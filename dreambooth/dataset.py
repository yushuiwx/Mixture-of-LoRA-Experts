import argparse
import gc
import hashlib
import itertools
import logging
import math
import os, sys
import threading
import warnings
from pathlib import Path
from typing import Optional
sys.path.append("../tools")
import datasets
import diffusers_mole
import numpy as np
import psutil
import cv2
import random
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformersmole
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers_mole import (
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler,
    PNDMScheduler,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
from diffusers_mole.image_processor import VaeImageProcessor
from diffusers_mole.optimization import get_scheduler
from diffusers_mole.utils import check_min_version
from diffusers_mole.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformersmole import AutoTokenizer, PretrainedConfig
from utils import *
from dataset import *
from peft import LoraConfig, get_peft_model
import torch.nn as nn 
from diffusers_mole.utils import randn_tensor


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        # self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.instance_images_path = []
        sub_dirs = os.listdir(instance_data_root)
        for sub_dir in sub_dirs:
            sub_images = os.listdir(os.path.join(instance_data_root, sub_dir))
            for sub_image in sub_images:
                self.instance_images_path.append(Path(os.path.join(instance_data_root, sub_dir, sub_image)))

        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        
        # get instance name
        ins_path = self.instance_images_path[index % self.num_instance_images]
        ins_name = str(os.path.basename(os.path.dirname(ins_path)))

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_names"] = ins_name
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            # change class prmpt into no <>
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

        return example


def collate_fn(examples, with_prior_preservation=False):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    instance_pixel_values = [example["instance_images"] for example in examples]
    ins_names = [example["instance_names"] for example in examples]
    
    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        class_pixel_values = [example["class_images"] for example in examples]
    
    instance_pixel_values = torch.stack(instance_pixel_values)
    instance_pixel_values = instance_pixel_values.to(memory_format=torch.contiguous_format).float()

    if with_prior_preservation:
        class_pixel_values = torch.stack(class_pixel_values)
        class_pixel_values = class_pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)

    if with_prior_preservation:
        batch = {
            "input_ids": input_ids,
            "instance_pixel_values": instance_pixel_values,
            "class_pixel_values": class_pixel_values,
            "ins_names": ins_names
        }
    else:
        batch = {
            "input_ids": input_ids,
            "instance_pixel_values": instance_pixel_values,
            "ins_names": ins_names
        }
    return batch


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example
