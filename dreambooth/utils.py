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

from peft import LoraConfig, get_peft_model
import torch.nn as nn 
from diffusers_mole.utils import randn_tensor


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformersmole import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers_mole.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


# Converting Bytes to Megabytes
def b2mb(x):
    return int(x / 2**20)

def mark_only_gate_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "wg" in n:
            p.requires_grad = True
        else:
            p.requires_grad = False

# This context manager is used to track the peak memory usage of the process
class TorchTracemalloc:
    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()  # reset the peak gauge to zero
        self.begin = torch.cuda.memory_allocated()
        self.process = psutil.Process()

        self.cpu_begin = self.cpu_mem_used()
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        return self

    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_peak = -1

        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)

            # can't sleep or will not catch the peak right (this comment is here on purpose)
            # time.sleep(0.001) # 1msec

            if not self.peak_monitoring:
                break

    def __exit__(self, *exc):
        self.peak_monitoring = False

        gc.collect()
        torch.cuda.empty_cache()
        self.end = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)

        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = b2mb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = b2mb(self.cpu_peak - self.cpu_begin)
        # print(f"delta used/peak {self.used:4d}/{self.peaked:4d}")



def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def load_adapter(unet, text_encoder, ckpt_dir, adapter_name):
    unet_sub_dir = os.path.join(ckpt_dir, "unet")
    text_encoder_sub_dir = os.path.join(ckpt_dir, "text_encoder")
    unet.load_adapter(unet_sub_dir, adapter_name=adapter_name)
    text_encoder.load_adapter(text_encoder_sub_dir, adapter_name=adapter_name)

def set_adapter(net, adapter_name):
    net.set_adapter(adapter_name)

def save_gates(unet, text_encoder, save_dir, epoch):
    all_state_dict = {}
    for n, p in unet.named_parameters():
        if "wg" in n:
            assert n not in all_state_dict.keys()
            all_state_dict[n] = p
    for n, p in text_encoder.named_parameters():
        if "wg" in n:
            assert n not in all_state_dict.keys()
            all_state_dict[n] = p

    save_pah = os.path.join(save_dir, 'gates_epoch{}.bin'.format(epoch))
    torch.save(all_state_dict, save_pah)


def set_and_load_loras(unet, text_encoder, dir, adapter_name):
    load_adapter(unet, text_encoder, dir, adapter_name=adapter_name)
    set_adapter(unet, adapter_name=adapter_name)
    set_adapter(text_encoder, adapter_name=adapter_name)



def entropy(probs):
    logits = torch.distributions.utils.probs_to_logits(probs)
    p_log_p = probs * logits
    return -p_log_p.sum(-1)

def product_of_list_elements(lst):
    result = 1
    for element in lst:
        result *= element
    return result


def get_all_entropy(unet, text_encoder):
    text_encoder_gates_record = 0.
    count = 0
    for n, p in text_encoder.named_parameters():
        if 'gates_record' in n:
            count += 1
            text_encoder_gates_record += p.data
    text_encoder_gates_record /= count
    text_encoder_gates_record = text_encoder_gates_record / (text_encoder_gates_record.sum())
    text_encoder_gates_record = text_encoder_gates_record.tolist()
    text_encoder_entropy = product_of_list_elements(text_encoder_gates_record)


    unet_gates_record = 0.
    count = 0
    for n, p in unet.named_parameters():
        if 'gates_record' in n:
            count += 1
            unet_gates_record += p.data
    unet_gates_record /= count
    unet_gates_record = unet_gates_record / (unet_gates_record.sum())
    unet_gates_record = unet_gates_record.tolist()
    unet_entropy = product_of_list_elements(unet_gates_record)

    return unet_entropy, text_encoder_entropy
