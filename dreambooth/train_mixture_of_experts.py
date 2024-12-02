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
from parse import *
from peft import LoraConfig, get_peft_model
import torch.nn as nn 
from diffusers_mole.utils import randn_tensor

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__)

UNET_TARGET_MODULES = ["to_q", "to_v", "query", "value"]  # , "ff.net.0.proj"]
TEXT_ENCODER_TARGET_MODULES = ["q_proj", "v_proj"]



def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
    )
    if args.report_to == "wandb" and accelerator.is_main_process:
        import wandb
        wandb_path = os.path.join(args.output_dir, 'wandb')
        os.makedirs(wandb_path, exist_ok = True)
        wandb.login(key=args.wandb_key)
        wandb.init(project=args.wandb_project_name, dir=wandb_path)
    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformersmole.utils.logging.set_verbosity_warning()
        diffusers_mole.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformersmole.utils.logging.set_verbosity_error()
        diffusers_mole.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)


    # Generate class images if prior preservation is enabled.
    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
            if args.prior_generation_precision == "fp32":
                torch_dtype = torch.float32
            elif args.prior_generation_precision == "fp16":
                torch_dtype = torch.float16
            elif args.prior_generation_precision == "bf16":
                torch_dtype = torch.bfloat16

            from diffusers import StableDiffusionPipeline
            
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                safety_checker=None,
                revision=args.revision,
                num_experts=0,
                enable_mole=False
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)

            for example in tqdm(
                sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)  # noqa: F841

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    number_of_lora = len(args.lora_list)

    if args.use_lora:
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=UNET_TARGET_MODULES,
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            number_of_lora=number_of_lora
        )
        unet = get_peft_model(unet, config)
        unet.print_trainable_parameters()

    vae.requires_grad_(False)
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)

    elif args.train_text_encoder and args.use_lora:
        config = LoraConfig(
            r=args.lora_text_encoder_r,
            lora_alpha=args.lora_text_encoder_alpha,
            target_modules=TEXT_ENCODER_TARGET_MODULES,
            lora_dropout=args.lora_text_encoder_dropout,
            bias=args.lora_text_encoder_bias,
            number_of_lora=number_of_lora
        )
        text_encoder = get_peft_model(text_encoder, config)

    LoRA_root_path = '/mnt1/msranlpintern/wuxun/StyleHub/exp/multi_lora_exp/cv_multi_lora/dreambooth/cpkt'

    lora_list = args.lora_list
    for lora in lora_list:
        set_and_load_loras(unet, text_encoder, os.path.join(LoRA_root_path, lora), adapter_name=lora)
    unet.delete_adapter('default')
    text_encoder.delete_adapter('default')

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        # below fails when using lora so commenting it out
        if args.train_text_encoder and not args.use_lora:
            text_encoder.gradient_checkpointing_enable()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = (
        itertools.chain(unet.parameters(), text_encoder.parameters()) if args.train_text_encoder else unet.parameters()
    )
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
        num_workers=1,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1]
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(args.output_dir, path))
        global_step = int(path.split("-")[1])

        resume_global_step = global_step * args.gradient_accumulation_steps
        first_epoch = resume_global_step // num_update_steps_per_epoch
        resume_step = resume_global_step % num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # mark only as gates trainable
    mark_only_gate_as_trainable(unet)
    mark_only_gate_as_trainable(text_encoder)

    for n, p in unet.named_parameters():
        print(n, p.requires_grad)
    for n, p in text_encoder.named_parameters():
        print(n, p.requires_grad)

    for n, p in unet.named_parameters():
        if 'lora' in n and 'transformer_blocks' not in n:
            assert 0
    for n, p in text_encoder.named_parameters():
        if 'lora' in n and 'self_attn' not in n:
            assert 0
            

    def prepare_latents(batch_size, dtype, device, noise_scheduler, generator=None, latents=None):
        # generator = torch.Generator(device="cuda").manual_seed(8)
        shape = (batch_size, 4, 64, 64)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * noise_scheduler.init_noise_sigma
        return latents


    def normalize(images):
        """
        Normalize an image array to [-1,1].
        """
        return 2.0 * images - 1.0

    from loss.clip_loss import CLIPLoss
    clip_loss_func = CLIPLoss(
        accelerator.device,
        lambda_direction=1,
        lambda_patch=0,
        lambda_global=1,
        lambda_manifold=0,
        lambda_texture=0,
        lambda_sementic=0,
        lambda_class_global=0,
        clip_model='ViT-B/16'
    )

    # num_inference_steps = 50
    num_loss_backward_step = args.num_loss_backward_step
    backward_step = 0
    # noise_scheduler.set_timesteps(num_inference_steps, device=accelerator.device)
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
    visualization = True

    for epoch in range(first_epoch, args.num_train_epochs):
        epoch_loss = 0.
        unet.train()
        if args.train_text_encoder:
            text_encoder.train()
        with TorchTracemalloc() as tracemalloc:
            count = 0
            for step, batch in enumerate(train_dataloader):
                # Skip steps until we reach the resumed step
                if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                        if args.report_to == "wandb" and accelerator.is_main_process:
                            accelerator.print(progress_bar)
                    continue

                # set num_inference_steps
                num_inference_steps = random.randint(20, 50)
                noise_scheduler.set_timesteps(num_inference_steps, device=accelerator.device)

                with accelerator.accumulate(unet):
                    # Convert images to latent space
                    if args.with_prior_preservation:
                        class_latents = vae.encode(batch["class_pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                        class_latents = class_latents * 0.18215
                    ins_latents = vae.encode(batch["instance_pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                    ins_latents = ins_latents * 0.18215
                    if args.with_prior_preservation:
                        gt_latents = torch.cat((ins_latents, class_latents), dim = 0)
                    else:
                        gt_latents = ins_latents

                    # Sample noise that we'll add to the latents
                    bsz = batch["instance_pixel_values"].shape[0]
                    # Sample a random timestep for each image
                    timesteps_list = noise_scheduler.timesteps.tolist()
                    n = len(timesteps_list) - 1
                    step_s_ratio = args.step_s_ratio
                    step_e_ratio = args.step_e_ratio
                    random_index = random.randint(int(n * step_s_ratio), int(n * step_e_ratio))
                    timesteps = torch.Tensor([timesteps_list[random_index]]).expand(bsz)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    noise = torch.randn_like(gt_latents)
                    input_latents = noise_scheduler.add_noise(gt_latents, noise, timesteps)

                    ## generate the original RGB images
                    reserved_timesteps = timesteps_list[random_index:]

                    output_latents = None
                    if accelerator.is_main_process:
                        all_loss_dict = None
                    loss = 0.
                    loss_weights_sum = sum(reserved_timesteps)

                    for i, t in enumerate(reserved_timesteps):
                        encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                        input_latents = noise_scheduler.scale_model_input(input_latents.detach(), t)
                        model_pred = unet(input_latents, t, encoder_hidden_states).sample
                        output_latents = noise_scheduler.step(model_pred, t, input_latents, return_dict=False)[0]
                        input_latents = output_latents.clone().detach()
                    
                        if args.with_prior_preservation:
                            generated_latent, class_latent = torch.chunk(output_latents, 2, dim=0)
                        else:
                            generated_latent = output_latents
                        count += 1

                        ins_pixels = batch["instance_pixel_values"]
                        ins_pixels = (ins_pixels * 0.5) + 0.5
                        if args.with_prior_preservation:
                            class_pixels = batch["class_pixel_values"]
                            class_pixels = (class_pixels * 0.5) + 0.5

                        tar_image = vae.decode(generated_latent.to(dtype=weight_dtype) / vae.config.scaling_factor, return_dict=False)[0]
                        tar_image = image_processor.postprocess(tar_image, output_type='pt', do_denormalize=[True] * tar_image.shape[0]) # [3, 512, 512]
                        assert tar_image.requires_grad == True

                        if visualization and i == len(reserved_timesteps) - 1:
                            # print(tar_image.shape, ins_pixels.shape)
                            # torch.Size([2, 3, 512, 512]) torch.Size([1, 3, 512, 512])
                            for j in range(tar_image.shape[0]):
                                save_image = torch.cat((tar_image[j], ins_pixels[j]), dim = -1)
                                save_image = save_image.permute(1, 2, 0).detach().cpu().float().numpy()[:, :, ::-1]
                                cv2.imwrite(os.path.join(args.output_dir, 'middle_results_ddp/e{}_s{}_b{}_scale_model_input_img.png'.format(epoch, step, j)), np.uint8(np.clip(save_image * 255., 0., 255.)))
                        
                        if args.with_prior_preservation:
                            cls_image = vae.decode(class_latent.to(dtype=weight_dtype) / vae.config.scaling_factor, return_dict=False)[0]
                            cls_image = image_processor.postprocess(cls_image, output_type='pt', do_denormalize=[True] * cls_image.shape[0])# [3, 512, 512]
                            assert cls_image.requires_grad == True
                            # print("image", type(image), image.requires_grad, image.shape, image.max(), image.min())
                            if visualization and i == len(reserved_timesteps) - 1:
                                # print(cls_image.shape, tar_image.shape)
                                for j in range(cls_image.shape[0]):
                                    save_image = torch.cat((cls_image[j], class_pixels[j]), dim = -1)
                                    save_image = save_image.permute(1, 2, 0).detach().cpu().float().numpy()[:, :, ::-1]
                                    cv2.imwrite(os.path.join(args.output_dir, 'middle_results_ddp/class_e{}_s{}_b{}_scale_model_input_img.png'.format(epoch, step, j)), np.uint8(np.clip(save_image * 255., 0., 255.)))

                        
                        # compute L2 loss
                        L2_loss = torch.nn.functional.mse_loss(ins_pixels, tar_image) + torch.nn.functional.mse_loss(class_pixels, tar_image)
                        L2_loss = L2_loss * args.L2_weights

                        if accelerator.is_main_process:
                            if all_loss_dict is None:
                                all_loss_dict = {}
                                all_loss_dict['L2 loss'] = L2_loss
                            else:
                                all_loss_dict['L2 loss'] += L2_loss

                        loss = loss + L2_loss.to(weight_dtype)

                        if args.use_entropy_loss:
                            unet_entropy, text_encoder_entropy = get_all_entropy(unet, text_encoder)
                            # entropy_loss = (1.0 - (unet_entropy + text_encoder_entropy)) * 0.5
                            entropy_loss = -torch.log(torch.Tensor([unet_entropy])) - torch.log(torch.Tensor([text_encoder_entropy]))
                            entropy_loss = entropy_loss.to(loss.device)
                            if accelerator.is_main_process:
                                all_loss_dict['entropy loss'] = entropy_loss
                            loss = loss + entropy_loss.to(weight_dtype)

                        # compute CLIP loss
                        for j in range(len(batch['ins_names'])):
                            sub_loss_dict = clip_loss_func(ins_pixels[j].unsqueeze(0), batch['ins_names'][j], tar_image[j].unsqueeze(0), args.instance_prompt, ins_pixels[j].unsqueeze(0), None)
                            loss_clip = (2.5 - sub_loss_dict['clip_loss']) / 2.0
                            loss_clip = - torch.log(loss_clip)
                            loss = loss + loss_clip.to(weight_dtype)

                            if accelerator.is_main_process:
                                for key in sub_loss_dict.keys():
                                    if key not in all_loss_dict.keys():
                                        all_loss_dict[key] = sub_loss_dict[key]
                                    else:
                                        all_loss_dict[key] += sub_loss_dict[key]

                        backward_step += 1
                        if backward_step % num_loss_backward_step == 0:
                            loss = loss.to(weight_dtype)
                            accelerator.backward(loss)
                            if accelerator.sync_gradients:
                                params_to_clip = (
                                    itertools.chain(unet.parameters(), text_encoder.parameters())
                                    if args.train_text_encoder
                                    else unet.parameters()
                                )
                                accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                            with torch.no_grad():
                                for n, p in unet.named_parameters():
                                    if 'temperature' in n:
                                        p.data.clamp_(1.0, 3.0)
                                for n, p in text_encoder.named_parameters():
                                    if 'temperature' in n:
                                        p.data.clamp_(1.0, 3.0)
                            
                            optimizer.step()
                            lr_scheduler.step()
                            optimizer.zero_grad()
                            backward_step = 0
                            loss = 0.

                        # del output_latents
                        # del generated_latent
                        # del ins_pixels
                        # if args.with_prior_preservation:
                        #     del class_pixels
                        torch.cuda.empty_cache()

                    # ================================ record ================================
                    if accelerator.is_main_process:
                        for key in sub_loss_dict.keys():
                            all_loss_dict[key] = all_loss_dict[key] / (bsz * len(reserved_timesteps))

                        # ================================ compute gates ================================
                        gates_record = 0.
                        count = 0
                        for n, p in text_encoder.named_parameters():
                            if 'gates_record' in n:
                                count += 1
                                gates_record += p.data
                                p.data.fill_(0.0)
                        results_list = (gates_record / count).tolist()
                        for n in range(len(results_list)):
                            all_loss_dict['gates_text_encoder_lora{}'.format(n)] = results_list[n] / len(reserved_timesteps)


                        gates_record = 0.
                        count = 0
                        for n, p in unet.named_parameters():
                            if 'gates_record' in n:
                                count += 1
                                gates_record += p.data
                                p.data.fill_(0.0)
                        results_list = (gates_record / count).tolist()
                        for n in range(len(results_list)):
                            all_loss_dict['gates_unet_lora{}'.format(n)] = results_list[n] / len(reserved_timesteps)

                        entropy_record = 0.
                        count = 0
                        for n, p in text_encoder.named_parameters():
                            if 'entropy_record' in n:
                                count += 1
                                entropy_record += p.data
                                p.data.fill_(0.0)
                        all_loss_dict['entropy_text_encoder'] = entropy_record.item() / (count * len(reserved_timesteps))

                        entropy_record = 0.
                        count = 0
                        for n, p in unet.named_parameters():
                            if 'entropy_record' in n:
                                count += 1
                                entropy_record += p.data
                                p.data.fill_(0.0)
                        all_loss_dict['entropy_unet'] = entropy_record.item() / (count * len(reserved_timesteps))


                        temperature = 0.
                        count = 0
                        for n, p in unet.named_parameters():
                            if 'temperatures' in n:
                                count += 1
                                temperature += p.data
                        all_loss_dict['temperature_unet'] = temperature.item() / (count)


                        temperature = 0.
                        count = 0
                        for n, p in text_encoder.named_parameters():
                            if 'temperatures' in n:
                                count += 1
                                temperature += p.data
                        all_loss_dict['temperature_text_encoder'] = temperature.item() / (count)

                        all_loss_dict['lr'] = lr_scheduler.get_last_lr()[0]
                        all_loss_dict['bs'] = num_loss_backward_step * accelerator.num_processes
                        accelerator.log(all_loss_dict, step=global_step)
                        wandb.log(all_loss_dict)

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    if args.report_to == "wandb" and accelerator.is_main_process:
                        accelerator.print(progress_bar)
                    global_step += 1

                if global_step >= args.max_train_steps:
                    break

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if args.use_lora:
                save_gates(unet, text_encoder, args.output_dir, epoch)
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
