#!/bin/bash

lora_name=$1
export MODEL_NAME="CompVis/stable-diffusion-v1-4" #"stabilityai/stable-diffusion-2-1"
export INSTANCE_DIR=/mnt1/msranlpintern/wuxun/StyleHub/exp/multi_lora_exp/mixture_of_lora_experts/dreambooth_based_below_gate/dreambooth/datasets/cut_mix_datasets/bear_plushie_clock
# export CLASS_DIR=/mnt1/msranlpintern/wuxun/StyleHub/exp/multi_lora_exp/cv_multi_lora/dreambooth/datasets/cls_datasets/clock
export OUTPUT_DIR=/mnt1/msranlpintern/wuxun/StyleHub/exp/multi_lora_exp/cv_multi_lora/dreambooth/cpkt/bear_plushie_clock
mkdir -p $OUTPUT_DIR
mkdir -p $CLASS_DIR
echo "==========================================================================================="
echo "$count_bear_plushie_clock"
echo "a photo of <$count> bear_plushie_clock"
echo "==========================================================================================="
accelerate launch finetune_with_lora.py \
--pretrained_model_name_or_path=$MODEL_NAME  \
--instance_data_dir=$INSTANCE_DIR \
--output_dir=$OUTPUT_DIR \
--train_text_encoder \
--instance_prompt="a photo of a <19> bear_plushie and a <9> clock." \
--resolution=512 \
--train_batch_size=2 \
--lr_scheduler="constant" \
--lr_warmup_steps=0 \
--use_lora \
--lora_r 16 \
--lora_alpha 27 \
--lora_text_encoder_r 16 \
--lora_text_encoder_alpha 17 \
--learning_rate=1e-4 \
--gradient_accumulation_steps=1 \
--gradient_checkpointing \
--max_train_steps=800
