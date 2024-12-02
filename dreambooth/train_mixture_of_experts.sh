#!/bin/bash
TAG=$1
PROMPT=$2
LR=$3
step_s_ratio=$4
step_e_ratio=$5
L2_weights=$6
PROMPT_cleaned=$(echo "$PROMPT" | sed 's/[^a-zA-Z0-9]/_/g')
ROOT_PATH=../results/$TAG/$PROMPT_cleaned/LR$LR-S_RATIO$step_s_ratio-E_RATIO$step_e_ratio-L2WEIGHTS$L2_weights
export MODEL_NAME="CompVis/stable-diffusion-v1-4" #"stabilityai/stable-diffusion-2-1"
export INSTANCE_DIR=/mnt1/msranlpintern/wuxun/StyleHub/exp/multi_lora_exp/mixture_of_lora_experts/dreambooth_based_below_gate_under_text_guidance_multi_sup/dreambooth/datasets/ins_dataset/dog6_wolf_plushie_backpack
export CLASS_DIR=$ROOT_PATH/cls_datasets
export OUTPUT_DIR=$ROOT_PATH

CLASS_PROMPT=$(echo "$PROMPT" | sed 's/<[^>]*>//g')

echo "#### USE PROMPT: $PROMPT"
echo "#### USE CLASS PROMPT: $CLASS_PROMPT"
mkdir -p $OUTPUT_DIR/middle_results_ddp
mkdir -p $CLASS_DIR

accelerate  launch train_mixture_of_experts.py \
--pretrained_model_name_or_path=$MODEL_NAME  \
--instance_data_dir=$INSTANCE_DIR \
--output_dir=$OUTPUT_DIR \
--train_text_encoder \
--instance_prompt="$PROMPT" \
--resolution=512 \
--train_batch_size=1 \
--lr_scheduler="constant" \
--lr_warmup_steps=20 \
--use_lora \
--lora_r 16 \
--lora_alpha 27 \
--lora_text_encoder_r 16 \
--lora_text_encoder_alpha 17 \
--learning_rate=$LR \
--gradient_accumulation_steps=1 \
--gradient_checkpointing \
--max_train_steps=400 \
--report_to wandb \
--wandb_key 9ea8068a1f9d45d54de5a570aec38b1b152b3274 \
--wandb_project_name mole_block \
--lora_list dog6 wolf_plushie backpack \
--with_prior_preservation \
--prior_loss_weight=1.0 \
--class_prompt="$CLASS_PROMPT" \
--class_data_dir=$CLASS_DIR \
--num_class_images=200 \
--step_s_ratio $step_s_ratio \
--num_loss_backward_step 2 \
--step_e_ratio $step_e_ratio \
--L2_weights $L2_weights \
--use_entropy_loss \
# --mixed_precision fp16 
