import os, sys
import torch
sys.path.append('../')
from diffusers import StableDiffusionPipeline
from peft import PeftModel, LoraConfig

MODEL_NAME = "/mnt1/msranlpintern/wuxun/envs/hub/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b"

def get_lora_sd_pipeline(
    ckpt_dir, base_model_name_or_path=None, dtype=torch.float16, device="cuda", adapter_name="default"
):
    unet_sub_dir = os.path.join(ckpt_dir, "unet")
    text_encoder_sub_dir = os.path.join(ckpt_dir, "text_encoder")
    if os.path.exists(text_encoder_sub_dir) and base_model_name_or_path is None:
        config = LoraConfig.from_pretrained(text_encoder_sub_dir)
        base_model_name_or_path = config.base_model_name_or_path

    if base_model_name_or_path is None:
        raise ValueError("Please specify the base model name or path")

    pipe = StableDiffusionPipeline.from_pretrained(base_model_name_or_path, torch_dtype=dtype).to(device)
    pipe.unet = PeftModel.from_pretrained(pipe.unet, unet_sub_dir, adapter_name=adapter_name)

    if os.path.exists(text_encoder_sub_dir):
        pipe.text_encoder = PeftModel.from_pretrained(
            pipe.text_encoder, text_encoder_sub_dir, adapter_name=adapter_name
        )

    if dtype in (torch.float16, torch.bfloat16):
        pipe.unet.half()
        pipe.text_encoder.half()

    pipe.to(device)
    return pipe

def load_adapter(pipe, ckpt_dir, adapter_name):
    unet_sub_dir = os.path.join(ckpt_dir, "unet")
    text_encoder_sub_dir = os.path.join(ckpt_dir, "text_encoder")
    pipe.unet.load_adapter(unet_sub_dir, adapter_name=adapter_name)
    if os.path.exists(text_encoder_sub_dir):
        pipe.text_encoder.load_adapter(text_encoder_sub_dir, adapter_name=adapter_name)

def set_adapter(pipe, adapter_name):
    pipe.unet.set_adapter(adapter_name)
    if isinstance(pipe.text_encoder, PeftModel):
        pipe.text_encoder.set_adapter(adapter_name)

def create_weighted_lora_adapter(pipe, adapters, weights, adapter_name="default", combination_type='svd'):
    pipe.unet.add_weighted_adapter(adapters, weights, adapter_name)
    if isinstance(pipe.text_encoder, PeftModel):
        pipe.text_encoder.add_weighted_adapter(adapters, weights, adapter_name, combination_type=combination_type)

    return pipe

if __name__ == "__main__":
    pipe = get_lora_sd_pipeline("/mnt1/msranlpintern/wuxun/StyleHub/exp/multi_lora_exp/mixture_of_lora_experts/dreambooth_based_below_gate/dreambooth/cpkt/dog6", adapter_name="dog6")
    load_adapter(pipe, '/mnt1/msranlpintern/wuxun/StyleHub/exp/multi_lora_exp/cv_multi_lora/dreambooth/cpkt/wolf_plushie', adapter_name="wolf_plushie")
    load_adapter(pipe, '/mnt1/msranlpintern/wuxun/StyleHub/exp/multi_lora_exp/cv_multi_lora/dreambooth/cpkt/backpack', adapter_name="backpack")
    set_adapter(pipe, adapter_name="wolf_plushie")
    set_adapter(pipe, adapter_name="backpack")

    # loading gates
    gate_path = '/mnt1/msranlpintern/wuxun/StyleHub/exp/multi_lora_exp/results/block_wise/baseline_sum_weights2/a_photo_of__18__dog6_and_a__3__wolf_plushie__with_a__15__backpack_/LR1e-4-S_RATIO0-E_RATIO0.1-L2WEIGHTS0/gates_epoch6.bin'
    gates = torch.load(gate_path)
    # print(pipe, type(pipe))
    pipe.unet.load_state_dict(gates, strict=False)

    prompt = "a photo of <18> dog6 and <3> wolf_plushie, with a <15> backpack."
    lora_list = ["dog6", "backpack", "wolf_plushie"]

    save_root = './generated_results'
    save_path = os.path.join(save_root, "_".join(lora_list), 'samples')
    os.makedirs(save_path, exist_ok=True)
    prompts = {}
    for i in range(20):
        negative_prompt = "low quality, blurry, unfinished"
        image = pipe(prompt, num_inference_steps=50, guidance_scale=7, negative_prompt=negative_prompt).images[0]
        # print("image", image.requires_grad)
        image.save(os.path.join(save_path, '{}.png'.format(i)))
        prompts['{}.png'.format(i)] =  prompt
