# Mixture of LoRA Experts
[![Paper](https://img.shields.io/badge/cs.CV-Paper-b31b1b?logo=arxiv&logoColor=red)](https://openreview.net/pdf?id=uWvKBCYh4S)


Official implementation of "**Mixture of LoRA Experts**"

 [Xun Wu](https://yushuiwx.github.io/), [Shaohan Huang](https://buaahsh.github.io/)+, [Furu Wei](https://thegenerality.com/)

accepted by International Conference on Learning Representations (**ICLR**), 2024

## Abstract
> LoRA has emerged as a pivotal technique for fine-tuning large pre-trained models, renowned for its efficacy across diverse tasks. Its modular design has spurred investigations into the composition of multiple trained LoRAs to enhance task performance. Nevertheless, the effective composition of these LoRAs remains a formidable challenge: (1) Linear arithmetic composition method may lead to the loss of the generative capabilities inherent in the original pre-trained model or the distinctive attributes of the trained LoRAs, resulting in suboptimal outcomes. (2) Reference tuning-based composition method exhibits limitations in terms of the necessary adaptability for effectively composing multiple LoRAs and incurs significant costs due to retraining a sizable model. In response to these challenges, we propose Mixture of LoRA Experts (MoLE). MoLE treats each layer of trained LoRAs as distinct experts and implements hierarchical weight control by integrating a learnable gating function within each layer to learn optimal composition weights tailored to a specific domain objective. MoLE not only surpasses linear arithmetic composition in terms of LoRA composition performance but also preserves the essential flexibility required for the effective composition of trained LoRAs with minimal computational overhead. Extensive experimental evaluations conducted in both Natural Language Processing (NLP) and Vision & Language (V&L) domains validate the efficacy of MoLE.

## Project Structure

```
|-- dreambooth
    |-- loss.py # clip loss implementation
    -- dataset.py # datatset class for lora finetuning
    -- finetune_with_lora.py # main code for finetuning to get the lora candidates
    -- id.log # The correspondence between id and lora candidates.
    -- train_mixture_of_experts.py # main code for training the MoLE
    -- train_mixture_of_experts.sh # script for run the train_mixture_of_experts.py
    -- run.sh # a training example
    -- inference.py # inference the images with MoLE
|-- tools # tools code
    |-- diffusers_mole # modified diffusers for supporting block-wise MoLE training
    |-- peft # modified peft for supporting block-wise MoLE training
    |-- transformers_mole # modified transformers for supporting block-wise MoLE training

```
## Setup

```
conda create -n MoLE python=3.8 -y
conda activate MoLE
cd dreambooth
bash setup.sh
```


### Prepare LoRA Candidates

------

#### 1. NLP tasks

For the NLP task, we use LoRA candidates provided in LoRAHub, available at [LoRAHub - Hugging Face](https://huggingface.co/models?search=lorahub). If you are using these candidates, please consider citing their paper  [LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition](https://arxiv.org/abs/2307.13269).

#### 2. Text-to-Images tasks

we provide text-to-image LoRA candidates trained on DreamBooth dataset.

* Candidates Link: [Google Drive](https://drive.google.com/file/d/1DwWq6k-fAXaPmDOnJxORhENq2PMAq7by/view?usp=sharing).

* DreamBooth Dataset Link:  https://github.com/google/dreambooth

Or you can download our candidates directly from [Hugging Face](https://huggingface.co/collections/YUSHUIWX/mixture-of-lora-experts-67239ad28cb487fa22a0bd74).

---

## Training with LoRA Candidates
```
cd dreambooth
bash run.sh # An example
```
or you can modify the run.sh as:

```
bash train_mixture_of_experts.sh <YOUR_EXP_TAG> <YOUR_PROMPT> 1e-5 0 0.1 0 0.5
```


## Citation

If our work is useful for you, please consider citing our paper:

```
@inproceedings{wu2023mole,
  title={Mole: Mixture of lora experts},
  author={Wu, Xun and Huang, Shaohan and Wei, Furu},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2023}
}
```



