pip install transformers==4.34.0
pip install accelerate==0.20.3
pip install bitsandbytes
pip install SentencePiece
pip install git+https://github.com/openai/CLIP.git
pip install diffusers==0.20.2
pip install wandb
pip install tokenizers==0.14.1
cd ../tools/transformers_mole
pip install -e .
pip install huggingface_hub==0.16.4
pip uninstall -y transformers
pip install datasets==2.6.0
pip install opencv-python
pip list