from transformers import AutoTokenizer, CLIPModel
from transformers import AutoProcessor
from PIL import Image
import requests

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

text = ["a photo of a <19> bear_plushie.", "a photo of a <9> clock.", "a photo of a <19> bear_plushie and a <9> clock."]
image = '/mnt1/msranlpintern/wuxun/StyleHub/exp/multi_lora_exp/mixture_of_lora_experts/dreambooth_based_below_gate/dreambooth/datasets/cut_mix_datasets/bear_plushie_clock/01.png'


inputs = tokenizer(text, padding=True, return_tensors="pt")
text_features = model.get_text_features(**inputs)
print(text_features.shape)


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open(image)

inputs = processor(images=image, return_tensors="pt")

image_features = model.get_image_features(**inputs)
print(image_features.shape)
print("processor", processor)

inputs = processor(
    text=text, images=image, return_tensors="pt", padding=True
)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
print(probs)