from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Paths
IMAGE_PATH = "data/car.jpeg"

'''
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

image = Image.open("data/car.jpeg").convert("RGB")
inputs = processor(image, return_tensors="pt")
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)

print(caption)  # → "A red car parked on the street"
'''

from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch

# Load BLIP-2 processor and model (make sure to match versions)
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", use_fast=True)
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16, device_map="auto"
)

# Load and preprocess image
image = Image.open("your_image.jpg").convert("RGB")

# Prompt to extract components
prompt = "List all visible parts or components of the car in this image."

# Preprocess inputs
inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

# Generate output
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=50)

# Decode and print
result = processor.decode(output[0], skip_special_tokens=True)
print(result)