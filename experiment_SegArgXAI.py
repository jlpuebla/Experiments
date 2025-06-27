import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import warnings

from torchvision import models, transforms
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from groundingdino.util.inference import load_model, predict, load_image
from ollama import get_components

from utils.model_utils import download_if_not_exists

# Paths
IMAGE_PATH = "data/car.jpeg"
SAM_CHECKPOINT = "models/sam_vit_h.pth"
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
DINO_CHECKPOINT = "models/groundingdino.pth"
DINO_URL = "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth"
DINO_CONFIG = "configs/GroundingDINO_SwinT_OGC.py"

# Automatically use GPU if available, else default to CPU (for compatibility)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
def load_sam():
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
    predictor = SamPredictor(sam)
    return predictor

# Run segmentation 
def run_segmentation_and_grounding(caption: str):
    image = cv2.imread(IMAGE_PATH)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_source, image_tensor = load_image(IMAGE_PATH)

    predictor = load_sam()
    predictor.set_image(image_rgb)

    dino_model = load_model(DINO_CONFIG, DINO_CHECKPOINT)
    return predict(model=dino_model,
                   image=image_tensor,
                   caption=caption,
                   box_threshold=0.3,
                   text_threshold=0.25,
                   device=device
                   )

def preprocess_image(image_path: str):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),  # converts to [C, H, W] and [0, 1] range
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],) # ImageNet normalization
        ])
    img = Image.open(image_path).convert("RGB")
    return preprocess(img).unsqueeze(0)  # Add batch dimension

def format_for_grounding_dino(components, main_class=None):
    if main_class:
        components = [main_class] + components

    # Normalize to lowercase and singular if possible
    return " . ".join(c.lower() for c in components)

if __name__ == "__main__":
    # Ignore warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Download sam and dino models
    download_if_not_exists(SAM_URL, SAM_CHECKPOINT)
    download_if_not_exists(DINO_URL, DINO_CHECKPOINT)

    ''' 1. Classify Image '''
    # Load pretrained ResNet-50
    model = models.resnet50(pretrained=True)
    model.eval()  # set to inference mode

    # Preprocess input image
    img_tensor = preprocess_image(IMAGE_PATH)

    # Classify image
    predicted_class_id = ''
    with torch.no_grad(): # do not track gradients, faster
        output = model(img_tensor)
        predicted_class_id = output.argmax().item()

    # Decode class id to label
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    imagenet_labels = urllib.request.urlopen(url).read().decode("utf-8").splitlines()
    predicted_class_name = imagenet_labels[predicted_class_id]
    print("Predicted class:", predicted_class_name)

    ''' 2. Get list of components with LLM '''
    components = get_components(predicted_class_name)
    print(f'Components: {components}')
    
    dino_caption = format_for_grounding_dino(components=components, main_class=predicted_class_name)
    print(f'Dino caption: {dino_caption}')

    ''' 3. Check component presence '''
    boxes, logits, phrases = run_segmentation_and_grounding(dino_caption)

    print('Detected:')
    for box, label in zip(boxes, phrases):
        print(f"- {label}")
        x0, y0, x1, y1 = map(int, box)

    ''' 4. Argumentation framework generation '''

    ''' 5. Explanation'''