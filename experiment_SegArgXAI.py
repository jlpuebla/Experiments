import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamPredictor
from groundingdino.util.inference import load_model, predict, load_image

from model_utils import download_if_not_exists

# Paths
IMAGE_PATH = "data/car.jpeg"
SAM_CHECKPOINT = "models/sam_vit_h.pth"
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
DINO_CHECKPOINT = "models/groundingdino.pth"
DINO_URL = "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth"
DINO_CONFIG = "configs/GroundingDINO_SwinT_OGC.py"

# Automatically use GPU if available, else default to CPU (for compatibility)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Models
def load_sam():
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
    predictor = SamPredictor(sam)
    return predictor

# Run Segmentation 
def run_segmentation_and_grounding():
    image = cv2.imread(IMAGE_PATH)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_source, image_tensor = load_image(IMAGE_PATH)

    predictor = load_sam()
    predictor.set_image(image_rgb)

    dino_model = load_model(DINO_CONFIG, DINO_CHECKPOINT)
    TEXT_PROMPT = "car . wheel . headlight . door . windshield" # TODO: use LLM to find components for this list
    boxes, logits, phrases = predict(model=dino_model,
                                     image=image_tensor,
                                     caption=TEXT_PROMPT,
                                     box_threshold=0.3,
                                     text_threshold=0.25,
                                     device=device
                                     )

    for box, label in zip(boxes, phrases):
        print(f"Detected: {label}")
        x0, y0, x1, y1 = map(int, box)
        cv2.rectangle(image_rgb, (x0, y0), (x1, y1), (255, 0, 0), 2)
        cv2.putText(image_rgb, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    plt.imshow(image_rgb)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    # Download Models
    download_if_not_exists(SAM_URL, SAM_CHECKPOINT)
    download_if_not_exists(DINO_URL, DINO_CHECKPOINT)
    
    run_segmentation_and_grounding()