import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import warnings
import mlflow
import tempfile
import os

from torchvision import models, transforms
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from groundingdino.util.inference import load_model, predict, load_image
from ollama import get_components
from captum.attr import IntegratedGradients, LayerGradCam

from utils.model_utils import download_if_not_exists

# Paths
IMAGE_PATH = "data/minivan.png"
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
    #if main_class:
    #    components = [main_class] + components

    # Normalize to lowercase and singular if possible
    return " . ".join(c.lower() for c in components)
 
def build_arguments( all_components: list, component_importances: dict,) -> list:
    detected = set(component_importances.keys())
    arguments = []

    # Support arguments for detected components with their importance scores
    for component, score in component_importances.items():
        arguments.append({
            "component": component,
            "relation":  "support",
            "weight":    float(score),
        })

    # Attack arguments for missing components with zero weight
    for component in all_components:
        if component not in detected:
            arguments.append({
                "component": component,
                "relation":  "attack",
                "weight":    0.0,
            })
 
    return arguments
 
def build_argumentation_framework(
    claim: str,
    all_components: list,
    component_importances: dict,
) -> dict:
    # If no components were detected, we cannot support the claim at all. Return early with an empty argument list and rejected claim.
    if not all_components:
        return {"claim": claim, "arguments": [], "accepted": False}
    
    # Build arguments based on detected components and their importance scores
    arguments = build_arguments(all_components, component_importances)
    supports = [a for a in arguments if a["relation"] == "support"]

    # return the argumentation framework dict with claim, arguments, and acceptance status
    return {
        "claim":     claim,
        "arguments": arguments,
        "accepted":  len(supports) > 0,
    }

def compute_attention_scores(
    component_importances: dict,
    attributions_ig_sum: np.ndarray,
) -> dict:
    # Calculate total attribution across the entire image
    ig_total = float(np.sum(np.maximum(attributions_ig_sum, 0)))

    if ig_total == 0:
        # return zero attention scores if the total attribution is zero to avoid division by zero
        return {c: 0.0 for c in component_importances}
    
    # return attention scores as the proportion of each component's importance relative to the total positive attribution
    return {
        component: float(score / ig_total)
        for component, score in component_importances.items()
    }

def generate_explanation(argumentation: dict, attention_scores: dict, supported: str) -> str:
    claim    = argumentation["claim"]
    supports = [a for a in argumentation["arguments"] if a["relation"] == "support"]
    attacks  = [a for a in argumentation["arguments"] if a["relation"] == "attack"]
 
    # Opening sentence
    explanation = (f"The image was classified as '{claim}'."
                   f" This claim is {supported} by {len(supports)} components detected in the image."
    )

    # Supporting evidence with attention scores
    if supports:
        support_details = ", ".join(
            f"{a['component']} ({attention_scores.get(a['component'], 0.0):.3%} contribution)"
            for a in supports
        )
        explanation += (
            f" Detected components and their contribution to the classification: "
            f"{support_details}."
        )
    else:
        explanation += " No expected components were detected to support this classification."
 
    # Attacking evidence — noted but not weighted
    if attacks:
        #attack_names = ", ".join(a["component"] for a in attacks)
        explanation += (
            f" A total of {len(attacks)} expected components were not detected. "
            f"Their absence does not invalidate the classification."
        )
    else:
        explanation += (
            f" All expected features were detected. "
        )
 
    return explanation

if __name__ == "__main__":
    # Ignore warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Download sam and dino models
    download_if_not_exists(SAM_URL, SAM_CHECKPOINT)
    download_if_not_exists(DINO_URL, DINO_CHECKPOINT)

    # Set the experiment name for MLflow tracking (create it if it doesn't exist)
    mlflow.set_experiment("VisLocArgXAI")

    with mlflow.start_run():

        # Log static run parameters
        mlflow.log_param("image_path", IMAGE_PATH)
        mlflow.log_param("device", str(device))

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

        # Step 1: log classification results
        mlflow.log_param("predicted_class", predicted_class_name)
        mlflow.log_param("predicted_class_id", predicted_class_id)

        ''' 2. Get list of components with LLM '''
        components = get_components(predicted_class_name)
        print(f'Components: {components}')
        
        dino_caption = format_for_grounding_dino(components=components, main_class=predicted_class_name)
        print(f'Dino caption: {dino_caption}')

        # Step 2: log components and formatted caption for grounding dino
        mlflow.log_param("components_expected", components)
        mlflow.log_metric("num_components_expected", len(components))
        mlflow.log_param("dino_caption", dino_caption)

        ''' 3. Check component presence '''
        boxes, logits, phrases = run_segmentation_and_grounding(dino_caption)

        print('Detected:')
        for box, label in zip(boxes, phrases):
            print(f"- {label}")
            x0, y0, x1, y1 = map(int, box)
            print(box)

        # Step 3: logging
        mlflow.log_metric("num_components_detected", len(phrases))
        mlflow.log_param("components_detected", phrases)
        mlflow.log_metric("detection_rate", len(phrases) / len(components) if components else 0)

        ''' 4. Evaluate feature importance'''
        mlflow.log_param("attribution_method", "IG")

        ''' 4a. Gradient-based attribution method '''
        # Tracks gradients from here on
        img_tensor.requires_grad_()

        # Initialize Integrated Gradients
        ig = IntegratedGradients(model)

        # Define a baseline: black image
        baseline = torch.zeros_like(img_tensor)

        # Compute attributions
        attributions_ig, delta = ig.attribute(
            img_tensor,
            baselines=baseline,
            target=predicted_class_id,
            return_convergence_delta=True
            )

        # Convert to numpy for aggregation
        attributions_ig = attributions_ig.squeeze().detach().cpu().numpy()

        # Sum across color channels to get single-channel attribution
        attributions_ig_sum = np.abs(attributions_ig).sum(axis=0)

        # Visualize and log the IG heatmap as an artifact
        fig, ax = plt.subplots()
        ax.imshow(attributions_ig_sum, cmap='hot')
        ax.set_title("Integrated Gradients Attribution")
        ax.axis("off")
        with tempfile.TemporaryDirectory() as tmp:
            heatmap_path = os.path.join(tmp, "ig_heatmap.png")
            fig.savefig(heatmap_path, bbox_inches="tight")
            mlflow.log_artifact(heatmap_path)
        plt.show()

        ''' 4b. Quantify the component importance '''
        img_pil = Image.open(IMAGE_PATH)
        W_orig, H_orig = img_pil.size

        scaled_boxes = []
        for box in boxes:
            x_center, y_center, width, height = box.tolist()
        
            x0 = int(max(0, min(224, (x_center - width / 2) * 224)))
            y0 = int(max(0, min(224, (y_center - height / 2) * 224)))
            x1 = int(max(0, min(224, (x_center + width / 2) * 224)))
            y1 = int(max(0, min(224, (y_center + height / 2) * 224)))
        
            scaled_boxes.append([x0, y0, x1, y1])

        # Visualize and log the IG heatmap with detected boxes overlaid as an artifact
        fig2, ax2 = plt.subplots()
        ax2.imshow(attributions_ig_sum, cmap="hot")
        for box in scaled_boxes:
            x0, y0, x1, y1 = box
            ax2.add_patch(
                plt.Rectangle((x0, y0), x1-x0, y1-y0, edgecolor="cyan", fill=False, lw=2)
            )
        ax2.set_title("Scaled boxes over attribution heatmap")
        with tempfile.TemporaryDirectory() as tmp:
            boxes_path = os.path.join(tmp, "ig_boxes.png")
            fig2.savefig(boxes_path, bbox_inches="tight")
            mlflow.log_artifact(boxes_path)
        plt.show()

        component_importances = {}
        
        for label, box in zip(phrases, scaled_boxes):
            x0, y0, x1, y1 = box
            print(f"Label: {label}, Scaled box: {x0},{y0} to {x1},{y1}")

            # Ensure box is within bounds
            x0 = max(x0, 0)
            y0 = max(y0, 0)
            x1 = min(x1, attributions_ig_sum.shape[1])
            y1 = min(y1, attributions_ig_sum.shape[0])
            
            # Crop attribution map
            attribution_crop = attributions_ig_sum[y0:y1, x0:x1]
            
            # Sum absolute attribution
            component_importances[label] = np.abs(attribution_crop).sum()

        print(f'Importance scores:\n{component_importances}')

        # Step 4: log component importance scores and total attribution
        mlflow.log_dict(component_importances, "component_importances.json")
        mlflow.log_metric("total_attribution", float(np.sum(attributions_ig_sum)))

        ''' 5. Build argumentation framework '''
        argumentation = build_argumentation_framework(claim=predicted_class_name, all_components=components, component_importances=component_importances)
    
        supports = [a for a in argumentation["arguments"] if a["relation"] == "support"]
        attacks  = [a for a in argumentation["arguments"] if a["relation"] == "attack"]
    
        print("\nArgumentation framework:")
        print(f"Claim: '{argumentation['claim']}'")
        print(f"\nSupporting arguments ({len(supports)} detected components):")
        for a in supports:
            print(f" [+] {a['component']}  (IG={a['weight']:.2f})")
        print(f"\nAttacking arguments ({len(attacks)} missing components):")
        for a in attacks:
            print(f" [-] {a['component']}")

        # Step 5: log argumentation framework details
        mlflow.log_metric("num_support_args", len(supports))
        mlflow.log_metric("num_attack_args", len(attacks))
        mlflow.log_dict(argumentation, "argumentation_framework.json")

        ''' 6. Generate Explanation'''
        supported = "SUPPORTED" if argumentation["accepted"] else "UNSUPPORTED"
        attention_scores = compute_attention_scores(component_importances, attributions_ig_sum)
        explanation = generate_explanation(argumentation, attention_scores, supported)
        print(f"Explanation : {explanation}")

        # Step 6: log final explanation text
        mlflow.log_param("result", supported)
        mlflow.log_text(explanation, "explanation.txt")
        mlflow.log_dict(attention_scores, "attention_scores.json")

    #TODO: visualize the bounding boxes on the original image
    #TODO: visualize the argumentation framework as a graph, with support and attack edges, and confidence score.
    #TODO: add more examples with different images and classes, and compare the explanations generated by the framework.