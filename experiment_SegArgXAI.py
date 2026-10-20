import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
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
IMAGE_PATH = "data/baldeagle1.jpg"
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

# Run segmentation — accepts an explicit image_path so it can operate on the cropped image
def run_segmentation_and_grounding(caption: str, image_path: str):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_source, image_tensor = load_image(image_path)

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
 
def build_arguments(
        all_components: list,
        component_importances: dict,
        ) -> list:
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

def generate_explanation(
        argumentation: dict,
        attention_scores: dict,
        supported: str,
        ) -> str:
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

def visualize_argumentation_graph(
    argumentation: dict,
    attention_scores: dict,
) -> plt.Figure:
    claim = argumentation["claim"]
    arguments = argumentation["arguments"]

    supports = [a for a in arguments if a["relation"] == "support"]
    attacks  = [a for a in arguments if a["relation"] == "attack"]

    graph = nx.DiGraph()
    graph.add_node(claim)
    for a in arguments:
        graph.add_node(a["component"])
        graph.add_edge(a["component"], claim)

    # Claim on top, supports on the bottom-left, attacks on the bottom-right —
    # widths are proportional to how many of each so the claim stays centered
    # over the full row even when one side is empty.
    total = max(len(supports) + len(attacks), 1)
    support_frac = len(supports) / total
    attack_frac = len(attacks) / total

    pos = {claim: (0.5, 1.0)}
    for i, a in enumerate(supports):
        pos[a["component"]] = ((i + 0.5) / max(len(supports), 1) * support_frac, 0.0)
    for i, a in enumerate(attacks):
        pos[a["component"]] = (support_frac + (i + 0.5) / max(len(attacks), 1) * attack_frac, 0.0)

    node_colors = {claim: "steelblue"}
    for a in supports:
        node_colors[a["component"]] = "seagreen"
    for a in attacks:
        node_colors[a["component"]] = "firebrick"

    edge_labels = {}
    for a in supports:
        edge_labels[(a["component"], claim)] = f"{attention_scores.get(a['component'], 0.0):.1%}"
    for a in attacks:
        edge_labels[(a["component"], claim)] = "absent"

    fig, ax = plt.subplots(figsize=(max(6, len(arguments) * 1.4), 5))

    nx.draw_networkx_nodes(
        graph, pos, ax=ax,
        node_color=[node_colors[node] for node in graph.nodes()],
        node_size=2200,
    )
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=7, font_color="white")

    nx.draw_networkx_edges(
        graph, pos, ax=ax, edgelist=[(a["component"], claim) for a in supports],
        edge_color="seagreen", arrowstyle="-|>", arrowsize=20, node_size=2200,
        connectionstyle="arc3,rad=0.1",
    )
    nx.draw_networkx_edges(
        graph, pos, ax=ax, edgelist=[(a["component"], claim) for a in attacks],
        edge_color="firebrick", arrowstyle="-|>", arrowsize=20, node_size=2200,
        connectionstyle="arc3,rad=0.1",
    )
    nx.draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=edge_labels, font_size=7)

    ax.set_title(f"Argumentation graph for claim: '{claim}'")
    ax.set_ylim(-0.15, 1.15)
    ax.axis("off")

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="seagreen", markersize=12, label="Support (detected)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="firebrick", markersize=12, label="Attack (undetected)"),
    ]
    ax.legend(handles=legend_handles, loc="lower center", ncol=2, frameon=False)

    return fig

def visualize_boxes(
    image,
    scaled_boxes: list,
    title: str,
    box_color: str = "cyan",
    labels: list = None,
    cmap: str = None,
) -> plt.Figure:
    fig, ax = plt.subplots(tight_layout=True)
    ax.imshow(image, cmap=cmap)
    for i, box in enumerate(scaled_boxes):
        x0, y0, x1, y1 = box
        ax.add_patch(
            plt.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor=box_color, fill=False, lw=2)
        )
        if labels:
            ax.text(x0, y0 - 4, labels[i], color=box_color, fontsize=8,
                    bbox=dict(facecolor="black", alpha=0.5, pad=1))
    ax.axis("off")
    #plt.show()
    return fig

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

        ''' 0. Crop image to model input space (shared by ResNet, IG, and DINO) '''
        img_original = Image.open(IMAGE_PATH)

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])
        img_cropped = preprocess(img_original)

        CROPPED_IMAGE_PATH = IMAGE_PATH.replace(".", "_cropped.")
        img_cropped.save(CROPPED_IMAGE_PATH)

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
        print(f"\nPredicted class: {predicted_class_name}")

        # Step 1: log classification results
        mlflow.log_param("predicted_class", predicted_class_name)
        mlflow.log_param("predicted_class_id", predicted_class_id)

        ''' 2. Get list of components with LLM '''
        components = get_components(predicted_class_name)
        print(f'\nExpected components: {components}')
        
        dino_caption = " . ".join(c.lower() for c in components)

        # Step 2: log components and formatted caption for grounding dino
        mlflow.log_param("components_expected", components)
        mlflow.log_metric("num_components_expected", len(components))
        mlflow.log_param("dino_caption", dino_caption)

        ''' 3. Check component presence — DINO runs on the cropped image '''
        boxes, logits, phrases = run_segmentation_and_grounding(dino_caption, CROPPED_IMAGE_PATH)
        detected_components_boxes = "\n".join(f"{label}: {box}" for label, box in zip(phrases, boxes))
        print(f'\nDetected components: {phrases}')

        # Step 3: logging
        mlflow.log_metric("num_components_detected", len(phrases))
        mlflow.log_param("components_detected", phrases)
        mlflow.log_text(detected_components_boxes, "detected_components_boxes.txt")
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
        ax.axis("off")
        with tempfile.TemporaryDirectory() as tmp:
            heatmap_path = os.path.join(tmp, "ig_heatmap.png")
            fig.savefig(heatmap_path, bbox_inches="tight", pad_inches=0)
            mlflow.log_artifact(heatmap_path)
        #plt.show()

        ''' 4b. Quantify the component importance '''
        # DINO boxes are normalized to the 224x224 cropped image — scale directly to pixel space
        scaled_boxes = []
        for box in boxes:
            x_center, y_center, width, height = box.tolist()
        
            x0 = int(max(0, min(224, (x_center - width / 2) * 224)))
            y0 = int(max(0, min(224, (y_center - height / 2) * 224)))
            x1 = int(max(0, min(224, (x_center + width / 2) * 224)))
            y1 = int(max(0, min(224, (y_center + height / 2) * 224)))
        
            scaled_boxes.append([x0, y0, x1, y1])

        # Visualize and log the images with detected boxes overlaid as an artifact
        fig2 = visualize_boxes(
            image=attributions_ig_sum,
            scaled_boxes=scaled_boxes,
            title="Detected components on attribution heatmap",
            box_color="cyan",
            labels=phrases,
            cmap="hot",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ig_boxes.png")
            fig2.savefig(path, bbox_inches="tight", pad_inches=0)
            mlflow.log_artifact(path)

        fig3 = visualize_boxes(
            image=img_cropped,
            scaled_boxes=scaled_boxes,
            title="Detected components on image (224×224)",
            box_color="lime",
            labels=phrases,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "original_image_boxes.png")
            fig3.savefig(path, bbox_inches="tight", pad_inches=0)
            mlflow.log_artifact(path)

        individual_component_importance = []
        component_pixel_masks = {}

        for label, box in zip(phrases, scaled_boxes):
            x0, y0, x1, y1 = box

            # Ensure box is within bounds
            x0 = max(x0, 0)
            y0 = max(y0, 0)
            x1 = min(x1, attributions_ig_sum.shape[1])
            y1 = min(y1, attributions_ig_sum.shape[0])

            # Crop attribution map
            attribution_crop = attributions_ig_sum[y0:y1, x0:x1]

            # single component importance is the sum of attributions within the box
            importance_score = float(np.abs(attribution_crop).sum())
            individual_component_importance.append((label, importance_score))

            # Union the pixels covered by this box into the label's mask, so that
            # duplicate/overlapping detections of the same component (GroundingDINO
            # frequently returns more than one box per phrase) don't get their
            # shared pixels counted twice when we aggregate importance below.
            mask = component_pixel_masks.setdefault(label, np.zeros_like(attributions_ig_sum, dtype=bool))
            mask[y0:y1, x0:x1] = True

        # Aggregate importance once per label from the deduplicated pixel mask
        aggregated_component_importance = {
            label: float(attributions_ig_sum[mask].sum())
            for label, mask in component_pixel_masks.items()
        }

        print(f'\nImportance scores: {aggregated_component_importance}')

        # Step 4: log component importance scores and total attribution
        mlflow.log_dict(aggregated_component_importance, "aggregated_component_importance.json")
        mlflow.log_text("\n".join([f"{label}: {score}" for label, score in individual_component_importance]), "individual_component_importance.txt")
        mlflow.log_metric("total_attribution", float(np.sum(attributions_ig_sum)))

        ''' 5. Build argumentation framework '''
        argumentation = build_argumentation_framework(claim=predicted_class_name, all_components=components, component_importances=aggregated_component_importance)
    
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
        attention_scores = compute_attention_scores(aggregated_component_importance, attributions_ig_sum)
        explanation = generate_explanation(argumentation, attention_scores, supported)
        print(f"\nExplanation : {explanation}")

        # Step 6: log final explanation text
        mlflow.log_param("result", supported)
        mlflow.log_text(explanation, "explanation.txt")
        mlflow.log_dict(attention_scores, "attention_scores.json")

        ''' 7. Visualize the argumentation framework as a graph '''
        fig4 = visualize_argumentation_graph(argumentation, attention_scores)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "argumentation_graph.png")
            fig4.savefig(path, bbox_inches="tight", pad_inches=0.2)
            mlflow.log_artifact(path)