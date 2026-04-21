"""
GradCAM Baseline — XAI Comparison Suite
=========================================
Runs Grad-CAM on ResNet-50 for a fair comparison.

Usage:
    python gradcam_baseline.py <image_path>
    python gradcam_baseline.py <image_path> --layer layer4   # default
    python gradcam_baseline.py <image_path> --topk 5

Output:
    - <image_stem>_gradcam.png  : side-by-side original | heatmap overlay
    - Console: top-k predictions with confidence scores
"""

import argparse
import json
from pathlib import Path

import mlflow

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

IMAGE_PATH = "data/schoolbus.png"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MLFLOW_TRACKING_URI = "mlruns"

IMAGENET_LABELS_URL = (
    "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels"
    "/master/imagenet-simple-labels.json"
)

PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_imagenet_labels() -> list[str]:
    """Load human-readable ImageNet class labels."""
    import urllib.request
    with urllib.request.urlopen(IMAGENET_LABELS_URL) as resp:
        return json.loads(resp.read().decode())


def load_model(device: torch.device) -> models.ResNet:
    """Load pretrained ResNet-50"""
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.eval()
    model.to(device)
    return model


def preprocess_image(image_path: str, device: torch.device) -> tuple[torch.Tensor, np.ndarray]:
    """
    Returns:
        tensor : (1, 3, 224, 224) normalised input for the model
        rgb_img: (224, 224, 3) float32 in [0, 1] for overlay rendering
    """
    img = Image.open(image_path).convert("RGB")
    tensor = PREPROCESS(img).unsqueeze(0).to(device)

    # Render-ready version (no normalisation, resized to 224)
    render_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    rgb_img = render_tf(img).permute(1, 2, 0).numpy().astype(np.float32)
    return tensor, rgb_img


def get_target_layer(model: models.ResNet, layer_name: str):
    """Return the conv layer used as the GradCAM target."""
    layer_map = {
        "layer1": model.layer1[-1],
        "layer2": model.layer2[-1],
        "layer3": model.layer3[-1],
        "layer4": model.layer4[-1],   # default — deepest semantic features
    }
    if layer_name not in layer_map:
        raise ValueError(f"Unknown layer '{layer_name}'. Choose from: {list(layer_map)}")
    return layer_map[layer_name]


def run_inference(model, tensor: torch.Tensor, labels: list[str], topk: int) -> list[dict]:
    """Return top-k predictions as [{rank, class_id, label, confidence}]."""
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    top = torch.topk(probs, topk)
    return [
        {
            "rank": i + 1,
            "class_id": idx.item(),
            "label": labels[idx.item()] if labels else str(idx.item()),
            "confidence": prob.item(),
        }
        for i, (prob, idx) in enumerate(zip(top.values, top.indices))
    ]


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def run_gradcam(
    image_path: str,
    layer_name: str = "layer4",
    topk: int = 5,
) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[GradCAM] Device: {device}")

    # --- Setup ---
    print("[GradCAM] Loading model (ResNet-50, ImageNet pretrained)...")
    model = load_model(device)
    labels = load_imagenet_labels()

    target_layer = get_target_layer(model, layer_name)
    tensor, rgb_img = preprocess_image(image_path, device)

    # --- Inference ---
    predictions = run_inference(model, tensor, labels, topk)
    top_pred = predictions[0]

    print(f"\n[GradCAM] Top-{topk} predictions:")
    for p in predictions:
        bar = "█" * int(p["confidence"] * 30)
        print(f"  {p['rank']:>2}. {p['label']:<30} {p['confidence']:.2%}  {bar}")

    # --- GradCAM ---
    # Target: gradient of the top-predicted class score w.r.t. target layer activations.
    # No explicit ClassifierOutputTarget needed — GradCAM defaults to argmax (top class).
    print(f"\n[GradCAM] Computing GradCAM on '{layer_name}' for class '{top_pred['label']}'...")

    cam_engine = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam_engine(input_tensor=tensor, targets=None)   # None → top class
    grayscale_cam = grayscale_cam[0]   # (224, 224), values in [0, 1]

    # --- Visualisation ---
    overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    original_u8 = (rgb_img * 255).astype(np.uint8)

    src = Path(image_path)
    heatmap_path = src.parent / f"{src.stem}_gradcam_{layer_name}.png"
    original_path = src.parent / f"{src.stem}_original.png"

    cv2.imwrite(str(heatmap_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(original_path), cv2.cvtColor(original_u8, cv2.COLOR_RGB2BGR))
    print(f"\n[GradCAM] Saved heatmap  → {heatmap_path}")
    print(f"[GradCAM] Saved original → {original_path}")

    # --- MLflow ---
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("GradCAM")
    with mlflow.start_run():
        mlflow.log_param("attribution_method", "GradCAM")
        mlflow.log_param("image_path", image_path)
        mlflow.log_param("predicted_class", top_pred["label"])
        mlflow.log_param("predicted_class_id", top_pred["class_id"])
        mlflow.log_metric("top1_confidence", top_pred["confidence"])
        mlflow.log_artifact(str(heatmap_path))
        mlflow.log_artifact(str(original_path))
    print("[GradCAM] MLflow run logged.")

    return heatmap_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GradCAM baseline over ResNet-50 (XAI comparison suite)"
    )
    parser.add_argument("image", nargs="?", default=IMAGE_PATH, help="Path to input image (default: IMAGE_PATH)")
    parser.add_argument(
        "--layer", default="layer4",
        choices=["layer1", "layer2", "layer3", "layer4"],
        help="ResNet-50 layer to hook (default: layer4)",
    )
    parser.add_argument(
        "--topk", type=int, default=5,
        help="Number of top predictions to display (default: 5)",
    )
    args = parser.parse_args()
    run_gradcam(args.image, layer_name=args.layer, topk=args.topk)


if __name__ == "__main__":
    main()
