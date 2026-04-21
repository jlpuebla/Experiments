"""
LIME Baseline — XAI Comparison Suite
======================================
Runs LIME on ResNet-50 for a fair comparison.

LIME perturbs superpixel segments of the input image and fits a local linear
model to approximate the classifier's decision boundary around that input.
Positive-contributing segments are highlighted; negative ones are suppressed.

Usage:
    python baseline_lime.py <image_path>
    python baseline_lime.py <image_path> --num-samples 1000   # default
    python baseline_lime.py <image_path> --num-segments 50    # default
    python baseline_lime.py <image_path> --topk 5

Output:
    - <image_stem>_lime.png     : LIME explanation overlay (positive segments)
    - <image_stem>_original.png : preprocessed original image
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
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

IMAGE_PATH = "data/schoolbus.png"  # default input image (can be overridden via CLI)

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

# LIME operates on raw uint8 numpy images — no normalisation here.
# Normalisation is applied inside the batch_predict wrapper.
RENDER_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
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
    """Load pretrained ResNet-50 — identical backbone to SegArgXAI."""
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.eval()
    model.to(device)
    return model


def preprocess_image(image_path: str, device: torch.device) -> tuple[torch.Tensor, np.ndarray]:
    """
    Returns:
        tensor  : (1, 3, 224, 224) normalised input for inference
        rgb_img : (224, 224, 3) uint8 numpy array — used as LIME's input image
    """
    img = Image.open(image_path).convert("RGB")
    tensor = PREPROCESS(img).unsqueeze(0).to(device)

    # LIME needs a uint8 (H, W, 3) image — no normalisation
    rgb_img = np.array(RENDER_TRANSFORM(img))   # (224, 224, 3), uint8
    return tensor, rgb_img


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


def make_batch_predict(model: models.ResNet, device: torch.device):
    """
    Returns a function that LIME can call: (N, H, W, 3) uint8 → (N, num_classes) float32.

    LIME perturbs the image by zeroing out superpixel segments and needs a
    classifier function that accepts a batch of numpy images and returns class
    probabilities. This closure captures the model and handles the uint8→tensor
    conversion + ImageNet normalisation internally.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    def batch_predict(images: np.ndarray) -> np.ndarray:
        # images: (N, H, W, 3) uint8
        batch = torch.stack([
            normalize(transforms.ToTensor()(Image.fromarray(img)))
            for img in images
        ]).to(device)

        with torch.no_grad():
            logits = model(batch)
            probs = F.softmax(logits, dim=1)

        return probs.cpu().numpy()   # (N, 1000)

    return batch_predict


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def run_lime(
    image_path: str,
    num_samples: int = 1000,
    num_segments: int = 50,
    topk: int = 5,
) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LIME] Device: {device}")

    # --- Setup ---
    print("[LIME] Loading model (ResNet-50, ImageNet pretrained)...")
    model = load_model(device)
    labels = load_imagenet_labels()

    tensor, rgb_img = preprocess_image(image_path, device)

    # --- Inference ---
    predictions = run_inference(model, tensor, labels, topk)
    top_pred = predictions[0]

    print(f"\n[LIME] Top-{topk} predictions:")
    for p in predictions:
        bar = "█" * int(p["confidence"] * 30)
        print(f"  {p['rank']:>2}. {p['label']:<30} {p['confidence']:.2%}  {bar}")

    # --- LIME ---
    # Segmentation: quickshift over the rendered (non-normalised) image.
    # num_segments controls granularity — ~50 matches a typical semantic
    # component count without over-fragmenting.
    print(f"\n[LIME] Running explanation "
          f"(num_samples={num_samples}, num_segments={num_segments})...")
    print(f"[LIME] Target class: '{top_pred['label']}' (id={top_pred['class_id']})")

    explainer = lime_image.LimeImageExplainer()
    segmenter = SegmentationAlgorithm(
        "quickshift",
        kernel_size=4,
        max_dist=200,
        ratio=0.2,
    )
    batch_predict = make_batch_predict(model, device)

    explanation = explainer.explain_instance(
        image=rgb_img,
        classifier_fn=batch_predict,
        top_labels=1,                       # explain only top predicted class
        hide_color=0,                       # occlude hidden segments with black
        num_samples=num_samples,
        segmentation_fn=segmenter,
    )

    # --- Visualisation ---
    # Positive-only mode: show only the superpixels that support the prediction.
    # This mirrors GradCAM's convention of highlighting "what the model looks at."
    temp_img, mask = explanation.get_image_and_mask(
        label=top_pred["class_id"],
        positive_only=True,
        num_features=10,
        hide_rest=True,       # black out non-contributing segments
        )
    # Blend: highlighted segments at full brightness, rest dimmed to 30%
    highlighted = rgb_img.astype(np.float32) / 255.0
    dimmed = highlighted * 0.3
    blended = np.where(mask[:, :, np.newaxis], highlighted, dimmed)
    overlay = mark_boundaries(blended, mask, color=(1, 1, 0))  # yellow borders
    overlay_u8 = (overlay * 255).astype(np.uint8)

    original_u8 = rgb_img                                        # already uint8

    src = Path(image_path)
    heatmap_path = src.parent / f"{src.stem}_lime.png"
    original_path = src.parent / f"{src.stem}_original.png"

    cv2.imwrite(str(heatmap_path), cv2.cvtColor(overlay_u8, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(original_path), cv2.cvtColor(original_u8, cv2.COLOR_RGB2BGR))
    print(f"\n[LIME] Saved heatmap  → {heatmap_path}")
    print(f"[LIME] Saved original → {original_path}")

    # --- MLflow ---
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("LIME")
    with mlflow.start_run():
        mlflow.log_param("attribution_method", "LIME")
        mlflow.log_param("image_path", image_path)
        mlflow.log_param("predicted_class", top_pred["label"])
        mlflow.log_param("predicted_class_id", top_pred["class_id"])
        mlflow.log_param("num_samples", num_samples)
        mlflow.log_param("num_segments", num_segments)
        mlflow.log_metric("top1_confidence", top_pred["confidence"])
        mlflow.log_metric("surrogate_r2", explanation.score)
        mlflow.log_artifact(str(heatmap_path))
        mlflow.log_artifact(str(original_path))
    print("[LIME] MLflow run logged.")

    return heatmap_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LIME baseline over ResNet-50 (XAI comparison suite)"
    )
    parser.add_argument("image", nargs="?", default=IMAGE_PATH,
                        help="Path to input image (default: IMAGE_PATH)")
    parser.add_argument(
        "--num-samples", type=int, default=1000,
        help="LIME perturbation samples (default: 1000; higher = more stable)",
    )
    parser.add_argument(
        "--num-segments", type=int, default=50,
        help="Quickshift superpixel count (default: 50)",
    )
    parser.add_argument(
        "--topk", type=int, default=5,
        help="Number of top predictions to display (default: 5)",
    )
    args = parser.parse_args()
    run_lime(
        args.image,
        num_samples=args.num_samples,
        num_segments=args.num_segments,
        topk=args.topk,
    )


if __name__ == "__main__":
    main()
