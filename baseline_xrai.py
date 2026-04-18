"""
XRAI Baseline — XAI Comparison Suite
======================================
Runs XRAI on ResNet-50 for a fair comparison.

XRAI (eXplanation with Ranked Area Inlays) extends Integrated Gradients to
the region level. Rather than attributing importance to individual pixels,
XRAI aggregates pixel-level IG scores over image segments (via SLIC), then
ranks those segments from most to least salient. The result is a region-level
attribution map that answers: "which parts of the image — as coherent regions —
most influenced the model's prediction?"

Three outputs capture XRAI's distinct contributions:
  1. Heatmap overlay  : pixel-level IG scores colormapped and blended onto original
  2. Inlay image      : XRAI's signature — top-ranked segments revealed on a dark
                        background, colored by rank (red = most salient)
  3. Segments overlay : segmentation boundaries drawn on the original image,
                        each segment tinted by its attribution rank

Usage:
    python baseline_xrai.py <image_path>
    python baseline_xrai.py <image_path> --num-segments 50        # SLIC granularity
    python baseline_xrai.py <image_path> --topk-segments 10       # segments to highlight
    python baseline_xrai.py <image_path> --algorithm full         # more accurate, slower
    python baseline_xrai.py <image_path> --topk 5                 # top-k predictions

Output:
    - <stem>_xrai_heatmap.png   : pixel-level IG saliency overlaid on image
    - <stem>_xrai_inlay.png     : ranked area inlays (XRAI's signature visual)
    - <stem>_xrai_segments.png  : segment boundaries colored by attribution rank
    - <stem>_original.png       : preprocessed original image
    - Console: top-k predictions + ranked segment attribution table
    - MLflow: all artifacts + segment_attributions JSON
"""

import argparse
import json
import urllib.request
from pathlib import Path

import cv2
import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

import saliency.core as saliency


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

IMAGE_PATH = "data/schoolbus.png"

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

# XRAI works on raw float images in [0, 1]; normalisation is applied inside
# the call_model_function so the model still receives properly normalised input.
RENDER_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
])

NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_imagenet_labels() -> list[str]:
    with urllib.request.urlopen(IMAGENET_LABELS_URL) as resp:
        return json.loads(resp.read().decode())


def load_model(device: torch.device) -> models.ResNet:
    """Pretrained ResNet-50 — identical backbone to SegArgXAI."""
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.eval()
    model.to(device)
    return model


def preprocess_image(image_path: str, device: torch.device) -> tuple[torch.Tensor, np.ndarray]:
    """
    Returns:
        tensor    : (1, 3, 224, 224) normalised tensor for inference
        float_img : (224, 224, 3) float32 array in [0, 1] for XRAI
    """
    img = Image.open(image_path).convert("RGB")
    tensor = PREPROCESS(img).unsqueeze(0).to(device)
    float_img = np.array(RENDER_TRANSFORM(img)).astype(np.float32) / 255.0
    return tensor, float_img


def run_inference(
    model: models.ResNet,
    tensor: torch.Tensor,
    labels: list[str],
    topk: int,
) -> list[dict]:
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


def make_call_model_function(model: models.ResNet, device: torch.device, class_id: int):
    """
    Wraps ResNet-50 for the saliency library's expected interface.

    XRAI uses Integrated Gradients internally: at each interpolation step along
    the path from baseline (black image) to input, it queries this function for
    d(logit_class_id) / d(input_pixels). Those gradients are then integrated
    and aggregated per segment.

    The function signature is fixed by the saliency library:
        fn(images: np.ndarray, call_model_args, expected_keys) -> dict
    where images is (N, H, W, 3) float32 in [0, 1].
    """
    def call_model_fn(images: np.ndarray, call_model_args=None, expected_keys=None):
        # Convert (N, H, W, 3) float32 → normalised (N, 3, H, W) tensor
        images_t = torch.stack([
            NORMALIZE(torch.tensor(img, dtype=torch.float32).permute(2, 0, 1))
            for img in images
        ]).to(device).requires_grad_(True)

        logits = model(images_t)
        target = logits[:, class_id].sum()

        grads = torch.autograd.grad(target, images_t)[0]  # (N, 3, H, W)
        grads_np = grads.detach().cpu().numpy().transpose(0, 2, 3, 1)  # (N, H, W, 3)

        return {saliency.base.INPUT_OUTPUT_GRADIENTS: grads_np}

    return call_model_fn


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def make_heatmap_overlay(float_img: np.ndarray, attr_map: np.ndarray) -> np.ndarray:
    """
    Standard saliency heatmap: pixel-level IG scores colormapped (JET) and
    alpha-blended onto the original image.

    Returns uint8 BGR image.
    """
    attr_norm = (attr_map - attr_map.min()) / (attr_map.max() - attr_map.min() + 1e-8)
    heatmap = cv2.applyColorMap((attr_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    original_bgr = cv2.cvtColor((float_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(original_bgr, 0.5, heatmap, 0.5, 0)


def make_inlay_image(
    float_img: np.ndarray,
    attr_map: np.ndarray,
    segments: np.ndarray,
    segment_scores: dict[int, float],
    topk_segments: int,
) -> np.ndarray:
    """
    XRAI's signature visualisation: ranked area inlays.

    The top-ranked segments are revealed on a darkened background, each tinted
    from red (most salient) through yellow to green (least salient among shown).
    This directly communicates which regions matter most and in what order —
    the core claim of the XRAI paper.

    Returns uint8 BGR image.
    """
    original_rgb = (float_img * 255).astype(np.uint8)

    # Dark background
    canvas = (original_rgb.astype(np.float32) * 0.15).astype(np.uint8)

    # Rank segments by score, take top-k
    ranked = sorted(segment_scores.items(), key=lambda x: x[1], reverse=True)
    top_segments = ranked[:topk_segments]

    # Color ramp: red (rank 1) → yellow → green (rank N)
    # HSV hue: 0° = red, 60° = yellow, 120° = green
    for rank_idx, (seg_id, _score) in enumerate(top_segments):
        t = rank_idx / max(len(top_segments) - 1, 1)   # 0.0 → 1.0
        hue = int(t * 120)  # 0 (red) → 120 (green)
        color_hsv = np.array([[[hue // 2, 220, 255]]], dtype=np.uint8)  # OpenCV hue is /2
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()

        mask = segments == seg_id
        # Blend original pixel content with rank color
        region = canvas[mask].astype(np.float32)
        tint = np.array(color_bgr, dtype=np.float32)
        canvas[mask] = (region * 0.4 + tint * 0.6).astype(np.uint8)

        # Draw segment boundary
        seg_mask_u8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(seg_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, color_bgr, 1)

    return canvas


def make_segments_overlay(
    float_img: np.ndarray,
    segments: np.ndarray,
    segment_scores: dict[int, float],
) -> np.ndarray:
    """
    All segments drawn on the original image with boundaries colored by
    attribution rank. High-scoring segments get warmer boundary colors.

    Returns uint8 BGR image.
    """
    original_bgr = cv2.cvtColor((float_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    canvas = original_bgr.copy()

    ranked = sorted(segment_scores.items(), key=lambda x: x[1], reverse=True)
    n = len(ranked)

    for rank_idx, (seg_id, _score) in enumerate(ranked):
        t = rank_idx / max(n - 1, 1)
        hue = int(t * 120)
        color_hsv = np.array([[[hue // 2, 200, 255]]], dtype=np.uint8)
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()

        mask = (segments == seg_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, color_bgr, 1)

    return canvas


# ---------------------------------------------------------------------------
# Segment attribution extraction
# ---------------------------------------------------------------------------

def extract_segment_scores(
    attr_map: np.ndarray,
    segments: np.ndarray,
) -> dict[int, float]:
    """
    Compute mean attribution per segment. This is the scalar score XRAI
    uses to rank regions — the quantity that drives the inlay ordering.
    """
    scores = {}
    for seg_id in np.unique(segments):
        mask = segments == seg_id
        scores[int(seg_id)] = float(attr_map[mask].mean())
    return scores


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def run_xrai(
    image_path: str,
    num_segments: int = 50,
    topk_segments: int = 10,
    algorithm: str = "fast",
    topk: int = 5,
) -> dict[str, Path]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[XRAI] Device: {device}")

    # --- Setup ---
    print("[XRAI] Loading model (ResNet-50, ImageNet pretrained)...")
    model = load_model(device)
    labels = load_imagenet_labels()
    tensor, float_img = preprocess_image(image_path, device)

    # --- Inference ---
    predictions = run_inference(model, tensor, labels, topk)
    top_pred = predictions[0]

    print(f"\n[XRAI] Top-{topk} predictions:")
    for p in predictions:
        bar = "█" * int(p["confidence"] * 30)
        print(f"  {p['rank']:>2}. {p['label']:<30} {p['confidence']:.2%}  {bar}")

    # --- XRAI attribution ---
    print(f"\n[XRAI] Running explanation "
          f"(algorithm={algorithm}, num_segments={num_segments})...")
    print(f"[XRAI] Target class: '{top_pred['label']}' (id={top_pred['class_id']})")

    call_model_fn = make_call_model_function(model, device, top_pred["class_id"])

    xrai_object = saliency.XRAI()
    xrai_params = saliency.XRAIParameters()
    xrai_params.algorithm = algorithm
    xrai_params.num_segments = num_segments

    # attr_map : (H, W) float32  — pixel-level aggregated IG scores
    # segments : (H, W) int      — SLIC segment label map (available via extra_returns)
    xrai_output = xrai_object.GetMask(
        x_value=float_img,
        call_model_function=call_model_fn,
        extra_parameters=xrai_params,
    )

    # GetMask returns the XRAI map directly; raw IG is in xrai_output.ig_attributions
    # when return_ig_attributions=True — collapse channels to scalar per pixel
    attr_map = xrai_output  # (H, W) — XRAI region-level map

    # Re-run SLIC to get segment labels (same params XRAI used internally)
    from skimage.segmentation import slic
    segments = slic(
        float_img,
        n_segments=num_segments,
        compactness=10,
        sigma=1,
        start_label=0,
    )

    segment_scores = extract_segment_scores(attr_map, segments)

    # --- Print ranked segment table ---
    ranked = sorted(segment_scores.items(), key=lambda x: x[1], reverse=True)
    print(f"\n[XRAI] Top-{topk_segments} segments by mean attribution:")
    print(f"  {'Rank':<6} {'Segment ID':<12} {'Mean Attribution':>18}")
    print(f"  {'-'*6} {'-'*12} {'-'*18}")
    for i, (seg_id, score) in enumerate(ranked[:topk_segments], 1):
        bar = "█" * int(max(score, 0) * 40)
        print(f"  {i:<6} {seg_id:<12} {score:>18.6f}  {bar}")

    # --- Visualisations ---
    heatmap_img   = make_heatmap_overlay(float_img, attr_map)
    inlay_img     = make_inlay_image(float_img, attr_map, segments, segment_scores, topk_segments)
    segments_img  = make_segments_overlay(float_img, segments, segment_scores)
    original_bgr  = cv2.cvtColor((float_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    src = Path(image_path)
    paths = {
        "heatmap":  src.parent / f"{src.stem}_xrai_heatmap.png",
        "inlay":    src.parent / f"{src.stem}_xrai_inlay.png",
        "segments": src.parent / f"{src.stem}_xrai_segments.png",
        "original": src.parent / f"{src.stem}_original.png",
    }

    cv2.imwrite(str(paths["heatmap"]),  heatmap_img)
    cv2.imwrite(str(paths["inlay"]),    inlay_img)
    cv2.imwrite(str(paths["segments"]), segments_img)
    cv2.imwrite(str(paths["original"]), original_bgr)

    for key, path in paths.items():
        print(f"[XRAI] Saved {key:<10} → {path}")

    # --- MLflow ---
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("XRAI")
    with mlflow.start_run():
        mlflow.log_param("attribution_method", "XRAI")
        mlflow.log_param("image_path", image_path)
        mlflow.log_param("predicted_class", top_pred["label"])
        mlflow.log_param("predicted_class_id", top_pred["class_id"])
        mlflow.log_param("num_segments", num_segments)
        mlflow.log_param("topk_segments", topk_segments)
        mlflow.log_param("algorithm", algorithm)
        mlflow.log_metric("top1_confidence", top_pred["confidence"])
        mlflow.log_metric("attribution_mean", float(attr_map.mean()))
        mlflow.log_metric("attribution_max",  float(attr_map.max()))
        mlflow.log_dict(
            {str(k): v for k, v in segment_scores.items()},
            "segment_attributions.json",
        )
        for path in paths.values():
            mlflow.log_artifact(str(path))
    print("[XRAI] MLflow run logged.")

    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="XRAI baseline over ResNet-50 (XAI comparison suite)"
    )
    parser.add_argument("image", nargs="?", default=IMAGE_PATH,
                        help="Path to input image (default: IMAGE_PATH)")
    parser.add_argument(
        "--num-segments", type=int, default=50,
        help="SLIC superpixel count — controls region granularity (default: 50)",
    )
    parser.add_argument(
        "--topk-segments", type=int, default=10,
        help="Number of top segments to highlight in the inlay image (default: 10)",
    )
    parser.add_argument(
        "--algorithm", choices=["fast", "full"], default="fast",
        help="XRAI algorithm variant: 'fast' (one IG pass) or 'full' (per-merge IG, slower)",
    )
    parser.add_argument(
        "--topk", type=int, default=5,
        help="Number of top predictions to display (default: 5)",
    )
    args = parser.parse_args()
    run_xrai(
        args.image,
        num_segments=args.num_segments,
        topk_segments=args.topk_segments,
        algorithm=args.algorithm,
        topk=args.topk,
    )


if __name__ == "__main__":
    main()
