"""Run WildDet3D inference with OWLv2 auto-detection (no manual text prompt needed).

OWLv2 detects 2D bounding boxes + class labels from the lvisplus vocabulary.
Those boxes and labels are passed together to WildDet3D as VISUAL+LABEL or
GEOMETRY+LABEL prompts in a single forward pass.

Data layout expected
--------------------
    ~/data/<dataset>/<scene>/
        images/          original images (may contain sub-directories)
        sparse/          COLMAP sparse reconstruction

Prerequisite
------------
    Run extract_intrinsics.py first:
        python extract_intrinsics.py <dataset> <scene> <image>

    Place the OWLv2 checkpoint at:
        <project>/ckpt/owlv2-base-patch16-ensemble.pt

Usage
-----
    python wildDet3D_infer_auto.py <dataset> <scene> <image> [options]

Outputs  (written to <project>/outputs/<dataset>/<scene>/3Dresults/auto_owlv2/)
--------
    <stem>_boxes3d.npy
    <stem>_depth_map.npy
    <stem>_class_ids.npy      indices into unique_labels.txt
    <stem>_output.png
    unique_labels.txt         comma-separated unique detected class names
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from wilddet3d import build_model, preprocess
from wilddet3d.data_types import WildDet3DInput
from wilddet3d.vis.visualize import draw_3d_boxes
from owl import OwlWrapper

DATA_ROOT  = Path.home() / "data"
CHECKPOINT = PROJECT_ROOT / "ckpt" / "wilddet3d_alldata_all_prompt_v1.0.pt"
AUTO_FOLDER = "auto_owlv2"


# ---------------------------------------------------------------------------
# COLMAP helpers (copied from wildDet3D_infer.py — kept self-contained)
# ---------------------------------------------------------------------------

def _find_sparse_dir(colmap_root: Path) -> Path:
    if (colmap_root / "cameras.bin").exists():
        return colmap_root
    for sub in sorted(colmap_root.iterdir()):
        if sub.is_dir() and (sub / "cameras.bin").exists():
            return sub
    raise FileNotFoundError(f"cameras.bin not found under {colmap_root}")


def _read_images_bin(path: Path) -> dict[str, int]:
    name_to_cam: dict[str, int] = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            f.read(4)
            f.read(32)
            f.read(24)
            (cam_id,) = struct.unpack("<I", f.read(4))
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            (num_pts,) = struct.unpack("<Q", f.read(8))
            f.read(num_pts * 24)
            name_to_cam[name.decode()] = cam_id
    return name_to_cam


def _resolve_camera_id(sparse_dir: Path, image: str) -> int:
    name_to_cam = _read_images_bin(sparse_dir / "images.bin")
    if image in name_to_cam:
        return name_to_cam[image]
    matches = [k for k in name_to_cam if k.endswith(image)]
    if len(matches) == 1:
        return name_to_cam[matches[0]]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous image name '{image}'. Matches: {matches}")
    raise KeyError(f"Image '{image}' not found in images.bin.")


# ---------------------------------------------------------------------------
# Batch builder
# ---------------------------------------------------------------------------

def _build_mixed_batch(
    images_tensor: torch.Tensor,
    intrinsics_tensor: torch.Tensor,
    owl_boxes_xyxy: torch.Tensor,   # (N, 4) x1y1x2y2 in original pixel space
    labels: list[str],              # length N
    mode: str,                      # "visual" or "geometric"
    data: dict,
    device: torch.device,
) -> tuple[WildDet3DInput, list[str]]:
    """Build a WildDet3DInput with per-box per-label prompts in one batch."""
    n = len(labels)
    assert owl_boxes_xyxy.shape[0] == n

    # Prefixed texts: "visual: chair", "geometric: table", etc.
    prefixed = [f"{mode}: {lbl}" for lbl in labels]
    unique_texts = sorted(set(prefixed))
    text_id_map = {t: i for i, t in enumerate(unique_texts)}
    text_ids = torch.tensor(
        [text_id_map[t] for t in prefixed], dtype=torch.long, device=device
    )

    # Transform OWLv2 boxes from original pixel space to padded+resized input_hw space,
    # then normalise to [0,1] cxcywh for the geometry encoder.
    pad_left, pad_right, pad_top, pad_bottom = data["padding"]
    H_inp, W_inp = data["input_hw"]
    H_orig, W_orig = data["original_hw"]
    resized_w = W_inp - pad_left - pad_right
    resized_h = H_inp - pad_top - pad_bottom
    sx = resized_w / W_orig
    sy = resized_h / H_orig

    geo_boxes = []
    for i in range(n):
        x1, y1, x2, y2 = owl_boxes_xyxy[i].tolist()
        x1p = x1 * sx + pad_left
        y1p = y1 * sy + pad_top
        x2p = x2 * sx + pad_left
        y2p = y2 * sy + pad_top
        cx = (x1p + x2p) / 2.0 / W_inp
        cy = (y1p + y2p) / 2.0 / H_inp
        w  = (x2p - x1p) / W_inp
        h  = (y2p - y1p) / H_inp
        geo_boxes.append([cx, cy, w, h])

    geo_boxes_t = torch.tensor(
        geo_boxes, dtype=torch.float32, device=device
    ).unsqueeze(1)  # (N, 1, 4)

    batch = WildDet3DInput(
        images=images_tensor,
        intrinsics=intrinsics_tensor,
        img_ids=torch.zeros(n, dtype=torch.long, device=device),
        text_ids=text_ids,
        unique_texts=unique_texts,
        geo_boxes=geo_boxes_t,
        geo_boxes_mask=torch.zeros(n, 1, dtype=torch.bool, device=device),
        geo_box_labels=torch.ones(n, 1, dtype=torch.long, device=device),
        padding=[data["padding"]],
    )

    # Clean label names (strip mode prefix) aligned with unique_texts ordering
    clean_names = [t.split(": ", 1)[-1] if ": " in t else t for t in unique_texts]
    return batch, clean_names


# ---------------------------------------------------------------------------
# Postprocessing (mirrors WildDet3DPredictor.forward logic)
# ---------------------------------------------------------------------------

def _postprocess(output, score_threshold: float, data: dict):
    """Filter by score and rescale 2D boxes to original image space."""
    boxes    = output.boxes[0]
    boxes3d  = output.boxes3d[0]
    scores   = output.scores[0]
    scores_2d = output.scores_2d[0] if output.scores_2d is not None else torch.zeros_like(scores)
    scores_3d = output.scores_3d[0] if output.scores_3d is not None else torch.zeros_like(scores)
    class_ids = output.class_ids[0]

    mask = scores >= score_threshold
    boxes     = boxes[mask]
    boxes3d   = boxes3d[mask]
    scores    = scores[mask]
    scores_2d = scores_2d[mask]
    scores_3d = scores_3d[mask]
    class_ids = class_ids[mask]

    pad_left, pad_right, pad_top, pad_bottom = data["padding"]
    orig_h, orig_w = data["original_hw"]
    H_inp, W_inp   = data["input_hw"]
    padded_h = H_inp - pad_top - pad_bottom
    padded_w = W_inp - pad_left - pad_right
    scale_x = orig_w / padded_w
    scale_y = orig_h / padded_h

    boxes = boxes.clone()
    boxes[:, 0] -= pad_left
    boxes[:, 2] -= pad_left
    boxes[:, 1] -= pad_top
    boxes[:, 3] -= pad_top
    boxes[:, 0::2] *= scale_x
    boxes[:, 1::2] *= scale_y
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, orig_w)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, orig_h)

    return boxes, boxes3d, scores, scores_2d, scores_3d, class_ids


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="WildDet3D auto-detection pipeline via OWLv2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("dataset", help='Dataset name, e.g. "mipnerf360"')
    parser.add_argument("scene",   help='Scene name, e.g. "room"')
    parser.add_argument(
        "image",
        help="Image name as in images.bin "
             '(e.g. "DSCF4702.JPG" or "pano_camera1/pano_00051.jpg")',
    )
    parser.add_argument(
        "--mode",
        choices=["visual", "geometric"],
        default="visual",
        help="WildDet3D prompt mode: 'visual' (one-to-many, default) or "
             "'geometric' (one-to-one).",
    )
    parser.add_argument(
        "--owl-confidence",
        type=float,
        default=0.5,
        help="OWLv2 minimum detection confidence (default: 0.5).",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.5,
        help="OWLv2 NMS IoU threshold (default: 0.5).",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.3,
        help="WildDet3D minimum output score (default: 0.3).",
    )
    args = parser.parse_args()

    stem          = Path(args.image).stem
    sparse_dir    = _find_sparse_dir(DATA_ROOT / args.dataset / args.scene / "sparse")
    cam_id        = _resolve_camera_id(sparse_dir, args.image)
    image_file    = DATA_ROOT / args.dataset / args.scene / "images" / args.image
    intrinsics_file = (
        PROJECT_ROOT / "outputs" / args.dataset / args.scene
        / "cam_info" / f"{cam_id}_intrinsics.npy"
    )
    out_dir = (
        PROJECT_ROOT / "outputs" / args.dataset / args.scene
        / "3Dresults" / AUTO_FOLDER
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if not image_file.exists():
        raise FileNotFoundError(f"Image not found: {image_file}")
    if not intrinsics_file.exists():
        raise FileNotFoundError(
            f"Intrinsics not found: {intrinsics_file}\n"
            f"Run first: python extract_intrinsics.py {args.dataset} {args.scene} {args.image}"
        )

    print(f"Image      : {image_file}")
    print(f"Intrinsics : {intrinsics_file}")
    print(f"Output     : {out_dir}/{stem}_*")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load image and intrinsics ---
    image_np   = np.array(Image.open(image_file).convert("RGB")).astype(np.float32)
    intrinsics = np.load(intrinsics_file)

    # --- OWLv2 detection ---
    print("\n[1/3] Running OWLv2 auto-detection...")
    owl = OwlWrapper(
        device=device,
        min_confidence=args.owl_confidence,
        nms_iou_threshold=args.nms_iou,
    )

    # OWLv2 expects (1, 3, H, W) float32 in [0, 255]
    image_t = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
    owl_boxes, owl_scores, owl_label_ids, _ = owl(image_t)

    if len(owl_boxes) == 0:
        print("OWLv2 found no detections above confidence threshold. Exiting.")
        return

    labels = [owl.text_prompts[i] for i in owl_label_ids.tolist()]
    print(f"  OWLv2 detected {len(owl_boxes)} object(s): "
          f"{sorted(set(labels))}")

    # --- Preprocess image for WildDet3D ---
    data = preprocess(image_np, intrinsics)

    # --- Build WildDet3D model ---
    print("\n[2/3] Running WildDet3D 3D detection...")
    wilddet3d_model = build_model(
        checkpoint=str(CHECKPOINT),
        score_threshold=args.score_threshold,
        skip_pretrained=True,
    )

    images_tensor     = data["images"].to(device)
    intrinsics_tensor = data["intrinsics"].to(device)[None]  # (1, 3, 3)

    batch, clean_unique_names = _build_mixed_batch(
        images_tensor=images_tensor,
        intrinsics_tensor=intrinsics_tensor,
        owl_boxes_xyxy=owl_boxes,   # (N, 4) x1y1x2y2 in original pixel space
        labels=labels,
        mode=args.mode,
        data=data,
        device=torch.device(device),
    )

    with torch.no_grad():
        output = wilddet3d_model.wilddet3d(batch)

    boxes, boxes3d, scores, scores_2d, scores_3d, class_ids = _postprocess(
        output, args.score_threshold, data
    )

    print(f"  WildDet3D kept {len(boxes3d)} box(es) after score filtering.")

    # --- Save outputs ---
    print(f"\n[3/3] Saving outputs to {out_dir}/")
    np.save(out_dir / f"{stem}_boxes3d.npy",   boxes3d.cpu().numpy())
    np.save(out_dir / f"{stem}_depth_map.npy", output.depth_maps[0].cpu().numpy())
    np.save(out_dir / f"{stem}_class_ids.npy", class_ids.cpu().numpy())

    # unique_labels.txt: comma-separated clean label names (indexed by class_ids)
    unique_labels_path = out_dir / "unique_labels.txt"
    unique_labels_path.write_text(",".join(clean_unique_names))

    output_png = out_dir / f"{stem}_output.png"
    draw_3d_boxes(
        image=image_np.astype(np.uint8),
        boxes3d=boxes3d,
        intrinsics=intrinsics,
        scores_2d=scores_2d,
        scores_3d=scores_3d,
        class_ids=class_ids,
        class_names=clean_unique_names,
        save_path=str(output_png),
    )

    print(f"\nSaved:")
    print(f"  {out_dir}/{stem}_boxes3d.npy")
    print(f"  {out_dir}/{stem}_depth_map.npy")
    print(f"  {out_dir}/{stem}_class_ids.npy")
    print(f"  {unique_labels_path}")
    print(f"  {output_png}")


if __name__ == "__main__":
    main()