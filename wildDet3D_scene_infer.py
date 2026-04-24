"""Batch WildDet3D scene inference (OWLv2 → WildDet3D, models loaded once).

Discovers all COLMAP-registered images in a scene, runs OWLv2 + WildDet3D on
each, and writes per-frame results.  Both OWLv2 and WildDet3D are loaded once
and reused for every frame.

Data layout expected
--------------------
    ~/data/<dataset>/<scene>/
        images/          flat or camera-id subdirectories
        sparse/          COLMAP sparse reconstruction

Outputs
-------
    outputs/<dataset>/<scene>/3Dresults/auto_owlv2/per_frame/
        <stem>_boxes3d.npy          [M, 10] camera-space metres
        <stem>_class_ids.npy        [M]  indices into unique_labels.txt
        <stem>_unique_labels.txt    comma-separated label vocab for this frame
        <stem>_scores_2d.npy        [M]
        <stem>_scores_3d.npy        [M]
        <stem>_depth_map.npy        [H, W] metric depth
        <stem>_output.png           detection overlay

    outputs/<dataset>/<scene>/3Dresults/auto_owlv2/scene/
        processed_images.json       manifest: stem → {colmap_path, original_hw}

Usage
-----
    python wildDet3D_scene_infer.py <dataset> <scene> [options]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from wilddet3d import build_model, preprocess
from wilddet3d.vis.visualize import draw_3d_boxes
from owl import OwlWrapper

# Reuse batch-builder and postprocessor from the single-frame script
from wildDet3D_infer_auto import _build_mixed_batch, _postprocess

# Intrinsics utilities from extract_intrinsics
from extract_intrinsics import (
    build_K,
    find_sparse_dir,
    read_cameras_bin,
    read_images_bin,
)

DATA_ROOT    = Path.home() / "data"
CHECKPOINT   = PROJECT_ROOT / "ckpt" / "wilddet3d_alldata_all_prompt_v1.0.pt"
AUTO_FOLDER  = "auto_owlv2"
FRAME_FOLDER = "per_frame"
SCENE_FOLDER = "scene"

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif",
             ".JPG", ".JPEG", ".PNG", ".TIFF", ".TIF"}


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------

def discover_scene_images(
    images_dir: Path,
    images_bin: Path,
) -> list[tuple[str, str]]:
    """Return [(stem, colmap_relative_path)] for all COLMAP-registered images.

    Handles both flat layouts (image.JPG) and subdirectory layouts
    (pano_camera1/image.jpg).  Only images present in images.bin are returned.
    """
    name_to_cam = read_images_bin(images_bin)
    colmap_names = set(name_to_cam.keys())

    found: list[tuple[str, str]] = []
    seen_stems: set[str] = set()

    for fpath in sorted(images_dir.rglob("*")):
        if not fpath.is_file() or fpath.suffix not in _IMG_EXTS:
            continue
        rel_posix = fpath.relative_to(images_dir).as_posix()
        if rel_posix in colmap_names:
            key = rel_posix
        elif fpath.name in colmap_names:
            key = fpath.name
        else:
            continue

        stem = fpath.stem
        if stem in seen_stems:
            # Disambiguate colliding stems (rare but possible with subdirs)
            stem = rel_posix.replace("/", "_").replace(".", "_")
        seen_stems.add(stem)
        found.append((stem, key))

    n_disk = sum(
        1 for f in images_dir.rglob("*")
        if f.is_file() and f.suffix in _IMG_EXTS
    )
    print(f"  {len(found)} registered images "
          f"(disk: {n_disk}, COLMAP registry: {len(colmap_names)})")
    return found


# ---------------------------------------------------------------------------
# Intrinsics
# ---------------------------------------------------------------------------

def ensure_all_intrinsics(
    sparse_dir: Path,
    image_names: list[str],
    cam_info_dir: Path,
) -> dict[str, np.ndarray]:
    """Return {colmap_image_name: K} and write missing .npy files."""
    name_to_cam = read_images_bin(sparse_dir / "images.bin")
    cameras     = read_cameras_bin(sparse_dir / "cameras.bin")

    result: dict[str, np.ndarray] = {}
    for img_name in image_names:
        cam_id = name_to_cam.get(img_name)
        if cam_id is None:
            continue
        npy_path = cam_info_dir / f"{cam_id}_intrinsics.npy"
        if npy_path.exists():
            K = np.load(npy_path)
        else:
            K = build_K(cameras[cam_id])
            npy_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(npy_path, K)
        result[img_name] = K
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch WildDet3D scene inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("dataset")
    parser.add_argument("scene")
    parser.add_argument(
        "--mode", choices=["visual", "geometric"], default="visual",
        help="WildDet3D prompt mode (default: visual).",
    )
    parser.add_argument(
        "--owl-confidence", type=float, default=0.5,
        help="OWLv2 min confidence (default: 0.5).",
    )
    parser.add_argument(
        "--nms-iou", type=float, default=0.5,
        help="OWLv2 NMS IoU threshold (default: 0.5).",
    )
    parser.add_argument(
        "--score-threshold", type=float, default=0.3,
        help="WildDet3D output score filter (default: 0.3).",
    )
    args = parser.parse_args()

    images_dir   = DATA_ROOT / args.dataset / args.scene / "images"
    sparse_dir   = find_sparse_dir(DATA_ROOT / args.dataset / args.scene / "sparse")
    cam_info_dir = PROJECT_ROOT / "outputs" / args.dataset / args.scene / "cam_info"
    base_out     = PROJECT_ROOT / "outputs" / args.dataset / args.scene / "3Dresults" / AUTO_FOLDER
    frame_dir    = base_out / FRAME_FOLDER
    scene_dir    = base_out / SCENE_FOLDER
    frame_dir.mkdir(parents=True, exist_ok=True)
    scene_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*48}")
    print(f" WildDet3D scene inference")
    print(f"  dataset : {args.dataset}")
    print(f"  scene   : {args.scene}")
    print(f"  mode    : {args.mode}")
    print(f"{'='*48}\n")

    print("Discovering images...")
    image_list = discover_scene_images(images_dir, sparse_dir / "images.bin")
    if not image_list:
        print("No registered images found. Exiting.")
        return

    colmap_paths = [rel for _, rel in image_list]

    print(f"\nEnsuring intrinsics for {len(image_list)} images...")
    intrinsics_map = ensure_all_intrinsics(sparse_dir, colmap_paths, cam_info_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nLoading OWLv2 on {device}...")
    owl = OwlWrapper(
        device=device,
        min_confidence=args.owl_confidence,
        nms_iou_threshold=args.nms_iou,
    )

    print(f"Loading WildDet3D on {device}...")
    wilddet3d_model = build_model(
        checkpoint=str(CHECKPOINT),
        score_threshold=args.score_threshold,
        skip_pretrained=True,
    )

    manifest: dict[str, dict] = {}
    n_total = len(image_list)

    for idx, (stem, colmap_rel) in enumerate(image_list):
        print(f"\n[{idx + 1}/{n_total}] {colmap_rel}")

        image_file = images_dir / colmap_rel
        intrinsics = intrinsics_map.get(colmap_rel)
        if intrinsics is None:
            print("  Skip: no intrinsics")
            continue

        try:
            pil_img  = PILImage.open(image_file).convert("RGB")
            image_np = np.array(pil_img).astype(np.float32)
            H_orig, W_orig = pil_img.height, pil_img.width
        except Exception as exc:
            print(f"  Skip: cannot load image ({exc})")
            continue

        # --- OWLv2 ---
        image_t = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
        owl_boxes, owl_scores, owl_label_ids, _ = owl(image_t)

        empty_arrays = [
            ("_boxes3d.npy",   np.zeros((0, 10), dtype=np.float32)),
            ("_class_ids.npy", np.zeros((0,),    dtype=np.int64)),
            ("_scores_2d.npy", np.zeros((0,),    dtype=np.float32)),
            ("_scores_3d.npy", np.zeros((0,),    dtype=np.float32)),
            ("_depth_map.npy", np.zeros((0,),    dtype=np.float32)),
        ]

        if len(owl_boxes) == 0:
            print("  OWLv2: no detections")
            for suffix, arr in empty_arrays:
                np.save(frame_dir / f"{stem}{suffix}", arr)
            (frame_dir / f"{stem}_unique_labels.txt").write_text("")
            manifest[stem] = {"colmap_path": colmap_rel, "original_hw": [H_orig, W_orig]}
            continue

        labels = [owl.text_prompts[i] for i in owl_label_ids.tolist()]
        print(f"  OWLv2: {len(owl_boxes)} detection(s): {sorted(set(labels))}")

        # --- WildDet3D ---
        data              = preprocess(image_np, intrinsics)
        images_tensor     = data["images"].to(device)
        intrinsics_tensor = data["intrinsics"].to(device)[None]

        batch, clean_unique_names = _build_mixed_batch(
            images_tensor=images_tensor,
            intrinsics_tensor=intrinsics_tensor,
            owl_boxes_xyxy=owl_boxes,
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
        print(f"  WildDet3D: {len(boxes3d)} box(es) after score filter")

        # --- Save ---
        np.save(frame_dir / f"{stem}_boxes3d.npy",   boxes3d.cpu().numpy())
        np.save(frame_dir / f"{stem}_class_ids.npy", class_ids.cpu().numpy())
        np.save(frame_dir / f"{stem}_scores_2d.npy", scores_2d.cpu().numpy())
        np.save(frame_dir / f"{stem}_scores_3d.npy", scores_3d.cpu().numpy())
        np.save(frame_dir / f"{stem}_depth_map.npy",
                output.depth_maps[0].cpu().numpy())
        (frame_dir / f"{stem}_unique_labels.txt").write_text(
            ",".join(clean_unique_names)
        )
        draw_3d_boxes(
            image=image_np.astype(np.uint8),
            boxes3d=boxes3d,
            intrinsics=intrinsics,
            scores_2d=scores_2d,
            scores_3d=scores_3d,
            class_ids=class_ids,
            class_names=clean_unique_names,
            save_path=str(frame_dir / f"{stem}_output.png"),
        )

        manifest[stem] = {"colmap_path": colmap_rel, "original_hw": [H_orig, W_orig]}

    # --- Manifest ---
    manifest_path = scene_dir / "processed_images.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    n_with_boxes = sum(
        1 for stem in manifest
        if (frame_dir / f"{stem}_boxes3d.npy").exists()
        and np.load(frame_dir / f"{stem}_boxes3d.npy").shape[0] > 0
    )
    print(f"\nDone. {len(manifest)}/{n_total} images processed, "
          f"{n_with_boxes} with detections.")
    print(f"Per-frame outputs : {frame_dir}/")
    print(f"Manifest          : {manifest_path}")


if __name__ == "__main__":
    main()