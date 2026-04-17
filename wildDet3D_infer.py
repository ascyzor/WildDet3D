"""Run WildDet3D inference on a single image and save all outputs.

Data layout expected
--------------------
    ~/data/<dataset>/<scene>/
        images/          original images (may contain sub-directories)
        sparse/          COLMAP sparse reconstruction
            0/
                cameras.bin
                images.bin

Prerequisite
------------
    Run extract_intrinsics.py first so the intrinsics .npy exists:
        python extract_intrinsics.py <dataset> <scene> <image>

Usage
-----
    python wildDet3D_infer.py <dataset> <scene> <image> --prompt <categories>

Arguments
---------
  dataset           Dataset name, e.g. "mipnerf360".
  scene             Scene name,   e.g. "room".
  image             Image name exactly as stored in images.bin.
                    Include the sub-directory if present:
                      e.g. "DSCF4702.JPG"  or  "pano_camera1/pano_00051.jpg"
  --prompt STR      Comma-separated detection categories, e.g. "chair,table".
                    Required.
  --score-threshold FLOAT
                    Minimum confidence score to keep a detection. Default: 0.3.

Outputs  (all written to <project>/outputs/<dataset>/<scene>/3Dresults/<prompt>/)
--------
    <stem>_boxes3d.npy    (N, 10) float32 — 3D boxes in OpenCV camera space
    <stem>_depth_map.npy  (1, 1008, 1008) float32 — metric depth in metres
    <stem>_class_ids.npy  (N,) int64 — class index into the prompt list
    <stem>_output.png     visualisation of 3D boxes on the original image

where <stem> is the image filename without extension
(e.g. "DSCF4702" or "pano_00051"), and <prompt> is the --prompt value
with commas replaced by underscores (e.g. "chair_table").
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
from PIL import Image

from wilddet3d import build_model, preprocess
from wilddet3d.vis.visualize import draw_3d_boxes

PROJECT_ROOT = Path(__file__).parent
DATA_ROOT    = Path.home() / "data"
CHECKPOINT   = PROJECT_ROOT / "ckpt" / "wilddet3d_alldata_all_prompt_v1.0.pt"


# ---------------------------------------------------------------------------
# Minimal COLMAP readers (to resolve camera_id without importing
# extract_intrinsics.py, keeping this script self-contained)
# ---------------------------------------------------------------------------

def _find_sparse_dir(colmap_root: Path) -> Path:
    if (colmap_root / "cameras.bin").exists():
        return colmap_root
    for sub in sorted(colmap_root.iterdir()):
        if sub.is_dir() and (sub / "cameras.bin").exists():
            return sub
    raise FileNotFoundError(f"cameras.bin not found under {colmap_root}")


def _read_images_bin(path: Path) -> dict[str, int]:
    """Read images.bin → {image_name: camera_id}."""
    name_to_cam: dict[str, int] = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            f.read(4)              # image_id
            f.read(32)             # quaternion
            f.read(24)             # translation
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
    """Return the camera_id for the given image name."""
    name_to_cam = _read_images_bin(sparse_dir / "images.bin")
    if image in name_to_cam:
        return name_to_cam[image]
    matches = [k for k in name_to_cam if k.endswith(image)]
    if len(matches) == 1:
        return name_to_cam[matches[0]]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous image name '{image}'. Matches: {matches}")
    raise KeyError(
        f"Image '{image}' not found in images.bin. "
        f"Example entries: {list(name_to_cam)[:5]}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run WildDet3D inference and save outputs",
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
        "--prompt",
        required=True,
        help='Comma-separated detection categories, e.g. "chair,table".',
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.3,
        help="Minimum confidence score (default: 0.3).",
    )
    args = parser.parse_args()

    text_prompt   = [p.strip() for p in args.prompt.split(",")]
    prompt_folder = args.prompt.replace(",", "_").replace(" ", "_")
    stem          = Path(args.image).stem   # "DSCF4702" or "pano_00051"

    # --- Paths ---
    image_file  = DATA_ROOT / args.dataset / args.scene / "images" / args.image
    sparse_dir  = _find_sparse_dir(DATA_ROOT / args.dataset / args.scene / "sparse")
    cam_id      = _resolve_camera_id(sparse_dir, args.image)
    intrinsics_file = (PROJECT_ROOT / "outputs" / args.dataset / args.scene
                       / "cam_info" / f"{cam_id}_intrinsics.npy")
    out_dir     = (PROJECT_ROOT / "outputs" / args.dataset / args.scene
                   / "3Dresults" / prompt_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not image_file.exists():
        raise FileNotFoundError(f"Image not found: {image_file}")
    if not intrinsics_file.exists():
        raise FileNotFoundError(
            f"Intrinsics not found: {intrinsics_file}\n"
            f"Run first:  python extract_intrinsics.py {args.dataset} {args.scene} {args.image}"
        )

    print(f"Image      : {image_file}")
    print(f"Intrinsics : {intrinsics_file}")
    print(f"Prompts    : {text_prompt}")
    print(f"Outputs    : {out_dir}/{stem}_*")

    # --- Load image and intrinsics ---
    image_np   = np.array(Image.open(image_file).convert("RGB")).astype(np.float32)
    intrinsics = np.load(intrinsics_file)   # (3, 3) float64

    # --- Preprocess ---
    data = preprocess(image_np, intrinsics)

    # --- Build model ---
    model = build_model(
        checkpoint=str(CHECKPOINT),
        score_threshold=args.score_threshold,
        skip_pretrained=True,
    )

    # --- Inference ---
    results = model(
        images=data["images"].cuda(),
        intrinsics=data["intrinsics"].cuda()[None],
        input_hw=[data["input_hw"]],
        original_hw=[data["original_hw"]],
        padding=[data["padding"]],
        input_texts=text_prompt,
    )
    boxes, boxes3d, scores, scores_2d, scores_3d, class_ids, depth_maps = results

    print(f"Detected {len(boxes3d[0])} box(es).")

    # --- Save outputs ---
    np.save(out_dir / f"{stem}_boxes3d.npy",   boxes3d[0].cpu().numpy())
    np.save(out_dir / f"{stem}_depth_map.npy", depth_maps[0].cpu().numpy())
    np.save(out_dir / f"{stem}_class_ids.npy", class_ids[0].cpu().numpy())

    output_png = out_dir / f"{stem}_output.png"
    draw_3d_boxes(
        image=image_np.astype(np.uint8),
        boxes3d=boxes3d[0],
        intrinsics=intrinsics,
        scores_2d=scores_2d[0],
        scores_3d=scores_3d[0],
        class_ids=class_ids[0],
        class_names=text_prompt,
        save_path=str(output_png),
    )

    print(f"\nSaved:")
    print(f"  {out_dir}/{stem}_boxes3d.npy")
    print(f"  {out_dir}/{stem}_depth_map.npy")
    print(f"  {out_dir}/{stem}_class_ids.npy")
    print(f"  {output_png}")


if __name__ == "__main__":
    main()