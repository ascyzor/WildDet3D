"""Extract camera intrinsics from a COLMAP sparse reconstruction and save as .npy.

Data layout expected
--------------------
    ~/data/<dataset>/<scene>/
        images/          original images (may contain sub-directories)
        sparse/          COLMAP sparse reconstruction
            0/           (or directly here)
                cameras.bin
                images.bin

Usage
-----
    python extract_intrinsics.py <dataset> <scene> <image>

Arguments
---------
  dataset   Dataset name, e.g. "mipnerf360".
  scene     Scene name,   e.g. "room".
  image     Image name exactly as stored in images.bin.
            If the image lives in a sub-directory, include it:
              e.g. "DSCF4702.JPG"  or  "pano_camera1/pano_00051.jpg"

Optional
--------
  --list    Print all cameras in the reconstruction and exit (no file written).

Output
------
    <project>/outputs/<dataset>/<scene>/cam_info/<camera_id>_intrinsics.npy

    A (3, 3) float64 pinhole K matrix:
        [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]

COLMAP camera models supported
-------------------------------
  SIMPLE_PINHOLE  params: [f, cx, cy]
  PINHOLE         params: [fx, fy, cx, cy]
  SIMPLE_RADIAL   params: [f, cx, cy, k]        -> uses f for fx/fy, ignores k
  RADIAL          params: [f, cx, cy, k1, k2]   -> uses f for fx/fy, ignores k
  OPENCV          params: [fx, fy, cx, cy, k1, k2, p1, p2]
  OPENCV_FISHEYE  params: [fx, fy, cx, cy, k1, k2, k3, k4]
  FULL_OPENCV     params: [fx, fy, cx, cy, k1..k6, p1, p2]
  FOV             params: [fx, fy, cx, cy, omega]
  SIMPLE_RADIAL_FISHEYE params: [f, cx, cy, k]
  RADIAL_FISHEYE  params: [f, cx, cy, k1, k2]
  THIN_PRISM_FISHEYE    params: [fx, fy, cx, cy, ...]

Only the pinhole part (fx, fy, cx, cy) is written to K.
Distortion coefficients are intentionally ignored.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent
DATA_ROOT    = Path.home() / "data"

# ---------------------------------------------------------------------------
# COLMAP binary readers
# ---------------------------------------------------------------------------

# Maps model_id -> (model_name, num_params, (fx_idx, fy_idx, cx_idx, cy_idx))
# fy_idx == fx_idx means the model uses a single focal length.
_CAMERA_MODELS: dict[int, tuple[str, int, tuple[int, int, int, int]]] = {
    0:  ("SIMPLE_PINHOLE",         1,  (0, 0, 1, 2)),
    1:  ("PINHOLE",                4,  (0, 1, 2, 3)),
    2:  ("SIMPLE_RADIAL",          4,  (0, 0, 1, 2)),
    3:  ("RADIAL",                 5,  (0, 0, 1, 2)),
    4:  ("OPENCV",                 8,  (0, 1, 2, 3)),
    5:  ("OPENCV_FISHEYE",         8,  (0, 1, 2, 3)),
    6:  ("FULL_OPENCV",           12,  (0, 1, 2, 3)),
    7:  ("FOV",                    5,  (0, 1, 2, 3)),
    8:  ("SIMPLE_RADIAL_FISHEYE",  4,  (0, 0, 1, 2)),
    9:  ("RADIAL_FISHEYE",         5,  (0, 0, 1, 2)),
    10: ("THIN_PRISM_FISHEYE",    12,  (0, 1, 2, 3)),
}


def read_cameras_bin(path: Path) -> dict[int, dict]:
    """Read cameras.bin → {camera_id: {model, width, height, fx, fy, cx, cy, params}}."""
    cameras: dict[int, dict] = {}
    with open(path, "rb") as f:
        (num_cameras,) = struct.unpack("<Q", f.read(8))
        for _ in range(num_cameras):
            (cam_id,)   = struct.unpack("<I", f.read(4))
            (model_id,) = struct.unpack("<i", f.read(4))
            (width,)    = struct.unpack("<Q", f.read(8))
            (height,)   = struct.unpack("<Q", f.read(8))

            if model_id not in _CAMERA_MODELS:
                raise ValueError(
                    f"Unknown COLMAP camera model id {model_id} for camera {cam_id}. "
                    "Add it to _CAMERA_MODELS."
                )

            name, num_params, (fi, fj, ci, cj) = _CAMERA_MODELS[model_id]
            params = struct.unpack(f"<{num_params}d", f.read(8 * num_params))

            cameras[cam_id] = dict(
                model=name,
                width=width,
                height=height,
                fx=params[fi],
                fy=params[fj],
                cx=params[ci],
                cy=params[cj],
                params=params,
            )
    return cameras


def read_images_bin(path: Path) -> dict[str, int]:
    """Read images.bin → {image_name: camera_id}."""
    name_to_cam: dict[str, int] = {}
    with open(path, "rb") as f:
        (num_images,) = struct.unpack("<Q", f.read(8))
        for _ in range(num_images):
            f.read(4)             # image_id
            f.read(32)            # qw qx qy qz
            f.read(24)            # tx ty tz
            (cam_id,) = struct.unpack("<I", f.read(4))
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            (num_pts,) = struct.unpack("<Q", f.read(8))
            f.read(num_pts * 24)  # x(8) + y(8) + point3d_id(8)
            name_to_cam[name.decode()] = cam_id
    return name_to_cam


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_K(cam: dict) -> np.ndarray:
    """Build a 3×3 pinhole intrinsics matrix."""
    return np.array(
        [[cam["fx"],        0.0, cam["cx"]],
         [      0.0, cam["fy"], cam["cy"]],
         [      0.0,       0.0,       1.0]],
        dtype=np.float64,
    )


def find_sparse_dir(colmap_root: Path) -> Path:
    """Resolve the folder that directly contains cameras.bin."""
    if (colmap_root / "cameras.bin").exists():
        return colmap_root
    for sub in sorted(colmap_root.iterdir()):
        if sub.is_dir() and (sub / "cameras.bin").exists():
            return sub
    raise FileNotFoundError(
        f"Could not find cameras.bin under {colmap_root}."
    )


def print_cameras(cameras: dict[int, dict]) -> None:
    print(f"{'ID':>4}  {'Model':<22}  {'W':>5}  {'H':>5}  "
          f"{'fx':>10}  {'fy':>10}  {'cx':>10}  {'cy':>10}")
    print("-" * 82)
    for cam_id, c in sorted(cameras.items()):
        print(f"{cam_id:>4}  {c['model']:<22}  {c['width']:>5}  {c['height']:>5}  "
              f"{c['fx']:>10.4f}  {c['fy']:>10.4f}  {c['cx']:>10.4f}  {c['cy']:>10.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract camera intrinsics from COLMAP and save as .npy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("dataset", help='Dataset name, e.g. "mipnerf360"')
    parser.add_argument("scene",   help='Scene name, e.g. "room"')
    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        help="Image name as stored in images.bin "
             '(e.g. "DSCF4702.JPG" or "pano_camera1/pano_00051.jpg"). '
             "Required unless --list is used.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print all cameras and exit (no file written).",
    )
    args = parser.parse_args()

    if not args.list and args.image is None:
        parser.error("image argument is required unless --list is used.")

    colmap_root = DATA_ROOT / args.dataset / args.scene / "sparse"
    sparse_dir  = find_sparse_dir(colmap_root)
    cameras     = read_cameras_bin(sparse_dir / "cameras.bin")

    if args.list:
        print_cameras(cameras)
        return

    # Look up camera_id from images.bin
    images_bin = sparse_dir / "images.bin"
    if not images_bin.exists():
        parser.error(f"images.bin not found in {sparse_dir}")

    name_to_cam = read_images_bin(images_bin)
    image_key   = args.image

    if image_key not in name_to_cam:
        matches = [k for k in name_to_cam if k.endswith(image_key)]
        if len(matches) == 1:
            image_key = matches[0]
        elif len(matches) > 1:
            parser.error(
                f"Ambiguous image name '{image_key}'. Matches: {matches}"
            )
        else:
            parser.error(
                f"Image '{image_key}' not found in images.bin. "
                f"Example entries: {list(name_to_cam)[:5]}"
            )

    cam_id = name_to_cam[image_key]
    print(f"Image '{image_key}' -> camera_id={cam_id}")

    if cam_id not in cameras:
        parser.error(
            f"camera_id={cam_id} not found. Available: {sorted(cameras)}"
        )

    cam = cameras[cam_id]
    K   = build_K(cam)

    print(f"\nCamera {cam_id}: {cam['model']}  {cam['width']}x{cam['height']}")
    print(f"  fx={cam['fx']:.6f}  fy={cam['fy']:.6f}  "
          f"cx={cam['cx']:.6f}  cy={cam['cy']:.6f}")
    print(f"\nIntrinsics matrix K:\n{K}")

    out_path = (PROJECT_ROOT / "outputs" / args.dataset / args.scene
                / "cam_info" / f"{cam_id}_intrinsics.npy")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, K)
    print(f"\nSaved -> {out_path}  (shape {K.shape}, dtype {K.dtype})")


if __name__ == "__main__":
    main()