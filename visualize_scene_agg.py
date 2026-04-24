"""Visualise aggregated scene-level 3D boxes as a PLY point cloud.

Reads the output of wildDet3D_scene_agg.py and produces a single PLY that
contains the COLMAP point cloud plus box wireframes, one distinct colour per
object ID.

All box data is in COLMAP world space — no per-frame R/t transform needed.

Usage
-----
    python visualize_scene_agg.py <dataset> <scene> [options]

Options
-------
    --box-type    median|best_score|best_location  (default: best_location)
    --pts-per-edge INT        Sampled points per box edge (default: 200)
    --clip-radius FLOAT       Discard COLMAP points farther than this from
                              the scene centroid (COLMAP units, default: None)
    --show-frame-centers      Also plot per-frame detection centers as grey dots
    --exp-scene-name NAME     Also inject boxes into a gsplat .ply

Outputs
-------
    auto_owlv2/scene/scene_agg.ply
    auto_owlv2/scene/scene_agg_gsplat.ply  (only if --exp-scene-name given)
"""

from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent
DATA_ROOT    = Path.home() / "data"
GSPLAT_ROOT  = Path.home() / "PycharmProjects" / "gsplat" / "examples" / "results"
AUTO_FOLDER  = "auto_owlv2"
SCENE_FOLDER = "scene"

from visualize_boxes3d import (
    _build_gsplat_box_gaussians,
    cam_to_world,
    find_sparse_dir,
    read_all_extrinsics,
    read_points3d_bin,
    sample_box_edges_world,
    save_ply,
)

_BOX_TYPE_FILE = {
    "median":        "scene_median_boxes3d_world.npy",
    "best_score":    "scene_best_score_boxes3d_world.npy",
    "best_location": "scene_best_location_boxes3d_world.npy",
}


def object_colors(n: int) -> list[np.ndarray]:
    """Generate N visually distinct colours using evenly-spaced HSV hues."""
    colors = []
    for i in range(n):
        h = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
        colors.append(
            np.array([int(r * 255), int(g * 255), int(b * 255)], dtype=np.uint8)
        )
    return colors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise aggregated 3D boxes as a PLY",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("dataset")
    parser.add_argument("scene")
    parser.add_argument(
        "--box-type",
        choices=["median", "best_score", "best_location"],
        default="best_location",
        help="Which box type to visualise (default: best_location).",
    )
    parser.add_argument(
        "--pts-per-edge", type=int, default=200,
        help="Sampled points per box edge (default: 200).",
    )
    parser.add_argument(
        "--clip-radius", type=float, default=None,
        help="Discard COLMAP points farther than this from the scene centroid "
             "(COLMAP units, default: None).",
    )
    parser.add_argument(
        "--show-frame-centers", action="store_true",
        help="Plot per-frame detection centers as grey dots for debugging.",
    )
    parser.add_argument(
        "--exp-scene-name", type=str, default=None,
        help="gsplat experiment folder name (produces scene_agg_gsplat.ply).",
    )
    parser.add_argument(
        "--gsplat-box-radius", type=float, default=0.003,
        help="Gaussian cross-section radius in metres (default: 0.003).",
    )
    parser.add_argument(
        "--gsplat-segs-per-edge", type=int, default=20,
        help="Gaussians per box edge (default: 20).",
    )
    parser.add_argument(
        "--gsplat-opacity", type=float, default=0.95,
        help="Gaussian opacity (default: 0.95).",
    )
    args = parser.parse_args()

    scene_dir  = (PROJECT_ROOT / "outputs" / args.dataset / args.scene
                  / "3Dresults" / AUTO_FOLDER / SCENE_FOLDER)
    sparse_dir = find_sparse_dir(DATA_ROOT / args.dataset / args.scene / "sparse")

    # --- COLMAP scale ---
    scale_file   = scene_dir / "scene_colmap_scale.txt"
    colmap_scale = float(scale_file.read_text().strip()) if scale_file.exists() else 1.0
    print(f"COLMAP scale : {colmap_scale:.5f} m/unit")
    print(f"Box type     : {args.box_type}")

    # --- Objects ---
    objects_path = scene_dir / "scene_objects.json"
    if not objects_path.exists():
        raise FileNotFoundError(
            f"scene_objects.json not found: {objects_path}\n"
            f"Run wildDet3D_scene_agg.py first."
        )
    objects = json.loads(objects_path.read_text())
    K = len(objects)
    print(f"Objects      : {K}")

    if K == 0:
        print("No objects in scene_objects.json — saving point cloud only.")
        xyz, rgb = read_points3d_bin(sparse_dir / "points3D.bin")
        cloud_rgb = np.tile(np.array([180, 180, 180], dtype=np.uint8), (len(xyz), 1))
        save_ply(scene_dir / "scene_agg.ply", xyz.astype(np.float32), cloud_rgb)
        return

    # --- Boxes ---
    npy_path = scene_dir / _BOX_TYPE_FILE[args.box_type]
    if not npy_path.exists():
        raise FileNotFoundError(f"Box file not found: {npy_path}")
    boxes = np.load(npy_path)   # (K, 10)

    # --- Per-object colours ---
    colors = object_colors(K)

    # --- COLMAP point cloud ---
    print("Reading COLMAP point cloud...")
    xyz, rgb = read_points3d_bin(sparse_dir / "points3D.bin")
    print(f"  {len(xyz):,} points")

    if args.clip_radius is not None:
        centroid = xyz.mean(axis=0)
        mask     = np.linalg.norm(xyz - centroid, axis=1) <= args.clip_radius
        xyz, rgb = xyz[mask], rgb[mask]
        print(f"  After clip: {len(xyz):,} points")

    cloud_rgb = np.tile(np.array([180, 180, 180], dtype=np.uint8), (len(xyz), 1))
    all_xyz = [xyz]
    all_rgb = [cloud_rgb]

    # --- Box wireframes (one colour per object) ---
    for obj_id, (obj, color) in enumerate(zip(objects, colors)):
        box    = boxes[obj_id][np.newaxis]   # (1, 10)
        pts    = sample_box_edges_world(box, pts_per_edge=args.pts_per_edge)
        label  = obj["label"]
        print(f"  obj {obj_id:>3}: '{label}'  color={color.tolist()}")
        all_xyz.append(pts)
        all_rgb.append(np.tile(color, (len(pts), 1)))

    # --- Per-frame centers (debug) ---
    if args.show_frame_centers:
        frame_dir = (PROJECT_ROOT / "outputs" / args.dataset / args.scene
                     / "3Dresults" / AUTO_FOLDER / "per_frame")
        manifest_path = scene_dir / "processed_images.json"
        if frame_dir.exists() and manifest_path.exists():
            all_extr = read_all_extrinsics(sparse_dir / "images.bin")
            manifest = json.loads(manifest_path.read_text())
            center_pts = []
            for stem, info in manifest.items():
                boxes_path = frame_dir / f"{stem}_boxes3d.npy"
                if not boxes_path.exists():
                    continue
                b = np.load(boxes_path)
                if b.ndim != 2 or len(b) == 0:
                    continue
                R, t = all_extr.get(info["colmap_path"], (None, None))
                if R is None:
                    continue
                center_pts.append(
                    cam_to_world(b[:, :3], R, t, scale=colmap_scale)
                )
            if center_pts:
                pts_all = np.concatenate(center_pts, axis=0)
                all_xyz.append(pts_all)
                all_rgb.append(np.tile(
                    np.array([120, 120, 120], dtype=np.uint8), (len(pts_all), 1)
                ))
                print(f"  Per-frame centers: {len(pts_all)} points")

    # --- Merge and save ---
    merged_xyz = np.concatenate(all_xyz, axis=0).astype(np.float32)
    merged_rgb = np.concatenate(all_rgb, axis=0)
    out_ply    = scene_dir / "scene_agg.ply"
    save_ply(out_ply, merged_xyz, merged_rgb)

    # --- Gaussian Splatting output ---
    if args.exp_scene_name is not None:
        from plyfile import PlyData, PlyElement
        _C0 = 0.28209479177387814

        gsplat_ply = GSPLAT_ROOT / args.exp_scene_name / "ply" / "point_cloud_29999.ply"
        if not gsplat_ply.exists():
            raise FileNotFoundError(f"gsplat .ply not found: {gsplat_ply}")

        print(f"\nReading gsplat PLY: {gsplat_ply} ...")
        ply_in     = PlyData.read(str(gsplat_ply))
        el_in      = ply_in.elements[0]
        prop_names = [p.name for p in el_in.properties]
        print(f"  {len(el_in.data):,} Gaussians")

        opacity_logit = float(
            np.log(args.gsplat_opacity / (1.0 - args.gsplat_opacity))
        )
        R_id = np.eye(3)
        t_0  = np.zeros(3)

        all_new_g = []
        for obj_id, (obj, color) in enumerate(zip(objects, colors)):
            box = boxes[obj_id][np.newaxis]
            c   = np.clip(color.astype(np.float64) / 255.0, 0.001, 0.999)
            f_dc = tuple(float(np.log(c[i] / (1.0 - c[i])) / _C0) for i in range(3))
            new_g = _build_gsplat_box_gaussians(
                box, R_id, t_0, 1.0,
                prop_names, f_dc,
                args.gsplat_box_radius / colmap_scale,
                opacity_logit,
                args.gsplat_segs_per_edge,
            )
            all_new_g.append(new_g)

        if all_new_g:
            merged_g   = np.concatenate([el_in.data] + all_new_g)
            out_gsplat = scene_dir / "scene_agg_gsplat.ply"
            PlyData([PlyElement.describe(merged_g, "vertex")],
                    text=False).write(str(out_gsplat))
            print(f"Saved {len(merged_g):,} Gaussians → {out_gsplat}")


if __name__ == "__main__":
    main()