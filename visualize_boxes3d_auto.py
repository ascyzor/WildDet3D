"""Export auto-detected WildDet3D 3D boxes + COLMAP point cloud to .ply files.

Companion to wildDet3D_infer_auto.py. Reads class names from unique_labels.txt
instead of a --prompt argument. Output subfolder is always auto_owlv2.

Usage
-----
    python visualize_boxes3d_auto.py <dataset> <scene> <image> [options]

All options from visualize_boxes3d.py are supported except --prompt.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Import all shared utilities from the existing visualize_boxes3d module.
# That script is importable because it guards execution with if __name__ == "__main__".
from visualize_boxes3d import (
    _BOX_EDGES,
    _build_gsplat_box_gaussians,
    _print_colmap_depths,
    cam_to_world,
    estimate_colmap_scale,
    find_sparse_dir,
    read_image_extrinsics,
    read_points3d_bin,
    sample_box_edges,
    save_ply,
)

PROJECT_ROOT = Path(__file__).parent
DATA_ROOT    = Path.home() / "data"
GSPLAT_ROOT  = Path.home() / "PycharmProjects" / "gsplat" / "examples" / "results"
AUTO_FOLDER  = "auto_owlv2"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export auto-detected 3D boxes + COLMAP point cloud to PLY",
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
        "--colmap-scale", type=float, default=None,
        help="Metres per COLMAP unit. If omitted, estimated from depth_map.npy.",
    )
    parser.add_argument(
        "--clip-radius", type=float, default=None,
        help="Remove COLMAP points farther than this distance from the camera.",
    )
    parser.add_argument(
        "--box-color", type=str, default="255,0,0",
        help="Box edge colour R,G,B uint8 (default: 255,0,0).",
    )
    parser.add_argument(
        "--cloud-color", type=str, default="180,180,180",
        help="Point cloud colour R,G,B uint8 (default: 180,180,180).",
    )
    parser.add_argument(
        "--pts-per-edge", type=int, default=200,
        help="Sampled points per box edge (default: 200).",
    )
    parser.add_argument(
        "--print-colmap-depths", action="store_true",
        help="Print COLMAP depth statistics for the image and exit.",
    )
    parser.add_argument(
        "--exp-scene-name", type=str, default=None,
        help="gsplat experiment folder name. When given, produces <stem>_gsplat_annotated.ply.",
    )
    parser.add_argument(
        "--gsplat-box-radius", type=float, default=0.003,
        help="Gaussian cross-section radius in metres (default: 0.003).",
    )
    parser.add_argument(
        "--gsplat-segs-per-edge", type=int, default=20,
        help="Elongated Gaussians per box edge (default: 20).",
    )
    parser.add_argument(
        "--gsplat-opacity", type=float, default=0.95,
        help="Gaussian opacity in (0, 1) (default: 0.95).",
    )
    args = parser.parse_args()

    stem       = Path(args.image).stem
    sparse_dir = find_sparse_dir(DATA_ROOT / args.dataset / args.scene / "sparse")
    out_dir    = (PROJECT_ROOT / "outputs" / args.dataset / args.scene
                  / "3Dresults" / AUTO_FOLDER)
    image_file = DATA_ROOT / args.dataset / args.scene / "images" / args.image

    print(f"Reading extrinsics for '{args.image}'...")
    R, t = read_image_extrinsics(sparse_dir / "images.bin", args.image)
    cam_center_world = cam_to_world(np.zeros((1, 3)), R, t, scale=1.0).squeeze(0)

    if args.print_colmap_depths:
        _print_colmap_depths(sparse_dir, args.image, R, t)
        return

    # --- COLMAP scale ---
    if args.colmap_scale is not None:
        colmap_scale = args.colmap_scale
        print(f"Using colmap_scale={colmap_scale}")
    else:
        depth_map_path = out_dir / f"{stem}_depth_map.npy"
        if depth_map_path.exists():
            print(f"Estimating COLMAP scale from '{depth_map_path.name}'...")
            colmap_scale = estimate_colmap_scale(
                sparse_dir, args.image, R, t, depth_map_path, image_file
            )
        else:
            print(
                f"WARNING: depth map not found at {depth_map_path}\n"
                "  Using scale=1.0 (box positions will likely be wrong).\n"
                "  Run wildDet3D_infer_auto.py first, or pass --colmap-scale manually."
            )
            colmap_scale = 1.0

    # --- Point cloud ---
    print("Reading point cloud...")
    xyz, rgb = read_points3d_bin(sparse_dir / "points3D.bin")
    print(f"  {len(xyz):,} points")

    if args.clip_radius is not None:
        dists = np.linalg.norm(xyz - cam_center_world, axis=1)
        xyz, rgb = xyz[dists <= args.clip_radius], rgb[dists <= args.clip_radius]
        print(f"  After clip (radius={args.clip_radius}): {len(xyz):,} points")

    # --- Boxes ---
    boxes3d_path = out_dir / f"{stem}_boxes3d.npy"
    if not boxes3d_path.exists():
        raise FileNotFoundError(
            f"boxes3d not found: {boxes3d_path}\n"
            f"Run first:  python wildDet3D_infer_auto.py {args.dataset} {args.scene} {args.image}"
        )
    boxes3d = np.load(boxes3d_path).astype(np.float64)
    print(f"  {len(boxes3d)} box(es) loaded")

    # --- Camera marker ---
    rng = np.random.default_rng(0)
    sphere_pts = rng.standard_normal((500, 3))
    sphere_pts /= np.linalg.norm(sphere_pts, axis=1, keepdims=True)
    cam_radius = max(np.linalg.norm(xyz - cam_center_world, axis=1).min() * 0.3, 0.05)
    cam_pts = cam_center_world + sphere_pts * cam_radius
    cam_rgb = np.tile(np.array([0, 255, 0], dtype=np.uint8), (len(cam_pts), 1))

    # --- Box edges ---
    box_color = np.array([int(v) for v in args.box_color.split(",")], dtype=np.uint8)
    print(f"Sampling box edges ({args.pts_per_edge} pts/edge × "
          f"{len(_BOX_EDGES)} edges × {len(boxes3d)} box(es))...")
    box_xyz = sample_box_edges(boxes3d, R, t, pts_per_edge=args.pts_per_edge,
                               scale=colmap_scale)
    box_rgb = np.tile(box_color, (len(box_xyz), 1))

    # --- Merge and save ---
    cloud_color = np.array([int(v) for v in args.cloud_color.split(",")], dtype=np.uint8)
    cloud_rgb   = np.tile(cloud_color, (len(xyz), 1))

    merged_xyz = np.concatenate([xyz, cam_pts, box_xyz], axis=0)
    merged_rgb = np.concatenate([cloud_rgb, cam_rgb, box_rgb], axis=0)

    out_ply = out_dir / f"{stem}_scene.ply"
    save_ply(out_ply, merged_xyz, merged_rgb)

    # --- Gaussian Splatting output ---
    if args.exp_scene_name is not None:
        from plyfile import PlyData, PlyElement

        gsplat_ply = GSPLAT_ROOT / args.exp_scene_name / "ply" / "point_cloud_29999.ply"
        if not gsplat_ply.exists():
            raise FileNotFoundError(f"gsplat .ply not found: {gsplat_ply}")

        print(f"\nReading gsplat PLY: {gsplat_ply} ...")
        ply_in     = PlyData.read(str(gsplat_ply))
        el_in      = ply_in.elements[0]
        prop_names = [p.name for p in el_in.properties]
        print(f"  {len(el_in.data):,} existing Gaussians")

        c    = np.clip(box_color.astype(np.float64) / 255.0, 0.001, 0.999)
        _C0  = 0.28209479177387814
        f_dc = tuple(float(np.log(c[i] / (1.0 - c[i])) / _C0) for i in range(3))
        opacity_logit = float(np.log(args.gsplat_opacity / (1.0 - args.gsplat_opacity)))

        print(f"Sampling gsplat box Gaussians "
              f"({args.gsplat_segs_per_edge} seg(s)/edge × "
              f"{len(_BOX_EDGES)} edges × {len(boxes3d)} box(es))...")
        new_g = _build_gsplat_box_gaussians(
            boxes3d, R, t, colmap_scale, prop_names,
            f_dc, args.gsplat_box_radius, opacity_logit, args.gsplat_segs_per_edge,
        )

        merged_g   = np.concatenate([el_in.data, new_g])
        out_gsplat = out_dir / f"{stem}_gsplat_annotated.ply"
        PlyData([PlyElement.describe(merged_g, "vertex")], text=False).write(str(out_gsplat))
        print(f"Saved {len(merged_g):,} Gaussians → {out_gsplat}")


if __name__ == "__main__":
    main()