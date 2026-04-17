"""Diagnostic: check whether the gsplat .ply world space is aligned with COLMAP.

Hardcoded for mipnerf360 / room / room_default.

Steps
-----
1. Bounding-box comparison  — numeric check, printed to stdout.
2. Camera-center injection  — writes a .ply with the COLMAP camera centre
   (for every image in the reconstruction) injected as bright green Gaussians.
   Open in SuperSplat; markers should sit exactly at each camera position.
3. COLMAP-cloud overlay     — writes a .ply with ~1 000 random COLMAP 3D points
   injected as cyan Gaussians.  They should land on scene surfaces.

Outputs (written to the same folder as the gsplat .ply)
-------
    room_default_cam_markers.ply      — camera centres (green)
    room_default_colmap_overlay.ply   — COLMAP point sample (cyan)
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# Hardcoded paths
# ---------------------------------------------------------------------------

DATA_ROOT   = Path.home() / "data"
GSPLAT_ROOT = Path.home() / "PycharmProjects" / "gsplat" / "examples" / "results"

COLMAP_DIR   = DATA_ROOT   / "mipnerf360" / "room" / "sparse" / "0"
GSPLAT_PLY   = GSPLAT_ROOT / "room_default" / "ply" / "point_cloud_29999.ply"
OUT_DIR      = GSPLAT_ROOT / "room_default" / "ply"

# Number of COLMAP points to sample for the overlay
N_OVERLAY = 1000

# Gaussian radius for markers (in scene units — tuned for mipnerf360 room scale)
CAM_MARKER_RADIUS   = 0.05   # camera spheres
CLOUD_MARKER_RADIUS = 0.01   # COLMAP point dots

# SH DC constant
_C0 = 0.28209479177387814


# ---------------------------------------------------------------------------
# Tiny COLMAP readers (self-contained, no imports from project)
# ---------------------------------------------------------------------------

def _read_points3d(path: Path) -> np.ndarray:
    """Return (N, 3) float64 xyz from points3D.bin."""
    pts = []
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            f.read(8)                              # point3d_id
            x, y, z = struct.unpack("<3d", f.read(24))
            f.read(11)                             # r g b error
            (tl,) = struct.unpack("<Q", f.read(8))
            f.read(tl * 8)                         # track
            pts.append((x, y, z))
    return np.array(pts, dtype=np.float64)


def _read_camera_centres(path: Path) -> np.ndarray:
    """Return (N, 3) float64 world-space camera centres from images.bin."""
    centres = []
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            f.read(4)                                      # image_id
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz      = struct.unpack("<3d", f.read(24))
            f.read(4)                                      # camera_id
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            (num_pts,) = struct.unpack("<Q", f.read(8))
            f.read(num_pts * 24)
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            t = np.array([tx, ty, tz])
            # Camera centre in world = -R^T @ t
            centres.append(-R.T @ t)
    return np.array(centres, dtype=np.float64)


# ---------------------------------------------------------------------------
# Gaussian builder
# ---------------------------------------------------------------------------

def _colour_to_sh_dc(rgb_float: np.ndarray) -> tuple[float, float, float]:
    """rgb_float in [0,1] → (f_dc_0, f_dc_1, f_dc_2)."""
    c = np.clip(rgb_float, 0.001, 0.999)
    return tuple(float(np.log(c[i] / (1.0 - c[i])) / _C0) for i in range(3))


def _make_sphere_gaussians(
    positions: np.ndarray,      # (N, 3) world-space centres
    colour: np.ndarray,         # (3,) float [0,1]
    log_radius: float,
    prop_names: list[str],
    opacity_logit: float = 3.0, # sigmoid(3) ≈ 0.95
) -> np.ndarray:
    """Build spherical Gaussians at the given positions."""
    f_dc  = _colour_to_sh_dc(colour)
    dtype = [(name, np.float32) for name in prop_names]
    n     = len(positions)
    data  = np.zeros(n, dtype=dtype)
    data["x"] = positions[:, 0].astype(np.float32)
    data["y"] = positions[:, 1].astype(np.float32)
    data["z"] = positions[:, 2].astype(np.float32)
    data["f_dc_0"] = np.float32(f_dc[0])
    data["f_dc_1"] = np.float32(f_dc[1])
    data["f_dc_2"] = np.float32(f_dc[2])
    data["opacity"] = np.float32(opacity_logit)
    data["scale_0"] = np.float32(log_radius)
    data["scale_1"] = np.float32(log_radius)
    data["scale_2"] = np.float32(log_radius)
    data["rot_0"]   = np.float32(1.0)   # identity quaternion
    return data


def _save_merged(out_path: Path, base: np.ndarray, markers: np.ndarray,
                 prop_names: list[str]) -> None:
    merged = np.concatenate([base, markers])
    PlyData([PlyElement.describe(merged, "vertex")], text=False).write(str(out_path))
    print(f"  Saved {len(merged):,} Gaussians → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print(" gsplat ↔ COLMAP alignment check: mipnerf360 / room")
    print("=" * 60)

    # --- Load COLMAP points ---
    print("\n[1] Loading COLMAP points3D ...")
    colmap_pts = _read_points3d(COLMAP_DIR / "points3D.bin")
    print(f"  {len(colmap_pts):,} points")
    print(f"  X: [{colmap_pts[:,0].min():.3f}, {colmap_pts[:,0].max():.3f}]")
    print(f"  Y: [{colmap_pts[:,1].min():.3f}, {colmap_pts[:,1].max():.3f}]")
    print(f"  Z: [{colmap_pts[:,2].min():.3f}, {colmap_pts[:,2].max():.3f}]")
    colmap_centre = colmap_pts.mean(axis=0)
    colmap_extent = colmap_pts.max(axis=0) - colmap_pts.min(axis=0)
    print(f"  Centre : {colmap_centre.round(3)}")
    print(f"  Extent : {colmap_extent.round(3)}")

    # --- Load gsplat Gaussians ---
    print(f"\n[2] Loading gsplat PLY: {GSPLAT_PLY} ...")
    ply_in     = PlyData.read(str(GSPLAT_PLY))
    el_in      = ply_in.elements[0]
    prop_names = [p.name for p in el_in.properties]
    base       = el_in.data
    gs_xyz     = np.stack([base["x"], base["y"], base["z"]], axis=1).astype(np.float64)
    print(f"  {len(gs_xyz):,} Gaussians  ({len(prop_names)} properties)")
    print(f"  X: [{gs_xyz[:,0].min():.3f}, {gs_xyz[:,0].max():.3f}]")
    print(f"  Y: [{gs_xyz[:,1].min():.3f}, {gs_xyz[:,1].max():.3f}]")
    print(f"  Z: [{gs_xyz[:,2].min():.3f}, {gs_xyz[:,2].max():.3f}]")
    gs_centre = gs_xyz.mean(axis=0)
    gs_extent = gs_xyz.max(axis=0) - gs_xyz.min(axis=0)
    print(f"  Centre : {gs_centre.round(3)}")
    print(f"  Extent : {gs_extent.round(3)}")

    # --- Bounding-box comparison ---
    print("\n[3] Bounding-box comparison:")
    centre_offset = np.linalg.norm(gs_centre - colmap_centre)
    extent_ratio  = gs_extent / (colmap_extent + 1e-9)
    print(f"  Centre offset (L2) : {centre_offset:.4f}")
    print(f"  Extent ratio (gs/colmap): {extent_ratio.round(3)}")
    if centre_offset < 0.5 * colmap_extent.max() and extent_ratio.max() < 2.0 and extent_ratio.min() > 0.5:
        print("  ✓ Spaces appear ALIGNED (offset small, extents similar)")
    else:
        print("  ✗ Spaces appear MISALIGNED — normalization likely applied")
        print("    Check gsplat training output for a scene_scale / scene_center file.")

    # --- Camera-centre injection ---
    print("\n[4] Injecting camera centres (green) ...")
    cam_centres = _read_camera_centres(COLMAP_DIR / "images.bin")
    print(f"  {len(cam_centres)} cameras")
    log_r_cam = float(np.log(CAM_MARKER_RADIUS))
    cam_markers = _make_sphere_gaussians(
        cam_centres,
        colour=np.array([0.0, 1.0, 0.0]),
        log_radius=log_r_cam,
        prop_names=prop_names,
    )
    out_cam = OUT_DIR / "room_default_cam_markers.ply"
    _save_merged(out_cam, base, cam_markers, prop_names)
    print("  Open in SuperSplat — green spheres should sit at each camera position.")

    # --- COLMAP cloud overlay ---
    print("\n[5] Injecting COLMAP point sample (cyan) ...")
    rng  = np.random.default_rng(42)
    idx  = rng.choice(len(colmap_pts), size=min(N_OVERLAY, len(colmap_pts)), replace=False)
    sample = colmap_pts[idx]
    log_r_pt = float(np.log(CLOUD_MARKER_RADIUS))
    cloud_markers = _make_sphere_gaussians(
        sample,
        colour=np.array([0.0, 0.8, 0.8]),
        log_radius=log_r_pt,
        prop_names=prop_names,
    )
    out_cloud = OUT_DIR / "room_default_colmap_overlay.ply"
    _save_merged(out_cloud, base, cloud_markers, prop_names)
    print("  Open in SuperSplat — cyan dots should land on scene surfaces.")

    print("\nDone.")


if __name__ == "__main__":
    main()