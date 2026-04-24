"""Export WildDet3D 3D boxes + COLMAP point cloud to .ply files.

Data layout expected
--------------------
    ~/data/<dataset>/<scene>/
        images/          original images (may contain sub-directories)
        sparse/          COLMAP sparse reconstruction
            0/
                cameras.bin
                images.bin
                points3D.bin

Prerequisite
------------
    Run extract_intrinsics.py and wildDet3D_infer.py first so the required files exist:
        python extract_intrinsics.py <dataset> <scene> <image>
        python wildDet3D_infer.py    <dataset> <scene> <image> --prompt <categories>

Usage
-----
    python visualize_boxes3d.py <dataset> <scene> <image> [options]

Arguments
---------
  dataset   Dataset name, e.g. "mipnerf360".
  scene     Scene name,   e.g. "room".
  image     Image name exactly as stored in images.bin.
            Include the sub-directory if present:
              e.g. "DSCF4702.JPG"  or  "pano_camera1/pano_00051.jpg"

Scale options (one of the two, or neither for scale=1.0 fallback):
  --colmap-scale FLOAT   Metres per COLMAP unit, e.g. 0.5. Skips auto-estimation.
                         Default: auto-estimated from depth_map saved by wildDet3D_infer.py.

COLMAP point cloud output  (<stem>_scene.ply):
  --clip-radius FLOAT    Discard COLMAP points farther than this distance
                         (COLMAP units) from the camera centre.
                         Default: None (keep all).
  --box-color R,G,B      Box edge colour as uint8 R,G,B. Default: 255,0,0 (red).
  --cloud-color R,G,B    Point cloud colour as uint8 R,G,B. Default: 180,180,180.
  --pts-per-edge INT     Sampled points per box edge. Default: 200.

Gaussian Splatting output  (<stem>_gsplat_annotated.ply):
  --exp-scene-name STR   gsplat experiment folder name, e.g. "room_default".
                         When provided, box edges are merged into the gsplat .ply
                         as elongated Gaussians and saved separately.
                         gsplat .ply location:
                           ~/PycharmProjects/gsplat/examples/results/<exp_scene_name>/
                               ply/point_cloud_29999.ply
  --gsplat-box-radius FLOAT
                         Gaussian cross-section radius in metres (default: 0.003).
  --gsplat-segs-per-edge INT
                         Elongated Gaussians per edge (default: 20). Higher values
                         reduce corner protrusion; at N=20 tails extend ~10% of
                         edge length past each corner.
  --gsplat-opacity FLOAT Gaussian opacity in (0, 1) (default: 0.95).

Diagnostic:
  --print-colmap-depths  Print depth statistics for visible COLMAP 3D points
                         in the chosen image, then exit.

Outputs
-------
    <project>/outputs/<dataset>/<scene>/3Dresults/<prompt>/
        <stem>_scene.ply              COLMAP point cloud + box edges (always)
        <stem>_gsplat_annotated.ply   gsplat + box Gaussians (if --exp-scene-name given)

    <prompt> is the --prompt value with commas replaced by underscores.

Coordinate transform
--------------------
    COLMAP:    P_cam_colmap = R @ P_world + t
    WildDet3D: P_cam is in metric metres.
    Scale s (metres/COLMAP-unit) converts:
        P_world = Rᵀ @ (P_cam_metres / s  −  t)
    s is estimated by comparing WildDet3D metric depths against COLMAP point
    depths at the same 2D feature locations.
"""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent
DATA_ROOT    = Path.home() / "data"
GSPLAT_ROOT  = Path.home() / "PycharmProjects" / "gsplat" / "examples" / "results"


# ---------------------------------------------------------------------------
# COLMAP readers
# ---------------------------------------------------------------------------

def read_points3d_bin(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (xyz, rgb) from points3D.bin.  xyz: float64,  rgb: uint8."""
    xyz_list, rgb_list = [], []
    with open(path, "rb") as f:
        (num_pts,) = struct.unpack("<Q", f.read(8))
        for _ in range(num_pts):
            f.read(8)
            x, y, z = struct.unpack("<3d", f.read(24))
            r, g, b = struct.unpack("<3B", f.read(3))
            f.read(8)
            (tl,) = struct.unpack("<Q", f.read(8))
            f.read(tl * 8)
            xyz_list.append((x, y, z))
            rgb_list.append((r, g, b))
    return np.array(xyz_list, dtype=np.float64), np.array(rgb_list, dtype=np.uint8)


def read_points3d_xyz(path: Path) -> dict[int, np.ndarray]:
    """Return {point3d_id: xyz} from points3D.bin."""
    pts: dict[int, np.ndarray] = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            (pid,) = struct.unpack("<Q", f.read(8))
            x, y, z = struct.unpack("<3d", f.read(24))
            f.read(11)
            (tl,) = struct.unpack("<Q", f.read(8))
            f.read(tl * 8)
            pts[pid] = np.array([x, y, z])
    return pts


def read_image_tracks(
    images_bin: Path, image_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return (xy2d, point3d_ids) for the named image.

    xy2d:        (N, 2) float64 — pixel coords in original image
    point3d_ids: (N,)  int64
    """
    with open(images_bin, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            f.read(4)
            f.read(32 + 24)
            f.read(4)
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            (num_pts,) = struct.unpack("<Q", f.read(8))
            if name.decode() != image_name:
                f.read(num_pts * 24)
                continue
            xy_list, pid_list = [], []
            for _ in range(num_pts):
                x2d, y2d = struct.unpack("<2d", f.read(16))
                (pid,) = struct.unpack("<q", f.read(8))
                xy_list.append((x2d, y2d))
                pid_list.append(pid)
            return np.array(xy_list, dtype=np.float64), np.array(pid_list, dtype=np.int64)
    raise KeyError(f"Image '{image_name}' not found in {images_bin}")


def read_all_extrinsics(
    images_bin: Path,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read (R, t) world-to-camera extrinsics for ALL images in one pass.

    Returns {image_name: (R, t)}  where  P_cam = R @ P_world + t.
    """
    from scipy.spatial.transform import Rotation
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    with open(images_bin, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            f.read(4)                                   # image_id
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz      = struct.unpack("<3d", f.read(24))
            f.read(4)                                   # cam_id
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
            result[name.decode()] = (R, t)
    return result


def read_all_image_tracks(
    images_bin: Path,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read 2D feature tracks for ALL images in images.bin in one pass.

    Returns {image_name: (xy2d, point3d_ids)} where
        xy2d:         (N, 2) float64 — pixel coords in original image
        point3d_ids:  (N,)   int64
    """
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    with open(images_bin, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            f.read(4)   # image_id
            f.read(32)  # qvec
            f.read(24)  # tvec
            f.read(4)   # cam_id
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            (num_pts,) = struct.unpack("<Q", f.read(8))
            xy_list, pid_list = [], []
            for _ in range(num_pts):
                x2d, y2d = struct.unpack("<2d", f.read(16))
                (pid,)   = struct.unpack("<q", f.read(8))
                xy_list.append((x2d, y2d))
                pid_list.append(pid)
            result[name.decode()] = (
                np.array(xy_list,  dtype=np.float64),
                np.array(pid_list, dtype=np.int64),
            )
    return result


def read_image_extrinsics(
    images_bin: Path, image_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return (R, t) world-to-camera extrinsics.  P_cam = R @ P_world + t."""
    with open(images_bin, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            f.read(4)
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz      = struct.unpack("<3d", f.read(24))
            f.read(4)
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            (num_pts,) = struct.unpack("<Q", f.read(8))
            f.read(num_pts * 24)
            if name.decode() == image_name:
                from scipy.spatial.transform import Rotation
                R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
                t = np.array([tx, ty, tz])
                return R, t
    raise KeyError(f"Image '{image_name}' not found in {images_bin}")


def find_sparse_dir(root: Path) -> Path:
    if (root / "cameras.bin").exists():
        return root
    for sub in sorted(root.iterdir()):
        if sub.is_dir() and (sub / "cameras.bin").exists():
            return sub
    raise FileNotFoundError(f"cameras.bin not found under {root}")


# ---------------------------------------------------------------------------
# COLMAP scale estimation
# ---------------------------------------------------------------------------

def estimate_colmap_scale(
    sparse_dir: Path,
    image_name: str,
    R: np.ndarray,
    t: np.ndarray,
    depth_map_path: Path,
    image_file: Path,
) -> float:
    """Estimate COLMAP scale (metres per COLMAP unit) from a WildDet3D depth map.

    For each COLMAP 3D point visible in this image:
        z_colmap = (R @ P_world + t)[2]
        z_metric = depth_map[y', x']   (sampled at preprocessed coordinates)
    scale = median(z_metric / z_colmap), with p10–p90 IQR filter.

    Coordinate mapping from original pixel (x2d, y2d) to depth-map pixel:
        WildDet3D resizes (aspect-ratio-preserving) so width = 1008,
        then center-pads to 1008×1008.
        x' = round(x2d * resized_w / W_orig + left_pad)
        y' = round(y2d * resized_h / H_orig + top_pad)
    """
    from PIL import Image as PILImage

    depth_raw = np.load(depth_map_path)
    if depth_raw.ndim == 4:
        depth_raw = depth_raw[0]
    if depth_raw.ndim == 3:
        depth_raw = depth_raw[0]
    depth_map = depth_raw.astype(np.float32)
    H_dm, W_dm = depth_map.shape

    valid = depth_map[depth_map > 0]
    print(f"  Depth map: {depth_map.shape}  "
          f"range [{valid.min():.2f}, {valid.max():.2f}] m  "
          f"median {np.median(valid):.2f} m")

    with PILImage.open(image_file) as img_pil:
        H_orig, W_orig = img_pil.height, img_pil.width

    # Replicate GenResizeParameters(shape=(1008, 1008))
    TARGET = 1008
    resize_scale = TARGET / W_orig if W_orig >= H_orig else TARGET / H_orig
    resized_h = math.ceil(H_orig * resize_scale - 0.5)
    resized_w = math.ceil(W_orig * resize_scale - 0.5)
    top_pad   = (TARGET - resized_h) // 2
    left_pad  = (TARGET - resized_w) // 2
    scale_x   = resized_w / W_orig
    scale_y   = resized_h / H_orig

    print(f"  Image: {H_orig}×{W_orig}  →  resized: {resized_h}×{resized_w}  "
          f"→  padded: {TARGET}×{TARGET}  (top_pad={top_pad}, left_pad={left_pad})")

    xy2d, pids = read_image_tracks(sparse_dir / "images.bin", image_name)
    pts3d      = read_points3d_xyz(sparse_dir / "points3D.bin")

    z_c_list, z_m_list = [], []
    for (x2d, y2d), pid in zip(xy2d, pids):
        if pid < 0 or pid not in pts3d:
            continue
        p_cam = R @ pts3d[pid] + t
        if p_cam[2] <= 0:
            continue
        xi = int(np.clip(round(x2d * scale_x + left_pad), 0, W_dm - 1))
        yi = int(np.clip(round(y2d * scale_y + top_pad),  0, H_dm - 1))
        z_m = float(depth_map[yi, xi])
        if z_m <= 0:
            continue
        z_c_list.append(p_cam[2])
        z_m_list.append(z_m)

    if len(z_c_list) < 10:
        print("  WARNING: too few matches, falling back to scale=1.0")
        return 1.0

    ratios = np.array(z_m_list) / np.array(z_c_list)
    lo, hi  = np.percentile(ratios, 10), np.percentile(ratios, 90)
    inliers = ratios[(ratios >= lo) & (ratios <= hi)]
    scale   = float(np.median(inliers))

    print(f"  {len(z_c_list)} points matched  →  "
          f"scale = {scale:.4f} m/unit  (IQR [{lo:.4f}, {hi:.4f}])")
    return scale


# ---------------------------------------------------------------------------
# Coordinate transform
# ---------------------------------------------------------------------------

def cam_to_world(p_cam: np.ndarray, R: np.ndarray, t: np.ndarray,
                 scale: float = 1.0) -> np.ndarray:
    """Transform (N, 3) metric camera-space points to COLMAP world space.

    P_world = Rᵀ @ (P_cam_metres / scale  −  t)
    """
    return (R.T @ (p_cam.T / scale - t[:, None])).T


# ---------------------------------------------------------------------------
# Box edge sampling
# ---------------------------------------------------------------------------

_BOX_EDGES = np.array([
    [0, 1], [0, 2], [1, 3], [2, 3],  # top face
    [4, 5], [4, 6], [5, 7], [6, 7],  # bottom face
    [0, 4], [1, 5], [2, 6], [3, 7],  # verticals
], dtype=np.int32)


def _box_corners(center: np.ndarray, R_world: np.ndarray,
                 W: float, L: float, H: float) -> np.ndarray:
    """Return (8, 3) world-space corners.  Local axes: X→L, Y→H, Z→W."""
    dx, dy, dz = L / 2, H / 2, W / 2
    local = np.array([
        [ dx,  dy, -dz],  # 0
        [ dx,  dy,  dz],  # 1
        [-dx,  dy, -dz],  # 2
        [-dx,  dy,  dz],  # 3
        [ dx, -dy, -dz],  # 4
        [ dx, -dy,  dz],  # 5
        [-dx, -dy, -dz],  # 6
        [-dx, -dy,  dz],  # 7
    ], dtype=np.float64)
    return (R_world @ local.T).T + center


def sample_box_edges(boxes3d: np.ndarray, R: np.ndarray, t: np.ndarray,
                     pts_per_edge: int = 200, scale: float = 1.0) -> np.ndarray:
    """Sample dense points along every box edge. Returns (M, 3) world-space xyz."""
    from scipy.spatial.transform import Rotation

    all_pts = []
    for row in boxes3d.astype(np.float64):
        W, L, H = float(row[3]), float(row[4]), float(row[5])
        qw, qx, qy, qz = float(row[6]), float(row[7]), float(row[8]), float(row[9])
        R_box        = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        center_world = cam_to_world(row[:3][None], R, t, scale=scale).squeeze(0)
        R_box_world  = R.T @ R_box
        corners = np.ascontiguousarray(
            _box_corners(center_world, R_box_world, W / scale, L / scale, H / scale),
            dtype=np.float64,
        )
        for i, j in _BOX_EDGES:
            alphas = np.linspace(0, 1, pts_per_edge)[:, None]
            all_pts.append(corners[i] * (1 - alphas) + corners[j] * alphas)
    return np.concatenate(all_pts, axis=0)


def sample_box_edges_world(
    boxes3d_world: np.ndarray,
    pts_per_edge: int = 200,
) -> np.ndarray:
    """Sample dense points along box edges for world-space boxes.

    boxes3d_world: (K, 10) already in COLMAP world units.
        Format: [cx, cy, cz, W, L, H, qw, qx, qy, qz]
    Returns (M, 3) world-space xyz.  No R/t/scale conversion needed.
    """
    from scipy.spatial.transform import Rotation
    all_pts = []
    for row in boxes3d_world.astype(np.float64):
        W, L, H = float(row[3]), float(row[4]), float(row[5])
        qw, qx, qy, qz = float(row[6]), float(row[7]), float(row[8]), float(row[9])
        R_box   = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        center  = row[:3].copy()
        corners = np.ascontiguousarray(
            _box_corners(center, R_box, W, L, H), dtype=np.float64
        )
        for i, j in _BOX_EDGES:
            alphas = np.linspace(0, 1, pts_per_edge)[:, None]
            all_pts.append(corners[i] * (1 - alphas) + corners[j] * alphas)
    if not all_pts:
        return np.zeros((0, 3), dtype=np.float64)
    return np.concatenate(all_pts, axis=0)


# ---------------------------------------------------------------------------
# Gaussian Splatting output
# ---------------------------------------------------------------------------

# SH DC basis constant: colour = sigmoid(f_dc * _C0)
_C0 = 0.28209479177387814


def _rotation_z_to_vec(d: np.ndarray) -> np.ndarray:
    """Return (w, x, y, z) quaternion rotating local +Z → world direction d."""
    d = d / (np.linalg.norm(d) + 1e-12)
    cos_a = float(np.clip(d[2], -1.0, 1.0))   # dot([0,0,1], d) = d[2]
    if cos_a > 0.9999:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if cos_a < -0.9999:
        return np.array([0.0, 1.0, 0.0, 0.0])
    axis = np.array([-d[1], d[0], 0.0])
    axis /= np.linalg.norm(axis)
    half = np.arccos(cos_a) / 2.0
    s = np.sin(half)
    return np.array([np.cos(half), s * axis[0], s * axis[1], s * axis[2]])


def _build_gsplat_box_gaussians(
    boxes3d: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    colmap_scale: float,
    prop_names: list[str],
    f_dc: tuple[float, float, float],
    thin_radius_m: float,
    opacity_logit: float,
    segs_per_edge: int,
) -> np.ndarray:
    """Build elongated Gaussians for all box edges in COLMAP world space.

    Each segment spans `1/segs_per_edge` of an edge.  With segs_per_edge=1 a
    single Gaussian covers the whole edge, giving the thinnest possible result.

    scale_0 = scale_1 = log(thin_radius_colmap)  ← cross-section
    scale_2            = log(half_segment_length) ← along the edge
    rot                = quaternion aligning local +Z to the edge direction
    """
    from scipy.spatial.transform import Rotation as Rot

    thin_colmap = thin_radius_m / colmap_scale
    log_thin    = float(np.log(thin_colmap))
    dtype       = [(name, np.float32) for name in prop_names]

    positions: list[np.ndarray] = []
    quats:     list[np.ndarray] = []
    log_halfs: list[float]      = []

    for row in boxes3d.astype(np.float64):
        W, L, H       = float(row[3]), float(row[4]), float(row[5])
        qw, qx, qy, qz = float(row[6]), float(row[7]), float(row[8]), float(row[9])
        R_box         = Rot.from_quat([qx, qy, qz, qw]).as_matrix()
        center_world  = cam_to_world(row[:3][None], R, t, scale=colmap_scale).squeeze(0)
        R_box_world   = R.T @ R_box
        corners = np.ascontiguousarray(
            _box_corners(center_world, R_box_world,
                         W / colmap_scale, L / colmap_scale, H / colmap_scale),
            dtype=np.float64,
        )
        for i, j in _BOX_EDGES:
            ev  = corners[j] - corners[i]
            el  = float(np.linalg.norm(ev))
            if el < 1e-8:
                continue
            seg_half = el / (2.0 * segs_per_edge)
            for k in range(segs_per_edge):
                alpha = (k + 0.5) / segs_per_edge
                positions.append(corners[i] + alpha * ev)
                quats.append(_rotation_z_to_vec(ev / el))
                log_halfs.append(float(np.log(seg_half)))

    n    = len(positions)
    data = np.zeros(n, dtype=dtype)
    pa   = np.array(positions, dtype=np.float32)
    data["x"] = pa[:, 0]
    data["y"] = pa[:, 1]
    data["z"] = pa[:, 2]
    data["f_dc_0"] = np.float32(f_dc[0])
    data["f_dc_1"] = np.float32(f_dc[1])
    data["f_dc_2"] = np.float32(f_dc[2])
    # f_rest stays 0 → no view-dependent colour
    data["opacity"] = np.float32(opacity_logit)
    data["scale_0"] = np.float32(log_thin)
    data["scale_1"] = np.float32(log_thin)
    data["scale_2"] = np.array(log_halfs, dtype=np.float32)
    qa = np.array(quats, dtype=np.float32)
    data["rot_0"] = qa[:, 0]
    data["rot_1"] = qa[:, 1]
    data["rot_2"] = qa[:, 2]
    data["rot_3"] = qa[:, 3]
    return data


# ---------------------------------------------------------------------------
# PLY writer
# ---------------------------------------------------------------------------

def save_ply(path: Path, xyz: np.ndarray, rgb_uint8: np.ndarray) -> None:
    """Write a binary PLY file with xyz (float32) and RGB (uint8) per vertex."""
    assert xyz.shape == rgb_uint8.shape and xyz.shape[1] == 3
    n = len(xyz)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        )
        f.write(header.encode())
        data = np.zeros(n, dtype=[
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ])
        data["x"] = xyz[:, 0].astype(np.float32)
        data["y"] = xyz[:, 1].astype(np.float32)
        data["z"] = xyz[:, 2].astype(np.float32)
        data["red"]   = rgb_uint8[:, 0]
        data["green"] = rgb_uint8[:, 1]
        data["blue"]  = rgb_uint8[:, 2]
        f.write(data.tobytes())
    print(f"Saved {n:,} points → {path}")


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _print_colmap_depths(sparse_dir: Path, image_name: str,
                         R: np.ndarray, t: np.ndarray) -> None:
    pts3d: dict[int, np.ndarray] = {}
    with open(sparse_dir / "points3D.bin", "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            (pid,) = struct.unpack("<Q", f.read(8))
            x, y, z = struct.unpack("<3d", f.read(24))
            f.read(11)
            (tl,) = struct.unpack("<Q", f.read(8))
            f.read(tl * 8)
            pts3d[pid] = np.array([x, y, z])

    depths: list[float] = []
    with open(sparse_dir / "images.bin", "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            f.read(4)
            f.read(32 + 24 + 4)
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            (num_pts,) = struct.unpack("<Q", f.read(8))
            tracks = []
            for _ in range(num_pts):
                f.read(16)
                (pid,) = struct.unpack("<q", f.read(8))
                tracks.append(pid)
            if name.decode() == image_name:
                for pid in tracks:
                    if pid >= 0 and pid in pts3d:
                        p_cam = R @ pts3d[pid] + t
                        if p_cam[2] > 0:
                            depths.append(float(p_cam[2]))
                break

    if not depths:
        print("No visible COLMAP points found for this image.")
        return

    d = np.array(depths)
    print(f"\nCOLMAP depth statistics for '{image_name}' ({len(d)} points):")
    print(f"  median : {np.median(d):.4f} COLMAP units")
    print(f"  mean   : {np.mean(d):.4f} COLMAP units")
    print(f"  p10/p90: {np.percentile(d, 10):.4f} / {np.percentile(d, 90):.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export WildDet3D 3D boxes + COLMAP point cloud to PLY",
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
        "--prompt", type=str, required=True,
        help='Comma-separated detection categories used during inference, '
             'e.g. "chair" or "chair,table". Determines the 3Dresults sub-folder.',
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
    # --- Gaussian Splatting output ---
    parser.add_argument(
        "--exp-scene-name", type=str, default=None,
        help="gsplat experiment folder name, e.g. \"room_default\". "
             "When given, also produces <stem>_gsplat_annotated.ply.",
    )
    parser.add_argument(
        "--gsplat-box-radius", type=float, default=0.003,
        help="Gaussian cross-section radius in metres (default: 0.003).",
    )
    parser.add_argument(
        "--gsplat-segs-per-edge", type=int, default=20,
        help="Elongated Gaussians per box edge (default: 20). "
             "Higher values reduce corner protrusion: at N=20 the tail "
             "extends ~10%% of edge length past each corner.",
    )
    parser.add_argument(
        "--gsplat-opacity", type=float, default=0.95,
        help="Gaussian opacity in (0, 1) (default: 0.95).",
    )
    args = parser.parse_args()

    stem          = Path(args.image).stem
    prompt_folder = args.prompt.replace(",", "_").replace(" ", "_")
    sparse_dir    = find_sparse_dir(DATA_ROOT / args.dataset / args.scene / "sparse")
    out_dir       = (PROJECT_ROOT / "outputs" / args.dataset / args.scene
                     / "3Dresults" / prompt_folder)
    image_file    = DATA_ROOT / args.dataset / args.scene / "images" / args.image

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
                "  Run wildDet3D_infer.py first, or pass --colmap-scale manually."
            )
            colmap_scale = 1.0

    # --- Point cloud ---
    print("Reading point cloud...")
    xyz, rgb = read_points3d_bin(sparse_dir / "points3D.bin")
    print(f"  {len(xyz):,} points  "
          f"(bounds: {xyz.min(axis=0).round(2)} .. {xyz.max(axis=0).round(2)})")

    if args.clip_radius is not None:
        dists = np.linalg.norm(xyz - cam_center_world, axis=1)
        xyz, rgb = xyz[dists <= args.clip_radius], rgb[dists <= args.clip_radius]
        print(f"  After clip (radius={args.clip_radius}): {len(xyz):,} points  "
              f"(bounds: {xyz.min(axis=0).round(2)} .. {xyz.max(axis=0).round(2)})")

    # --- Boxes ---
    boxes3d_path = out_dir / f"{stem}_boxes3d.npy"
    if not boxes3d_path.exists():
        raise FileNotFoundError(
            f"boxes3d not found: {boxes3d_path}\n"
            f"Run first:  python wildDet3D_infer.py {args.dataset} {args.scene} {args.image} --prompt ..."
        )
    boxes3d = np.load(boxes3d_path).astype(np.float64)
    print(f"  {len(boxes3d)} box(es) loaded")

    # --- Camera marker: green sphere ---
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
        print(f"  {len(el_in.data):,} existing Gaussians  ({len(prop_names)} properties)")

        # Convert box colour to SH DC coefficients: f_dc = logit(colour) / C0
        c    = np.clip(box_color.astype(np.float64) / 255.0, 0.001, 0.999)
        f_dc = tuple(float(np.log(c[i] / (1.0 - c[i])) / _C0) for i in range(3))
        opacity_logit = float(np.log(args.gsplat_opacity / (1.0 - args.gsplat_opacity)))

        print(f"Sampling gsplat box Gaussians "
              f"({args.gsplat_segs_per_edge} seg(s)/edge × "
              f"{len(_BOX_EDGES)} edges × {len(boxes3d)} box(es))...")
        new_g = _build_gsplat_box_gaussians(
            boxes3d, R, t, colmap_scale, prop_names,
            f_dc, args.gsplat_box_radius, opacity_logit, args.gsplat_segs_per_edge,
        )
        print(f"  {len(new_g):,} new Gaussians  "
              f"(radius={args.gsplat_box_radius}m → "
              f"{args.gsplat_box_radius / colmap_scale:.5f} COLMAP units)")

        merged_g    = np.concatenate([el_in.data, new_g])
        out_gsplat  = out_dir / f"{stem}_gsplat_annotated.ply"
        PlyData([PlyElement.describe(merged_g, "vertex")], text=False).write(str(out_gsplat))
        print(f"Saved {len(merged_g):,} Gaussians → {out_gsplat}")


if __name__ == "__main__":
    main()