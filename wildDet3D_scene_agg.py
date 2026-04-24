"""Aggregate per-frame 3D boxes across all scene frames into scene-level objects.

Algorithm
---------
1.  Load per-frame boxes3d, labels, scores.
2.  Compute scene-level COLMAP scale (global median of all depth-map / COLMAP-point pairs).
3.  Transform all boxes to COLMAP world space.
4.  Encode unique labels with the OWLv2 CLIP text encoder.
5.  Semantic label grouping: complete-linkage clustering on label embeddings so every
    pair within a group has cosine similarity > sem_threshold.
6.  Greedy IoU clustering within each label group:
      - Sort detections by combined confidence (descending) → highest-confidence box seeds
      - For each box: compute 3D IoU against current median of every existing cluster
      - Assign to highest-IoU cluster if IoU >= iou_threshold; else start new cluster
      - After each assignment update the cluster's median box
7.  Discard clusters with fewer than ceil(min_cluster_frac × n_frames_with_detections) boxes.
8.  Per cluster: produce median_box, best_score_box, best_location_box.
9.  Final label = label with highest Σ combined_confidence among cluster members.

Requires CUDA (vis4d_cuda_ops) for exact 3D volumetric IoU.

Outputs
-------
    auto_owlv2/scene/
        scene_colmap_scale.txt
        scene_objects.json
        scene_median_boxes3d_world.npy
        scene_best_score_boxes3d_world.npy
        scene_best_location_boxes3d_world.npy

    auto_owlv2/debug/   (only with --debug)
        debug_step1_all_detections.json
        debug_step2_colmap_scale.json
        debug_step3_world_centers.json
        debug_step4_embeddings.json
        debug_step5_label_groups.json
        debug_step6_clustering_trace.json
        debug_step7_clusters_filtered.json

Usage
-----
    python wildDet3D_scene_agg.py <dataset> <scene> [options]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from visualize_boxes3d import (
    cam_to_world,
    find_sparse_dir,
    read_all_extrinsics,
    read_all_image_tracks,
    read_points3d_xyz,
)

DATA_ROOT    = Path.home() / "data"
AUTO_FOLDER  = "auto_owlv2"
FRAME_FOLDER = "per_frame"
SCENE_FOLDER = "scene"
DEBUG_FOLDER = "debug"


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def R_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(R).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def chordal_mean_quat(quats_wxyz: list[np.ndarray]) -> np.ndarray:
    R_stack = np.stack([quat_wxyz_to_R(q) for q in quats_wxyz])
    R_bar   = R_stack.mean(axis=0)
    U, _, Vt = np.linalg.svd(R_bar)
    R_mean  = U @ Vt
    if np.linalg.det(R_mean) < 0:
        U[:, -1] *= -1
        R_mean = U @ Vt
    return R_to_quat_wxyz(R_mean)


# ---------------------------------------------------------------------------
# IoU helpers — CUDA (vis4d_cuda_ops) with numpy/scipy exact fallback
# ---------------------------------------------------------------------------

# Edge pairs for iou_box3d corner ordering
_IOU_EDGE_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
    (4, 5), (5, 6), (6, 7), (7, 4),  # top face
    (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
]

_IOU_FACE_QUADS = [
    [0, 1, 2, 3], [4, 5, 6, 7],  # bottom / top
    [0, 1, 5, 4], [3, 2, 6, 7],  # front / back
    [0, 3, 7, 4], [1, 2, 6, 5],  # left / right
]


def _box_face_planes(corners: np.ndarray) -> list[tuple[np.ndarray, float]]:
    """Inward-pointing half-space planes (normal, d). Inside iff normal @ p >= d."""
    center = corners.mean(axis=0)
    planes = []
    for fq in _IOU_FACE_QUADS:
        v0, v1, v2 = corners[fq[0]], corners[fq[1]], corners[fq[2]]
        n = np.cross(v1 - v0, v2 - v0)
        norm = float(np.linalg.norm(n))
        if norm < 1e-10:
            continue
        n = n / norm
        d = float(n @ v0)
        if float(n @ center) < d:
            n, d = -n, -d
        planes.append((n, d))
    return planes


def _inside_planes(p: np.ndarray, planes: list) -> bool:
    return all(float(n @ p) >= d - 1e-7 for n, d in planes)


def _edge_plane_isect(
    p1: np.ndarray, p2: np.ndarray, n: np.ndarray, d: float
) -> np.ndarray | None:
    denom = float(n @ (p2 - p1))
    if abs(denom) < 1e-10:
        return None
    t = (d - float(n @ p1)) / denom
    if -1e-7 <= t <= 1.0 + 1e-7:
        return p1 + float(np.clip(t, 0.0, 1.0)) * (p2 - p1)
    return None


def box3d_iou_numpy(corners1: np.ndarray, corners2: np.ndarray) -> float:
    """Exact oriented 3D IoU — pure numpy/scipy, no CUDA required.

    Intersection polytope vertices are found by collecting:
      - corners of box1 inside box2
      - corners of box2 inside box1
      - intersections of each box's edges with the other box's faces
    Volume is then computed via scipy ConvexHull.
    """
    from scipy.spatial import ConvexHull
    from scipy.spatial.qhull import QhullError

    try:
        vol1 = ConvexHull(corners1).volume
        vol2 = ConvexHull(corners2).volume
    except (QhullError, ValueError):
        return 0.0

    planes1 = _box_face_planes(corners1)
    planes2 = _box_face_planes(corners2)

    candidates: list[np.ndarray] = []

    for p in corners1:
        if _inside_planes(p, planes2):
            candidates.append(p)
    for p in corners2:
        if _inside_planes(p, planes1):
            candidates.append(p)
    for i, j in _IOU_EDGE_PAIRS:
        for n, d in planes2:
            pt = _edge_plane_isect(corners1[i], corners1[j], n, d)
            if pt is not None and _inside_planes(pt, planes2):
                candidates.append(pt)
    for i, j in _IOU_EDGE_PAIRS:
        for n, d in planes1:
            pt = _edge_plane_isect(corners2[i], corners2[j], n, d)
            if pt is not None and _inside_planes(pt, planes1):
                candidates.append(pt)

    if len(candidates) < 4:
        return 0.0

    try:
        inter_vol = ConvexHull(np.array(candidates)).volume
    except (QhullError, ValueError):
        return 0.0

    union_vol = vol1 + vol2 - inter_vol
    return float(inter_vol / union_vol) if union_vol > 1e-10 else 0.0


def detect_iou_backend() -> str:
    """Return 'cuda' if vis4d_cuda_ops is available, 'numpy' otherwise."""
    import torch
    if torch.cuda.is_available():
        try:
            from vis4d_cuda_ops import iou_box3d as _  # noqa: F401
            print("  IoU backend : CUDA (vis4d_cuda_ops)")
            return "cuda"
        except ImportError:
            pass
    print("  IoU backend : numpy/scipy (exact, no CUDA extension required)")
    return "numpy"


def box_world_to_corners_iou(box_world: np.ndarray) -> np.ndarray:
    """Convert [cx,cy,cz,W,L,H,qw,qx,qy,qz] → (8,3) corners in iou_box3d ordering.

    Corner layout (matches vis4d_cuda_ops.iou_box3d face topology):
        bottom face (corners 0-3): z = -H/2
        top    face (corners 4-7): z = +H/2
    """
    from scipy.spatial.transform import Rotation
    cx, cy, cz   = float(box_world[0]), float(box_world[1]), float(box_world[2])
    W,  L,  H    = float(box_world[3]), float(box_world[4]), float(box_world[5])
    qw, qx, qy, qz = (float(box_world[6]), float(box_world[7]),
                      float(box_world[8]), float(box_world[9]))

    R      = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    center = np.array([cx, cy, cz], dtype=np.float64)

    hw, hl, hh = W / 2.0, L / 2.0, H / 2.0
    local = np.array([
        [-hw, -hl, -hh],  # 0
        [ hw, -hl, -hh],  # 1
        [ hw,  hl, -hh],  # 2
        [-hw,  hl, -hh],  # 3
        [-hw, -hl,  hh],  # 4
        [ hw, -hl,  hh],  # 5
        [ hw,  hl,  hh],  # 6
        [-hw,  hl,  hh],  # 7
    ], dtype=np.float64)

    return (R @ local.T).T + center   # (8, 3)


def compute_iou_box_vs_cluster_medians(
    box_corners: np.ndarray,                  # (8, 3)
    median_corners_list: list[np.ndarray],    # each (8, 3)
    iou_backend: str,                         # 'cuda' or 'numpy'
) -> np.ndarray:
    """Compute IoU of one box against K cluster medians. Returns (K,) numpy array."""
    K = len(median_corners_list)
    if K == 0:
        return np.array([], dtype=np.float32)

    if iou_backend == "cuda":
        import torch
        from wilddet3d.ops.box3d import box3d_overlap
        box_t     = torch.tensor(box_corners, dtype=torch.float32, device="cuda").unsqueeze(0)
        medians_t = torch.tensor(np.stack(median_corners_list), dtype=torch.float32, device="cuda")
        iou = box3d_overlap(box_t, medians_t)
        return iou[0].cpu().numpy()
    else:
        return np.array(
            [box3d_iou_numpy(box_corners, m) for m in median_corners_list],
            dtype=np.float32,
        )


# ---------------------------------------------------------------------------
# Phase 1 — Load per-frame data
# ---------------------------------------------------------------------------

def load_per_frame_data(
    frame_dir: Path,
    scene_dir: Path,
) -> tuple[list[dict], int, dict]:
    manifest: dict = json.loads((scene_dir / "processed_images.json").read_text())
    n_frames = len(manifest)
    detections: list[dict] = []

    n_with_boxes = 0
    for stem, info in manifest.items():
        boxes_path = frame_dir / f"{stem}_boxes3d.npy"
        if not boxes_path.exists():
            continue
        boxes3d = np.load(boxes_path)
        if boxes3d.ndim != 2 or len(boxes3d) == 0:
            continue

        class_ids  = np.load(frame_dir / f"{stem}_class_ids.npy")
        scores_2d  = np.load(frame_dir / f"{stem}_scores_2d.npy")
        scores_3d  = np.load(frame_dir / f"{stem}_scores_3d.npy")
        labels_txt = (frame_dir / f"{stem}_unique_labels.txt").read_text().strip()
        vocab      = labels_txt.split(",") if labels_txt else []

        for i in range(len(boxes3d)):
            cid   = int(class_ids[i])
            label = vocab[cid] if cid < len(vocab) else "unknown"
            detections.append({
                "center_cam":    boxes3d[i, :3],
                "dims_m":        boxes3d[i, 3:6],
                "quat_wxyz_cam": boxes3d[i, 6:10],
                "label":         label,
                "score_2d":      float(scores_2d[i]),
                "score_3d":      float(scores_3d[i]),
                "frame_stem":    stem,
                "colmap_path":   info["colmap_path"],
                "original_hw":   info["original_hw"],
            })
        n_with_boxes += 1

    print(f"  {len(detections)} detections from {n_with_boxes}/{n_frames} frames")
    return detections, n_frames, n_with_boxes, manifest


# ---------------------------------------------------------------------------
# Phase 2 — Scene-level COLMAP scale
# ---------------------------------------------------------------------------

def compute_scene_colmap_scale(
    manifest: dict,
    frame_dir: Path,
    sparse_dir: Path,
) -> tuple[float, dict[str, float]]:
    TARGET = 1008
    print("  Reading COLMAP tracks and 3D points (one pass each)...")
    all_tracks = read_all_image_tracks(sparse_dir / "images.bin")
    pts3d      = read_points3d_xyz(sparse_dir / "points3D.bin")
    all_extr   = read_all_extrinsics(sparse_dir / "images.bin")

    all_ratios: list[float] = []
    per_frame_scales: dict[str, float] = {}

    for stem, info in manifest.items():
        depth_path = frame_dir / f"{stem}_depth_map.npy"
        if not depth_path.exists():
            continue
        depth_raw = np.load(depth_path)
        if depth_raw.ndim == 4: depth_raw = depth_raw[0]
        if depth_raw.ndim == 3: depth_raw = depth_raw[0]
        depth_map = depth_raw.astype(np.float32)
        if depth_map.shape[0] == 0:
            continue

        H_dm, W_dm     = depth_map.shape
        H_orig, W_orig = info["original_hw"]
        colmap_name    = info["colmap_path"]

        R, t = all_extr.get(colmap_name, (None, None))
        if R is None:
            continue
        xy2d, pids = all_tracks.get(colmap_name, (None, None))
        if xy2d is None or len(xy2d) == 0:
            continue

        resize_sc = TARGET / W_orig if W_orig >= H_orig else TARGET / H_orig
        resized_h = math.ceil(H_orig * resize_sc - 0.5)
        resized_w = math.ceil(W_orig * resize_sc - 0.5)
        top_pad   = (TARGET - resized_h) // 2
        left_pad  = (TARGET - resized_w) // 2
        sx = resized_w / W_orig
        sy = resized_h / H_orig

        frame_ratios: list[float] = []
        for (x2d, y2d), pid in zip(xy2d, pids):
            if pid < 0 or pid not in pts3d:
                continue
            p_cam = R @ pts3d[pid] + t
            if p_cam[2] <= 0:
                continue
            xi = int(np.clip(round(x2d * sx + left_pad), 0, W_dm - 1))
            yi = int(np.clip(round(y2d * sy + top_pad),  0, H_dm - 1))
            z_m = float(depth_map[yi, xi])
            if z_m <= 0:
                continue
            frame_ratios.append(z_m / p_cam[2])

        if frame_ratios:
            per_frame_scales[stem] = float(np.median(frame_ratios))
            all_ratios.extend(frame_ratios)

    if len(all_ratios) < 20:
        print(f"  WARNING: only {len(all_ratios)} depth pairs — using scale=1.0")
        return 1.0, per_frame_scales

    ratios  = np.array(all_ratios)
    lo, hi  = np.percentile(ratios, 10), np.percentile(ratios, 90)
    inliers = ratios[(ratios >= lo) & (ratios <= hi)]
    scale   = float(np.median(inliers))
    print(f"  Scale: {scale:.5f} m/unit  "
          f"({len(inliers)}/{len(all_ratios)} pairs, IQR [{lo:.4f}, {hi:.4f}])")
    return scale, per_frame_scales


# ---------------------------------------------------------------------------
# Phase 3 — Transform to world space
# ---------------------------------------------------------------------------

def transform_to_world(
    detections: list[dict],
    all_extr: dict[str, tuple],
    colmap_scale: float,
    per_frame_scales: dict[str, float] | None = None,
    scale_mode: str = "global",
) -> None:
    for det in detections:
        R, t = all_extr[det["colmap_path"]]
        if scale_mode == "align":
            scale = 1.0
        elif scale_mode == "local" and per_frame_scales:
            scale = per_frame_scales.get(det["frame_stem"], colmap_scale)
        else:
            scale = colmap_scale
        det["_scale_used"] = scale   # stored for debug
        center_world = cam_to_world(
            det["center_cam"][None], R, t, scale=scale
        ).squeeze(0)
        dims_colmap  = det["dims_m"] / scale
        R_box_cam    = quat_wxyz_to_R(det["quat_wxyz_cam"])
        R_box_world  = R.T @ R_box_cam
        q_world      = R_to_quat_wxyz(R_box_world)

        det["center_world"]    = center_world
        det["dims_colmap"]     = dims_colmap
        det["R_box_world"]     = R_box_world
        det["quat_wxyz_world"] = q_world
        det["box_world"]       = np.concatenate([center_world, dims_colmap, q_world])


# ---------------------------------------------------------------------------
# Phase 4 — Semantic embeddings
# ---------------------------------------------------------------------------

def compute_label_embeddings(
    unique_labels: list[str],
    device: str,
) -> dict[str, np.ndarray]:
    from owl import OwlWrapper
    import torch
    print(f"  Encoding {len(unique_labels)} label(s) with OWLv2 CLIP...")
    owl  = OwlWrapper(device=device, warmup=False)
    with torch.no_grad():
        embs = owl._encode_text(unique_labels).float().cpu().numpy()
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
    embs  = embs / norms
    return {lbl: embs[i] for i, lbl in enumerate(unique_labels)}


# ---------------------------------------------------------------------------
# Phase 5 — Semantic label grouping (complete-linkage)
# ---------------------------------------------------------------------------

def semantic_label_groups(
    label_embs: dict[str, np.ndarray],
    sem_threshold: float,
) -> list[list[str]]:
    """Group labels so every pair within a group has cosine similarity > threshold.

    Uses complete-linkage hierarchical clustering: the within-group maximum
    pairwise cosine distance is <= (1 - sem_threshold).
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    unique_labels = sorted(label_embs.keys())
    if len(unique_labels) == 1:
        return [unique_labels]

    emb_mat  = np.stack([label_embs[l] for l in unique_labels])     # (L, D)
    cos_sim  = np.clip(emb_mat @ emb_mat.T, -1.0, 1.0)              # (L, L)
    cos_dist = np.clip(1.0 - cos_sim, 0.0, None)
    np.fill_diagonal(cos_dist, 0.0)

    condensed = squareform(cos_dist, checks=False)
    Z         = linkage(condensed, method="complete")
    cids      = fcluster(Z, t=1.0 - sem_threshold, criterion="distance")

    groups: dict[int, list[str]] = defaultdict(list)
    for lbl, cid in zip(unique_labels, cids):
        groups[int(cid)].append(lbl)

    return list(groups.values())


# ---------------------------------------------------------------------------
# Semantic group PLY export
# ---------------------------------------------------------------------------

def save_semantic_group_plys(
    label_groups: list[list[str]],
    label_to_group: dict[str, int],
    detections: list[dict],
    sparse_dir: Path,
    scene_dir: Path,
    pts_per_edge: int = 100,
) -> None:
    """Write one PLY per semantic group to scene_dir.

    Each PLY contains the COLMAP point cloud (grey) plus box wireframes (orange)
    for every detection whose label belongs to that group.  Files are named
    scene_<label>.ply using the most-frequent label in the group.
    """
    from collections import Counter
    from visualize_boxes3d import read_points3d_bin, sample_box_edges_world, save_ply

    print("  Reading COLMAP point cloud for group PLYs...")
    xyz, rgb = read_points3d_bin(sparse_dir / "points3D.bin")
    cloud_rgb = np.tile(np.array([180, 180, 180], dtype=np.uint8), (len(xyz), 1))
    box_color = np.array([255, 140, 0], dtype=np.uint8)   # orange

    for gid, group_labels in enumerate(label_groups):
        group_set = set(group_labels)
        group_dets = [d for d in detections if d["label"] in group_set]
        if not group_dets:
            continue

        # Name from most-frequent label in this group's detections
        label_counts = Counter(d["label"] for d in group_dets)
        name = label_counts.most_common(1)[0][0].replace(" ", "_").replace("/", "_")

        boxes = np.stack([d["box_world"] for d in group_dets])  # (N, 10)
        box_pts = sample_box_edges_world(boxes, pts_per_edge=pts_per_edge)
        box_rgb = np.tile(box_color, (len(box_pts), 1))

        merged_xyz = np.concatenate([xyz, box_pts], axis=0).astype(np.float32)
        merged_rgb = np.concatenate([cloud_rgb, box_rgb], axis=0)

        out_path = scene_dir / f"scene_{name}.ply"
        save_ply(out_path, merged_xyz, merged_rgb)
        print(f"    [{gid}] {sorted(group_labels)}  "
              f"({len(group_dets)} detections) → {out_path.name}")


# ---------------------------------------------------------------------------
# Phase 6 — Greedy IoU clustering
# ---------------------------------------------------------------------------

def _combined_score(det: dict, w2d: float, w3d: float) -> float:
    return (det["score_2d"] ** w2d) * (det["score_3d"] ** w3d)


def _median_box_from_indices(
    indices: list[int],
    detections: list[dict],
) -> np.ndarray:
    """Compute synthetic median box (world space) for a list of detection indices."""
    members    = [detections[i] for i in indices]
    centers    = np.stack([d["center_world"] for d in members])
    dims       = np.stack([d["dims_colmap"]   for d in members])
    quats      = [d["quat_wxyz_world"] for d in members]
    med_center = np.median(centers, axis=0)
    med_dims   = np.median(dims,    axis=0)
    med_quat   = chordal_mean_quat(quats) if len(quats) > 1 else quats[0].copy()
    return np.concatenate([med_center, med_dims, med_quat])


def greedy_iou_cluster(
    group_detection_indices: list[int],
    detections: list[dict],
    iou_threshold: float,
    score_w2d: float,
    score_w3d: float,
    iou_backend: str,
) -> list[list[int]]:
    """Greedy IoU-based clustering for one semantic group.

    Returns a list of clusters, each a list of detection indices into `detections`.
    If `trace` is not None, per-detection assignment events are appended to it.
    """
    # Sort by combined confidence descending — highest-confidence box seeds first
    sorted_idx = sorted(
        group_detection_indices,
        key=lambda i: _combined_score(detections[i], score_w2d, score_w3d),
        reverse=True,
    )

    clusters: list[list[int]]         = []
    cluster_medians: list[np.ndarray] = []   # (K,) of box_world arrays
    cluster_corners: list[np.ndarray] = []   # (K,) of (8,3) corner arrays

    for det_idx in sorted_idx:
        box     = detections[det_idx]["box_world"]
        corners = box_world_to_corners_iou(box)

        if cluster_corners:
            ious = compute_iou_box_vs_cluster_medians(corners, cluster_corners, iou_backend)
            best_c  = int(np.argmax(ious))
            best_iou = float(ious[best_c])
        else:
            best_c   = -1
            best_iou = 0.0

        if best_c >= 0 and best_iou >= iou_threshold:
            assigned_cluster = best_c
            clusters[best_c].append(det_idx)
            new_median = _median_box_from_indices(clusters[best_c], detections)
            cluster_medians[best_c] = new_median
            cluster_corners[best_c] = box_world_to_corners_iou(new_median)
        else:
            assigned_cluster = len(clusters)
            clusters.append([det_idx])
            cluster_medians.append(box.copy())
            cluster_corners.append(corners)


    return clusters




# ---------------------------------------------------------------------------
# Phase 8 — Aggregation
# ---------------------------------------------------------------------------

def aggregate_cluster(
    cluster_indices: list[int],
    detections: list[dict],
    score_w2d: float,
    score_w3d: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (median_box, best_score_box, best_location_box) as (10,) arrays."""
    members  = [detections[i] for i in cluster_indices]
    centers  = np.stack([d["center_world"]    for d in members])
    dims     = np.stack([d["dims_colmap"]      for d in members])
    quats    = [d["quat_wxyz_world"] for d in members]
    s2d      = np.array([d["score_2d"] for d in members])
    s3d      = np.array([d["score_3d"] for d in members])

    med_center = np.median(centers, axis=0)
    med_dims   = np.median(dims,    axis=0)
    med_quat   = chordal_mean_quat(quats)
    median_box = np.concatenate([med_center, med_dims, med_quat])

    combined = (s2d ** score_w2d) * (s3d ** score_w3d)
    best_score_box   = members[int(np.argmax(combined))]["box_world"].copy()
    dists_to_med     = np.linalg.norm(centers - med_center, axis=1)
    best_loc_box     = members[int(np.argmin(dists_to_med))]["box_world"].copy()

    return median_box, best_score_box, best_loc_box


# ---------------------------------------------------------------------------
# Phase 9 — Confidence-weighted label scoring
# ---------------------------------------------------------------------------

def confidence_weighted_label(
    cluster_indices: list[int],
    detections: list[dict],
    score_w2d: float,
    score_w3d: float,
) -> tuple[str, dict[str, float]]:
    """Final label = label with highest Σ combined_confidence."""
    label_scores: dict[str, float] = {}
    for i in cluster_indices:
        d     = detections[i]
        score = _combined_score(d, score_w2d, score_w3d)
        label_scores[d["label"]] = label_scores.get(d["label"], 0.0) + score
    final_label = max(label_scores, key=lambda k: label_scores[k])
    sorted_scores = dict(
        sorted(label_scores.items(), key=lambda kv: -kv[1])
    )
    return final_label, {k: round(v, 4) for k, v in sorted_scores.items()}


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

def _save_debug(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, np.ndarray):
        np.save(path, data)
    else:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate per-frame 3D boxes into scene-level objects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("dataset")
    parser.add_argument("scene")
    parser.add_argument(
        "--sem-threshold", type=float, default=0.9,
        help="Cosine similarity threshold for label grouping (default: 0.9). "
             "Every pair of labels within a group satisfies cos_sim > threshold.",
    )
    parser.add_argument(
        "--iou-threshold", type=float, default=0.5,
        help="Minimum 3D IoU to assign a box to an existing cluster (default: 0.5).",
    )
    threshold_group = parser.add_mutually_exclusive_group()
    threshold_group.add_argument(
        "--min-cluster-frac", type=float, default=None,
        help="Min fraction of frames-with-detections a cluster must contain "
             "to survive (default: 0.15, ignored when --min-cluster-no is set).",
    )
    threshold_group.add_argument(
        "--min-cluster-no", type=int, default=None,
        help="Min absolute number of boxes a cluster must contain to survive "
             "(default: 10 when this flag is used). "
             "Mutually exclusive with --min-cluster-frac.",
    )
    parser.set_defaults(min_cluster_frac=0.15)
    parser.add_argument(
        "--score-w2d", type=float, default=1.0,
        help="Exponent for scores_2d in combined score (default: 1.0).",
    )
    parser.add_argument(
        "--score-w3d", type=float, default=1.0,
        help="Exponent for scores_3d in combined score (default: 1.0).",
    )
    scale_group = parser.add_mutually_exclusive_group()
    scale_group.add_argument(
        "--global", dest="scale_mode", action="store_const", const="global",
        help="Use scene-level (global) COLMAP scale for all frames (default).",
    )
    scale_group.add_argument(
        "--local", dest="scale_mode", action="store_const", const="local",
        help="Use per-frame COLMAP scale to correct per-frame depth bias.",
    )
    scale_group.add_argument(
        "--align", dest="scale_mode", action="store_const", const="align",
        help="Assume COLMAP space is already metric (scale=1.0). "
             "Skips scale estimation entirely.",
    )
    parser.set_defaults(scale_mode="global")
    parser.add_argument(
        "--debug", action="store_true",
        help="Save intermediate debug files to auto_owlv2/debug/.",
    )
    args = parser.parse_args()

    base_out  = (PROJECT_ROOT / "outputs" / args.dataset / args.scene
                 / "3Dresults" / AUTO_FOLDER)
    frame_dir = base_out / FRAME_FOLDER
    scene_dir = base_out / SCENE_FOLDER
    debug_dir = base_out / DEBUG_FOLDER
    scene_dir.mkdir(parents=True, exist_ok=True)

    sparse_dir = find_sparse_dir(DATA_ROOT / args.dataset / args.scene / "sparse")

    print(f"\n{'='*50}")
    print(f" WildDet3D scene aggregation")
    print(f"  dataset        : {args.dataset}")
    print(f"  scene          : {args.scene}")
    print(f"  sem-threshold  : {args.sem_threshold}")
    print(f"  iou-threshold  : {args.iou_threshold}")
    if args.min_cluster_no is not None:
        print(f"  min-cluster-no  : {args.min_cluster_no}")
    else:
        print(f"  min-cluster-frac: {args.min_cluster_frac}")
    print(f"{'='*50}\n")

    import torch
    torch_device  = "cuda" if torch.cuda.is_available() else "cpu"
    iou_backend   = detect_iou_backend()

    # --- Phase 1 ---
    print("[1/9] Loading per-frame data...")
    detections, n_frames, n_frames_with_detections, manifest = load_per_frame_data(frame_dir, scene_dir)
    if not detections:
        print("No detections found. Writing empty scene_objects.json and exiting.")
        scene_dir.mkdir(parents=True, exist_ok=True)
        with open(scene_dir / "scene_objects.json", "w") as f:
            json.dump([], f)
        return

    if args.debug:
        _save_debug(debug_dir / "debug_step1_all_detections.json", [
            {"frame": d["frame_stem"], "label": d["label"],
             "score_2d": d["score_2d"], "score_3d": d["score_3d"]}
            for d in detections
        ])

    # --- Phase 2 ---
    if args.scale_mode == "align":
        print("\n[2/9] Skipping scale estimation (--align: COLMAP space assumed metric).")
        colmap_scale    = 1.0
        per_frame_scales = {}
    else:
        print("\n[2/9] Computing scene-level COLMAP scale...")
        colmap_scale, per_frame_scales = compute_scene_colmap_scale(
            manifest, frame_dir, sparse_dir
        )
    (scene_dir / "scene_colmap_scale.txt").write_text(str(colmap_scale))

    if args.debug:
        _save_debug(debug_dir / "debug_step2_colmap_scale.json", {
            "scene_colmap_scale":  colmap_scale,
"per_frame_scales":    per_frame_scales,
        })

    # --- Phase 3 ---
    print(f"\n[3/9] Transforming to COLMAP world space (scale mode: {args.scale_mode})...")
    if args.scale_mode == "global":
        print(f"  scale={colmap_scale:.5f} m/unit (applied to all frames)")
    elif args.scale_mode == "local":
        n_local = sum(1 for d in detections if d["frame_stem"] in per_frame_scales)
        n_fallback = len(detections) - n_local
        print(f"  per-frame scale available: {len(per_frame_scales)} frames  "
              f"({n_fallback} detection(s) fall back to global scale={colmap_scale:.5f})")
    else:  # align
        print("  scale=1.0 (COLMAP space assumed already metric)")
    all_extr = read_all_extrinsics(sparse_dir / "images.bin")
    transform_to_world(
        detections, all_extr, colmap_scale,
        per_frame_scales=per_frame_scales,
        scale_mode=args.scale_mode,
    )
    print(f"  {len(detections)} detections transformed")

    if args.debug:
        _save_debug(debug_dir / "debug_step3_world_centers.json", [
            {"frame": d["frame_stem"], "label": d["label"],
             "scale_used": round(d.get("_scale_used", colmap_scale), 5),
             "center_world": d["center_world"].tolist()}
            for d in detections
        ])

    # --- Phase 4 ---
    print("\n[4/9] Computing label embeddings...")
    unique_labels = sorted(set(d["label"] for d in detections))
    label_embs    = compute_label_embeddings(unique_labels, torch_device)

    if args.debug:
        emb_mat    = np.stack([label_embs[l] for l in unique_labels])
        cos_matrix = emb_mat @ emb_mat.T
        _save_debug(debug_dir / "debug_step4_embeddings.json", {
            lbl: [
                {"label": unique_labels[j], "cos_sim": round(float(cos_matrix[i, j]), 4)}
                for j in np.argsort(-cos_matrix[i])[:6] if j != i
            ][:5]
            for i, lbl in enumerate(unique_labels)
        })

    # --- Phase 5 ---
    print(f"\n[5/9] Semantic label grouping (threshold={args.sem_threshold})...")
    label_groups = semantic_label_groups(label_embs, args.sem_threshold)
    print(f"  {len(unique_labels)} unique labels → {len(label_groups)} group(s)")
    for i, g in enumerate(label_groups):
        print(f"    group {i}: {sorted(g)}")

    if args.debug:
        _save_debug(debug_dir / "debug_step5_label_groups.json", [
            {"group_id": i, "labels": sorted(g)}
            for i, g in enumerate(label_groups)
        ])

    # Build label → group_id lookup (used by PLY export and clustering)
    label_to_group: dict[str, int] = {}
    for gid, grp in enumerate(label_groups):
        for lbl in grp:
            label_to_group[lbl] = gid

    if args.debug:
        print("\n[5b/9] Saving per-semantic-group PLYs (debug)...")
        save_semantic_group_plys(
            label_groups, label_to_group, detections, sparse_dir, debug_dir
        )

    # --- Phase 6 ---
    if args.min_cluster_no is not None:
        min_size_global = max(1, args.min_cluster_no)
        threshold_desc  = f"min_size={min_size_global} (--min-cluster-no)"
    else:
        min_size_global = max(1, math.ceil(args.min_cluster_frac * n_frames_with_detections))
        threshold_desc  = (f"min_size={min_size_global} "
                           f"({args.min_cluster_frac*100:.0f}% × "
                           f"{n_frames_with_detections} frames with detections)")
    print(f"\n[6/9] Greedy IoU clustering  "
          f"iou_threshold={args.iou_threshold}  {threshold_desc}")

    # Partition detections by group
    group_det_indices: dict[int, list[int]] = defaultdict(list)
    for i, det in enumerate(detections):
        gid = label_to_group.get(det["label"], -1)
        if gid >= 0:
            group_det_indices[gid].append(i)

    all_clusters: list[list[int]] = []
    clustering_trace: list[dict]  = []
    n_total_raw = 0
    n_total_kept = 0

    for gid, indices in group_det_indices.items():
        clusters = greedy_iou_cluster(
            indices, detections,
            args.iou_threshold, args.score_w2d, args.score_w3d,
            iou_backend,
        )
        if args.debug:
            print(f"  [debug] group {gid} {sorted(label_groups[gid])}: "
                  f"{len(clusters)} cluster(s) before filter")
            for ci, c in enumerate(clusters):
                print(f"    cluster {ci}: {len(c)} boxes")
            clustering_trace.append({
                "group_id":  gid,
                "labels":    sorted(label_groups[gid]),
                "n_clusters": len(clusters),
                "clusters":  [{"cluster_id": ci, "n_boxes": len(c)}
                               for ci, c in enumerate(clusters)],
            })

        # Size filter: threshold relative to frames that had any OWLv2 detection
        kept      = [c for c in clusters if len(c) >= min_size_global]
        discarded = len(clusters) - len(kept)
        all_clusters.extend(kept)
        n_total_raw  += len(clusters)
        n_total_kept += len(kept)
        print(f"  Group {gid} ({sorted(label_groups[gid])}): "
              f"{len(indices)} dets → {len(clusters)} clusters "
              f"→ {len(kept)} kept, {discarded} discarded")

    print(f"  Total: {n_total_raw} clusters → {n_total_kept} kept")

    if args.debug:
        _save_debug(debug_dir / "debug_step6_clustering_trace.json", clustering_trace)

    kept_clusters = all_clusters

    if args.debug:
        _save_debug(debug_dir / "debug_step7_clusters_filtered.json", [
            {
                "cluster_idx": ci,
                "n_boxes":     len(c),
                "labels":      dict(
                    sorted(
                        {d["label"]: 0 for d in [detections[i] for i in c]}.items()
                    )
                ),
                "frames":      sorted({detections[i]["frame_stem"] for i in c}),
            }
            for ci, c in enumerate(kept_clusters)
        ])

    if not kept_clusters:
        print("No clusters survived size threshold. Writing empty scene_objects.json and exiting.")
        with open(scene_dir / "scene_objects.json", "w") as f:
            json.dump([], f)
        return

    # Sort by cluster size descending → object_id 0 = most-observed object
    kept_clusters.sort(key=len, reverse=True)

    # --- Phase 7 + 8 ---
    print(f"\n[7/8] Aggregating {len(kept_clusters)} cluster(s)...")
    objects      = []
    median_boxes = []
    score_boxes  = []
    loc_boxes    = []

    for obj_id, cluster in enumerate(kept_clusters):
        med, best_s, best_l = aggregate_cluster(
            cluster, detections, args.score_w2d, args.score_w3d
        )
        final_label, label_scores = confidence_weighted_label(
            cluster, detections, args.score_w2d, args.score_w3d
        )

        median_boxes.append(med)
        score_boxes.append(best_s)
        loc_boxes.append(best_l)

        objects.append({
            "object_id":             obj_id,
            "label":                 final_label,
            "label_scores":          label_scores,
            "n_boxes":               len(cluster),
            "frames_with_detection": sorted({detections[i]["frame_stem"] for i in cluster}),
            "median_box":            med.tolist(),
            "best_score_box":        best_s.tolist(),
            "best_location_box":     best_l.tolist(),
        })

        print(f"  obj {obj_id:>3}: '{final_label}'  "
              f"n={len(cluster)}  scores={label_scores}")

    # --- Save ---
    print(f"\n[8/8] Saving to {scene_dir}/")
    with open(scene_dir / "scene_objects.json", "w") as f:
        json.dump(objects, f, indent=2)

    np.save(scene_dir / "scene_median_boxes3d_world.npy",        np.stack(median_boxes))
    np.save(scene_dir / "scene_best_score_boxes3d_world.npy",    np.stack(score_boxes))
    np.save(scene_dir / "scene_best_location_boxes3d_world.npy", np.stack(loc_boxes))

    print(f"\nDone. {len(objects)} object(s) found in scene.")
    if args.debug:
        print(f"Debug  : {debug_dir}/")


if __name__ == "__main__":
    main()