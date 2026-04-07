"""Zero-Shot 3D Object Tracking Pipeline.

Tracks objects in video with 3D bounding boxes using WildDet3D.

Input:
    - Video file (mp4)
    - Object masks as RLE JSON (per-frame, from SAM2 or any tracker)
    - Category labels per object (from VLM or manual)
    - Camera intrinsics (3x3 matrix)

Output:
    - Tracked video with 3D bounding box overlays
    - Results JSON with per-track 3D boxes

Usage:
    python -m demo.tracking.run_pipeline \
        --video video.mp4 \
        --masks masks.json \
        --categories categories.json \
        --intrinsics intrinsics.json
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from . import config
from .inference import load_model, run_inference_single_frame
from .kalman_filter import smooth_tracks
from .visualization import render_video


def normalize_rotation_yaw(box_3d):
    """Normalize quaternion yaw to [0, pi) range.

    Removes 180-degree ambiguity for symmetric objects in OPENCV axis mode.

    Args:
        box_3d: numpy array (10,) [cx,cy,cz,w,l,h,qr,qi,qj,qk]

    Returns:
        box_3d with normalized rotation (copy)
    """
    from scipy.spatial.transform import Rotation

    box = box_3d.copy()
    qr, qi, qj, qk = box[6], box[7], box[8], box[9]
    rot = Rotation.from_quat([qi, qj, qk, qr])

    yaw, pitch, roll = rot.as_euler("YXZ")

    yaw = yaw % (2 * np.pi)
    if yaw >= np.pi:
        yaw -= np.pi

    rot_norm = Rotation.from_euler("YXZ", [yaw, pitch, roll])
    qx, qy, qz, qw = rot_norm.as_quat()
    box[6] = qw
    box[7] = qx
    box[8] = qy
    box[9] = qz
    return box


def fix_90deg_rotation_flips(tracks):
    """Fix 90-degree rotation flips for near-square objects (w ~ l).

    For objects where width/length ratio > 0.7, yaw can flip by 90 degrees
    between frames since both orientations look identical. This picks the
    yaw closest to the previous frame for temporal consistency.

    Args:
        tracks: dict mapping track_id -> list[n_frames] of box_3d or None

    Returns:
        New dict with rotation-consistent tracks.
    """
    from scipy.spatial.transform import Rotation

    WL_RATIO_THRESHOLD = 0.7

    fixed = {}
    for tid, track in tracks.items():
        wl_ratios = []
        for box in track:
            if box is not None:
                w, l = box[3], box[4]
                if max(w, l) > 0:
                    wl_ratios.append(min(w, l) / max(w, l))
        if not wl_ratios:
            fixed[tid] = track
            continue

        if np.median(wl_ratios) <= WL_RATIO_THRESHOLD:
            fixed[tid] = track
            continue

        symmetry_period = np.pi / 2
        new_track = []
        prev_yaw = None

        for box in track:
            if box is None:
                new_track.append(None)
                continue

            box = box.copy()
            qr, qi, qj, qk = box[6], box[7], box[8], box[9]
            rot = Rotation.from_quat([qi, qj, qk, qr])
            yaw, pitch, roll = rot.as_euler("YXZ")

            if prev_yaw is not None:
                diff = yaw - prev_yaw
                diff = (diff + symmetry_period / 2) % symmetry_period
                diff -= symmetry_period / 2
                yaw = prev_yaw + diff

                rot_new = Rotation.from_euler("YXZ", [yaw, pitch, roll])
                qx, qy, qz, qw = rot_new.as_quat()
                box[6] = qw
                box[7] = qx
                box[8] = qy
                box[9] = qz

            prev_yaw = yaw
            new_track.append(box)

        fixed[tid] = new_track

    return fixed


def build_raw_tracks(tracked_results, n_frames):
    """Convert per-frame detection lists to per-track arrays.

    Args:
        tracked_results: list[n_frames] of list[detections]
        n_frames: total number of frames

    Returns:
        Dict mapping track_id -> list[n_frames] of box_3d (10,) or None
    """
    raw_tracks = {}
    for frame_idx, frame_dets in enumerate(tracked_results):
        for det in frame_dets:
            tid = det["track_id"]
            if tid not in raw_tracks:
                raw_tracks[tid] = [None] * n_frames
            raw_tracks[tid][frame_idx] = det["box_3d"]
    return raw_tracks


def load_video_frames(video_path):
    """Load all frames from a video file.

    Args:
        video_path: Path to video file (mp4, avi, etc.)

    Returns:
        List of RGB numpy arrays (H, W, 3), uint8
    """
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    print(f"Loaded {len(frames)} frames from {video_path}")
    return frames


def load_masks_json(masks_path):
    """Load per-frame object masks from JSON.

    Expected format: list of n_frames elements, each a list of n_objects
    RLE dicts (or null for invisible objects).

    Example:
        [
            [{"counts": "...", "size": [H, W]}, null, ...],  // frame 0
            [null, {"counts": "...", "size": [H, W]}, ...],  // frame 1
            ...
        ]

    Args:
        masks_path: Path to masks JSON file.

    Returns:
        Dict mapping obj_id (int) -> list[n_frames] of RLE dict or None.
    """
    with open(masks_path) as f:
        all_frames = json.load(f)

    n_frames = len(all_frames)
    n_objects = len(all_frames[0])

    trajectories = {}
    for obj_id in range(n_objects):
        obj_masks = []
        for frame_idx in range(n_frames):
            rle = all_frames[frame_idx][obj_id]
            obj_masks.append(rle)
        trajectories[obj_id] = obj_masks

    print(f"Loaded {n_objects} object trajectories across {n_frames} frames")
    return trajectories


def load_categories_json(categories_path):
    """Load object category labels from JSON.

    Expected format:
        {"0": "car", "1": "person", "3": "bicycle", ...}

    Only objects present in this file are tracked. Objects not listed
    (or with null value) are skipped.

    Args:
        categories_path: Path to categories JSON file.

    Returns:
        Dict mapping obj_id (int) -> category string.
    """
    with open(categories_path) as f:
        data = json.load(f)

    categories = {}
    for obj_id_str, cat in data.items():
        if cat is not None:
            categories[int(obj_id_str)] = cat

    print(f"Loaded {len(categories)} object categories")
    return categories


def load_intrinsics(intrinsics_input):
    """Load camera intrinsics from JSON or numpy file.

    Supports:
        - JSON file: {"K": [[fx,0,cx],[0,fy,cy],[0,0,1]]}
        - NPY file: (3, 3) numpy array
        - "fx,fy,cx,cy" string: construct K matrix

    Args:
        intrinsics_input: Path or "fx,fy,cx,cy" string.

    Returns:
        numpy array (3, 3)
    """
    if "," in intrinsics_input:
        parts = [float(x) for x in intrinsics_input.split(",")]
        fx, fy, cx, cy = parts
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    path = Path(intrinsics_input)
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)
    elif path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        return np.array(data["K"], dtype=np.float32)
    else:
        return np.load(path).astype(np.float32)


def run_pipeline(args):
    """Run the full zero-shot 3D tracking pipeline."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_name = Path(args.video).stem
    print(f"=== Zero-Shot 3D Tracking: {video_name} ===")

    # Step 1: Load data
    print("\n[1/5] Loading data...")
    frames = load_video_frames(args.video)
    n_frames = len(frames)

    mask_trajectories = load_masks_json(args.masks)
    categories = load_categories_json(args.categories)
    intrinsics = load_intrinsics(args.intrinsics)

    print(f"  Frames: {n_frames}")
    print(f"  Tracked objects: {len(mask_trajectories)}")
    print(f"  Categories: {len(categories)}")
    for oid, cat in categories.items():
        print(f"    obj_{oid}: {cat}")

    # Step 2: Load model
    print("\n[2/5] Loading WildDet3D model...")
    model = load_model(
        checkpoint=args.checkpoint,
        device=args.device,
    )

    # Step 3: Per-frame inference
    print(f"\n[3/5] Running inference on {n_frames} frames...")
    tracked_results = []
    total_dets = 0

    for frame_idx in range(n_frames):
        frame_masks = {}
        for obj_id, masks in mask_trajectories.items():
            if obj_id in categories and masks[frame_idx] is not None:
                frame_masks[obj_id] = masks[frame_idx]

        t0 = time.time()
        frame_dets = run_inference_single_frame(
            model=model,
            frame_rgb=frames[frame_idx],
            intrinsics=intrinsics,
            obj_masks=frame_masks,
            categories=categories,
            device=args.device,
        )
        dt = time.time() - t0

        tracked_results.append(frame_dets)
        total_dets += len(frame_dets)

        if (frame_idx + 1) % 10 == 0 or frame_idx == 0:
            print(
                f"  Frame {frame_idx + 1}/{n_frames}: "
                f"{len(frame_dets)} detections, {dt:.2f}s"
            )

    print(f"  Total detections: {total_dets}")

    # Step 4: Post-processing (rotation normalization + Kalman smoothing)
    print("\n[4/5] Post-processing tracks...")

    # Normalize rotations
    for frame_dets in tracked_results:
        for det in frame_dets:
            det["box_3d"] = normalize_rotation_yaw(det["box_3d"])

    # Fix 90-degree flips
    temp_tracks = build_raw_tracks(tracked_results, n_frames)
    temp_tracks = fix_90deg_rotation_flips(temp_tracks)
    for frame_idx, frame_dets in enumerate(tracked_results):
        for det in frame_dets:
            tid = det["track_id"]
            if tid in temp_tracks and temp_tracks[tid][frame_idx] is not None:
                det["box_3d"][6:10] = temp_tracks[tid][frame_idx][6:10]

    # Kalman filter smoothing
    raw_tracks = build_raw_tracks(tracked_results, n_frames)
    smoothed_tracks = smooth_tracks(tracked_results, n_frames)

    print(f"  Smoothed {len(smoothed_tracks)} tracks")
    for tid, track in smoothed_tracks.items():
        visible = sum(1 for t in track if t is not None)
        print(
            f"    Track #{tid} ({categories.get(tid, '?')}): "
            f"{visible}/{n_frames} frames"
        )

    # Step 5: Render video
    print("\n[5/5] Rendering output video...")
    if args.side_by_side:
        output_path = output_dir / f"{video_name}_tracked_comparison.mp4"
        render_video(
            frames, smoothed_tracks, intrinsics, categories,
            output_path, fps=config.FPS, raw_tracks=raw_tracks,
        )
    else:
        output_path = output_dir / f"{video_name}_tracked.mp4"
        render_video(
            frames, smoothed_tracks, intrinsics, categories,
            output_path, fps=config.FPS,
        )

    # Save results JSON
    results_path = output_dir / f"{video_name}_results.json"
    results = {
        "video_name": video_name,
        "n_frames": n_frames,
        "n_tracks": len(smoothed_tracks),
        "categories": {str(k): v for k, v in categories.items()},
        "tracks": {},
    }
    for tid, track in smoothed_tracks.items():
        results["tracks"][str(tid)] = {
            "category": categories.get(tid, "unknown"),
            "visible_frames": sum(1 for t in track if t is not None),
            "boxes_3d": [
                t.tolist() if t is not None else None for t in track
            ],
        }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Done! ===")
    print(f"Output video: {output_path}")
    print(f"Results JSON: {results_path}")

    return output_path, results_path


def main():
    parser = argparse.ArgumentParser(
        description="Zero-Shot 3D Object Tracking with WildDet3D"
    )
    parser.add_argument(
        "--video", type=str, required=True,
        help="Path to input video (mp4)",
    )
    parser.add_argument(
        "--masks", type=str, required=True,
        help="Path to object masks JSON (per-frame RLE from SAM2/tracker)",
    )
    parser.add_argument(
        "--categories", type=str, required=True,
        help="Path to category labels JSON ({obj_id: category_name})",
    )
    parser.add_argument(
        "--intrinsics", type=str, required=True,
        help="Camera intrinsics: JSON file, npy file, or 'fx,fy,cx,cy'",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Model checkpoint (auto-downloaded if not provided)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=str(config.OUTPUT_DIR),
        help="Output directory",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device (cuda or cpu)",
    )
    parser.add_argument(
        "--side_by_side", action="store_true",
        help="Render side-by-side comparison (raw vs smoothed)",
    )

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
