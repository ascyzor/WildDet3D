"""3D bbox visualization and video rendering.

Uses vis4d's boxes3d_to_corners for correct 3D box format handling.
Box format: [cx, cy, cz, w, l, h, qr, qi, qj, qk] in OPENCV axis mode.
"""

from pathlib import Path

import cv2
import numpy as np
import torch

from vis4d.data.const import AxisMode
from vis4d.op.box.box3d import boxes3d_to_corners


# Consistent color palette for track IDs
COLORS = [
    (255, 0, 0),     # red
    (0, 255, 0),     # green
    (0, 0, 255),     # blue
    (255, 255, 0),   # yellow
    (255, 0, 255),   # magenta
    (0, 255, 255),   # cyan
    (255, 128, 0),   # orange
    (128, 0, 255),   # purple
    (0, 128, 255),   # sky blue
    (255, 0, 128),   # pink
    (128, 255, 0),   # lime
    (0, 255, 128),   # spring green
    (128, 128, 255), # light blue
    (255, 128, 128), # light red
    (128, 255, 128), # light green
    (255, 255, 128), # light yellow
]


def get_track_color(track_id):
    """Get consistent BGR color for a track ID."""
    color = COLORS[track_id % len(COLORS)]
    return (color[2], color[1], color[0])


def project_corners_to_2d(corners_3d, intrinsics):
    """Project 3D corners to 2D pixel coordinates.

    Args:
        corners_3d: numpy array (8, 3) in camera space (OPENCV)
        intrinsics: numpy array (3, 3) camera matrix

    Returns:
        corners_2d: numpy array (8, 2) pixel coordinates
        valid: bool, True if all corners are in front of camera
    """
    if np.any(corners_3d[:, 2] <= 0):
        return None, False

    projected = (intrinsics @ corners_3d.T).T
    corners_2d = projected[:, :2] / projected[:, 2:3]
    return corners_2d, True


def draw_3d_bbox(image, corners_2d, color, track_id, category, score=None):
    """Draw a 3D bounding box wireframe on an image.

    Uses vis4d corner ordering:
               (back)
        (6) +---------+. (7)
            | ` .     |  ` .
            | (4) +---+-----+ (5)
            |     |   |     |
        (2) +-----+---+. (3)|
            ` .   |     ` . |
            (0) ` +---------+ (1)
                     (front)

    Front face: 0-1-5-4
    Back face: 2-3-7-6
    Side edges: 0-2, 1-3, 4-6, 5-7
    """
    pts = corners_2d.astype(int)
    thickness = 2

    # Front face (0-1-5-4)
    for i, j in [(0, 1), (1, 5), (5, 4), (4, 0)]:
        cv2.line(image, tuple(pts[i]), tuple(pts[j]), color, thickness)

    # Back face (2-3-7-6)
    for i, j in [(2, 3), (3, 7), (7, 6), (6, 2)]:
        cv2.line(image, tuple(pts[i]), tuple(pts[j]), color, max(1, thickness - 1))

    # Side edges
    for i, j in [(0, 2), (1, 3), (4, 6), (5, 7)]:
        cv2.line(image, tuple(pts[i]), tuple(pts[j]), color, max(1, thickness - 1))

    # Heading indicator (front face bottom center)
    center_bottom_front = (pts[0] + pts[1]) / 2
    center_bottom = (pts[0] + pts[1] + pts[2] + pts[3]) / 4
    cv2.line(
        image,
        tuple(center_bottom.astype(int)),
        tuple(center_bottom_front.astype(int)),
        color,
        thickness,
    )

    # Label
    label = f"#{track_id} {category}"
    if score is not None:
        label += f" {score:.2f}"

    label_x = int(pts[:, 0].min())
    label_y = int(pts[:, 1].min()) - 5

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(
        image,
        (label_x, label_y - th - 4),
        (label_x + tw + 4, label_y + 4),
        color,
        -1,
    )
    cv2.putText(
        image,
        label,
        (label_x + 2, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return image


def compute_corners(box_3d):
    """Convert a single box_3d (10,) to 8 corner points using vis4d.

    Args:
        box_3d: numpy array (10,) [cx, cy, cz, w, l, h, qr, qi, qj, qk]

    Returns:
        corners: numpy array (8, 3) or None if invalid
    """
    box_tensor = torch.from_numpy(box_3d).float().unsqueeze(0)
    corners = boxes3d_to_corners(box_tensor, axis_mode=AxisMode.OPENCV)
    return corners[0].numpy()


def render_frame(
    frame_rgb,
    frame_detections,
    intrinsics,
    categories,
    show_scores=True,
):
    """Render 3D bboxes on a single frame.

    Args:
        frame_rgb: RGB image (H, W, 3)
        frame_detections: dict mapping track_id -> box_3d (10,) or None
        intrinsics: numpy array (3, 3)
        categories: dict mapping track_id -> category string
        show_scores: whether to show score labels

    Returns:
        BGR image with 3D bbox overlays
    """
    image = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    for track_id, box_3d in frame_detections.items():
        if box_3d is None:
            continue

        corners_3d = compute_corners(box_3d)
        corners_2d, valid = project_corners_to_2d(corners_3d, intrinsics)

        if not valid:
            continue

        h, w = image.shape[:2]
        if (
            np.all(corners_2d[:, 0] < 0)
            or np.all(corners_2d[:, 0] > w)
            or np.all(corners_2d[:, 1] < 0)
            or np.all(corners_2d[:, 1] > h)
        ):
            continue

        color = get_track_color(track_id)
        category = categories.get(track_id, "object")

        image = draw_3d_bbox(image, corners_2d, color, track_id, category)

    return image


def render_video(
    frames,
    smoothed_tracks,
    intrinsics,
    categories,
    output_path,
    fps=30,
    raw_tracks=None,
    save_frames_dir=None,
):
    """Render output video with 3D bbox overlays.

    Args:
        frames: list of RGB numpy arrays (H, W, 3)
        smoothed_tracks: dict mapping track_id -> list[n_frames] of
            box_3d (10,) or None
        intrinsics: numpy array (3, 3)
        categories: dict mapping track_id -> category string
        output_path: path to save output video
        fps: output video FPS
        raw_tracks: optional, same format as smoothed_tracks, for
            side-by-side comparison
        save_frames_dir: if provided, save each frame as jpg to this dir
    """
    n_frames = len(frames)
    h, w = frames[0].shape[:2]

    if save_frames_dir is not None:
        save_frames_dir = Path(save_frames_dir)
        save_frames_dir.mkdir(parents=True, exist_ok=True)

    if raw_tracks is not None:
        out_w = w * 2
    else:
        out_w = w
    out_h = h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))

    for frame_idx in range(n_frames):
        frame_dets = {}
        for track_id, track in smoothed_tracks.items():
            if track[frame_idx] is not None:
                frame_dets[track_id] = track[frame_idx]

        rendered = render_frame(
            frames[frame_idx], frame_dets, intrinsics, categories
        )

        if raw_tracks is not None:
            raw_dets = {}
            for track_id, track in raw_tracks.items():
                if track[frame_idx] is not None:
                    raw_dets[track_id] = track[frame_idx]

            raw_rendered = render_frame(
                frames[frame_idx], raw_dets, intrinsics, categories
            )

            cv2.putText(
                raw_rendered, "Raw", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
            )
            cv2.putText(
                rendered, "Smoothed", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
            )

            combined = np.hstack([raw_rendered, rendered])
            if save_frames_dir is not None:
                cv2.imwrite(
                    str(save_frames_dir / f"frame_{frame_idx:04d}.jpg"),
                    combined,
                    [cv2.IMWRITE_JPEG_QUALITY, 80],
                )
            writer.write(combined)
        else:
            if save_frames_dir is not None:
                cv2.imwrite(
                    str(save_frames_dir / f"frame_{frame_idx:04d}.jpg"),
                    rendered,
                    [cv2.IMWRITE_JPEG_QUALITY, 80],
                )
            writer.write(rendered)

    writer.release()
    print(f"Video saved to: {output_path}")
