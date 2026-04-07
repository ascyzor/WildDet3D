"""WildDet3D model utilities for VLM demo.

Thin wrapper around wilddet3d.build_model and wilddet3d.preprocess
to match the interface expected by the VLM notebook.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Ensure project root is in path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "third_party" / "sam3"))
sys.path.insert(0, str(project_root / "third_party" / "lingbot_depth"))

from wilddet3d import build_model, preprocess


def get_wilddet3d_model(
    checkpoint: str,
    sam3_checkpoint: str = "pretrained/sam3/sam3_detector.pt",
    score_threshold: float = 0.3,
    nms: bool = True,
    iou_threshold: float = 0.6,
    device: str = "cuda",
    canonical_rotation: bool = True,
    use_depth_input_test: bool = False,
    use_predicted_intrinsics: bool = True,
    skip_pretrained: bool = True,
):
    """Build WildDet3D model for inference.

    This is a convenience wrapper around wilddet3d.build_model
    with defaults tuned for the VLM demo.

    Returns:
        WildDet3DPredictor model
    """
    return build_model(
        checkpoint=checkpoint,
        sam3_checkpoint=sam3_checkpoint,
        score_threshold=score_threshold,
        nms=nms,
        iou_threshold=iou_threshold,
        device=device,
        canonical_rotation=canonical_rotation,
        use_depth_input_test=use_depth_input_test,
        use_predicted_intrinsics=use_predicted_intrinsics,
        skip_pretrained=skip_pretrained,
    )


def preprocess_wilddet3d(
    image: np.ndarray,
    intrinsics: Optional[np.ndarray] = None,
) -> dict:
    """Preprocess image for WildDet3D.

    This is a convenience wrapper around wilddet3d.preprocess.

    Args:
        image: RGB image as numpy array (H, W, 3), float32
        intrinsics: Camera intrinsics (3, 3), or None to use predicted

    Returns:
        Dict with preprocessed tensors and metadata
    """
    return preprocess(image, intrinsics)
