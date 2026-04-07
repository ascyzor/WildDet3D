"""Preprocessing utilities for WildDet3D inference.

Handles image resizing, normalization, center padding, and intrinsics
adjustment to prepare raw inputs for the WildDet3D model.
"""

from typing import Optional

import numpy as np

from vis4d.data.transforms.base import compose
from vis4d.data.transforms.normalize import NormalizeImages
from vis4d.data.transforms.resize import ResizeImages, ResizeIntrinsics
from vis4d.data.transforms.to_tensor import ToTensor

from wilddet3d.data.transforms.pad import (
    CenterPadImages,
    CenterPadIntrinsics,
)
from wilddet3d.data.transforms.resize import GenResizeParameters

# WildDet3D expects 1008x1008 images
IMAGE_SIZE = (1008, 1008)


def preprocess(
    image: np.ndarray,
    intrinsics: Optional[np.ndarray] = None,
) -> dict:
    """Preprocess image for WildDet3D.

    Args:
        image: RGB image as numpy array (H, W, 3)
        intrinsics: Camera intrinsics (3, 3), or None to use default/predicted

    Returns:
        Dict with preprocessed tensors and metadata
    """
    images = image.astype(np.float32)[None, ...]
    H, W = images.shape[1], images.shape[2]

    # If no intrinsics provided, create a placeholder.
    # When use_predicted_intrinsics=True in the model, the geometry backend's
    # K_pred will be used for 3D box decoding instead of this placeholder.
    # The placeholder is still needed so the data pipeline doesn't crash.
    if intrinsics is None:
        focal = max(H, W)
        intrinsics = np.array(
            [
                [focal, 0, W / 2],
                [0, focal, H / 2],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

    data_dict = {
        "images": images,
        "original_images": images.copy(),
        "input_hw": (H, W),
        "original_hw": (H, W),
        "intrinsics": intrinsics.astype(np.float32),
        "original_intrinsics": intrinsics.astype(np.float32).copy(),
    }

    preprocess_transforms = compose(
        transforms=[
            GenResizeParameters(shape=IMAGE_SIZE),
            ResizeImages(),
            ResizeIntrinsics(),
            NormalizeImages(),
            CenterPadImages(
                stride=1, shape=IMAGE_SIZE, update_input_hw=True
            ),
            CenterPadIntrinsics(),
        ]
    )

    data = preprocess_transforms([data_dict])[0]
    to_tensor = ToTensor()
    data = to_tensor([data])[0]

    return data
