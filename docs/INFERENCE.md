# WildDet3D Inference Guide

## Overview

WildDet3D supports 5 prompt modes for 3D object detection:

| Mode | Prompt Input | Behavior | Use Case |
|---|---|---|---|
| **Text** | `input_texts=["car", "person"]` | Detect all instances of given categories | Open-vocabulary detection |
| **Visual** | `input_boxes` + `prompt_text="visual"` | Use box as visual example, find similar objects | One-to-many matching |
| **Visual+Label** | `input_boxes` + `prompt_text="visual: car"` | Visual example with category constraint | Filtered one-to-many |
| **Geometric** | `input_boxes` + `prompt_text="geometric"` | Lift the given 2D box to 3D | One-to-one, box prompt |
| **Geometric+Label** | `input_boxes` + `prompt_text="geometric: car"` | Lift 2D box to 3D with category label | One-to-one with label |

Point prompts (`input_points`) work the same way as box prompts with all prompt_text modes.

## Setup

```python
from wilddet3d import build_model, preprocess
import numpy as np
from PIL import Image

# Build model
model = build_model(
    checkpoint="ckpt/wilddet3d.pt",
    score_threshold=0.3,
    skip_pretrained=True,  # faster loading from full checkpoint
)

# Load and preprocess image
image = np.array(Image.open("image.jpg")).astype(np.float32)
data = preprocess(image, intrinsics=None)  # None = use predicted intrinsics
```

## `build_model()` Parameters

| Parameter | Default | Description |
|---|---|---|
| `checkpoint` | (required) | Path to WildDet3D `.ckpt` file |
| `sam3_checkpoint` | `"pretrained/sam3/sam3_detector.pt"` | Path to SAM3 pretrained weights |
| `score_threshold` | `0.3` | Confidence threshold for filtering detections |
| `nms` | `True` | Whether to apply NMS |
| `iou_threshold` | `0.6` | IoU threshold for NMS |
| `device` | `"cuda"` | Device to load model on |
| `skip_pretrained` | `False` | Skip loading SAM3/LingBot pretrained weights (faster if checkpoint has everything) |
| `use_predicted_intrinsics` | `False` | Use predicted intrinsics for 3D decoding (for in-the-wild images without GT K) |
| `canonical_rotation` | `False` | Use canonical rotation representation |

## `preprocess()` Parameters

| Parameter | Default | Description |
|---|---|---|
| `image` | (required) | RGB image as numpy array `(H, W, 3)`, dtype `float32` |
| `intrinsics` | `None` | Camera intrinsics `(3, 3)`. If `None`, a default is created and predicted intrinsics are used at inference. |

Returns a dict with keys: `images`, `intrinsics`, `input_hw`, `original_hw`, `padding`, `original_images`, `original_intrinsics`.

## Prompt Modes

### 1. Text Prompt

Detect all instances of given text categories.

```python
results = model(
    images=data["images"].cuda(),
    intrinsics=data["intrinsics"].cuda()[None],
    input_hw=[data["input_hw"]],
    original_hw=[data["original_hw"]],
    padding=[data["padding"]],
    input_texts=["car", "person", "bicycle"],
)
boxes, boxes3d, scores, scores_2d, scores_3d, class_ids, depth_maps = results
```

### 2. Visual Prompt (Box, one-to-many)

Use a 2D box as a visual example to find all similar objects in the image.

```python
results = model(
    images=data["images"].cuda(),
    intrinsics=data["intrinsics"].cuda()[None],
    input_hw=[data["input_hw"]],
    original_hw=[data["original_hw"]],
    padding=[data["padding"]],
    input_boxes=[[100, 200, 300, 400]],  # pixel xyxy
    prompt_text="visual",
)
boxes, boxes3d, scores, scores_2d, scores_3d, class_ids, depth_maps = results
```

### 3. Visual+Label Prompt (Box + category, one-to-many)

Same as visual but with a category constraint.

```python
results = model(
    ...,
    input_boxes=[[100, 200, 300, 400]],
    prompt_text="visual: car",
)
```

### 4. Geometric Prompt (Box, one-to-one)

Lift a specific 2D bounding box to a 3D bounding box.

```python
results = model(
    ...,
    input_boxes=[[100, 200, 300, 400]],
    prompt_text="geometric",
)
```

### 5. Geometric+Label Prompt (Box + category, one-to-one)

```python
results = model(
    ...,
    input_boxes=[[100, 200, 300, 400]],
    prompt_text="geometric: car",
)
```

### Point Prompts

Point prompts work with any `prompt_text` mode. Each point is `(x, y, label)` where label is 1 (positive) or 0 (negative).

```python
results = model(
    ...,
    input_points=[[(150, 250, 1), (200, 300, 0)]],  # pixel coords
    prompt_text="geometric",
)
```

## Output Format

All outputs are **per-image lists**:

| Output | Shape | Description |
|---|---|---|
| `boxes` | `list[Tensor[N, 4]]` | 2D bounding boxes in **pixel xyxy** (original image space) |
| `boxes3d` | `list[Tensor[N, 10]]` | 3D bounding boxes (center_x, center_y, center_z, w, h, l, rot_6d) |
| `scores` | `list[Tensor[N]]` | Combined confidence scores |
| `scores_2d` | `list[Tensor[N]]` | 2D detection confidence |
| `scores_3d` | `list[Tensor[N]]` | 3D detection confidence |
| `class_ids` | `list[Tensor[N]]` | Class indices (into `input_texts` or prompt order) |
| `depth_maps` | `list[Tensor[1, H, W]]` | Predicted metric depth maps (meters) |

### Getting Predicted Intrinsics

```python
results = model(
    ...,
    input_texts=["car"],
    return_predicted_intrinsics=True,
)
boxes, boxes3d, scores, scores_2d, scores_3d, class_ids, depth_maps, predicted_K = results
# predicted_K: Tensor[B, 3, 3] predicted camera intrinsics
```

## Visualization

Draw 3D bounding boxes with 2D/3D scores on the original image:

```python
from wilddet3d.vis.visualize import draw_3d_boxes

# After running inference
boxes, boxes3d, scores, scores_2d, scores_3d, class_ids, depth_maps = results

draw_3d_boxes(
    image=image.astype(np.uint8),     # original RGB image (H, W, 3)
    boxes3d=boxes3d[0],               # 3D boxes for first image
    intrinsics=intrinsics,            # camera intrinsics (3, 3)
    scores_2d=scores_2d[0],           # 2D confidence
    scores_3d=scores_3d[0],           # 3D confidence
    class_ids=class_ids[0],           # class indices
    class_names=["car", "person"],    # category names
    line_width=2,                     # 3D box edge width
    font_size=13,                     # label font size
    save_path="output.png",           # save to file
)
```

## Batch Inference

```python
# Preprocess multiple images
data_list = [preprocess(img) for img in images]

# Stack into batch
import torch
batch_images = torch.stack([d["images"] for d in data_list]).cuda()
batch_intrinsics = torch.stack([d["intrinsics"] for d in data_list]).cuda()

results = model(
    images=batch_images,
    intrinsics=batch_intrinsics,
    input_hw=[d["input_hw"] for d in data_list],
    original_hw=[d["original_hw"] for d in data_list],
    padding=[d["padding"] for d in data_list],
    input_texts=["car", "person"],
)
```
