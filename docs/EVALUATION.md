# WildDet3D Evaluation Guide

Evaluation uses the [vis4d](https://github.com/SysCV/vis4d) framework. Follow the [vis4d documentation](https://vis4d.readthedocs.io/) for general setup.

## Quick Start

```bash
# General format
vis4d test --config configs/eval/<benchmark>/<mode>.py \
    --gpus 1 --ckpt ckpt/wilddet3d.pt
```

## Metrics

| Metric | Description |
|---|---|
| **AP** | Average Precision (COCO-style, 3D IoU thresholds) |

## Benchmarks

### Omni3D

Training benchmark. Evaluates on KITTI, nuScenes, SUNRGBD, Hypersim, ARKitScenes, Objectron.

| Mode | Config | Description |
|---|---|---|
| Text | `configs/eval/omni3d/text.py` | Text prompt, monocular depth |
| Text + Depth | `configs/eval/omni3d/text_with_depth.py` | Text prompt, with GT depth |
| Box Prompt | `configs/eval/omni3d/box_prompt.py` | GT 2D box as prompt |
| Box Prompt + Depth | `configs/eval/omni3d/box_prompt_with_depth.py` | GT 2D box + GT depth |
| 2D Only | `configs/eval/omni3d/eval_2d.py` | 2D detection metrics only |

```bash
vis4d test --config configs/eval/omni3d/text.py --gpus 1 --ckpt ckpt/wilddet3d.pt
```

### ScanNet (Zero-Shot)

Indoor 3D scene understanding. Zero-shot evaluation (not in training set).

| Mode | Config | Description |
|---|---|---|
| Text | `configs/eval/scannet/text.py` | Text prompt, monocular depth |
| Text + Depth | `configs/eval/scannet/text_with_depth.py` | Text prompt, with GT depth |
| Box Prompt | `configs/eval/scannet/box_prompt.py` | GT 2D box as prompt |
| Box Prompt + Depth | `configs/eval/scannet/box_prompt_with_depth.py` | GT 2D box + GT depth |

```bash
vis4d test --config configs/eval/scannet/text.py --gpus 1 --ckpt ckpt/wilddet3d.pt
```

### Argoverse2 (Zero-Shot)

Outdoor autonomous driving. Zero-shot evaluation.

| Mode | Config | Description |
|---|---|---|
| Text | `configs/eval/argoverse/text.py` | Text prompt, monocular depth |
| Text + Depth | `configs/eval/argoverse/text_with_depth.py` | Text prompt, with GT depth |
| Box Prompt | `configs/eval/argoverse/box_prompt.py` | GT 2D box as prompt |
| Box Prompt + Depth | `configs/eval/argoverse/box_prompt_with_depth.py` | GT 2D box + GT depth |

```bash
vis4d test --config configs/eval/argoverse/text.py --gpus 1 --ckpt ckpt/wilddet3d.pt
```

### Stereo4D (Zero-Shot)

Dynamic 3D object detection from stereo video.

| Mode | Config | Description |
|---|---|---|
| Text | `configs/eval/stereo4d/text.py` | Text prompt, monocular depth |
| Text + Depth | `configs/eval/stereo4d/text_with_depth.py` | Text prompt, with GT depth |
| Box Prompt | `configs/eval/stereo4d/box_prompt.py` | GT 2D box as prompt |
| Box Prompt + Depth | `configs/eval/stereo4d/box_prompt_with_depth.py` | GT 2D box + GT depth |

```bash
vis4d test --config configs/eval/stereo4d/text.py --gpus 1 --ckpt ckpt/wilddet3d.pt
```

### In-the-Wild

Large-scale in-the-wild evaluation with diverse categories.

| Mode | Config | Description |
|---|---|---|
| Text | `configs/eval/in_the_wild/text.py` | Text prompt, monocular depth |
| Text + Depth | `configs/eval/in_the_wild/text_with_depth.py` | Text prompt, with GT depth |
| Box Prompt | `configs/eval/in_the_wild/box_prompt.py` | GT 2D box as prompt |
| Box Prompt + Depth | `configs/eval/in_the_wild/box_prompt_with_depth.py` | GT 2D box + GT depth |

```bash
vis4d test --config configs/eval/in_the_wild/text.py --gpus 1 --ckpt ckpt/wilddet3d.pt
```

## Config Naming Convention

```
configs/eval/<benchmark>/<prompt>[_with_depth].py
```

| Component | Options | Meaning |
|---|---|---|
| `<benchmark>` | `omni3d`, `scannet`, `argoverse`, `stereo4d`, `in_the_wild` | Evaluation dataset |
| `<prompt>` | `text`, `box_prompt` | Text prompt vs. GT 2D box as prompt |
| `with_depth` | present or absent | Whether GT depth is provided to the model |

## Data Setup

Evaluation datasets should follow the [vis4d data format](https://vis4d.readthedocs.io/). Default data paths:

```
data/
├── omni3d/                    # Omni3D (KITTI, nuScenes, SUNRGBD, etc.)
├── scannet/                   # ScanNet val
├── argoverse/                 # Argoverse2
├── stereo4d/                  # Stereo4D
└── in_the_wild/               # WildDet3D in-the-wild annotations
```

<!-- TODO: Add data preparation scripts and download links -->
