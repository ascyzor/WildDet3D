# VLM-Driven 3D Object Detection Demo

Use a Vision-Language Model (VLM) to drive WildDet3D for 3D object detection from a single RGB image.

<p align="center">
  <img src="../../assets/demo_vlm.png" width="600">
</p>

Given a natural language query (e.g., "Detect all the sheep in this image"),
the VLM analyzes the image and outputs spatial prompts (bounding boxes or points)
that guide WildDet3D to produce 3D bounding boxes.

Two VLM modes are supported:

| Mode | VLM | Prompt type |
|------|-----|-------------|
| `box` | Qwen3-VL-8B | 2D bounding boxes via tool calling |
| `point` | Molmo2-8B | 2D points via native pointing |

## Setup

### 1. Install vLLM environment

The VLM server runs in a **separate conda environment**:

```bash
conda create -n vllm python=3.11 -y
conda activate vllm
pip install -r requirements.txt
```

### 2. Run the notebook

```bash
conda activate wilddet3d
cd demo/vlm
jupyter notebook vlm_3d_detection_demo.ipynb
```
