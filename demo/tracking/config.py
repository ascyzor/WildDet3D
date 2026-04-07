"""Configuration for Zero-Shot 3D Tracking Pipeline."""

from pathlib import Path

# ============== Paths ==============
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Output directory
OUTPUT_DIR = Path(__file__).parent / "output"

# ============== Model ==============
# Default checkpoint (downloaded via HuggingFace)
HF_MODEL_REPO = "allenai/WildDet3D"
HF_CKPT_NAME = "wilddet3d_alldata_all_prompt_v1.0.pt"

# Local checkpoint paths (tried in order)
LOCAL_CHECKPOINTS = [
    "ckpt/wilddet3d.pt",
    str(PROJECT_ROOT / "ckpt" / "wilddet3d.pt"),
]

SAM3_CHECKPOINT = str(PROJECT_ROOT / "pretrained" / "sam3" / "sam3_detector.pt")

# ============== Inference ==============
SCORE_THRESHOLD = 0.15
NMS_IOU_THRESHOLD = 0.6
FPS = 30

# ============== Kalman Filter ==============
KF_PROCESS_NOISE_POS = 0.5
KF_PROCESS_NOISE_DIM = 0.1
KF_PROCESS_NOISE_VEL = 1.0
KF_MEASUREMENT_NOISE_POS = 1.0
KF_MEASUREMENT_NOISE_DIM = 0.5
ROTATION_SMOOTH_ALPHA = 0.5

# Minimum mask area (pixels) to consider valid
MIN_MASK_AREA = 100
