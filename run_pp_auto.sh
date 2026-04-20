#!/usr/bin/env bash
# WildDet3D auto-detection pipeline (OWLv2 → WildDet3D)
#
# Runs the three steps in order:
#   1. extract_intrinsics.py      — reads COLMAP cameras.bin, saves K as .npy
#   2. wildDet3D_infer_auto.py    — OWLv2 auto-detects 2D boxes, WildDet3D
#                                   lifts them to 3D, saves outputs under
#                                   3Dresults/auto_owlv2/
#   3. visualize_boxes3d_auto.py  — merges COLMAP point cloud + 3D boxes
#                                   into a .ply file
#
# Usage
# -----
#   ./run_pp_auto.sh <dataset> <scene> <image> [infer_args...] [-- viz_args...]
#
# Arguments
# ---------
#   dataset       Dataset name, e.g. "mipnerf360"
#   scene         Scene name,   e.g. "room"
#   image         Image name exactly as stored in COLMAP images.bin.
#                 Include sub-directory if present:
#                   e.g. "DSCF4702.JPG"  or  "pano_camera1/pano_00051.jpg"
#
# Optional inference args (passed to wildDet3D_infer_auto.py):
#   --mode visual|geometric       WildDet3D prompt mode (default: visual)
#   --owl-confidence FLOAT        OWLv2 min confidence   (default: 0.2)
#   --nms-iou FLOAT               OWLv2 NMS IoU threshold (default: 0.5)
#   --score-threshold FLOAT       WildDet3D output filter (default: 0.3)
#
# Optional viz args (placed after a "--" separator):
#   --clip-radius FLOAT           Discard distant COLMAP points
#   --box-color R,G,B             Box edge colour (default: 255,0,0)
#   --exp-scene-name NAME         Also produce a gsplat-annotated .ply
#   (and all other visualize_boxes3d_auto.py options)
#
# Examples
# --------
#   ./run_pp_auto.sh mipnerf360 room DSCF4702.JPG
#   ./run_pp_auto.sh mipnerf360 room DSCF4702.JPG --mode geometric --owl-confidence 0.3
#   ./run_pp_auto.sh mipnerf360 room DSCF4702.JPG -- --clip-radius 15
#   ./run_pp_auto.sh mipnerf360 room DSCF4702.JPG --score-threshold 0.4 -- --exp-scene-name room_default

set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <dataset> <scene> <image> [infer_args...] [-- viz_args...]" >&2
    exit 1
fi

DATASET=$1
SCENE=$2
IMAGE=$3
shift 3

# Split remaining args into infer_args (before "--") and viz_args (after "--")
INFER_ARGS=()
VIZ_ARGS=()
SEPARATOR_FOUND=0
for arg in "$@"; do
    if [ "$arg" = "--" ]; then
        SEPARATOR_FOUND=1
    elif [ "$SEPARATOR_FOUND" -eq 0 ]; then
        INFER_ARGS+=("$arg")
    else
        VIZ_ARGS+=("$arg")
    fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================"
echo " WildDet3D auto-detection pipeline"
echo "  dataset : $DATASET"
echo "  scene   : $SCENE"
echo "  image   : $IMAGE"
echo "  mode    : auto (OWLv2 → WildDet3D)"
echo "========================================"

echo ""
echo "[1/3] Extracting camera intrinsics..."
python "$SCRIPT_DIR/extract_intrinsics.py" "$DATASET" "$SCENE" "$IMAGE"

echo ""
echo "[2/3] Running OWLv2 + WildDet3D inference..."
python "$SCRIPT_DIR/wildDet3D_infer_auto.py" "$DATASET" "$SCENE" "$IMAGE" "${INFER_ARGS[@]+"${INFER_ARGS[@]}"}"

echo ""
echo "[3/3] Generating PLY..."
python "$SCRIPT_DIR/visualize_boxes3d_auto.py" "$DATASET" "$SCENE" "$IMAGE" "${VIZ_ARGS[@]+"${VIZ_ARGS[@]}"}"

echo ""
echo "Done. Outputs in: outputs/$DATASET/$SCENE/3Dresults/auto_owlv2/"