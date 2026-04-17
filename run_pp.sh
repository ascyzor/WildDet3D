#!/usr/bin/env bash
# WildDet3D inference pipeline
#
# Runs the three steps in order:
#   1. extract_intrinsics.py  — reads COLMAP cameras.bin and saves K as .npy
#   2. wildDet3D_infer.py     — runs WildDet3D, saves boxes3d / depth_map /
#                               class_ids / output.png
#   3. visualize_boxes3d.py   — merges COLMAP point cloud + 3D boxes into .ply
#                               and, optionally, into a gsplat .ply
#
# Usage
# -----
#   ./run_pp.sh <dataset> <scene> <image> <prompt> [viz_args...]
#
# Arguments
# ---------
#   dataset     Dataset name, e.g. "mipnerf360"
#   scene       Scene name,   e.g. "room"
#   image       Image name exactly as stored in COLMAP images.bin.
#               Include the sub-directory if present:
#                 e.g. "DSCF4702.JPG"  or  "pano_camera1/pano_00051.jpg"
#   prompt      Comma-separated detection categories, e.g. "chair,table"
#   viz_args    Optional extra arguments forwarded to visualize_boxes3d.py,
#               e.g. "--clip-radius 15 --box-color 0,255,0"
#               Pass --exp-scene-name <name> to also produce a gsplat-annotated .ply.
#
# Data layout expected
# --------------------
#   ~/data/<dataset>/<scene>/
#       images/     original images (may include sub-directories)
#       sparse/     COLMAP sparse reconstruction (cameras.bin, images.bin, ...)
#
# Outputs  (all written to <project>/outputs/<dataset>/<scene>/)
# -------
#   cam_info/
#       <camera_id>_intrinsics.npy
#   3Dresults/<prompt>/         prompt commas replaced by underscores
#       <stem>_boxes3d.npy
#       <stem>_depth_map.npy
#       <stem>_class_ids.npy
#       <stem>_output.png
#       <stem>_scene.ply
#
# Examples
# --------
#   ./run_pp.sh mipnerf360 room DSCF4702.JPG "chair"
#   ./run_pp.sh mipnerf360 room DSCF4702.JPG "chair,table" --clip-radius 15
#   ./run_pp.sh mecenapolis scene1 "pano_camera1/pano_00051.jpg" "person,car"
#   ./run_pp.sh mipnerf360 room DSCF4702.JPG "chair" --exp-scene-name room_default

set -euo pipefail

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <dataset> <scene> <image> <prompt> [viz_args...]" >&2
    exit 1
fi

DATASET=$1
SCENE=$2
IMAGE=$3
PROMPT=$4
shift 4
VIZ_ARGS=("$@")   # remaining args forwarded to visualize_boxes3d.py

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================"
echo " WildDet3D pipeline"
echo "  dataset : $DATASET"
echo "  scene   : $SCENE"
echo "  image   : $IMAGE"
echo "  prompt  : $PROMPT"
echo "========================================"

echo ""
echo "[1/3] Extracting camera intrinsics..."
python "$SCRIPT_DIR/extract_intrinsics.py" "$DATASET" "$SCENE" "$IMAGE"

echo ""
echo "[2/3] Running inference..."
python "$SCRIPT_DIR/wildDet3D_infer.py" "$DATASET" "$SCENE" "$IMAGE" --prompt "$PROMPT"

echo ""
echo "[3/3] Generating PLY..."
python "$SCRIPT_DIR/visualize_boxes3d.py" "$DATASET" "$SCENE" "$IMAGE" --prompt "$PROMPT" "${VIZ_ARGS[@]}"

echo ""
echo "Done."