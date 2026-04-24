#!/usr/bin/env bash
# WildDet3D full-scene pipeline
#
# Runs the three stages in order:
#   1. wildDet3D_scene_infer.py  — OWLv2 + WildDet3D on every scene image
#                                   (models loaded once)
#   2. wildDet3D_scene_agg.py   — cross-frame aggregation into scene objects
#   3. visualize_scene_agg.py   — PLY output for all aggregated boxes
#
# Usage
# -----
#   ./run_scene.sh <dataset> <scene> [infer_args...] [-- agg_args...] [-- viz_args...]
#
# Arguments
# ---------
#   dataset       e.g. "mipnerf360"
#   scene         e.g. "room"
#
# Arg groups (separated by "--"):
#   First  "--" separates inference args from aggregation args
#   Second "--" separates aggregation args from visualization args
#
# Inference args (wildDet3D_scene_infer.py):
#   --mode visual|geometric      (default: visual)
#   --owl-confidence FLOAT       (default: 0.5)
#   --nms-iou FLOAT              (default: 0.5)
#   --score-threshold FLOAT      (default: 0.3)
#
# Aggregation args (wildDet3D_scene_agg.py):
#   --sem-threshold FLOAT        cosine similarity gate for label grouping (default: 0.9)
#   --iou-threshold FLOAT        min 3D IoU to join an existing cluster  (default: 0.5)
#   --min-cluster-frac FLOAT     min fraction of detected frames per cluster (default: 0.15)
#   --min-cluster-no INT         min absolute box count per cluster (default: 10); overrides --min-cluster-frac
#   --score-w2d FLOAT            exponent for scores_2d in combined score (default: 1.0)
#   --score-w3d FLOAT            exponent for scores_3d in combined score (default: 1.0)
#   --global                     use scene-level COLMAP scale for all frames (default)
#   --local                      use per-frame COLMAP scale (corrects per-frame depth bias)
#   --align                      assume COLMAP space is already metric, scale=1.0 (no estimation)
#   --debug
#
# Visualization args (visualize_scene_agg.py):
#   --box-type median|best_score|best_location       (default: best_location)
#   --pts-per-edge INT           (default: 200)
#   --clip-radius FLOAT
#   --show-frame-centers
#   --exp-scene-name NAME
#
# Flags
# -----
#   --skip-infer   Skip Stage 1 (use existing per_frame/ outputs). Useful when
#                  tuning aggregation parameters without re-running inference.
#
# Examples
# --------
#   ./run_scene.sh mipnerf360 room
#   ./run_scene.sh mipnerf360 room --mode geometric
#   ./run_scene.sh mipnerf360 room -- --debug -- --show-frame-centers
#   ./run_scene.sh mipnerf360 room --mode geometric -- --min-cluster-frac 0.1 -- --exp-scene-name room_default
#   ./run_scene.sh mipnerf360 room --skip-infer -- --iou-threshold 0.3 --debug
#   ./run_scene.sh mipnerf360 room --skip-infer -- --sem-threshold 0.85 -- --show-frame-centers

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <dataset> <scene> [--skip-infer] [infer_args...] [-- agg_args...] [-- viz_args...]" >&2
    exit 1
fi

DATASET=$1
SCENE=$2
shift 2

# Parse --skip-infer before splitting into arg groups
SKIP_INFER=0
REMAINING=()
for arg in "$@"; do
    if [ "$arg" = "--skip-infer" ]; then
        SKIP_INFER=1
    else
        REMAINING+=("$arg")
    fi
done
set -- "${REMAINING[@]+"${REMAINING[@]}"}"

# Split remaining args into three groups separated by "--"
INFER_ARGS=()
AGG_ARGS=()
VIZ_ARGS=()
SEP_COUNT=0

for arg in "$@"; do
    if [ "$arg" = "--" ]; then
        SEP_COUNT=$((SEP_COUNT + 1))
    elif [ "$SEP_COUNT" -eq 0 ]; then
        INFER_ARGS+=("$arg")
    elif [ "$SEP_COUNT" -eq 1 ]; then
        AGG_ARGS+=("$arg")
    else
        VIZ_ARGS+=("$arg")
    fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================"
echo " WildDet3D scene pipeline"
echo "  dataset : $DATASET"
echo "  scene   : $SCENE"
echo "========================================"

if [ "$SKIP_INFER" -eq 1 ]; then
    echo ""
    echo "[1/3] Skipping inference (--skip-infer)."
else
    echo ""
    echo "[1/3] Per-frame inference (OWLv2 + WildDet3D)..."
    python "$SCRIPT_DIR/wildDet3D_scene_infer.py" \
        "$DATASET" "$SCENE" \
        "${INFER_ARGS[@]+"${INFER_ARGS[@]}"}"
fi

echo ""
echo "[2/3] Cross-frame aggregation..."
python "$SCRIPT_DIR/wildDet3D_scene_agg.py" \
    "$DATASET" "$SCENE" \
    "${AGG_ARGS[@]+"${AGG_ARGS[@]}"}"

echo ""
echo "[3/3] Generating PLY..."
python "$SCRIPT_DIR/visualize_scene_agg.py" \
    "$DATASET" "$SCENE" \
    "${VIZ_ARGS[@]+"${VIZ_ARGS[@]}"}"

echo ""
echo "Done. Outputs in: outputs/$DATASET/$SCENE/3Dresults/auto_owlv2/"