#!/bin/bash
# Pack WildDet3D HuggingFace Space into a self-contained directory.
#
# Usage:
#   cd WildDet3D
#   bash demo/huggingface/pack_hf_space.sh [output_dir]
#
# Default output: hf_space/
# Then push to HF:
#   cd hf_space && git init && git remote add origin https://huggingface.co/spaces/allenai/WildDet3D
#   git add . && git commit -m "update" && git push

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:-${REPO_ROOT}/hf_space}"

# Use cp -r instead of rsync (not always available)
# Helper: copy dir excluding __pycache__ and .pyc
copy_clean() {
    local src="$1" dst="$2"
    cp -r "${src}" "${dst}"
    find "${dst}" -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
    find "${dst}" -name '*.pyc' -delete 2>/dev/null || true
}

echo "=== Packing WildDet3D HuggingFace Space ==="
echo "Repo root: ${REPO_ROOT}"
echo "Output:    ${OUT}"
echo

# Clean previous build
rm -rf "${OUT}"
mkdir -p "${OUT}"

# ---- Demo files ----
echo "[1/5] Copying demo files..."
cp "${REPO_ROOT}/demo/huggingface/app.py"           "${OUT}/"
cp "${REPO_ROOT}/demo/huggingface/vis3d_glb.py"     "${OUT}/"
cp "${REPO_ROOT}/demo/huggingface/requirements.txt" "${OUT}/"
cp -r "${REPO_ROOT}/demo/huggingface/assets"         "${OUT}/"

# ---- wilddet3d package ----
echo "[2/5] Copying wilddet3d/ ..."
copy_clean "${REPO_ROOT}/wilddet3d" "${OUT}/wilddet3d"

# ---- third_party (sam3, lingbot_depth) ----
# Only copy python packages, skip assets/examples/docs to save space.
echo "[3/5] Copying third_party/sam3 (python only)..."
mkdir -p "${OUT}/third_party/sam3"
copy_clean "${REPO_ROOT}/third_party/sam3/sam3" "${OUT}/third_party/sam3/sam3"
for f in setup.py setup.cfg pyproject.toml __init__.py; do
    [ -f "${REPO_ROOT}/third_party/sam3/${f}" ] && \
        cp "${REPO_ROOT}/third_party/sam3/${f}" "${OUT}/third_party/sam3/"
done

echo "[3/5] Copying third_party/lingbot_depth (python only)..."
mkdir -p "${OUT}/third_party/lingbot_depth"
copy_clean "${REPO_ROOT}/third_party/lingbot_depth/mdm" "${OUT}/third_party/lingbot_depth/mdm"
for f in setup.py setup.cfg pyproject.toml __init__.py; do
    [ -f "${REPO_ROOT}/third_party/lingbot_depth/${f}" ] && \
        cp "${REPO_ROOT}/third_party/lingbot_depth/${f}" "${OUT}/third_party/lingbot_depth/"
done

# ---- vis4d framework ----
# vis4d is a dependency but not in the release repo.
# Copy from old bundled demo or from installed package.
echo "[4/5] Copying vis4d/ ..."
OLD_VIS4D="${REPO_ROOT}/../../molmo_3det_huggingface_demo/vis4d"
if [ -d "${OLD_VIS4D}" ]; then
    copy_clean "${OLD_VIS4D}" "${OUT}/vis4d"
else
    VIS4D_PKG=$(python -c "import vis4d; import os; print(os.path.dirname(vis4d.__file__))" 2>/dev/null || true)
    if [ -n "${VIS4D_PKG}" ] && [ -d "${VIS4D_PKG}" ]; then
        copy_clean "${VIS4D_PKG}" "${OUT}/vis4d"
    else
        echo "ERROR: Cannot find vis4d. Provide path or install vis4d."
        exit 1
    fi
fi

# ---- Summary ----
echo "[5/5] Done!"
echo
echo "=== Space contents ==="
du -sh "${OUT}"/*/ "${OUT}"/*.py 2>/dev/null | sort -rh
echo
TOTAL=$(du -sh "${OUT}" | cut -f1)
echo "Total: ${TOTAL}"
echo
echo "Next steps:"
echo "  cd ${OUT}"
echo "  # Test locally: python app.py"
echo "  # Push to HF Spaces:"
echo "  git init && git lfs install"
echo "  git remote add origin https://huggingface.co/spaces/allenai/WildDet3D"
echo "  git add . && git commit -m 'WildDet3D demo' && git push"
