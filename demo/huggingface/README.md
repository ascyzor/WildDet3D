# WildDet3D HuggingFace Demo

Interactive web demo for WildDet3D with text, point, and box prompts.

**Live demo**: [https://huggingface.co/spaces/allenai/WildDet3D](https://huggingface.co/spaces/allenai/WildDet3D)

## Prompt Modes

- **Text**: Enter object names (e.g., "chair.table"), click Run
- **Box-to-Multi-Object**: Draw box -> detect ALL similar objects (one-to-many)
- **Box-to-Single-Object**: Draw box -> detect ONLY the boxed object (one-to-one)
- **Point**: Click on object, click Run
- **+ Label**: Attach a category name (e.g., "chair") to box/point prompts

## Run Locally

```bash
cd WildDet3D

# Install demo dependencies
pip install gradio>=5.0.0
pip install -r demo/huggingface/requirements.txt

# Run (checkpoint auto-downloaded from HuggingFace Hub)
python demo/huggingface/app.py
```

Then open http://localhost:7860 in your browser.

## Deploy to HuggingFace Spaces

Bundle all code into a self-contained directory using the pack script:

```bash
cd WildDet3D
bash demo/huggingface/pack_hf_space.sh

# Push to HF Spaces
cd hf_space
git init && git lfs install
git remote add origin https://huggingface.co/spaces/allenai/WildDet3D
git add . && git commit -m "update" && git push
```

### HuggingFace Space Layout

```
hf_space/
├── app.py                  # Gradio UI
├── vis3d_glb.py            # 3D visualization
├── requirements.txt
├── assets/demo/            # Default demo images
├── wilddet3d/              # Core model code
├── vis4d/                  # Framework (transforms, geometry)
└── third_party/
    ├── sam3/               # SAM3 backbone
    └── lingbot_depth/      # LingBot-Depth backend
```
