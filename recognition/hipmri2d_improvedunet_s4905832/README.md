# HipMRI 2D Prostate Segmentation (Improved U-Net), Clayton Nagle/s4905832

## Problem & Goal
Segment prostate on HipMRI 2D slices. **Target**: test prostate **Dice ≥ 0.75** (Normal difficulty). :contentReference[oaicite:13]{index=13}

## How It Works
Improved U-Net (2D) with BN/Dropout. Train on 70%, validate 15%, test 15% (patient-level split if metadata allows). Loss: Dice (prostate class). Metrics: per-class Dice with focus on prostate.

## Data
Rangpur: `/home/groups/comp3710/HipMRI_Study_open/keras_slices_data`. Do **not** commit data/models. :contentReference[oaicite:14]{index=14} :contentReference[oaicite:15]{index=15}

## Environment
Python 3.10, PyTorch (CUDA if available), nibabel/nilearn, scikit-image, matplotlib, tqdm. Deterministic with seed=42.

## Usage
```bash
python recognition/hipmri2d-improvedunet-<yourid>/train.py
python recognition/hipmri2d-improvedunet-<yourid>/predict.py
