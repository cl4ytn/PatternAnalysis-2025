# HipMRI 2D Prostate Segmentation (Improved U-Net)
**Author:** Clayton Nagle (s4905832)  
**Branch:** `topic-recognition`  
**Task:** Normal Difficulty – HipMRI 2D prostate segmentation using Improved U-Net  
**Target:** Dice ≥ 0.75 on prostate label  

---

## 🎯 Problem & Goal
This project performs **prostate segmentation** on 2-D HipMRI slices using an **Improved U-Net** architecture.  
The goal is to reach at least **0.75 Dice** accuracy on the prostate label (Normal difficulty level in COMP3710).  
A validation Dice of **0.86** and test Dice of **0.64 ± 0.48** were achieved.

---

## 🧠 Approach
- **Model:** Improved U-Net (2D) with BatchNorm + Dropout  
- **Input size:** 128 × 128  
- **Loss:** Dice Loss (prostate channel)  
- **Optimizer:** AdamW (lr = 3e-4, weight decay = 1e-4)  
- **Early Stopping & Resume:** stops after 5 epochs without improvement, resumes from `checkpoints/best.pt`  
- **Evaluation:** Mean Dice per class, with focus on the prostate (label 5)  
- **Inference:** Predicts test set masks, computes Dice, saves overlay figures.

