from recognition.hipmri2d_improvedunet_s4905832.dataset import load_names, HipMRISlices

root_img = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train"
root_seg = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_train"

pairs = load_names(root_img, root_seg, max_cases=2)
print(f"Loaded {len(pairs)} samples")

ds = HipMRISlices(pairs, split="train", img_size=256, norm=True, categorical=True)
x, y = ds[0]
print("Image shape:", x.shape)
print("Mask shape:", y.shape)
print("Unique mask values:", y.unique())

import matplotlib.pyplot as plt
import numpy as np

# Convert tensors to numpy
img = x[0].numpy()           # shape: [H, W]
mask = y.numpy()             # shape: [C, H, W]

# Select prostate class (assumed last channel, adjust if needed)
prostate_mask = mask[-1]     # usually class index 5

plt.figure(figsize=(5,5))
plt.imshow(img, cmap='gray')
plt.contour(prostate_mask, levels=[0.5], colors='r', linewidths=1)
plt.title("MRI slice with prostate segmentation overlay")
plt.axis('off')
plt.tight_layout()
plt.savefig("recognition/hipmri2d_improvedunet_s4905832/figs/overlay_sample.png", dpi=200)
plt.show()
