import torch, os
import numpy as np
import matplotlib.pyplot as plt
from modules import build_model
from dataset import load_names, make_loader
from torch.nn.functional import softmax

# -----------------------------
# 1. Device setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", device)

# -----------------------------
# 2. Paths
# -----------------------------
root_img_test = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test"
root_seg_test = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_test"

# -----------------------------
# 3. Data loader
# -----------------------------
pairs = load_names(root_img_test, root_seg_test)
test_loader = make_loader(pairs, batch_size=4, img_size=256, shuffle=False)

# -----------------------------
# 4. Model load
# -----------------------------
model = build_model(name="unet", in_ch=1, out_ch=6, base=32, depth=3)
ckpt_path = "checkpoints/best.pt"
assert os.path.exists(ckpt_path), "❌ No checkpoint found! Train first."
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.to(device)
model.eval()
print("✅ Model loaded from", ckpt_path)

# -----------------------------
# 5. Metric (Dice)
# -----------------------------
def dice_score(pred, target, eps=1e-6):
    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2. * inter + eps) / (union + eps)

# -----------------------------
# 6. Evaluation
# -----------------------------
os.makedirs("figs_test", exist_ok=True)
dice_scores = []
with torch.no_grad():
    for i, (x, y) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)
        out = softmax(model(x), dim=1)           # [B,C,H,W]
        pred = (out[:, 5] > 0.5).float()         # prostate mask only (channel 5)
        target = y[:, 5]
        # 🔧 Ensure prediction and target have the same spatial size
        if pred.shape[-1] != target.shape[-1]:
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=target.shape[-2:], mode="bilinear", align_corners=False
            ).squeeze(1)
        for b in range(x.size(0)):
            dice = dice_score(pred[b], target[b])
            dice_scores.append(dice.item())

            # save overlay for first few samples
            if i < 2 and b < 2:
                img = x[b, 0].cpu().numpy()
                m_pred = pred[b].cpu().numpy()
                m_true = target[b].cpu().numpy()
                plt.figure(figsize=(5,5))
                plt.imshow(img, cmap="gray")
                plt.contour(m_true, levels=[0.5], colors="lime", linewidths=1.0)
                plt.contour(m_pred, levels=[0.5], colors="red", linewidths=1.0)
                plt.title(f"Green = GT, Red = Pred | Dice {dice:.3f}")
                plt.axis("off")
                plt.tight_layout()
                plt.savefig(f"figs_test/overlay_{i}_{b}.png", dpi=200)
                plt.close()

print(f"✅ Mean Prostate Dice = {np.mean(dice_scores):.3f} ± {np.std(dice_scores):.3f}")

