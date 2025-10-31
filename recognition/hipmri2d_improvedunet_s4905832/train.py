import torch, torch.nn as nn, torch.optim as optim
from modules import build_model
from dataset import load_names, make_loader
import matplotlib.pyplot as plt, os, json

# --------------------------------------------------------
# 1. Dataset paths
# --------------------------------------------------------
root_img_train = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train"
root_seg_train = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_train"

root_img_val = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_validate"
root_seg_val = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_validate"

# --------------------------------------------------------
# 2. Load datasets
# --------------------------------------------------------
train_pairs = load_names(root_img_train, root_seg_train)
val_pairs   = load_names(root_img_val,  root_seg_val)

train_loader = make_loader(train_pairs, batch_size=8, img_size=128, shuffle=True)
val_loader   = make_loader(val_pairs,   batch_size=8, img_size=128, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------
# 3. Model setup + optimizer
# --------------------------------------------------------
model = build_model(name="unet", in_ch=1, out_ch=6, base=32, depth=3).to(device)
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

# --------------------------------------------------------
# 4. Dice Loss & evaluation
# --------------------------------------------------------
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6, class_ix=5):
        super().__init__()
        self.eps, self.class_ix = eps, class_ix

    def forward(self, logits, y):
        probs = torch.softmax(logits, dim=1)
        # auto-resize if needed
        if probs.shape[-1] != y.shape[-1]:
            probs = torch.nn.functional.interpolate(
                probs, size=y.shape[-2:], mode="bilinear", align_corners=False)
        p = probs[:, self.class_ix]
        t = y[:, self.class_ix]
        num = (2 * (p * t).sum(dim=(1, 2)) + self.eps)
        den = (p.pow(2).sum(dim=(1, 2)) + t.pow(2).sum(dim=(1, 2)) + self.eps)
        return 1 - (num / den).mean()

def evaluate(model, loader, class_ix=5):
    model.eval()
    dsum, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = torch.softmax(model(x), dim=1)

            # 🔧 Fix: ensure output and target have same spatial size
            if out.shape[-1] != y.shape[-1]:
                out = torch.nn.functional.interpolate(
                    out, size=y.shape[-2:], mode="bilinear", align_corners=False
                )

            p = out[:, class_ix]
            t = y[:, class_ix]
            num = (2 * (p * t).sum(dim=(1, 2)) + 1e-6)
            den = (p.pow(2).sum(dim=(1, 2)) + t.pow(2).sum(dim=(1, 2)) + 1e-6)
            dsum += (num / den).sum().item()
            n += x.size(0)
    return dsum / n

# --------------------------------------------------------
# 5. Training loop with resume + early stop
# --------------------------------------------------------
criterion = DiceLoss(class_ix=5)
best_dice = 0.0
patience, patience_counter = 5, 0
history = {"val_dice": []}

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("runs", exist_ok=True)

# --- Resume if checkpoint exists ---
ckpt_path = "checkpoints/best.pt"
if os.path.exists(ckpt_path):
    print("Resuming from checkpoint...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print("Checkpoint loaded.")

num_epochs = 40
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    val_dice = evaluate(model, val_loader, class_ix=5)
    history["val_dice"].append(val_dice)
    print(f"Epoch {epoch+1}/{num_epochs} | Loss {epoch_loss/len(train_loader):.4f} | Val Dice {val_dice:.3f}")

    # --- Save best ---
    if val_dice > best_dice + 1e-4:
        best_dice = val_dice
        torch.save(model.state_dict(), ckpt_path)
        print(f"✅ New best model saved ({best_dice:.3f})")
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"No improvement ({patience_counter}/{patience})")

    # --- Early stopping ---
    if patience_counter >= patience:
        print("⏹ Early stopping triggered.")
        break

# --------------------------------------------------------
# 6. Save metrics & plot
# --------------------------------------------------------
plt.figure()
plt.plot(history["val_dice"])
plt.xlabel("Epoch")
plt.ylabel("Validation Dice (Prostate)")
plt.title("Validation Dice over Epochs")
plt.grid(True)
plt.tight_layout()
plt.savefig("runs/val_dice_curve.png", dpi=150)

with open("runs/history.json", "w") as f:
    json.dump(history, f)

print(f"Training complete! Best Val Dice = {best_dice:.3f}")

