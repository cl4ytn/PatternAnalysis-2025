import torch, torch.nn as nn, torch.optim as optim
from modules import UNet2D
from dataset import load_names, make_loader
import matplotlib.pyplot as plt, os, json

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6, class_ix=5):  # assume prostate label index=5; adjust to your mapping
        super().__init__(); self.eps=eps; self.class_ix=class_ix
    def forward(self, logits, y):
        # logits: [B,C,H,W], y: [B,C,H,W] one-hot
        probs = torch.softmax(logits, dim=1)
        p = probs[:, self.class_ix]; t = y[:, self.class_ix]
        num = (2*(p*t).sum(dim=(1,2)) + self.eps)
        den = (p.pow(2).sum(dim=(1,2)) + t.pow(2).sum(dim=(1,2)) + self.eps)
        return 1 - (num/den).mean()

def evaluate(model, loader, class_ix=5):
    model.eval(); dsum=0; n=0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.cuda(), y.cuda()
            p = torch.softmax(model(x), dim=1)[:,class_ix]
            t = y[:,class_ix]
            num = (2*(p*t).sum(dim=(1,2)) + 1e-6)
            den = (p.pow(2).sum(dim=(1,2)) + t.pow(2).sum(dim=(1,2)) + 1e-6)
            dsum += (num/den).sum().item(); n += x.size(0)
    return dsum/n

def main():
    root = "/path/to/HipMRI_Study_open/keras_slices_data"
    names = load_names(root)
    tr = make_loader(names, "train", bs=8, img_size=256)
    va = make_loader(names, "val",   bs=8, img_size=256)
    model = UNet2D(in_ch=1, out_ch=6, base=32).cuda()
    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    crit = DiceLoss(class_ix=5)  # prostate
    best, history = 0.0, {"val_dice":[]}
    os.makedirs("checkpoints", exist_ok=True)
    for ep in range(40):
        model.train()
        for x,y in tr:
            x,y = x.cuda(), y.cuda()
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward(); opt.step()
        val_dice = evaluate(model, va, class_ix=5)
        history["val_dice"].append(val_dice)
        if val_dice>best:
            best=val_dice; torch.save(model.state_dict(), "checkpoints/best.pt")
        print(f"Epoch {ep}: val prostate Dice={val_dice:.3f} (best {best:.3f})")
    plt.figure(); plt.plot(history["val_dice"]); plt.title("Val Prostate Dice"); plt.savefig("runs/val_dice.png")
    with open("runs/history.json","w") as f: json.dump(history,f)

if __name__=="__main__": main()
