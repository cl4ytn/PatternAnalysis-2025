import torch, os
from modules import UNet2D
from dataset import load_names, make_loader
import matplotlib.pyplot as plt

@torch.no_grad()
def main():
    root = "/path/to/HipMRI_Study_open/keras_slices_data"
    names = load_names(root)
    te = make_loader(names, "test", bs=4, img_size=256)
    model = UNet2D().cuda()
    model.load_state_dict(torch.load("checkpoints/best.pt", map_location="cuda"))
    model.eval()
    os.makedirs("figs", exist_ok=True)
    # quick metric + overlay examples
    tot, n = 0.0, 0
    for i,(x,y) in enumerate(te):
        x,y = x.cuda(), y.cuda()
        p = torch.softmax(model(x), dim=1)[:,5]  # prostate
        t = y[:,5]
        num = (2*(p*t).sum(dim=(1,2))+1e-6); den = (p.pow(2).sum(dim=(1,2))+t.pow(2).sum(dim=(1,2))+1e-6)
        tot += (num/den).sum().item(); n += x.size(0)
        if i<4:  # save overlays for README
            import numpy as np
            img = x[0,0].detach().cpu().numpy()
            mask= t[0].detach().cpu().numpy()
            pred= (p[0].detach().cpu().numpy()>0.5).astype(float)
            plt.figure(); plt.imshow(img, cmap="gray");
            plt.contour(mask, levels=[0.5], linewidths=1); plt.contour(pred, levels=[0.5], linewidths=1)
            plt.title("GT (blue) vs Pred (orange)"); plt.savefig(f"figs/overlay_{i}.png")
    print(f"Test prostate Dice: {tot/n:.3f}")

if __name__=="__main__": main()
