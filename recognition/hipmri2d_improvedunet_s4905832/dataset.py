import os, glob, numpy as np, nibabel as nib
from torch.utils.data import Dataset, DataLoader
import torch
import skimage.transform as skt

def to_channels(arr: np.ndarray, num_classes: int = 6, dtype=np.uint8):
    # assumes labels in {0..num_classes-1}; adjust if needed
    h,w = arr.shape
    y = np.zeros((num_classes, h, w), dtype=dtype)
    for c in range(num_classes):
        y[c] = (arr == c).astype(dtype)
    return y

def load_names(root_img, root_seg, max_cases=None):
    """
    Match image (.nii.gz) and segmentation (.nii.gz) files by slice index.
    """
    imgs = sorted(glob.glob(os.path.join(root_img, "*.nii*")))
    labs = sorted(glob.glob(os.path.join(root_seg, "*.nii*")))

    # Extract slice identifiers (e.g., week_0_slice_10)
    img_map = {os.path.basename(f).replace("case_", ""): f for f in imgs}
    lab_map = {os.path.basename(f).replace("seg_", ""): f for f in labs}

    # Intersect by slice ID
    common_keys = sorted(set(img_map.keys()) & set(lab_map.keys()))
    pairs = [(img_map[k], lab_map[k]) for k in common_keys]

    return pairs[:max_cases] if max_cases else pairs

class HipMRISlices(Dataset):
    def __init__(self, names, split="train", img_size=256, norm=True, categorical=True):
        self.names = names
        n = len(names)
        n_train = int(0.7*n); n_val = int(0.15*n)
        if split=="train":   self.names = names[:n_train]
        elif split=="val":   self.names = names[n_train:n_train+n_val]
        else:                self.names = names[n_train+n_val:]
        self.img_size, self.norm, self.categorical = img_size, norm, categorical

    def __len__(self): return len(self.names)

    def __getitem__(self, i):
        x_nii = nib.load(self.names[i][0]).get_fdata(caching='unchanged')
        y_nii = nib.load(self.names[i][1]).get_fdata(caching='unchanged')
        if x_nii.ndim == 3: x_nii = x_nii[:,:,0]
        if y_nii.ndim == 3: y_nii = y_nii[:,:,0]
        if self.norm:
            x_nii = (x_nii - x_nii.mean()) / (x_nii.std() + 1e-8)  # spec example uses mean/std norm
        x_nii = skt.resize(x_nii, (self.img_size, self.img_size), preserve_range=True, anti_aliasing=True)
        y_nii = skt.resize(y_nii, (self.img_size, self.img_size), order=0, preserve_range=True, anti_aliasing=False)
        if self.categorical:
            y = to_channels(y_nii.astype(np.int64), num_classes=6).astype(np.float32)
        else:
            y = y_nii[None,...].astype(np.float32)
        x = x_nii[None,...].astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)

def make_loader(names, split, bs=8, **kw):
    return DataLoader(HipMRISlices(names, split=split, **kw), batch_size=bs, shuffle=(split=="train"), num_workers=4, pin_memory=True)
