import os, glob, numpy as np, nibabel as nib, torch
from torch.utils.data import Dataset, DataLoader
import skimage.transform as skt

# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------

def to_channels(arr: np.ndarray, num_classes: int = 6, dtype=np.uint8):
    """Convert integer label mask -> one-hot channels [C,H,W]."""
    h, w = arr.shape
    y = np.zeros((num_classes, h, w), dtype=dtype)
    for c in range(num_classes):
        y[c] = (arr == c).astype(dtype)
    return y


def load_names(root_img, root_seg, max_cases=None):
    """
    Pair MRI image and segmentation files by matching slice identifiers.
    e.g. case_004_week_0_slice_10.nii.gz ↔ seg_004_week_0_slice_10.nii.gz
    """
    imgs = sorted(glob.glob(os.path.join(root_img, "*.nii*")))
    labs = sorted(glob.glob(os.path.join(root_seg, "*.nii*")))

    img_map = {os.path.basename(f).replace("case_", ""): f for f in imgs}
    lab_map = {os.path.basename(f).replace("seg_", ""): f for f in labs}

    keys = sorted(set(img_map.keys()) & set(lab_map.keys()))
    pairs = [(img_map[k], lab_map[k]) for k in keys]

    if max_cases:
        pairs = pairs[:max_cases]
    return pairs


# -------------------------------------------------------------------
# Dataset class
# -------------------------------------------------------------------

class HipMRISlices(Dataset):
    """
    PyTorch Dataset for 2D HipMRI slices.
    Performs normalisation, resizing, and one-hot encoding.
    """

    def __init__(self, pairs, img_size=256, norm=True, categorical=True):
        self.pairs = pairs
        self.img_size = img_size
        self.norm = norm
        self.categorical = categorical

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        x_path, y_path = self.pairs[i]
        x_nii = nib.load(x_path).get_fdata()
        y_nii = nib.load(y_path).get_fdata()

        # Handle (H,W,1) or (H,W) shapes
        if x_nii.ndim == 3:
            x_nii = x_nii[:, :, 0]
        if y_nii.ndim == 3:
            y_nii = y_nii[:, :, 0]

        # Normalise
        if self.norm:
            x_nii = (x_nii - x_nii.mean()) / (x_nii.std() + 1e-8)

        # Resize
        x_nii = skt.resize(x_nii, (self.img_size, self.img_size), preserve_range=True, anti_aliasing=True)
        y_nii = skt.resize(y_nii, (self.img_size, self.img_size), order=0, preserve_range=True, anti_aliasing=False)

        # One-hot encode
        if self.categorical:
            y = to_channels(y_nii.astype(np.int64), num_classes=6).astype(np.float32)
        else:
            y = y_nii[None, ...].astype(np.float32)

        x = x_nii[None, ...].astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


# -------------------------------------------------------------------
# Dataloader helper
# -------------------------------------------------------------------

def make_loader(pairs, batch_size=8, shuffle=True, **kwargs):
    """Return a DataLoader for given (image,label) pairs."""
    ds = HipMRISlices(pairs, **kwargs)
    import os
    num_workers = min(4, os.cpu_count() // 2 or 1)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)
