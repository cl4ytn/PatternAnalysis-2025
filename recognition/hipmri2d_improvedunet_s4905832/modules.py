import torch
import torch.nn as nn
from typing import Tuple, Literal

# -------------------------------
# Helpers
# -------------------------------

def kaiming_init(module: nn.Module):
    """Kaiming init for Conv/Norm layers."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -------------------------------
# Building blocks
# -------------------------------

class DoubleConv(nn.Module):
    """Conv-BN-ReLU ×2 (Improved U-Net style: BN, optional dropout outside)."""
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    """Upsample via transposed conv, then DoubleConv on concatenated skip."""
    def __init__(self, c_in: int, c_skip: int, c_out: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_out, kernel_size=2, stride=2)
        self.conv = DoubleConv(c_out + c_skip, c_out)

    def forward(self, x, skip):
        x = self.up(x)
        # handle odd-sized inputs by center-cropping skip if needed
        if x.shape[-2:] != skip.shape[-2:]:
            dh = skip.shape[-2] - x.shape[-2]
            dw = skip.shape[-1] - x.shape[-1]
            skip = skip[..., dh//2:skip.shape[-2]-((dh+1)//2), dw//2:skip.shape[-1]-((dw+1)//2)]
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# -------------------------------
# Improved U-Net 2D
# -------------------------------

class UNet2D(nn.Module):
    """
    Improved U-Net for 2D segmentation (BN + Dropout).
    Output is raw logits [B, C, H, W]; apply softmax in loss/eval code.
    """
    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 6,
        base: int = 32,
        drop: float = 0.1,
        depth: int = 4,
    ):
        super().__init__()
        assert depth in (3, 4, 5), "depth must be 3/4/5 for typical U-Nets"
        self.drop = nn.Dropout2d(drop)

        # Encoder
        self.enc1 = DoubleConv(in_ch, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)

        if depth >= 4:
            self.enc4 = DoubleConv(base * 4, base * 8)
        if depth == 5:
            self.enc5 = DoubleConv(base * 8, base * 16)

        # Bottleneck
        bott_in = {3: base * 4, 4: base * 8, 5: base * 16}[depth]
        bott_out = bott_in
        self.bott = DoubleConv(bott_in, bott_out)

        # Decoder (mirror)
        if depth == 5:
            self.up5 = UpBlock(bott_out, base * 8, base * 8)
        if depth >= 4:
            self.up4 = UpBlock({5: base * 8, 4: bott_out}[depth], base * 4, base * 4)
        self.up3 = UpBlock(base * 4, base * 2, base * 2)
        self.up2 = UpBlock(base * 2, base, base)

        self.out_conv = nn.Conv2d(base, out_ch, kernel_size=1)

        # init
        self.apply(kaiming_init)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)                  # [B, base, H, W]
        p1 = self.pool(e1)
        e2 = self.enc2(p1)                 # [B, 2b, H/2, W/2]
        p2 = self.pool(e2)
        e3 = self.enc3(p2)                 # [B, 4b, H/4, W/4]
        p3 = self.pool(e3)

        if hasattr(self, "enc4"):
            e4 = self.enc4(p3)             # [B, 8b, H/8, W/8]
            p4 = self.pool(e4)
        if hasattr(self, "enc5"):
            e5 = self.enc5(p4)             # [B, 16b, H/16, W/16]
            p5 = self.pool(e5)

        # Bottleneck
        b_in = p5 if hasattr(self, "enc5") else (p4 if hasattr(self, "enc4") else p3)
        b = self.bott(self.drop(b_in))

        # Decoder
        if hasattr(self, "enc5"):
            d5 = self.up5(b, e4)
            d4_in = d5
        elif hasattr(self, "enc4"):
            d4_in = b
        else:
            d4_in = None

        if hasattr(self, "enc4"):
            d4 = self.up4(d4_in, e3)
            d3_in = d4
        else:
            d3_in = b

        d3 = self.up3(d3_in, e2)
        d2 = self.up2(d3, e1)

        logits = self.out_conv(d2)         # raw scores [B, out_ch, H, W]
        return logits


# -------------------------------
# Optional: CAN-style (dilated) head
# -------------------------------

class DilatedCAN2D(nn.Module):
    """
    A light CAN-style network using progressive dilations to aggregate context.
    Good as a compact alternative/baseline.
    """
    def __init__(self, in_ch: int = 1, out_ch: int = 6, base: int = 48):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )
        # context aggregation with increasing dilation
        blocks = []
        ch = base
        for d in [1, 2, 4, 8, 16]:
            blocks += [
                nn.Conv2d(ch, ch, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
            ]
        self.body = nn.Sequential(*blocks)
        self.head = nn.Conv2d(ch, out_ch, 1)
        self.apply(kaiming_init)

    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        x = self.head(x)  # logits
        return x


# -------------------------------
# Factory
# -------------------------------

ModelName = Literal["unet", "can"]

def build_model(
    name: ModelName = "unet",
    in_ch: int = 1,
    out_ch: int = 6,
    base: int = 32,
    drop: float = 0.1,
    depth: int = 4,
) -> nn.Module:
    """
    Create a model by name.
    - 'unet': Improved U-Net 2D (default)
    - 'can' : Dilated CAN-style network
    """
    if name == "unet":
        return UNet2D(in_ch=in_ch, out_ch=out_ch, base=base, drop=drop, depth=depth)
    elif name == "can":
        return DilatedCAN2D(in_ch=in_ch, out_ch=out_ch, base=max(base, 48))
    else:
        raise ValueError(f"Unknown model name: {name}")

