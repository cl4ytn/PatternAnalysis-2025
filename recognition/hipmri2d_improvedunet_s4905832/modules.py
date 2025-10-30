import torch, torch.nn as nn

def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))

class UNet2D(nn.Module):
    def __init__(self, in_ch=1, out_ch=6, base=32, drop=0.1):
        super().__init__()
        self.enc1 = conv_block(in_ch, base)
        self.enc2 = conv_block(base, base*2)
        self.enc3 = conv_block(base*2, base*4)
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout2d(drop)
        self.bott = conv_block(base*4, base*8)
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, 2)
        self.dec3 = conv_block(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, 2)
        self.dec2 = conv_block(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, 2)
        self.dec1 = conv_block(base*2, base)
        self.out  = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x); p1 = self.pool(e1)
        e2 = self.enc2(p1); p2 = self.pool(e2)
        e3 = self.enc3(p2); p3 = self.pool(e3)
        b  = self.bott(self.drop(p3))
        d3 = self.up3(b); d3 = self.dec3(torch.cat([d3, e3], 1))
        d2 = self.up2(d3); d2 = self.dec2(torch.cat([d2, e2], 1))
        d1 = self.up1(d2); d1 = self.dec1(torch.cat([d1, e1], 1))
        return self.out(d1)
