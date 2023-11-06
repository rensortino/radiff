from torch import nn
from einops import rearrange

class MaskEncoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 7, 1, 1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            # nn.Conv2d(128, out_channels, 3, 2, 1),
            nn.Conv2d(128, out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            # nn.Conv2d(256, out_channels, 1, 1, 0),
            # nn.BatchNorm2d(out_channels),
            # nn.GELU(),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class ImageEncoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=768):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 7, 1, 1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Conv2d(512, out_channels, 3, 2, 0),
            # nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        out = self.net(x)
        out = rearrange(out, 'b c h w -> b (h w) c')
        return out
    
if __name__ == '__main__':
    import torch
    model = MaskEncoder(1, 1)
    x = torch.randn(6, 1, 128, 128)
    y = model(x)
    print("Mask", y.shape)
    model = ImageEncoder(3, 768)
    x = torch.randn(6, 3, 128, 128)
    y = model(x)
    print("Image", y.shape)