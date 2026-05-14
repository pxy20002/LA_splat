import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv2d → BatchNorm → ReLU) × 2, stride-1, no spatial change."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class EncoderBlock(nn.Module):
    """DoubleConv + MaxPool. Returns (downsampled, pre_pool) for skip connection."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.conv(x)
        down = self.pool(skip)
        return down, skip


class DecoderBlock(nn.Module):
    """Upsample → Conv(match skip size) → Concat(skip) → DoubleConv.

    Args:
        in_ch:  channels from the previous decoder stage.
        skip_ch: channels of the skip connection being concatenated.
        out_ch:  desired output channels.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, skip_ch, 1, bias=False),
        )
        self.conv = DoubleConv(skip_ch * 2, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class LightweightUNet(nn.Module):
    """
    Lightweight 2D U-Net for multi-view feature extraction.

    Input:  [B, 10, H, W]   (RGB 3 + Depth 1 + Plücker Ray 6)
    Output: [B, C_out, H, W] feature maps.

    Architecture: 4-level encoder-decoder with skip connections.
    Channel progression: 10→32→64→128→256 (bottleneck).
    """

    def __init__(self, in_channels: int = 10, out_channels: int = 64):
        super().__init__()
        c = [32, 64, 128, 256]

        # Encoder
        self.enc1 = EncoderBlock(in_channels, c[0])   # 10→32
        self.enc2 = EncoderBlock(c[0], c[1])           # 32→64
        self.enc3 = EncoderBlock(c[1], c[2])           # 64→128
        self.enc4 = EncoderBlock(c[2], c[3])           # 128→256

        self.bottleneck = DoubleConv(c[3], c[3])       # 256→256

        # Decoder  (in, skip, out)
        self.dec4 = DecoderBlock(c[3], c[3], c[3])     # 256 + 256 = 512 → 256
        self.dec3 = DecoderBlock(c[3], c[2], c[2])     # 256 + 128 = 384 → 128
        self.dec2 = DecoderBlock(c[2], c[1], c[1])     # 128 + 64  = 192 → 64
        self.dec1 = DecoderBlock(c[1], c[0], c[0])     # 64  + 32  = 96  → 32

        self.out_conv = nn.Conv2d(c[0], out_channels, 1)  # 32 → C_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        d1, s1 = self.enc1(x)
        d2, s2 = self.enc2(d1)
        d3, s3 = self.enc3(d2)
        d4, s4 = self.enc4(d3)

        # Bottleneck
        b = self.bottleneck(d4)

        # Decoder
        u4 = self.dec4(b, s4)
        u3 = self.dec3(u4, s3)
        u2 = self.dec2(u3, s2)
        u1 = self.dec1(u2, s1)

        return self.out_conv(u1)
