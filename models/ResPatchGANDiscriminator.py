import torch
import torch.nn as nn


def get_gn_groups(channels, target_per=16):
    groups = channels
    groups = max(1, groups)
    while channels % groups != 0:
        groups -= 1
    return groups


class ResPatchBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: bool = True):
        super().__init__()
        stride = 2
        # Main branch: same as original block
        self.conv = nn.Conv2d(in_ch, out_ch, 4, stride, 1, bias=False)
        self.norm = nn.GroupNorm(get_gn_groups(out_ch), out_ch) if norm else nn.Identity()
        self.act = nn.LeakyReLU(0.2, inplace=True)

        # Shortcut path to match size & channel
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 4, stride, 1, bias=False),
                nn.GroupNorm(get_gn_groups(out_ch), out_ch) if norm else nn.Identity()
            )

    def forward(self, x):
        main = self.norm(self.conv(x))
        skip = self.shortcut(x)
        out = main + skip
        return self.act(out)


class ResPatchGANDiscriminator(nn.Module):
    def __init__(self, in_ch: int = 4) -> None:
        super().__init__()
        self.model = nn.Sequential(
            ResPatchBlock(in_ch, 64, norm=False),
            ResPatchBlock(64, 128),
            ResPatchBlock(128, 256),
            ResPatchBlock(256, 512),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    @staticmethod
    def block(in_ch: int, out_ch: int, norm: bool = True) -> nn.Sequential:
        layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1)]
        if norm:
            layers.append(nn.GroupNorm(get_gn_groups(out_ch), out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)
