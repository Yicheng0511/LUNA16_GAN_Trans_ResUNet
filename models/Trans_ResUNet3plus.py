import torch
import torch.nn as nn
import torch.nn.functional as F
from models.SegViT import SegViTBottleneck as transformer


def get_gn_groups(channels, target_per=16):
    groups = channels // target_per
    groups = max(1, groups)
    while channels % groups != 0:
        groups -= 1
    return groups
    
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(get_gn_groups(in_channels), in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.gn2 = nn.GroupNorm(get_gn_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(get_gn_groups(out_channels), out_channels)
        )

    def forward(self, x):
        residual = self.downsample(x)
        out = self.gn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.gn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        return out                        
    
    
class ResUNet3plus(nn.Module):
    def __init__(self, base_ch: int = 64) -> None:
        super().__init__()
        self.e1 = ResBlock(3, base_ch)
        self.e2 = ResBlock(base_ch, base_ch * 2)
        self.e3 = ResBlock(base_ch * 2, base_ch * 4)
        self.e4 = ResBlock(base_ch * 4, base_ch * 8)
        self.bottleneck = transformer(base_ch * 8)

        self.fusion = self.conv_block(base_ch * 5, base_ch)
        def make_feat_reduce(in_ch, out_ch):
            g = get_gn_groups(out_ch)
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding="same", bias=False),
                nn.GroupNorm(g, out_ch),
                nn.ReLU(inplace=True)
            )

        self.conv1 = make_feat_reduce(base_ch, base_ch)
        self.conv2 = make_feat_reduce(base_ch * 2, base_ch)
        self.conv3 = make_feat_reduce(base_ch * 4, base_ch)
        self.conv4 = make_feat_reduce(base_ch * 8, base_ch)
        self.conv5 = make_feat_reduce(base_ch * 8, base_ch)
        self.conv_dec = make_feat_reduce(base_ch, base_ch)

        self.pool = nn.MaxPool2d(2, 2)

        self.out1 = nn.Conv2d(base_ch, 1, 1)
        self.out2 = nn.Conv2d(base_ch, 1, 1)
        self.out3 = nn.Conv2d(base_ch, 1, 1)
        self.out4 = nn.Conv2d(base_ch, 1, 1)
        self.out5 = nn.Conv2d(base_ch * 8, 1, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        e5 = self.bottleneck(e4)

        e1_d4 = self.pool(self.pool(self.pool(e1)))
        e2_d4 = self.pool(self.pool(e2))
        e3_d4 = self.pool(e3)
        e5_d4 = F.interpolate(e5, e4.shape[2:], mode="bilinear", align_corners=False)

        e1_d4 = self.conv1(e1_d4)
        e2_d4 = self.conv2(e2_d4)
        e3_d4 = self.conv3(e3_d4)
        e4_d4 = self.conv4(e4)
        e5_d4 = self.conv5(e5_d4)
        d4 = self.fusion(torch.cat([e5_d4, e4_d4, e3_d4, e2_d4, e1_d4], 1))

        e1_d3 = self.pool(self.pool(e1))
        e2_d3 = self.pool(e2)
        d4_d3 = F.interpolate(d4, e3.shape[2:], mode="bilinear", align_corners=False)
        e5_d3 = F.interpolate(e5, e3.shape[2:], mode="bilinear", align_corners=False)

        e1_d3 = self.conv1(e1_d3)
        e2_d3 = self.conv2(e2_d3)
        e3_d3 = self.conv3(e3)
        d4_d3 = self.conv_dec(d4_d3)
        e5_d3 = self.conv5(e5_d3)
        d3 = self.fusion(torch.cat([e5_d3, d4_d3, e3_d3, e2_d3, e1_d3], 1))

        e1_d2 = self.pool(e1)
        d3_d2 = F.interpolate(d3, e2.shape[2:], mode="bilinear", align_corners=False)
        d4_d2 = F.interpolate(d4, e2.shape[2:], mode="bilinear", align_corners=False)
        e5_d2 = F.interpolate(e5, e2.shape[2:], mode="bilinear", align_corners=False)

        e1_d2 = self.conv1(e1_d2)
        e2_d2 = self.conv2(e2)
        d3_d2 = self.conv_dec(d3_d2)
        d4_d2 = self.conv_dec(d4_d2)
        e5_d2 = self.conv5(e5_d2)
        d2 = self.fusion(torch.cat([d3_d2, d4_d2, e5_d2, e2_d2, e1_d2], 1))

        d2_d1 = F.interpolate(d2, e1.shape[2:], mode="bilinear", align_corners=False)
        d3_d1 = F.interpolate(d3, e1.shape[2:], mode="bilinear", align_corners=False)
        d4_d1 = F.interpolate(d4, e1.shape[2:], mode="bilinear", align_corners=False)
        e5_d1 = F.interpolate(e5, e1.shape[2:], mode="bilinear", align_corners=False)

        e1_d1 = self.conv1(e1)
        d2_d1 = self.conv_dec(d2_d1)
        d3_d1 = self.conv_dec(d3_d1)
        d4_d1 = self.conv_dec(d4_d1)
        e5_d1 = self.conv5(e5_d1)
        d1 = self.fusion(torch.cat([d2_d1, d3_d1, d4_d1, e5_d1, e1_d1], 1))

        sup1 = self.out1(d1)
        sup2 = self.out2(d2)
        sup3 = self.out3(d3)
        sup4 = self.out4(d4)
        sup5 = self.out5(e5)

        sup1 = F.interpolate(sup1, x.shape[2:], mode='bilinear', align_corners=False)
        sup2 = F.interpolate(sup2, x.shape[2:], mode='bilinear', align_corners=False)
        sup3 = F.interpolate(sup3, x.shape[2:], mode='bilinear', align_corners=False)
        sup4 = F.interpolate(sup4, x.shape[2:], mode='bilinear', align_corners=False)
        sup5 = F.interpolate(sup5, x.shape[2:], mode='bilinear', align_corners=False)

        return sup1, sup2, sup3, sup4, sup5
    

    @staticmethod
    def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        g = get_gn_groups(out_channels)
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding="same", bias=False),
            nn.GroupNorm(g, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding="same", bias=False),
            nn.GroupNorm(g, out_channels),
            nn.ReLU(inplace=True), 
            nn.Dropout2d(0.2)
        )

    
if __name__ == "__main__":
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")
    model = Unet3plus(base_ch=64).to(device)
    
    # 仅做前向维度校验，无任何探测钩子
    dummy_x = torch.randn(1, 3, 512, 512).to(device)
    sup1, sup2, sup3, sup4, sup5 = model(dummy_x)
    
    print("==== 分割输出尺寸 ====")
    print(f"sup1:{sup1.shape}")
    print(f"sup2:{sup2.shape}")
    print(f"sup3:{sup3.shape}")
    print(f"sup4:{sup4.shape}")
    print(f"sup5:{sup5.shape}")


    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:.2f} M")
    print(f"可训练参数量: {trainable_params:.2f} M")
    
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
