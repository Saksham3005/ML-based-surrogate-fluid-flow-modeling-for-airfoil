import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        return x, self.pool(x)
    
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        """
        F_g   : channels of decoder feature (gating signal)
        F_l   : channels of skip connection
        F_int : intermediate channels
        """
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.InstanceNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.InstanceNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        g: decoder feature
        x: skip connection
        """
        psi = self.relu(self.W_g(g) + self.W_x(x))
        alpha = self.psi(psi)
        return x * alpha



# class Up(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
#         self.conv = ConvBlock(in_ch, out_ch)

#     def forward(self, x, skip):
#         x = self.up(x)
#         x = torch.cat([x, skip], dim=1)
#         return self.conv(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

        self.att = AttentionGate(
            F_g=out_ch,      # decoder channels
            F_l=out_ch,      # skip channels
            F_int=out_ch // 2
        )

        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        skip = self.att(x, skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class AirfoilUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, base=64):
        super().__init__()

        self.d1 = Down(in_channels, base)
        self.d2 = Down(base, base*2)
        self.d3 = Down(base*2, base*4)
        self.d4 = Down(base*4, base*8)

        self.bottleneck = ConvBlock(base*8, base*16)

        self.u4 = Up(base*16, base*8)
        self.u3 = Up(base*8, base*4)
        self.u2 = Up(base*4, base*2)
        self.u1 = Up(base*2, base)

        self.out = nn.Conv2d(base, out_channels, kernel_size=1)

    def forward(self, x):
        s1, x = self.d1(x)
        s2, x = self.d2(x)
        s3, x = self.d3(x)
        s4, x = self.d4(x)

        x = self.bottleneck(x)

        x = self.u4(x, s4)
        x = self.u3(x, s3)
        x = self.u2(x, s2)
        x = self.u1(x, s1)

        return self.out(x)
    
if __name__ == "__main__":
    model = AirfoilUNet(in_channels=1, out_channels=3, base=64)
    x = torch.randn(2, 1, 256, 256)
    y = model(x)
    print(y.shape)  # Expected: (2, 3, 256, 256)