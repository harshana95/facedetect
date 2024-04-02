import einops
import torch
from torch import nn

from modelzoo.Blocks import ConvBlock


class AutoEncoder(nn.Module):
    def __init__(self, n_channel_in=1, n_channel_out=1, activation='selu', residual=False):
        super(AutoEncoder, self).__init__()

        c0 = n_channel_in
        c1 = 64 if n_channel_in > 64 else 32  # sometimes we get input with channels >> 32 (e.g. 800)
        c2 = c1*2
        c3 = c2*2
        c4 = c3*2
        cn = n_channel_out

        self.down1 = nn.Conv2d(c1, c1, kernel_size=2, stride=2, groups=c1)
        self.down2 = nn.Conv2d(c2, c2, kernel_size=2, stride=2, groups=c2)
        self.down3 = nn.Conv2d(c3, c3, kernel_size=2, stride=2, groups=c3)
        self.down4 = nn.Conv2d(c4, c4, kernel_size=2, stride=2, groups=c4)

        self.down1.weight.data = 0.01 * self.down1.weight.data + 0.25
        self.down2.weight.data = 0.01 * self.down2.weight.data + 0.25
        self.down3.weight.data = 0.01 * self.down3.weight.data + 0.25
        self.down4.weight.data = 0.01 * self.down4.weight.data + 0.25

        self.down1.bias.data = 0.01 * self.down1.bias.data + 0
        self.down2.bias.data = 0.01 * self.down2.bias.data + 0
        self.down3.bias.data = 0.01 * self.down3.bias.data + 0
        self.down4.bias.data = 0.01 * self.down4.bias.data + 0

        self.up1 = nn.ConvTranspose2d(c4, c4, kernel_size=2, stride=2, groups=c4)
        self.up2 = nn.ConvTranspose2d(c3, c3, kernel_size=2, stride=2, groups=c3)
        self.up3 = nn.ConvTranspose2d(c2, c2, kernel_size=2, stride=2, groups=c2)
        self.up4 = nn.ConvTranspose2d(c1, c1, kernel_size=2, stride=2, groups=c1)

        self.up1.weight.data = 0.01 * self.up1.weight.data + 0.25
        self.up2.weight.data = 0.01 * self.up2.weight.data + 0.25
        self.up3.weight.data = 0.01 * self.up3.weight.data + 0.25
        self.up4.weight.data = 0.01 * self.up4.weight.data + 0.25

        self.up1.bias.data = 0.01 * self.up1.bias.data + 0
        self.up2.bias.data = 0.01 * self.up2.bias.data + 0
        self.up3.bias.data = 0.01 * self.up3.bias.data + 0
        self.up4.bias.data = 0.01 * self.up4.bias.data + 0

        self.conv1 = ConvBlock(c0, c1, residual=residual, activation=activation)
        self.conv2 = ConvBlock(c1, c2, residual=residual, activation=activation)
        self.conv3 = ConvBlock(c2, c3, residual=residual, activation=activation)
        self.conv4 = ConvBlock(c3, c4, residual=residual, activation=activation)

        self.conv5 = ConvBlock(c4, c4, residual=residual, activation=activation)

        self.conv6 = ConvBlock(c4, c3, residual=residual, activation=activation)
        self.conv7 = ConvBlock(c3, c2, residual=residual, activation=activation)
        self.conv8 = ConvBlock(c2, c1, residual=residual, activation=activation)
        self.conv9 = ConvBlock(c1, cn, residual=residual, activation=activation)

    def forward(self, x):
        if len(x.shape)==5:
            x = einops.rearrange(x, 'b n c h w -> b (n c) h w')
        c0 = x
        c1 = self.conv1(x)
        x = self.down1(c1)
        c2 = self.conv2(x)
        x = self.down2(c2)
        c3 = self.conv3(x)
        x = self.down3(c3)
        c4 = self.conv4(x)
        x = self.down4(c4)
        lat = self.conv5(x)

        x = self.up1(lat)
        x = self.conv6(x)
        x = self.up2(x)
        x = self.conv7(x)
        x = self.up3(x)
        x = self.conv8(x)
        x = self.up4(x)
        x = self.conv9(x)

        return x, lat


class AutoEncoder2(nn.Module):
    def __init__(self, n_channel_in=1, n_channel_out=1, activation='selu', residual=False):
        super(AutoEncoder2, self).__init__()

        c0 = n_channel_in
        c1 = 64 if n_channel_in > 64 else 32  # sometimes we get input with channels >> 32 (e.g. 800)
        c2 = c1*2
        c3 = c2*2
        c4 = c3*2
        c5 = c4*2
        cn = n_channel_out

        self.down1 = nn.Conv2d(c1, c1, kernel_size=2, stride=2, groups=c1)
        self.down2 = nn.Conv2d(c2, c2, kernel_size=2, stride=2, groups=c2)
        self.down3 = nn.Conv2d(c3, c3, kernel_size=2, stride=2, groups=c3)
        self.down4 = nn.Conv2d(c4, c4, kernel_size=2, stride=2, groups=c4)
        self.down5 = nn.Conv2d(c5, c5, kernel_size=2, stride=2, groups=c5)

        self.down1.weight.data = 0.01 * self.down1.weight.data + 0.25
        self.down2.weight.data = 0.01 * self.down2.weight.data + 0.25
        self.down3.weight.data = 0.01 * self.down3.weight.data + 0.25
        self.down4.weight.data = 0.01 * self.down4.weight.data + 0.25
        self.down5.weight.data = 0.01 * self.down5.weight.data + 0.25

        self.down1.bias.data = 0.01 * self.down1.bias.data + 0
        self.down2.bias.data = 0.01 * self.down2.bias.data + 0
        self.down3.bias.data = 0.01 * self.down3.bias.data + 0
        self.down4.bias.data = 0.01 * self.down4.bias.data + 0
        self.down5.bias.data = 0.01 * self.down5.bias.data + 0

        self.up0 = nn.ConvTranspose2d(c5, c5, kernel_size=2, stride=2, groups=c5)
        self.up1 = nn.ConvTranspose2d(c4, c4, kernel_size=2, stride=2, groups=c4)
        self.up2 = nn.ConvTranspose2d(c3, c3, kernel_size=2, stride=2, groups=c3)
        self.up3 = nn.ConvTranspose2d(c2, c2, kernel_size=2, stride=2, groups=c2)
        self.up4 = nn.ConvTranspose2d(c1, c1, kernel_size=2, stride=2, groups=c1)

        self.up0.weight.data = 0.01 * self.up0.weight.data + 0.25
        self.up1.weight.data = 0.01 * self.up1.weight.data + 0.25
        self.up2.weight.data = 0.01 * self.up2.weight.data + 0.25
        self.up3.weight.data = 0.01 * self.up3.weight.data + 0.25
        self.up4.weight.data = 0.01 * self.up4.weight.data + 0.25

        self.up0.bias.data = 0.01 * self.up0.bias.data + 0
        self.up1.bias.data = 0.01 * self.up1.bias.data + 0
        self.up2.bias.data = 0.01 * self.up2.bias.data + 0
        self.up3.bias.data = 0.01 * self.up3.bias.data + 0
        self.up4.bias.data = 0.01 * self.up4.bias.data + 0

        self.conv1 = ConvBlock(c0, c1, residual=residual, activation=activation)
        self.conv2 = ConvBlock(c1, c2, residual=residual, activation=activation)
        self.conv3 = ConvBlock(c2, c3, residual=residual, activation=activation)
        self.conv4 = ConvBlock(c3, c4, residual=residual, activation=activation)

        self.conv50 = ConvBlock(c4, c5, residual=residual, activation=activation)
        self.conv51 = ConvBlock(c5, c5, residual=residual, activation=activation)
        self.conv52 = ConvBlock(c5, c4, residual=residual, activation=activation)

        self.conv6 = ConvBlock(c4, c3, residual=residual, activation=activation)
        self.conv7 = ConvBlock(c3, c2, residual=residual, activation=activation)
        self.conv8 = ConvBlock(c2, c1, residual=residual, activation=activation)
        self.conv9 = ConvBlock(c1, cn, residual=residual, activation=activation)

    def forward(self, x):
        if len(x.shape)==5:
            x = einops.rearrange(x, 'b n c h w -> b (n c) h w')
        c0 = x
        c1 = self.conv1(x)
        x = self.down1(c1)
        c2 = self.conv2(x)
        x = self.down2(c2)
        c3 = self.conv3(x)
        x = self.down3(c3)
        c4 = self.conv4(x)
        x = self.down4(c4)
        c5 = self.conv50(x)
        x = self.down5(c5)
        lat = self.conv51(x)

        x = self.up0(lat)
        x = self.conv52(x)
        x = self.up1(x)
        x = self.conv6(x)
        x = self.up2(x)
        x = self.conv7(x)
        x = self.up3(x)
        x = self.conv8(x)
        x = self.up4(x)
        x = self.conv9(x)

        return x, lat