import torch
import torch.nn as nn


@torch.no_grad()
def init_weights(init_type='xavier'):
    if init_type == 'xavier':
        init = nn.init.xavier_normal_
    elif init_type == 'he':
        init = nn.init.kaiming_normal_
    else:
        init = nn.init.orthogonal_

    def initializer(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            init(m.weight)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight, 1.0, 0.01)
            nn.init.zeros_(m.bias)

    return initializer


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.actv = nn.PReLU(out_channels)

    def forward(self, x):
        return self.actv(self.conv(x))


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, cat_channels, out_channels):
        super(UpsampleBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels + cat_channels, out_channels, 3, padding=1)
        self.conv_t = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)
        self.actv = nn.PReLU(out_channels)
        self.actv_t = nn.PReLU(in_channels)

    def forward(self, x):
        upsample, concat = x
        upsample = self.actv_t(self.conv_t(upsample))
        return self.actv(self.conv(torch.cat([concat, upsample], 1)))


class InputBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InputBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.actv_1 = nn.PReLU(out_channels)
        self.actv_2 = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.actv_1(self.conv_1(x))
        return self.actv_2(self.conv_2(x))


class OutputBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutputBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv_2 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.actv_1 = nn.PReLU(in_channels)
        self.actv_2 = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.actv_1(self.conv_1(x))
        return self.actv_2(self.conv_2(x))


class DenoisingBlock(nn.Module):
    def __init__(self, in_channels, inner_channels, out_channels):
        super(DenoisingBlock, self).__init__()
        self.conv_0 = nn.Conv2d(in_channels, inner_channels, 3, padding=1)
        self.conv_1 = nn.Conv2d(in_channels + inner_channels, inner_channels, 3, padding=1)
        self.conv_2 = nn.Conv2d(in_channels + 2 * inner_channels, inner_channels, 3, padding=1)
        self.conv_3 = nn.Conv2d(in_channels + 3 * inner_channels, out_channels, 3, padding=1)

        self.actv_0 = nn.PReLU(inner_channels)
        self.actv_1 = nn.PReLU(inner_channels)
        self.actv_2 = nn.PReLU(inner_channels)
        self.actv_3 = nn.PReLU(out_channels)

    def forward(self, x):
        out_0 = self.actv_0(self.conv_0(x))

        out_0 = torch.cat([x, out_0], 1)
        out_1 = self.actv_1(self.conv_1(out_0))

        out_1 = torch.cat([out_0, out_1], 1)
        out_2 = self.actv_2(self.conv_2(out_1))

        out_2 = torch.cat([out_1, out_2], 1)
        out_3 = self.actv_3(self.conv_3(out_2))

        return out_3 + x


class CompleteNet(nn.Module):
    """Denoising model."""
    def __init__(self, **kwargs):
        super().__init__()

        channels = kwargs['channels']
        filters_0 = kwargs['base filters']
        filters_1 = 2 * filters_0
        filters_2 = 4 * filters_0
        filters_3 = 8 * filters_0

        # Encoder:
        # Level 0:
        self.input_block = InputBlock(channels, filters_0)
        self.dn_block_0_0 = DenoisingBlock(filters_0, filters_0 // 2, filters_0)
        self.dn_block_0_1 = DenoisingBlock(filters_0, filters_0 // 2, filters_0)
        self.downsample_0 = DownsampleBlock(filters_0, filters_1)

        # Level 1:
        self.dn_block_1_0 = DenoisingBlock(filters_1, filters_1 // 2, filters_1)
        self.dn_block_1_1 = DenoisingBlock(filters_1, filters_1 // 2, filters_1)
        self.downsample_1 = DownsampleBlock(filters_1, filters_2)

        # Level 2:
        self.dn_block_2_0 = DenoisingBlock(filters_2, filters_2 // 2, filters_2)
        self.dn_block_2_1 = DenoisingBlock(filters_2, filters_2 // 2, filters_2)
        self.downsample_2 = DownsampleBlock(filters_2, filters_3)

        # Level 3 (Bottleneck)
        self.dn_block_3_0 = DenoisingBlock(filters_3, filters_3 // 2, filters_3)
        self.dn_block_3_1 = DenoisingBlock(filters_3, filters_3 // 2, filters_3)

        # Decoder
        # Level 2:
        self.upsample_2 = UpsampleBlock(filters_3, filters_2, filters_2)
        self.dn_block_4_0 = DenoisingBlock(filters_2, filters_2 // 2, filters_2)
        self.dn_block_4_1 = DenoisingBlock(filters_2, filters_2 // 2, filters_2)

        # Level 1:
        self.upsample_1 = UpsampleBlock(filters_2, filters_1, filters_1)
        self.dn_block_5_0 = DenoisingBlock(filters_1, filters_1 // 2, filters_1)
        self.dn_block_5_1 = DenoisingBlock(filters_1, filters_1 // 2, filters_1)

        # Level 0:
        self.upsample_0 = UpsampleBlock(filters_1, filters_0, filters_0)
        self.dn_block_6_0 = DenoisingBlock(filters_0, filters_0 // 2, filters_0)
        self.dn_block_6_1 = DenoisingBlock(filters_0, filters_0 // 2, filters_0)

        self.output_block = OutputBlock(filters_0, channels)

    def forward(self, inputs):
        out_0 = self.input_block(inputs)            # Level 0
        out_0 = self.dn_block_0_0(out_0)
        out_0 = self.dn_block_0_1(out_0)

        out_1 = self.downsample_0(out_0)            # Level 1
        out_1 = self.dn_block_1_0(out_1)
        out_1 = self.dn_block_1_1(out_1)

        out_2 = self.downsample_1(out_1)            # Level 2
        out_2 = self.dn_block_2_0(out_2)
        out_2 = self.dn_block_2_1(out_2)

        out_3 = self.downsample_2(out_2)            # Level 3 (Bottom)
        out_3 = self.dn_block_3_0(out_3)            # Bottleneck
        out_3 = self.dn_block_3_1(out_3)            # Bottleneck

        out_4 = self.upsample_2([out_3, out_2])     # Level 2
        out_4 = self.dn_block_4_0(out_4)
        out_4 = self.dn_block_4_1(out_4)

        out_5 = self.upsample_1([out_4, out_1])     # Level 1
        out_5 = self.dn_block_5_0(out_5)
        out_5 = self.dn_block_5_1(out_5)

        out_6 = self.upsample_0([out_5, out_0])     # Level 0
        out_6 = self.dn_block_6_0(out_6)
        out_6 = self.dn_block_6_1(out_6)

        return self.output_block(out_6) + inputs
