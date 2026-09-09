import torch.nn as nn


class BasicConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=1)
        self.normalize = nn.GroupNorm(out_channels // 16, out_channels)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activate(self.normalize(self.conv1(x)))


class BasicResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, shared_group_norm=False):
        super().__init__()
        assert out_channels % 16 == 0
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=1)
        self.activate = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=1)
        self.shared_group_norm = shared_group_norm
        if shared_group_norm:
            # Early experiment checkpoints reused one GroupNorm module after both
            # convolutions. Preserve their parameter names and exact computation.
            self.normalize = nn.GroupNorm(out_channels // 16, out_channels)
        else:
            self.normalize1 = nn.GroupNorm(out_channels // 16, out_channels)
            self.normalize2 = nn.GroupNorm(out_channels // 16, out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(max(1, out_channels // 16), out_channels),
            )

    def forward(self, x):
        if self.shared_group_norm:
            out = self.activate(self.normalize(self.conv1(x)))
            out = self.normalize(self.conv2(out))
        else:
            out = self.activate(self.normalize1(self.conv1(x)))
            out = self.normalize2(self.conv2(out))

        residual = self.downsample(x) if self.downsample is not None else x
        return self.activate(out + residual)


class ResidualBlockA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(out_channels // 16, out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlockB(nn.Module):
    def __init__(self, in_channels, out_channels, shared_group_norm=False):
        super().__init__()
        self.res1 = BasicResidualBlock(
            in_channels, out_channels, kernel_size=3, stride=1, shared_group_norm=shared_group_norm
        )
        self.res2 = BasicResidualBlock(
            out_channels, out_channels, kernel_size=3, stride=1, shared_group_norm=shared_group_norm
        )

    def forward(self, x):
        return self.res2(self.res1(x))


class ResidualBlockC(nn.Module):
    def __init__(self, in_channels, out_channels, shared_group_norm=False):
        super().__init__()
        self.res1 = BasicResidualBlock(
            in_channels, out_channels, kernel_size=3, stride=2, shared_group_norm=shared_group_norm
        )
        self.res2 = BasicResidualBlock(
            out_channels, out_channels, kernel_size=3, stride=1, shared_group_norm=shared_group_norm
        )

    def forward(self, x):
        return self.res2(self.res1(x))


class ResidualBlockD(ResidualBlockC):
    pass


class ResidualBlockE(ResidualBlockC):
    pass


class OursDiffusionRgbEncoder(nn.Module):
    """Extract multi-scale RGB feature maps used by SCDP."""

    def __init__(self, config=None):
        super().__init__()
        shared_group_norm = bool(
            getattr(config, "use_shared_group_norm_in_residual_blocks", False)
        )
        self.blockA = ResidualBlockA(in_channels=3, out_channels=64)
        self.blockB = ResidualBlockB(
            in_channels=64,
            out_channels=64,
            shared_group_norm=shared_group_norm,
        )
        self.blockC = ResidualBlockC(
            in_channels=64,
            out_channels=128,
            shared_group_norm=shared_group_norm,
        )
        self.blockD = ResidualBlockD(
            in_channels=128,
            out_channels=256,
            shared_group_norm=shared_group_norm,
        )
        self.blockE = ResidualBlockE(
            in_channels=256,
            out_channels=512,
            shared_group_norm=shared_group_norm,
        )

        self.convA = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.convB = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.convC = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.convD = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.convE = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.channel = 3
        self.feature_dim = 64 * 5

    def forward(self, x):
        """Return five feature maps for an ``(N, 3, H, W)`` image batch."""
        if x.shape[1] != self.channel:
            raise ValueError(f"Expected {self.channel} RGB channels, got shape {tuple(x.shape)}.")

        feat_a = self.blockA(x)
        feat_b = self.blockB(feat_a)
        feat_c = self.blockC(feat_b)
        feat_d = self.blockD(feat_c)
        feat_e = self.blockE(feat_d)

        return [
            self.convA(feat_a),
            self.convB(feat_b),
            self.convC(feat_c),
            self.convD(feat_d),
            self.convE(feat_e),
        ]
