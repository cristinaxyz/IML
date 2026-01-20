import torch
import torch.nn as nn
from torchvision.ops import SqueezeExcitation

KERNEL_SIZE = 3
STRIDE = 1

# This is not stated in the paper but this is the default from the SE paper.
SE_REDUCTION = 16


class ConvolutionBlock(nn.Module):
    """
    Implements the red (dilation=2, se_module=False) and black (defaults) blocks
        from the paper (Figure 2).
    """

    def __init__(
        self, in_channels: int, out_channels: int, dilation=1, se_module=True
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=KERNEL_SIZE,
            stride=STRIDE,
            dilation=dilation,
            padding=dilation,  # to maintain spatial dimensions
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if se_module:
            self.se = SqueezeExcitation(
                out_channels, out_channels // SE_REDUCTION
            )
        else:
            self.se = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.se is not None:
            x = self.se(x)
        return x


class CNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            # 224x224x64
            ConvolutionBlock(3, 64, dilation=2, se_module=False),
            ConvolutionBlock(64, 64),
            # 112x112x128
            nn.MaxPool2d(2, 2),
            ConvolutionBlock(64, 128),
            ConvolutionBlock(128, 128),
            # 56x56x256
            nn.MaxPool2d(2, 2),
            ConvolutionBlock(128, 256),
            ConvolutionBlock(256, 256),
            ConvolutionBlock(256, 256),
            # 28x28x512
            nn.MaxPool2d(2, 2),
            ConvolutionBlock(256, 512),
            ConvolutionBlock(512, 512),
            ConvolutionBlock(512, 512),
            # 14x14x1024
            nn.MaxPool2d(2, 2),
            ConvolutionBlock(512, 1024),
            ConvolutionBlock(1024, 1024),
            # 7x7x1024
            nn.MaxPool2d(2, 2),
            ConvolutionBlock(1024, 1024, dilation=2, se_module=False),
        )
        # 1x1x1024
        self.gap = nn.AdaptiveAvgPool2d(1)
        # 1x1xnum_classes
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.network(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = CNN(num_classes=10)
