import torch
import torch.nn as nn

from models import config


class Block(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ):

        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.shortcut = nn.Sequential()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Stem(nn.Module):
    def __init__(self, in_channels: int, model_config: config.ResNetStemConfig):
        super().__init__()
        self.config = model_config
        self.conv = nn.Conv2d(
            in_channels,
            out_channels=model_config.conv.out_channels,
            kernel_size=model_config.conv.kernel_size,
            stride=model_config.conv.stride,
            padding=model_config.conv.padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(model_config.conv.out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = None
        if model_config.maxpool is not None:
            self.maxpool = nn.MaxPool2d(
                kernel_size=model_config.maxpool.kernel_size,
                stride=model_config.maxpool.stride,
                padding=model_config.maxpool.padding,
            )

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        if self.maxpool is not None:
            out = self.maxpool(out)
        return out


class ResNetShallow(nn.Module):
    def __init__(
        self,
        img_size: int,
        n_classes: int,
        model_config: config.ResNetShallowModelConfig,
    ):
        super().__init__()

        self.in_channels = model_config.stem.conv.out_channels
        self.stem = Stem(img_size, model_config.stem)

        layers = []
        for layer in model_config.layers:
            layers.append(
                self._make_layer(
                    Block,
                    out_channels=layer.out_channels,
                    blocks=layer.blocks,
                    stride=layer.stride,
                )
            )

        self.layers = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(model_config.layers[-1].out_channels, n_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None

        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(
            block(
                in_channels=self.in_channels,
                out_channels=out_channels,
                stride=stride,
                downsample=downsample,
            )
        )
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)
