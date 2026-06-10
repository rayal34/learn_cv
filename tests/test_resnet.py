import torch
import torch.nn as nn
from models.config import (
    ConvSpec,
    MaxPoolSpec,
    ResNetBlockConfig,
    ResNetShallowModelConfig,
    ResNetStemConfig,
)
from models.resnet import Block, ResNetShallow, Stem


def test_block_no_downsample():
    # Instantiating Block without downsample
    block = Block(in_channels=16, out_channels=16, stride=1)
    x = torch.randn(2, 16, 8, 8)
    out = block(x)
    assert out.shape == (2, 16, 8, 8)


def test_block_with_downsample():
    # Instantiating Block with downsample
    downsample = nn.Sequential(
        nn.Conv2d(16, 32, kernel_size=1, stride=2, bias=False),
        nn.BatchNorm2d(32),
    )
    block = Block(in_channels=16, out_channels=32, stride=2, downsample=downsample)
    x = torch.randn(2, 16, 8, 8)
    out = block(x)
    assert out.shape == (2, 32, 4, 4)


def test_stem_no_maxpool():
    conv_spec = ConvSpec(out_channels=16, kernel_size=3, padding=1, pool=None, stride=1)
    stem_config = ResNetStemConfig(conv=conv_spec, maxpool=None)
    stem = Stem(in_channels=3, model_config=stem_config)
    x = torch.randn(2, 3, 32, 32)
    out = stem(x)
    assert out.shape == (2, 16, 32, 32)


def test_stem_with_maxpool():
    conv_spec = ConvSpec(out_channels=16, kernel_size=3, padding=1, pool=None, stride=1)
    maxpool_spec = MaxPoolSpec(kernel_size=2, stride=2, padding=0)
    stem_config = ResNetStemConfig(conv=conv_spec, maxpool=maxpool_spec)
    stem = Stem(in_channels=3, model_config=stem_config)
    x = torch.randn(2, 3, 32, 32)
    out = stem(x)
    assert out.shape == (2, 16, 16, 16)


def test_resnet18_forward():
    stem_conv = ConvSpec(out_channels=16, kernel_size=3, padding=1, pool=None, stride=1)
    stem_config = ResNetStemConfig(conv=stem_conv, maxpool=None)

    layers_config = [
        ResNetBlockConfig(
            out_channels=16, kernel_size=3, padding=1, stride=1, blocks=2
        ),
        ResNetBlockConfig(
            out_channels=32, kernel_size=3, padding=1, stride=2, blocks=2
        ),
    ]

    model_config = ResNetShallowModelConfig(stem=stem_config, layers=layers_config)

    # In ResNet18, img_size is used as the number of input channels (e.g. 3)
    model = ResNetShallow(img_size=3, n_classes=10, model_config=model_config)

    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    assert out.shape == (4, 10)
