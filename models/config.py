from dataclasses import dataclass
from typing import Optional


@dataclass
class MaxPoolSpec:
    kernel_size: int
    stride: Optional[int] = None
    padding: int = 0


@dataclass
class ConvSpec:
    out_channels: int
    kernel_size: int
    padding: int
    pool: int | None
    stride: int


@dataclass
class SimpleCNNModelConfig:
    conv_layers: list[ConvSpec]
    fc_hidden: list[int]
    dropout: Optional[float]


@dataclass
class ResNetStemConfig:
    conv: ConvSpec
    maxpool: Optional[MaxPoolSpec] = None


@dataclass
class ResNetBlockConfig:
    out_channels: int
    kernel_size: int
    padding: int
    stride: int
    blocks: int


@dataclass
class ResNetShallowModelConfig:
    stem: ResNetStemConfig
    layers: list[ResNetBlockConfig]
