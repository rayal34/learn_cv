from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING


@dataclass
class MaxPoolSpec:
    kernel_size: int
    stride: int | None = None
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
    conv_layers: list[ConvSpec] = MISSING
    fc_hidden: list[int] = MISSING
    dropout: float | None = MISSING


@dataclass
class ResNetStemConfig:
    conv: ConvSpec = MISSING
    maxpool: Optional[MaxPoolSpec] = None


@dataclass
class ResNetBlockConfig:
    out_channels: int
    kernel_size: int
    padding: int
    stride: int
    blocks: int


@dataclass
class ResNet18ModelConfig:
    stem: ResNetStemConfig = MISSING
    layers: list[ResNetBlockConfig] = MISSING
