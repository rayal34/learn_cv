from dataclasses import asdict, dataclass, field

from base.config import ConvSpec, DataConfig, TrainingConfig
from omegaconf import MISSING
from utils import train_utils


@dataclass
class DataAugmentationConfig:
    h_flip_prob: float = 0.5
    rotate_range: list[float] = field(default_factory=lambda: [-5.0, 5.0])
    crop_padding: int = 2


@dataclass
class ModelConfig:
    conv_layers: list[ConvSpec] = MISSING
    fc_hidden: tuple[int, ...] = MISSING
    dropout: float | None = MISSING


@dataclass
class ExperimentConfig:
    name: str = field(default_factory=lambda: train_utils.generate_default_exp_name())
    seed: int = 42
    dataset: DataConfig = MISSING
    data_augmentations: DataAugmentationConfig = field(
        default_factory=DataAugmentationConfig
    )
    training: TrainingConfig = MISSING
    model: ModelConfig = field(default_factory=ModelConfig)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "seed": self.seed,
            "dataset": asdict(self.dataset),
            "training": asdict(self.training),
            "data_augmentations": asdict(self.data_augmentations),
            "model": asdict(self.model),
        }
