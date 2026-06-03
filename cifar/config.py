from dataclasses import asdict, dataclass, field
from typing import Any

from base.config import DataConfig, TrainingConfig
from omegaconf import MISSING
from utils import train_utils


@dataclass
class DataAugmentationConfig:
    h_flip_prob: float = 0.5

    rotate_range: list[float] = field(default_factory=lambda: [-5.0, 5.0])

    crop_padding: int = 2

    mixup_alpha: float = 0.0


@dataclass
class ExperimentConfig:
    name: str = field(default_factory=lambda: train_utils.generate_default_exp_name())
    seed: int = 42
    dry_run: bool = False
    dataset: DataConfig = MISSING
    data_augmentations: DataAugmentationConfig = field(
        default_factory=DataAugmentationConfig
    )
    training: TrainingConfig = MISSING
    model: Any = MISSING

    def to_dict(self) -> dict:

        return {
            "name": self.name,
            "seed": self.seed,
            "dry_run": self.dry_run,
            "dataset": asdict(self.dataset),
            "training": asdict(self.training),
            "data_augmentations": asdict(self.data_augmentations),
            "model": asdict(self.model),
        }
