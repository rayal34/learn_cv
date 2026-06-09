from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from base.config import DataConfig, GenericConfig, TrainingConfig
from omegaconf import MISSING


@dataclass
class DataAugmentationConfig:
    h_flip_prob: float = 0.5

    rotate_range: list[float] = field(default_factory=lambda: [-5.0, 5.0])

    crop_padding: int = 2


@dataclass
class ExperimentConfig:
    name: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    seed: int = 42
    dry_run: bool = False
    dataset: DataConfig = MISSING
    data_augmentations: DataAugmentationConfig = field(
        default_factory=DataAugmentationConfig
    )
    training: TrainingConfig = MISSING
    model: Any = MISSING
    scheduler: GenericConfig = MISSING
    optimizer: GenericConfig = MISSING

    def to_dict(self) -> dict:

        return {
            "name": self.name,
            "seed": self.seed,
            "dry_run": self.dry_run,
            "dataset": asdict(self.dataset),
            "training": asdict(self.training),
            "data_augmentations": asdict(self.data_augmentations),
            "model": asdict(self.model),
            "scheduler": asdict(self.scheduler),
            "optimizer": asdict(self.optimizer),
        }
