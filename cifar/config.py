from dataclasses import asdict, dataclass, field
from typing import Any

from base.config import (
    DataAugmentationConfig,
    DataConfig,
    GenericConfig,
    TrainingConfig,
)
from omegaconf import MISSING
from utils import train_utils


@dataclass
class ExperimentConfig:
    name: str = field(default_factory=lambda: train_utils.generate_default_exp_name())
    seed: int = 42
    dry_run: bool = False
    dataset: DataConfig = MISSING
    train_augmentations: DataAugmentationConfig = MISSING
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
            "train_augmentations": asdict(self.train_augmentations),
            "model": asdict(self.model),
            "scheduler": asdict(self.scheduler),
            "optimizer": asdict(self.optimizer),
        }
