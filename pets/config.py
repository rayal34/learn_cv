from dataclasses import asdict, dataclass, field
from typing import Optional

from core import config
from core.config import (
    DataAugmentationConfig,
    ExperimentConfig,
    GenericConfig,
)


@dataclass(kw_only=True)
class DataConfig(config.DataConfig):
    torch_home: str

    train_path: str = field(init=False, default="${.data_path}/train.npz")
    val_path: str = field(init=False, default="${.data_path}/val.npz")
    test_path: str = field(init=False, default="${.data_path}/test.npz")

    val_ratio: float = 0.2


@dataclass(kw_only=True)
class ExperimentConfig(ExperimentConfig):
    label: str
    dataset: DataConfig
    train_augmentations: DataAugmentationConfig
    fine_tune_freezing_strategy: Optional[GenericConfig] = None

    def to_dict(self) -> dict:

        return {
            "name": self.name,
            "seed": self.seed,
            "dry_run": self.dry_run,
            "dataset": asdict(self.dataset),
            "training": asdict(self.training),
            "train_augmentations": asdict(self.train_augmentations),
            "model": asdict(self.model) if self.model is not None else None,
            "scheduler": asdict(self.scheduler),
            "optimizer": asdict(self.optimizer),
            "early_stopping": asdict(self.early_stopping)
            if self.early_stopping is not None
            else None,
            "fine_tune_freezing_strategy": asdict(self.fine_tune_freezing_strategy)
            if self.fine_tune_freezing_strategy is not None
            else None,
        }


#
