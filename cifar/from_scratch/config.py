from dataclasses import asdict, dataclass
from typing import Optional

from core.config import (
    DataAugmentationConfig,
    EarlyStoppingConfig,
)
from core.config import (
    ExperimentConfig as CoreExperimentConfig,
)


@dataclass(kw_only=True)
class ExperimentConfig(CoreExperimentConfig):
    train_augmentations: DataAugmentationConfig
    early_stopping: Optional[EarlyStoppingConfig] = None

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
        }
