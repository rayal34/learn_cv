from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Optional

from base.config import (
    DataAugmentationConfig,
    DataConfig,
    EarlyStoppingConfig,
    GenericConfig,
    SchedulerConfig,
    TrainingConfig,
)
from omegaconf import MISSING


@dataclass
class ExperimentConfig:
    name: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    seed: int = 42
    dry_run: bool = False
    dataset: DataConfig = MISSING
    train_augmentations: DataAugmentationConfig = MISSING
    training: TrainingConfig = MISSING
    model: Any = MISSING

    scheduler: SchedulerConfig = MISSING

    optimizer: GenericConfig = MISSING

    early_stopping: Optional[EarlyStoppingConfig] = None

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
            "early_stopping": asdict(self.early_stopping)
            if self.early_stopping is not None
            else None,
        }
