from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Optional

from core.config import (
    DataAugmentationConfig,
    DataConfig,
    EarlyStoppingConfig,
    GenericConfig,
    SchedulerConfig,
    TrainingConfig,
)


@dataclass
class ExperimentConfig:
    dataset: DataConfig
    train_augmentations: DataAugmentationConfig
    training: TrainingConfig
    scheduler: SchedulerConfig
    optimizer: GenericConfig

    model: Any

    name: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    seed: int = 42
    dry_run: bool = False
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
