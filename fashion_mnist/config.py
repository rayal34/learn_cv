from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Optional

from core.config import (
    DataConfig,
    EarlyStoppingConfig,
    GenericConfig,
    SchedulerConfig,
    TrainingConfig,
)


@dataclass
class DataAugmentationConfig:
    h_flip_prob: float = 0.5

    rotate_range: list[float] = field(default_factory=lambda: [-5.0, 5.0])

    crop_padding: int = 2


@dataclass
class ExperimentConfig:
    dataset: DataConfig
    training: TrainingConfig
    model: Any
    optimizer: GenericConfig
    scheduler: SchedulerConfig
    name: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    seed: int = 42
    dry_run: bool = False
    data_augmentations: DataAugmentationConfig = field(
        default_factory=DataAugmentationConfig
    )

    early_stopping: Optional[EarlyStoppingConfig] = None

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
            "early_stopping": asdict(self.early_stopping)
            if self.early_stopping is not None
            else None,
        }
