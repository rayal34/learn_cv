from dataclasses import asdict, dataclass, field
from typing import Optional

from core.config import DataConfig as CoreDataConfig
from core.config import ExperimentConfig as CoreExperimentConfig


@dataclass
class DataConfig(CoreDataConfig):
    train_images_filename: Optional[str] = None
    train_labels_filename: Optional[str] = None
    test_images_filename: Optional[str] = None
    test_labels_filename: Optional[str] = None


@dataclass
class DataAugmentationConfig:
    h_flip_prob: float = 0.5

    rotate_range: list[float] = field(default_factory=lambda: [-5.0, 5.0])

    crop_padding: int = 2


@dataclass
class ExperimentConfig(CoreExperimentConfig):
    dataset: DataConfig
    data_augmentations: DataAugmentationConfig = field(
        default_factory=DataAugmentationConfig
    )

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
