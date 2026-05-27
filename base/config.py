import os
from dataclasses import asdict, dataclass, field
from typing import Any

from omegaconf import MISSING
from utils import train_utils


@dataclass
class DataConfig:
    root: str

    train_images_filename: str | None = None
    train_labels_filename: str | None = None
    test_images_filename: str | None = None
    test_labels_filename: str | None = None

    data_path: str = field(init=False, default="${.root}/data")
    model_path: str = field(init=False, default="${.root}/models")
    experiment_path: str = field(init=False, default="${.root}/experiments")

    def __post_init__(self):
        self.data_path = os.path.join(self.root, "data")
        self.model_path = os.path.join(self.root, "models")
        self.experiment_path = os.path.join(self.root, "experiments")


@dataclass
class TrainingConfig:
    learning_rate: float
    batch_size: int
    early_stopping_patience: int
    scheduler_patience: int
    scheduler_factor: float
    num_epochs: int

    weight_decay: float
    early_stopping: bool


@dataclass
class DataAugmentationConfig:
    h_flip_prob: float = 0.5
    rotate_range: list[float] = field(default_factory=lambda: [-5.0, 5.0])
    crop_padding: int = 2


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
