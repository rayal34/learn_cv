import os
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    root: str
    num_workers: int
    pin_memory: bool

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
