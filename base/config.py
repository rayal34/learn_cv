import os
from dataclasses import dataclass, field

from omegaconf import MISSING


@dataclass
class DataConfig:
    root: str

    train_images_filename: str
    train_labels_filename: str
    test_images_filename: str
    test_labels_filename: str

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
class ConvSpec:
    out_channels: int
    kernel_size: int
    padding: int
    pool: int | None
    stride: int


@dataclass
class SimpleCNNModelConfig:
    conv_layers: list[ConvSpec] = MISSING
    fc_hidden: list[int] = MISSING
    dropout: float | None = MISSING

