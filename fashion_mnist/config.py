import os
from dataclasses import asdict, dataclass, field

from utils import train_utils


@dataclass
class DataConfig:
    root: str = "/Volumes/satechi/ml_projects/FashionMNIST"
    data_path: str = field(init=False)
    model_path: str = field(init=False)
    experiment_path: str = field(init=False)

    train_images_filename: str = "train-images-idx3-ubyte.gz"
    train_labels_filename: str = "train-labels-idx1-ubyte.gz"
    test_images_filename: str = "t10k-images-idx3-ubyte.gz"
    test_labels_filename: str = "t10k-labels-idx1-ubyte.gz"

    def __post_init__(self):
        self.data_path = os.path.join(self.root, "data")
        self.model_path = os.path.join(self.root, "models")
        self.experiment_path = os.path.join(self.root, "experiments")


@dataclass
class TrainingConfig:
    learning_rate: float = 0.002
    batch_size: int = 256
    early_stopping_patience: int = 20
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    num_epochs: int = 100

    weight_decay: float = 0.001
    early_stopping: bool = True


@dataclass
class DataAugmentationConfig:
    h_flip_prob: float = 0.5
    rotate_range: list[float] = field(default_factory=lambda: [-5.0, 5.0])
    crop_padding: int = 2


@dataclass
class ConvSpec:
    out_channels: int
    kernel_size: int = 3
    padding: int = 0
    pool: int | None = 2
    stride: int = 1


@dataclass
class ModelConfig:
    conv_layers: list[ConvSpec] = field(
        default_factory=lambda: [
            ConvSpec(64, 3, pool=2),
        ]
    )
    fc_hidden: tuple[int, ...] = (64,)
    dropout: float | None = 0.3


@dataclass
class ExperimentConfig:
    name: str = field(default_factory=lambda: train_utils.generate_default_exp_name())
    seed: int = 42
    dataset: DataConfig = field(default_factory=DataConfig)
    data_augmentations: DataAugmentationConfig = field(
        default_factory=DataAugmentationConfig
    )
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "seed": self.seed,
            "dataset": asdict(self.dataset),
            "training": asdict(self.training),
            "data_augmentations": asdict(self.data_augmentations),
            "model": asdict(self.model),
        }
