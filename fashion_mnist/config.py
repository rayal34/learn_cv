from dataclasses import asdict, dataclass, field

from utils import train_utils


@dataclass
class DataConfig:
    data_path: str = "/Volumes/satechi/ml_projects/FashionMNIST/data"
    train_images_filename: str = "train-images-idx3-ubyte.gz"
    train_labels_filename: str = "train-labels-idx1-ubyte.gz"
    test_images_filename: str = "t10k-images-idx3-ubyte.gz"
    test_labels_filename: str = "t10k-labels-idx1-ubyte.gz"

    model_path: str = "/Volumes/satechi/ml_projects/FashionMNIST/models/"

    experiment_path: str = "/Volumes/satechi/ml_projects/FashionMNIST/experiments/"


@dataclass
class TrainingConfig:
    learning_rate: float = 0.002
    batch_size: int = 256
    early_stopping_patience: int = 20
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    num_epochs: int = 100

    weight_decay: float = 0.001

    input_height = 28
    input_width = 28
    input_channels = 1

    h_flip_prob = 0.5
    v_flip_prob = 0.0
    rotate_range = [-5.0, 5.0]

    mean: float = 0.286
    std: float = 0.353


@dataclass
class ConvSpec:
    out_channels: int
    kernel_size: int = 3
    padding: int = 0
    pool: int | None = 2
    stride: int = 1


@dataclass
class ModelConfig:
    num_classes: int = 10
    conv_layers: list[ConvSpec] = field(
        default_factory=lambda: [
            ConvSpec(64, 3, pool=None),
            ConvSpec(128, 3, pool=2),
            ConvSpec(256, 3, pool=2),
            ConvSpec(512, 3, pool=2),
        ]
    )
    fc_hidden: tuple[int, ...] = (256, 128, 64)
    dropout: float | None = 0.3


@dataclass
class ExperimentConfig:
    name: str = field(default_factory=lambda: train_utils.generate_default_exp_name())
    seed: int = 42
    dataset: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "seed": self.seed,
            "dataset": asdict(self.dataset),
            "training": asdict(self.training),
            "model": asdict(self.model),
        }
