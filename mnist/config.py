from dataclasses import asdict, dataclass, field


@dataclass
class DataConfig:
    data_path: str = "/Volumes/satechi/ml_projects/mnist"

    model_path: str = "/Volumes/satechi/ml_projects/mnist/models/"

    experiment_path: str = "/Volumes/satechi/ml_projects/mnist/experiments/"


@dataclass
class TrainingConfig:
    learning_rate: float = 0.01
    batch_size: int = 256
    early_stopping_patience: int = 20
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    num_epochs: int = 100


@dataclass
class ConvSpec:
    out_channels: int
    kernel_size: int = 3
    padding: int = 0
    pool: int | None = 2


@dataclass
class ModelConfig:
    num_classes: int = 10
    conv_layers: list[ConvSpec] = field(
        default_factory=lambda: [
            ConvSpec(256, 7, pool=2),
            ConvSpec(32, 5, pool=2),
        ]
    )
    fc_hidden: tuple[int, ...] = (128,)
    dropout: float | None = 0.5


@dataclass
class ExperimentConfig:
    name: str = "default"
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
