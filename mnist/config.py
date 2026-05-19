from dataclasses import asdict, dataclass, field

from base.config import DataConfig, SimpleCNNModelConfig, TrainingConfig
from omegaconf import MISSING
from utils import train_utils


@dataclass
class ExperimentConfig:
    name: str = field(default_factory=lambda: train_utils.generate_default_exp_name())
    seed: int = 42
    dataset: DataConfig = MISSING
    training: TrainingConfig = MISSING
    model: SimpleCNNModelConfig = field(default_factory=SimpleCNNModelConfig)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "seed": self.seed,
            "dataset": asdict(self.dataset),
            "training": asdict(self.training),
            "model": asdict(self.model),
        }
