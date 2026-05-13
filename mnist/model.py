import config
import torch
import torch.nn as nn
import torch.nn.functional as F
import train_utils


class EarlyStoppingWithCheckpoint:
    def __init__(
        self,
        model_path: str,
        model_name: str,
        patience: int = 5,
        min_delta: float = 0.0,
        higher_is_better: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.model_path = model_path
        self.model_name = model_name
        self.higher_is_better = higher_is_better

        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float, model: torch.nn.Module):
        if self.best_score is None:
            # initialize the best score to the current score
            self.best_score = score

        if self.higher_is_better:
            improved = True if score > self.best_score + self.min_delta else False
        else:
            improved = True if score < self.best_score - self.min_delta else False

        if improved:
            self.best_score = score
            train_utils.save_model(model, self.model_path, f"{self.model_name}.pt")
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class SimpleCNN(nn.Module):
    def __init__(self, model_config: config.ModelConfig | None = None):
        super().__init__()
        if model_config is None:
            model_config = config.ModelConfig()
        self.config = model_config

        c_in = 1
        size = 28
        self.conv_blocks = nn.ModuleList()
        for spec in model_config.conv_layers:
            layers: list[nn.Module] = [
                nn.Conv2d(
                    c_in, spec.out_channels, spec.kernel_size, padding=spec.padding
                ),
                nn.BatchNorm2d(spec.out_channels),
                nn.ReLU(),
            ]
            c_in = spec.out_channels
            size = size - spec.kernel_size + 1 + 2 * spec.padding
            if spec.pool is not None:
                layers.append(nn.MaxPool2d(spec.pool))
                size //= spec.pool
            self.conv_blocks.append(nn.Sequential(*layers))

        in_features = c_in * size * size
        fc_sizes = [in_features, *model_config.fc_hidden, model_config.num_classes]
        fcs = []
        for i in range(len(fc_sizes) - 1):
            fcs.append(nn.Linear(fc_sizes[i], fc_sizes[i + 1]))
            fcs.append(nn.ReLU())
            if model_config.dropout is not None:
                fcs.append(nn.Dropout(model_config.dropout))
        self.fcs = nn.ModuleList(fcs)

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        x = torch.flatten(x, 1)
        for fc in self.fcs[:-1]:
            x = F.relu(fc(x))
        x = self.fcs[-1](x)
        return x
