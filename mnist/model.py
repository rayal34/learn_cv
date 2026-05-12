import config
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.fcs = nn.ModuleList(
            nn.Linear(fc_sizes[i], fc_sizes[i + 1]) for i in range(len(fc_sizes) - 1)
        )

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        x = torch.flatten(x, 1)
        for fc in self.fcs[:-1]:
            x = F.relu(fc(x))
        x = self.fcs[-1](x)
        return x
