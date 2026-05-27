import torch
from base.config import DataConfig, TrainingConfig
from models.cnn import SimpleCNN
from models.config import ConvSpec, SimpleCNNModelConfig

# ==========================================
# Tests for base/config.py
# ==========================================


def test_data_config():
    config = DataConfig(
        root="/dummy/root",
        train_images_filename="train-images-idx3-ubyte",
        train_labels_filename="train-labels-idx1-ubyte",
        test_images_filename="t10k-images-idx3-ubyte",
        test_labels_filename="t10k-labels-idx1-ubyte",
    )

    assert config.root == "/dummy/root"
    assert config.train_images_filename == "train-images-idx3-ubyte"
    assert config.data_path == "/dummy/root/data"
    assert config.model_path == "/dummy/root/models"
    assert config.experiment_path == "/dummy/root/experiments"


def test_training_config():
    config = TrainingConfig(
        learning_rate=0.001,
        batch_size=64,
        early_stopping_patience=5,
        scheduler_patience=2,
        scheduler_factor=0.5,
        num_epochs=10,
        weight_decay=1e-4,
        early_stopping=True,
    )
    assert config.learning_rate == 0.001
    assert config.batch_size == 64
    assert config.early_stopping is True


# ==========================================
# Tests for base/model.py
# ==========================================


def test_simple_cnn_init_and_forward():
    # Define a small CNN configuration
    input_size = 28
    n_classes = 10
    conv_specs = [
        ConvSpec(
            out_channels=8, kernel_size=3, padding=1, pool=2, stride=1
        ),  # size: 28 -> 28 -> 14
        ConvSpec(
            out_channels=16, kernel_size=3, padding=0, pool=None, stride=1
        ),  # size: 14 -> 12
    ]
    model_config = SimpleCNNModelConfig(
        conv_layers=conv_specs, fc_hidden=[32], dropout=0.2
    )

    # Instantiate model
    model = SimpleCNN(
        input_channels=1, input_size=input_size, model_config=model_config
    )

    # Validate structure
    assert len(model.conv_blocks) == 2
    assert len(model.fcs) == 2  # hidden (32) + output (constants.NUM_CLASSES)
    assert model.dropout is not None

    # Forward pass test
    n_samples = 4
    dummy_input = torch.randn(n_samples, 1, input_size, input_size)
    output = model(dummy_input)
    assert output.shape == (n_samples, n_classes)


def test_simple_cnn_none_dropout_none_pool():
    conv_specs = [
        ConvSpec(
            out_channels=4, kernel_size=3, padding=0, pool=None, stride=1
        ),  # size: 28 -> 26
    ]
    model_config = SimpleCNNModelConfig(
        conv_layers=conv_specs, fc_hidden=[], dropout=None
    )

    model = SimpleCNN(input_channels=1, input_size=28, model_config=model_config)
    assert model.dropout is None
    assert len(model.fcs) == 1  # only input_features -> NUM_CLASSES

    dummy_input = torch.randn(2, 1, 28, 28)
    output = model(dummy_input)
    assert output.shape == (2, 10)
