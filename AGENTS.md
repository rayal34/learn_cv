# learn-cv

## Stack
- Python 3.12, **uv** package manager
- PyTorch 2.11, torchvision, datasets, matplotlib
- No test framework, no linter/formatter, no CI configured

## Commands
- `uv run <script.py>` — run any script
- `uv sync` — install deps (incl. dev: `uv sync --group dev`)
- `uv add <package>` — add dependency

## Structure
- `mnist/` — single module (not a package; no `__init__.py`), uses bare `import config` etc.
- `main.py` — placeholder entry point

## Quirks
- `config.py:DatasetConfig.dir` defaults to external path `/Volumes/satechi/mnist` — will fail on machines without this mount
- `mnist/test.ipynb` — Jupyter notebook; run with `uv run jupyter lab` (available in dev deps)
- `mnist/load_data.py` provides both `datasets.MNIST` download API and a custom `MNISTDataset` class for local arrow/csv files

## Experiment tracking
- Every run creates `runs/<timestamp>_<name>/` with `config.json`, `metrics.json`, and `model.pth`
- `runs/` is gitignored
- Define an experiment in `mnist/main.py` by creating an `ExperimentConfig` with a descriptive `name`
- To vary model architecture, change `ModelConfig` fields (conv channels, kernel size, fc units) — `SimpleCNN` derives flattened dimension from conv params automatically