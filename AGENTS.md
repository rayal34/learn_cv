# learn-cv

## Stack
- Python 3.12, **uv** package manager
- PyTorch 2.11, torchvision, datasets, matplotlib
- No test framework, no linter/formatter, no CI configured

## Commands
- `uv sync` — install deps (incl. dev: `uv sync --group dev`)
- `uv pip install -e .` — (re-)install project in editable mode after clone
- `uv add <package>` — add dependency
- `uv run python -c "..."` — run inline snippet
- `uv run python -m fashion_mnist.main --exp-name <name>` — train a Fashion-MNIST experiment (uses fully qualified `fashion_mnist.*` imports, works with both `-m` and direct script invocation)
- `uv run mnist/main.py --exp-name <name>` — train an MNIST experiment

## Structure
- `mnist/` — single module (not a package; no `__init__.py`), uses bare `import config` etc.
- `utils/` — package with shared utilities (`train_utils.py`, imported by `mnist/main.py` via editable install)

## Quirks
- `config.py:DatasetConfig.dir` defaults to external path `/Volumes/satechi/mnist` — will fail on machines without this mount
- `mnist/test.ipynb` — Jupyter notebook; run with `uv run jupyter lab` (available in dev deps)
- `mnist/load_data.py` provides both `datasets.MNIST` download API and a custom `MNISTDataset` class for local arrow/csv files