# Repository Guidelines

## Project Structure & Module Organization
`latent_action_model/` contains the Lightning-based latent action model and its configs. `prismatic/` holds the main VLA package, model code, preprocessing, and training utilities. Use `vla-scripts/` for top-level training, finetuning, and deployment entrypoints; `experiments/robot/` contains evaluation code; `docs/` contains deployment notes; `assets/` stores figures. Treat `vla-scripts/extern/` as vendored third-party code and avoid mixing project-specific changes into it unless necessary.

## Build, Test, and Development Commands
The documented local setup is Python 3.10 with an editable install:

- `conda create -n univla python=3.10 -y && conda activate univla`
- `pip install -e .` installs the package and runtime dependencies.
- `pip install -e ".[dev]"` adds `black`, `ruff`, and `pre-commit`.
- `(cd latent_action_model && bash train.sh)` launches stage-1 latent-action training.
- `bash vla-scripts/train.sh` starts distributed UniVLA pretraining.
- `torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune_libero.py ...` is the pattern for benchmark finetuning.

## Coding Style & Naming Conventions
Formatting is defined in `pyproject.toml`: Black and Ruff, 121-character lines, Python 3.8+ targets. Use 4-space indentation, `snake_case` for modules, functions, and config fields, and `PascalCase` for classes. Keep new configs close to the subsystem they control, for example `latent_action_model/config/` or `prismatic/conf/`, and mirror existing script naming such as `finetune_<benchmark>.py`.

## Testing Guidelines
There is no dedicated root test suite in the current tree, so every change should include lightweight verification. At minimum run `python -m ruff check .` and `python -m black --check .`. For training or deployment changes, add a smoke run from the affected area, such as a single-node `torchrun` command or a script `--help` invocation, and record the exact command in the PR.

## Commit & Pull Request Guidelines
Recent commits use brief subjects and occasional prefixes, for example `[fix] ...` and `upload training logs for reference`. Keep subjects concise, imperative, and focused on one change. Pull requests should link the related issue or experiment, state dataset and checkpoint assumptions, list required GPUs, and attach logs, metrics, or screenshots for training, evaluation, or deployment updates.
