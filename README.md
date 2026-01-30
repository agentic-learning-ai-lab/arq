# Local Reinforcement Learning with Action-Conditioned Root Mean Squared Q-Functions

## Abstract

The Forward-Forward (FF) Algorithm is a recently proposed learning procedure for neural networks that employs two forward passes instead of the traditional forward and backward passes used in backpropagation. However, FF remains largely confined to supervised settings, leaving a gap at domains where learning signals can be yielded more naturally such as RL. In this work, inspired by FF's goodness function using layer activity statistics, we introduce Action-conditioned Root mean squared Q-Functions (ARQ), a novel value estimation method that applies a goodness function and action conditioning for local RL using temporal difference learning. Despite its simplicity and biological grounding, our approach achieves superior performance compared to state-of-the-art local backprop-free RL methods in the MinAtar and the DeepMind Control Suite benchmarks, while also outperforming algorithms trained with backpropagation on most tasks.

## Repository Structure

- `scripts/` - Training scripts
- `src/` - Source code implementation
- `dmc2gym/` - DeepMind Control Suite to Gymnasium adapter

**Note:** The current codebase does not fully replicate paper accuracies for MinAtar/Seaquest-v1 and MinAtar/Asterix-v1; a slightly different script was used to produce those results. We're working on unifying them.

## Environment Setup

Create and activate the conda environment:

```bash
conda create -n arq python=3.10
conda activate arq
pip install poetry
poetry install
```

For DeepMind Control Suite tasks, install `xvfb`:
```bash
sudo apt-get install xvfb  # Ubuntu/Debian
```

## Supported Environments

### MinAtar Tasks
- `MinAtar/Breakout-v1`
- `MinAtar/Freeway-v1`
- `MinAtar/SpaceInvaders-v1`
- `MinAtar/Seaquest-v1`
- `MinAtar/Asterix-v1`

### DeepMind Control Suite Tasks
- `walker` (walker walk)
- `runner` (walker run)
- `hopper` (hopper hop)
- `cheetah` (cheetah run)
- `reacher_hard` (reacher hard)

## Training

For MinAtar tasks:
```bash
poetry run python scripts/train.py <ENV_ID> --seed=<SEED>
```

For DMC tasks:
```bash
xvfb-run python scripts/train.py <ENV_ID> --seed=<SEED>
```

### Examples

```bash
# MinAtar
poetry run python scripts/train.py MinAtar/Freeway-v1 --seed=42

# DMC
xvfb-run python scripts/train.py walker --seed=42
```
