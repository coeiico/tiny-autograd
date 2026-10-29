# tiny-autograd

**Backpropagation from scratch.** A micrograd-style scalar autograd engine and tiny MLP, built to make loss.backward() stop being magic.

Part of the **Latent Space Project**: 18 small ML repos that test one idea, that neural networks encode concepts and strategy in their latent space, and that we can read and edit those representations. Everything trains on a single consumer GPU or a laptop.

| Roadmap | Phase | Depends on | Status |
| --- | --- | --- | --- |
| 1 of 18 | 1. Foundations | Nothing | Not started (scaffolded) |

A learning exercise in the style of Karpathy's micrograd. Nothing else imports it.

## Goal

Make `loss.backward()` stop being magic by building backpropagation from scratch.

## Plan

- **Core:** a scalar `Value` class supporting +, −, ×, ÷, power, exp, tanh and relu, each with its local derivative. `backward()` topologically sorts the graph and applies the chain rule in reverse.
- **On top:** `Neuron`, `Layer` and `MLP` classes, plain SGD, and a training run on a 2D toy dataset (two moons) with a decision-boundary plot.
- **Extras:** draw the computation graph (graphviz). Tests compare every gradient against PyTorch.
- **Stretch goal:** a tensor version on NumPy arrays (matmul, broadcasting-aware gradients, softmax with cross-entropy, AdamW), then retrain clockwork-tfm on it.

## Done when

The definition of done every Latent Space Project repo shares:

Apply this checklist with the project-specific requirements and exceptions in
[CLAUDE.md](CLAUDE.md#project-specific-scope).

- `just quick` runs end to end on CPU in under 5 minutes, as a smoke test.
- The scalar implementation runs on CPU; GPU detection and PyTorch model interfaces are
  not required. PyTorch is used as a reference in tests.
- Each run saves reloadable scalar parameters, architecture settings, `metrics.json`
  and its plots to `runs/<run-name>/`; safetensors is not required.
- This README shows the headline result with plots, compares against baselines, and states negative results plainly.
- Tests cover data generation, local and composed derivatives, reused graph nodes,
  gradient accumulation/reset behaviour and one tiny learning task. Tensor-shape checks
  apply if the optional NumPy tensor extension is implemented.

## Notes

- Default plan: the owner writes this by hand following Karpathy, and Claude Code scaffolds tests and reviews (open question in the handoff doc).
- The graph drawing needs the Graphviz system package as well as the `graphviz` Python package (`sudo apt install graphviz` or `brew install graphviz`).

## Quickstart

Needs [uv](https://docs.astral.sh/uv/) and [just](https://just.systems/). Everything else, Python 3.12 included, comes from uv.

```sh
just setup              # uv sync, then install the pre-commit hooks
just quick              # CPU smoke test, under 5 minutes
just train              # full run; flags pass through, e.g. --run-name wd1.0-seed0
just analyze <run>      # baselines, probes, interventions and plots
just test               # pytest
just check              # ruff and basedpyright, as in CI
```

Copy `.env.example` to `.env` (gitignored) for `HF_TOKEN` and `HF_USER`.

## Layout

```
tiny-autograd/
├── README.md                goal, plan, results with plots
├── CLAUDE.md                instructions for coding agents; AGENTS.md points here
├── .agents/                 instructions/ (Python formatting, commits) and plans/
├── pyproject.toml           metadata, dependencies, tool settings
├── uv.lock                  committed lockfile
├── .python-version          pinned by uv
├── justfile                 every routine command
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml
├── src/tiny_autograd/
│   └── engine.py, nn.py, data.py, viz.py, utils.py
├── train.py                 CLI; --quick runs the CPU smoke test
├── predict.py               load weights and run inference
├── analyze.py               baselines, probes, interventions, plots
├── notebooks/               marimo notebooks for exploration
├── tests/
└── runs/                    outputs, gitignored
```

## Results

_No results yet._ The headline result with plots, the comparison against baselines, and any negative results go here.

## Sources

- micrograd
- Neural Networks: Zero to Hero

## License

MIT
