"""
Train an MLP on two moons with the hand-built engine.

`just quick` runs this with --quick: a tiny CPU smoke test that must finish in under 5
minutes. `just train` runs the full experiment. `--model.hidden` with no sizes trains the
linear baseline (logistic regression), and `--engine tensor` trains the same network, from
the same starting weights, on the NumPy tensor engine.
"""

import gc
import math
import random
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

import numpy as np
import tyro

from tiny_autograd import tensor
from tiny_autograd.data import MoonsArgs, Split, describe, make_splits
from tiny_autograd.engine import total
from tiny_autograd.nn import (
  MLP,
  SGD,
  Nonlinearity,
  accuracy,
  logistic_loss,
  mlp_from_dict,
  mlp_to_dict,
  predict_logits,
  save_mlp,
)
from tiny_autograd.utils import code_state, environment, make_run_dir, save_json, seed_everything
from tiny_autograd.viz import plot_curves


@dataclass(frozen=True)
class ModelArgs:
  hidden: tuple[int, ...] = (16, 16)
  """Hidden layer sizes. Empty gives logistic regression, the linear baseline."""
  nonlin: Nonlinearity = 'relu'


@dataclass(frozen=True)
class OptimArgs:
  steps: int = 1000
  """Full-batch steps: every step uses all the training points."""
  lr: float = 0.5
  optimizer: Literal['sgd', 'adamw'] = 'sgd'
  """adamw needs --engine tensor; the scalar engine has plain SGD only."""
  weight_decay: float = 0.0
  """AdamW only."""


@dataclass
class Config:
  run_name: str = 'default'
  """Outputs go to runs/<run-name>/, which must not exist yet."""
  seed: int = 0
  """Seeds the initial weights. The data has its own seed, so seeds share one dataset."""
  data_seed: int = 0
  quick: bool = False
  """Run the tiny CPU smoke test instead of the full experiment."""
  engine: Literal['scalar', 'tensor'] = 'scalar'
  """scalar builds a Value per number; tensor runs the same maths on NumPy arrays."""
  eval_every: int = 25
  data: MoonsArgs = field(default_factory=MoonsArgs)
  model: ModelArgs = field(default_factory=ModelArgs)
  optim: OptimArgs = field(default_factory=OptimArgs)


def quick_config(cfg: Config) -> Config:
  """Shrink everything so the whole pipeline runs on a CPU in well under 5 minutes."""
  return replace(
    cfg,
    eval_every=10,
    data=MoonsArgs(n_train=40, n_val=100, n_test=200, noise=cfg.data.noise),
    model=replace(cfg.model, hidden=(8,)),
    optim=replace(cfg.optim, steps=60),
  )


def stop_if_not_finite(loss: float, step: int, record: dict[str, Any], history: list[dict[str, float]], run_dir: Any) -> None:
  if not math.isfinite(loss):
    record.update(status=f'failed: non-finite loss {loss} at step {step}', history=history)
    save_json(run_dir / 'metrics.json', record)
    raise SystemExit(f'Stopping: loss is {loss} at step {step}.')


def evaluate(logits: np.ndarray, split: Split) -> tuple[float, float]:
  """Mean logistic loss and accuracy, from logits computed by either engine."""
  # The same softplus as the losses in nn.py and tensor.py, on plain arrays.
  z = np.where(split.y == 1, -logits, logits)
  loss = float(np.mean(np.maximum(z, 0) + np.log1p(np.exp(-np.abs(z)))))
  return loss, accuracy(logits, split.y)


def train_scalar(mlp: MLP, splits: dict[str, Split], cfg: Config, log: Any) -> MLP:
  # Python's cycle collector runs as objects are allocated, and every pass re-scans the tens
  # of thousands of live Values each forward pass builds: it tripled the step time. The
  # graph holds no reference cycles (a test checks this), so plain reference counting frees
  # it and the collector can be paused while training.
  gc.disable()
  try:
    return _train_scalar(mlp, splits, cfg, log)
  finally:
    gc.enable()


def _train_scalar(mlp: MLP, splits: dict[str, Split], cfg: Config, log: Any) -> MLP:
  train, val = splits['train'], splits['val']
  points = [row.tolist() for row in train.x]
  optimizer = SGD(mlp.parameters(), lr=cfg.optim.lr)
  for step in range(cfg.optim.steps + 1):
    # Forward: one graph through every training point, averaged into a single loss Value.
    logits = [mlp(x) for x in points]
    loss = total(logistic_loss(logit, int(y)) for logit, y in zip(logits, train.y, strict=True)) / len(points)
    if step % cfg.eval_every == 0 or step == cfg.optim.steps:
      log(step, loss.data, np.array([v.data for v in logits]), predict_logits(mlp, val.x))
    if step == cfg.optim.steps:
      break  # step `steps` is evaluated, not trained on, so the log ends on the final weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  return mlp


def train_tensor(mlp: MLP, splits: dict[str, Split], cfg: Config, log: Any) -> MLP:
  train, val = splits['train'], splits['val']
  # Copy the scalar model's starting weights, so both engines start from the same point.
  model = tensor.TensorMLP(mlp_to_dict(mlp))
  if cfg.optim.optimizer == 'adamw':
    optimizer = tensor.AdamW(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
  else:
    optimizer = tensor.SGD(model.parameters(), lr=cfg.optim.lr)
  for step in range(cfg.optim.steps + 1):
    logits = model(train.x)
    loss = tensor.logistic_loss(logits, train.y)
    if step % cfg.eval_every == 0 or step == cfg.optim.steps:
      log(step, loss.data.item(), logits.data, model(val.x).data)
    if step == cfg.optim.steps:
      break
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  return mlp_from_dict(model.to_dict())


def main(cfg: Config) -> None:
  if cfg.quick:
    cfg = quick_config(cfg)
  if cfg.optim.optimizer == 'adamw' and cfg.engine != 'tensor':
    raise SystemExit('adamw is only implemented for --engine tensor')
  seed_everything(cfg.seed)
  run_dir = make_run_dir(cfg.run_name)

  splits = make_splits(cfg.data, cfg.data_seed)
  train, val, test = splits['train'], splits['val'], splits['test']
  for x, y in zip(train.x[:3], train.y[:3], strict=True):
    print(f'  example: ({x[0]:+.2f}, {x[1]:+.2f}) -> class {y}')
  print(f'  class-1 fraction: train {train.y.mean():.2f}  val {val.y.mean():.2f}  test {test.y.mean():.2f}')

  mlp = MLP(2, cfg.model.hidden, cfg.model.nonlin, random.Random(cfg.seed))
  n_params = len(mlp.parameters())
  print(f'tiny-autograd: {n_params} parameters, {cfg.engine} engine, writing to {run_dir}')

  record: dict[str, Any] = {
    'config': asdict(cfg),
    'parameters': n_params,
    'data': describe(cfg.data, cfg.data_seed, splits),
    'code': code_state(run_dir),
    'environment': environment(),
    'resumable': False,
    'status': 'running',
  }
  save_json(run_dir / 'metrics.json', record)

  history: list[dict[str, float]] = []

  def log(step: int, train_loss: float, train_logits: np.ndarray, val_logits: np.ndarray) -> None:
    stop_if_not_finite(train_loss, step, record, history, run_dir)
    val_loss, val_acc = evaluate(val_logits, val)
    history.append({
      'step': step,
      'train_loss': train_loss,
      'train_accuracy': accuracy(train_logits, train.y),
      'val_loss': val_loss,
      'val_accuracy': val_acc,
      'seconds': round(time.time() - start, 2),
    })
    h = history[-1]
    print(f'step {step:5d}  train {train_loss:.4f} ({h["train_accuracy"]:.3f})  val {val_loss:.4f} ({val_acc:.3f})')

  start = time.time()
  trained = (train_scalar if cfg.engine == 'scalar' else train_tensor)(mlp, splits, cfg, log)
  seconds = time.time() - start

  save_mlp(run_dir / 'weights.json', trained)
  # Test is scored once, on the final weights: nothing was chosen by looking at it.
  test_loss, test_acc = evaluate(predict_logits(trained, test.x), test)
  final = history[-1]
  plot_curves(
    history,
    run_dir / 'training.png',
    f'{"Linear model" if not cfg.model.hidden else "MLP"} reaches {final["val_accuracy"]:.1%} '
    f'validation accuracy in {final["step"]} steps',
  )
  record.update(
    status='finished',
    seconds=round(seconds, 1),
    seconds_per_step=seconds / max(1, cfg.optim.steps),
    history=history,
    final={'train_accuracy': final['train_accuracy'], 'val_accuracy': final['val_accuracy'], 'val_loss': final['val_loss']},
    test={'loss': test_loss, 'accuracy': test_acc},
  )
  save_json(run_dir / 'metrics.json', record)
  print(f'done in {seconds:.1f}s: test accuracy {test_acc:.3f}, {run_dir}')


if __name__ == '__main__':
  main(tyro.cli(Config))
