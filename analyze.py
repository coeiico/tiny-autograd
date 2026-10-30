"""
Baselines, computation graphs, hidden units, ablations and plots for a saved run.

Analysis reads saved weights, so it never needs to retrain. Outputs go to
runs/<run>/<name>/, a new folder, so the training record stays untouched.
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tyro

from tiny_autograd.data import MoonsArgs, describe, make_splits
from tiny_autograd.engine import Value
from tiny_autograd.nn import MLP, accuracy, load_mlp, logistic_loss, mlp_to_dict, predict_logits
from tiny_autograd.tensor import TensorMLP
from tiny_autograd.utils import code_state, make_output_dir, save_json
from tiny_autograd.viz import (
  grid,
  plot_ablation,
  plot_curves,
  plot_decision_boundary,
  plot_hidden_units,
  save_graph,
)


@dataclass
class Config:
  run: Path
  """Run directory, e.g. runs/default-seed0."""
  name: str = 'analysis'
  """Outputs go to <run>/<name>/, which must not exist yet."""
  resolution: int = 200
  """Grid points per side for the boundary and hidden-unit maps."""


def mean_ablate(record: dict[str, Any], layer: int, unit: int, mean_activation: float) -> dict[str, Any]:
  """
  The weights of the same network with one hidden unit frozen at its mean activation.

  The unit's output only reaches the next layer through its outgoing weights, so replacing
  the activation a with a constant ā is the same as zeroing those weights and adding ā × w
  to the next layer's biases. Mean rather than zero, so the ablation removes the information
  the unit carries without also shifting every downstream unit by its average.
  """
  ablated = json.loads(json.dumps(record))  # a deep copy
  nxt = ablated['layers'][layer + 1]
  for j, w_row in enumerate(nxt['weights']):
    nxt['biases'][j] += mean_activation * w_row[unit]
    w_row[unit] = 0.0
  return ablated


def main(cfg: Config) -> None:
  out_dir = make_output_dir(cfg.run / cfg.name)
  train_record = json.loads((cfg.run / 'metrics.json').read_text())
  run_cfg = train_record['config']
  mlp = load_mlp(cfg.run / 'weights.json')
  weights = mlp_to_dict(mlp)

  # Regenerate the run's data and check it is byte-for-byte what training saw.
  data_args = MoonsArgs(**run_cfg['data'])
  splits = make_splits(data_args, run_cfg['data_seed'])
  data = describe(data_args, run_cfg['data_seed'], splits)
  if data['sha256'] != train_record['data']['sha256']:
    raise SystemExit('regenerated data does not match the run (generator or settings changed)')
  train, test = splits['train'], splits['test']

  # The grid has tens of thousands of points, so it runs on the tensor engine: the same
  # weights and maths (tests check the two engines agree), without a graph node per multiply.
  model = TensorMLP(weights)

  def logits_of(points: np.ndarray) -> np.ndarray:
    return model(points).data

  test_logits = predict_logits(mlp, test.x)
  test_acc = accuracy(test_logits, test.y)

  # Baselines on the same test set. The untrained model is this run's exact starting point,
  # rebuilt from its seed; the majority class is what you get by ignoring the input.
  untrained = MLP(2, run_cfg['model']['hidden'], run_cfg['model']['nonlin'], random.Random(run_cfg['seed']))
  majority = int(train.y.mean() >= 0.5)
  baselines = {
    'majority class (from train)': float((test.y == majority).mean()),
    'untrained network (same init)': accuracy(predict_logits(untrained, test.x), test.y),
  }
  print(f'test accuracy {test_acc:.3f}; baselines: ' + ', '.join(f'{k} {v:.3f}' for k, v in baselines.items()))

  kind = 'linear model' if not mlp.hidden else f'{len(mlp.hidden)}-hidden-layer MLP'
  plot_decision_boundary(
    logits_of,
    train,
    test,
    out_dir / 'decision-boundary.png',
    f'The {kind} classifies {test_acc:.1%} of {len(test.y)} held-out points correctly',
    cfg.resolution,
  )
  history = train_record['history']
  plot_curves(
    history,
    out_dir / 'training.png',
    f'Validation accuracy goes from {history[0]["val_accuracy"]:.0%} untrained to {history[-1]["val_accuracy"]:.1%}',
  )

  # Computation graphs, drawn after backward() so each box shows its gradient. The whole
  # network's graph for one point has hundreds of nodes, so draw two readable pieces: one
  # active first-layer neuron of this model on one test point, and the full loss of a 2-2-1 network.
  x0 = [Value(v, label=f'x{sub}') for v, sub in zip(test.x[0], '₁₂', strict=True)]
  # The first unit active on this point: an inactive ReLU passes no gradient, so its graph
  # would show nothing but zeros.
  neuron = next((n for n in mlp.layers[0].neurons if n(x0).data > 0), mlp.layers[0].neurons[0])
  for v, name in zip(neuron.w, ('w₁', 'w₂'), strict=True):
    v.label = name
  neuron.b.label = 'b'
  neuron_out = neuron(x0)
  neuron_out.label = 'output'
  neuron_out.backward()
  graphs = {'neuron': str(save_graph(neuron_out, out_dir / 'graph-neuron'))}
  for v in x0:
    v.grad = 0.0  # the inputs are leaves shared with the graph above, so their gradients accumulate
  toy = MLP(2, (2,), 'tanh', random.Random(0))
  toy_loss = logistic_loss(toy(x0), int(test.y[0]))
  toy_loss.label = 'loss'
  toy_loss.backward()
  graphs['toy_network'] = str(save_graph(toy_loss, out_dir / 'graph-toy-network'))

  hidden_report: list[dict[str, Any]] = []
  if mlp.hidden:
    # What each hidden unit computes over the whole plane: the network's internal picture.
    xs, ys, points = grid(test, cfg.resolution // 2)
    boundary_logits, hidden_on_grid = model(points, return_hidden=True)
    plot_hidden_units(
      hidden_on_grid,
      xs,
      ys,
      boundary_logits.data.reshape(xs.shape),
      out_dir / 'hidden-units.png',
      'Layer-1 units each cut the plane along one line; layer 2 combines the cuts into curves',
    )

    # Ablate each unit in turn on the test set and measure what breaks: decodable is not the
    # same as used, and this asks which units the decision actually depends on.
    _, hidden_on_test = model(test.x, return_hidden=True)
    for layer, acts in enumerate(hidden_on_test):
      for unit in range(acts.shape[1]):
        mean_act = float(acts[:, unit].mean())
        ablated = TensorMLP(mean_ablate(weights, layer, unit, mean_act))
        acc = accuracy(ablated(test.x).data, test.y)
        hidden_report.append({
          'layer': layer + 1,
          'unit': unit,
          'dead_on_test': bool(acts[:, unit].max() <= 0) if mlp.nonlin == 'relu' else False,
          'mean_activation': mean_act,
          'test_accuracy_ablated': acc,
          'accuracy_drop': test_acc - acc,
        })
    worst = max(hidden_report, key=lambda r: r['accuracy_drop'])
    print(
      f'dead units: {sum(r["dead_on_test"] for r in hidden_report)} of {len(hidden_report)}; '
      f'largest single-unit ablation drop: {worst["accuracy_drop"]:.3f} (layer {worst["layer"]} unit {worst["unit"]})'
    )
    plot_ablation(hidden_report, test_acc, out_dir / 'ablation.png')

  save_json(
    out_dir / 'analysis.json',
    {
      'config': {k: str(v) for k, v in vars(cfg).items()},
      'code': code_state(out_dir),
      'data_sha256': data['sha256'],
      'test_accuracy': test_acc,
      'baselines': baselines,
      'graphs': graphs,
      'hidden_units': hidden_report,
    },
  )
  print(f'wrote {out_dir}')


if __name__ == '__main__':
  main(tyro.cli(Config))
