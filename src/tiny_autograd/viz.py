"""Computation-graph drawing with graphviz, and the plots analyze.py and train.py save."""

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import graphviz
import matplotlib

matplotlib.use('Agg')  # write PNGs without needing a display
import matplotlib.pyplot as plt
import numpy as np

from tiny_autograd.data import Split
from tiny_autograd.engine import Value, topological_order

CLASS_COLOURS = ('#2a6fdb', '#e8702a')  # class 0 (upper moon), class 1 (lower moon)


def graph(root: Value) -> graphviz.Digraph:
  """
  The computation graph behind root: a box per Value, a small node per op.

  Each box shows the Value's label, its data and its grad, so after root.backward() the
  picture reads as the chain rule worked out on paper.
  """
  dot = graphviz.Digraph(graph_attr={'rankdir': 'LR'}, node_attr={'fontname': 'Helvetica', 'fontsize': '11'})
  for node in topological_order(root):
    uid = str(id(node))
    label = f'{node.label} | ' if node.label else ''
    dot.node(uid, f'{{ {label}data {node.data:.4f} | grad {node.grad:.4f} }}', shape='record')
    if node._op:
      dot.node(uid + node._op, node._op, shape='ellipse', style='filled', fillcolor='#eeeeee')
      dot.edge(uid + node._op, uid)
    for child in node._prev:
      dot.edge(str(id(child)), uid + node._op)
  return dot


def graph_record(root: Value) -> dict[str, Any]:
  """
  The graph behind root as plain data, inputs first, for drawing the README figure in TikZ.

  Each node has an id, its label, the op that made it, its data and grad, and the ids of its
  inputs.
  """
  order = topological_order(root)
  ids = {id(node): i for i, node in enumerate(order)}
  return {
    'nodes': [
      {
        'id': ids[id(node)],
        'label': node.label,
        'op': node._op,
        'data': node.data,
        'grad': node.grad,
        'inputs': [ids[id(child)] for child in node._prev],
      }
      for node in order
    ]
  }


def save_graph(root: Value, path: Path) -> Path:
  """
  Write the graph's DOT source, and render it to PNG when Graphviz is installed.

  The Python package only writes DOT text; turning it into an image needs the `dot` program.
  Without it, say so instead of failing, so the rest of the analysis still runs.
  """
  dot = graph(root)
  source = path.with_suffix('.dot')
  source.write_text(dot.source)
  if shutil.which('dot') is None:
    print(f'Graphviz `dot` not found: wrote {source} but no PNG (install graphviz to render it)')
    return source
  dot.render(outfile=path.with_suffix('.png'), cleanup=True)
  return path.with_suffix('.png')


def plot_curves(history: list[dict[str, float]], path: Path, title: str) -> None:
  steps = [h['step'] for h in history]
  fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
  ax_loss.plot(steps, [h['train_loss'] for h in history], label='train')
  ax_loss.plot(steps, [h['val_loss'] for h in history], label='validation')
  ax_loss.set(xlabel='step', ylabel='logistic loss (nats)')
  ax_loss.legend()
  ax_acc.plot(steps, [h['train_accuracy'] for h in history], label='train')
  ax_acc.plot(steps, [h['val_accuracy'] for h in history], label='validation')
  ax_acc.set(xlabel='step', ylabel='accuracy', ylim=(0.4, 1.02))
  ax_acc.legend()
  fig.suptitle(title)
  fig.tight_layout()
  fig.savefig(path, dpi=120)
  plt.close(fig)


def grid(split: Split, steps: int, margin: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """A steps × steps grid of points covering the data, as (xs, ys, points [steps², 2])."""
  lo, hi = split.x.min(axis=0) - margin, split.x.max(axis=0) + margin
  xs, ys = np.meshgrid(np.linspace(lo[0], hi[0], steps), np.linspace(lo[1], hi[1], steps))
  return xs, ys, np.stack([xs.ravel(), ys.ravel()], axis=1)


def plot_decision_boundary(
  predict_logits: Callable[[np.ndarray], np.ndarray],
  train: Split,
  test: Split,
  path: Path,
  title: str,
  resolution: int = 200,
) -> None:
  """The plane coloured by predicted probability, the 0.5 boundary in black, and the points."""
  xs, ys, points = grid(test, resolution)
  prob = 1 / (1 + np.exp(-predict_logits(points).reshape(xs.shape)))
  fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True, sharey=True)
  for ax, split, name in zip(axes, (train, test), ('train', 'test'), strict=True):
    ax.contourf(xs, ys, prob, levels=np.linspace(0, 1, 11), cmap='RdBu_r', alpha=0.35)
    ax.contour(xs, ys, prob, levels=[0.5], colors='black', linewidths=1.5)
    for label, colour in enumerate(CLASS_COLOURS):
      pts = split.x[split.y == label]
      ax.scatter(pts[:, 0], pts[:, 1], s=12 if name == 'test' else 22, c=colour, edgecolors='white', linewidths=0.4)
    ax.set(title=f'{name} ({len(split.y)} points)', aspect='equal')
  fig.suptitle(title)
  fig.tight_layout()
  fig.savefig(path, dpi=120)
  plt.close(fig)


def plot_hidden_units(
  hidden_on_grid: list[np.ndarray],
  xs: np.ndarray,
  ys: np.ndarray,
  boundary: np.ndarray,
  path: Path,
  title: str,
) -> None:
  """
  Each hidden unit's activation over the input plane, one panel per unit, a row per layer.

  The model's boundary is overlaid in black, to compare each unit's region with the result.
  """
  n_rows, n_cols = len(hidden_on_grid), max(h.shape[1] for h in hidden_on_grid)
  fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.6 * n_cols, 1.7 * n_rows + 0.6), squeeze=False)
  for row, acts in enumerate(hidden_on_grid):
    for col in range(n_cols):
      ax = axes[row, col]
      ax.set_xticks([])
      ax.set_yticks([])
      if col >= acts.shape[1]:
        ax.axis('off')
        continue
      unit = acts[:, col].reshape(xs.shape)
      ax.contourf(xs, ys, unit, levels=12, cmap='viridis')
      ax.contour(xs, ys, boundary, levels=[0.0], colors='white', linewidths=0.8)
      if unit.max() <= 0:
        ax.set_title('dead', fontsize=8, color='red')
      if col == 0:
        ax.set_ylabel(f'layer {row + 1}')
  fig.suptitle(title)
  fig.tight_layout()
  fig.savefig(path, dpi=110)
  plt.close(fig)


def plot_ablation(report: list[dict[str, Any]], test_acc: float, path: Path) -> None:
  """Test accuracy lost when each hidden unit is mean-ablated, one panel per layer."""
  layers = sorted({r['layer'] for r in report})
  fig, axes = plt.subplots(1, len(layers), figsize=(5.5 * len(layers), 3.4), squeeze=False, sharey=True)
  for ax, layer in zip(axes[0], layers, strict=True):
    rows = [r for r in report if r['layer'] == layer]
    colours = ['#bbbbbb' if r['dead_on_test'] else '#2a6fdb' for r in rows]
    ax.bar([r['unit'] for r in rows], [r['accuracy_drop'] for r in rows], color=colours)
    ax.set(title=f'layer {layer} (grey: dead on every test point)', xlabel='unit', xticks=[r['unit'] for r in rows])
  axes[0][0].set_ylabel('test accuracy lost')
  n_big = sum(r['accuracy_drop'] > 0.05 for r in report)
  fig.suptitle(
    f'Mean-ablating one unit at a time: {n_big} of {len(report)} units cost more than 5 points '
    f'of the {test_acc:.1%} accuracy'
  )
  fig.tight_layout()
  fig.savefig(path, dpi=120)
  plt.close(fig)
