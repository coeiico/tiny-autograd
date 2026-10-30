"""
Neuron, Layer and MLP built from Values, plus plain SGD and the logistic loss.

Nothing here knows about gradients beyond calling backward(): a network is just a
function that builds a graph of Values, and its parameters are the leaves of that graph
we choose to update.
"""

import json
import math
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

from tiny_autograd.engine import Number, Value, total

Nonlinearity = Literal['relu', 'tanh']


class Module:
  def parameters(self) -> list[Value]:
    return []

  def zero_grad(self) -> None:
    # backward() accumulates into leaves, so stale gradients from the last step must go.
    for p in self.parameters():
      p.grad = 0.0


class Neuron(Module):
  """A weighted sum of the inputs plus a bias, optionally passed through a nonlinearity."""

  def __init__(self, n_in: int, nonlin: Nonlinearity | None, rng: random.Random) -> None:
    # PyTorch's nn.Linear default: uniform in ±1/√n_in. Scaling by fan-in keeps the weighted
    # sum's spread roughly the same whatever the number of inputs, so wide layers don't start
    # with huge activations.
    bound = 1 / math.sqrt(n_in)
    self.w = [Value(rng.uniform(-bound, bound), label='w') for _ in range(n_in)]
    self.b = Value(rng.uniform(-bound, bound), label='b')
    self.nonlin = nonlin

  def __call__(self, x: Sequence[Value | Number]) -> Value:
    act = total(wi * xi for wi, xi in zip(self.w, x, strict=True)) + self.b
    if self.nonlin == 'relu':
      return act.relu()
    if self.nonlin == 'tanh':
      return act.tanh()
    return act

  def parameters(self) -> list[Value]:
    return [*self.w, self.b]


class Layer(Module):
  """n_out neurons, all reading the same inputs."""

  def __init__(self, n_in: int, n_out: int, nonlin: Nonlinearity | None, rng: random.Random) -> None:
    self.neurons = [Neuron(n_in, nonlin, rng) for _ in range(n_out)]

  def __call__(self, x: Sequence[Value | Number]) -> list[Value]:
    return [n(x) for n in self.neurons]

  def parameters(self) -> list[Value]:
    return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
  """
  Hidden layers with a nonlinearity, then one linear output neuron: a binary classifier.

  The output is a logit (a score), not a probability: positive means class 1. hidden=()
  gives logistic regression, the linear baseline.
  """

  def __init__(self, n_in: int, hidden: Sequence[int], nonlin: Nonlinearity, rng: random.Random) -> None:
    self.n_in, self.hidden, self.nonlin = n_in, tuple(hidden), nonlin
    sizes = [n_in, *hidden]
    self.layers = [Layer(sizes[i], sizes[i + 1], nonlin, rng) for i in range(len(hidden))]
    # No nonlinearity on the output: the logit has to be free to be any real number.
    self.layers.append(Layer(sizes[-1], 1, None, rng))

  def __call__(self, x: Sequence[Value | Number]) -> Value:
    h = self.layers[0](x)
    for layer in self.layers[1:]:
      h = layer(h)
    return h[0]

  def parameters(self) -> list[Value]:
    return [p for layer in self.layers for p in layer.parameters()]


def logistic_loss(logit: Value, label: int) -> Value:
  """
  Binary cross-entropy on a logit: −log σ(logit) for class 1, −log(1 − σ(logit)) for class 0.

  Both cases are softplus(z) = log(1 + eᶻ) with z = −logit for class 1 and z = logit for class
  0. For large z, eᶻ overflows, so the positive branch uses the equal form
  z + log(1 + e⁻ᶻ), which only ever exponentiates a negative number (the same trick as
  PyTorch's BCEWithLogitsLoss). Choosing a branch by value is fine: both build the same
  function, so the gradient is the same either way.
  """
  z = -logit if label == 1 else logit
  if z.data > 0:
    return z + (1 + (-z).exp()).log()
  return (1 + z.exp()).log()


def accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
  """The fraction of points on the right side of the boundary: logit > 0 predicts class 1."""
  return float(((logits > 0) == (labels == 1)).mean())


def predict_logits(mlp: MLP, points: np.ndarray) -> np.ndarray:
  """Run the scalar model on each point [n, n_in] and collect the logits as an array [n]."""
  return np.array([mlp(row.tolist()).data for row in points])


class SGD:
  """Plain stochastic gradient descent: step every parameter a little way downhill."""

  def __init__(self, params: list[Value], lr: float) -> None:
    self.params, self.lr = params, lr

  def zero_grad(self) -> None:
    for p in self.params:
      p.grad = 0.0

  def step(self) -> None:
    # The gradient points uphill (the direction that increases the loss fastest), so move
    # against it. Only .data changes: the update itself must not join the graph.
    for p in self.params:
      p.data -= self.lr * p.grad


# Saving and loading. A run's weights go in plain JSON: the architecture plus every number,
# laid out as layers → neurons → weights, so the file is readable by eye and loads into
# either engine.


def mlp_to_dict(mlp: MLP) -> dict[str, Any]:
  return {
    'n_in': mlp.n_in,
    'hidden': list(mlp.hidden),
    'nonlin': mlp.nonlin,
    'layers': [
      {'weights': [[w.data for w in n.w] for n in layer.neurons], 'biases': [n.b.data for n in layer.neurons]}
      for layer in mlp.layers
    ],
  }


def mlp_from_dict(record: dict[str, Any]) -> MLP:
  """
  Rebuild an MLP and copy the saved numbers in.

  Strict: every layer and neuron must match the architecture exactly, so a mismatched file
  is an error rather than a silently half-loaded model.
  """
  mlp = MLP(record['n_in'], record['hidden'], record['nonlin'], random.Random(0))
  if len(record['layers']) != len(mlp.layers):
    raise ValueError(f'expected {len(mlp.layers)} layers, file has {len(record["layers"])}')
  for i, (layer, saved) in enumerate(zip(mlp.layers, record['layers'], strict=True)):
    weights, biases = saved['weights'], saved['biases']
    if len(weights) != len(layer.neurons) or len(biases) != len(layer.neurons):
      raise ValueError(f'layer {i}: expected {len(layer.neurons)} neurons')
    for j, (neuron, w_row, b) in enumerate(zip(layer.neurons, weights, biases, strict=True)):
      if len(w_row) != len(neuron.w):
        raise ValueError(f'layer {i} neuron {j}: expected {len(neuron.w)} weights, file has {len(w_row)}')
      for w, saved_w in zip(neuron.w, w_row, strict=True):
        w.data = float(saved_w)
      neuron.b.data = float(b)
  return mlp


def save_mlp(path: Path, mlp: MLP) -> Path:
  path.write_text(json.dumps(mlp_to_dict(mlp), indent=1) + '\n')
  return path


def load_mlp(path: Path) -> MLP:
  return mlp_from_dict(json.loads(path.read_text()))
