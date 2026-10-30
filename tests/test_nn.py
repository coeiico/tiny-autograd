"""Neuron, Layer, MLP, SGD and saving, checked against an identical PyTorch network."""

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from tiny_autograd.engine import Value, total
from tiny_autograd.nn import (
  MLP,
  SGD,
  Nonlinearity,
  accuracy,
  load_mlp,
  logistic_loss,
  mlp_from_dict,
  mlp_to_dict,
  predict_logits,
  save_mlp,
)

POINTS = np.array([[0.3, -1.2], [1.5, 0.4], [-0.8, 0.9], [0.0, 0.1], [2.0, -0.5]])
LABELS = np.array([1, 0, 0, 1, 1])


def torch_mlp(mlp: MLP) -> torch.nn.Sequential:
  """The same network in PyTorch, float64, with the weights copied over."""
  layers: list[torch.nn.Module] = []
  for i, layer in enumerate(mlp.layers):
    linear = torch.nn.Linear(len(layer.neurons[0].w), len(layer.neurons), dtype=torch.float64)
    with torch.no_grad():
      linear.weight.copy_(torch.tensor([[w.data for w in n.w] for n in layer.neurons], dtype=torch.float64))
      linear.bias.copy_(torch.tensor([n.b.data for n in layer.neurons], dtype=torch.float64))
    layers.append(linear)
    if i < len(mlp.layers) - 1:
      layers.append(torch.nn.ReLU() if mlp.nonlin == 'relu' else torch.nn.Tanh())
  return torch.nn.Sequential(*layers)


def scalar_loss(mlp: MLP) -> Value:
  return total(logistic_loss(mlp(x.tolist()), int(y)) for x, y in zip(POINTS, LABELS, strict=True)) / len(POINTS)


def torch_loss(net: torch.nn.Sequential) -> torch.Tensor:
  logits = net(torch.tensor(POINTS)).squeeze(1)
  return torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor(LABELS, dtype=torch.float64))


def test_parameter_count() -> None:
  # 2→16: 32 weights + 16 biases; 16→16: 256 + 16; 16→1: 16 + 1.
  assert len(MLP(2, (16, 16), 'relu', random.Random(0)).parameters()) == 48 + 272 + 17


def test_initialisation_is_seeded() -> None:
  a = [p.data for p in MLP(2, (4,), 'relu', random.Random(3)).parameters()]
  b = [p.data for p in MLP(2, (4,), 'relu', random.Random(3)).parameters()]
  c = [p.data for p in MLP(2, (4,), 'relu', random.Random(4)).parameters()]
  assert a == b != c


@pytest.mark.parametrize('nonlin', ['relu', 'tanh'])
def test_loss_and_every_parameter_gradient_match_pytorch(nonlin: Nonlinearity) -> None:
  mlp = MLP(2, (6, 5), nonlin, random.Random(1))
  net = torch_mlp(mlp)
  loss = scalar_loss(mlp)
  loss.backward()
  t_loss = torch_loss(net)
  t_loss.backward()
  assert loss.data == pytest.approx(t_loss.item(), rel=1e-12)

  # Flatten PyTorch's gradients in our order: per layer, per neuron, weights then bias.
  linears = [m for m in net if isinstance(m, torch.nn.Linear)]
  theirs = [
    g
    for linear in linears
    for w_row, b in zip(linear.weight.grad, linear.bias.grad, strict=True)  # pyright: ignore[reportArgumentType]
    for g in [*w_row.tolist(), b.item()]
  ]
  ours = [p.grad for p in mlp.parameters()]
  assert ours == pytest.approx(theirs, abs=1e-12)
  # Every parameter is reached. Only with tanh: a ReLU unit that is off for all five points
  # rightly gets zero gradient, so zeros there don't mean a cut graph.
  if nonlin == 'tanh':
    assert all(g != 0 for g in ours)


def test_twenty_sgd_steps_follow_pytorch_exactly() -> None:
  mlp = MLP(2, (8,), 'relu', random.Random(2))
  net = torch_mlp(mlp)
  ours, theirs = SGD(mlp.parameters(), lr=0.3), torch.optim.SGD(net.parameters(), lr=0.3)
  for _ in range(20):
    loss = scalar_loss(mlp)
    ours.zero_grad()
    loss.backward()
    ours.step()
    t_loss = torch_loss(net)
    theirs.zero_grad()
    t_loss.backward()
    theirs.step()
    assert loss.data == pytest.approx(t_loss.item(), rel=1e-10)


def test_sgd_steps_against_the_gradient_and_zero_grad_clears() -> None:
  mlp = MLP(2, (3,), 'relu', random.Random(0))
  before = [p.data for p in mlp.parameters()]
  optimizer = SGD(mlp.parameters(), lr=0.1)
  scalar_loss(mlp).backward()
  grads = [p.grad for p in mlp.parameters()]
  optimizer.step()
  assert [p.data for p in mlp.parameters()] == pytest.approx([b - 0.1 * g for b, g in zip(before, grads, strict=True)])
  optimizer.zero_grad()
  assert all(p.grad == 0.0 for p in mlp.parameters())


def test_mlp_learns_xor() -> None:
  # The tiny learning task: XOR is the classic problem no linear model can solve.
  points = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
  labels = np.array([0, 1, 1, 0])
  mlp = MLP(2, (8,), 'tanh', random.Random(0))
  optimizer = SGD(mlp.parameters(), lr=0.5)
  losses = []
  for _ in range(300):
    loss = total(logistic_loss(mlp(x.tolist()), int(y)) for x, y in zip(points, labels, strict=True)) / 4
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.data)
  assert accuracy(predict_logits(mlp, points), labels) == 1.0
  assert losses[-1] < 0.05


def test_accuracy_counts_positive_logits_as_class_one() -> None:
  assert accuracy(np.array([2.0, -1.0, 0.5, -0.1]), np.array([1, 0, 0, 0])) == 0.75


def test_save_and_load_give_identical_predictions(tmp_path: Path) -> None:
  mlp = MLP(2, (5, 3), 'tanh', random.Random(7))
  loaded = load_mlp(save_mlp(tmp_path / 'weights.json', mlp))
  assert (loaded.n_in, loaded.hidden, loaded.nonlin) == (2, (5, 3), 'tanh')
  assert np.array_equal(predict_logits(loaded, POINTS), predict_logits(mlp, POINTS))


def test_loading_a_mismatched_file_is_an_error(tmp_path: Path) -> None:
  record = mlp_to_dict(MLP(2, (4,), 'relu', random.Random(0)))
  record['layers'][0]['weights'][1].append(0.5)  # one neuron with an extra weight
  path = tmp_path / 'weights.json'
  path.write_text(json.dumps(record))
  with pytest.raises(ValueError, match='expected 2 weights'):
    load_mlp(path)
  record = mlp_to_dict(MLP(2, (4,), 'relu', random.Random(0)))
  record['hidden'] = [4, 4]
  with pytest.raises(ValueError, match='layers'):
    mlp_from_dict(record)
