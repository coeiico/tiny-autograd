"""The NumPy tensor engine against PyTorch, and against the scalar engine it generalises."""

import random
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import torch

from tiny_autograd.engine import total
from tiny_autograd.nn import MLP, Nonlinearity, logistic_loss, mlp_to_dict
from tiny_autograd.tensor import AdamW, Tensor, TensorMLP, cross_entropy, unbroadcast
from tiny_autograd.tensor import logistic_loss as tensor_logistic_loss

RNG = np.random.default_rng(0)


def compare(f: Callable[..., Any], *arrays: np.ndarray) -> None:
  """Run f on Tensors and on torch tensors, reduce to a scalar, and compare value and grads."""
  ours = [Tensor(a) for a in arrays]
  out = f(*ours)
  # Weight the output by fixed random numbers before summing, so every output element gets
  # a different upstream gradient and a transposed or misrouted gradient can't hide.
  weights = RNG.normal(size=out.shape)
  (out * weights).sum().backward()
  theirs = [torch.tensor(a, requires_grad=True) for a in arrays]
  t_out = f(*theirs)
  (t_out * torch.tensor(weights)).sum().backward()
  np.testing.assert_allclose(out.data, t_out.detach().numpy(), rtol=1e-12, atol=1e-12)
  for o, t in zip(ours, theirs, strict=True):
    assert o.grad.shape == o.shape
    np.testing.assert_allclose(o.grad, t.grad.numpy(), rtol=1e-10, atol=1e-12)  # pyright: ignore[reportOptionalMemberAccess]


CASES: dict[str, tuple[Callable[..., Any], list[tuple[int, ...]]]] = {
  'add same shape': (lambda a, b: a + b, [(3, 4), (3, 4)]),
  'add bias row': (lambda a, b: a + b, [(3, 4), (4,)]),
  'add column': (lambda a, b: a + b, [(3, 4), (3, 1)]),
  'mul outer broadcast': (lambda a, b: a * b, [(3, 1), (1, 4)]),
  'sub scalar tensor': (lambda a, b: a - b, [(2, 3), ()]),
  'div': (lambda a, b: a / (b * b + 1), [(2, 3), (3,)]),
  'matmul': (lambda a, b: a @ b, [(5, 3), (3, 4)]),
  'sum all': (lambda a: a.sum(), [(3, 4)]),
  'sum axis 0': (lambda a: a.sum(axis=0), [(3, 4)]),
  'sum axis 1 keepdims': (lambda a: a.sum(axis=1, keepdims=True), [(3, 4)]),
  'mean axis 1': (lambda a: a.mean(axis=1), [(3, 4)]),
  'exp': (lambda a: a.exp(), [(2, 3)]),
  'log': (lambda a: (a * a + 0.5).log(), [(2, 3)]),
  'tanh': (lambda a: a.tanh(), [(2, 3)]),
  'relu': (lambda a: a.relu(), [(4, 5)]),
  'pow': (lambda a: (a * a + 1) ** -0.5, [(2, 3)]),
  'mlp layer': (lambda x, w, b: (x @ w + b).relu() @ w.sum(axis=1, keepdims=True), [(6, 3), (3, 3), (3,)]),
}


@pytest.mark.parametrize('name', CASES)
def test_ops_match_pytorch_including_broadcasting(name: str) -> None:
  f, shapes = CASES[name]
  compare(f, *[RNG.normal(size=s) for s in shapes])


def test_softplus_matches_pytorch_at_extremes() -> None:
  x = np.array([-800.0, -30.0, -1.0, 0.0, 1.0, 30.0, 800.0])
  ours = Tensor(x)
  out = ours.softplus()
  out.sum().backward()
  t = torch.tensor(x, requires_grad=True)
  # PyTorch returns x itself above x = 20, where log(1 + eˣ) equals x to float precision.
  t_out = torch.nn.functional.softplus(t)
  t_out.sum().backward()
  assert np.all(np.isfinite(out.data))
  np.testing.assert_allclose(out.data, t_out.detach().numpy(), rtol=1e-12)
  np.testing.assert_allclose(ours.grad, t.grad.numpy(), rtol=1e-12)  # pyright: ignore[reportOptionalMemberAccess]


def test_cross_entropy_matches_pytorch() -> None:
  logits = RNG.normal(size=(8, 5)) * 10  # large logits, to exercise the max shift
  targets = RNG.integers(0, 5, size=8)
  ours = Tensor(logits)
  loss = cross_entropy(ours, targets)
  loss.backward()
  t = torch.tensor(logits, requires_grad=True)
  t_loss = torch.nn.functional.cross_entropy(t, torch.tensor(targets))
  t_loss.backward()
  assert loss.data.item() == pytest.approx(t_loss.item(), rel=1e-12)
  np.testing.assert_allclose(ours.grad, t.grad.numpy(), rtol=1e-10, atol=1e-15)  # pyright: ignore[reportOptionalMemberAccess]


def test_logistic_loss_matches_pytorch() -> None:
  logits = RNG.normal(size=10) * 5
  labels = RNG.integers(0, 2, size=10)
  ours = Tensor(logits)
  loss = tensor_logistic_loss(ours, labels)
  loss.backward()
  t = torch.tensor(logits, requires_grad=True)
  t_loss = torch.nn.functional.binary_cross_entropy_with_logits(t, torch.tensor(labels, dtype=torch.float64))
  t_loss.backward()
  assert loss.data.item() == pytest.approx(t_loss.item(), rel=1e-12)
  np.testing.assert_allclose(ours.grad, t.grad.numpy(), rtol=1e-10)  # pyright: ignore[reportOptionalMemberAccess]


def test_unbroadcast_sums_back_to_the_input_shape() -> None:
  grad = np.ones((2, 3, 4))
  assert unbroadcast(grad, (4,)).tolist() == [6.0] * 4
  assert unbroadcast(grad, (3, 1)).shape == (3, 1)
  assert unbroadcast(grad, (3, 1))[0, 0] == 8.0
  assert unbroadcast(grad, ()).item() == 24.0


def test_backward_twice_accumulates_into_leaves() -> None:
  w = Tensor(RNG.normal(size=(3, 2)))
  out = (Tensor(RNG.normal(size=(4, 3))) @ w).tanh().sum()
  out.backward()
  first = w.grad.copy()
  out.backward()
  np.testing.assert_allclose(w.grad, 2 * first)


@pytest.mark.parametrize('nonlin', ['relu', 'tanh'])
def test_tensor_mlp_matches_the_scalar_mlp(nonlin: Nonlinearity) -> None:
  points, labels = RNG.normal(size=(7, 2)), RNG.integers(0, 2, size=7)
  mlp = MLP(2, (5, 4), nonlin, random.Random(0))
  model = TensorMLP(mlp_to_dict(mlp))

  scalar_loss = total(logistic_loss(mlp(x.tolist()), int(y)) for x, y in zip(points, labels, strict=True)) / len(points)
  scalar_loss.backward()
  loss = tensor_logistic_loss(model(points), labels)
  loss.backward()
  assert loss.data.item() == pytest.approx(scalar_loss.data, rel=1e-12)

  # Scalar neuron j's weights are column j of the tensor layer's matrix.
  for layer, w, b in zip(mlp.layers, model.weights, model.biases, strict=True):
    np.testing.assert_allclose(w.grad, np.array([[p.grad for p in n.w] for n in layer.neurons]).T, atol=1e-12)
    np.testing.assert_allclose(b.grad, [n.b.grad for n in layer.neurons], atol=1e-12)

  logits, hidden = model(points, return_hidden=True)
  assert [h.shape for h in hidden] == [(7, 5), (7, 4)]
  assert logits.shape == (7,)
  assert TensorMLP(model.to_dict()).to_dict() == model.to_dict()


def test_tensor_mlp_rejects_a_mismatched_record() -> None:
  record = mlp_to_dict(MLP(2, (4,), 'relu', random.Random(0)))
  record['layers'][0]['biases'].append(0.0)
  with pytest.raises(ValueError, match='layer 0'):
    TensorMLP(record)


def test_adamw_matches_pytorch_over_several_steps() -> None:
  start = [RNG.normal(size=(3, 2)), RNG.normal(size=2)]
  x = RNG.normal(size=(5, 3))
  ours = [Tensor(a.copy()) for a in start]
  theirs = [torch.tensor(a.copy(), requires_grad=True) for a in start]
  opt = AdamW(ours, lr=0.05, weight_decay=0.1)
  t_opt = torch.optim.AdamW(theirs, lr=0.05, weight_decay=0.1)
  for _ in range(10):
    loss = ((Tensor(x) @ ours[0] + ours[1]).tanh() ** 2).sum()
    opt.zero_grad()
    loss.backward()
    opt.step()
    t_loss = ((torch.tensor(x) @ theirs[0] + theirs[1]).tanh() ** 2).sum()
    t_opt.zero_grad()
    t_loss.backward()
    t_opt.step()
  for o, t in zip(ours, theirs, strict=True):
    np.testing.assert_allclose(o.data, t.detach().numpy(), rtol=1e-10, atol=1e-12)
