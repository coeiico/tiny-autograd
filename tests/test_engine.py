"""
The scalar engine against PyTorch and finite differences.

PyTorch runs in float64 here, the same precision as Python floats, so gradients should agree
to rounding error, not just approximately.
"""

import gc
import math
import random
from collections.abc import Callable
from typing import Any

import pytest
import torch

from tiny_autograd.engine import Value, topological_order, total
from tiny_autograd.nn import logistic_loss

TOL = 1e-12

# Each case: a function applied identically to Values and to torch tensors, and the input
# points to check it at. Points avoid the non-differentiable spots (relu at 0, log at <= 0).
UNARY: dict[str, tuple[Callable, list[float]]] = {
  'neg': (lambda x: -x, [-1.5, 2.0]),
  'exp': (lambda x: x.exp(), [-2.0, 0.3, 1.7]),
  'log': (lambda x: x.log(), [0.2, 1.0, 5.0]),
  'tanh': (lambda x: x.tanh(), [-2.0, 0.1, 3.0]),
  'relu': (lambda x: x.relu(), [-1.0, 0.5]),
  'square': (lambda x: x**2, [-1.5, 3.0]),
  'cube': (lambda x: x**3, [-1.5, 2.0]),
  'inverse': (lambda x: x**-1, [-2.0, 0.5]),
  'sqrt': (lambda x: x**0.5, [0.3, 4.0]),
  'plus constant': (lambda x: x + 3, [1.0]),
  'constant plus': (lambda x: 3 + x, [1.0]),
  'constant minus': (lambda x: 3 - x, [1.0]),
  'constant times': (lambda x: 2.5 * x, [1.0]),
  'constant over': (lambda x: 2 / x, [0.7, -3.0]),
}

BINARY: dict[str, Callable] = {
  'add': lambda a, b: a + b,
  'sub': lambda a, b: a - b,
  'mul': lambda a, b: a * b,
  'div': lambda a, b: a / b,
}


def torch_grad(f: Callable, *xs: float) -> tuple[float, list[float]]:
  inputs = [torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in xs]
  out = f(*inputs)
  out.backward()
  return out.item(), [t.grad.item() for t in inputs]  # pyright: ignore[reportOptionalMemberAccess]


def our_grad(f: Callable, *xs: float) -> tuple[float, list[float]]:
  inputs = [Value(x) for x in xs]
  out = f(*inputs)
  out.backward()
  return out.data, [v.grad for v in inputs]


@pytest.mark.parametrize('name', UNARY)
def test_unary_ops_match_pytorch(name: str) -> None:
  f, points = UNARY[name]
  for x in points:
    ours, theirs = our_grad(f, x), torch_grad(f, x)
    assert ours[0] == pytest.approx(theirs[0], abs=TOL)
    assert ours[1] == pytest.approx(theirs[1], abs=TOL), f'{name} at {x}'


@pytest.mark.parametrize('name', BINARY)
def test_binary_ops_match_pytorch(name: str) -> None:
  for a, b in [(1.5, -0.7), (-2.0, 3.0)]:
    ours, theirs = our_grad(BINARY[name], a, b), torch_grad(BINARY[name], a, b)
    assert ours[0] == pytest.approx(theirs[0], abs=TOL)
    assert ours[1] == pytest.approx(theirs[1], abs=TOL)


def test_relu_gradient_at_zero_is_zero_like_pytorch() -> None:
  assert our_grad(lambda x: x.relu(), 0.0)[1] == torch_grad(lambda x: x.relu(), 0.0)[1] == [0.0]


def composed(a: Any, b: Any) -> Any:
  """A graph that reuses intermediates many times (micrograd's own sanity check, extended)."""
  c = a + b
  d = a * b + b**3
  c = c + c + 1
  c = c + 1 + c + (-a)
  d = d + d * 2 + (b + a).relu()
  d = d + 3 * d + (b - a).relu()
  e = c - d
  f = e**2
  g = f / 2.0
  g = g + 10.0 / f
  return (g * a.tanh() ** 2 + (b / 7).exp()).log()


def test_composed_graph_with_reuse_matches_pytorch() -> None:
  ours, theirs = our_grad(composed, -4.0, 2.0), torch_grad(composed, -4.0, 2.0)
  assert ours[0] == pytest.approx(theirs[0], rel=1e-12)
  assert ours[1] == pytest.approx(theirs[1], rel=1e-10)


def test_value_reused_on_two_paths_sums_both_gradients() -> None:
  # f = x·x + x has df/dx = 2x + 1. Overwriting instead of accumulating would give x or 1.
  x = Value(3.0)
  (x * x + x).backward()
  assert x.grad == 7.0


def test_composed_graph_matches_finite_differences() -> None:
  # Central differences, independent of PyTorch: (f(x + h) − f(x − h)) / 2h has error O(h²).
  rng = random.Random(0)
  for _ in range(5):
    a, b = rng.uniform(-3, -1), rng.uniform(1, 3)
    _, (grad_a, grad_b) = our_grad(composed, a, b)
    h = 1e-6
    fd_a = (composed(Value(a + h), Value(b)).data - composed(Value(a - h), Value(b)).data) / (2 * h)
    fd_b = (composed(Value(a), Value(b + h)).data - composed(Value(a), Value(b - h)).data) / (2 * h)
    assert grad_a == pytest.approx(fd_a, rel=1e-5)
    assert grad_b == pytest.approx(fd_b, rel=1e-5)


def test_backward_twice_doubles_leaf_gradients_like_pytorch() -> None:
  x, w = Value(2.0), Value(-3.0)
  hidden = (x * w).tanh()
  out = hidden * hidden + w
  out.backward()
  first = (x.grad, w.grad, hidden.grad)
  out.backward()
  # Leaves accumulate, as in PyTorch; the intermediate holds this call's gradient only.
  assert (x.grad, w.grad) == pytest.approx((2 * first[0], 2 * first[1]), abs=TOL)
  assert hidden.grad == pytest.approx(first[2], abs=TOL)

  tx = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
  tw = torch.tensor(-3.0, dtype=torch.float64, requires_grad=True)
  t_out = (tx * tw).tanh() ** 2 + tw
  t_out.backward(retain_graph=True)
  t_out.backward()
  assert (x.grad, w.grad) == pytest.approx((tx.grad.item(), tw.grad.item()), abs=TOL)  # pyright: ignore[reportOptionalMemberAccess]


def test_zeroing_then_backward_gives_a_fresh_gradient() -> None:
  x = Value(2.0)
  out = x**3
  out.backward()
  out.backward()
  x.grad = 0.0
  out.backward()
  assert x.grad == 12.0


def test_deep_graph_does_not_hit_the_recursion_limit() -> None:
  xs = [Value(1.0) for _ in range(5000)]
  out = total(xs)
  out.backward()
  assert out.data == 5000.0
  assert all(x.grad == 1.0 for x in xs)


def test_graphs_hold_no_reference_cycles() -> None:
  # train.py pauses the cycle collector on the strength of this: a dropped graph must be freed
  # by reference counting alone, leaving nothing for gc.collect() to find.
  gc.collect()
  gc.disable()
  try:
    x = Value(0.5)
    out = composed(x, Value(2.0)) + logistic_loss(x * 3, 1)
    out.backward()
    del out
    assert gc.collect() == 0
  finally:
    gc.enable()


def test_topological_order_puts_inputs_first_and_visits_each_node_once() -> None:
  a, b = Value(1.0), Value(2.0)
  c = a * b
  out = c + c + a
  order = topological_order(out)
  position = {id(v): i for i, v in enumerate(order)}
  assert len(position) == len(order)
  for node in order:
    for child in node._prev:
      assert position[id(child)] < position[id(node)]


@pytest.mark.parametrize('logit', [-800.0, -30.0, -1.2, 0.0, 0.7, 30.0, 800.0])
@pytest.mark.parametrize('label', [0, 1])
def test_logistic_loss_matches_pytorch_without_overflow(logit: float, label: int) -> None:
  v = Value(logit)
  loss = logistic_loss(v, label)
  loss.backward()
  t = torch.tensor(logit, dtype=torch.float64, requires_grad=True)
  t_loss = torch.nn.functional.binary_cross_entropy_with_logits(t, torch.tensor(float(label), dtype=torch.float64))
  t_loss.backward()
  assert math.isfinite(loss.data)
  # The value is checked against softplus computed exactly with log1p, not against PyTorch:
  # for a confident correct logit the loss is about e^-|logit|, and PyTorch's label-0 path
  # keeps fewer digits of a number that small than ours does. Absolute tolerance, because
  # log(1 + 1e-13) itself only keeps a few digits; an error of 1e-16 nats changes nothing.
  z = -logit if label == 1 else logit
  exact = max(z, 0.0) + math.log1p(math.exp(-abs(z)))
  assert loss.data == pytest.approx(exact, rel=1e-12, abs=1e-15)
  assert v.grad == pytest.approx(t.grad.item(), rel=1e-12, abs=1e-15)  # pyright: ignore[reportOptionalMemberAccess]
