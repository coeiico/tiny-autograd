"""
The tensor version of the engine: the same idea as Value, one NumPy array per node.

A Value node holds one number, so a 2-16-16-1 network builds hundreds of nodes per example.
A Tensor node holds a whole array, so a matrix multiply over the full batch is one node and
NumPy does the arithmetic in C. The chain rule is unchanged; what's new is that every local
derivative is now a matrix expression, and broadcasting has to be undone on the way back.
"""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

ArrayLike = np.ndarray | float | int


class Tensor:
  """An array in a computation graph, with the gradient of the final scalar w.r.t. it."""

  def __init__(self, data: ArrayLike, _prev: tuple['Tensor', ...] = (), _op: str = '') -> None:
    self.data = np.asarray(data, dtype=np.float64)
    self.grad = np.zeros_like(self.data)
    """d(output)/d(self), the same shape as data."""
    # Takes this node's gradient as an argument, as in Value, so no closure refers to its own
    # node and graphs are freed without the cycle collector.
    self._backward: Callable[[np.ndarray], None] = lambda grad: None
    self._prev = _prev
    self._op = _op

  @property
  def shape(self) -> tuple[int, ...]:
    return self.data.shape

  def __repr__(self) -> str:
    return f'Tensor(shape={self.shape}, op={self._op!r})'

  def __add__(self, other: 'Tensor | ArrayLike') -> 'Tensor':
    other = as_tensor(other)
    out = Tensor(self.data + other.data, (self, other), '+')

    def _backward(grad: np.ndarray) -> None:
      self.grad += unbroadcast(grad, self.shape)
      other.grad += unbroadcast(grad, other.shape)

    out._backward = _backward
    return out

  def __mul__(self, other: 'Tensor | ArrayLike') -> 'Tensor':
    # Elementwise, like Value's ×: each input's gradient is scaled by the other input.
    other = as_tensor(other)
    out = Tensor(self.data * other.data, (self, other), '×')

    def _backward(grad: np.ndarray) -> None:
      self.grad += unbroadcast(other.data * grad, self.shape)
      other.grad += unbroadcast(self.data * grad, other.shape)

    out._backward = _backward
    return out

  def __pow__(self, exponent: float) -> 'Tensor':
    assert isinstance(exponent, int | float), 'only constant exponents are supported'
    out = Tensor(self.data**exponent, (self,), f'^{exponent:g}')

    def _backward(grad: np.ndarray) -> None:
      self.grad += exponent * self.data ** (exponent - 1) * grad

    out._backward = _backward
    return out

  def __matmul__(self, other: 'Tensor') -> 'Tensor':
    """
    Matrix product of two 2D tensors: [n, k] @ [k, m] → [n, m].

    out[i, j] = Σₖ a[i, k] b[k, j], so d(loss)/d(a[i, k]) = Σⱼ g[i, j] b[k, j], which is
    g @ bᵀ; likewise d(loss)/db = aᵀ @ g. Check the shapes: [n, m] @ [m, k] = [n, k] for a.
    """
    assert self.data.ndim == 2 and other.data.ndim == 2, 'matmul expects 2D tensors'
    out = Tensor(self.data @ other.data, (self, other), '@')

    def _backward(grad: np.ndarray) -> None:
      self.grad += grad @ other.data.T
      other.grad += self.data.T @ grad

    out._backward = _backward
    return out

  def __neg__(self) -> 'Tensor':
    return self * -1.0

  def __sub__(self, other: 'Tensor | ArrayLike') -> 'Tensor':
    return self + (-as_tensor(other))

  def __truediv__(self, other: 'Tensor | ArrayLike') -> 'Tensor':
    return self * as_tensor(other) ** -1.0

  def __radd__(self, other: ArrayLike) -> 'Tensor':
    return self + other

  def __rmul__(self, other: ArrayLike) -> 'Tensor':
    return self * other

  def __rsub__(self, other: ArrayLike) -> 'Tensor':
    return as_tensor(other) - self

  def __rtruediv__(self, other: ArrayLike) -> 'Tensor':
    return as_tensor(other) / self

  def sum(self, axis: int | None = None, keepdims: bool = False) -> 'Tensor':
    out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), 'sum')

    def _backward(grad: np.ndarray) -> None:
      # Every summed element contributed with slope 1, so each one gets the output's gradient:
      # put back the summed-away axis, then copy the gradient along it.
      if axis is not None and not keepdims:
        grad = np.expand_dims(grad, axis)
      self.grad += np.broadcast_to(grad, self.shape)

    out._backward = _backward
    return out

  def mean(self, axis: int | None = None) -> 'Tensor':
    n = self.data.size if axis is None else self.shape[axis]
    return self.sum(axis) / n

  def exp(self) -> 'Tensor':
    e = np.exp(self.data)
    out = Tensor(e, (self,), 'exp')

    def _backward(grad: np.ndarray) -> None:
      self.grad += e * grad

    out._backward = _backward
    return out

  def log(self) -> 'Tensor':
    out = Tensor(np.log(self.data), (self,), 'log')

    def _backward(grad: np.ndarray) -> None:
      self.grad += grad / self.data

    out._backward = _backward
    return out

  def tanh(self) -> 'Tensor':
    t = np.tanh(self.data)
    out = Tensor(t, (self,), 'tanh')

    def _backward(grad: np.ndarray) -> None:
      self.grad += (1 - t**2) * grad

    out._backward = _backward
    return out

  def relu(self) -> 'Tensor':
    out = Tensor(np.maximum(self.data, 0.0), (self,), 'relu')

    def _backward(grad: np.ndarray) -> None:
      self.grad += (self.data > 0) * grad

    out._backward = _backward
    return out

  def softplus(self) -> 'Tensor':
    """
    log(1 + eˣ), computed stably as max(x, 0) + log(1 + e^−|x|).

    One fused op rather than a chain of exp and log: the stable form needs max and abs, and
    the derivative is simply the sigmoid σ(x) = 1 / (1 + e⁻ˣ).
    """
    out = Tensor(np.maximum(self.data, 0) + np.log1p(np.exp(-np.abs(self.data))), (self,), 'softplus')

    def _backward(grad: np.ndarray) -> None:
      # σ(x) written to only exponentiate non-positive numbers, so it can't overflow either.
      e = np.exp(-np.abs(self.data))
      sigmoid = np.where(self.data >= 0, 1 / (1 + e), e / (1 + e))
      self.grad += sigmoid * grad

    out._backward = _backward
    return out

  def backward(self) -> None:
    """
    Fill in .grad on every Tensor this scalar depends on, by the chain rule.

    Same rules as Value.backward(): leaves accumulate, intermediates are reset each call.
    """
    assert self.data.size == 1, 'backward() starts from a scalar, such as a loss'
    order = topological_order(self)
    for node in order:
      if node._prev:
        node.grad = np.zeros_like(node.data)
    self.grad = np.ones_like(self.data)
    for node in reversed(order):
      node._backward(node.grad)


def as_tensor(x: 'Tensor | ArrayLike') -> Tensor:
  return x if isinstance(x, Tensor) else Tensor(x)


def unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
  """
  Sum a gradient back down to the shape of the input it belongs to.

  Broadcasting copies an input along new or size-1 axes: a bias of shape [m] added to a
  batch [n, m] is used n times. A value used n times collects n gradients, summed (as with
  a Value used twice), so undo each copy by summing over the axis it was copied along.
  """
  # Axes broadcasting added on the left.
  while grad.ndim > len(shape):
    grad = grad.sum(axis=0)
  # Axes that were size 1 and got stretched.
  for axis, size in enumerate(shape):
    if size == 1 and grad.shape[axis] != 1:
      grad = grad.sum(axis=axis, keepdims=True)
  return grad


def topological_order(root: Tensor) -> list[Tensor]:
  """Every Tensor root depends on, inputs first. The same walk as engine.topological_order."""
  order: list[Tensor] = []
  visited: set[int] = set()
  stack: list[tuple[Tensor, bool]] = [(root, False)]
  while stack:
    node, children_done = stack.pop()
    if children_done:
      order.append(node)
      continue
    if id(node) in visited:
      continue
    visited.add(id(node))
    stack.append((node, True))
    for child in node._prev:
      if id(child) not in visited:
        stack.append((child, False))
  return order


# Losses.


def logistic_loss(logits: Tensor, labels: np.ndarray) -> Tensor:
  """
  Mean binary cross-entropy on logits [n] with labels in {0, 1}.

  Per example this is softplus(−logit) for class 1 and softplus(logit) for class 0, as in
  nn.logistic_loss; the sign flip turns both cases into one expression.
  """
  sign = np.where(labels == 1, -1.0, 1.0)
  return (logits * sign).softplus().mean()


def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
  """
  Mean cross-entropy of logits [n, classes] against integer targets [n], fused with softmax.

  Fused because the combined derivative is far simpler than its parts: d(loss)/d(logits) =
  (softmax(logits) − onehot(targets)) / n. Subtracting each row's max before exponentiating
  changes nothing mathematically (softmax ignores a constant shift) but keeps exp from
  overflowing.
  """
  n = logits.shape[0]
  shifted = logits.data - logits.data.max(axis=1, keepdims=True)
  log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
  rows = np.arange(n)
  out = Tensor(-log_probs[rows, targets].mean(), (logits,), 'cross_entropy')

  def _backward(grad: np.ndarray) -> None:
    d_logits = np.exp(log_probs)
    d_logits[rows, targets] -= 1
    logits.grad += d_logits / n * grad

  out._backward = _backward
  return out


# The network, the same architecture as nn.MLP with one weight matrix per layer.


class TensorMLP:
  """
  Hidden layers, then one linear output: nn.MLP with each layer as a single matmul.

  Layer weights are [n_in, n_out], so a batch [n, n_in] @ W gives [n, n_out]; column j of W
  holds the weights of nn.MLP's neuron j. It loads and saves the same dict as nn.MLP.
  """

  def __init__(self, record: dict[str, Any]) -> None:
    self.n_in, self.hidden, self.nonlin = record['n_in'], tuple(record['hidden']), record['nonlin']
    self.weights: list[Tensor] = []
    self.biases: list[Tensor] = []
    sizes = [self.n_in, *self.hidden, 1]
    if len(record['layers']) != len(sizes) - 1:
      raise ValueError(f'expected {len(sizes) - 1} layers, file has {len(record["layers"])}')
    for i, layer in enumerate(record['layers']):
      w = np.asarray(layer['weights'], dtype=np.float64).T
      b = np.asarray(layer['biases'], dtype=np.float64)
      if w.shape != (sizes[i], sizes[i + 1]) or b.shape != (sizes[i + 1],):
        raise ValueError(f'layer {i}: weights {w.shape} and biases {b.shape} do not match the architecture')
      self.weights.append(Tensor(w))
      self.biases.append(Tensor(b))

  def __call__(self, x: np.ndarray | Tensor, return_hidden: bool = False) -> Any:
    """
    Logits [n] for points [n, n_in]; with return_hidden, also each hidden layer's activations.

    The hidden activations are the network's internal representation of each point: what
    analyze.py plots to show how the layers carve up the plane.
    """
    h = as_tensor(x)
    hidden: list[np.ndarray] = []
    for w, b in zip(self.weights[:-1], self.biases[:-1], strict=True):
      h = h @ w + b
      h = h.relu() if self.nonlin == 'relu' else h.tanh()
      hidden.append(h.data)
    logits = h @ self.weights[-1] + self.biases[-1]
    # Flatten [n, 1] to [n] with a sum over the size-1 axis: a reshape op isn't needed for
    # anything else, and sum already knows how to send the gradient back.
    logits = logits.sum(axis=1)
    return (logits, hidden) if return_hidden else logits

  def parameters(self) -> list[Tensor]:
    return [p for pair in zip(self.weights, self.biases, strict=True) for p in pair]

  def to_dict(self) -> dict[str, Any]:
    return {
      'n_in': self.n_in,
      'hidden': list(self.hidden),
      'nonlin': self.nonlin,
      'layers': [
        {'weights': w.data.T.tolist(), 'biases': b.data.tolist()}
        for w, b in zip(self.weights, self.biases, strict=True)
      ],
    }


# Optimisers.


class SGD:
  """The same update as nn.SGD, applied to whole arrays."""

  def __init__(self, params: Sequence[Tensor], lr: float) -> None:
    self.params, self.lr = list(params), lr

  def zero_grad(self) -> None:
    for p in self.params:
      p.grad = np.zeros_like(p.data)

  def step(self) -> None:
    for p in self.params:
      p.data -= self.lr * p.grad


class AdamW:
  """
  Adam with decoupled weight decay (Loshchilov & Hutter, 2019), as torch.optim.AdamW.

  Adam keeps a running mean of each parameter's gradient (m, the momentum) and of its square
  (v), and steps by m / √v. Dividing by √v gives every parameter a step of about lr whatever
  its gradient's scale. Weight decay shrinks the weights directly rather than through the
  gradient, so Adam's rescaling can't weaken it.
  """

  def __init__(
    self,
    params: Sequence[Tensor],
    lr: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01,
  ) -> None:
    self.params, self.lr, self.betas, self.eps, self.weight_decay = list(params), lr, betas, eps, weight_decay
    self.m = [np.zeros_like(p.data) for p in self.params]
    self.v = [np.zeros_like(p.data) for p in self.params]
    self.t = 0

  def zero_grad(self) -> None:
    for p in self.params:
      p.grad = np.zeros_like(p.data)

  def step(self) -> None:
    self.t += 1
    beta1, beta2 = self.betas
    for p, m, v in zip(self.params, self.m, self.v, strict=True):
      p.data -= self.lr * self.weight_decay * p.data
      m[:] = beta1 * m + (1 - beta1) * p.grad
      v[:] = beta2 * v + (1 - beta2) * p.grad**2
      # m and v start at zero, so early on they underestimate the true averages. Dividing by
      # 1 − βᵗ corrects that bias; it fades to 1 as t grows.
      m_hat = m / (1 - beta1**self.t)
      v_hat = v / (1 - beta2**self.t)
      p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
