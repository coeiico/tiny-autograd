"""
The scalar Value class and its backward pass.

A Value wraps one number. Every arithmetic operation on Values returns a new Value that
remembers its inputs (`_prev`) and how to push a gradient back to them (`_backward`). The
result is a graph of the whole computation, built as a side effect of running it.

Value supports +, −, ×, ÷, power, exp, log, tanh and relu, each with its local derivative.
backward() topologically sorts the graph and applies the chain rule in reverse.
"""

import math
from collections.abc import Callable, Iterable

Number = int | float


class Value:
  """One scalar in a computation graph, with the gradient of the final output w.r.t. it."""

  def __init__(self, data: Number, _prev: tuple['Value', ...] = (), _op: str = '', label: str = '') -> None:
    self.data = float(data)
    self.grad = 0.0
    """d(output)/d(self), filled in by output.backward()."""
    # Each op sets _backward to a closure that takes this node's gradient (flowing in from
    # above) and adds its share to each input's grad. Leaves have nothing to do. The gradient
    # is passed in rather than read from the output node, so the closure never refers to its
    # own node: no reference cycle, and Python frees each step's graph the moment it's dropped
    # instead of leaving it to the cycle collector, which made training about 4× slower.
    self._backward: Callable[[float], None] = lambda grad: None
    self._prev = _prev
    self._op = _op
    self.label = label

  def __repr__(self) -> str:
    return f'Value(data={self.data:.6g}, grad={self.grad:.6g})'

  # Binary ops. Each one accepts a plain number on either side by wrapping it in a Value, so
  # `2 * x` and `x + 1` work. The wrapped constant is a leaf like any other: it receives a
  # gradient nobody reads.

  def __add__(self, other: 'Value | Number') -> 'Value':
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')

    def _backward(grad: float) -> None:
      # d(a+b)/da = d(a+b)/db = 1: addition passes the gradient through unchanged.
      # += rather than =, because a Value used twice gets gradient from both uses.
      self.grad += grad
      other.grad += grad

    out._backward = _backward
    return out

  def __mul__(self, other: 'Value | Number') -> 'Value':
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '×')

    def _backward(grad: float) -> None:
      # d(ab)/da = b: each input's gradient is scaled by the other input's value.
      self.grad += other.data * grad
      other.grad += self.data * grad

    out._backward = _backward
    return out

  def __pow__(self, exponent: Number) -> 'Value':
    # A constant exponent only: x ** y with y a Value would need d/dy = x^y · ln x, which
    # nothing here uses.
    assert isinstance(exponent, int | float), 'only constant exponents are supported'
    out = Value(self.data**exponent, (self,), f'^{exponent:g}')

    def _backward(grad: float) -> None:
      self.grad += exponent * self.data ** (exponent - 1) * grad

    out._backward = _backward
    return out

  # Everything else is built from the ops above, so it needs no derivative of its own.

  def __neg__(self) -> 'Value':
    return self * -1

  def __sub__(self, other: 'Value | Number') -> 'Value':
    return self + (-other)

  def __truediv__(self, other: 'Value | Number') -> 'Value':
    other = other if isinstance(other, Value) else Value(other)
    return self * other**-1

  # The reflected versions handle a plain number on the left: Python tries int.__add__(2, x)
  # first, which fails, then falls back to x.__radd__(2).

  def __radd__(self, other: Number) -> 'Value':
    return self + other

  def __rmul__(self, other: Number) -> 'Value':
    return self * other

  def __rsub__(self, other: Number) -> 'Value':
    return Value(other) - self

  def __rtruediv__(self, other: Number) -> 'Value':
    return Value(other) / self

  # Unary functions.

  def exp(self) -> 'Value':
    e = math.exp(self.data)
    out = Value(e, (self,), 'exp')

    def _backward(grad: float) -> None:
      # d(e^x)/dx = e^x, which is the output itself.
      self.grad += e * grad

    out._backward = _backward
    return out

  def log(self) -> 'Value':
    # Natural log, for the logistic loss. Undefined at x <= 0, and math.log says so loudly.
    out = Value(math.log(self.data), (self,), 'log')

    def _backward(grad: float) -> None:
      self.grad += grad / self.data

    out._backward = _backward
    return out

  def tanh(self) -> 'Value':
    t = math.tanh(self.data)
    out = Value(t, (self,), 'tanh')

    def _backward(grad: float) -> None:
      # d(tanh x)/dx = 1 − tanh²(x): reuse the forward result instead of recomputing it.
      self.grad += (1 - t**2) * grad

    out._backward = _backward
    return out

  def relu(self) -> 'Value':
    out = Value(max(0.0, self.data), (self,), 'relu')

    def _backward(grad: float) -> None:
      # The slope is 1 where the input is positive and 0 elsewhere. At exactly 0 the derivative
      # is undefined; 0 is the convention PyTorch uses too.
      self.grad += (self.data > 0) * grad

    out._backward = _backward
    return out

  # The backward pass.

  def backward(self) -> None:
    """
    Fill in .grad on every Value this one depends on, by the chain rule.

    Gradients accumulate into leaves (Values no op produced, such as parameters and
    inputs), exactly as in PyTorch: calling backward() twice without zeroing doubles them,
    so zero them between steps. Intermediate Values are reset here on every call, so their
    .grad always holds this call's gradient and repeated calls stay correct.
    """
    order = topological_order(self)
    for node in order:
      if node._prev:
        node.grad = 0.0
    # d(self)/d(self) = 1 seeds the chain rule.
    self.grad = 1.0
    # Reverse topological order visits a node only after every node that uses it, so its
    # .grad is complete (all paths summed) before it is pushed further down.
    for node in reversed(order):
      node._backward(node.grad)


def topological_order(root: Value) -> list[Value]:
  """
  Every Value root depends on, inputs before the Values computed from them.

  Iterative depth-first search rather than recursion: summing a loss over a batch chains
  one + per example, and a recursive walk that deep would hit Python's recursion limit.
  """
  order: list[Value] = []
  visited: set[int] = set()  # by id(): Values define arithmetic, so compare identity, not value
  stack: list[tuple[Value, bool]] = [(root, False)]
  while stack:
    node, children_done = stack.pop()
    if children_done:
      order.append(node)
      continue
    if id(node) in visited:
      continue
    visited.add(id(node))
    # Come back to this node once all its inputs are in the order.
    stack.append((node, True))
    for child in node._prev:
      if id(child) not in visited:
        stack.append((child, False))
  return order


def total(values: Iterable[Value]) -> Value:
  """
  Add up Values, starting from the first one.

  Python's sum() works too, but starts from the number 0, which puts a useless constant node
  into every sum and clutters the drawn graph.
  """
  it = iter(values)
  out = next(it)
  for v in it:
    out = out + v
  return out
