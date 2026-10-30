"""The two moons generator: shapes, labels, determinism and geometry."""

import numpy as np

from tiny_autograd.data import MoonsArgs, describe, make_moons, make_splits


def test_shapes_and_labels() -> None:
  split = make_moons(50, 0.1, np.random.default_rng(0))
  assert split.x.shape == (50, 2) and split.x.dtype == np.float64
  assert split.y.shape == (50,)
  assert set(split.y.tolist()) == {0, 1}


def test_same_seed_same_data_and_different_seed_different_data() -> None:
  args = MoonsArgs(n_train=20, n_val=10, n_test=10)
  a, b, c = make_splits(args, 0), make_splits(args, 0), make_splits(args, 1)
  assert np.array_equal(a['train'].x, b['train'].x) and np.array_equal(a['test'].y, b['test'].y)
  assert not np.array_equal(a['train'].x, c['train'].x)
  assert describe(args, 0, a)['sha256'] == describe(args, 0, b)['sha256'] != describe(args, 1, c)['sha256']


def test_split_sizes_and_no_shared_points() -> None:
  splits = make_splits(MoonsArgs(n_train=30, n_val=20, n_test=40), 0)
  assert {k: len(s.y) for k, s in splits.items()} == {'train': 30, 'val': 20, 'test': 40}
  rows = [tuple(r) for s in splits.values() for r in s.x]
  assert len(set(rows)) == len(rows)


def test_noiseless_points_lie_on_their_moons() -> None:
  split = make_moons(200, 0.0, np.random.default_rng(0))
  upper, lower = split.x[split.y == 0], split.x[split.y == 1]
  # Upper moon: the unit semicircle above the x axis, centred at the origin.
  np.testing.assert_allclose(np.linalg.norm(upper, axis=1), 1.0)
  assert np.all(upper[:, 1] >= 0)
  # Lower moon: a unit semicircle centred at (1, 0.5), below its centre.
  np.testing.assert_allclose(np.linalg.norm(lower - [1.0, 0.5], axis=1), 1.0)
  assert np.all(lower[:, 1] <= 0.5)


def test_classes_are_roughly_balanced() -> None:
  y = make_moons(4000, 0.1, np.random.default_rng(0)).y
  assert 0.47 < y.mean() < 0.53
