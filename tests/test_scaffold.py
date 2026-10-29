"""
Scaffold tests that keep CI green until the real ones land.

Tests to add:
- every Value gradient matches PyTorch's (torch is a dev dependency for this)
- the MLP overfits a tiny batch
"""

import numpy as np

from tiny_autograd.utils import seed_everything


def test_seeding_repeats_exactly() -> None:
  seed_everything(0)
  a = np.random.rand(3)
  seed_everything(0)
  b = np.random.rand(3)
  assert np.array_equal(a, b)
