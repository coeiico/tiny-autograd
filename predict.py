"""Load a trained tiny-autograd run and classify points with it."""

import math
from dataclasses import dataclass
from pathlib import Path

import tyro

from tiny_autograd.nn import load_mlp


@dataclass
class Config:
  run: Path
  """Run directory, e.g. runs/default-seed0."""
  points: tuple[tuple[float, float], ...] = ((0.0, 1.0), (1.0, -0.5), (0.5, 0.25))
  """Points to classify, e.g. --points 0 1 1 -0.5. The defaults sit on each moon and between them."""


def main(cfg: Config) -> None:
  mlp = load_mlp(cfg.run / 'weights.json')
  for x in cfg.points:
    # The scalar engine, as in training. The logit is a score; the sigmoid turns it into a
    # probability of class 1 (the lower moon).
    logit = mlp(list(x))
    # Written to only exponentiate a non-positive number, so a confident logit can't overflow.
    e = math.exp(-abs(logit.data))
    prob = 1 / (1 + e) if logit.data >= 0 else e / (1 + e)
    print(f'({x[0]:+.2f}, {x[1]:+.2f})  logit {logit.data:+7.3f}  P(lower moon) {prob:.3f}  -> class {int(logit.data > 0)}')


if __name__ == '__main__':
  main(tyro.cli(Config))
