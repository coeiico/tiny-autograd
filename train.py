"""
Train tiny-autograd.

`just quick` runs this with --quick: a tiny CPU smoke test that must finish in under 5
minutes. `just train` runs the full experiment.
"""

from dataclasses import dataclass

import tyro

from tiny_autograd.utils import make_run_dir, save_metrics, seed_everything


@dataclass
class Config:
  run_name: str = 'default'
  """Outputs go to runs/<run-name>/."""
  seed: int = 0
  """Seeds Python, NumPy."""
  quick: bool = False
  """Run the tiny CPU smoke test instead of the full experiment."""


def main(cfg: Config) -> None:
  seed_everything(cfg.seed)
  run_dir = make_run_dir(cfg.run_name)
  print(f'tiny-autograd: run_dir={run_dir}')

  # TODO: build an MLP from Values, train it on two moons with plain SGD, and save
  # the decision-boundary plot to run_dir.

  save_metrics(run_dir, cfg, {'status': 'scaffold: training not implemented yet'})


if __name__ == '__main__':
  main(tyro.cli(Config))
