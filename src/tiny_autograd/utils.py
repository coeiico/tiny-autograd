"""
Run bookkeeping shared by train.py, predict.py and analyze.py.

Every run writes to runs/<run-name>/ and records enough in metrics.json (all args, the
seed and the git commit) to reproduce it later.
"""

import json
import random
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def seed_everything(seed: int) -> None:
  """Seed every random number generator in play, so the same seed repeats a run."""
  random.seed(seed)
  np.random.seed(seed)


def git_commit() -> str | None:
  """The current commit, so metrics.json says exactly which code produced a result."""
  try:
    out = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True)
  except (OSError, subprocess.CalledProcessError):
    return None
  return out.stdout.strip()


def make_run_dir(run_name: str, root: Path = Path('runs')) -> Path:
  run_dir = root / run_name
  run_dir.mkdir(parents=True, exist_ok=True)
  return run_dir


def save_metrics(run_dir: Path, config: Any, metrics: dict[str, Any]) -> Path:
  """Write metrics.json: the full config (seed included), the git commit and results."""
  if is_dataclass(config) and not isinstance(config, type):
    config = asdict(config)
  record = {'config': config, 'git_commit': git_commit(), 'metrics': metrics}
  path = run_dir / 'metrics.json'
  path.write_text(json.dumps(record, indent=2, default=str))
  return path
