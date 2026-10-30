"""
Run bookkeeping shared by train.py, predict.py and analyze.py.

Every run writes to its own new folder, runs/<run-name>/, and records in metrics.json
enough to reproduce it: the resolved config, the seed, the data definition, the code state
and the environment.
"""

import hashlib
import json
import platform
import random
import subprocess
import tarfile
from pathlib import Path
from typing import Any

import numpy as np

LOCKFILE = Path('uv.lock')


def seed_everything(seed: int) -> None:
  """Seed every random number generator in play, so the same seed repeats a run."""
  random.seed(seed)
  np.random.seed(seed)


def make_run_dir(run_name: str, root: Path = Path('runs')) -> Path:
  """
  Create a fresh folder for a run.

  An existing folder is refused rather than reused: a run's outputs are its record, and
  mixing two runs' files would make that record lie.
  """
  run_dir = root / run_name
  if run_dir.exists():
    raise FileExistsError(f'{run_dir} already exists; pick a new run name')
  run_dir.mkdir(parents=True)
  return run_dir


def make_output_dir(path: Path) -> Path:
  """Create a fresh folder for a later output, such as an analysis, refusing an existing one."""
  if path.exists():
    raise FileExistsError(f'{path} already exists; pick a new name')
  path.mkdir(parents=True)
  return path


def _git(*args: str) -> str | None:
  try:
    out = subprocess.run(['git', *args], capture_output=True, text=True, check=True)
  except (OSError, subprocess.CalledProcessError):
    return None
  return out.stdout


def code_state(run_dir: Path) -> dict[str, Any]:
  """
  Identify the code that produced a run.

  A commit alone does not identify code with local changes, so when the working tree is
  dirty this also saves a snapshot of every tracked and untracked (not ignored) file.
  """
  commit = _git('rev-parse', 'HEAD')
  status = _git('status', '--porcelain')
  if commit is None or status is None:
    return {'commit': None, 'note': 'git unavailable: code version not recorded'}
  state: dict[str, Any] = {'commit': commit.strip(), 'dirty': bool(status.strip())}
  if state['dirty']:
    files = (_git('ls-files', '--cached', '--others', '--exclude-standard') or '').split()
    snapshot = run_dir / 'source-snapshot.tar.gz'
    with tarfile.open(snapshot, 'w:gz') as tar:
      for name in files:
        if Path(name).is_file():
          tar.add(name)
    state['snapshot'] = snapshot.name
  return state


def environment() -> dict[str, Any]:
  """The runtime a result came from: versions, device, precision and the lockfile's hash."""
  return {
    'python': platform.python_version(),
    'numpy': np.__version__,
    'device': 'cpu',
    'precision': 'float64 (Python floats and NumPy float64)',
    'uv_lock_sha256': hashlib.sha256(LOCKFILE.read_bytes()).hexdigest() if LOCKFILE.exists() else None,
  }


def save_json(path: Path, record: dict[str, Any]) -> Path:
  path.write_text(json.dumps(record, indent=2, default=str) + '\n')
  return path
