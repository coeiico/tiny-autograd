"""End-to-end runs of train.py: outputs, the run-folder rule, and both engines agreeing."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

import train
from tiny_autograd.data import MoonsArgs
from tiny_autograd.nn import load_mlp


def tiny(run_name: str, **changes: object) -> train.Config:
  cfg = train.Config(
    run_name=run_name,
    eval_every=5,
    data=MoonsArgs(n_train=20, n_val=20, n_test=20),
    model=train.ModelArgs(hidden=(4,)),
    optim=train.OptimArgs(steps=15),
  )
  return replace(cfg, **changes)


def test_a_run_writes_weights_metrics_and_plot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.chdir(tmp_path)
  train.main(tiny('smoke'))
  run_dir = tmp_path / 'runs' / 'smoke'
  record = json.loads((run_dir / 'metrics.json').read_text())
  assert record['status'] == 'finished'
  assert [h['step'] for h in record['history']] == [0, 5, 10, 15]
  assert record['data']['sha256'] and record['config']['seed'] == 0
  assert 0 <= record['test']['accuracy'] <= 1
  assert load_mlp(run_dir / 'weights.json').hidden == (4,)
  assert (run_dir / 'training.png').stat().st_size > 0


def test_an_existing_run_folder_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.chdir(tmp_path)
  train.main(tiny('once'))
  with pytest.raises(FileExistsError):
    train.main(tiny('once'))


def test_scalar_and_tensor_engines_train_identically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  # Same starting weights, same full-batch SGD: the two engines should trace the same path
  # to rounding error, step for step.
  monkeypatch.chdir(tmp_path)
  train.main(tiny('scalar', engine='scalar'))
  train.main(tiny('tensor', engine='tensor'))
  histories = [json.loads((tmp_path / 'runs' / name / 'metrics.json').read_text())['history'] for name in ('scalar', 'tensor')]
  for a, b in zip(*histories, strict=True):
    assert a['train_loss'] == pytest.approx(b['train_loss'], rel=1e-10)
    assert a['val_accuracy'] == b['val_accuracy']


def test_adamw_is_refused_on_the_scalar_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.chdir(tmp_path)
  with pytest.raises(SystemExit, match='adamw'):
    train.main(tiny('adam', optim=train.OptimArgs(optimizer='adamw')))
