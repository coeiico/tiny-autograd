"""
Prepare the decision-boundary figure: contour lines, regions and points in build/.

For each model, from its data/boundary-*.npz grid of logits: the boundary (logit 0) and the
lines where the model is 90% sure either way (logit ±ln 9), the region it classifies as the
lower moon (logit > 0) as polygons, and the test points with the model's verdict. Also the
plot extent as macros. Contour extraction only; the logits are the file's.
"""

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LN9 = np.log(9)


def write_segments(path: str, segments: list[np.ndarray]) -> None:
  """Write `x y` rows, with a blank line between segments (pgfplots: empty line=jump)."""
  with open(path, 'w') as f:
    f.write('x y\n')
    for segment in segments:
      f.writelines(f'{x:.4f} {y:.4f}\n' for x, y in segment)
      f.write('\n')


for run, tag in (('default-seed0', 'mlp'), ('linear-seed0', 'lin')):
  d = np.load(f'data/boundary-{run}.npz')
  xs, ys, logits = d['xs'], d['ys'], d['logits']
  fig, ax = plt.subplots()
  lines = ax.contour(xs, ys, logits, levels=[-LN9, 0, LN9])
  for level, segments in zip(('neg', 'zero', 'pos'), lines.allsegs, strict=True):
    write_segments(f'build/contour-{tag}-{level}.dat', segments)
  region = ax.contourf(xs, ys, logits, levels=[0, np.inf])
  write_segments(f'build/region-{tag}.dat', region.allsegs[0])
  plt.close(fig)

  wrong = (d['test_logits'] > 0) != (d['test_y'] == 1)
  with open(f'build/points-{tag}.dat', 'w') as f:
    f.write('x y cls wrong\n')
    f.writelines(f'{x:.4f} {y:.4f} {c} {int(w)}\n' for (x, y), c, w in zip(d['test_x'], d['test_y'], wrong, strict=True))

d = np.load('data/boundary-default-seed0.npz')
with open('build/extent.tex', 'w') as f:
  f.write(f'\\def\\xmin{{{d["xs"].min():.4f}}}\\def\\xmax{{{d["xs"].max():.4f}}}'
          f'\\def\\ymin{{{d["ys"].min():.4f}}}\\def\\ymax{{{d["ys"].max():.4f}}}\n')
