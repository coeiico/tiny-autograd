"""
What the hidden layers compute, drawn into build/units.pgf for units.tex.

Layer-1 units are drawn as the lines they switch on across, layer-2 units as contour maps.

Layer-1 unit n is ReLU(w₁x₁ + w₂x₂ + b): zero on one side of the line w·x + b = 0 and rising
on the other. The units whose removal costs more than 5 points of test accuracy are drawn in
spot, with their active side hatched. Weights and costs come from data/units-*.csv, the maps
from data/hidden-*.npz.
"""

import csv
import sys

sys.path.insert(0, 'tex')
import figstyle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RUN = 'default-seed0'
KEY = 0.05  # accuracy cost above which a unit counts as key

figstyle.setup()
hidden = np.load(f'data/hidden-{RUN}.npz')
points = np.load(f'data/boundary-{RUN}.npz')
with open(f'data/units-{RUN}.csv') as f:
  units = list(csv.DictReader(f))
xs, ys = hidden['xs'], hidden['ys']
x_lo, x_hi, y_lo, y_hi = xs.min(), xs.max(), ys.min(), ys.max()

w_in = figstyle.FIGWIDTH_PT / 72.27
fig = plt.figure(figsize=(w_in, w_in * 0.47), layout='constrained')
left, right = fig.subfigures(1, 2, width_ratios=[1.45, 1], wspace=0.05)


def visible_segment(w: np.ndarray, b: float) -> tuple[np.ndarray, np.ndarray]:
  """The two ends of the line w·x + b = 0 inside the plot frame."""
  norm = np.linalg.norm(w)
  p0, direction = -b / norm**2 * w, np.array([-w[1], w[0]]) / norm
  inside = [p0 + t * direction for t in np.linspace(-8, 8, 8001)]
  inside = [p for p in inside if x_lo <= p[0] <= x_hi and y_lo <= p[1] <= y_hi]
  return inside[0], inside[-1]


# (a) Layer 1.
ax = left.subplots()
for label, marker in ((0, 'o'), (1, '.')):
  pts = points['test_x'][points['test_y'] == label]
  ax.scatter(pts[:, 0], pts[:, 1], s=1.2 if marker == 'o' else 1.0, marker=marker, facecolors='none' if marker == 'o' else figstyle.FAINT,
             edgecolors=figstyle.FAINT, linewidths=0.25, alpha=0.6, rasterized=True)
key_units = []
for u in (u for u in units if u['layer'] == '1'):
  w, b = np.array([float(u['w1']), float(u['w2'])]), float(u['b'])
  start, end = visible_segment(w, b)
  if u['dead_on_test'] == '1':
    ax.plot(*zip(start, end, strict=True), color=figstyle.FAINT, linewidth=0.4, linestyle=(0, (2, 2)))
  elif float(u['accuracy_drop']) > KEY:
    key_units.append((u, w, b, start, end))
  else:
    ax.plot(*zip(start, end, strict=True), color=figstyle.INK, linewidth=0.35, alpha=0.4)
ax.contour(xs, ys, hidden['logits'], levels=[0], colors=figstyle.INK, linewidths=0.9)

placed: list[np.ndarray] = []
for u, w, _b, start, end in key_units:
  normal = w / np.linalg.norm(w)
  # The active side (w·x + b > 0) as a narrow hatched band along the line.
  band = np.array([start, end, end + 0.075 * normal, start + 0.075 * normal])
  ax.fill(band[:, 0], band[:, 1], facecolor='none', edgecolor=figstyle.SPOT, hatch='//////', linewidth=0)
  ax.plot(*zip(start, end, strict=True), color=figstyle.SPOT, linewidth=0.9)
  # The unit's number near one end of its line, whichever end keeps clear of the others.
  top, bottom = (start, end) if start[1] > end[1] else (end, start)
  for a, other in ((top, bottom), (bottom, top)):
    at = a + 0.2 * (other - a) / np.linalg.norm(other - a) + 0.3 * normal
    if all(np.linalg.norm(at - q) > 0.4 for q in placed):
      break
  placed.append(at)
  ax.text(*at, rf'\textit{{{int(u["unit"]) + 1}}}', color=figstyle.SPOT, fontsize=8, ha='center', va='center',
          bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 0.6})

costs = sorted(key_units, key=lambda k: -float(k[0]['accuracy_drop']))
ax.text(2.8, -0.44, r'\textit{removing one unit costs}', fontsize=7.5, color=figstyle.SOFT, ha='right', va='bottom',
        bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1.5})
rows = '\n'.join(rf'\textit{{unit {int(u["unit"]) + 1}: {100 * float(u["accuracy_drop"]):.1f} points}}' for u, *_ in costs)
ax.text(2.8, -0.48, rows, fontsize=7.5, color=figstyle.SPOT, ha='right', va='top', linespacing=1.35,
        bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1.5})
ax.set_aspect('equal')
ax.set_xlim(x_lo, x_hi)
ax.set_ylim(y_lo, y_hi)
ax.set_xticks([-1, 0, 1, 2])
ax.set_yticks([-1, 0, 1])
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
left.suptitle(r'\textit{(a)\enspace Layer 1: each unit switches on across one line.}', x=0.02, ha='left', fontsize=9, color=figstyle.INK)

# (b) Layer 2: each unit's activation as contour lines, its silent region hatched.
acts = hidden['layer2']
small = right.subplots(4, 4, sharex=True, sharey=True)
for n, sax in enumerate(small.flat):
  a = acts[:, :, n]
  if a.max() > 0:
    sax.contourf(xs, ys, a, levels=[-1, 1e-6], colors='none', hatches=['///'])
    sax.contour(xs, ys, a, levels=np.linspace(0, a.max(), 6)[1:-1], colors=figstyle.INK, linewidths=0.3)
  else:
    sax.contourf(xs, ys, a, levels=[-1, 1], colors='none', hatches=['///'])
  sax.contour(xs, ys, hidden['logits'], levels=[0], colors=figstyle.SPOT, linewidths=0.6)
  sax.set_xticks([])
  sax.set_yticks([])
  sax.set_aspect('equal')
  for spine in sax.spines.values():
    spine.set_linewidth(0.4)
  sax.set_title(rf'\textit{{{n + 1}{", silent" if a.max() <= 0 else ""}}}', loc='left', fontsize=7, pad=1.5, color=figstyle.INK)
right.suptitle(r'\textit{(b)\enspace Layer 2: what each unit responds to.}', x=0.02, ha='left', fontsize=9, color=figstyle.INK)
figstyle.save(fig, 'build/units')
