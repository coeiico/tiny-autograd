"""
Arrange the unit ablations for plotting: each layer's units sorted by cost, one column per seed.

Writes build/unit-costs.csv: rank (1 = the unit whose removal costs most), then the cost in
points of test accuracy for each layer and seed. Sorting only; the numbers are the files'.
"""

import csv

columns: dict[str, list[float]] = {}
for seed in (0, 1, 2):
  with open(f'data/units-default-seed{seed}.csv') as f:
    units = list(csv.DictReader(f))
  for layer in ('1', '2'):
    columns[f'l{layer}s{seed}'] = sorted((100 * float(u['accuracy_drop']) for u in units if u['layer'] == layer), reverse=True)

with open('build/unit-costs.csv', 'w', newline='') as f:
  writer = csv.writer(f)
  writer.writerow(['rank', *columns])
  for i in range(len(columns['l1s0'])):
    writer.writerow([i + 1, *(column[i] for column in columns.values())])
