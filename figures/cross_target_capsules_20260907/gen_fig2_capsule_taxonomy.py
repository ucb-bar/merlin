#!/usr/bin/env python3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

from plot_common import *

use_merlin_style()
d = load_snapshot()
caps = d["capsules"]
families = [k for k, _ in Counter(r["family"] for r in caps).most_common()]
families = families[:10]
counts = np.zeros((len(families), 3))
rates = np.full_like(counts, np.nan)
for yi, family in enumerate(families):
    for xi, target in enumerate(TARGETS):
        cell = [r for r in caps if r["target"] == target and r["family"] == family]
        counts[yi, xi] = len(cell)
        if cell:
            rates[yi, xi] = sum(r["status"] == "pass" for r in cell) / len(cell)

fig, ax = plt.subplots(figsize=(8.0, 6.0))
ax.set_facecolor(BG)
im = ax.imshow(rates, vmin=0, vmax=1, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("merlin", [MAUVE, GOLD, SAGE]))
ax.set_xticks(range(3), [LABELS[t] for t in TARGETS])
ax.set_yticks(range(len(families)), [f.title() for f in families])
for y in range(len(families)):
    for x in range(3):
        if counts[y, x]:
            passed = round(rates[y, x] * counts[y, x])
            ax.text(x, y, f"{passed}/{int(counts[y,x])}", ha="center", va="center",
                    color=INK, fontweight="bold", fontsize=10)
        else:
            ax.text(x, y, "—", ha="center", va="center", color=INK, alpha=.55)
for s in ax.spines.values():
    s.set_color(INK)
    s.set_linewidth(1.0)
cbar = fig.colorbar(im, ax=ax, fraction=.04, pad=.04)
cbar.set_label("Pass fraction")
ax.text(-.02, 1.025, "Cell labels are passed / graded capsules", transform=ax.transAxes, color=GOLD, fontweight="bold")
fig.tight_layout()
save_all(fig, "fig2_capsule_taxonomy")
