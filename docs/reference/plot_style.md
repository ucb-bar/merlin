---
title: Plotting house style
kind: reference
status: current
owner: plotting
last_verified: 2026-07-14
related: []
code_refs: [merlin/python/merlin/plotting]
---

# Merlin plotting style — the house style for every figure in this repo

**All plots in this repository should follow this style.** It is implemented once, in
[`scripts/merlin_plotstyle.py`](../scripts/merlin_plotstyle.py) — import it, don't re-derive it.
The reference implementation that exercises every rule is
[`scripts/plot_presentation.py`](../scripts/plot_presentation.py) (rendered into
`out/artifacts/presentation/`); look there for worked examples of grouped bars, stacked bars, annotated
line plots, flow-chart/tree diagrams, and info-cards.

---

## How to use it

```python
import matplotlib
matplotlib.use("Agg")
from merlin_plotstyle import *      # palette, SERIES, helpers
use_merlin_style()                  # register fonts + apply rcParams (warm cream canvas, ash ink)

fig, ax = plt.subplots(figsize=(8, 5))
style_ax(ax)                        # cream bg, ink left/bottom spines, dotted value grid
vbars(ax, x, heights, NAVY)         # solid bar + hard 3-D block shadow
emph(ax, x0, y0, "ours WINS")       # bold gold emphasis label
title(ax, "My panel")               # DM Serif Display, left-aligned
suptitle(fig, "My figure")
fig.savefig("…/myfig.png", bbox_inches="tight", dpi=170, facecolor=BG)
```

`scripts/` is on the path for repo scripts, so `from merlin_plotstyle import *` just works. New
plotting code goes in `scripts/` (or imports the module by path) — never copy the palette/rcParams
into a new file.

---

## The palette (and what each colour is for)

| hex | name | role |
|---|---|---|
| `#FDF7EF` | cream | **background of every axes + figure** (never white) |
| `#2E2D2C` | ash black (`INK`) | text, axis spines, bar borders, the 3-D block shadow |
| `#AB9A89` | california gold (`GOLD`) | **emphasis text, bold** (the winner, the headline number) |
| `#333351` | indigo (`BLUE`) | cooler emphasis text **and** a bar colour |
| `#0F3759` | deep navy (`NAVY`) | bars — the hero / "ours" series |
| `#8B93A6` | gray-blue (`SLATE`) | bars |
| `#815E5E` | mauve (`MAUVE`) | bars — baseline / killed |
| `#7D886C` | sage (`SAGE`) | bars — OpenBLAS |

Series identity is consistent across figures via the `SERIES` dict
(`baseline`→mauve, `ours`→navy, `xnnpack`→slate, `openblas`→sage, `ceiling`→blue). Reuse it so the
same thing is the same colour in every figure.

## Fonts

- **Titles / suptitles**: `DM Serif Display` (via `title()` / `suptitle()`).
- **Everything else** (axis labels, ticks, annotations, legends): `Inter`.

Both are bundled/installed; `use_merlin_style()` registers them. If you add a glyph that DM Serif
Display lacks (e.g. `→`, `↑`), put it in an Inter `ax.text`, not in a serif title.

---

## The rules

1. **Cream, never white.** `use_merlin_style()` sets figure + axes + savefig facecolor to `BG`;
   always pass `facecolor=BG` to `savefig` too.
2. **Ash ink for structure.** Text, spines, borders, and shadows are `INK`. Only the left and
   bottom spines show (top/right off), via `style_ax`.
3. **Solid fills — colour carries identity.** Do **not** hatch every bar. A hatch is allowed only
   when it *means* something (a conditional / not-directly-comparable bar — e.g. the "large-M-only"
   bars in fig3), and then use a single pattern (`"///"`) and say so in the label with a `*`. Never
   use patterns as decoration.
4. **Bars are 3-D blocks.** Use `vbars` / `hbars` — they draw a hard-edged solid-`INK` block offset
   behind each bar so it reads as a lifted block with volume. For **stacked** bars, pass
   `shadow=False` on the segments and call `block_shadow()` once spanning the whole bar (so the depth
   wraps the total, not each segment). Don't substitute matplotlib's soft blurred shadow.
5. **No caption paragraph baked into the figure.** Presentation figures carry a title and labels
   only; the narrative goes in speaker notes / the surrounding doc. (Keep legends and axis labels —
   those aren't captions.)
6. **Emphasis = gold bold (or blue).** Use `emph()` for the one or two numbers/verdicts that matter
   (`GOLD` default, `BLUE` for the cooler accent). Don't bold everything.
7. **Value axes need not start at 0.** When bars cluster (ratios, speedups), pick limits from the
   data range with `base=`/`set_ylim` so differences are legible — look at the shortest and longest
   bar and choose a sensible floor. Keep a 0 origin only where zero is meaningful (e.g. a count that
   really is 0).
8. **Nothing overlaps.** No text on top of a bar/line, nothing outside the axes. Stagger reference-
   line labels, use callout boxes with leader lines (`annotate(..., arrowprops=...)`), and leave
   headroom (`set_ylim(top=…)`) for value labels.
9. **Diagram cards** (flow charts, trees, info-cards) use rounded `FancyBboxPatch` with the soft
   `SHADOW` path-effect — *not* the hard `block_shadow` (a square shadow pokes out behind rounded
   corners). See `fig_beam_tree` / `fig_cca` for the pattern.
10. **Export both formats.** Save `.png` (dpi ≥ 160) and `.svg` for slides/print.

---

## Helper reference (`merlin_plotstyle`)

| symbol | purpose |
|---|---|
| `use_merlin_style()` | register fonts + apply rcParams — call once per script |
| `style_ax(ax, grid="y")` | cream bg, ink spines, dotted value-axis grid; `grid` ∈ {"y","x","both",None} |
| `vbars / hbars(ax, pos, vals, color, hatch="", base/left=0, shadow=True)` | solid bars + 3-D block shadow |
| `block_shadow(ax, x, y, w, h)` | the hard solid-ink block (used by vbars/hbars; call directly for stacked totals) |
| `title(ax, …)` / `suptitle(fig, …)` | DM Serif Display headings |
| `emph(ax, x, y, text, color=GOLD)` | bold emphasis label |
| `SHADOW` | soft path-effect for rounded diagram cards |
| `BG INK GOLD BLUE NAVY SLATE MAUVE SAGE` · `SERIES` · `SERIF SANS` | palette / identity / font names |

---

## Migrating an existing plot

The older paper figures live in `scripts/plot_paper_style.py` and `scripts/plot_rvv_comparisons.py`
with an inline (pre-house-style) palette. When you next touch one, port it: replace its local
palette/rcParams with `from merlin_plotstyle import *; use_merlin_style()`, swap `ax.bar/barh` for
`vbars/hbars`, drop baked-in caption paragraphs, and remove decorative hatches. The presentation
figures in `out/artifacts/presentation/` are the visual target.
