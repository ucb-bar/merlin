# AGENT.md — build_tools/plots

Standalone figure-generation scripts (run directly: `python build_tools/plots/<name>.py`). They
import the shared house style from the library (`from merlin.plotting.merlin_plotstyle import *`)
and write their renders under `out/artifacts/` -- never beside the script. They are tools, not
library code; nothing imports them. The reusable style lives in `merlin/python/merlin/plotting/`
(`merlin_plotstyle`, `plot_paper_style`).

- `paper_figures/` -- generators for the paper's K1, capsule-generation, gemmini phase-2 and
  cross-target campaign figures, with the frozen source data (json/csv) they plot kept beside them.
  Renders go to `out/artifacts/paper-figures/<set>/`. The `latex_includes.tex` files name the paper
  repository's own `figures/` paths, not this repository's.
- `snippet2svg/` -- code-snippet to SVG renderer (see its README).
