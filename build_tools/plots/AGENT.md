# AGENT.md — build_tools/plots

Standalone figure-generation scripts (run directly: `python build_tools/plots/<name>.py`). They
import the shared house style from the library (`from merlin.plotting.merlin_plotstyle import *`)
and write PNGs under `artifacts/`. They are tools, not library code — nothing imports them. The
reusable style lives in `merlin/python/merlin/plotting/` (`merlin_plotstyle`, `plot_paper_style`).
