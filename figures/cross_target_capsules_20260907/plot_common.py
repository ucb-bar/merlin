from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "merlin/python"))

from merlin.plotting.merlin_plotstyle import *  # noqa: F401,F403,E402

TARGETS = ["gemmini", "atlas", "radiance"]
LABELS = {"gemmini": "Gemmini", "atlas": "Atlas", "radiance": "Radiance"}
COLORS = {"gemmini": NAVY, "atlas": BLUE, "radiance": SAGE}


def load_snapshot():
    return json.loads((HERE / "cross_target_snapshot.json").read_text())


def save_all(fig, stem: str):
    for suffix, kwargs in (("pdf", {}), ("svg", {}), ("png", {"dpi": 190})):
        fig.savefig(HERE / f"{stem}.{suffix}", bbox_inches="tight", facecolor=BG, **kwargs)
