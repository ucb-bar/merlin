from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / "merlin/python"))  # for a run without the editable install

from merlin.common.paths import artifacts_dir, repo_root  # noqa: E402

ROOT = repo_root()
OUT = artifacts_dir() / "paper-figures" / HERE.name

from merlin.plotting.merlin_plotstyle import *  # noqa: F401,F403,E402

TARGETS = ["gemmini", "atlas", "radiance"]
LABELS = {"gemmini": "Gemmini", "atlas": "Atlas", "radiance": "Radiance"}
COLORS = {"gemmini": NAVY, "atlas": BLUE, "radiance": SAGE}


def load_snapshot():
    return json.loads((HERE / "cross_target_snapshot.json").read_text())


def save_all(fig, stem: str):
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (("pdf", {}), ("svg", {}), ("png", {"dpi": 190})):
        fig.savefig(OUT / f"{stem}.{suffix}", bbox_inches="tight", facecolor=BG, **kwargs)
