"""Original measured activity fixture shared by the three performance analyses.

This intake supplies observations only; each suite keeps its own independent
expected totals, classifications, attribution and unknown/refusal assertions.
"""

import json
from pathlib import Path

import pytest

from merlin.common.paths import env, repo_root
from merlin.perf.decompose import ResourceKind, activity_from_busy

BUCKET_KINDS = {
    "dma": ResourceKind.MOVEMENT,
    "mxu": ResourceKind.COMPUTE,
    "vpu": ResourceKind.COMPUTE,
    "none": ResourceKind.FIXED,
}


def suite(*, measured=False):
    root = env("MERLIN_MLC_DIR")
    label = "measured cycle suite" if measured else "activity fixture"
    missing = "measured cycle suite" if measured else "per-unit activity fixture"
    if not root:
        pytest.skip(f"MERLIN_MLC_DIR unset -- the {missing} lives in the mlc checkout")
    path = Path(root) / "mlc" / "validate" / "npu_model_suite.json"
    if not path.is_file():
        pytest.skip(f"{label} not present at {path}")
    assert repo_root().is_dir()
    return json.loads(path.read_text(encoding="utf-8"))


def sources(suite):
    out = []
    for name, body in suite["kernels"].items():
        arc = body["arc"]
        out.append(
            activity_from_busy(
                name,
                arc["truth"],
                {"dma": arc["dma_busy"], "mxu": arc["mxu"], "vpu": arc["vpu"], "none": arc["none"]},
                BUCKET_KINDS,
                partitioned=True,
                completion_observable=True,
                provenance="per-cycle activity decomposition from the cycle-accurate model",
            )
        )
    return out
