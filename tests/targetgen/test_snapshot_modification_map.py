"""Snapshot regression for modification_map output.

For each capability fixture under ``target_specs/examples/``, dump the
``ModificationMap`` to canonical JSON and diff against the committed golden
file at ``tests/targetgen/golden/<target>.modification_map.json``. Any
intentional planner change is reviewed via the diff in the snapshot file.

To refresh snapshots after an intentional planner change:

    UPDATE_SNAPSHOTS=1 conda run -n merlin-dev python -m pytest \
        tests/targetgen/test_snapshot_modification_map.py
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
from conftest import REPO_ROOT, all_capability_specs, assert_text_equals, normalise_json
from targetgen import build_support_plan, load_capability_spec
from targetgen.stage_map import build_modification_map

GOLDEN_DIR = REPO_ROOT / "tests" / "targetgen" / "golden"


@pytest.mark.parametrize(
    "capability_path",
    all_capability_specs(),
    ids=lambda p: p.parent.name,
)
def test_modification_map_snapshot(capability_path: Path) -> None:
    capabilities = load_capability_spec(capability_path)
    plan = build_support_plan(capabilities)
    modmap = build_modification_map(capabilities, targetgen_styles=plan.integration_styles)
    actual = normalise_json(asdict(modmap))
    snapshot = GOLDEN_DIR / f"{capability_path.parent.name}.modification_map.json"
    assert_text_equals(actual, snapshot, label=capability_path.parent.name)
