"""Every fresh campaign spec gives the broker the reference-analysis experiment owner."""

from pathlib import Path

import pytest
from test_fresh_campaign import F  # noqa: F401  (pytest fixture)

HERE = Path(__file__).resolve().parent
SPECS = sorted(HERE.glob("fresh-campaign-*.yaml"))


@pytest.mark.parametrize("path", SPECS, ids=lambda path: path.stem)
def test_campaign_pythonpath_includes_merlin_mining(F, path):
    environment = F.load_spec(path).get("plan", {}).get("environment", {})
    entries = environment["PYTHONPATH"].split(":")
    # reference_experiments.record_analysis imports merlin.mining in the broker environment.
    assert "${FRAMEWORK}/packages/merlin-mining/src" in entries
    assert entries.index("${FRAMEWORK}/packages/merlin-experiments/src") < entries.index(
        "${FRAMEWORK}/packages/merlin-mining/src")
