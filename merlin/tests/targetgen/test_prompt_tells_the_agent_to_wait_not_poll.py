"""The generated task must point at the BLOCKING waits, and must not prescribe polling.

`await_verdict.py`, `agent_selfcheck.py` and `simjob.py` are all staged into every workspace and all
block. The template nonetheless told every arm to "Re-read it periodically", and the arm-4 branch added
"read the `rtl_checks` block ... each round" — so the prompt argued with itself about whether to wait or
to look again. MEASURED on the gemmini arm-4 run of 2026-09-08: 118 of 786 tool calls were `sleep`/poll
loops, 56 min of wall clock and ~18M tokens, while its atlas sibling on the same treatment spent none.
Every look is a whole turn, so this is a prompt defect with a token price.

These assertions are on the RENDERED prompt (`render_prompt`), because that is what is served in the
`full` experiment — the on-disk `STARTER_PROMPT.md` is write-if-absent and never read there.
"""
from __future__ import annotations

import functools
import os

import pytest

from merlin.common.paths import merlin_dir, repo_root
from merlin.targetgen.generate_prompt import render_prompt
from merlin.targetgen.target_experiment import load_capability_manifest, load_target_experiment

TARGET = "gemmini"
DESCRIPTOR = merlin_dir() / "experiments/capsule_bench/targets/gemmini/target_experiment.yaml"
BUNDLES = merlin_dir() / "experiments/capsule_bench/targets/gemmini/input_bundles"
ARMS = ("raw_baseline_public_v0", "cpp_merlininfra_public_v0",
        "merlin_assisted_public_v0", "merlin_assisted_rtlchecks_public_v0")


@functools.lru_cache(maxsize=None)
def _rendered(stem: str) -> str:
    # render_prompt loads RTL facts + the capability manifest; caching keeps this suite seconds, not
    # minutes, since every assertion re-reads the same four prompts.
    if not DESCRIPTOR.is_file():
        pytest.skip(f"descriptor absent: {DESCRIPTOR}")
    os.environ.setdefault("MERLIN_TARGET_EXPERIMENT", str(DESCRIPTOR))
    te = load_target_experiment(DESCRIPTOR)
    manifest = load_capability_manifest(TARGET)
    allowed = BUNDLES / stem / "allowed_files.txt"
    granted = []
    if allowed.is_file():
        granted = [ln.strip() for ln in allowed.read_text().splitlines()
                   if ln.strip().startswith(("merlin/", "experiments/"))]
    return render_prompt(te, manifest, "full", stem, granted_tools=granted)


@pytest.mark.parametrize("stem", ARMS)
def test_every_arm_is_told_about_the_blocking_wait(stem):
    text = _rendered(stem)
    assert text.count("await_verdict") >= 2, "the blocking wait must be named, not left to discovery"
    assert "simjob" in text, "the async sim broker is staged for every arm and must be named"


@pytest.mark.parametrize("stem", ARMS)
def test_no_arm_is_told_to_poll(stem):
    text = _rendered(stem)
    assert "Re-read it periodically" not in text
    assert "each round" not in text, "a per-round re-read directive is a polling instruction"


@pytest.mark.parametrize("stem", ARMS)
def test_every_arm_gets_the_close_the_loop_cadence(stem):
    """The measured winning behaviour: edit -> build -> self-check -> read the verdict."""
    text = _rendered(stem)
    assert "CLOSE THE LOOP" in text
    assert "agent_selfcheck.py" in text


def test_the_rtl_arm_is_pointed_at_the_wait_too():
    """arm-4 carries an EXTRA verdict-reading directive; it must not become an extra poll."""
    text = _rendered("merlin_assisted_rtlchecks_public_v0")
    assert "rtl_checks" in text
    assert text.count("await_verdict") >= 3, \
        "arm-4's rtl_checks directive must itself point at the blocking wait"


def test_the_cadence_is_arm_invariant():
    """No arm may be coached better than another: only tool grants differ between rungs."""
    counts = {s: (_rendered(s).count("CLOSE THE LOOP"), _rendered(s).count("simjob")) for s in ARMS}
    assert len(set(counts.values())) == 1, f"cadence guidance differs per arm: {counts}"
