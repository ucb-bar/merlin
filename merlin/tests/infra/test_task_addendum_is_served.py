"""The merlin-arm addendum REACHES the served TASK.md, and is not silently empty.

WHY THIS IS A TEST AND NOT A CONVENTION. ``run_baseline_qa_loop._build_task`` reads the addendum
``.exists()``-guarded::

    add = (bdir / "TASK_ADDENDUM.md").read_text() if (bdir / "TASK_ADDENDUM.md").exists() else ""
    ws_task.write_text(pilot + "\\n\\n---\\n\\n" + add)

So a bundle that ships no addendum produces a task with an EMPTY addendum section and no error
anywhere: the arm's tool guidance, its close-the-loop cadence and its provenance ask all vanish, the
run completes, and the arm is scored as though it had been told. Measured on ``gemmini_universal``
before this test existed: all six merlin bundles shipped no ``TASK_ADDENDUM.md`` and no
``MERLIN_PROVENANCE_TEMPLATE.md``, so arm-3 and arm-4 would have been launched with the addendum
section blank — an arm contrast between two arms that were handed the same prompt.

Scoped to ``gemmini_universal`` deliberately, and that scope is itself a finding rather than
convenience: the same audit run across every capsule-bench target shows ``radiance``, ``mx_gemmini``,
``saturn_opu`` and the ``*_eqsat_*`` bundles of every target (``gemmini`` included) serving an empty
addendum today. Widening this test would red those targets' suites for a defect this change did not
introduce and is not authorized to fix (their prompts are live experiment inputs). The gap is
reported, not silently generalized.
"""

from __future__ import annotations

import sys

import pytest

from merlin.common.paths import merlin_dir

TARGET = "gemmini_universal"
EXP = merlin_dir() / "experiments/capsule_bench/targets" / TARGET
HARNESS = merlin_dir() / "experiments/capsule_bench/harness"

#: Every bundle ``_build_task`` can serve to the ``merlin_assisted`` arm on this target — the plain
#: merlin arm (arm-3) and the RTL-checks arm (arm-4), across the three experiment shapes. Named rather
#: than globbed: the point is that each one of these is REACHABLE by a launcher, and a glob would
#: silently stop covering a bundle that gets renamed.
SERVED_BUNDLES = [
    "merlin_assisted_public_v0",
    "merlin_assisted_hwbringup_v0",
    "merlin_assisted_realistic_v0",
    "merlin_assisted_rtlchecks_public_v0",
    "merlin_assisted_rtlchecks_hwbringup_v0",
    "merlin_assisted_rtlchecks_realistic_v0",
]

#: The docs ``_build_task`` copies into the agent's workspace (``MERLIN_WS_DOCS``). The addendum TELLS
#: the agent to fill the provenance template in, so shipping one without the other is a prompt that
#: names a file the workspace does not contain.
WS_DOCS = ["TASK_ADDENDUM.md", "ALLOWED_MERLIN_TOOLS.md", "MERLIN_PROVENANCE_TEMPLATE.md"]


@pytest.fixture
def target_loop(monkeypatch):
    """Load this target's native modules without changing another test's cached target."""
    monkeypatch.setenv("MERLIN_TARGET_EXPERIMENT", str(EXP / "target_experiment.yaml"))
    monkeypatch.syspath_prepend(str(HARNESS))
    names = ("_common", "run_baseline_qa_loop", "run_agent_experiment")
    original = {name: sys.modules[name] for name in names if name in sys.modules}
    try:
        for name in names:
            sys.modules.pop(name, None)
        import run_baseline_qa_loop

        yield run_baseline_qa_loop
    finally:
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(original)


@pytest.mark.parametrize("bundle", SERVED_BUNDLES)
@pytest.mark.parametrize("doc", WS_DOCS)
def test_every_served_merlin_bundle_ships_the_workspace_docs(bundle, doc):
    p = EXP / "input_bundles" / bundle / doc
    assert p.is_file(), f"{p} is missing — _build_task would serve an empty section, silently"
    assert p.stat().st_size > 0, f"{p} is empty"


@pytest.mark.parametrize("bundle", SERVED_BUNDLES)
def test_the_addendum_states_this_devices_readout_restriction(bundle):
    """The one fact this device's agent must have up front. It is INFRA information (which readout
    ports the hardware has), not a target answer: the graded cohort is already shaped by it, so
    withholding it only buys a run in which the agent discovers via all-zeros results that the mvout
    width it chose does not exist."""
    text = (EXP / "input_bundles" / bundle / "TASK_ADDENDUM.md").read_text(encoding="utf-8")
    assert "READOUT RESTRICTION" in text
    assert "ACC_READ_FULL_WIDTH" in text  # the ABI symbol the agent can check for itself
    assert "128" in text  # the store-DMA width
    # And the blindness of the fast oracle, which is the part an agent cannot discover from the ABI.
    assert "not evidence" in text or "does not enforce" in text.lower()


@pytest.mark.parametrize("bundle", SERVED_BUNDLES)
def test_the_addendum_names_this_target_and_not_the_sibling_elaboration(bundle):
    """A prompt copied from the sibling target would name the sibling's dialect, its RTL-facts
    directory and its seam-menu calls — pointing the agent at another device's facts."""
    text = (EXP / "input_bundles" / bundle / "TASK_ADDENDUM.md").read_text(encoding="utf-8")
    for stale in (
        "check_bijection('gemmini')",
        "target_profile('gemmini')",
        "load_facts('gemmini')",
        "emit gemmini-dialect",
        "merlin/targets/gemmini/contracts/rtl_facts/",
    ):
        assert stale not in text, f"{bundle}: addendum still names the sibling elaboration: {stale}"
    assert f"check_bijection('{TARGET}')" in text


@pytest.mark.parametrize(
    "bundle,experiment",
    [
        ("merlin_assisted_public_v0", "full"),
        ("merlin_assisted_rtlchecks_public_v0", "full"),
        ("merlin_assisted_realistic_v0", "realistic"),
        ("merlin_assisted_rtlchecks_realistic_v0", "realistic"),
    ],
)
def test_the_rendered_task_carries_a_non_empty_addendum_section(bundle, experiment, tmp_path, target_loop):
    """END TO END, through the real ``_build_task``: render the task the launcher would serve and
    assert the addendum section actually arrived. Asserting the FILE exists is not enough — the
    silent-empty defect lives in the render, not in the directory listing."""
    L = target_loop
    ws, run_dir = tmp_path / "ws", tmp_path / "run"
    ws.mkdir()
    run_dir.mkdir()
    from merlin_experiments.phase1 import task_staging as TS

    TS.build_task(
        "merlin_assisted",
        ws,
        run_dir,
        sandbox="bwrap",
        config=TS.TaskStagingConfig(L.C.CONTEXT, bundle, L.C.BUNDLES / bundle, experiment),
    )
    task = (ws / "TASK.md").read_text(encoding="utf-8")

    marker = "# Addendum —"
    assert marker in task, f"{bundle}/{experiment}: the served TASK.md has NO addendum section"
    body = task.split(marker, 1)[1].strip()
    assert len(body) > 2000, f"{bundle}/{experiment}: addendum section is {len(body)} bytes"
    assert "READOUT RESTRICTION" in body
    assert "MERLIN_PROVENANCE_TEMPLATE.md" in body  # the provenance ask survived
    # …and the file the ask names is actually in the workspace the agent gets.
    for doc in WS_DOCS:
        assert (ws / doc).is_file(), f"{doc} was not staged into the workspace"
