"""A bundle must resolve to ITS OWN arm, including an arm outside the default ladder.

`_arm_from_bundle_id` maps a bundle id back to the ladder rung whose tool grants the run should get.
Getting it wrong does not fail: the run proceeds with another arm's tools, under this arm's name, and
the result is attributed to a seam the agent never had. That is the one defect class an A/B ladder
cannot survive, because nothing downstream can detect it.

**The live case.** Every opt-in arm's stem deliberately CONTAINS `merlin_assisted`, because
`generate_prompt._is_assisted_arm` is a substring test and the arm should inherit the assisted prompt
with no prompt edit. So resolving against the default ladder alone does not REFUSE an opt-in bundle --
it silently mis-resolves it to `merlin_assisted`. Measured 2026-09-05:
`merlin_assisted_verify_public_v0` (a bundle that exists on disk for gemmini) resolved to
`merlin_assisted`, so the verify arm would have run with arm-3's grants and no verification seam at
all -- an arm gaining nothing while its result was read as evidence about the seam.
"""
from __future__ import annotations

import importlib.util

import pytest

from merlin.common.paths import merlin_dir

HARNESS = merlin_dir() / "experiments/capsule_bench/harness"


def _resolver():
    """Import the harness script by path — it is a script, not an installed package."""
    import sys

    path = HARNESS / "run_baseline_qa_loop.py"
    if not path.is_file():
        pytest.skip("the capsule-bench harness is not in this checkout")
    added = str(HARNESS)
    if added not in sys.path:
        sys.path.insert(0, added)
    spec = importlib.util.spec_from_file_location("run_baseline_qa_loop", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:                      # noqa: BLE001 — harness deps absent in this env
        pytest.skip(f"harness not importable here: {type(exc).__name__}: {exc}")
    return module._arm_from_bundle_id


def test_every_arm_including_an_opt_in_one_resolves_to_itself():
    """Derived from the arm table rather than a list typed here, so a new arm is covered on arrival."""
    from merlin.targetgen.generate_bundles import _ALL_ARMS

    resolve = _resolver()
    for arm, stem in _ALL_ARMS.items():
        for variant in ("public_v0", "hwbringup_v0"):
            bundle_id = f"{stem}_{variant}"
            assert resolve(bundle_id) == arm, (
                f"{bundle_id} resolved to {resolve(bundle_id)!r}, not {arm!r}; it would run with "
                f"another arm's tool grants under this arm's name")


def test_the_verify_arm_is_the_only_one_granted_the_verification_seam():
    """An arm that gains two things at once produces a result attributable to neither."""
    from merlin.targetgen.generate_bundles import _ALL_ARMS
    from merlin.targetgen.tool_registry import ARM_TOOLS

    holders = [arm for arm in _ALL_ARMS if "verify_seam" in ARM_TOOLS.get(arm, ())]
    assert holders == ["merlin_verify"], (
        f"the verification seam is granted to {holders}; it must reach exactly the verify arm, or the "
        f"comparison measures the seam against arms that also have it")


def test_an_unknown_bundle_id_raises_rather_than_defaulting():
    """Fail closed. A bundle that matches no rung must not quietly become the first plausible arm."""
    resolve = _resolver()
    with pytest.raises(KeyError):
        resolve("not_an_arm_public_v0")


def test_a_nested_stem_resolves_to_the_longer_one():
    """The original reason this function is longest-match: the stems nest.

    `merlin_assisted_rtlchecks_*` also starts with `merlin_assisted_`, and picking the shorter stem
    downgrades the CIRCT arm to the xDSL arm — the same silent mis-attribution as the opt-in case,
    which is why both belong in one test file.
    """
    resolve = _resolver()
    assert resolve("merlin_assisted_rtlchecks_public_v0") == "merlin_rtlchecks"
    assert resolve("merlin_assisted_public_v0") == "merlin_assisted"
