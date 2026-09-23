"""Which classes a target DECLARES, against which ones a capsule can ever be required to issue.

``corpus_spec._classes_source`` now reads the target's authored ``encoding.corpus_issue_order``.
The shared code must neither name nor silently filter out an instruction class. The target may still
declare more alternatives than its current baseline corpus demands, and this test keeps those
declarations visible until operation-specific capsules cover them.

This is a RATCHET, not a fix. It pins the gap so it is visible and cannot grow, and it fails if a
declared class starts being dropped that was reachable before. Closing it means deriving the
requirement from the declared role rather than extending the literal -- and deciding subsumption, since
a capsule owing LOOP_WS must not ALSO owe the PRELOAD/COMPUTE pair the loop performs itself.
"""

from __future__ import annotations

import pytest
import yaml

from merlin.common.paths import repo_root

pytestmark = pytest.mark.target("gemmini")

#: Classes these targets DECLARE that no capsule can be required to issue. Measured 2026-09-18. This
#: may only SHRINK: an entry removed means the requirement became reachable, which is the fix.
_UNREACHABLE: dict[str, set[str]] = {
    "gemmini": {"COMPUTE_ACCUMULATE", "LOOP_CONV", "LOOP_WS", "MVIN2"},
}


def _contract(target: str) -> dict:
    return yaml.safe_load((repo_root() / "examples" / target / "target/contracts/target_contract.yaml").read_text())


def _declared_pool(target: str) -> set[str]:
    """What the target's own encoding defines, exactly as `_classes_source` computes it."""
    contract = _contract(target)
    enc = contract.get("encoding") or {}
    semantic = set((enc.get("semantic_class") or {}).values())
    subtypes = set((enc.get("config_subtype") or {}).values())
    return semantic | subtypes


def _issue_order(target: str) -> set[str]:
    return set((_contract(target).get("encoding") or {}).get("corpus_issue_order") or [])


def test_the_declared_but_unreachable_set_is_exactly_what_was_measured():
    """If this fails because the set SHRANK, delete the entry -- the requirement became reachable."""
    for target, expected in _UNREACHABLE.items():
        pool = _declared_pool(target)
        assert pool, f"{target} declares no semantic classes at all"
        unreachable = pool - _issue_order(target) - {"CONFIG"}  # CONFIG has only declared subtypes here
        assert unreachable == expected, (
            f"{target}: declared-but-unrequirable classes changed from {sorted(expected)} to "
            f"{sorted(unreachable)}. A class that became reachable is the fix — remove it here. A NEW "
            f"one is a regression: the corpus can no longer demand a capability the target declares."
        )


def test_the_device_loop_family_is_among_the_unreachable():
    """Named explicitly because it is the one this cost: measured on ResNet-50, the program used 8 of
    the 25 functs the RTL declares, and the 17 it never emitted are this family."""
    unreachable = _declared_pool("gemmini") - _issue_order("gemmini") - {"CONFIG"}
    assert {"LOOP_WS", "LOOP_CONV"} <= unreachable, (
        "the loop family is now requirable — close the ratchet entry and make sure a capsule owing "
        "LOOP_WS does not ALSO owe the PRELOAD/COMPUTE pair the loop issues itself"
    )


def test_every_class_the_contract_requires_is_actually_declared():
    """The other direction: a demanded class the target does not declare is fabricated coverage."""
    pool = _declared_pool("gemmini")
    order = _issue_order("gemmini")
    assert order <= pool, f"the issue order lists classes gemmini does not declare: {sorted(order - pool)}"
