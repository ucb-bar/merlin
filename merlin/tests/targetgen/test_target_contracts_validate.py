"""Every in-tree target contract validates against the target_contract schema, or is RECORDED debt.

The schema was enforced only on generated manifests, so the hand-authored contracts it describes could
drift from it unseen: measured 2026-09-14, two of the six in-tree contracts miss a required field.
Rather than weaken the schema for every target, or invent contract content to satisfy it, the two are
listed below with the exact problems the validator reports. The test fails when an entry is FIXED (so it
must be deleted) or when its problems CHANGE, so the list can only shrink.
"""
from __future__ import annotations

import pytest

from merlin.common import schemas
from merlin.targetgen import target_registry

#: target -> the problems ``schemas.validate`` reports today. Adding a `legality` list to a tensor-resident
#: contract is a reviewed contract change (the four other contracts use it for human-readable
#: invariants), not something a schema edit or this test should decide.
KNOWN_INVALID: dict[str, list[str]] = {
    "gemmini": ["missing required field 'legality'"],
    "gemmini_universal": ["missing required field 'legality'"],
}


@pytest.mark.parametrize("target", target_registry.list_targets())
def test_contract_validates_or_is_recorded_debt(target):
    problems = schemas.validate(target_registry.load_contract(target), "target_contract")
    expected = KNOWN_INVALID.get(target, [])
    assert problems == expected, (
        f"{target}: validator reports {problems}, expected {expected}. If a recorded problem was fixed, "
        f"delete its KNOWN_INVALID entry; a NEW problem is a contract regression to fix.")


def test_every_recorded_exception_is_still_an_in_tree_target():
    assert set(KNOWN_INVALID) <= set(target_registry.list_targets())
