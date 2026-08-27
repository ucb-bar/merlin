"""A facts regeneration may not HOLLOW OUT an artifact.

The dangerous case is not the empty artifact (:class:`FactsEmpty` already covers that) but the one that
regenerates fully populated with individual facts emptied, because an optional extractor was missing.
Measured on the muon artifact: without ``MERLIN_MLC_DIR`` the regeneration keeps every key and still
turns ``instruction_classes`` into ``[]`` and ``address_spaces`` into ``None``. Nothing failed, and the
one guard that exists (``_warn_if_degraded``) sits in ``ensure_facts`` -- not on the path a person runs
by hand. Downstream, an empty ``instruction_classes`` reads as "this endpoint has no ISA", the opposite
of the truth.

These pin the ratchet: evidence may only get richer, unless a caller explicitly says the hardware lost it.
"""
from __future__ import annotations

import json

import pytest

from merlin.targetgen.rtl.facts import FactsDowngrade, hollowed_facts, write_facts_guarded

RICH = {"facts": {
    "isa": {"instruction_classes": ["AUIPC", "BRANCH"], "address_spaces": {"global": 0},
            "max_src_operands": 3},
    "simt": {"cores": 2},
}}
# what a regeneration WITHOUT the mlc toolchain actually produces: same keys, hollowed values
POOR = {"facts": {
    "isa": {"instruction_classes": [], "address_spaces": None, "max_src_operands": 4},
    "simt": {"cores": 2},
}}


def test_the_exact_mlc_absent_downgrade_is_named():
    lost = hollowed_facts(RICH, POOR)
    assert lost == ["isa.address_spaces", "isa.instruction_classes"], lost


def test_a_value_that_merely_changes_is_not_a_downgrade():
    """max_src_operands 3 -> 4 is a different value, not a lost one. If ordinary churn tripped the
    guard, the guard would be turned off, and it has to stay on to be worth anything."""
    assert "isa.max_src_operands" not in hollowed_facts(RICH, POOR)


def test_a_richer_regeneration_is_not_a_downgrade():
    richer = json.loads(json.dumps(RICH))
    richer["facts"]["console"] = {"base": 0x10020000}
    richer["facts"]["isa"]["instruction_classes"].append("CUSTOM0")
    assert hollowed_facts(RICH, richer) == []


def test_a_whole_dropped_fact_counts():
    dropped = {"facts": {"simt": RICH["facts"]["simt"]}}
    assert "isa" in hollowed_facts(RICH, dropped)


def test_the_write_refuses_and_leaves_the_artifact_untouched(tmp_path):
    p = tmp_path / "facts.json"
    write_facts_guarded(p, RICH)
    with pytest.raises(FactsDowngrade) as e:
        write_facts_guarded(p, POOR)
    # the message must NAME what would be lost and point at the cause, or it sends the reader hunting
    assert "instruction_classes" in str(e.value)
    assert "MERLIN_MLC_DIR" in str(e.value)
    assert json.loads(p.read_text())["facts"]["isa"]["instruction_classes"] == ["AUIPC", "BRANCH"]


def test_a_genuine_hardware_loss_can_be_declared(tmp_path):
    p = tmp_path / "facts.json"
    write_facts_guarded(p, RICH)
    write_facts_guarded(p, POOR, allow_downgrade=True)
    assert json.loads(p.read_text())["facts"]["isa"]["instruction_classes"] == []


def test_a_first_write_is_never_blocked(tmp_path):
    p = tmp_path / "facts.json"
    write_facts_guarded(p, POOR)          # nothing to lose yet
    assert p.is_file()
