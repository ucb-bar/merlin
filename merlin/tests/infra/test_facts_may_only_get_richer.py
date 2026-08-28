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

from merlin.targetgen.rtl.facts import (
    FactsDowngrade, _read_facts_doc, _refuse_hollowed, hollowed_facts, write_facts_guarded,
)

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


# --- the ratchet is a property of every FAMILY, not of the one target that exposed it -------------

def test_every_declared_family_routes_through_the_guard():
    """Each compute-unit kind declares a ``fact_extractor``; every one of those paths must be guarded.
    A family added later inherits it because the seam is shared — whoever adds the next NPU/GPU/spatial
    family should not have to know to opt in, which is exactly how the muon hole went unnoticed."""
    import inspect

    from merlin.targetgen import families
    from merlin.targetgen.rtl import circt_introspect, facts as F, spatial_introspect

    extractors = {families.family_profile(k).fact_extractor for k in families.known_kinds()}
    assert extractors == {"circt_static", "simt_config", "opu"}, (
        f"a new fact-extraction family appeared ({extractors}); guard its writer too")

    # simt_config (GPU-class, e.g. muon) — guarded inside the shared seam
    assert "write_facts_guarded" in inspect.getsource(F._dump_facts_for_kind)
    # circt_static (systolic/NPU, vector, scalar) — writes its own artifact
    assert "write_facts_guarded" in inspect.getsource(circt_introspect.dump_facts)
    # opu (spatial) — likewise
    assert "write_facts_guarded" in inspect.getsource(spatial_introspect.dump_fact_bundle)


def test_the_seam_restores_the_artifact_before_raising(tmp_path):
    """An extractor that writes the file ITSELF has already clobbered it by the time the seam looks.
    Reporting the downgrade while leaving the gutted file in place would destroy the very thing the
    guard exists to protect, so the previous artifact is put back first."""
    p = tmp_path / "facts.json"
    p.write_text(json.dumps(RICH))
    before = _read_facts_doc(p)
    p.write_text(json.dumps(POOR))        # stands in for the extractor's own write
    with pytest.raises(FactsDowngrade):
        _refuse_hollowed(p, before)
    assert json.loads(p.read_text())["facts"]["isa"]["instruction_classes"] == ["AUIPC", "BRANCH"]


def test_the_seam_is_silent_when_there_was_nothing_to_protect(tmp_path):
    """A cold cache is the NORMAL regeneration path — the guard must not turn a first extraction into
    an error."""
    p = tmp_path / "facts.json"
    p.write_text(json.dumps(POOR))
    _refuse_hollowed(p, None)             # no snapshot => nothing was lost
    assert json.loads(p.read_text())["facts"]["isa"]["instruction_classes"] == []
