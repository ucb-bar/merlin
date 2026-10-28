"""Port facts in a selected (verified-mode) fact bundle: derived from, and bound to, its own FIRRTL.

A selected profile performs no external reads, so ``explicit_completion`` can only be established
when the bundle itself carries the port list. These tests pin that the list is attached only from the
FIRRTL the bundle declares (digest-checked), that the profile accepts it only while that binding
holds, and that every unbound or missing case stays UNKNOWN rather than reading as absent.
"""
from __future__ import annotations

import hashlib

import pytest

from merlin.perf import profile as P
from merlin.targetgen.rtl import ports

_FIR = """FIRRTL version 3.3.0
circuit Top :
  module Mover :
    input clock : Clock
    output io : { flip cmd : { flip ready : UInt<1>, valid : UInt<1>, bits : UInt<8>}, completed : { flip ready : UInt<1>, valid : UInt<1>, bits : UInt<6>}}

    skip
"""


def _bundle(tmp_path, text=_FIR, digest=None):
    fir = tmp_path / "top.fir"
    fir.write_text(text)
    sha = digest or hashlib.sha256(fir.read_bytes()).hexdigest()
    return {"inputs": {"target": "phantom_target_under_test", "firrtl_inputs": [{"path": str(fir), "sha256": sha}]},
            "facts": {"interfaces": [{"name": "cmd", "evidence": "x"}]}}


def _trait(doc):
    src = P.load_sources("phantom_target_under_test", facts=doc, residual={}, contract={})
    assert src.external_reads is False
    return P._t_explicit_completion(src)


def test_attached_and_bound_port_facts_establish_completion_from_facts(tmp_path):
    doc = ports.attach_port_facts(_bundle(tmp_path))
    assert doc["port_facts"]["status"] == "derived"
    assert doc["port_facts"]["fields"]["completed"]["decoupled"] == ["Mover"]
    trait, tier = _trait(doc)
    assert trait.satisfied is True
    assert tier == P.TIER_FACTS


def test_attach_refuses_a_fir_whose_bytes_differ_from_the_declared_digest(tmp_path):
    with pytest.raises(ValueError, match="does not match"):
        ports.attach_port_facts(_bundle(tmp_path, digest="0" * 64))


def test_attach_refuses_without_exactly_one_declared_fir(tmp_path):
    doc = _bundle(tmp_path)
    doc["inputs"]["firrtl_inputs"] = []
    with pytest.raises(ValueError, match="exactly one"):
        ports.attach_port_facts(doc)


def test_port_facts_bound_to_another_elaboration_stay_unknown(tmp_path):
    doc = ports.attach_port_facts(_bundle(tmp_path))
    doc["inputs"]["firrtl_inputs"][0]["sha256"] = "f" * 64
    trait, _ = _trait(doc)
    assert trait.satisfied is None
    assert "different elaboration" in trait.evidence


def test_a_bundle_without_port_facts_stays_unknown_not_absent(tmp_path):
    trait, tier = _trait(_bundle(tmp_path))
    assert trait.satisfied is None
    assert tier == P.TIER_NONE


def test_a_completion_field_without_a_handshake_is_measured_false(tmp_path):
    text = _FIR.replace("completed : { flip ready : UInt<1>, valid : UInt<1>, bits : UInt<6>}",
                        "completed : { valid : UInt<1>, bits : UInt<6>}")
    trait, _ = _trait(ports.attach_port_facts(_bundle(tmp_path, text=text)))
    assert trait.satisfied is False
