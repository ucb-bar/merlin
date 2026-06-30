"""Emitted design-pressure / interface-candidate dicts validate against their schemas."""
import os

from merlin.common import schemas
from merlin.common.yaml import load_yaml
from merlin.design_pressure import synthesize as S
from merlin.design_pressure.emit.candidate_contracts import emit_candidate_contracts
from merlin.design_pressure.emit.design_pressure import emit_design_pressure
from merlin.design_pressure.emit.interface_candidate import emit_interface_candidate
from merlin.design_pressure.pressure_vector import compute_rpv
from merlin.design_pressure.workloads.vla_action_chunk_decode import build_region

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def test_emit_design_pressure_validates():
    region = build_region(H=8, K=256, epilogue=True)
    rpv = compute_rpv(region)
    pol = S.load_policies()
    feats = S.recommended_features(rpv, pol)
    report = emit_design_pressure(region["name"], rpv["cutpoints"], rpv, feats)
    assert schemas.validate(report, "design_pressure") == []
    assert report["candidate_contracts"] == feats
    assert "state" in report["pressure_classes"]


def test_emit_interface_candidate_validates():
    cand = emit_interface_candidate(
        "resident_packed_tensor", ["resident_pack", "resident_matmul", "evict"],
        ["resident_packed_tensor"], "vla_action_chunk_decode", ["packed_rhs_policy"])
    assert schemas.validate(cand, "interface_candidate") == []


def test_emit_candidate_contracts_shape():
    region = build_region(H=8, K=256, epilogue=True)
    rpv = compute_rpv(region)
    pol = S.load_policies()
    payload = emit_candidate_contracts(region["name"], S.legal_contracts(rpv, pol))
    assert payload["workload"] == region["name"]
    assert "resident_packed_tensor" in payload["legal_contracts"]
    assert {c["id"] for c in payload["contracts"]} == {"I0", "I1", "I2", "I3"}
