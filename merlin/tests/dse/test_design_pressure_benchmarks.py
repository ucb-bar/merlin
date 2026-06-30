"""End-to-end acceptance: workload_region -> pressure vector -> recommended contracts.

The five ``semantic_memory`` benchmarks carry golden ``expected_candidate_features``. This
test drives the full M1 spine (compute_rpv -> synthesize.recommended_features) and asserts the
synthesizer reproduces each golden — the headline correctness gate for the pressure pass.
"""
import glob
import os

import pytest

from merlin.common.yaml import load_yaml
from merlin.design_pressure import synthesize as S
from merlin.design_pressure.pressure_vector import compute_rpv

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
BENCH = os.path.join(ROOT, "merlin", "benchmarks", "semantic_memory")

# capacity_stress declares a resident_store sweep; the golden holds at the largest (fits).
RESIDENT_STORE = {"capacity_stress_reuse": 262144}


def _benchmarks():
    return sorted(glob.glob(os.path.join(BENCH, "*.yaml")))


def test_benchmarks_present():
    names = {os.path.splitext(os.path.basename(f))[0] for f in _benchmarks()}
    assert {"repeated_rhs_matmul", "matmul_bias_requant_relu", "no_reuse_matmul",
            "capacity_stress_reuse", "vla_action_chunk_decode"} <= names


@pytest.mark.parametrize("path", _benchmarks(), ids=lambda p: os.path.basename(p))
def test_recommended_features_match_golden(path):
    region = load_yaml(path)
    expected = region.get("expected_candidate_features", [])
    rpv = compute_rpv(region)
    pol = S.load_policies()
    rsb = RESIDENT_STORE.get(region["name"])
    got = S.recommended_features(rpv, pol, resident_store_bytes=rsb)
    assert got == expected


def test_capacity_phase_transition():
    """resident recommendation flips off when distinct weights exceed resident storage."""
    region = load_yaml(os.path.join(BENCH, "capacity_stress_reuse.yaml"))
    rpv = compute_rpv(region)
    pol = S.load_policies()
    # 16 distinct [128,64] i8 weights = 16 * 8192 = 131072 B.
    assert S.recommended_features(rpv, pol, resident_store_bytes=65536) == []
    assert S.recommended_features(rpv, pol, resident_store_bytes=131072) == ["resident_packed_tensor"]


def test_structural_legality_vs_endorsement_mechanism():
    """The split: a contraction-depth (K) condition gates *endorsement*, not *structural* legality.

    The shipped mined ``packed_rhs_policy`` currently has no K condition, so endorsement equals
    structural legality. This test proves the mechanism is correct *if* a K>=256 condition is
    present (e.g. re-mined): below K=256 residency stays structurally legal but unendorsed;
    at/above it, endorsed too. Uses an explicit K-bearing policy so it is independent of the
    shipped corpus.
    """
    from merlin.design_pressure.workloads.vla_action_chunk_decode import build_region
    k_policy = [{
        "policy": "packed_rhs_policy",
        "when": {"rhs_reuse_count": ">= 2", "rhs_mutable": "false", "K": ">= 256"},
        "actions": ["consider_resident_packed_tensor"],
    }]
    for k, endorsed in [(128, False), (256, True)]:
        rpv = compute_rpv(build_region(H=8, K=k))
        i2 = next(c for c in S.legal_contracts(rpv, k_policy) if c["id"] == "I2")
        assert i2["legal"] is True                # structural legality ignores K
        assert i2["policy_endorsed"] is endorsed   # endorsement honours K>=256


def test_shipped_policy_endorses_when_structurally_legal():
    """With the shipped mined policy (no K condition), endorsement matches structural legality."""
    from merlin.design_pressure.workloads.vla_action_chunk_decode import build_region
    pol = S.load_policies()
    rpv = compute_rpv(build_region(H=8, K=128))
    i2 = next(c for c in S.legal_contracts(rpv, pol) if c["id"] == "I2")
    assert i2["legal"] is True and i2["policy_endorsed"] is True
