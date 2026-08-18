"""The Muon runner is now a config over the shared capsule_runner — this pins the shim's config +
exception translation so the collapse preserves the SIMT grading behavior.
"""
from __future__ import annotations

import pytest

from merlin.runtime.backends.base import get_backend
MR = get_backend("muon").muon_capsule_runner    # evicted SIMT backend/CLI, resolved via plugin discovery
from merlin.targetgen.capsule_runner import OracleUnavailable
MuonUnavailable = get_backend("muon").MuonUnavailable


def test_muon_config_carries_the_simt_constants():
    c = MR._MUON_CONFIG
    assert c.target == "muon" and c.suite == "muon-perf-bench" and c.dtype == "f32"
    assert c.fourth_output_name == "kernel.cpp"                 # SIMT C++ kernel, not RoCC LLVM
    assert c.tier_sim == {"L2": "cyclotron", "L3": "vcs"}
    assert c.rtl_tiers == frozenset({"L3"}) and c.oracle_tiers == ("L2", "L3")
    assert c.trace_gate is None                                 # no RoCC trace gate on a SIMT target
    assert c.perf_fields == ("flops", "gflops", "pct_fp_peak")
    assert c.force_match_policy == {"compare": "float", "atol": 1e-3}   # fp-tolerant oracle equality


def test_wrap_translates_muon_unavailable_to_oracle_unavailable():
    def raises(cb, art, wd, to):
        raise MuonUnavailable("cyclotron not built")
    wrapped = MR._wrap_adapters({"L2": raises})["L2"]
    with pytest.raises(OracleUnavailable):
        wrapped({}, "k.cpp", "/tmp", 5)


def test_wrap_passes_through_a_normal_result():
    def ok(cb, art, wd, to):
        return {"outputs": {}, "cycles": 7, "gflops": 1.0, "pct_fp_peak": 5.0}
    wrapped = MR._wrap_adapters({"L2": ok})["L2"]
    assert wrapped({}, "k.cpp", "/tmp", 5)["cycles"] == 7


def test_perf_extractor_passes_through_adapter_perf():
    perf = MR._muon_perf({}, {"gflops": 3.5, "pct_fp_peak": 40.0, "cycles": 100})
    assert perf == {"gflops": 3.5, "pct_fp_peak": 40.0}
