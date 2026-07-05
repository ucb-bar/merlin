"""Board-free unit tests for the external-baseline K1-RVV harness (merlin.baselines).

These never touch a board: they exercise the RVV-coverage classifier, march enforcement, the
not_run_is_not_pass contract, marker profiling, bundle resolution, the compare-spec external kind,
and matrix rendering — so the gate stays green without a live K1.
"""
from __future__ import annotations

import pytest

from merlin.baselines import BaselineResult, RegionProfile, ScalarFallback
from merlin.baselines import aggregate, bundle, profile, rvv_audit


# --- RVV coverage classifier ------------------------------------------------------------------

# A tiny synthetic objdump dump: one vectorized kernel, one scalar fallback kernel.
_DISASM = """
0000000000010120 <gemm_rvv>:
   10120:\t02008557          \tvsetvli\ta0,a1,e32,m1,ta,ma
   10124:\t0205f007          \tvle32.v\tv0,(a1)
   10128:\t02008557          \tvfmacc.vv\tv8,v0,v4
   1012c:\t00008067          \tret
0000000000010200 <gemm_scalar>:
   10200:\t00b50533          \tadd\ta0,a0,a1
   10204:\t02c58533          \tmul\ta0,a1,a2
   10208:\t0005a007          \tflw\tfa0,0(a1)
   1020c:\t00008067          \tret
"""


def test_classify_disasm_counts_vector_and_scalar():
    rep = rvv_audit.classify_disasm(_DISASM)
    assert rep.by_symbol["gemm_rvv"].vector == 3          # vsetvli, vle32.v, vfmacc.vv
    assert rep.by_symbol["gemm_rvv"].scalar_compute == 0
    assert rep.by_symbol["gemm_scalar"].vector == 0
    assert rep.by_symbol["gemm_scalar"].scalar_compute == 3  # add, mul, flw
    assert rep.coverage_overall == pytest.approx(3 / 6)


def test_scalar_fallback_detection():
    rep = rvv_audit.classify_disasm(_DISASM)
    assert rep.by_symbol["gemm_scalar"].is_scalar_fallback is True
    assert rep.by_symbol["gemm_rvv"].is_scalar_fallback is False
    assert rep.scalar_fallback_symbols() == ["gemm_scalar"]


def test_enforce_rvv_march():
    assert rvv_audit.enforce_rvv_march("rv64gcv") == "rv64gcv"
    assert rvv_audit.enforce_rvv_march("-mattr=+v") == "-mattr=+v"   # +v style accepted
    with pytest.raises(ValueError):
        rvv_audit.enforce_rvv_march("rv64gc")
    with pytest.raises(ValueError):
        rvv_audit.enforce_rvv_march("rv64imafd")


# --- contract: not_run_is_not_pass ------------------------------------------------------------

def test_not_built_is_not_pass():
    r = BaselineResult(framework="tvm", model="tiny_llama", built=False, gap_reason="import failed")
    assert r.passed is False
    assert r.status() == "not_built"
    r.validate()  # gap_reason present -> ok


def test_ran_but_below_tolerance_is_fail():
    r = BaselineResult(framework="buddy", model="rdt2", built=True, ran=True,
                       cos=0.5, rel=0.9, cos_threshold=0.9999, rel_threshold=1e-3)
    assert r.passed is False
    assert r.status() == "fail"


def test_pass_requires_build_run_and_tolerance():
    r = BaselineResult(framework="ggml", model="tiny_llama", built=True, ran=True,
                       cos=0.99999, rel=1e-4, cos_threshold=0.9999, rel_threshold=1e-3)
    assert r.passed is True
    assert r.status() == "pass"


def test_gap_without_reason_raises():
    r = BaselineResult(framework="exo", model="openvla", built=False)  # no gap_reason
    with pytest.raises(ValueError):
        r.validate()


def test_unknown_framework_rejected():
    with pytest.raises(ValueError):
        BaselineResult(framework="tensorrt", model="rdt2")


def test_result_roundtrip(tmp_path):
    r = BaselineResult(
        framework="buddy", model="tiny_llama", variant="fp32", built=True, ran=True,
        cos=0.99999, rel=1e-4, cos_threshold=0.9999, rel_threshold=1e-3,
        e2e_rdtime_ticks=1000, e2e_cycles=66666,
        regions=[RegionProfile(name="gemm", rdtime_ticks=800, cycles=53333, rvv_coverage=0.9)],
        rvv_coverage_overall=0.85,
        scalar_fallbacks=[ScalarFallback(symbol="softmax", reason="no rvv microkernel", region="attention")],
    )
    path = r.write(tmp_path)
    back = BaselineResult.load(path)
    assert back.passed is True
    assert back.regions[0].name == "gemm"
    assert back.scalar_fallbacks[0].symbol == "softmax"
    assert back.rvv_coverage_overall == pytest.approx(0.85)


# --- profiling markers ------------------------------------------------------------------------

def test_parse_profile_markers():
    stdout = (
        "hello\n"
        "MERLIN_E2E ticks=2400 wall_ns=100000\n"
        "MERLIN_REGION name=gemm ticks=1800 calls=7\n"
        "MERLIN_REGION name=attention ticks=400\n"
        "DONE\n"
    )
    e2e, regions = profile.parse_profile(stdout)
    assert e2e.rdtime_ticks == 2400
    assert e2e.wall_ns == 100000
    # 2400 ticks * (1.6e9/24e6) = 160000 cycles
    assert e2e.cycles == profile.ticks_to_cycles(2400) == 160000
    assert [r.name for r in regions] == ["gemm", "attention"]
    assert regions[0].calls == 7


# --- bundle resolution ------------------------------------------------------------------------

def test_bundle_resolve_paths():
    b = bundle.resolve("bitvla", "int8")
    assert b.model == "bitvla" and b.variant == "int8"
    assert b.root.name == "bitvla_int8_consistent"
    assert b.mlir.name == "model.mlir" and b.golden.name == "golden.npy"
    assert b.tolerance == (0.999, 5e-3)


def test_bundle_default_tolerance():
    assert bundle.tolerance("some_unlisted_model") == (0.9999, 1e-3)


def test_bundle_bad_variant():
    with pytest.raises(ValueError):
        bundle.resolve("bitvla", "bf16")


# --- compare-spec external kind ---------------------------------------------------------------

def test_spec_external_framework_kind():
    from merlin.compare.spec import Config
    for fw in ("tvm", "executorch", "buddy", "exo", "ggml"):
        assert Config.parse(fw).kind == "external"
    assert Config.parse("baseline").kind == "baseline"
    assert Config.parse("xnnpack").kind == "kernel_backend"
    with pytest.raises(ValueError):
        Config.parse("tensorrt")


# --- matrix rendering -------------------------------------------------------------------------

def test_render_matrix_shows_gaps_and_coverage():
    results = [
        BaselineResult(framework="buddy", model="tiny_llama", built=True, ran=True,
                       cos=0.99999, rel=1e-4, cos_threshold=0.9999, rel_threshold=1e-3,
                       e2e_cycles=2_000_000, rvv_coverage_overall=0.9),
        BaselineResult(framework="ggml", model="tiny_llama", built=False,
                       gap_reason="no gguf converter for this arch"),
    ]
    md = aggregate.render_markdown(results)
    assert "tiny_llama/fp32" in md
    assert "pass" in md and "90%RVV" in md
    assert "not_built" in md            # the gap is shown, not blank
    csv = aggregate.render_csv(results)
    assert "no gguf converter" in csv
