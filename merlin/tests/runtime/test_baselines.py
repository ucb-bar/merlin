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
    # prefers the full-fidelity recapture when present, else the legacy _consistent bundle
    assert b.root.name in ("bitvla_int8_full", "bitvla_int8_consistent")
    assert b.mlir.name == "model.mlir" and b.golden.name == "golden.npy"
    assert b.tolerance == (0.999, 5e-3)


def test_k1_runnable_and_full_env():
    assert "tiny_llama" in bundle.K1_RUNNABLE and "openvla" not in bundle.K1_RUNNABLE
    assert bundle.K1_RUNNABLE.isdisjoint(bundle.K1_RAM_INFEASIBLE)
    assert bundle.full_env("bitvla") == {"BITVLA_LLM_LAYERS": "30"}
    assert bundle.full_env("tiny_llama") == {}


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

# --- TVM arm (board-free: target string, region mapping, availability gating, RVV enforce) ------

def test_tvm_target_enables_rvv():
    from merlin.baselines import tvm as tvm_arm
    # The plan-locked target string must pass the same march enforcement every arm uses.
    assert rvv_audit.enforce_rvv_march(tvm_arm.TVM_TARGET) == tvm_arm.TVM_TARGET
    assert "+v" in tvm_arm.TVM_TARGET and "riscv64" in tvm_arm.TVM_TARGET


def test_tvm_driver_uses_onnx_path():
    # The import path must be ONNX (torch-exported-program frontend lacks HF-transformer ops), with
    # the onnx.mapping + isnan/isinf legalize compat shims wired into the driver.
    from merlin.baselines import tvm as tvm_arm
    d = tvm_arm._DRIVER_TEMPLATE
    assert "torch.onnx.export" in d and "from_onnx" in d
    assert "onnx.mapping" in d and "TENSOR_TYPE_TO_NP_TYPE" in d
    assert "relax.isnan" in d and "register_legalize" in d
    # correctness is gated vs the torch reference for the exported instance, gold_cos reported too.
    assert "torch_ref" in d and "gold_cos" in d


def test_tvm_rpc_run_driver_wired():
    # The on-board execution path: a riscv64-runtime detector + an m2m-venv RPC-run driver that
    # deploys the runtime, connects host->board directly, runs the relax VM, times + gates cos.
    from merlin.baselines import tvm as tvm_arm
    assert hasattr(tvm_arm, "rv64_runtime_built") and hasattr(tvm_arm, "_rpc_run_driver")
    d = tvm_arm._RPC_RUN_TEMPLATE
    assert "rpc.connect" in d and "VirtualMachine" in d and "tvm_rpc server" in d
    assert "wall_ns" in d and "cos" in d and "rel" in d


def test_tvm_golden_prefers_w8a8(tmp_path):
    # int8-first: when a W8A8 golden is present it must be the correctness reference, else golden.npy.
    from merlin.baselines import bundle as _bundle
    from merlin.baselines import tvm as tvm_arm
    root = tmp_path / "m_int8_consistent"
    root.mkdir()
    (root / "golden.npy").write_bytes(b"\x00")
    b = _bundle.CaptureBundle(model="m", variant="int8", root=root)
    assert tvm_arm.golden_path(b).name == "golden.npy"      # only fp golden present
    (root / "golden_w8a8.npy").write_bytes(b"\x00")
    assert tvm_arm.golden_path(b).name == "golden_w8a8.npy"  # W8A8 golden preferred once present


def test_tvm_region_of_symbol():
    from merlin.baselines import tvm as tvm_arm
    assert tvm_arm._region_of_symbol("tvmgen_default_fused_matmul_add") == "gemm"
    assert tvm_arm._region_of_symbol("tvmgen_default_fused_softmax") == "attention"
    assert tvm_arm._region_of_symbol("tvmgen_default_fused_rms_norm") == "norm"
    assert tvm_arm._region_of_symbol("tvmgen_default_fused_add_multiply") == "elementwise"
    assert tvm_arm._region_of_symbol("tvmgen_default_transpose") == "other"


def test_tvm_audit_ignores_runtime_symbols():
    # A synthetic .so-style dump: one vectorized TVM kernel + a scalar TVM-runtime shim that must
    # NOT be reported as a model scalar-fallback (it's plumbing, filtered by _AUDIT_IGNORE).
    from merlin.baselines import tvm as tvm_arm
    disasm = (
        "0000000000010120 <tvmgen_default_fused_matmul>:\n"
        "   10120:\t02008557          \tvsetvli\ta0,a1,e32,m1,ta,ma\n"
        "   10124:\t0205f007          \tvfmacc.vv\tv8,v0,v4\n"
        "   10128:\t00008067          \tret\n"
        "0000000000010200 <__tvm_set_device>:\n"
        "   10200:\t00b50533          \tadd\ta0,a0,a1\n"
        "   10204:\t00008067          \tret\n"
    )
    rep = rvv_audit.classify_disasm(disasm)
    fb = rep.scalar_fallback_symbols(ignore=tvm_arm._AUDIT_IGNORE)
    assert fb == []  # the __tvm_ runtime shim is ignored, not labeled a model fallback


def test_tvm_not_built_when_lib_absent(monkeypatch, tmp_path):
    # With no built lib, run_model must produce an honest not_built gap (never a fabricated pass).
    from merlin.baselines import bundle as _bundle
    from merlin.baselines import tvm as tvm_arm
    # Fake a resolvable bundle (golden + loader present) so the gate we exercise is TVM availability,
    # not a missing capture — keeps the test independent of what's on disk.
    fake = tmp_path / "bundle"
    fake.mkdir()
    (fake / "golden.npy").write_bytes(b"\x00")
    b = _bundle.CaptureBundle(model="tiny_llama", variant="fp32", root=fake)
    monkeypatch.setattr(tvm_arm, "resolve_bundle", lambda m, v="fp32": b)
    monkeypatch.setattr(_bundle.CaptureBundle, "torch_loader",
                        property(lambda self: None))  # skip the loader-missing gate
    monkeypatch.setattr(tvm_arm, "tvm_built", lambda: False)
    monkeypatch.setattr(tvm_arm, "m2m_python", lambda: None)
    r = tvm_arm.run_model("tiny_llama", "fp32", work_root=tmp_path, write=False, run_board=False)
    assert r.status() == "not_built"
    assert r.gap_reason and "unavailable" in r.gap_reason.lower()
    r.validate()


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


def test_dedupe_latest_executed_beats_absence():
    """A timed-out re-verification (not_run, later ts) must NOT erase an earlier real on-board pass."""
    early_pass = BaselineResult(
        framework="exo", model="tiny_llama", variant="int8", built=True, ran=True,
        cos=1.0, rel=1e-6, cos_threshold=0.9999, rel_threshold=1e-3,
        e2e_cycles=11_680_000_000, timestamp="20260707T013739Z")
    later_notrun = BaselineResult(
        framework="exo", model="tiny_llama", variant="int8", built=True, ran=False,
        gap_reason="re-verification batch timed out", timestamp="20260707T014259Z")
    kept = aggregate.dedupe_latest([early_pass, later_notrun])
    assert len(kept) == 1
    assert kept[0].ran and kept[0].passed          # the real pass survives the later not_run
    # but a genuine later FAIL (ran=True) DOES supersede an earlier pass
    later_fail = BaselineResult(
        framework="exo", model="tiny_llama", variant="int8", built=True, ran=True,
        cos=0.5, rel=1.0, cos_threshold=0.9999, rel_threshold=1e-3, timestamp="20260707T020000Z")
    kept2 = aggregate.dedupe_latest([early_pass, later_fail])
    assert len(kept2) == 1 and kept2[0].ran and not kept2[0].passed
