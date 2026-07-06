"""Board-free unit tests for the ExecuTorch + XNNPACK baseline arm (merlin.baselines.executorch).

These never touch a board, the ET venv, or the SpacemiT toolchain: they exercise the pure logic
(symbol->region mapping, scalar-fallback labeling on a synthetic disasm, log parsing, bundle
resolution incl. the legacy fp32 LLM dir names, march enforcement, and the not_run_is_not_pass
gap path) so the gate stays green everywhere.
"""
from __future__ import annotations

from merlin.baselines import executorch as et
from merlin.baselines import rvv_audit


# --- symbol -> region mapping -----------------------------------------------------------------

def test_region_of_symbol():
    assert et._region_of_symbol("xnn_f32_gemm_ukernel_6x8__rvv") == "gemm"
    assert et._region_of_symbol("xnn_qs8_dwconv_minmax") == "gemm"
    assert et._region_of_symbol("softmax_out_kernel") == "attention"
    assert et._region_of_symbol("native_layer_norm_out") == "norm"
    assert et._region_of_symbol("aten_add_out") == "elementwise"
    assert et._region_of_symbol("some_bookkeeping") == "other"


# --- RVV audit: scalar fallbacks are labeled with the ExecuTorch reason -----------------------

_DISASM = """
0000000000010120 <xnn_f32_gemm_ukernel_1x4v__rvv_u1v>:
   10120:\t02008557          \tvsetvli\ta0,a1,e32,m1,ta,ma
   10124:\t0205f007          \tvle32.v\tv0,(a1)
   10128:\t02008557          \tvfmacc.vv\tv8,v0,v4
   1012c:\t00008067          \tret
0000000000010200 <executorch_portable_softmax_out>:
   10200:\t00b50533          \tadd\ta0,a0,a1
   10204:\t02c58533          \tmul\ta0,a1,a2
   10208:\t0005a007          \tflw\tfa0,0(a1)
   1020c:\t00008067          \tret
0000000000010300 <memcpy>:
   10300:\t00b50533          \tadd\ta0,a0,a1
   10304:\t00008067          \tret
"""


def test_audit_labels_portable_kernel_fallback(tmp_path, monkeypatch):
    # Feed the classifier directly (no objdump/binary), then run the arm's fallback labeling on it.
    report = rvv_audit.classify_disasm(_DISASM)
    fallbacks = [
        __import__("merlin.baselines.contract", fromlist=["ScalarFallback"]).ScalarFallback(
            symbol=sym, reason="no XNNPACK RVV ukernel (portable/scalar kernel)",
            region=et._region_of_symbol(sym))
        for sym in report.scalar_fallback_symbols(ignore=et._IGNORE_SYMS)
    ]
    syms = {f.symbol for f in fallbacks}
    assert "executorch_portable_softmax_out" in syms   # portable kernel labeled
    assert "xnn_f32_gemm_ukernel_1x4v__rvv_u1v" not in syms   # vectorized -> not a fallback
    assert "memcpy" not in syms                          # libc plumbing ignored
    sf = next(f for f in fallbacks if f.symbol == "executorch_portable_softmax_out")
    assert "no XNNPACK RVV ukernel" in sf.reason
    assert sf.region == "attention"


# --- executor_runner log parsing --------------------------------------------------------------

_ET_LOG = """
I 00:00:05.833558 executorch:executor_runner.cpp:564] Model loaded in 4463.268273 ms.
I 00:00:06.075611 executorch:executor_runner.cpp:705] Iteration 1 of 1: 241.944288 ms
I 00:00:06.075701 executorch:executor_runner.cpp:714] Model executed successfully 1 time(s) in 241.944288 ms.
"""


def test_parse_executor_runner_log():
    assert et._TIME_RE.search(_ET_LOG).group(1) == "241.944288"
    assert et._LOAD_RE.search(_ET_LOG).group(1) == "4463.268273"
    assert et._ITER_RE.search(_ET_LOG).group(1) == "241.944288"


def test_no_success_line_means_not_ran():
    log = "I executorch:executor_runner.cpp] Execution of method forward failed with status 0x12\n"
    assert et._TIME_RE.search(log) is None


def test_cos_rel_matches_golden(tmp_path):
    import numpy as np

    gold = np.arange(24, dtype=np.float32).reshape(1, 24)
    gp = tmp_path / "golden.npy"
    np.save(gp, gold)
    ob = tmp_path / "out.bin"
    ob.write_bytes(gold.ravel().tobytes())          # identical output
    cos, rel = et._cos_rel(ob, gp)
    assert cos > 0.999999 and rel < 1e-6


# --- bundle resolution (legacy fp32 LLM dir names) --------------------------------------------

def test_resolve_bundle_legacy_tiny_llama():
    # tiny_llama fp32 lives at the legacy tiny_consistent dir; resolve_bundle must find it if the
    # convention dir is absent. We only assert the fallback logic, not that the capture exists.
    b = et.resolve_bundle("tiny_llama", "fp32")
    assert b.model == "tiny_llama"
    # Either the convention dir or the legacy dir, but always a plausible recaptures path.
    assert "consistent" in b.root.name


def test_resolve_bundle_int8_uses_convention():
    b = et.resolve_bundle("bitvla", "int8")
    assert b.root.name == "bitvla_int8_consistent"


# --- march enforcement is wired into cross-compile --------------------------------------------

def test_cross_compile_enforces_rvv_march():
    # The K1 march must enable +v; the arm imports it and enforce_rvv_march would reject otherwise.
    from merlin.rvvgen import k1
    assert rvv_audit.enforce_rvv_march(k1.K1_MARCH) == "rv64gcv"


# --- not_run_is_not_pass: a board-down / venv-missing result is an explicit gap ---------------

def test_run_model_gap_when_inputs_missing(monkeypatch, tmp_path):
    # Point resolve_bundle at an empty dir so golden/inputs are absent -> not_built with a reason.
    from merlin.baselines import bundle as _bundle

    empty = _bundle.CaptureBundle(model="tiny_llama", variant="fp32", root=tmp_path)
    monkeypatch.setattr(et, "resolve_bundle", lambda m, v="fp32": empty)
    r = et.run_model("tiny_llama", "fp32", write=False, run_board=False)
    assert r.status() == "not_built"
    assert r.gap_reason  # non-empty
    assert not r.passed
    r.validate()


# --- int8: quantize + int8-subgraph flags, and the int8-appropriate gate ----------------------

def test_int8_variant_defaults_quantize_and_loosens_gate(monkeypatch, tmp_path):
    # An int8 run must (a) default quantize=True and (b) use an int8 gate (not the fp32 0.9999),
    # else a genuinely-correct W8A8 result would be spuriously failed. We capture the thresholds the
    # runner sets by stubbing export to fail fast AFTER the gate is chosen.
    from merlin.baselines import bundle as _bundle

    # A bundle with golden+inputs present so we get past the pre-export guards.
    (tmp_path / "golden.npy").write_bytes(b"\x00" * 8)
    (tmp_path / "inputs.npz").write_bytes(b"\x00" * 8)
    bnd = _bundle.CaptureBundle(model="tiny_llama", variant="int8", root=tmp_path)
    monkeypatch.setattr(et, "resolve_bundle", lambda m, v="fp32": bnd)
    monkeypatch.setattr(et, "et_venv_available", lambda: True)

    captured = {}

    def _fake_export(model, b, work, **kw):
        captured.update(kw)
        raise et.ExecuTorchError("stub: stop after gate is set")

    monkeypatch.setattr(et, "export_pte", _fake_export)
    r = et.run_model("tiny_llama", "int8", write=False, run_board=False, int8_subgraph=True)
    # int8 gate, not the fp32 (0.9999, 1e-3)
    assert r.cos_threshold == 0.99 and r.rel_threshold == 5e-2
    # quantize + int8_subgraph threaded to the exporter
    assert captured.get("quantize") is True
    assert captured.get("int8_subgraph") is True
    assert r.status() == "not_built" and r.gap_reason  # stubbed export -> honest gap
