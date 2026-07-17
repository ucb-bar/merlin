"""Board-free unit tests for the ExecuTorch + XNNPACK baseline arm (merlin.baselines.executorch).

These never touch a board, the ET venv, or the SpacemiT toolchain: they exercise the pure logic
(symbol->region mapping, scalar-fallback labeling on a synthetic disasm, log parsing, bundle
resolution incl. the legacy fp32 LLM dir names, march enforcement, and the not_run_is_not_pass
gap path) so the gate stays green everywhere.
"""
from __future__ import annotations

import importlib.util

from merlin.common.paths import merlin_dir

from merlin.baselines import executorch as et
from merlin.baselines import rvv_audit


def _load_et_export():
    """Load the ET-venv AOT helper by path (it is dependency-light; the pure fqn-extraction logic
    imports fine under merlin's venv even though the export path itself runs under the ET venv)."""
    p = merlin_dir() / "python" / "merlin" / "baselines" / "_et_export.py"
    spec = importlib.util.spec_from_file_location("_et_export_under_test", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeNode:
    def __init__(self, op, name, target, meta):
        self.op, self.name, self.target, self.meta = op, name, target, meta


class _FakeGraphModule:
    def __init__(self, nodes):
        self.graph = type("g", (), {"nodes": nodes})()


# --- ET-side provenance: node -> originating model-layer fqn (the cross-compiler join key) -----

def test_extract_fqn_map_picks_deepest_module_and_skips_non_compute():
    m = _load_et_export()
    nodes = [
        _FakeNode("call_function", "lin_q", "aten.linear.default",
                  {"nn_module_stack": {"a": ("model.layers.0", "M"),
                                       "b": ("model.layers.0.self_attn", "Attn")}}),
        _FakeNode("call_function", "act", "aten.silu.default",
                  {"nn_module_stack": {"a": ("model.layers.0", "M"),
                                       "b": ("model.layers.0.mlp", "MLP")}}),
        _FakeNode("placeholder", "x", "x", {}),                 # not a compute op -> skipped
        _FakeNode("call_function", "untagged", "aten.view.default", {}),  # no nn_module_stack -> skipped
    ]
    fmap = m.extract_fqn_map(_FakeGraphModule(nodes))
    assert fmap["lin_q"]["fqn"] == "model.layers.0.self_attn"   # deepest module, not the wrapper
    assert fmap["act"]["fqn"] == "model.layers.0.mlp"
    assert "x" not in fmap and "untagged" not in fmap
    # the fqns are the SAME key space role_from_fqn / recognize_regions consume.
    from merlin.dse_guidance.attribution import role_from_fqn
    assert role_from_fqn(fmap["lin_q"]["fqn"]) == "repeated_head"


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


def test_mmap_model_adds_runner_flag(tmp_path, monkeypatch):
    """--mmap_model=true is passed to executor_runner iff mmap_model=True (whole-model int8 path).

    The const-folded whole-model int8 .pte carries the dequantized fp32 weights as program
    constants (multi-GB), so the board must mmap it (MmapDataLoader, NoMlock) to demand-page the
    weight pages under its RAM ceiling instead of reading them fully resident.
    """
    captured_argv = {}

    class _Proc:
        stdout = "Model executed successfully 1 time(s) in 5.0 ms\n"
        stderr = ""

    import contextlib

    monkeypatch.setattr(et.k1_exec, "board_lock", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(et, "_board_free_bytes", lambda: 10 * 1024**3)
    monkeypatch.setattr(et.k1_exec, "push", lambda p, r=None, **k: r or f"/remote/{p.name}")

    def _fake_run(argv, **kw):
        if any("--model_path" in str(a) for a in argv):
            captured_argv["argv"] = argv
        return _Proc()

    monkeypatch.setattr(et.k1_exec, "run", _fake_run)
    monkeypatch.setattr(et, "_scp_from_board", lambda *a, **k: None)

    pte = tmp_path / "model.pte"
    pte.write_bytes(b"x" * 16)
    exp = et.ExportResult(pte=pte, ptd_files=[], input_files=[], golden=tmp_path / "g.npy")
    res = et.BaselineResult(framework="executorch", model="tiny_llama", variant="int8")

    et._run_on_board(res, pte, exp, mmap_model=True)
    assert any("--mmap_model=true" == str(a) for a in captured_argv["argv"])

    captured_argv.clear()
    et._run_on_board(res, pte, exp, mmap_model=False)
    assert not any("mmap_model" in str(a) for a in captured_argv["argv"])


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

def test_resolve_bundle_tiny_llama():
    # resolve_bundle finds a plausible tiny_llama fp32 recaptures dir: the full-fidelity
    # <model>_fp32_full, the convention _consistent, or the legacy tiny_consistent.
    b = et.resolve_bundle("tiny_llama", "fp32")
    assert b.model == "tiny_llama"
    assert b.root.name in ("tiny_llama_fp32_full", "tiny_llama_fp32_consistent", "tiny_consistent")


def test_resolve_bundle_int8_prefers_full_fidelity():
    # bundle.resolve prefers the full-fidelity <model>_int8_full recapture (real/native arch) over
    # the older truncated _consistent bundle when present.
    b = et.resolve_bundle("bitvla", "int8")
    assert b.root.name in ("bitvla_int8_full", "bitvla_int8_consistent")


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


def test_int8_variant_defaults_to_whole_model(monkeypatch, tmp_path):
    # The int8 variant defaults to the OFFICIAL llama whole-model recipe (int8_whole_model=True),
    # NOT the subgraph — the path that unblocks full-model int8 on HF Llama.
    from merlin.baselines import bundle as _bundle

    (tmp_path / "golden.npy").write_bytes(b"\x00" * 8)
    (tmp_path / "inputs.npz").write_bytes(b"\x00" * 8)
    bnd = _bundle.CaptureBundle(model="tiny_llama", variant="int8", root=tmp_path)
    monkeypatch.setattr(et, "resolve_bundle", lambda m, v="fp32": bnd)
    monkeypatch.setattr(et, "et_venv_available", lambda: True)
    captured = {}

    def _fake_export(model, b, work, **kw):
        captured.update(kw)
        raise et.ExecuTorchError("stub")

    monkeypatch.setattr(et, "export_pte", _fake_export)
    r = et.run_model("tiny_llama", "int8", write=False, run_board=False)  # no int8_subgraph
    assert captured.get("int8_whole_model") is True
    assert captured.get("int8_subgraph") is False
    assert r.cos_threshold == 0.99  # int8 gate


def test_ram_infeasible_models_are_built_not_run(monkeypatch, tmp_path):
    # openvla/molmoact/pi05 are RAM-infeasible whole-model on the K1: they must be BUILT (export +
    # audit) but recorded not_run with a specific RAM gap, never a false fit. We stub past export +
    # cross-compile + audit so the RAM short-circuit is what sets the gap.
    from merlin.baselines import bundle as _bundle

    (tmp_path / "golden.npy").write_bytes(b"\x00" * 8)
    (tmp_path / "inputs.npz").write_bytes(b"\x00" * 8)
    bnd = _bundle.CaptureBundle(model="openvla", variant="int8", root=tmp_path)
    monkeypatch.setattr(et, "resolve_bundle", lambda m, v="fp32": bnd)
    monkeypatch.setattr(et, "et_venv_available", lambda: True)

    class _Exp:
        summary = {}
        delegated_nodes = None
        total_call_nodes = None
        pte = tmp_path / "model.pte"
        ptd_files = []
        input_files = []
        golden = tmp_path / "golden.npy"

    monkeypatch.setattr(et, "export_pte", lambda *a, **k: _Exp())
    fake_runner = tmp_path / "executor_runner"
    fake_runner.write_bytes(b"\x7fELF")
    monkeypatch.setattr(et, "audit_binary", lambda r: (0.1, [], {}))
    monkeypatch.setattr(et.k1_exec, "board_vlenb", lambda: 32)
    r = et.run_model("openvla", "int8", write=False, run_board=True,
                     runner_override=fake_runner)
    assert r.built is True          # export + audit happened
    assert r.status() == "not_run"  # but not run on-board
    assert "RAM-infeasible" in r.gap_reason
    assert not r.passed
    r.validate()
