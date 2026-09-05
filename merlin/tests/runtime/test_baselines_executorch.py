"""Board-free unit tests for the ExecuTorch + XNNPACK baseline arm (merlin.baselines.executorch).

These never touch a board, the ET venv, or the SpacemiT toolchain: they exercise the pure logic
(symbol->region mapping, scalar-fallback labeling on a synthetic disasm, log parsing, bundle
resolution incl. the legacy fp32 LLM dir names, march enforcement, and the not_run_is_not_pass
gap path) so the gate stays green everywhere.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

from merlin.common.paths import merlin_dir

from merlin.baselines import executorch as et
from merlin.baselines import executorch_session as ets
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


def _load_et_inspect():
    p = merlin_dir() / "python" / "merlin" / "baselines" / "_et_inspect.py"
    spec = importlib.util.spec_from_file_location("_et_inspect_under_test", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeEvent:
    def __init__(self, name, module_hierarchy, perf_ms=None, delegated=False):
        self.name = name
        self.module_hierarchy = module_hierarchy
        self.is_delegated_op = delegated
        self.perf_data = type("pd", (), {"p50": perf_ms})() if perf_ms is not None else None


def test_et_inspect_recovers_layer_fqn_and_buckets_delegated():
    """The per-region etdump post-processor keys a portable op to its L__self__ module path (the same
    fqn Merlin uses), and buckets a delegated (opaque XNNPACK) op as 'other'."""
    m = _load_et_inspect()
    # a portable addmm: the module path rides in module_hierarchy behind the L__self__ marker,
    # alongside class-name noise that must be filtered out.
    portable = _FakeEvent(
        "native_call_addmm.out",
        {"torch.nn.modules.linear.Linear": {}, "aten_addmm_default_1_.L__self__layers.0.mlp": {}},
        perf_ms=0.08)
    assert m._deepest_fqn(portable) == "layers.0.mlp"
    assert m._clean_module_path("aten_addmm_default_.L__self__layers.0.self_attn") == "layers.0.self_attn"
    # a delegated op has no L__self__ path -> None -> 'other' bucket (the honest asymmetry).
    delegated = _FakeEvent("DELEGATE_CALL", {"XNNPACKBackend": {}}, perf_ms=1.0, delegated=True)
    assert m._deepest_fqn(delegated) is None


def test_et_region_json_to_profiles_feeds_the_compare(tmp_path):
    """The full C7 data path: _et_inspect's per-region etdump JSON -> RegionProfiles -> align_regions
    -> the region×framework matrix. Uses the exact shape recovered from the real K1 run."""
    import json as _json

    from merlin.baselines.contract import RegionProfile
    from merlin.compare.attribution import align_regions
    from merlin.compare.report import region_alignment_md

    et_json = tmp_path / "et_regions.json"
    et_json.write_text(_json.dumps([
        {"fqn": "layers.0.mlp", "wall_ns": 157707, "n_events": 8, "delegated": False},
        {"fqn": "layers.0.self_attn", "wall_ns": 82043, "n_events": 4, "delegated": False},
        {"fqn": "other", "wall_ns": 679048, "n_events": 3, "delegated": False},
    ]))
    et_profiles = et.region_profiles_from_et_json(et_json)
    by = {p.fqn: p for p in et_profiles}
    assert by["layers.0.self_attn"].role == "repeated_head"       # role derived on the merlin side
    assert by["layers.0.self_attn"].wall_ns == 82043
    assert any(p.name == "other" and p.fqn == "" for p in et_profiles)

    # align an ExecuTorch region against a Merlin region on the shared fqn -> the apples-to-apples row.
    merlin = [RegionProfile(name="attention", fqn="layers.0.self_attn", role="repeated_head",
                            wall_ns=60000, cos=1.0)]
    md = region_alignment_md(align_regions(merlin, et_profiles))
    assert "layers.0.self_attn" in md and "0.73×" in md          # 60000/82043 ≈ 0.73 (Merlin faster here)


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


def _exec_reading(console):
    return et._read_console_ms(console, et._EXEC_MARKER, joiner=et._EXEC_JOINER)


def _load_reading(console):
    return et._read_console_ms(console, et._LOAD_MARKER)


def test_parse_executor_runner_log():
    """The real ET_LOG banner + format strings, read structurally (no regex).

    These are the numbers the retired patterns produced on this exact log, so the conversion is
    value-for-value: `Model executed successfully .* in ([\\d.]+) ms` -> 241.944288 and
    `Model loaded in ([\\d.]+) ms` -> 4463.268273.
    """
    executed = _exec_reading(_ET_LOG)
    assert executed.state == et._PARSED
    assert executed.ms == 241.944288
    assert executed.ns == 241944288          # ms -> ns, the only conversion applied
    loaded = _load_reading(_ET_LOG)
    assert loaded.state == et._PARSED
    assert loaded.ms == 4463.268273


def test_no_success_line_means_not_ran():
    log = "I executorch:executor_runner.cpp] Execution of method forward failed with status 0x12\n"
    r = _exec_reading(log)
    assert r.state == et._ABSENT and r.ms is None and r.ns is None


def test_present_but_unreadable_time_is_unparseable_not_absent():
    """A marker line that IS there but whose ms field cannot be read must NOT look like "never ran".

    This is the state the retired regex could not express: it returned no match either way, so a
    console-format drift was indistinguishable from a model that never executed.
    """
    drifted = ("I 00:00:06.075701 executorch:executor_runner.cpp:714] "
               "Model executed successfully 1 time(s) in <n/a> ms.\n")
    r = _exec_reading(drifted)
    assert r.state == et._UNPARSEABLE
    assert r.ms is None and r.ns is None     # UNKNOWN, never 0
    assert "Model executed successfully" in r.detail


def test_unreadable_number_is_refused_never_coerced():
    """Tokens the old `[\\d.]+` class matched but `float()` could not build (it raised) are refused."""
    for token in ("1.2.3", ".", "inf", "nan", "-5", "1e3"):
        line = f"I x] Model loaded in {token} ms.\n"
        r = _load_reading(line)
        assert r.ms is None, token
        assert r.state == et._UNPARSEABLE, token


def test_greedy_last_in_wins_like_the_old_pattern():
    """The old `.*` was greedy, so the LAST ` in <n> ms` on the line supplied the number."""
    line = "I x] Model executed successfully 1 time(s) after 9 ms in 241.5 ms.\n"
    assert _exec_reading(line).ms == 241.5


def test_board_run_reports_an_unreadable_console_instead_of_zero(monkeypatch, tmp_path):
    """A present-but-unreadable timing must reach the result as a stated gap, not as ran/0 ns."""
    br = et.BoardRun()
    br.parse_warnings = ["executor_runner reported success but its time field is unreadable: 'x'"]
    res = et.BaselineResult(framework=et.FRAMEWORK, model="tiny_llama", variant="fp32")
    monkeypatch.setattr(et, "_run_on_board", lambda *a, **k: br)
    et._do_board(res, tmp_path / "runner", None)
    assert res.ran is False
    assert res.e2e_wall_ns is None           # UNKNOWN, never 0
    assert "unreadable" in res.gap_reason and "console-parse" in res.notes


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
    # The board ISA also carries the half-precision extensions (zfh/zvfh — see /proc/cpuinfo), so the
    # default march includes them; enforce_rvv_march passes it through unchanged (it only gates on +v).
    from merlin.mining import k1
    assert rvv_audit.enforce_rvv_march(k1.K1_MARCH) == "rv64gcv_zfh_zvfh"
    assert "v" in k1.K1_MARCH


def test_external_baselines_share_merlins_physical_board_lock(monkeypatch, tmp_path):
    """Framework and compiler arms must never overlap on the single K1."""
    from merlin.baselines import k1_exec

    lock_path = tmp_path / "shared-k1.lock"
    opened = []

    class _Handle:
        def close(self):
            pass

    monkeypatch.setattr(k1_exec.k1, "_board_lock_path", lambda: lock_path)
    monkeypatch.setattr("builtins.open", lambda path, mode: opened.append((path, mode)) or _Handle())
    monkeypatch.setattr(k1_exec.fcntl, "flock", lambda *args: None)
    with k1_exec.board_lock():
        pass
    assert opened == [(lock_path, "w")]


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


# --- semantic continuous sessions (never --num_executions on one unchanged input) -------------

def _session_manifest(tmp_path, **overrides):
    import json

    tmp_path.mkdir(parents=True, exist_ok=True)
    for name, data in {
        "step.pte": b"pte", "hidden.bin": b"\0" * 8, "frame0.bin": b"\0" * 12,
        "frames.bin": b"\1" * 24, "correctness.npz": b"correct", "quality.npz": b"quality",
    }.items():
        (tmp_path / name).write_bytes(data)
    doc = {
        "schema": ets.SCHEMA,
        "protocol_version": 1,
        "kind": "recurrent_frames",
        "paper_ready": True,
        "precision": "fp32",
        "reset": "restore_initial_inputs",
        "observations": 2,
        "warmups": 1,
        "measurement_repeats": 3,
        "programs": [{
            "name": "step", "pte": "step.pte", "ptd": [], "method": "forward",
            "inputs": [
                {"dtype": "float32", "shape": [1, 2]},
                {"dtype": "float32", "shape": [1, 3]},
            ],
        }],
        "bindings": [
            {"target": {"program": "step", "index": 0}, "kind": "initial",
             "tensor": {"dtype": "float32", "shape": [1, 2]}, "file": "hidden.bin"},
            {"target": {"program": "step", "index": 1}, "kind": "initial",
             "tensor": {"dtype": "float32", "shape": [1, 3]}, "file": "frame0.bin"},
            {"target": {"program": "step", "index": 1}, "kind": "stream",
             "tensor": {"dtype": "float32", "shape": [1, 3]}, "file": "frames.bin"},
        ],
        "routes": [{
            "source": {"program": "step", "index": 1},
            "target": {"program": "step", "index": 0},
            "tensor": {"dtype": "float32", "shape": [1, 2]},
        }],
        "execution_schedule": [
            {"stage": "recurrent_step", "program": "step", "observation": 0, "timed": True},
            {"stage": "recurrent_step", "program": "step", "observation": 1, "timed": True},
        ],
        "observation_output": {
            "source": {"program": "step", "index": 0},
            "tensor": {"dtype": "float32", "shape": [1, 2]},
        },
        "correctness": "correctness.npz",
        "quality": "quality.npz",
        "logical_stages": ["recurrent_step"],
        "stage_schedule": [
            {"name": "recurrent_step", "steps": 2,
             "execution": "compiled_recurrent", "timed": True},
        ],
        "parameters": {},
        "provenance": {"checkpoint": "test/checkpoint", "checkpoint_sha256": "f" * 64,
                       "full_checkpoint": True, "input_source": "test/trajectory",
                       "input_sha256": "1" * 64, "synthetic_inputs": False},
    }
    doc.update(overrides)
    path = tmp_path / "executorch_session.json"
    path.write_text(json.dumps(doc))
    return path


def _session_package(tmp_path):
    import hashlib
    import json

    manifest = _session_manifest(tmp_path)
    elf = bytearray(64)
    elf[:6] = b"\x7fELF\x02\x01"
    elf[18:20] = (243).to_bytes(2, "little")
    runner = tmp_path / "executorch_session_runner"
    runner.write_bytes(elf)
    runner.chmod(0o755)
    packages = ["executorch==test", "torch==test"]
    package_text = "\n".join(packages) + "\n"
    identity_digest = ets.session_identity_sha256(
        ets.plan_session_identity(ets.load_plan(manifest)))
    invocation = {"MERLIN_K1_TOOLCHAIN": "/exact/tc",
                  "MERLIN_K1_TOOLCHAIN_ROOT": "/exact/tc",
                  "MERLIN_MODEL2MLIR": "/exact/model2mlir",
                  "MERLIN_M2M_DIR": "/exact/model2mlir"}
    build_environment = {
        "invocation_environment": invocation,
        "invocation_environment_sha256": ets._json_sha256(invocation),
        "python": "Python test", "python_packages": packages,
        "executorch_identity": {
            "exporter_version": "test",
            "exporter_git_sha": "9" * 40,
            "source_git_sha": "9" * 40,
            "matches": True,
        },
        "python_packages_sha256": hashlib.sha256(
            package_text.encode()).hexdigest(),
        "toolchain_identity": {
            "root": "/exact/tc",
            "c_compiler": {"path": "/exact/tc/bin/clang", "sha256": "1" * 64,
                           "version": "clang test"},
            "cxx_compiler": {"path": "/exact/tc/bin/clang++", "sha256": "2" * 64,
                             "version": "clang test"},
        },
        "model2mlir_identity": {
            "path": "/exact/model2mlir",
            "git_sha": "c" * 40, "loader_sha256": "d" * 64,
            "capture_source_sha256": "e" * 64,
        },
        "external_model_source": None,
    }
    (tmp_path / "session_package.json").write_text(json.dumps({
        "schema": ets.PACKAGE_SCHEMA,
        "model": "model", "variant": "fp32", "precision": "fp32",
        "capture_sha256": "a" * 64, "framework_source_sha256": "b" * 64,
        "build_environment_sha256": ets._json_sha256(build_environment),
        "build_invocation_environment_sha256": ets._json_sha256(invocation),
        "capture_session_identity_sha256": identity_digest,
        "build_environment": build_environment,
        "xnnpack": True, "manifest": manifest.name, "runner": "executorch_session_runner",
        "observations": 2, "warmups": 1, "measurement_repeats": 3,
    }))
    from merlin.compare.freeze import sha256_paths
    return tmp_path, sha256_paths([tmp_path])


def test_continuous_session_package_is_content_addressed_and_run_only(tmp_path):
    import json
    import pytest

    root, digest = _session_package(tmp_path / "package")
    package = ets.load_session_package(root, expected_sha256=digest)
    assert package.model == "model" and package.variant == "fp32"
    assert package.plan.observations == 2
    assert package.runner == root / "executorch_session_runner"
    assert package.executorch_identity["source_git_sha"] == "9" * 40
    assert package.model2mlir_identity["git_sha"] == "c" * 40
    assert package.toolchain_identity["c_compiler"]["sha256"] == "1" * 64
    manifest = root / "executorch_session.json"
    original_manifest = manifest.read_text()
    manifest_doc = json.loads(original_manifest)
    manifest_doc["provenance"]["checkpoint"] = "different/checkpoint"
    manifest.write_text(json.dumps(manifest_doc))
    with pytest.raises(ets.ExecuTorchSessionError, match="identity digest differs"):
        ets.load_session_package(root)
    manifest.write_text(original_manifest)
    (root / "step.pte").write_bytes(b"drift")
    with pytest.raises(ets.ExecuTorchSessionError, match="package digest mismatch"):
        ets.load_session_package(root, expected_sha256=digest)


def test_paper_producer_receipt_separates_public_build_from_private_session_bytes(tmp_path):
    import json

    from merlin.compare.paper_model_object_builder import executorch_session_resources

    root, _digest = _session_package(tmp_path / "package")
    receipt = ets.write_paper_producer_receipt(root)
    compiler_input = root / ets.PAPER_COMPILER_INPUT
    resources = executorch_session_resources(compiler_input)
    assert resources.runner == root / "executorch_session_runner"
    assert resources.producer_receipt == receipt
    document = json.loads(receipt.read_text(encoding="ascii"))
    assert document["producer_id"] == ets.PAPER_PRODUCER_ID
    assert document["runner_architecture"] == {
        "elf_class": 64, "endianness": "little", "machine": "riscv", "machine_id": 243}
    private = root / document["private_files"][0]["path"]
    private.write_bytes(private.read_bytes() + b"tampered")
    # The public producer barrier must not open private measurement bytes.
    executorch_session_resources(compiler_input)
    with pytest.raises(ValueError, match="private files.*identity differs"):
        executorch_session_resources(compiler_input, include_private=True)


def test_paper_producer_receipt_rejects_runner_architecture_and_public_tampering(tmp_path):
    from merlin.compare.paper_model_object_builder import executorch_session_resources

    root, _digest = _session_package(tmp_path / "package")
    ets.write_paper_producer_receipt(root)
    compiler_input = root / ets.PAPER_COMPILER_INPUT
    runner = root / "executorch_session_runner"
    runner.write_bytes(b"\x7fELF" + b"\0" * 60)
    with pytest.raises(ValueError, match="public files.*identity differs"):
        executorch_session_resources(compiler_input)


def test_paper_producer_refuses_a_non_riscv_runner_before_issuing_authority(tmp_path):
    root, _digest = _session_package(tmp_path / "package")
    runner = root / "executorch_session_runner"
    image = bytearray(runner.read_bytes())
    image[18:20] = (62).to_bytes(2, "little")  # EM_X86_64
    runner.write_bytes(image)
    with pytest.raises(
            ets.ExecuTorchSessionError, match="ELF64 little-endian RISC-V"):
        ets.write_paper_producer_receipt(root)
    assert not (root / ets.PAPER_PRODUCER_RECEIPT).exists()
    assert not (root / ets.PAPER_COMPILER_INPUT).exists()


def test_paper_executorch_recipe_reproduces_the_sealed_runner_without_invoking_a_compiler(tmp_path):
    import hashlib

    from merlin.compare.paper_model_object_builder import (
        EXECUTORCH_RECIPE,
        regenerate_model_object,
    )

    root, _digest = _session_package(tmp_path / "package")
    receipt = ets.write_paper_producer_receipt(root)
    output = tmp_path / "verified-runner"
    result = regenerate_model_object(
        recipe=EXECUTORCH_RECIPE, registry_id="executorch_v1", target="k1",
        compiler_input=root / ets.PAPER_COMPILER_INPUT, tool=tmp_path / "must-not-run",
        output=output, source_identity_sha256="b" * 64, capture_sha256="a" * 64,
        runtime_artifact_sha256=hashlib.sha256(
            (root / "session_package.json").read_bytes()).hexdigest())
    assert output.read_bytes() == (root / "executorch_session_runner").read_bytes()
    assert result["generated_source_sha256"] == hashlib.sha256(receipt.read_bytes()).hexdigest()
    assert result["object_build_argv"][0] == "verify_executorch_sealed_session"


def test_paper_controller_materializes_sealed_executorch_runner_without_cross_compiler_execution(
        tmp_path, monkeypatch):
    import hashlib
    import json
    import shutil
    from pathlib import Path

    from merlin.compare import paper_measurement_controller as controller
    from merlin.compare.paper_model_object_builder import EXECUTORCH_RECIPE
    from merlin.compare.paper_toolchain_authority import write_toolchain_authority

    root, _digest = _session_package(tmp_path / "package")
    producer = ets.write_paper_producer_receipt(root)
    compiler_input = root / ets.PAPER_COMPILER_INPUT
    builder = root / "paper_model_object_builder.py"
    shutil.copy2(Path(controller.__file__).with_name("paper_model_object_builder.py"), builder)
    tool = root / "cross-compiler"
    shutil.copy2("/bin/true", tool)
    authority = write_toolchain_authority(
        root / "toolchain-authority.json", authority_id="sealed-executorch-test",
        target="k1", build_tool=tool)
    package_receipt = root / "package-receipt.json"
    runtime_artifact_sha256 = hashlib.sha256(
        (root / "session_package.json").read_bytes()).hexdigest()
    package_receipt.write_text(json.dumps({
        "object_recipe": EXECUTORCH_RECIPE,
        "compiler_or_framework_source_sha256": "b" * 64,
        "runtime_artifact_sha256": runtime_artifact_sha256,
        "generated_model_source_sha256": hashlib.sha256(producer.read_bytes()).hexdigest(),
    }), encoding="utf-8")

    def ref(path):
        path = Path(path)
        return {"path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    source_identity = controller._canonical_sha({
        "compiler_input": ref(compiler_input)["sha256"],
        "model_object": ref(root / "executorch_session_runner")["sha256"],
        "object_builder": ref(builder)["sha256"],
        "runner": ref(root / "executorch_session_runner")["sha256"],
    })
    authority_document = json.loads(authority.read_text(encoding="utf-8"))
    contract = {
        "registry_id": "executorch_v1", "target": "k1", "artifact_sha256": "a" * 64,
        "build": {
            "tool": ref(tool), "toolchain_authority": ref(authority),
            "build_tool_identity_sha256": authority_document["tool"]["identity_sha256"],
            "sources": {"runner": ref(root / "executorch_session_runner"),
                        "model_object": ref(root / "executorch_session_runner")},
            "inputs": {"compiler_input": ref(compiler_input), "object_builder": ref(builder),
                       "package_receipt": ref(package_receipt)},
            "argv": list(controller._EXECUTORCH_PRODUCTION_BUILD_ARGV),
            "environment": {}, "timeout_seconds": 10,
            "expected_executable_sha256": ref(root / "executorch_session_runner")["sha256"],
            "source_identity_sha256": source_identity,
        },
    }
    monkeypatch.setattr(
        controller.subprocess, "run",
        lambda *_args, **_kwargs: pytest.fail("sealed ExecuTorch verification invoked a compiler"))
    output = tmp_path / "verified-executable"
    result = controller._build_executable(root, contract, output, lambda: 10.0)
    assert result["operation"] == "sealed_executable_verification"
    assert result["executable_sha256"] == ref(root / "executorch_session_runner")["sha256"]
    assert output.read_bytes() == (root / "executorch_session_runner").read_bytes()


def test_continuous_session_package_rejects_exporter_runtime_source_mismatch(tmp_path):
    import json
    import pytest

    root, _digest = _session_package(tmp_path / "package")
    metadata = root / "session_package.json"
    doc = json.loads(metadata.read_text())
    doc["build_environment"]["executorch_identity"].update(
        source_git_sha="8" * 40, matches=False)
    doc["build_environment_sha256"] = ets._json_sha256(doc["build_environment"])
    metadata.write_text(json.dumps(doc))
    with pytest.raises(ets.ExecuTorchSessionError, match="build identity.*mismatched"):
        ets.load_session_package(root)


def test_continuous_session_package_rejects_misnamed_full_environment_digest(tmp_path):
    import json

    root, _digest = _session_package(tmp_path / "package")
    metadata = root / "session_package.json"
    doc = json.loads(metadata.read_text())
    doc["build_environment_sha256"] = doc["build_invocation_environment_sha256"]
    metadata.write_text(json.dumps(doc))
    with pytest.raises(ets.ExecuTorchSessionError, match="full build environment digest"):
        ets.load_session_package(root)


def test_session_build_requires_explicit_toolchain_environment_not_dotenv(
        monkeypatch, tmp_path):
    monkeypatch.delenv("MERLIN_K1_TOOLCHAIN", raising=False)
    monkeypatch.delenv("MERLIN_K1_TOOLCHAIN_ROOT", raising=False)
    monkeypatch.setattr(ets.k1, "_toolchain_root", lambda: tmp_path / "hidden-dotenv-toolchain")
    with pytest.raises(ets.ExecuTorchSessionError, match="explicit MERLIN_K1_TOOLCHAIN"):
        ets._explicit_toolchain_root()


@pytest.mark.parametrize("variant", ["fp32", "w8a8"])
def test_continuous_session_build_blocks_identity_mismatch_before_precision_dispatch(
        monkeypatch, tmp_path, variant):
    def mismatch():
        raise ets.ExecuTorchSessionError(
            "ExecuTorch exporter/runtime source identity mismatch")

    monkeypatch.setattr(ets, "_require_executorch_identity", mismatch)
    with pytest.raises(ets.ExecuTorchSessionError, match="identity mismatch"):
        ets.build_session_package(
            "unused", variant, tmp_path / "session_contract.yaml", tmp_path / "package",
            observations=1, warmups=1, measurement_repeats=1,
            framework_source_sha256="a" * 64,
            build_invocation_environment_sha256="b" * 64)


def test_continuous_session_plan_is_abi_closed_and_renders_execution_only_timer(tmp_path):
    plan = ets.load_plan(_session_manifest(tmp_path))
    assert plan.observations == 2
    assert plan.stages == ("recurrent_step",)
    assert plan.routes[0].source == ets.Endpoint("step", 1)

    source = ets.render_runner_source(plan)
    # Timer surrounds exactly Module::execute; input streams and state copies are outside it.
    before = source.index("const uint64_t before = now_ns();")
    execute = source.index("->execute(", before)
    after = source.index("now_ns() - before", execute)
    assert before < execute < after
    assert "num_executions" not in source
    assert "buffers[i] = initial[i]" in source            # reset each full-session repeat
    assert "ET_SESSION_AFFINITY" in source                # observed, not echoed core count
    assert "_unsafe_reset_threadpool(requested_threads)" in source
    assert "NoThreadPoolGuard" in source                  # official 1-core no-pool semantics
    assert "ET_SESSION_THREADS" in source


def test_continuous_session_parser_requires_exact_stage_sum_and_affinity(tmp_path):
    plan = ets.load_plan(_session_manifest(tmp_path))
    log = "\n".join([
        "ET_SESSION_AFFINITY 4 sched_getaffinity",
        "ET_SESSION_THREADS 4 extension_threadpool",
        "ET_SESSION_VLEN 256 csr",
        "ET_SESSION_STAGE 0 recurrent_step 10", "ET_SESSION_REPEAT 0 10",
        "ET_SESSION_STAGE 1 recurrent_step 11", "ET_SESSION_REPEAT 1 11",
        "ET_SESSION_STAGE 2 recurrent_step 12", "ET_SESSION_REPEAT 2 12",
        "ET_SESSION_RSS 4096", "ET_SESSION_DONE 2 3",
    ])
    result = ets.parse_session_console(log, plan, requested_cores=4)
    assert result.samples == (10, 11, 12)
    assert result.stage_samples == {"recurrent_step": (10, 11, 12)}
    assert result.affinity_count == 4 and result.affinity_source == "sched_getaffinity"
    assert result.worker_threads == 4 and result.worker_thread_source == "extension_threadpool"
    assert result.median == 11 and result.p95 == 12

    one_core_log = log.replace(
        "AFFINITY 4 sched_getaffinity", "AFFINITY 1 sched_getaffinity").replace(
        "THREADS 4 extension_threadpool",
        "THREADS 1 extension_threadpool_no_pool_guard")
    one_core = ets.parse_session_console(one_core_log, plan, requested_cores=1)
    assert one_core.worker_threads == 1
    assert one_core.worker_thread_source == "extension_threadpool_no_pool_guard"

    import pytest
    with pytest.raises(ets.ExecuTorchSessionError, match="affinity"):
        ets.parse_session_console(log, plan, requested_cores=8)
    with pytest.raises(ets.ExecuTorchSessionError, match="exact sum"):
        ets.parse_session_console(log.replace("REPEAT 1 11", "REPEAT 1 99"), plan,
                                  requested_cores=4)
    with pytest.raises(ets.ExecuTorchSessionError, match="worker threads"):
        ets.parse_session_console(
            one_core_log.replace("extension_threadpool_no_pool_guard", "extension_threadpool"),
            plan, requested_cores=1)


def test_continuous_session_fails_closed_on_precision_and_route_abi(tmp_path):
    import json
    import pytest

    with pytest.raises(ets.ExecuTorchSessionError, match="fp32 only"):
        ets.load_plan(_session_manifest(tmp_path, precision="w8a8"))

    path = _session_manifest(tmp_path)
    doc = json.loads(path.read_text())
    doc["routes"][0]["tensor"]["shape"] = [1, 3]
    path.write_text(json.dumps(doc))
    with pytest.raises(ets.ExecuTorchSessionError, match="route tensor differs"):
        ets.load_plan(path)


def test_continuous_session_wrapper_cmake_is_well_formed(tmp_path):
    plan = ets.load_plan(_session_manifest(tmp_path / "session"))
    project = ets.write_runner_project(plan, tmp_path / "build")
    cmake = (project / "CMakeLists.txt").read_text()
    assert 'add_subdirectory("${EXECUTORCH_ROOT}" executorch)' in cmake
    assert 'target_link_libraries(executorch_session_runner' in cmake
    assert "extension_threadpool" in cmake
    assert "EXECUTORCH_BUILD_PTHREADPOOL ON" in cmake
    assert "EXECUTORCH_BUILD_CPUINFO ON" in cmake
    assert '\\n"' not in cmake


def test_continuous_session_paper_sections_use_full_trajectory_and_opaque_stage_truth(tmp_path):
    import json
    from dataclasses import replace

    import numpy as np

    path = _session_manifest(tmp_path)
    doc = json.loads(path.read_text())
    doc.update(correctness_key="commands", quality_key="commands",
               logical_stages=["visual_encode", "recurrent_step", "predict"],
               stage_attribution="opaque_whole_forward",
               parameters={"timed_stages": ["visual_encode", "recurrent_step", "predict"]})
    path.write_text(json.dumps(doc))
    expected = np.array([[[1.0, 2.0]], [[3.0, 4.0]]], dtype=np.float32)
    np.savez(tmp_path / "correctness.npz", commands=expected)
    np.savez(tmp_path / "quality.npz", commands=expected)
    (tmp_path / "trajectory.bin").write_bytes(expected.tobytes())
    plan = ets.load_plan(path)
    log = "\n".join([
        "ET_SESSION_AFFINITY 2 sched_getaffinity",
        "ET_SESSION_THREADS 2 extension_threadpool", "ET_SESSION_VLEN 256 csr",
        "ET_SESSION_STAGE 0 recurrent_step 10", "ET_SESSION_REPEAT 0 10",
        "ET_SESSION_STAGE 1 recurrent_step 11", "ET_SESSION_REPEAT 1 11",
        "ET_SESSION_STAGE 2 recurrent_step 12", "ET_SESSION_REPEAT 2 12",
        "ET_SESSION_RSS 4096", "ET_SESSION_DONE 2 3",
    ])
    run = ets.parse_session_console(
        log, plan, requested_cores=2, trajectory=tmp_path / "trajectory.bin")
    conditions = {"governor": "performance", "current_khz": 1600000,
                  "max_khz": 1600000, "max_thermal_millic": 42000}
    run = replace(run, board_conditions={"before": conditions, "after": conditions})
    sections = ets.paper_sections(
        plan, run, requested_cores=2, quality_metric="output_cosine", quality_min=0.99,
        framework_source_sha256="a" * 64)
    assert sections["correctness"]["gate_ok"] is True
    assert sections["quality"]["value"] > 0.999999
    assert sections["timing"]["samples"] == [10, 11, 12]
    assert sections["timing"]["timed_stages"] == [
        "visual_encode", "recurrent_step", "predict"]
    assert sections["timing"]["stage_samples"] == {}  # never fabricate an internal split
    assert sections["provenance"]["stage_attribution"] == "opaque_whole_forward"
    assert sections["execution"]["core_count"] == 2


# --- the runner's kernel registry is a LINK-TIME set, derived per .pte -------------------------
#
# Measured on the board 2026-09-04: `spectformer_int8_full` aborted at Method::load with "There are
# 12 instructions don't have corresponding operator registered". Reproduced host-side with an x86
# executor_runner built from the same kernel configuration: the 12 were aten::view_as_complex_copy
# .out (x4), aten::_fft_r2c.out (x4) and aten::_fft_c2r.out (x4). The two FFT operators live only in
# ExecuTorch's `optimized` kernel library, which the build did not link; view_as_complex_copy.out is
# in no ExecuTorch kernel library at all. These tests hold both halves of that distinction.

def test_kernel_yaml_operators_reads_both_entry_spellings(tmp_path):
    # ExecuTorch spells an entry either `- op: <name>` (implicitly the aten namespace) or
    # `- func: <ns>::<name>(<schema>)`, and mixes the two in one file. Both must land on the
    # registry key the runtime actually looks up, or the plan compares nothing.
    y = tmp_path / "functions.yaml"
    y.write_text(
        "- op: mul.out\n"
        "  kernels:\n"
        "    - arg_meta: null\n"
        "      kernel_name: torch::executor::mul_out\n"
        "- func: quantized_decomposed::add.out(Tensor a, float s, *, Tensor(a!) out) -> Tensor(a!)\n"
        "  kernels:\n"
        "    - arg_meta: null\n"
        "      kernel_name: torch::executor::quantized_add_out\n")
    assert et._kernel_yaml_operators(y) == {"aten::mul.out", "quantized_decomposed::add.out"}


def test_pinned_kernel_yamls_place_the_fft_ops_outside_portable():
    # The fact that made the board fail, asserted against the PINNED source rather than restated:
    # the FFT ops are optimized-only, so a portable-only runner cannot load spectformer.
    portable = et._ET_SRC / "kernels/portable/functions.yaml"
    optimized = et._ET_SRC / "kernels/optimized/optimized.yaml"
    if not portable.is_file() or not optimized.is_file():
        pytest.skip("ExecuTorch submodule not checked out")
    p, o = et._kernel_yaml_operators(portable), et._kernel_yaml_operators(optimized)
    assert "aten::mul.out" in p                     # the always-linked set is being read at all
    for op in ("aten::_fft_r2c.out", "aten::_fft_c2r.out"):
        assert op in o and op not in p, f"{op} moved between kernel libraries; re-plan the build"


def test_plan_routes_ops_to_the_library_that_owns_them(monkeypatch, tmp_path):
    # Portable-only operators must NOT turn on an extra library (that would change every existing
    # cell's kernel set); an optimized-only one must turn on exactly its cmake option.
    monkeypatch.setattr(et, "pte_operators", lambda pte, **kw: {"aten::mul.out": 3})
    plan = et.plan_kernels(tmp_path / "model.pte")
    assert plan.libraries == set() and plan.cmake_options == () and not plan.missing

    monkeypatch.setattr(et, "pte_operators",
                        lambda pte, **kw: {"aten::mul.out": 3, "aten::_fft_r2c.out": 4})
    plan = et.plan_kernels(tmp_path / "model.pte")
    assert plan.libraries == {"optimized"}
    assert plan.cmake_options == ("EXECUTORCH_BUILD_KERNELS_OPTIMIZED",)
    assert not plan.missing


def test_plan_names_an_operator_executorch_does_not_implement(monkeypatch, tmp_path):
    # No build configuration registers view_as_complex_copy.out, so the gap must be NAMED (and
    # counted in the runtime's own unit, instructions) rather than spent as a board slot.
    monkeypatch.setattr(et, "pte_operators", lambda pte, **kw: {
        "aten::mul.out": 3, "aten::view_as_complex_copy.out": 4})
    plan = et.plan_kernels(tmp_path / "model.pte")
    assert plan.missing == {"aten::view_as_complex_copy.out": 4}
    assert plan.n_missing_instructions == 4
    reason = plan.missing_reason()
    assert "aten::view_as_complex_copy.out" in reason and "4 instructions" in reason


def test_run_model_reports_the_missing_operator_instead_of_running_the_board(monkeypatch, tmp_path):
    # A program the runtime would refuse at load must come back not_run WITH the operator named,
    # and must never reach the board — the board can only rediscover the same abort.
    from merlin.baselines import bundle as _bundle

    (tmp_path / "golden.npy").write_bytes(b"\x00" * 8)
    (tmp_path / "inputs.npz").write_bytes(b"\x00" * 8)
    bnd = _bundle.CaptureBundle(model="spectformer", variant="int8", root=tmp_path)
    monkeypatch.setattr(et, "resolve_bundle", lambda m, v="fp32": bnd)
    monkeypatch.setattr(et, "et_venv_available", lambda: True)
    monkeypatch.setattr(et, "export_pte", lambda model, b, work, **kw: et.ExportResult(
        pte=tmp_path / "model.pte", ptd_files=[], input_files=[], golden=bnd.golden))
    monkeypatch.setattr(et, "pte_operators", lambda pte, **kw: {
        "aten::mul.out": 3, "aten::view_as_complex_copy.out": 4})
    monkeypatch.setattr(et, "cross_compile_runner", lambda work, **kw: tmp_path / "executor_runner")
    monkeypatch.setattr(et, "audit_binary", lambda runner: (0.5, [], {}))
    board = []
    monkeypatch.setattr(et, "_do_board", lambda *a, **k: board.append(1))

    r = et.run_model("spectformer", "int8", write=False, work_root=tmp_path)
    assert board == [], "a program the runtime refuses at load must not consume a board slot"
    assert r.built is True                       # export + binary are real; only the run is blocked
    assert r.status() == "not_run" and "aten::view_as_complex_copy.out" in r.gap_reason


def test_kernel_options_reach_cmake_and_get_their_own_build_dir(monkeypatch, tmp_path):
    # Two kernel sets are two different binaries. They must not share a build dir (a cached runner
    # with the wrong registry is exactly the failure this whole path exists to prevent), and the
    # options must actually reach the configure line.
    monkeypatch.setattr(et, "et_identity", lambda: None)
    monkeypatch.setattr(et.rvv_audit, "enforce_rvv_march", lambda m: m)
    monkeypatch.setattr(et, "_toolchain_root", lambda: tmp_path / "tc")
    monkeypatch.setattr(et, "et_venv_python", lambda: tmp_path / "python")
    seen = []

    def _fake_run(argv, **kw):
        seen.append([str(a) for a in argv])
        if argv[0] == "cmake" and argv[1] == "--build":
            out = pathlib.Path(argv[2]) / "executor_runner"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"\x7fELF")
        class _P:
            returncode, stdout, stderr = 0, "ELF 64-bit LSB, UCB RISC-V", ""
        return _P()

    monkeypatch.setattr(et.subprocess, "run", _fake_run)
    plain = et.cross_compile_runner(tmp_path / "w")
    opt = et.cross_compile_runner(
        tmp_path / "w", kernel_options=("EXECUTORCH_BUILD_KERNELS_OPTIMIZED",))
    assert plain.parent != opt.parent, "kernel set is part of the binary's identity"
    configures = [a for a in seen if a[0] == "cmake" and a[1] == "-S"]
    assert not any("-DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON" in a for a in configures[:1])
    assert "-DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON" in configures[-1]
