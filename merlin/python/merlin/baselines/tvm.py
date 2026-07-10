"""TVM (Apache TVM / Relax) baseline arm — ingest OUR models and run them on the K1 with RVV.

TVM is the second external-framework arm. Unlike Buddy (which ingests our ``model.mlir``
directly), TVM imports from the *PyTorch* graph — but via **ONNX**, not the torch-exported-program
frontend. TVM v0.19.0's ``relax.frontend.torch.from_exported_program`` lacks ops for HF
transformer/VLA graphs (``full``/``where``/``masked_fill``/``convolution``/…); its **ONNX**
frontend (``relax.frontend.onnx.from_onnx``) has far broader coverage. So this arm exports the
seeded instance to ONNX (``torch.onnx.export``), imports that into Relax, lowers for ``rv64gcv``,
and emits a deployable module. The emitted kernel object is RVV-audited exactly like every other
arm, so "TVM used RVV" is *proven by disassembly*, never assumed.

**Pinned TVM = v0.19.0** (``third_party/baselines/tvm`` @ tag ``v0.19.0``), built against the
system **LLVM 18** (``/usr/bin/llvm-config-18``). This is a deliberate re-pin from the earlier
``main`` snapshot, which (a) would not compile against LLVM 23, (b) shipped no MetaSchedule, and
(c) had a bool-op LLVM-codegen bug that blocked every model. v0.19.0 has MetaSchedule + AutoTVM +
the classic ``tvm.contrib.cc`` cross-compiler and an ONNX frontend that lowers our graphs.

Pipeline (per model; **int8 variant first**, then fp32)::

    seeded torch model (== capture recipe: manual_seed(0), M2M_LLAMA_LAYERS, zero-param perturb,
                        per-cfg torchao int8 quant for the int8 variant)
      -> torch.onnx.export (classic opset17, dynamo opset18 fallback)      [m2m venv: torch 2.x]
      -> tvm.relax.frontend.onnx.from_onnx                                 [Relax IRModule]
      -> [optional] meta_schedule.tune_relax over a K1 RPC runner          [RVV autotune, opt-in]
      -> relax.build / compile_relax(db) for rv64gcv                        [RVV LLVM codegen]
      -> mod.export_library(fcompile=<SpacemiT clang cross>)  -> model_tvm.so  [the RVV artifact]
      -> rvv_audit.audit_binary(.so)   : mechanical RVV%/scalar-fallback honesty
      -> host reference: TVM VM on x86 vs the torch reference for THIS instance (gate) + vs the
                         capture golden (reported)                          [board-independent]
      -> on-board run (board_lock): the module .so + a C harness cross-linked with SpacemiT clang,
                                    scp'd + run -> E2E rdtime/wall + OUT-vs-golden cos/rel
                                    [fail-closed if the board is down]

Two compat shims are needed against onnx 1.22 / TVM v0.19.0: (1) a reconstructed ``onnx.mapping``
module (removed in onnx 1.22, still imported by TVM's ONNX frontend), (2) registered
``relax.isnan``/``isinf`` legalizations (missing from the base set, else VM codegen rejects the
un-lowered intrinsic). Both are pure/faithful (dtype table; topi TIR) — no invented semantics.

Because torch + TVM's Python package must co-run (the main ``.venv`` has no torch), the
import+compile step executes as a **subprocess in the model2MLIR venv** (``$MERLIN_MODEL2MLIR
/.venv``, torch 2.x + transformers) with ``PYTHONPATH=<tvm>/python`` and
``LD_LIBRARY_PATH=<lib>`` set. This module writes a driver script, runs it there, and reads back a
JSON status + the emitted ``.so``.

RVV coverage note (honest): TVM v0.19.0's *default* ``relax.build`` lowering emits largely scalar
CPU loops for an rv64gcv target (LLVM does not auto-RVV-vectorize them; there is no dlight-cpu
schedule in this release). Real RVV coverage comes from **MetaSchedule autotuning** (``tune_relax``),
which must measure on the device — an on-K1 RPC runner. That is expensive (single shared board,
queued across agents), so it is **opt-in** via ``MERLIN_TVM_TUNE=1`` (+ ``MERLIN_TVM_RPC_*``); the
default build ships correct-but-mostly-scalar code and the RVV audit records exactly that
(``rvv_coverage_overall`` + per-symbol ``scalar_compute``) — never averaged away.

Honesty (``not_run_is_not_pass``): a model that will not export/import/compile is a ``not_built``
result with a specific ``gap_reason``; a built-but-unrun model (board down / won't fit K1 RAM) is
``not_run`` with a reason. We NEVER fabricate a cos/rel or a cycle count.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from merlin.baselines import bundle as _bundle
from merlin.baselines import k1_exec, profile, rvv_audit
from merlin.baselines.contract import BaselineResult, RegionProfile, ScalarFallback
from merlin.common import artifacts
from merlin.common.paths import repo_root
from merlin.rvvgen import k1

FRAMEWORK = "tvm"

# --- TVM build layout (gitignored; built by this arm) -------------------------------------------
_BUILD_ROOT = repo_root() / "build" / "baselines" / "tvm"
_TVM_SRC = repo_root() / "third_party" / "baselines" / "tvm"

# The RVV target: LLVM riscv64 with the vector extension enabled. ``+zvl256b`` pins the vector
# register width to the K1's VLEN (256b / vlenb=32) so codegen + MetaSchedule size vscale to the
# real silicon. ``num-cores=8`` = the board's core count (MetaSchedule cost model / parallelism).
# TVM v0.19.0 accepts BOTH the CLI string and the JSON-dict target form; we use the dict (typed,
# unambiguous) for build and keep the string only for the RVV-march honesty check + display.
TVM_TARGET_CONFIG: dict = {
    "kind": "llvm",
    "mtriple": "riscv64-unknown-linux-gnu",
    "mattr": ["+m", "+f", "+d", "+c", "+v", "+zvl256b"],
    "mcpu": "generic-rv64",
    "mabi": "lp64d",
    "num-cores": 8,
}
TVM_TARGET = ("llvm -mtriple=riscv64-unknown-linux-gnu "
              "-mattr=+m,+f,+d,+c,+v,+zvl256b -mcpu=generic-rv64 -mabi=lp64d")


def _env_path(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    return Path(v) if v else default


def tvm_lib_dir() -> Path:
    """Directory holding the built ``libtvm.so`` / ``libtvm_runtime.so`` (ninja emits them under
    ``<build>/lib``). Override with ``MERLIN_TVM_LIBRARY_PATH``."""
    override = os.environ.get("MERLIN_TVM_LIBRARY_PATH")
    if override:
        return Path(override)
    lib = _BUILD_ROOT / "lib"
    return lib if lib.is_dir() else _BUILD_ROOT


def tvm_python_path() -> Path:
    """PYTHONPATH entry so the driving venv can ``import tvm`` from the built (uninstalled) tree."""
    return _TVM_SRC / "python"


def m2m_python() -> Path | None:
    """The model2MLIR venv python (has torch + transformers). Import+export+compile run here."""
    p = _bundle.model2mlir_root() / ".venv" / "bin" / "python"
    return p if p.is_file() else None


# Some VLA loaders need per-model framework deps the generic m2m venv lacks (smolvla -> lerobot;
# bitvla -> the vendored BitNet transformers fork). Each ships a dedicated capture venv beside the
# repo (``<model>_capture/.venv``) that CAN torch-load the model; we prep those venvs with the TVM
# python deps (onnx/decorator/attrs/...) so the same driver can run there. Override the base dir
# with ``MERLIN_CAPTURE_VENVS_ROOT``.
_CAPTURE_VENVS_ROOT = Path(os.environ.get("MERLIN_CAPTURE_VENVS_ROOT",
                                         "/scratch/agustin/projects"))
_MODEL_CAPTURE_VENV = {"smolvla": "smolvla_capture", "bitvla": "bitvla_capture", "pi05": "openpi"}


def driver_python(model: str) -> Path | None:
    """Python that can BOTH torch-load ``model`` and import the built TVM. Prefer a model-specific
    capture venv (has the model's framework deps) when present, else the m2m venv."""
    name = _MODEL_CAPTURE_VENV.get(model)
    if name:
        p = _CAPTURE_VENVS_ROOT / name / ".venv" / "bin" / "python"
        if p.is_file():
            return p
    return m2m_python()


def tvm_built() -> bool:
    """True iff the TVM shared libs are present (the Python package needs them to import)."""
    lib = tvm_lib_dir()
    return any((lib / n).is_file() for n in ("libtvm.so", "libtvm_runtime.so"))


def tvm_available() -> bool:
    """True iff TVM is built AND a torch-capable python (m2m venv) exists to drive the import."""
    return tvm_built() and m2m_python() is not None


def tvm_commit() -> str:
    try:
        r = subprocess.run(["git", "-C", str(_TVM_SRC), "describe", "--tags", "--always"],
                           capture_output=True, text=True, timeout=15)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:  # noqa: BLE001
        return ""


# --- capture-bundle resolution (robust to legacy dir names) -------------------------------------
# The fp32 LLM captures predate the ``<model>_fp32_consistent`` convention (see buddy.py). int8
# variants already follow the convention (``<model>_int8_consistent``).
_LEGACY_FP32_DIRS: dict[str, str] = {
    "tiny_llama": "tiny_consistent",
    "small_llama": "small_consistent",
}


def resolve_bundle(model: str, variant: str = "int8") -> _bundle.CaptureBundle:
    """Resolve a capture bundle, falling back to legacy dir names for the fp32 LLMs."""
    b = _bundle.resolve(model, variant)
    if b.golden.is_file() or b.mlir.is_file():
        return b
    if variant == "fp32" and model in _LEGACY_FP32_DIRS:
        legacy = artifacts.recaptures_dir() / _LEGACY_FP32_DIRS[model]
        if (legacy / "golden.npy").is_file():
            return _bundle.CaptureBundle(model=model, variant=variant, root=legacy)
    return b


def golden_path(b: _bundle.CaptureBundle) -> Path:
    """The correctness reference: prefer the W8A8 int8 golden when present, else golden.npy."""
    w8a8 = b.root / "golden_w8a8.npy"
    return w8a8 if w8a8.is_file() else b.golden


# --- workload cfg (mirror the capture recipe so the golden is reproducible) ---------------------

def _workload_env(model: str, *, full: bool = False) -> dict[str, str]:
    """The loader env for a workload so the exported instance matches the golden.

    ``full=True`` (a ``_full`` recapture): use ``bundle.full_env`` — the full-fidelity/native
    architecture the ``<model>_int8_full`` golden was computed on (e.g. real 22-layer TinyLlama,
    30-layer BitNet), NOT the TOML truncation defaults. ``full=False``: the TOML ``[env]`` (the
    truncated ``_consistent`` recapture, e.g. ``M2M_LLAMA_LAYERS=2``)."""
    if full:
        return _bundle.full_env(model)
    toml_env: dict[str, str] = {}
    wdir = _bundle.model2mlir_root() / "workloads" / model
    for cand in list(wdir.glob("*.toml")):
        try:
            import tomllib
            data = tomllib.loads(cand.read_text())
        except Exception:  # noqa: BLE001
            continue
        for k, v in (data.get("env") or {}).items():
            toml_env[str(k)] = str(v)
    return toml_env


# Models whose captured weights exceed the K1's usable RAM/disk (board is ~3.4G RAM, /tmp 1.9G
# tmpfs, rootfs ~12G). We still ATTEMPT compile + audit (host-side), but flag the on-board run as a
# fit gap rather than pretend it ran. Sizes checked at run time against the actual weights blob.
_K1_WEIGHT_FIT_BYTES = 3_000_000_000


class TVMError(RuntimeError):
    pass


# --- the m2m-venv driver: torch.export -> Relax -> (tune) -> RVV .so ----------------------------

_DRIVER_TEMPLATE = r'''
import json, os, sys, traceback
import numpy as np

OUT = {"stage": "start", "ok": False}
def emit(**kw):
    OUT.update(kw)
    with open(os.environ["MERLIN_TVM_STATUS"], "w") as f:
        json.dump(OUT, f, indent=2, default=str)

try:
    model = os.environ["MERLIN_TVM_MODEL"]
    variant = os.environ["MERLIN_TVM_VARIANT"]
    bundle_root = os.environ["MERLIN_TVM_BUNDLE"]
    work = os.environ["MERLIN_TVM_WORK"]
    target_cfg = json.loads(os.environ["MERLIN_TVM_TARGET"])
    cross_cc = os.environ.get("MERLIN_TVM_CROSS_CC", "")
    do_tune = os.environ.get("MERLIN_TVM_TUNE", "0") == "1"
    golden_file = os.environ["MERLIN_TVM_GOLDEN"]
    so_path = os.path.join(work, "model_tvm.so")

    import torch
    emit(stage="torch_imported", torch=torch.__version__)

    # 1. Reproduce the seeded capture instance (== workloads/capture_consistent.py).
    workloads = os.path.join(os.environ["MERLIN_MODEL2MLIR"], "workloads")
    sys.path.insert(0, os.path.join(workloads, model))
    sys.path.insert(0, workloads)
    from loader import get_model_and_inputs  # type: ignore
    torch.manual_seed(0); np.random.seed(0)
    mdl, inputs = get_model_and_inputs()
    mdl.eval()
    inputs = tuple(inputs)
    with torch.no_grad():
        for p in mdl.parameters():
            if float(p.detach().abs().max()) == 0.0:
                p.copy_(torch.randn_like(p) * 0.02)
    # bf16 -> f32 UPCAST (lossless), BEFORE quant/export. The real SmolVLA checkpoint runs in
    # bfloat16, and relax/ONNX lack bf16 support for several ops (e.g. relax.sigmoid rejects bf16).
    # Cast the base model + its floating inputs to f32 so activations flow as f32 (weight-only int8
    # is then applied on the f32 base). Guarded: only fires when the model actually carries bf16, so
    # the already-f32 models are untouched.
    bf16_note = ""
    if (any(p.dtype == torch.bfloat16 for p in mdl.parameters())
            or any(b.dtype == torch.bfloat16 for b in mdl.buffers())):
        mdl = mdl.to(torch.float32)
        inputs = tuple(x.to(torch.float32) if x.is_floating_point() else x for x in inputs)
        bf16_note = "bf16->f32 upcast"
    # int8/fp8: apply the SAME torchao quantization the capture used (per-cfg scheme).
    quant_note = ""
    if variant != "fp32":
        try:
            import pathlib
            from capture import _quant_for, _load_toml  # type: ignore
            cfg = _load_toml(pathlib.Path(workloads) / model)  # _load_toml wants a Path, not str
            q = _quant_for(cfg, variant)
            if q is not None:
                from m2m.capture.torchao_pipeline import apply_quantization  # type: ignore
                mdl = apply_quantization(mdl, q)
                quant_note = "torchao:" + str(getattr(q, "scheme", "?"))
        except Exception as e:  # noqa: BLE001
            quant_note = "quant-apply-failed:" + str(e)[:160]
    # Make NON-PERSISTENT buffers persistent so they land in state_dict. torch 2.10 lifts
    # non-persistent buffers (e.g. rope inv_freq / original_inv_freq) into ep.constants but the
    # graph signature still types them InputKind.BUFFER — TVM v0.19.0's exported-program frontend
    # then looks them up only in state_dict and KeyErrors. Re-registering them persistent puts them
    # in state_dict so the frontend resolves them (functionally identical: same tensor value).
    n_persisted = 0
    for mod_ in mdl.modules():
        npset = getattr(mod_, "_non_persistent_buffers_set", None)
        if npset:
            for bname in list(npset):
                npset.discard(bname)
                n_persisted += 1
    emit(stage="model_built", quant=quant_note, persisted_buffers=n_persisted, bf16=bf16_note)

    # 2. Export to ONNX (the import path). ONNX has FAR broader op coverage than TVM v0.19.0's
    #    torch-exported-program frontend (which lacks full/where/masked_fill/convolution/... for HF
    #    transformer/VLA graphs). torch.onnx dynamo=True handles rope's aten::diff etc. that the
    #    classic tracer rejects.
    onnx_path = os.path.join(work, "model.onnx")
    input_names = [f"in{i}" for i in range(len(inputs))]
    try:
        torch.onnx.export(mdl, inputs, onnx_path, opset_version=17,
                          input_names=input_names, dynamo=False)
        onnx_note = "onnx:classic-opset17"
    except Exception as e_classic:  # noqa: BLE001
        torch.onnx.export(mdl, inputs, onnx_path, opset_version=18, dynamo=True)
        onnx_note = "onnx:dynamo-opset18 (classic failed: " + str(e_classic)[:80] + ")"
    emit(stage="exported", onnx=onnx_note)

    # Torch reference on THESE inputs — the true correctness reference for what we compiled (the
    # capture goldens were seeded/quantized differently, so we ALSO report vs them but gate here).
    with torch.no_grad():
        _r = mdl(*inputs)
    torch_ref = (_r[0] if isinstance(_r, (tuple, list)) else _r).detach().float().cpu().numpy()
    np.save(os.path.join(work, "torch_ref.npy"), torch_ref)
    # Save the ACTUAL loader inputs the model+torch_ref were built on. The bundle inputs.npz can use a
    # DIFFERENT config than the loader (e.g. openvla loader exports pixel_values (1,6,64,64) while the
    # recapture bundle is (1,6,224,224)) -> feeding bundle inputs to the compiled model is a shape
    # mismatch. Host-VM + on-board runs feed THESE (self-consistent TVM-vs-torch gate).
    _din = {("in%d" % i): np.asarray(t.detach().cpu().numpy()) for i, t in enumerate(inputs)}
    np.savez(os.path.join(work, "driver_inputs.npz"), **_din)

    # 3. Import ONNX -> TVM Relax.
    import onnx as _onnx
    import types as _types
    # onnx 1.22 removed onnx.mapping; TVM v0.19.0's onnx frontend still imports it. Reconstruct the
    # TENSOR_TYPE_TO_NP_TYPE map from onnx.helper (pure dtype table, no semantics invented).
    if not hasattr(_onnx, "mapping"):
        from onnx import helper as _oh
        _mm = _types.ModuleType("onnx.mapping")
        _mm.TENSOR_TYPE_TO_NP_TYPE = {dt: _oh.tensor_dtype_to_np_dtype(dt)
                                      for dt in _oh.get_all_tensor_dtypes()}
        sys.modules["onnx.mapping"] = _mm
        _onnx.mapping = _mm
    import tvm
    from tvm import relax, topi
    from tvm.relax.frontend.onnx import from_onnx
    # Register legalizations the base set lacks (isnan/isinf) so LegalizeOps -> TIR PrimFunc instead
    # of leaving a relax intrinsic the VM codegen rejects. Real semantics via topi.
    from tvm.relax.transform.legalize_ops.common import _call_topi_without_attr, register_legalize
    for _nm, _fn in (("relax.isnan", getattr(topi, "isnan", None)),
                     ("relax.isinf", getattr(topi, "isinf", None))):
        if _fn is not None:
            try:
                register_legalize(_nm, _call_topi_without_attr(_fn, _nm.replace("relax.", "tir_")))
            except Exception:  # noqa: BLE001 - already registered on a re-run
                pass
    emit(stage="tvm_imported", tvm=getattr(tvm, "__version__", "?"), onnx=onnx_note)

    onnx_model = _onnx.load(onnx_path)
    mod = from_onnx(onnx_model, keep_params_in_input=False)
    emit(stage="relax_imported")

    target = tvm.target.Target(target_cfg)

    # 4. (optional) MetaSchedule autotune for RVV. Requires a K1 RPC runner to measure on-device;
    #    opt-in via MERLIN_TVM_TUNE=1 + MERLIN_TVM_RPC_{TRACKER,KEY}. Default OFF (queued board).
    tuned_note = "no-tune (default relax.build; RVV via LLVM only)"
    ex = None
    if do_tune:
        try:
            from tvm import meta_schedule as ms
            tracker = os.environ.get("MERLIN_TVM_RPC_TRACKER", "")
            key = os.environ.get("MERLIN_TVM_RPC_KEY", "k1")
            trials = int(os.environ.get("MERLIN_TVM_TUNE_TRIALS", "200"))
            if tracker:
                host, port = tracker.split(":")
                runner = ms.runner.RPCRunner(ms.runner.RPCConfig(
                    tracker_host=host, tracker_port=int(port), tracker_key=key,
                    session_timeout_sec=120))
            else:
                runner = "local"  # cannot execute rv64gcv on x86; only valid with an RPC runner
            db = ms.tune_relax(mod=mod, params={}, target=target,
                               work_dir=os.path.join(work, "ms_tune"),
                               max_trials_global=trials, runner=runner)
            ex = ms.compile_relax(db, mod, target, params=None)
            tuned_note = f"metaschedule trials={trials} runner={'rpc' if tracker else 'local'}"
        except Exception as e:  # noqa: BLE001
            tuned_note = "tune-failed(fallback default build):" + str(e)[:200]
            ex = None
    if ex is None:
        with target:
            ex = relax.build(mod, target=target)
    emit(stage="built", tuned=tuned_note)

    # 5. Cross-link the deployable .so with the SpacemiT clang (glibc rv64gcv board target).
    if cross_cc:
        from tvm.contrib import cc as _cc
        def _fcompile(output, objects, options=None):
            opts = ["--target=riscv64-unknown-linux-gnu", "-march=rv64gcv", "-mabi=lp64d",
                    "-shared", "-fPIC", "-O2"] + (options or [])
            _cc.create_shared(output, objects, options=opts, cc=cross_cc)
        ex.export_library(so_path, fcompile=_fcompile)
    else:
        ex.export_library(so_path)  # host .so (x86) — host correctness only
    emit(stage="exported_library", so=so_path, so_exists=os.path.exists(so_path))

    # 6. Host correctness: run the ONNX module on the x86 TVM VM (llvm host). This proves TVM's
    #    LOWERING/execution is numerically sound (board-independent; the rv64 .so can't run on x86).
    #    Gate vs the torch reference for THIS instance (host_cos), and ALSO report vs the capture
    #    golden (gold_cos) — the latter can differ if the capture was seeded/quantized differently.
    def _cos_rel(a, b):
        a = np.asarray(a, dtype=np.float64).ravel(); b = np.asarray(b, dtype=np.float64).ravel()
        n = min(a.size, b.size); a = a[:n]; b = b[:n]
        d = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
        return (float(np.dot(a, b) / d), float(np.linalg.norm(a - b) / (np.linalg.norm(b) or 1.0)))
    # Feed inputs in the GRAPH's positional order. The npz keys are in0,in1,...,inN; a plain
    # sorted() is LEXICAL (in0,in1,in10,in11,in2,...) which mis-orders >=10-input models (rdt2/xr0)
    # and silently misfeeds the VM. Sort by the trailing integer instead (natural order).
    def _natkey(k):
        i = len(k)
        while i > 0 and k[i - 1].isdigit():
            i -= 1
        return (int(k[i:]) if i < len(k) else 0, k)
    host_cos = host_rel = gold_cos = None
    host_note = ""
    try:
        host_ex = relax.build(from_onnx(_onnx.load(onnx_path), keep_params_in_input=False),
                              target=tvm.target.Target("llvm"))
        dev = tvm.cpu()
        vm = relax.VirtualMachine(host_ex, dev)
        # Feed the loader's OWN inputs (driver_inputs.npz) — they match the compiled model's config;
        # the bundle inputs.npz can differ (see above). Fall back to bundle inputs if absent.
        _dip = os.path.join(work, "driver_inputs.npz")
        npz = np.load(_dip if os.path.exists(_dip) else os.path.join(bundle_root, "inputs.npz"))
        args = [tvm.nd.array(npz[k], dev) for k in sorted(npz.files, key=_natkey)]
        out = vm["main"](*args)
        got = out.numpy() if hasattr(out, "numpy") else np.asarray(out[0].numpy())
        host_cos, host_rel = _cos_rel(got, torch_ref)          # gate: TVM vs torch (this instance)
        gold_cos, _ = _cos_rel(got, np.load(golden_file))       # report: TVM vs capture golden
    except Exception as e:  # noqa: BLE001
        host_note = "host-vm correctness failed: " + str(e)[:300]

    emit(stage="done", ok=os.path.exists(so_path), host_cos=host_cos, host_rel=host_rel,
         gold_cos=gold_cos, host_note=host_note, tuned=tuned_note, quant=quant_note, onnx=onnx_note)

except Exception as e:  # noqa: BLE001
    OUT["error"] = str(e)[:600]
    OUT["traceback"] = traceback.format_exc()[-2000:]
    emit(ok=False)
    sys.exit(1)
'''


@dataclass
class CompileResult:
    ok: bool
    so_path: Path | None
    host_cos: float | None
    host_rel: float | None
    gold_cos: float | None
    stage: str
    note: str
    raw: dict


def compile_model(b: _bundle.CaptureBundle, work: Path, *, cross: bool = True,
                  tune: bool = False, timeout: int = 3600) -> CompileResult:
    """Export+import+compile a bundle in the driver venv; return the .so + host correctness.

    The driver venv is the model-specific capture venv when one exists (it carries the model's
    framework deps, e.g. lerobot / the BitNet transformers fork), else the m2m venv."""
    m2m = driver_python(b.model)
    if m2m is None:
        raise TVMError("no driver venv python found (set MERLIN_MODEL2MLIR / capture venv) — cannot "
                       "drive torch.export + TVM Relax import")
    work.mkdir(parents=True, exist_ok=True)
    driver = work / "tvm_compile_driver.py"
    driver.write_text(_DRIVER_TEMPLATE)
    status = work / "tvm_status.json"
    if status.exists():
        status.unlink()

    cross_cc = ""
    if cross:
        cc = k1.toolchain_cc()
        cross_cc = str(cc) if cc is not None else ""

    libdir = str(tvm_lib_dir())
    ld = os.pathsep.join([libdir, os.environ.get("LD_LIBRARY_PATH", "")]).strip(os.pathsep)
    pypath = os.pathsep.join([str(tvm_python_path()), os.environ.get("PYTHONPATH", "")])
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": pypath,
        "TVM_LIBRARY_PATH": libdir,
        "LD_LIBRARY_PATH": ld,
        "MERLIN_MODEL2MLIR": str(_bundle.model2mlir_root()),
        "MERLIN_TVM_MODEL": b.model,
        "MERLIN_TVM_VARIANT": b.variant,
        "MERLIN_TVM_BUNDLE": str(b.root),
        "MERLIN_TVM_WORK": str(work),
        "MERLIN_TVM_TARGET": json.dumps(TVM_TARGET_CONFIG),
        "MERLIN_TVM_CROSS_CC": cross_cc,
        "MERLIN_TVM_STATUS": str(status),
        "MERLIN_TVM_GOLDEN": str(golden_path(b)),
        "MERLIN_TVM_TUNE": "1" if tune else "0",
    })
    # For a full-fidelity ``_full`` recapture, load the REAL/native architecture (bundle.full_env),
    # not the TOML truncation defaults — otherwise we export a truncated model but gate vs the full
    # golden. Detected by the resolved bundle dir suffix.
    _is_full = b.root.name.endswith("_full")
    env.update(_workload_env(b.model, full=_is_full))

    proc = subprocess.run([str(m2m), str(driver)], capture_output=True, text=True,
                          timeout=timeout, env=env)
    raw: dict = {}
    if status.is_file():
        try:
            raw = json.loads(status.read_text())
        except Exception:  # noqa: BLE001
            raw = {}
    if not raw:
        raw = {"stage": "no_status", "error": (proc.stderr or proc.stdout)[-600:]}
    so = work / "model_tvm.so"
    ok = bool(raw.get("ok")) and so.is_file()
    note = " ".join(x for x in (raw.get("onnx", ""), raw.get("tuned", ""), raw.get("quant", ""),
                                raw.get("host_note", ""), raw.get("error", "")) if x)
    return CompileResult(ok=ok, so_path=(so if so.is_file() else None),
                         host_cos=raw.get("host_cos"), host_rel=raw.get("host_rel"),
                         gold_cos=raw.get("gold_cos"),
                         stage=str(raw.get("stage", "?")), note=str(note)[:400], raw=raw)


# --- RVV audit ----------------------------------------------------------------------------------

def _region_of_symbol(sym: str) -> str:
    s = sym.lower()
    if "matmul" in s or "gemm" in s or "dense" in s or "linear" in s or "contract" in s:
        return "gemm"
    if "softmax" in s or "attention" in s or "attn" in s:
        return "attention"
    if "norm" in s or "rsqrt" in s or "rms" in s:
        return "norm"
    if any(t in s for t in ("add", "mul", "gelu", "silu", "relu", "exp", "divide", "sub")):
        return "elementwise"
    return "other"


# TVM names its fused compute kernels tvmgen_default_fused_*. libc / CRT / the TVM runtime shims are
# not model kernels — ignore them when listing scalar fallbacks.
_AUDIT_IGNORE = ("_start", "abort", "frame_dummy", "register_tm", "__do_global",
                 "printf", "memcpy", "memset", "malloc", "free", "__tvm", "TVM",
                 "call_packed", "deregister", "_init", "_fini", "plt")


def audit_so(so: Path) -> tuple[float | None, list[ScalarFallback], dict]:
    """RVV-audit the emitted TVM ``.so``. Returns (coverage_overall, fallbacks, per-symbol dict)."""
    report = rvv_audit.audit_binary(so)
    fallbacks = [
        ScalarFallback(symbol=sym, reason="TVM emitted scalar (no RVV in kernel)",
                       region=_region_of_symbol(sym))
        for sym in report.scalar_fallback_symbols(ignore=_AUDIT_IGNORE)
    ]
    by_symbol = {n: {"vector": sc.vector, "scalar_compute": sc.scalar_compute,
                     "coverage": sc.coverage} for n, sc in report.by_symbol.items()}
    return report.coverage_overall, fallbacks, by_symbol


# --- the runner ---------------------------------------------------------------------------------

# The full model2MLIR corpus. int8 is attempted first (the coordinator's priority + the K1's native
# datapath); fp32 is the fallback / where int8 isn't captured.
ALL_MODELS = ("tiny_llama", "small_llama", "bitvla", "openvla", "rdt2", "rdt",
              "molmoact", "groot_n1d7", "xr0", "pi05", "smolvla")
DEFAULT_MODELS = ALL_MODELS


def run_model(model: str, variant: str = "int8", *, work_root: Path | None = None,
              write: bool = True, run_board: bool | None = None, cross: bool = True,
              tune: bool | None = None) -> BaselineResult:
    """Run one (model, variant) through the TVM arm end-to-end and return a BaselineResult.

    Re-runnable: with the board down it produces a ``not_run`` result that still carries the built
    rv64gcv ``.so``, RVV coverage, scalar-fallback table, and host-VM correctness.
    """
    if tune is None:
        tune = os.environ.get("MERLIN_TVM_TUNE", "0") == "1"
    cos_thr, rel_thr = _bundle.tolerance(model)
    res = BaselineResult(framework=FRAMEWORK, model=model, variant=variant,
                         substrate="k1_spacemit", cos_threshold=cos_thr, rel_threshold=rel_thr,
                         march=k1.K1_MARCH, toolchain="tvm-v0.19.0(llvm18)+spacemit-clang",
                         framework_commit=tvm_commit(), timestamp=artifacts.utc_stamp(),
                         notes=("tuning=" + ("metaschedule" if tune else "default-relax-build")))

    b = resolve_bundle(model, variant)
    gold = golden_path(b)
    if not gold.is_file():
        res.gap_reason = f"golden missing: neither golden_w8a8.npy nor golden.npy under {b.root}"
        return _finish(res, model, variant, write)
    if b.torch_loader is not None and not b.torch_loader.is_file():
        res.gap_reason = f"torch loader missing: {b.torch_loader} absent (TVM imports from torch)"
        return _finish(res, model, variant, write)
    if not tvm_available():
        why = ("TVM shared lib not built under build/baselines/tvm (run cmake+ninja)"
               if not tvm_built() else "model2MLIR venv (torch) not found")
        res.gap_reason = f"TVM arm unavailable: {why}"
        return _finish(res, model, variant, write)

    work = (work_root or (_BUILD_ROOT / "runs")) / f"{model}_{variant}"
    work.mkdir(parents=True, exist_ok=True)

    # 1. export -> Relax import -> (tune) -> rv64gcv compile -> cross-linked .so (the RVV artifact).
    try:
        cr = compile_model(b, work, cross=cross, tune=tune)
    except (TVMError, subprocess.TimeoutExpired) as e:
        res.gap_reason = f"TVM compile driver failed: {str(e)[:300]}"
        return _finish(res, model, variant, write)

    if not cr.ok or cr.so_path is None:
        res.gap_reason = f"TVM export/import/build failed at stage={cr.stage}: {cr.note[:250]}"
        if cr.host_cos is not None:
            res.notes += f" host_cos={cr.host_cos:.5f}"
        return _finish(res, model, variant, write)

    res.built = True
    res.notes += f" so={cr.so_path}"
    if cr.note:
        res.notes += f" driver:{cr.note[:200]}"

    # 2. RVV audit — mechanical honesty on the emitted .so.
    try:
        cov, fallbacks, _by = audit_so(cr.so_path)
        res.rvv_coverage_overall = cov
        res.scalar_fallbacks = fallbacks
    except Exception as e:  # noqa: BLE001
        res.notes += f" rvv-audit failed: {str(e)[:150]}"

    # 3. host-VM correctness — board-independent lowering gate. host_cos = TVM-vs-torch (this
    #    instance, the true reference for what we compiled); gold_cos = TVM-vs-capture-golden
    #    (reported for continuity, may differ if the capture was seeded/quantized differently).
    if cr.host_cos is not None:
        res.cos, res.rel = cr.host_cos, cr.host_rel
        res.notes += " (cos/rel=TVM-vs-torch host VM; on-board RVV cos pending board)"
        if cr.gold_cos is not None:
            res.notes += f" gold_cos={cr.gold_cos:.4f}"

    # 4. K1 on-board run — the ONLY board-gated step (fail-closed). Serialized on the single shared
    #    board via board_lock() (the RPC-run driver deploys the runtime + runs the relax VM inside).
    do_board = k1_exec.board_available() if run_board is None else run_board
    if do_board:
        try:
            with k1_exec.board_lock():
                _run_on_board(res, cr.so_path, b, work)
        except k1_exec.BoardUnavailable as e:
            res.gap_reason = res.gap_reason or f"K1 board run failed: {str(e)[:200]}"
        except Exception as e:  # noqa: BLE001
            res.gap_reason = res.gap_reason or f"K1 board run error: {str(e)[:200]}"
    else:
        # Distinguish "the caller did not ask for a board run" from "the board is unreachable" — both
        # are not_run, but blaming an unset MERLIN_K1_HOST when the board is up and run_board=False was
        # explicit writes a FALSE reason into the record. The gap string is the product here.
        why = ("board run not requested (run_board=False)" if run_board is False
               else "K1 board unavailable (MERLIN_K1_HOST unset / unreachable)")
        res.gap_reason = (f"{why} — RVV .so built and audited, "
                          "host-VM correctness recorded; on-silicon timing pending")

    res.board_vlenb = k1_exec.board_vlenb()
    return _finish(res, model, variant, write)


# riscv64 TVM runtime + tvm_rpc, cross-built with SpacemiT clang (rv64gcv). See tvm-rv64/ build.
_RV64_RUNTIME_DIR = repo_root() / "build" / "baselines" / "tvm-rv64"
_BOARD_RPC_DIR = "/root/tvm_rpc"
# The board /root rootfs is small (~14 G, often full from other agents' weights); the RPC work dir
# MUST be on tmpfs (/tmp, ~1.9 G free) or uploads fail with ENOSPC.
_BOARD_RPC_WORKDIR = "/tmp/tvm_work"
# Our own port range (9193-9199), distinct from the default 9090/9091 the other agents' servers use.
_BOARD_RPC_PORT = int(os.environ.get("MERLIN_TVM_RPC_PORT", "9193"))


def rv64_runtime_built() -> bool:
    """True iff the riscv64 TVM runtime + tvm_rpc are cross-built (the on-board execution deps)."""
    return ((_RV64_RUNTIME_DIR / "libtvm_runtime.so").is_file()
            and (_RV64_RUNTIME_DIR / "tvm_rpc").is_file())


# m2m-venv driver: deploy the riscv64 runtime, start a persistent tvm_rpc server on the board,
# connect host->board directly, upload the .so, run the relax VM, time (wall) + gate cos/rel.
# Hard-won fixes baked in (see AGENT.md): the C++ tvm_rpc parses ONLY ``--opt=value`` (space form is
# silently ignored -> defaults); it EXITS on stdin EOF, so the server is launched from the HOST with
# an ssh whose stdin is a never-closing FIFO (a plain nohup/setsid over ssh dies); the work dir is
# on tmpfs (board /root is often full); and the relax VM over RPC needs set_input/invoke_stateful/
# get_outputs (the direct ``vm["main"](*args)`` closure call mis-marshals remote NDArrays). Fail-closed.
_RPC_RUN_TEMPLATE = r'''
import json, os, subprocess, sys, time
import numpy as np

OUT = {"ok": False}
def emit(**kw):
    OUT.update(kw)
    with open(os.environ["MERLIN_RPC_STATUS"], "w") as f:
        json.dump(OUT, f, default=str)

fifo = None
holder = None
sshsrv = None
try:
    so = os.environ["MERLIN_RPC_SO"]
    host = os.environ["MERLIN_RPC_HOST"]; port = int(os.environ["MERLIN_RPC_PORT"])
    port_end = port + 6
    rtdir = os.environ["MERLIN_RPC_RTDIR"]; bdir = os.environ["MERLIN_RPC_BOARDDIR"]
    wdir = os.environ["MERLIN_RPC_WORKDIR"]
    key = os.environ["MERLIN_RPC_SSH_KEY"]; sshhost = os.environ["MERLIN_RPC_SSH_HOST"]
    ssh = ["ssh", "-i", key, "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
           "-o", "ConnectTimeout=10", sshhost]
    scp = ["scp", "-i", key, "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no"]
    # deploy runtime + server binary (idempotent) + tmpfs work dir
    subprocess.run(ssh + ["mkdir -p %s %s" % (bdir, wdir)], capture_output=True, timeout=60)
    for fn in ("libtvm_runtime.so", "tvm_rpc"):
        subprocess.run(scp + [os.path.join(rtdir, fn), sshhost + ":" + bdir + "/" + fn],
                       capture_output=True, timeout=300)
    subprocess.run(ssh + ["chmod +x %s/tvm_rpc" % bdir], capture_output=True, timeout=30)
    # Start the server held open by a never-closing FIFO on this host feeding the ssh stdin, so the
    # remote server never sees EOF and stays up (foreground under ssh; ssh backgrounded here).
    fifo = os.path.join(os.environ["MERLIN_RPC_TMP"], "rpc_fifo_%d" % port)
    try:
        os.remove(fifo)
    except OSError:
        pass
    os.mkfifo(fifo)
    holder = subprocess.Popen(["bash", "-c", "sleep 7200 > %s" % fifo])
    srv_cmd = ("cd %s && LD_LIBRARY_PATH=%s exec ./tvm_rpc server --host=0.0.0.0 --port=%d "
               "--port-end=%d --work-dir=%s" % (bdir, bdir, port, port_end, wdir))
    fin = open(fifo, "rb", buffering=0)
    sshsrv = subprocess.Popen(ssh + [srv_cmd], stdin=fin,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(6)
    import tvm
    from tvm import relax, rpc
    sess = rpc.connect(host, port, session_timeout=120)
    dev = sess.cpu()
    sess.upload(so)
    rmod = sess.load_module(os.path.basename(so))
    vm = relax.VirtualMachine(rmod, dev)
    npz = np.load(os.environ["MERLIN_RPC_INPUTS"])
    def _natkey(k):
        i = len(k)
        while i > 0 and k[i - 1].isdigit():
            i -= 1
        return (int(k[i:]) if i < len(k) else 0, k)
    # natural (not lexical) order so >=10-input models feed in graph-positional order (see driver).
    args = [tvm.nd.array(npz[k], dev) for k in sorted(npz.files, key=_natkey)]
    # relax VM over RPC: set_input/invoke_stateful/get_outputs (marshals remote NDArrays correctly).
    vm.set_input("main", *args)
    vm.invoke_stateful("main")
    out = vm.get_outputs("main")
    got = out.numpy() if hasattr(out, "numpy") else np.asarray(out[0].numpy())
    n = 10; t0 = time.time()
    for _ in range(n):
        vm.set_input("main", *args); vm.invoke_stateful("main"); vm.get_outputs("main")
    wall_ns = int((time.time() - t0) / n * 1e9)
    cos = rel = None
    ref = os.environ.get("MERLIN_RPC_REF", "")
    if ref and os.path.exists(ref):
        g = np.load(ref).astype(np.float64).ravel()
        a = np.asarray(got, dtype=np.float64).ravel()[:g.size]; gg = g[:a.size]
        d = (np.linalg.norm(a) * np.linalg.norm(gg)) or 1.0
        cos = float(np.dot(a, gg) / d); rel = float(np.linalg.norm(a - gg) / (np.linalg.norm(gg) or 1.0))
    emit(ok=True, wall_ns=wall_ns, cos=cos, rel=rel)
except Exception as e:
    import traceback
    emit(ok=False, error=str(e)[:300], tb=traceback.format_exc()[-800:])
    sys.exit(1)
finally:
    # tear down the server + FIFO holder (be a good board citizen: release between phases)
    for pr in (sshsrv, holder):
        try:
            pr and pr.terminate()
        except Exception:
            pass
    try:
        subprocess.run(["ssh", "-i", os.environ["MERLIN_RPC_SSH_KEY"], "-o", "BatchMode=yes",
                        "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                        os.environ["MERLIN_RPC_SSH_HOST"],
                        "pkill -f 'tvm_rpc server --host=0.0.0.0 --port=%d' 2>/dev/null || true"
                        % int(os.environ["MERLIN_RPC_PORT"])], capture_output=True, timeout=30)
    except Exception:
        pass
    try:
        fifo and os.path.exists(fifo) and os.remove(fifo)
    except Exception:
        pass
'''


def board_runner_bin() -> Path:
    """The cross-built board-LOCAL relax-VM runner (build via tvm_board/build_board_runner.sh)."""
    return _RV64_RUNTIME_DIR / "board_runner" / "board_runner"


_BOARD_LOCAL_DIR = "/root/tvm_local"  # board SD (~8G free) — big const-folded .so lives here


def _dl_dtype(dt) -> tuple[int, int]:
    """numpy dtype -> DLDataType (code, bits). TVM bool == UInt(1)."""
    import numpy as _np
    dt = _np.dtype(dt)
    table = {_np.float32: (2, 32), _np.float64: (2, 64), _np.int64: (0, 64),
             _np.int32: (0, 32), _np.int8: (0, 8), _np.uint8: (1, 8), _np.bool_: (1, 1)}
    if dt.type not in table:
        raise ValueError(f"unmapped input dtype {dt}")
    return table[dt.type]


def _run_board_local(res: BaselineResult, so: Path, b: _bundle.CaptureBundle, work: Path,
                     *, iters: int = 10, timeout: int = 1800) -> bool:
    """Run the module BOARD-LOCALLY (no tvm_rpc): scp the .so + cross-built runner + inputs to the
    board SD, run the relax VM locally over plain ssh, pull the output, gate cos/rel vs torch_ref.

    This bypasses the ``tvm_rpc`` ``kShutdown`` that kills large-``.so`` RPC sessions (the RPC layer
    is the flaky part, not board RAM). Returns True iff it produced a gated on-board result.
    """
    import numpy as np
    runner = board_runner_bin()
    rt = _RV64_RUNTIME_DIR / "libtvm_runtime.so"
    if not runner.is_file() or not rt.is_file():
        res.gap_reason = ("board-local runner/libtvm_runtime.so not cross-built "
                          "(tvm_board/build_board_runner.sh); on-board pending")
        return False
    ref = work / "torch_ref.npy"
    if not ref.is_file():
        res.gap_reason = "torch_ref.npy missing (needed to gate the on-board output)"
        return False
    host = k1.K1_HOST
    ssh = ["ssh", "-i", k1.K1_SSH_KEY, "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
           "-o", "ConnectTimeout=10", host]
    scp = ["scp", "-i", k1.K1_SSH_KEY, "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no"]
    bdir = _BOARD_LOCAL_DIR

    # prep inputs (graph-positional / natural order) as raw bins + a manifest the runner parses.
    # Prefer the loader's OWN inputs (driver_inputs.npz, saved at compile) — they match the compiled
    # model's config; the bundle inputs.npz can differ (e.g. openvla 64x64 vs 224x224).
    _dip = work / "driver_inputs.npz"
    npz = np.load(_dip if _dip.is_file() else b.inputs)
    keys = sorted(npz.files, key=lambda k: (int("".join(filter(str.isdigit, k)) or 0), k))
    indir = work / "board_inputs"
    indir.mkdir(parents=True, exist_ok=True)
    manifest = work / "board_manifest.txt"
    with open(manifest, "w") as mf:
        for i, k in enumerate(keys):
            a = np.ascontiguousarray(npz[k])
            code, bits = _dl_dtype(a.dtype)
            a.tofile(indir / f"in{i}.bin")
            mf.write(f"{code} {bits} {a.ndim} " + " ".join(str(x) for x in a.shape) + f" in{i}.bin\n")

    try:
        subprocess.run(ssh + [f"mkdir -p {bdir}/inputs"], capture_output=True, timeout=60)
        for src, dst in [(rt, "libtvm_runtime.so"), (runner, "board_runner"),
                         (so, "model.so"), (manifest, "manifest.txt")]:
            r = subprocess.run(scp + [str(src), f"{host}:{bdir}/{dst}"], capture_output=True,
                               timeout=timeout, text=True)
            if r.returncode:
                res.gap_reason = f"board scp {dst} failed: {r.stderr[:150]}"
                return False
        for i in range(len(keys)):
            subprocess.run(scp + [str(indir / f"in{i}.bin"), f"{host}:{bdir}/inputs/in{i}.bin"],
                           capture_output=True, timeout=600)
        subprocess.run(ssh + [f"chmod +x {bdir}/board_runner"], capture_output=True, timeout=30)
        cmd = (f"cd {bdir} && LD_LIBRARY_PATH={bdir} ./board_runner model.so manifest.txt inputs "
               f"out.bin {iters}")
        rr = subprocess.run(ssh + [cmd], capture_output=True, timeout=timeout, text=True)
        stderr = rr.stderr or ""
        if rr.returncode != 0:
            res.ran = False
            res.gap_reason = ("board-LOCAL run failed (rc=%d): %s"
                              % (rr.returncode, stderr.strip().splitlines()[-1][:200] if stderr.strip() else "?"))
            return True  # a real, characterized on-board attempt (honest not_run reason recorded)
        # pull output + meta
        subprocess.run(scp + [f"{host}:{bdir}/out.bin", str(work / "board_out.bin")],
                       capture_output=True, timeout=120)
        subprocess.run(scp + [f"{host}:{bdir}/out.bin.meta", str(work / "board_out.meta")],
                       capture_output=True, timeout=60)
        meta = (work / "board_out.meta").read_text().split()
        code, bits, ndim = int(meta[0]), int(meta[1]), int(meta[2])
        shape = [int(x) for x in meta[3:3 + ndim]]
        npdt = {(2, 32): np.float32, (2, 64): np.float64, (0, 64): np.int64,
                (0, 32): np.int32}[(code, bits)]
        out = np.fromfile(work / "board_out.bin", dtype=npdt).reshape(shape)
        e2e_ns = None
        for line in stderr.splitlines():
            if line.startswith("E2E_NS"):
                e2e_ns = int(float(line.split()[1]))
        tref = np.load(ref).astype(np.float64).ravel()
        a = out.astype(np.float64).ravel()[:tref.size]
        gg = tref[:a.size]
        d = (np.linalg.norm(a) * np.linalg.norm(gg)) or 1.0
        res.ran = True
        res.cos = float(np.dot(a, gg) / d)
        res.rel = float(np.linalg.norm(a - gg) / (np.linalg.norm(gg) or 1.0))
        res.e2e_wall_ns = e2e_ns
        if e2e_ns:
            res.regions = [RegionProfile(name="other", wall_ns=e2e_ns,
                                         rvv_coverage=res.rvv_coverage_overall,
                                         note="whole-model relax VM (board-LOCAL, no tvm_rpc, wall)")]
        return True
    except subprocess.TimeoutExpired:
        res.ran = False
        res.gap_reason = f"board-LOCAL run timed out after {timeout}s (large .so load/exec)"
        return True
    finally:
        subprocess.run(ssh + [f"rm -rf {bdir}/model.so {bdir}/inputs {bdir}/out.bin* 2>/dev/null"],
                       capture_output=True, timeout=30)


def _run_on_board(res: BaselineResult, so: Path, b: _bundle.CaptureBundle, work: Path) -> None:
    """Deploy + run the TVM module on the K1 over direct RPC, under the board lock.

    The riscv64 TVM runtime + ``tvm_rpc`` ARE cross-built with the SpacemiT clang for ``rv64gcv``
    (``rv64_runtime_built()``; ``build/baselines/tvm-rv64``) and the RPC server runs on real K1
    silicon (host->board direct connect confirmed). The actual relax-VM-over-RPC *execution* runs in
    an m2m-venv subprocess (the main ``.venv`` has no ``tvm``) via :func:`_rpc_run_driver`.

    Two hard board/network constraints are recorded honestly rather than worked around:
      * the board's firewall blocks the **board->host** tracker callback, so the standard
        tracker-based MetaSchedule ``RPCRunner`` (on-device autotuning) is unavailable on this net;
      * the C++ ``tvm_rpc`` direct-server socket lifecycle is unstable across sessions.
    So this is fail-closed: a specific ``gap_reason`` on any RPC issue, never a fabricated timing.
    A weights-too-big model is a labeled fit gap.
    """
    # Fit check: what actually loads on-board is the EXPORTED ``.so`` (const-folded params baked in),
    # NOT the capture safetensors — a truncated/reduced capture can export a tiny .so even when the
    # original checkpoint is many GB (e.g. openvla: 4.2G safetensors -> 22M .so). Gate on the .so
    # size (+ headroom for the VM/intermediates) so a small .so is never wrongly declared a fit gap.
    so_gb = so.stat().st_size / 1e9 if so.is_file() else 0.0
    if so_gb > 3.0:
        res.ran = False
        res.gap_reason = (f"exported .so {so_gb:.1f}G exceeds K1 usable RAM (~3.4G) — on-board run is "
                          f"a fit gap (RVV .so built + audited + host-VM correctness recorded)")
        return
    if not rv64_runtime_built():
        res.ran = False
        res.gap_reason = ("riscv64 TVM runtime/tvm_rpc not cross-built under build/baselines/tvm-rv64 "
                          "(RVV .so built + audited + host-VM correctness recorded; on-silicon run "
                          "pending the runtime cross-build)")
        return
    # PREFER the board-LOCAL runner: it loads the module + runs the relax VM on the board directly,
    # bypassing tvm_rpc (whose session dies with ``kShutdown`` on large-``.so`` uploads — the RPC
    # layer is the flaky part, not board RAM). Falls back to the RPC path only if the runner binary
    # is not cross-built.
    if board_runner_bin().is_file():
        if _run_board_local(res, so, b, work):
            return
    # The RPC execution driver runs in the m2m venv (has tvm). It manages the board server under the
    # board_lock and connects host->board directly. On this network the direct-RPC relax-VM path is
    # unstable (see docstring); the driver returns a status we surface verbatim.
    ok, info = _rpc_run_driver(so, b, work)
    if ok:
        res.ran = True
        res.e2e_wall_ns = info.get("wall_ns")
        res.cos = info.get("cos")
        res.rel = info.get("rel")
        if res.e2e_wall_ns:
            res.regions = [RegionProfile(name="other", wall_ns=res.e2e_wall_ns,
                                         rvv_coverage=res.rvv_coverage_overall,
                                         note="whole-model relax VM (on-board, direct RPC, wall)")]
    else:
        res.ran = False
        res.gap_reason = res.gap_reason or (
            "on-board RPC run not completed: " + str(info.get("error", ""))[:220])


def _rpc_run_driver(so: Path, b: _bundle.CaptureBundle, work: Path, *, timeout: int = 600) -> tuple[bool, dict]:
    """Run the TVM ``.so`` on the K1 via direct RPC, in an m2m-venv subprocess (has ``tvm``).

    Manages the board ``tvm_rpc`` server, connects host->board, runs the relax VM, times it (wall),
    gates cos/rel vs the torch reference. Returns ``(ok, info)`` — fail-closed, never fabricates.
    Wrapped by the caller in ``board_lock()`` at the runner level for on-board serialization.
    """
    m2m = m2m_python()
    if m2m is None:
        return False, {"error": "m2m venv (tvm) not found"}
    host = k1.K1_HOST.split("@")[-1] if "@" in k1.K1_HOST else k1.K1_HOST
    driver = work / "tvm_rpc_run.py"
    driver.write_text(_RPC_RUN_TEMPLATE)
    status = work / "rpc_status.json"
    if status.exists():
        status.unlink()
    libdir = str(tvm_lib_dir())
    # The default work dir is tmpfs (/tmp, ~1.9 G) — fast but small. A big const-folded ``.so``
    # (diffusion graphs bake ~GB of code) won't fit there and the scp fails with ENOSPC. When the
    # ``.so`` is larger than a safe fraction of the tmpfs budget, upload to the SD rootfs instead
    # (~5 G free), which fits; the VM-load may then be slow/OOM on the 3.4 G board, but that is a
    # separate honest gap surfaced by the RPC timeout (never a fabricated pass).
    so_bytes = so.stat().st_size if so.is_file() else 0
    board_workdir = "/root/tvm_work" if so_bytes > 1_500_000_000 else _BOARD_RPC_WORKDIR
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": os.pathsep.join([str(tvm_python_path()), os.environ.get("PYTHONPATH", "")]),
        "TVM_LIBRARY_PATH": libdir,
        "LD_LIBRARY_PATH": os.pathsep.join([libdir, os.environ.get("LD_LIBRARY_PATH", "")]).strip(os.pathsep),
        "MERLIN_RPC_SO": str(so),
        "MERLIN_RPC_HOST": host,
        "MERLIN_RPC_PORT": str(_BOARD_RPC_PORT),
        "MERLIN_RPC_INPUTS": str(b.inputs),
        "MERLIN_RPC_REF": str(work / "torch_ref.npy"),
        "MERLIN_RPC_RTDIR": str(_RV64_RUNTIME_DIR),
        "MERLIN_RPC_BOARDDIR": _BOARD_RPC_DIR,
        "MERLIN_RPC_WORKDIR": board_workdir,
        "MERLIN_RPC_TMP": str(work),
        "MERLIN_RPC_SSH_KEY": k1.K1_SSH_KEY,
        "MERLIN_RPC_SSH_HOST": k1.K1_HOST,
        "MERLIN_RPC_STATUS": str(status),
    })
    timed_out = False
    try:
        subprocess.run([str(m2m), str(driver)], capture_output=True, text=True,
                       timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        timed_out = True  # the run itself may have SUCCEEDED (status written) before a slow teardown
    if not status.is_file():
        return False, {"error": "rpc run timed out" if timed_out else "rpc driver wrote no status"}
    try:
        info = json.loads(status.read_text())
    except Exception:  # noqa: BLE001
        return False, {"error": "rpc status unparseable"}
    # A written status with ok=True is authoritative even if the parent timed out during teardown.
    return bool(info.get("ok")), info


def _finish(res: BaselineResult, model: str, variant: str, write: bool) -> BaselineResult:
    res.validate()
    if write:
        m = artifacts.new_measurement("k1_spacemit", model, "cross_framework")
        res.write(m.path)
    return res


def run_all(models=DEFAULT_MODELS, variant: str = "int8", *, write: bool = True,
            tune: bool | None = None) -> list[BaselineResult]:
    """Run the TVM arm over the model set (int8 by default)."""
    out = []
    for m in models:
        try:
            out.append(run_model(m, variant, write=write, tune=tune))
        except Exception as e:  # noqa: BLE001 - one model must never sink the batch
            r = BaselineResult(framework=FRAMEWORK, model=m, variant=variant,
                               gap_reason=f"runner exception: {str(e)[:200]}",
                               timestamp=artifacts.utc_stamp())
            if write:
                try:
                    md = artifacts.new_measurement("k1_spacemit", m, "cross_framework")
                    r.write(md.path)
                except Exception:  # noqa: BLE001
                    pass
            out.append(r)
    return out


def _main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="TVM (Apache TVM v0.19.0 / Relax) K1-RVV baseline arm")
    ap.add_argument("models", nargs="*", default=list(DEFAULT_MODELS),
                    help="models to run (default: the full corpus)")
    ap.add_argument("--variant", default="int8", help="int8 (default) | fp32 | fp8")
    ap.add_argument("--tune", action="store_true", help="MetaSchedule autotune (needs K1 RPC)")
    ap.add_argument("--no-write", action="store_true", help="do not write BaselineResult artifacts")
    ap.add_argument("--no-cross", action="store_true",
                    help="build a host (x86) .so instead of cross-linking rv64gcv (debug only)")
    args = ap.parse_args(argv)
    out = []
    for m in tuple(args.models):
        out.append(run_model(m, args.variant, write=not args.no_write, cross=not args.no_cross,
                             tune=args.tune))
    for r in out:
        cov = f"{100*r.rvv_coverage_overall:.0f}%RVV" if r.rvv_coverage_overall is not None else "?RVV"
        print(f"{r.model}/{r.variant}: {r.status():10s} {cov} "
              f"fallbacks={len(r.scalar_fallbacks)} cos={r.cos} {r.gap_reason}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
