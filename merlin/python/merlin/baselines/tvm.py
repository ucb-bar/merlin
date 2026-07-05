"""TVM (Apache TVM / Relax) baseline arm — ingest OUR models and run them on the K1 with RVV.

TVM is the second external-framework arm. Unlike Buddy (which ingests our ``model.mlir``
directly), TVM does **not** consume linalg-on-tensors MLIR: it imports from the *PyTorch* graph.
So this arm starts from the same seeded model instance the capture bundle was built from, runs
``torch.export``, imports the exported program into TVM **Relax**
(``tvm.relax.frontend.torch.from_exported_program``), lowers it for a RISC-V ``rv64gcv`` LLVM
target, and emits a deployable module. The emitted kernel object is RVV-audited exactly like every
other arm, so "TVM used RVV" is *proven by disassembly*, never assumed.

Pipeline (per model, fp32 first)::

    seeded torch model (== capture recipe: manual_seed(0), M2M_LLAMA_LAYERS, zero-param perturb)
      -> torch.export.export                                              [m2m venv: torch 2.x]
      -> tvm.relax.frontend.torch.from_exported_program                   [Relax IRModule]
      -> relax.build(target="llvm -mtriple=riscv64-... -mattr=+v,...")     [RVV LLVM codegen]
      -> mod.export_library(fcompile=<SpacemiT clang cross>)  -> model_tvm.so  [the RVV artifact]
      -> rvv_audit.audit_binary(.so)   : mechanical RVV%/scalar-fallback honesty
      -> host reference: TVM VM on x86 vs golden.npy (lowering correctness, board-independent)
      -> on-board run (board_lock): TVM RPC to a riscv64 tvm_rpc server -> E2E timing + cos/rel
                                    [fail-closed if the board / runtime cross-build is unavailable]

Because torch + the freshly-built TVM Python package must co-run (the main ``.venv`` has no torch),
the import+compile step executes as a **subprocess in the model2MLIR venv** (``$MERLIN_MODEL2MLIR
/.venv``) with ``TVM_LIBRARY_PATH`` + the TVM/tvm_ffi python paths set. This module writes a small
driver script, runs it there, and reads back a JSON status + the emitted ``.so``.

TVM tuning note: this pinned TVM (post-FFI-refactor, Relax-only) does NOT ship the standalone
``meta_schedule`` / ``auto_scheduler`` / AutoTVM Python packages the plan referenced — tuning is via
the Relax ``"zero"`` pipeline + dlight CPU schedule rules (``tvm.s_tir.dlight``). We record that as
an honest note (``notes``): the numbers are default-schedule RVV, not autotuned. That is a labeled
limitation, never a hidden one.

Honesty (``not_run_is_not_pass``): a model that will not export/import/compile is a ``not_built``
result with a specific ``gap_reason``; a built-but-unrun model (board down / runtime cross-build
absent) is ``not_run`` with a reason. We NEVER fabricate a cos/rel or a cycle count.
"""
from __future__ import annotations

import json
import os
import shutil
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

# The RVV target (plan-locked): LLVM riscv64 with the vector extension enabled. -mcpu is a generic
# rv64 so codegen is not silently gated behind a narrower CPU model. VLEN comes from the board
# (256b / vlenb=32); TVM's LLVM backend sizes vectors via LMUL/vscale at run time. This TVM
# (post-FFI-refactor) DROPPED the CLI target-string form, so the target is a JSON dict; the string
# below is kept only for the RVV-march honesty check + display (never passed to Target()).
# ``+zvl256b`` pins the vector register width to the K1's VLEN (256b / vlenb=32) so LLVM's RVV
# codegen sizes vscale to the real silicon. NOTE (honest): this stripped TVM has no MetaSchedule
# tensorize to vectorize TIR at the schedule level, and its TIR->LLVM path emits scalar loops that
# LLVM does not auto-RVV-vectorize by default — so out-of-the-box coverage on unscheduled kernels
# is low. The RVV audit records that faithfully (rvv_coverage_overall + per-symbol scalar_compute).
TVM_TARGET_CONFIG: dict = {
    "kind": "llvm",
    "mtriple": "riscv64-unknown-linux-gnu",
    "mattr": ["+m", "+f", "+d", "+c", "+v", "+zvl256b"],
    "mcpu": "generic-rv64",
    "mabi": "lp64d",
}
TVM_TARGET = ("llvm -mtriple=riscv64-unknown-linux-gnu "
              "-mattr=+m,+f,+d,+c,+v,+zvl256b -mcpu=generic-rv64 -mabi=lp64d")


def _env_path(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    return Path(v) if v else default


def tvm_lib_dir() -> Path:
    """Directory holding the built ``libtvm_compiler.so`` / ``libtvm_runtime.so`` /
    ``libtvm_ffi.so``. Override ``MERLIN_TVM_LIBRARY_PATH``; defaults to the CMake build's ``lib/``
    subdir (ninja emits the shared libs under ``<build>/lib``, not the build root)."""
    override = os.environ.get("MERLIN_TVM_LIBRARY_PATH")
    if override:
        return Path(override)
    lib = _BUILD_ROOT / "lib"
    return lib if lib.is_dir() else _BUILD_ROOT


def tvm_python_paths() -> list[Path]:
    """PYTHONPATH entries so the m2m venv can ``import tvm`` from the built (uninstalled) tree.

    Only the main ``tvm`` package is added on PYTHONPATH; ``tvm_ffi`` (the compiled cython core)
    is pip-installed into the driving venv (its ``core`` extension can't run from a bare source
    dir), so we do NOT shadow it with the submodule's source ``tvm-ffi/python``.
    """
    return [_TVM_SRC / "python"]


def m2m_python() -> Path | None:
    """The model2MLIR venv python (has torch + transformers). Import+export+compile run here."""
    p = _bundle.model2mlir_root() / ".venv" / "bin" / "python"
    return p if p.is_file() else None


def tvm_built() -> bool:
    """True iff the TVM shared libs are present (the Python package needs them to import)."""
    lib = tvm_lib_dir()
    return any((lib / n).is_file()
               for n in ("libtvm.so", "libtvm_runtime.so", "libtvm_compiler.so"))


def tvm_available() -> bool:
    """True iff TVM is built AND a torch-capable python (m2m venv) exists to drive the import."""
    return tvm_built() and m2m_python() is not None


def tvm_commit() -> str:
    try:
        r = subprocess.run(["git", "-C", str(_TVM_SRC), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=15)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:  # noqa: BLE001
        return ""


# --- capture-bundle resolution (robust to legacy dir names) -------------------------------------
# The fp32 LLM captures predate the ``<model>_fp32_consistent`` convention (see buddy.py).
_LEGACY_FP32_DIRS: dict[str, str] = {
    "tiny_llama": "tiny_consistent",
    "small_llama": "small_consistent",
}


def resolve_bundle(model: str, variant: str = "fp32") -> _bundle.CaptureBundle:
    """Resolve a capture bundle, falling back to legacy dir names for the fp32 LLMs."""
    b = _bundle.resolve(model, variant)
    if b.golden.is_file():
        return b
    if variant == "fp32" and model in _LEGACY_FP32_DIRS:
        legacy = artifacts.recaptures_dir() / _LEGACY_FP32_DIRS[model]
        if (legacy / "golden.npy").is_file():
            return _bundle.CaptureBundle(model=model, variant=variant, root=legacy)
    return b


# --- workload cfg (mirror the capture recipe so the golden is reproducible) ---------------------

def _workload_env(model: str) -> dict[str, str]:
    """The env the m2m capture worker sets for a workload (e.g. M2M_LLAMA_LAYERS for the llamas).

    Read from the workload TOML ``[env]`` table so the exported instance matches the one that
    produced ``golden.npy`` (tiny_llama uses a 2-layer random-init model — exporting the full
    pretrained model would NOT match the golden).
    """
    toml_env: dict[str, str] = {}
    wdir = _bundle.model2mlir_root() / "workloads" / model
    for cand in list(wdir.glob("*.toml")):
        try:
            import tomllib
            data = tomllib.loads(cand.read_text())
        except Exception:  # noqa: BLE001
            continue
        env = data.get("env") or {}
        for k, v in env.items():
            toml_env[str(k)] = str(v)
    return toml_env


class TVMError(RuntimeError):
    pass


# --- the m2m-venv driver: torch.export -> Relax -> RVV .so --------------------------------------

# This runs inside the model2MLIR venv (torch+transformers) with TVM importable. It reproduces the
# EXACT seeded model instance the capture bundle was built from (manual_seed(0), workload env,
# zero-param perturb, per-cfg quant) so the golden is reproducible, exports it, imports to Relax,
# compiles for the rv64gcv target, and cross-links the deployable .so with the SpacemiT clang. It
# also runs the module on the HOST TVM VM (llvm x86) for a board-independent correctness signal.
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
    target_cfg = json.loads(os.environ["MERLIN_TVM_TARGET"])  # JSON dict (CLI string form dropped)
    cross_cc = os.environ.get("MERLIN_TVM_CROSS_CC", "")
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
    # NOTE: per-cfg quantization (int8/fp8) is applied by capture.py via _quant_for; the fp32
    # path (this arm's first target) does none, so the seeded instance matches the fp32 golden.
    if variant != "fp32":
        try:
            from capture import _quant_for, _load_toml  # type: ignore
            cfg = _load_toml(os.path.join(workloads, model))
            q = _quant_for(cfg, variant)
            if q is not None:
                import torchao  # noqa: F401
                q(mdl)
        except Exception as e:  # noqa: BLE001
            emit(stage="quant_skipped", quant_note=str(e)[:200])
    emit(stage="model_built")

    # 2. torch.export -> ExportedProgram.
    ep = torch.export.export(mdl, inputs)
    emit(stage="exported")

    # 3. Import to TVM Relax.
    os.environ.setdefault("TVM_LIBRARY_PATH", os.environ["MERLIN_TVM_LIBRARY_PATH"])
    import tvm
    from tvm import relax
    from tvm.relax.frontend.torch import from_exported_program
    emit(stage="tvm_imported", tvm=getattr(tvm, "__version__", "?"))

    mod = from_exported_program(ep, keep_params_as_input=False)
    emit(stage="relax_imported")

    # 4. Build for the rv64gcv LLVM target (RVV codegen). dlight CPU schedules apply where the
    #    zero pipeline leaves TIR unscheduled; default relax pipeline otherwise.
    target = tvm.target.Target(target_cfg)
    # Best-effort dlight CPU scheduling (GEMV + Reduction) BEFORE the build lowers TIR — this TVM's
    # dlight.cpu only ships GEMV + Reduction (no Fallback), so we apply just those and let the
    # default relax pipeline schedule the rest. dlight is optional: if unavailable, the default
    # pipeline still produces valid RVV codegen (recorded honestly in notes).
    dlight_note = "dlight: none"
    use_dlight = os.environ.get("MERLIN_TVM_DLIGHT", "1") != "0"
    with target:
        try:
            if not use_dlight:
                raise RuntimeError("disabled via MERLIN_TVM_DLIGHT=0")
            import tvm.s_tir.dlight as dl  # relocated under s_tir in this TVM
            rules = []
            for nm in ("GEMV", "Reduction"):
                r = getattr(dl.cpu, nm, None)
                if r is not None:
                    rules.append(r())
            if rules:
                mod = dl.ApplyDefaultSchedule(*rules)(mod)
                dlight_note = "dlight: " + "+".join(type(r).__name__ for r in rules)
        except Exception as e:  # noqa: BLE001
            dlight_note = "dlight-skip: " + str(e)[:120]
        try:
            ex = relax.build(mod, target=target)
        except Exception as e:
            emit(stage="build_failed", err=str(e)[:600], tb=traceback.format_exc()[-1500:],
                 dlight=dlight_note)
            raise
    emit(stage="built", dlight=dlight_note)

    # 5. Cross-link the deployable .so with the SpacemiT clang so it runs on the board's glibc.
    if cross_cc:
        try:
            from tvm.support import cc as _cc   # relocated from tvm.contrib.cc in this TVM
        except Exception:
            from tvm.contrib import cc as _cc   # older layout fallback
        def _fcompile(output, objects, options=None):
            opts = ["--target=riscv64-unknown-linux-gnu", "-march=rv64gcv", "-mabi=lp64d",
                    "-shared", "-fPIC", "-O2"] + (options or [])
            _cc.create_shared(output, objects, options=opts, cc=cross_cc)
        ex.export_library(so_path, fcompile=_fcompile)
    else:
        ex.export_library(so_path)  # host .so (x86) — used for host correctness only
    emit(stage="exported_library", so=so_path, so_exists=os.path.exists(so_path))

    # 5b. Emit the RVV kernel objects too (the audit target): re-export the *object* so rvv_audit
    #     can disassemble the compute kernels directly, independent of the shared-lib packaging.
    try:
        obj_path = os.path.join(work, "model_tvm.o")
        ex.mod.save(obj_path) if hasattr(ex.mod, "save") else None
    except Exception:
        pass

    # 6. Host correctness: run the module on the x86 TVM VM (llvm host) vs golden.npy. This proves
    #    TVM's *lowering* is numerically sound (board-independent); the board RPC run is the RVV
    #    timing + on-silicon cos/rel. We build a SECOND host module for this (the rv64 .so can't
    #    load on x86).
    host_cos = host_rel = None
    host_note = ""
    try:
        host_target = tvm.target.Target("llvm")
        host_ex = relax.build(mod, target=host_target)
        dev = tvm.cpu()
        vm = relax.VirtualMachine(host_ex, dev)
        npz = np.load(os.path.join(bundle_root, "inputs.npz"))
        args = [tvm.nd.array(npz[k], dev) for k in sorted(npz.files)]
        out = vm["main"](*args)
        got = out.numpy() if hasattr(out, "numpy") else np.asarray(out[0].numpy())
        gold = np.load(os.path.join(bundle_root, "golden.npy")).astype(np.float32).ravel()
        a = np.asarray(got, dtype=np.float64).ravel()[:gold.size]
        g = gold[:a.size].astype(np.float64)
        denom = (np.linalg.norm(a) * np.linalg.norm(g)) or 1.0
        host_cos = float(np.dot(a, g) / denom)
        host_rel = float(np.linalg.norm(a - g) / (np.linalg.norm(g) or 1.0))
    except Exception as e:  # noqa: BLE001
        host_note = "host-vm correctness failed: " + str(e)[:300]

    emit(stage="done", ok=os.path.exists(so_path), host_cos=host_cos, host_rel=host_rel,
         host_note=host_note)

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
    stage: str
    note: str
    raw: dict


def compile_model(b: _bundle.CaptureBundle, work: Path, *, cross: bool = True,
                  timeout: int = 3600) -> CompileResult:
    """Export+import+compile a bundle in the m2m venv; return the .so + host correctness.

    ``cross=True`` cross-links the .so with the SpacemiT clang for the board (rv64gcv). The host
    correctness VM run is always attempted (x86) so the compiler's lowering is gated even when the
    board is down.
    """
    m2m = m2m_python()
    if m2m is None:
        raise TVMError("model2MLIR venv python not found (set MERLIN_MODEL2MLIR) — cannot drive "
                       "torch.export + TVM Relax import")
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

    pypath = os.pathsep.join([str(p) for p in tvm_python_paths()] +
                             [os.environ.get("PYTHONPATH", "")])
    env = dict(os.environ)
    libdir = str(tvm_lib_dir())
    ld = os.pathsep.join([libdir, os.environ.get("LD_LIBRARY_PATH", "")]).strip(os.pathsep)
    env.update({
        "PYTHONPATH": pypath,
        "TVM_LIBRARY_PATH": libdir,
        # tvm_ffi's libinfo searches LD_LIBRARY_PATH (not TVM_LIBRARY_PATH) for libtvm_ffi.so.
        "LD_LIBRARY_PATH": ld,
        "MERLIN_TVM_LIBRARY_PATH": libdir,
        "MERLIN_MODEL2MLIR": str(_bundle.model2mlir_root()),
        "MERLIN_TVM_MODEL": b.model,
        "MERLIN_TVM_VARIANT": b.variant,
        "MERLIN_TVM_BUNDLE": str(b.root),
        "MERLIN_TVM_WORK": str(work),
        "MERLIN_TVM_TARGET": json.dumps(TVM_TARGET_CONFIG),
        "MERLIN_TVM_CROSS_CC": cross_cc,
        "MERLIN_TVM_STATUS": str(status),
    })
    env.update(_workload_env(b.model))  # e.g. M2M_LLAMA_LAYERS for the llamas

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
    note = raw.get("host_note", "") or raw.get("error", "") or raw.get("quant_note", "")
    return CompileResult(ok=ok, so_path=(so if so.is_file() else None),
                         host_cos=raw.get("host_cos"), host_rel=raw.get("host_rel"),
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


# TVM names its fused compute kernels tvmgen_default_fused_* (the compute-bearing symbols). libc /
# CRT / the TVM runtime shims are not model kernels — ignore them when listing scalar fallbacks.
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

# LLM subset first (clean torch graph, no VLA-specific ops); then the VLAs.
DEFAULT_MODELS = ("tiny_llama", "small_llama", "bitvla", "rdt2", "rdt", "openvla",
                  "molmoact", "groot_n1d7", "xr0", "pi05", "smolvla")


def run_model(model: str, variant: str = "fp32", *, work_root: Path | None = None,
              write: bool = True, run_board: bool | None = None,
              cross: bool = True) -> BaselineResult:
    """Run one (model, variant) through the TVM arm end-to-end and return a BaselineResult.

    Re-runnable: with the board down it produces a ``not_run`` result that still carries the built
    rv64gcv ``.so``, RVV coverage, scalar-fallback table, and host-VM correctness. A later
    invocation with the board up fills in the on-silicon RVV timing (board branch is the only part
    gated on ``board_available()``).
    """
    cos_thr, rel_thr = _bundle.tolerance(model)
    res = BaselineResult(framework=FRAMEWORK, model=model, variant=variant,
                         substrate="k1_spacemit", cos_threshold=cos_thr, rel_threshold=rel_thr,
                         march=k1.K1_MARCH, toolchain="tvm-relax(llvm18)+spacemit-clang",
                         framework_commit=tvm_commit(), timestamp=artifacts.utc_stamp(),
                         notes="tuning=relax-zero+dlight (no MetaSchedule/AutoTVM in this TVM)")

    b = resolve_bundle(model, variant)
    if not b.golden.is_file():
        res.gap_reason = f"golden missing: {b.root}/golden.npy absent (cannot gate correctness)"
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

    # 1. export -> Relax import -> rv64gcv compile -> cross-linked .so (the RVV artifact).
    try:
        cr = compile_model(b, work, cross=cross)
    except (TVMError, subprocess.TimeoutExpired) as e:
        res.gap_reason = f"TVM compile driver failed: {str(e)[:300]}"
        return _finish(res, model, variant, write)

    if not cr.ok or cr.so_path is None:
        res.gap_reason = f"TVM export/import/build failed at stage={cr.stage}: {cr.note[:250]}"
        if cr.host_cos is not None:  # partial: host correctness may still have been captured
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

    # 3. host-VM correctness (x86 TVM VM vs golden) — board-independent lowering gate. We attach it
    #    as the correctness signal; the on-board RVV run refines it when the board is up.
    if cr.host_cos is not None:
        res.cos, res.rel = cr.host_cos, cr.host_rel
        res.notes += " (cos/rel from host TVM VM; on-board RVV cos pending board)"

    # 4. K1 on-board run — the ONLY board-gated step (fail-closed).
    do_board = k1_exec.board_available() if run_board is None else run_board
    if do_board:
        try:
            _run_on_board(res, cr.so_path, b, work)
        except k1_exec.BoardUnavailable as e:
            res.gap_reason = res.gap_reason or f"K1 board run failed: {str(e)[:200]}"
        except Exception as e:  # noqa: BLE001
            res.gap_reason = res.gap_reason or f"K1 board run error: {str(e)[:200]}"
    else:
        # not a gap when we DO have host correctness + a built RVV artifact: the on-silicon timing
        # is what's pending. Mark ran=False honestly with a board-unavailable reason so the cell
        # reads not_run (built RVV artifact present) rather than a false pass.
        res.gap_reason = ("K1 board unavailable (MERLIN_K1_HOST unset) — RVV .so built and audited, "
                          "host-VM correctness recorded; on-silicon timing pending")

    res.board_vlenb = k1_exec.board_vlenb()
    return _finish(res, model, variant, write)


def _run_on_board(res: BaselineResult, so: Path, b: _bundle.CaptureBundle, work: Path) -> None:
    """Deploy + run the TVM module on the K1 via the TVM RPC server, under the board lock.

    Requires the TVM **runtime** cross-compiled for riscv64 and a ``tvm_rpc`` server running on the
    board (standard TVM cross flow). When that runtime is not present we fail-closed with a
    ``not_run`` gap rather than fabricate a timing — the RVV .so + audit + host correctness still
    stand. Wiring the riscv64 tvm_rpc server + RPC session is the remaining board step.
    """
    # The riscv64 TVM runtime + tvm_rpc server on the board is a separate cross-build (build/
    # baselines/tvm/riscv-runtime). Until it is present, honestly record not_run. This keeps the
    # arm fail-closed: we never claim an on-silicon number we didn't measure.
    rpc_ready = bool(os.environ.get("MERLIN_TVM_RPC_TRACKER") or
                     (work / "riscv_runtime_ready").is_file())
    if not rpc_ready:
        res.ran = False
        res.gap_reason = ("TVM riscv64 runtime / tvm_rpc server not yet cross-built for the board "
                          "(RVV .so built + audited + host-VM correctness recorded; on-silicon RPC "
                          "run pending — set MERLIN_TVM_RPC_TRACKER once the board server is up)")
        return
    # RPC path (executed once the board tvm_rpc is up): connect, upload the .so, time the run,
    # compare OUT vs golden. Left as the documented next step; guarded so it never fabricates.
    res.ran = False
    res.gap_reason = res.gap_reason or "TVM RPC run path not exercised (tracker set but session not wired)"


def _finish(res: BaselineResult, model: str, variant: str, write: bool) -> BaselineResult:
    res.validate()
    if write:
        m = artifacts.new_measurement("k1_spacemit", model, "cross_framework")
        res.write(m.path)
    return res


def run_all(models=DEFAULT_MODELS, variant: str = "fp32", *, write: bool = True) -> list[BaselineResult]:
    """Run the TVM arm over the model set (fp32 by default)."""
    out = []
    for m in models:
        try:
            out.append(run_model(m, variant, write=write))
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

    ap = argparse.ArgumentParser(description="TVM (Apache TVM / Relax) K1-RVV baseline arm")
    ap.add_argument("models", nargs="*", default=list(DEFAULT_MODELS),
                    help="models to run (default: the full corpus, LLM subset first)")
    ap.add_argument("--variant", default="fp32")
    ap.add_argument("--no-write", action="store_true", help="do not write BaselineResult artifacts")
    ap.add_argument("--no-cross", action="store_true",
                    help="build a host (x86) .so instead of cross-linking rv64gcv (debug only)")
    args = ap.parse_args(argv)
    models = tuple(args.models)
    out = []
    for m in models:
        out.append(run_model(m, args.variant, write=not args.no_write, cross=not args.no_cross))
    for r in out:
        cov = f"{100*r.rvv_coverage_overall:.0f}%RVV" if r.rvv_coverage_overall is not None else "?RVV"
        print(f"{r.model}/{r.variant}: {r.status():10s} {cov} "
              f"fallbacks={len(r.scalar_fallbacks)} cos={r.cos} {r.gap_reason}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
