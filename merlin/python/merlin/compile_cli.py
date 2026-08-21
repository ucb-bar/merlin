"""``merlin-compile`` — one command to compile (and optionally build/run/verify) a workload.

The single front door over the compile pipeline. You name a workload + target and it handles the rest
in the background:

  merlin-compile --workload bitvla --dtype int8 --target rvv --run k1 --verify

Target-appropriate semantics (they are genuinely different pipelines, not a false unification):
  * ``--target rvv``     — compile a whole captured MODEL: resolve/capture the model2MLIR bundle →
                           lower (native RVV) → cross-compile the runtime binary → optionally run on
                           host/K1 → gate the output vs the captured ``golden.npy``.
  * ``--target gemmini`` — compile a Gemmini OOT backend PACKAGE and run a capsule through it: build
                           the package → run the capsule on spike/verilator → three-way correctness
                           gate (the accelerator runs kernels/capsules, not whole VLA models).

Fail-closed + honest: a missing toolchain / board / sim yields a clear ``status`` (never a fake pass);
correctness gates before any success is reported. This CLI only ORCHESTRATES the existing, tested API
(``llvmlower.lower``, ``rvvgen.k1``, ``rvvgen.registry``, ``runtime.backends.zephyr_model``,
``targetgen.oot_runner``) — it adds no new compile logic.
"""
from __future__ import annotations

import argparse
import functools
import threading
import json
import sys
import tempfile
from pathlib import Path

# Workloads that ship as model2MLIR capture bundles (RVV whole-model path). Not exhaustive — any
# workloads/<name> with a loader can be captured; this is the "known-good" convenience set for --list.
_RVV_DTYPES = ("fp32", "int8", "fp16", "fp8")


def _bundle_dir(workload: str, dtype: str):
    """The capture bundle `merlin-compile` uses, preferring the FULL-FIDELITY recapture.

    Delegates to ``baselines.bundle.resolve``, which prefers ``<w>_<dtype>_full`` (the real/native
    architecture) and falls back to ``<w>_<dtype>_consistent``. This used to hard-code the
    ``_consistent`` suffix, which for tiny_llama is a 2-layer RANDOM-INIT stand-in rather than the
    real 22-layer TinyLlama-1.1B — so `merlin-compile` silently compiled a toy while every baseline
    arm (which already goes through ``resolve``) compiled the real model.
    """
    from .baselines.bundle import resolve
    return resolve(workload, dtype).root


def _ensure_bundle(workload: str, dtype: str, *, auto_capture: bool) -> Path:
    """Resolve the RVV capture bundle, auto-capturing via model2MLIR if absent (and allowed)."""
    bundle = _bundle_dir(workload, dtype)
    if (bundle / "model.mlir").is_file():
        return bundle
    if not auto_capture:
        raise SystemExit(f"[merlin-compile] no bundle {bundle} (and --no-capture). Capture it with:\n"
                         f"  <model venv>/bin/python $MERLIN_M2M_DIR/workloads/capture_consistent.py "
                         f"{workload} {dtype} {bundle}")
    # Auto-capture goes through model2MLIR's CONSISTENT-capture worker, which is the only one that
    # writes a bundle: `capture.py` emits `.mlir` + `.safetensors` into `workloads/<model>/` and
    # never produces `<bundle>/model.mlir`, so calling it here always ended in the "did not
    # produce" error below no matter how well the capture went. `capture_consistent.py` takes the
    # destination explicitly and emits the inputs/golden/extra the runtime needs.
    #
    # It also runs in the MODEL's venv (from `workloads/<model>/capture.toml`), not merlin's:
    # a model's upstream stack is pinned per model and is generally not importable from here.
    # Honest: a fresh machine without that venv cannot capture — we say so rather than fake a
    # bundle.
    import subprocess
    from .common.paths import env as _env
    m2m = _env("MERLIN_M2M_DIR") or _env("MERLIN_MODEL2MLIR")
    worker = (Path(m2m) / "workloads" / "capture_consistent.py") if m2m else None
    if not worker or not worker.is_file():
        raise SystemExit(f"[merlin-compile] bundle {bundle} missing and MERLIN_M2M_DIR unset/invalid — "
                         f"cannot auto-capture. Set MERLIN_M2M_DIR in .env or pre-capture the bundle.")
    py = _capture_python(Path(m2m), workload)
    print(f"[merlin-compile] bundle absent → auto-capturing {workload} ({dtype}) via "
          f"{worker.name} in {py}…", flush=True)
    r = subprocess.run([str(py), str(worker), workload, dtype, str(bundle)],
                       capture_output=True, text=True, cwd=str(m2m))
    if (bundle / "model.mlir").is_file():
        return bundle
    raise SystemExit(f"[merlin-compile] auto-capture did not produce {bundle}.\n{r.stdout[-800:]}\n{r.stderr[-800:]}")


def _capture_python(m2m_dir: Path, workload: str) -> Path:
    """The interpreter to capture ``workload`` with: its own venv when its capture.toml names one.

    A workload's upstream stack is pinned per model (model2MLIR's `capture.toml [venv]`), so the
    capture must not run under merlin's interpreter just because that is what is executing.
    Falls back to model2MLIR's own venv, then to this interpreter, and never fails here — the
    caller surfaces a capture failure with the worker's output attached.
    """
    toml = m2m_dir / "workloads" / workload / "capture.toml"
    if toml.is_file():
        try:
            import tomllib
            cfg = tomllib.loads(toml.read_text())
        except Exception:                                          # noqa: BLE001
            cfg = {}
        venv = cfg.get("venv")
        if venv:
            cand = Path(venv)
            if not cand.is_absolute():
                cand = m2m_dir / "workloads" / workload / venv
            if (cand / "bin" / "python").is_file():
                return cand / "bin" / "python"
    fallback = m2m_dir / ".venv" / "bin" / "python"
    return fallback if fallback.is_file() else Path(sys.executable)


#: knobs.yaml ``dtype_strategy`` for each ``--dtype``, used to pick a champion package of the
#: RIGHT datatype. An fp32 schedule applied to an int8 workload builds a silently wrong
#: datapath rather than failing, so this must never fall back across datatypes.
#:
#: These strings must match what packages actually declare (``rvvgen.tuning_agent``'s strategy
#: set: fp32, int8_w8a8, bf16_f32acc, fp16_f32acc). "fp16"/"fp8" matched nothing, so
#: ``select_champion`` raised and the fallback below handed back ``hand_v0`` — an **fp32**
#: package — which is exactly the cross-datatype substitution the paragraph above forbids.
_DTYPE_STRATEGY = {"int8": "int8_w8a8", "fp32": "fp32", "fp16": "fp16_f32acc",
                   "bf16": "bf16_f32acc", "fp8": "fp8"}


def default_package(dtype: str) -> str:
    """The package `merlin-compile` uses when `--package` is not given.

    Resolves the CERTIFIED CHAMPION for this datatype via ``targetgen.publish.select_champion``
    rather than hard-coding a name. The previous default was ``hand_v0``/``hand_v0_int8`` — the
    FROZEN, hand-authored, UNOPTIMIZED control that exists to be the before/after baseline. Every
    default invocation therefore shipped the slowest package in the repo while the tuned ones sat
    unused. Falls back to the hand baseline only when no package of this dtype is certified, and
    says so.
    """
    from .targetgen.publish import PublishError, select_champion
    strategy = _DTYPE_STRATEGY.get(dtype)
    if strategy is None:
        raise SystemExit(f"[merlin-compile] no dtype_strategy known for --dtype {dtype}; add one "
                         f"to _DTYPE_STRATEGY together with a package that declares it")
    try:
        sel = select_champion("rvv", dtype_strategy=strategy)
        return str(sel.package_dir)
    except PublishError:
        # Fall back only WITHIN the datatype. The frozen hand baselines exist for fp32 and int8
        # only, so any other dtype has no same-datatype control to fall back to — and handing
        # back an fp32 schedule there would build a silently wrong datapath (see _DTYPE_STRATEGY).
        # Fail with the fix instead: mint and certify a package for this strategy.
        fallback = {"int8": "hand_v0_int8", "fp32": "hand_v0"}.get(dtype)
        if fallback is None:
            raise SystemExit(
                f"[merlin-compile] no package declares dtype_strategy={strategy!r} (for --dtype "
                f"{dtype}), and there is no {dtype} baseline to fall back to. Refusing to "
                f"substitute a package of a different datatype. Either pass --package explicitly "
                f"or mint one: see docs/guides/targetgen.md.") from None
        print(f"[merlin-compile] no certified {strategy} package; falling back to the frozen "
              f"baseline {fallback} (this is the UNOPTIMIZED control)", flush=True)
        from .common.artifacts import artifacts_dir
        return str(artifacts_dir() / "targets" / "rvv" / fallback)


def _workload_features(pkg, bundle, out: dict, harts: int = 1) -> list[str]:
    """The package's compiler features, with a register block that cannot lower re-derived.

    A package's register block is a claim about extents, not a property of the target: a block that
    masks a parallel dim of a contraction it must cover does not lower at all on the integer path
    (LLVM-23 rejects the multi-op ``vector.mask`` that a masked ``transfer_write`` needs) and
    degrades ~34x on fp32. The certified block was chosen on transformer shapes; a model with a
    small or awkward parallel extent -- an FFT's frequency bins, a single-token decode step, a
    3-DoF output head -- fails to build with it.

    Re-derived PER OP CLASS and only where the frozen block provably fails, so a workload that
    already fits compiles byte-identically and existing measurements stand. The substitution is
    reported in the result dict and on stderr, never silently: it changes the emitted kernel.
    """
    frozen = list(pkg.compiler_features)
    try:
        from .rvvgen.apply import blocking_risks, shape_adapted_features
        # A package whose block lives in its SCHEDULE TEXT (no feature) cannot be re-resolved. Say so
        # rather than letting the resulting ~34x fp32 degradation read as "this model is slow".
        for risk in blocking_risks(pkg, bundle):
            out.setdefault("blocking_risks", []).append(risk)
            print(f"[merlin-compile] WARNING: {risk}", file=sys.stderr, flush=True)
        feats = shape_adapted_features(pkg, bundle, harts=harts)
    except Exception as exc:                                        # noqa: BLE001
        print(f"[merlin-compile] shape adaptation unavailable ({type(exc).__name__}: {exc}); "
              f"using the package block as pinned", file=sys.stderr)
        return frozen
    if sorted(feats) != sorted(frozen):
        # A re-derived point may leave an op class UNCLAIMED: no block wider than one lane is legal
        # for its extents, so its contractions go through convert-linalg-to-loops (scalar). Correct,
        # but a perf fact a reader must not have to decode from a feature name -- so say it.
        from .llvmlower.impr_features import unclaimed_op_classes
        unclaimed = sorted({cls for f in feats for cls in unclaimed_op_classes(f)})
        out["features_shape_adapted"] = {"pinned": frozen, "used": feats,
                                         **({"unclaimed_op_classes": unclaimed} if unclaimed else {})}
        print(f"[merlin-compile] the package register block does not lower for this workload's "
              f"contraction extents; re-derived per op class: {frozen} -> {feats}", flush=True)
        if unclaimed:
            print(f"[merlin-compile] NOT VECTORIZED (no multi-lane block is legal for their "
                  f"extents, so they run scalar): {', '.join(unclaimed)}", file=sys.stderr,
                  flush=True)
    return feats


def compile_rvv(workload: str, dtype: str, *, run: str, verify: bool, package: str | None,
                auto_capture: bool, timeout: int, harts: int = 1, iters: int = 1,
                warmup: int = 0, kernel_backend: str | None = None,
                mesh_target: str | None = None, mesh_package: str | None = None) -> dict:
    """RVV whole-model: resolve/capture → lower → build → (run) → (gate vs golden).

    ``kernel_backend='mesh'`` + ``mesh_target`` runs the model's matmul LAYERS on that target's
    accelerator mesh (host dispatch runtime, each matmul injected onto the mesh oracle)."""
    import numpy as np
    from .rvvgen import k1
    from .rvvgen.registry import load_rvv_package
    from .runtime.backends import zephyr_model as zm

    # Sustained mode runs warmup + iters passes inside ONE invocation, so the run deadline has to
    # cover all of them. Without this, `--iters 5 --warmup 2` on a model that takes 148 s per pass
    # needs 1034 s and dies on the 900 s default with a bare TimeoutExpired that names ssh rather
    # than the real cause. Scale the deadline by the pass count the caller actually asked for
    # (never shrink it — an explicit --timeout larger than the scaled value still wins).
    passes = max(1, int(iters)) + max(0, int(warmup))
    if passes > 1:
        timeout = max(timeout, timeout * passes)

    bundle = _ensure_bundle(workload, dtype, auto_capture=auto_capture)
    pkg_dir = package or default_package(dtype)
    pkg = load_rvv_package(pkg_dir)
    work = Path(tempfile.mkdtemp(prefix=f"merlin_compile_{workload}_{dtype}_"))
    out: dict = {"tool": "merlin-compile", "target": "rvv", "workload": workload, "dtype": dtype,
                 "bundle": str(bundle), "package": pkg_dir, "run": run,
                 "harts": harts, "iters": iters}
    refs: dict = {}
    if verify:
        refs["fp32"] = np.load(bundle / "golden.npy")
        w8 = bundle / "golden_w8a8.npy"
        if pkg.is_int8 and w8.is_file():
            refs["w8a8"] = np.load(w8)
        elif pkg.is_int8:
            # `golden.npy` in an int8 bundle is a WEIGHT-ONLY-int8 reference (int8 weights, fp32
            # activations); this package computes W8A8. Grading one against the other measures
            # activation-quantization error, not correctness, and reads as a large cos drop. Say
            # so rather than letting the run be silently judged by the wrong yardstick.
            out["reference_warning"] = (
                f"{bundle.name} has no golden_w8a8.npy — an int8 (W8A8) run graded only against "
                f"the weight-only golden.npy. A low fp32_cos here is expected quantization "
                f"divergence, NOT evidence of a defect. Generate the W8A8 reference first.")
            print(f"[merlin-compile] WARNING: {out['reference_warning']}", file=sys.stderr)

    # `host` runs the model through the x86 dispatch runtime (per-kernel JIT via the SAME
    # MLIR lowering, minus the RVV/cross-compile tail). It needs no board, simulator or
    # cross-toolchain, so it is the first correctness stage on a fresh machine — and the
    # discriminator when a board run disagrees: host-vs-board separates a quantization-math
    # bug (both wrong) from a codegen/RVV bug (only the board wrong).
    if run == "host":
        from .runtime.dispatch_runtime import run_model

        res = run_model(bundle, work, int8_compute=pkg.is_int8,
                        kernel_backend=kernel_backend, mesh_target=mesh_target,
                        mesh_package=mesh_package)
        if kernel_backend == "mesh":
            # Read the counters off THIS run's result. They used to be function attributes on a
            # module-global (`_dr.execute.mesh_ran`), which concurrent grades clobber -- and a model
            # verdict now depends on them. `UNKNOWN` rather than None when absent: None reads as "zero
            # layers ran", when what it means is "nobody could tell".
            from .common.provenance import UNKNOWN as _UNKNOWN
            out["mesh_execution"] = {
                "target": mesh_target,
                "matmul_layers_on_mesh": res.get("mesh_ran", _UNKNOWN),
                "matmul_layers_host_fallback": res.get("mesh_fell_back", _UNKNOWN),
                # coverage, NOT a gate: matmuls the classifier never routed (e.g. batched attention
                # generics), so a row can say "15 of 19 layers on the mesh" instead of implying 19.
                "matmul_layers_unrouted": res.get("mesh_unrouted_matmuls", _UNKNOWN)}
        out["status"] = "ran"
        out["n_kernels"] = res.get("n_kernels")
        if refs:
            g = zm._gate(res["output"], refs)
            out["verify"] = {"gate_ok": bool(g.get("ok")), **g}
            out["status"] = "verified" if g.get("ok") else "run_mismatch"
        return out

    # The Zephyr/spike/verilator routes build a Zephyr image (and are the only ones that can be
    # multicore or sustained); the K1 route builds a Linux binary for the board.
    if run in ("spike", "zephyr", "verilator"):
        board = "chipyard_riscv64" if run == "verilator" else "spike_riscv64"
        if not zm.available():
            out["status"] = "not_run"
            out["reason"] = "Zephyr/spike toolchain unavailable (ZEPHYR_BASE / SDK / MERLIN_CHIPYARD)"
            return out
        feats = _workload_features(pkg, bundle, out)

        def _build(fs):
            return zm.build_app(bundle, work, board=board, backend="rvv", rvv_hart=0,
                                int8_compute=pkg.is_int8, rvv_schedule=pkg.schedule_text,
                                cflags_override=pkg.cflags + zm._CFLAGS_COMMON,
                                features=frozenset(fs) or None,
                                n_harts=harts, iters=iters, warmup=warmup,
                                cpus=max(2, harts))

        try:
            b = _build(feats)
        except Exception as exc:                                    # noqa: BLE001
            # A multicore build splits each matmul over N BEFORE the package schedule runs, so the
            # block has to cover the PER-HART tile, not the model's N. Narrowing the block for that
            # up front would be wrong: measured on spectformer at 3 harts, the un-narrowed block
            # lowers fine and is 2.04x FASTER (2.56e9 vs 5.21e9 cycles), because the legality
            # predicate is conservative on the dynamic tile a forall produces. So prefer the fast
            # block and fall back only when the build actually rejects it -- one wasted build on the
            # rare model that needs it (lstmnetvit at 3 harts, whose N=2 splits to a 1-wide tile),
            # instead of a permanent 2x on every model that does not.
            if harts < 2 or "vector.mask" not in str(exc):
                raise
            narrowed = _workload_features(pkg, bundle, out, harts=harts)
            print(f"[merlin-compile] the register block does not lower once each matmul is split "
                  f"across {harts} harts; re-deriving against the per-hart tile: "
                  f"{feats} -> {narrowed}", file=sys.stderr, flush=True)
            out["harts_split_block_retry"] = {"tried": list(feats), "used": list(narrowed)}
            b = _build(narrowed)
            feats = narrowed
        out["binary"] = str(b["elf"]); out["status"] = "compiled"
        if run == "verilator":
            sim = zm.verilator_sim()
            if sim is None:
                out["status"] = "not_run"
                out["reason"] = ("no multicore Saturn Verilator sim built (see "
                                 "docs/guides/tinyllama_int8_rvv_zephyr.md)")
                return out
            res = zm.run_on_verilator(b["elf"], timeout=timeout,
                                      references=refs or None)
        else:
            res = zm.run_on_spike(b["elf"], harts=max(2, harts), mem_bytes=b["ram_bytes"],
                                  timeout=timeout)
            if refs:
                res.update(zm._gate(res["prefix"], refs))
        out["status"] = "ran"
        out["cycles"] = res.get("metrics", {}).get("cycles")
        if res.get("sustained"):
            out["sustained"] = res["sustained"]
        if refs:
            out["verify"] = {"gate_ok": bool(res.get("ok")), **{k: v for k, v in res.items()
                                                                if k.endswith(("_cos", "_rel", "_max_rel"))}}
            out["status"] = "verified" if res.get("ok") else "run_mismatch"
        return out

    # K1 / compile-only. `run_on_k1` does its own build (into <work>/v), so building here too
    # would compile the whole model TWICE — for TinyLlama int8 that is ~40 min of clang thrown
    # away, enough to push the run past its own timeout. Build directly only when nothing else
    # will.
    if run != "k1":
        binary = k1.build_k1_binary(bundle, work, pkg, inputs_npz=bundle / "inputs.npz")
        out["binary"] = str(binary)
    out["status"] = "compiled"
    if run == "none":
        return out
    if run == "k1":
        if not k1.available():
            out["status"] = "not_run"; out["reason"] = "K1 board unreachable (see MERLIN_K1_HOST/port 2222)"
            return out
        res = k1.run_on_k1(bundle, work, pkg, timeout=timeout, iters=iters, warmup=warmup)
        out["binary"] = str(Path(work) / "v" / "merlin_k1")
        out["status"] = "ran"; out["cycles"] = res.get("cycles"); out["vlen"] = res.get("vlen")
        if res.get("sustained"):
            out["sustained"] = res["sustained"]
        if verify:
            g = zm._gate(res["prefix"], refs)
            out["verify"] = {"gate_ok": bool(g.get("ok")), **g}
            out["status"] = "verified" if out["verify"]["gate_ok"] else "run_mismatch"
        return out
    out["status"] = "not_run"
    out["reason"] = f"run mode {run!r} not supported for rvv (use none|k1|spike|zephyr|verilator)"
    return out


_MESH_NTILE_WIDTH: dict[tuple, int] = {}   # N-tile width a target's backend accepts
_MESH_PKG_CACHE: dict[str, object] = {}
_MESH_PKG_LOCK = threading.Lock()


def _built_mesh_package(pkg_dir: str, timeout: int):
    """The loaded+scanned+built package for ``pkg_dir``, built at most once per process.

    The whole-model mesh path calls this once per matmul layer. Rebuilding per layer is pure waste and
    turns one broken build into a per-layer storm of identical failures.
    """
    key = str(Path(pkg_dir).resolve())
    with _MESH_PKG_LOCK:
        hit = _MESH_PKG_CACHE.get(key)
    if hit is not None:
        return hit
    from .targetgen.oot_runner import build_package, integrity_scan, load_package
    obj = load_package(pkg_dir, contract=None)     # the same sequence run_entrypoints does when pkg is None
    integrity_scan(obj)
    build_package(obj)
    with _MESH_PKG_LOCK:
        _MESH_PKG_CACHE[key] = obj
    return obj


@functools.lru_cache(maxsize=None)
def mesh_datapath(target: str) -> tuple[str, str, bool]:
    """``(operand_dtype, accum_dtype, is_integer)`` the target's own compute unit declares.

    Derived from the target contract via :func:`_mesh_tile_binding`, never assumed: addressing an fp32
    SIMT mesh as int8 would quantize operands the datapath never asked to have quantized.
    """
    b = _mesh_tile_binding(target, None, None)
    return b.operand_dtype, b.accum_dtype, bool(b.integer)


def _default_oot_package(target: str) -> str | None:
    """The conventional OOT backend package for ``target``, or None when the target ships no default (a
    bespoke-layout target must name it via ``--package``). Prefers the agent-submission package
    (``.../targets/<target>/agent_spec_v1_mlir_oot`` — the SAME default ``compile_oot`` resolves, so on-mesh
    tile verification reuses it verbatim) and falls back to a hand-curated REFERENCE backend package
    (``.../reference_v0``) when a target ships one instead of an agent submission (e.g. a SIMT core whose
    deterministic reference lowering is the package). Target-agnostic — both are directory conventions, not
    target literals."""
    from .common.artifacts import artifacts_dir
    base = artifacts_dir() / "targets" / target
    for pkg_id in ("agent_spec_v1_mlir_oot", "reference_v0"):
        cand = base / pkg_id
        if (cand / "manifest.yaml").is_file():
            return str(cand)
    return None


def _dtype_bytes(tok: str | None) -> int:
    """Byte width of an operand dtype token (i8/i16/i32/f16/f32/…). Defaults to 1 (int8-class)."""
    if not tok:
        return 1
    digits = "".join(ch for ch in str(tok) if ch.isdigit())
    return max(1, int(digits) // 8) if digits else 1


def _scratchpad_capacity_elems(target: str, dtype_bytes: int) -> int | None:
    """On-chip operand-store capacity (in elements of ``dtype_bytes``) DERIVED from the target's RTL
    memory facts (the ``scratchpad`` memory's byte size); None when facts don't declare it (fail-open —
    the caller then keeps the untiled extent). Never a hardcoded capacity."""
    try:
        from .targetgen.rtl import facts as _facts
        mems = (_facts.load_facts(target).get("facts") or {}).get("memories") or []
    except Exception:  # noqa: BLE001 — no facts bundle → capacity unknown, caller falls back
        return None
    sp = next((m for m in mems if m.get("name") == "scratchpad"), None)
    nbytes = sp.get("bytes") if sp else None
    return int(nbytes) // max(1, dtype_bytes) if nbytes else None


def _capacity_fit_tile(M: int, K: int, N: int, D: int, cap_elems: int) -> tuple[int, int, int, int]:
    """Shrink a matmul (M,K,N) to the largest D-aligned tile whose on-chip working set — the weight
    tile K·N plus the activation tile M·K — fits ``cap_elems``, and return (mt,kt,nt,n_tiles) where
    n_tiles is how many such tiles cover the layer. Keeps M whole (the row dim streams); halves the
    larger of K/N (never below D) until the tile fits. Pure arithmetic, target-agnostic."""
    def _fits(mt, kt, nt):
        return kt * nt + mt * kt <= cap_elems

    mt, kt, nt = M, K, N
    while not _fits(mt, kt, nt) and (kt > D or nt > D):
        if nt >= kt and nt > D:
            nt = max(D, (nt // 2 // D) * D or D)
        elif kt > D:
            kt = max(D, (kt // 2 // D) * D or D)
        else:
            nt = max(D, (nt // 2 // D) * D or D)
    import math
    n_tiles = math.ceil(M / mt) * math.ceil(K / kt) * math.ceil(N / nt)
    return mt, kt, nt, n_tiles


def _mesh_tile_binding(target: str, operand_dtype: str | None, accum_dtype: str | None,
                       requant_output_dtype: str | None = None):
    """A ``corpus_spec.CorpusBinding`` for synthesizing a single systolic mesh tile of ``target``, with the
    operand/accumulate datapath pinned to what the routed op actually needs (derived, never assumed). The
    tile dim, compare policy, and instruction-class deriver all come from the target's own descriptor +
    manifest via ``derive_binding`` — no target literal, no hand-set dims.

    ``requant_output_dtype`` (the narrow dtype an ``acc_scale`` epilogue requants the i32 accumulator to,
    e.g. i8) is passed only when the caller needs a requant handoff — an int8 matmul chain commits each
    layer's accumulator back to i8 so it can feed the next mesh layer."""
    from types import SimpleNamespace
    from .targetgen import corpus_spec as CS
    from .targetgen.capsule_runner import _bespoke_sim_via
    te = SimpleNamespace(target=target, sim_via=_bespoke_sim_via(target))
    datapath: dict = {}
    if operand_dtype:
        datapath["operand_dtype"] = operand_dtype
    if accum_dtype:
        datapath["accum_dtype"] = accum_dtype
    if requant_output_dtype:
        datapath["requant_output_dtype"] = requant_output_dtype
    return CS.derive_binding(te, datapath)


def _mesh_verify(plan: dict, *, target: str, package: str | None, timeout: int) -> dict:
    """Execute each mesh-routed matmul as a single systolic tile on the target's REAL mesh oracle.

    For every op the router placed on the accelerator mesh (``plan["mesh"]``), synthesize its
    ``merlin_iface`` tile capsule (``corpus_spec.build_matmul`` — one ``DxD`` tile at the target's derived
    mesh dimension, in the op's routed operand/accumulate dtype) and run it through the EXISTING
    ``oot_runner.certify`` accelerator path (the same one ``compile_oot`` uses). ``certify`` gates the
    emitted kernel's output bit-exact (integer) / within-tolerance (float) against the command buffer's
    mathematical reference == simulate == the RTL/mesh oracle — so a PASS is proof the matmul executed
    correctly ON the mesh, not merely that a routing plan was produced.

    FAIL-CLOSED: a tile whose oracle is unavailable is recorded ``oracle_unavailable`` (never a silent pass);
    an op with no single-tile synthesizer is recorded honestly and never counted as executed."""
    import tempfile
    from .targetgen import oot_runner
    from .targetgen import corpus_spec as CS
    from .benchharness import runs_root

    pkg_dir = package or _default_oot_package(target)
    out: dict = {"n_tiles": 0, "n_passed": 0, "n_failed": 0, "n_unavailable": 0,
                 "n_unsynthesizable": 0, "package": pkg_dir, "per_tile": []}
    if pkg_dir is None:
        out["status"] = "not_run"
        out["reason"] = (f"no default OOT backend package for target {target!r}; pass --mesh-package "
                         f"to name one")
        return out
    # Build the OOT backend ONCE up front so a broken build is a single honest not_run, not a per-tile
    # storm. certify re-runs an incremental (no-op) build per tile — harmless.
    try:
        oot_runner.build_package(oot_runner.load_package(pkg_dir), timeout=timeout)
    except Exception as e:  # noqa: BLE001 — a build failure is honest not_run, never a fake mesh pass
        out["status"] = "not_run"
        out["reason"] = f"OOT backend build failed: {type(e).__name__}: {str(e)[-400:]}"
        return out

    # Real RTL oracle by default (verilator); MERLIN_MESH_SIM can select spike (functional bootstrap) or
    # verilator (cycle-accurate RTL). The whole point of on-hardware validation is the RTL, not the ISS.
    import os
    sim = os.environ.get("MERLIN_MESH_SIM", "verilator")
    rr = runs_root(target, "mesh_verify")
    for i, r in enumerate(plan.get("mesh", [])):
        d = r.demand
        op = "matmul" if d.op in ("matmul", "linear") else d.op
        rec: dict = {"op": d.op, "site": d.site}
        if op not in CS.BUILDERS:
            out["n_unsynthesizable"] += 1
            rec.update(status="no_tile_synthesizer",
                       reason=f"no single-tile capsule synthesizer for mesh op {d.op!r}")
            out["per_tile"].append(rec)
            continue
        try:
            binding = _mesh_tile_binding(target, d.in_fmt, r.acc)
            D = binding.tile_dim
            # compile the matmul LAYER at its REAL extent when the router carried one (rounded up to the
            # mesh dim — the backend tiles it into DxD tiles); else a single DxD tile.
            def _rup(x):
                return D if not x else ((int(x) + D - 1) // D) * D
            M, K, N = (_rup(d.m), _rup(d.k), _rup(d.n)) if (d.m and d.k and d.n) else (D, D, D)
            # A whole layer's weight tile (K·N) + activation (M·K) may exceed the on-chip operand store;
            # shrink to the largest capacity-fit tile (derived from the target's scratchpad memory fact)
            # and record how many such tiles cover the layer. The certified tile is the layer's real
            # repeating unit — a fit tile that passes proves the layer runs on the mesh once tiled. When
            # the capacity fact is absent we keep the untiled extent (prior behavior).
            layer_extent = f"{M}x{K}x{N}"
            n_subtiles = 1
            sp_cap = _scratchpad_capacity_elems(target, _dtype_bytes(binding.operand_dtype))
            if sp_cap and (K * N + M * K) > sp_cap:
                M, K, N, n_subtiles = _capacity_fit_tile(M, K, N, D, sp_cap)
            entry = {"name": f"mesh_tile_{i}_{op}", "op": op, "kind": "op",
                     "source_role": "mesh_tile_synthesized",
                     "source_reference": f"whole-model op {d.op} [{d.site}]: layer {layer_extent} on the "
                                         f"mesh as {n_subtiles} capacity-fit {M}x{K}x{N} tile(s)",
                     "M": M, "K": K, "N": N}
            capsule, mlir = CS.build(entry, binding)
        except Exception as e:  # noqa: BLE001 — a synthesis failure is recorded, never a fake pass
            out["n_unsynthesizable"] += 1
            rec.update(status="synth_error", reason=f"{type(e).__name__}: {str(e)[-300:]}")
            out["per_tile"].append(rec)
            continue
        rec.update(M=M, K=K, N=N, layer_extent=layer_extent, n_subtiles=n_subtiles,
                   operand_dtype=binding.cap_dtype(binding.operand_dtype),
                   output_dtype=binding.cap_dtype(binding.accum_dtype), sim=sim)
        with tempfile.TemporaryDirectory(prefix="mesh_tile_") as td:
            iface = Path(td) / f"{entry['name']}.interface.mlir"
            iface.write_text(mlir, encoding="utf-8")
            res = oot_runner.certify(pkg_dir, iface, runs_root=str(rr), run_id=entry["name"],
                                     simulator=sim, target=target, timeout=timeout)
        oracle = res.get("oracle") or {}
        out["n_tiles"] += 1
        rec.update(oracle_kind=oracle.get("kind"), oracle_result=oracle.get("result"),
                   cycles=oracle.get("cycles"))
        if oracle.get("result") == "skipped":
            out["n_unavailable"] += 1
            rec["status"] = "oracle_unavailable"
            rec["reason"] = (res.get("failure") or {}).get("detail") or "mesh oracle unavailable in this env"
        elif res.get("status") == "pass":
            out["n_passed"] += 1
            rec["status"] = "pass"
        else:
            out["n_failed"] += 1
            rec["status"] = "fail"
            rec["reason"] = (res.get("failure") or {}).get("detail")
        out["per_tile"].append(rec)

    if out["n_tiles"] == 0:
        out["status"] = "no_mesh_matmuls" if out["n_unsynthesizable"] == 0 else "no_synthesizable_mesh_ops"
    elif out["n_unavailable"] == out["n_tiles"]:
        out["status"] = "oracle_unavailable"
    elif out["n_passed"] == out["n_tiles"]:
        out["status"] = "verified"
    else:
        out["status"] = "partial"
    out["note"] = (
        "each mesh-routed matmul LAYER is verified by EXECUTING its capacity-fit tile on the target mesh "
        f"oracle (simulator={sim}): the tile is the largest D-aligned unit whose weight+activation working "
        "set fits the target's DERIVED scratchpad capacity (n_subtiles = how many tile it into the layer); "
        "the emitted kernel's output is gated (bit-exact int / tolerance float) against the command buffer's "
        "reference == simulate == oracle three-way. A fit tile that certifies proves the layer runs on the "
        "mesh once tiled. The remaining gap is the single SPLICED image (all layers' mesh kernels + the "
        "scalar/RVV remainder co-scheduled in one binary with activations handed between layers) — see "
        "compile_model's docstring.")
    return out


def run_matmul_on_mesh(target: str, A: list, W: list, *, operand_dtype: str | None = None,
                       accum_dtype: str | None = None, simulator: str | None = None,
                       package: str | None = None, epilogue: list | None = None,
                       acc_scale: float | None = None, timeout: int = 900,
                       _padded: bool = False) -> list | None:
    """Execute ``A @ W`` on the target's mesh oracle with the REAL operand values INJECTED (not
    materialized-from-name), and return the mesh's output tensor (nested list) — or ``None`` when this
    target has no reachable mesh path. ``A`` / ``W`` are the layer's real activations / weights.

    ``epilogue`` (e.g. ``["acc_scale"]``) applies the accumulator epilogue the layer commits with; an
    ``acc_scale`` epilogue re-quantizes the i32 accumulator to the target's narrow requant dtype (the
    target-declared ``requant_output_dtype``, e.g. i8 for an integer mesh) by the float ``acc_scale`` +
    saturating cast, so the layer's output is a same-dtype activation that can feed the NEXT mesh layer —
    this is what makes an int8 matmul CHAIN executable on the mesh end to end (the requant handoff).

    TARGET-AGNOSTIC: it dispatches on the target's DERIVED ``endpoint_kind`` (from the contract, never a
    literal) to that target's OWN oracle — a systolic/RoCC target (``inline_asm_insn``) through its generated
    OOT-package cert path, a self-hosted-ISA target (``external_backend``) through the program oracle whose
    RTL cosim mlc DERIVES from the target. The per-target kernel + harness come from the GENERATED package;
    this tool only builds the merlin_iface interface, injects the operands, and reads the output back.
    Fail-closed (``None``) on an unavailable oracle or an endpoint with no mesh path (never a fabricated
    result)."""
    from .targetgen import corpus_spec as CS
    from .targetgen.capsule_runner import _bespoke_sim_via, _endpoint_of, _SIM_ORACLES

    endpoint, model_ext = _endpoint_of(target)
    M, K, N = len(A), len(A[0]), len(W[0])
    # An acc_scale requant commits the accumulator back to the operand's narrow dtype so the layer's
    # output can feed the next mesh layer (the int8 chain handoff); tell the binding that narrow dtype.
    requant_out = (operand_dtype or "i8") if (epilogue and "acc_scale" in epilogue) else None
    binding = _mesh_tile_binding(target, operand_dtype, accum_dtype, requant_output_dtype=requant_out)
    entry = {"name": "mesh_layer", "op": "matmul", "kind": "op",
             "source_role": "mesh_layer_real_operands",
             "source_reference": f"whole-model matmul layer {M}x{K}x{N} on the mesh with real operands",
             "M": M, "K": K, "N": N, "lhs": "A0", "weight": "W", "out": "Y0"}
    if epilogue:
        entry["epilogue"] = list(epilogue)
        if acc_scale is not None:
            entry["acc_scale"] = float(acc_scale)
    _, mlir = CS.build(entry, binding)

    # DISPATCH ORDER is deliberate (mirrors capsule_runner.oracle_adapters): a target whose contract
    # DECLARES an EXCLUSIVE bespoke sim (a self-hosted SIMT core graded on its own emitted kernel by its
    # own oracle, e.g. a cyclotron/muon backend) is routed FIRST — its endpoint is ALSO external_backend,
    # but the arc command-buffer program oracle grades the WRONG artifact for a SIMT kernel, so the bespoke
    # executor must take precedence. Only when no exclusive sim is declared does the endpoint kind pick the
    # path. Derived from the contract's sim_via + the _SIM_ORACLES registry — never a target-name branch.
    so = _SIM_ORACLES.get(_bespoke_sim_via(target))
    if so is not None and so.exclusive:
        out = _matmul_via_bespoke_sim(target, mlir, A, W, package=package, timeout=timeout)
    elif endpoint in (None, "inline_asm_insn", "upstream_target"):
        out = _matmul_via_oot_cert(target, mlir, A, W, simulator=simulator, package=package,
                                   timeout=timeout)
    elif endpoint == "external_backend":
        out = _matmul_via_program_oracle(target, mlir, A, W, model_ext=model_ext,
                                         package=package, timeout=timeout)
    else:
        return None                              # no mesh-execution path derived for this endpoint kind
    if out is not None or _padded:
        return out
    # The backend refused this extent. On a target that declares no scratchpad capacity fact there is
    # nothing to derive a tile size FROM, and inventing one would be a target literal in disguise. Find
    # the width this backend accepts by halving, then run the layer as N-tiles of that width: splitting
    # along N is exact, because C[:, a:b] is A @ W[:, a:b]. Costs a few refused probes ONCE per target.
    D = int(getattr(binding, "tile_dim", 0) or 0)
    if D <= 0 or N <= D:
        return None
    key = (target, operand_dtype or "", accum_dtype or "")
    width = _MESH_NTILE_WIDTH.get(key)
    if width is None:
        w = N
        while w > D:
            w = max(D, (w // 2 // D) * D or D)
            probe = run_matmul_on_mesh(target, A, [row[:w] for row in W],
                                       operand_dtype=operand_dtype, accum_dtype=accum_dtype,
                                       simulator=simulator, package=package, timeout=timeout,
                                       epilogue=epilogue, acc_scale=acc_scale, _padded=True)
            if probe is not None:
                width = w
                break
        if width is None:
            return None                          # even a single tile-dim-wide slice is refused
        _MESH_NTILE_WIDTH[key] = width
    cols: list[list] = []
    for a in range(0, N, width):
        b = min(a + width, N)
        piece = run_matmul_on_mesh(target, A, [row[a:b] for row in W],
                                   operand_dtype=operand_dtype, accum_dtype=accum_dtype,
                                   simulator=simulator, package=package, timeout=timeout,
                                   epilogue=epilogue, acc_scale=acc_scale, _padded=True)
        if piece is None:
            return None                          # a tile the discovered width should have covered
        cols.append(piece)
    return [sum((c[r] for c in cols), []) for r in range(M)]


def _matmul_via_oot_cert(target, mlir, A, W, *, simulator, package, timeout) -> list | None:
    """Systolic/RoCC path: certify the matmul through the target's generated OOT package on its ELF/mesh
    oracle, with the real operands injected (``inputs`` -> the package harness's ``materialize_inputs``)."""
    import os
    import tempfile
    from .benchharness import runs_root
    from .targetgen import oot_runner
    sim = simulator or os.environ.get("MERLIN_MESH_SIM", "verilator")
    pkg = package or _default_oot_package(target)
    if pkg is None:
        return None
    with tempfile.TemporaryDirectory(prefix="mesh_layer_") as td:
        iface = Path(td) / "mesh_layer.interface.mlir"
        iface.write_text(mlir, encoding="utf-8")
        res = oot_runner.certify(pkg, iface, runs_root=str(runs_root(target, "mesh_run")),
                                 run_id="mesh_layer", simulator=sim, target=target, timeout=timeout,
                                 inputs={"A0": A, "W": W})
    if res.get("status") != "pass":
        return None
    return (res.get("oracle_outputs") or {}).get("Y0")


def _matmul_via_program_oracle(target, mlir, A, W, *, model_ext, package, timeout) -> list | None:
    """Self-hosted-ISA (external_backend) path: emit the target's kernel from the interface through its
    generated OOT package (target-agnostic ``run_entrypoints``), inject the real operands onto the command
    buffer's leaf tensors, and run on the target's mlc-DERIVED arc cosim via the generic program oracle.
    Returns the output or ``None`` (fail-closed). The kernel/codegen is the GENERATED package's; the operand
    injection + oracle dispatch are target-agnostic. Requires ``model_ext`` (the operand-layout model)."""
    if not model_ext:
        return None
    from .targetgen import mesh_program_run
    return mesh_program_run.matmul_on_program_oracle(
        target, mlir, A, W, model_ext=model_ext, package=package or _default_oot_package(target),
        timeout=timeout)


def _matmul_via_bespoke_sim(target, mlir, A, W, *, package, timeout) -> list | None:
    """Exclusive bespoke-sim path: a self-hosted SIMT core (endpoint ``external_backend``) graded on the
    kernel its OWN generated package emits, by its OWN declared bespoke oracle (e.g. cyclotron via the muon
    backend) — NOT the arc command-buffer program oracle, which would grade the wrong artifact for a SIMT
    kernel. Emits the matmul kernel from the target's generated OOT package via the SAME target-agnostic
    entrypoint runner the grader uses (``capsule_common.run_entrypoints``), INJECTS the real ``A``/``W``
    operands onto the command buffer's leaf tensors (``preload_b64``, decoded harness-side by
    ``muon_harness``), runs on the target's DECLARED exclusive bespoke oracle, and reads the named output
    tensor (``Y0``) back off the device. Target-agnostic: the sim engine + adapters come from the DERIVED
    ``_SIM_ORACLES`` entry (contract ``sim_via``); the kernel/codegen is the generated package's. Fail-closed
    (``None``) on a missing package, an unavailable oracle, or an entrypoint/oracle failure — never a
    fabricated result."""
    import base64
    import tempfile

    from .benchharness import runs_root
    from .targetgen import capsule_common as CC
    from .targetgen import capsule_runner as CR
    from .targetgen import mesh_program_run as MP
    from .targetgen.capsule_common import make_run_paths

    pkg = package or _default_oot_package(target)
    if pkg is None:
        return None
    so = CR._SIM_ORACLES.get(CR._bespoke_sim_via(target))
    if so is None or not so.exclusive:
        return None
    ok, _reason = so.available(target)
    if not ok:
        return None
    adapters = so.adapters(target)                    # {tier: adapter}
    # The FUNCTIONAL tier (L2) carries the numeric grade for a SIMT core; fall back to any adapter present.
    run = adapters.get("L2") or next(iter(adapters.values()), None)
    if run is None:
        return None

    with tempfile.TemporaryDirectory(prefix="mesh_bsim_") as td:
        tdp = Path(td)
        cdir = tdp / "cap"
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / "capsule.interface.mlir").write_text(mlir, encoding="utf-8")
        capsule = {"name": "mesh_layer", "kind": "op", "interface_mlir": "capsule.interface.mlir",
                   "operation": {"op": "matmul", "attributes": {}}, "__dir__": str(cdir),
                   "required_oracle_tiers": ["L2"]}
        # A run id per LAYER SHAPE: `mesh_layer` was hardcoded, so every layer of a model wrote into one
        # run dir and overwrote the last, leaving nothing to attribute a failure to afterwards.
        _rid = f"mesh_layer_{len(A)}x{len(A[0]) if A else 0}x{len(W[0]) if W else 0}"
        paths = make_run_paths(runs_root(target, "mesh_bsim"), _rid, suite="mesh",
                               target=target, dtype="prog", benchmark=_rid)
        try:
            _built = _built_mesh_package(pkg, timeout)     # built once per process, not once per layer
            _pkg, cb, kernel_text = CC.run_entrypoints(_built, pkg, capsule, paths, contract=None,
                                                       timeout=timeout, fourth_output_name="kernel.cpp")
        except Exception:                             # noqa: BLE001 — package can't emit this kernel: honest None
            return None
        if cb is None or not kernel_text:
            return None
        # INJECT the real operands onto the cb's leaf tensors (encoded for each tensor's declared dtype);
        # the muon harness decodes ``preload_b64`` and embeds THESE values instead of the materialized ones.
        operands = {"A0": A, "W": W}
        for tname, tspec in (cb.get("tensors") or {}).items():
            if tspec.get("role") in ("input", "weight", "bias") and tname in operands:
                raw = MP._encode_operand(operands[tname], tspec.get("dtype", "f32"))
                if raw is None:
                    return None
                tspec["preload_b64"] = base64.b64encode(raw).decode()
        try:
            res = run(cb, kernel_text, tdp / "oracle", timeout)
        except Exception:                             # noqa: BLE001 — oracle unavailable / run failure: fail closed
            return None
        outs = res.get("outputs") or {}
        return outs.get("Y0") or next(iter(outs.values()), None)


def run_whole_model_on_mesh(target: str, module, *, in_fmt: str = "f32",
                            weight_fmt: str | None = None, leaf_values: dict | None = None,
                            operand_dtype: str | None = None, accum_dtype: str | None = None,
                            simulator: str | None = None, package: str | None = None,
                            ref_target: str = "toy_npu", seed: int = 0, timeout: int = 900) -> dict:
    """Execute a WHOLE MODEL co-scheduled across the target's mesh + scalar lanes on the REAL oracle.

    Routes ``module``'s ops (``mesh_program_run.demands_from_module``), builds the co-scheduled
    whole-model program, then walks it IN ORDER: each mesh matmul LAYER runs on the target's real oracle
    (``run_matmul_on_mesh`` — the operands injected, the kernel emitted by the target's generated OOT
    package, the output read back off the device), while norms/activations/elementwise run inline on the
    scalar lane, and every layer's on-device output is handed to the op that consumes it. The final tensor
    is gated against the whole-model engine reference (``lower_module`` of the entire module) — a PASS is
    proof the model ran end-to-end with its matmul layers ON the mesh, not merely that a plan was produced.

    This is the single co-scheduled whole-model run (mesh layers on hardware + scalar lane inline +
    inter-lane activation handoff). It is TARGET-AGNOSTIC: the lane split is READ from the routing plan, the
    layer extents from the module's def-use edges, the kernel from the generated package. FAIL-CLOSED: a
    layer with no reachable oracle returns ``status="oracle_unavailable"`` (never a fabricated result).

    Residual toward a single fused binary: this co-schedules the mesh kernels with the scalar lane
    host-driven (one program, multiple dispatched kernels), not yet ONE fused kernel in a single device
    address space — that final slice is the OOT backend emitting the whole loop nest inline. Seeded with
    small-integer operands by default so an integer mesh reproduces the f32 reference bit-exact."""
    import os

    import numpy as np

    from .targetgen import compute_units as _cu
    from .targetgen import mesh_program_run as mp
    from .targetgen import routing as _routing
    from .targetgen import target_registry as tr
    from .xdsl_dialects.lowering import execute, lower_module

    fn = next(op for op in module.walk() if op.name == "func.func")
    args = list(fn.body.blocks[0].args)

    def _shape(v):
        return [int(d) for d in v.type.get_shape()]

    rng = np.random.default_rng(seed)
    if leaf_values is None:
        # Small integer operands: an integer mesh (e.g. int8·int8->int32) reproduces the f32 engine
        # reference EXACTLY, so the whole-model gate is bit-exact rather than tolerance-bounded.
        arrays = {a: np.rint(rng.standard_normal(tuple(_shape(a))) * 3).clip(-8, 7).astype(np.float32)
                  for a in args}
    else:
        by_index = {a.index: a for a in args}
        arrays = {by_index[int(k[1:])]: np.asarray(v, dtype=np.float32) for k, v in leaf_values.items()}

    # Route + build the co-scheduled program, then run its mesh lane on the REAL oracle.
    demands = mp.demands_from_module(module, in_fmt, weight_fmt)
    plan = _routing.route_plan_on(demands, _cu.compute_units(tr.load_contract(target)))
    program = mp.build_whole_model_program(plan, target, module)
    seed_leaves = {f"L{a.index}": arrays[a] for a in args if f"L{a.index}" in program.leaves}

    # Whole-model reference (the numeric gate). A module in the engine's op vocabulary lowers+runs as ONE
    # module on the target-agnostic engine (through an in-tree ``ref_target``); a module carrying a
    # transcendental / fused op the engine cannot evaluate (softmax/rmsnorm/rope/attention/…) is gated
    # against a host-eager numpy recomputation of the whole model instead. Either way the mesh execution
    # below runs the matmul layers on the REAL ``target`` hardware; both must agree on the whole-model result.
    if mp._engine_can_lower(module):
        ref_names = mp._reference_leaf_names(module)
        ref_inj = {ref_names[a]: arrays[a].tolist() for a in args}
        ref_final = np.asarray(next(iter(
            execute(lower_module(module, target=ref_target), ref_inj)["outputs"].values())))
        ref_kind = "engine"
    else:
        ref_final = mp._host_eager_final(program, seed_leaves)
        ref_kind = "host_eager"

    per_layer: list = []

    def _mesh_exec(lhs, rhs, step):
        la, ra = np.asarray(lhs), np.asarray(rhs)
        got = run_matmul_on_mesh(target, la.tolist(), ra.tolist(),
                                 operand_dtype=operand_dtype, accum_dtype=accum_dtype,
                                 simulator=simulator, package=package, timeout=timeout)
        # extents read from the actual operands (a fused op's matmul sub-ops carry the sub-op shapes, not
        # the enclosing step's), so the per-layer log is honest for attention/geglu as well as plain matmuls.
        per_layer.append({"index": step.index, "m": int(la.shape[0]), "k": int(la.shape[1]),
                          "n": int(ra.shape[1]), "unit": step.unit,
                          "oracle": "ok" if got is not None else "unavailable"})
        return got

    base = {"target": target, "ref_target": ref_target, "ref_kind": ref_kind,
            "n_steps": len(program.steps), "n_mesh": program.n_mesh(),
            "n_scalar": program.n_scalar(), "output_id": program.output, "per_layer": per_layer,
            "simulator": simulator or os.environ.get("MERLIN_MESH_SIM", "verilator")}
    try:
        spliced = mp.run_whole_model_program(program, seed_leaves, mesh_exec=_mesh_exec)
    except mp.MeshLayerUnavailable as e:
        return {**base, "status": "oracle_unavailable", "reason": str(e)}

    spliced_final = spliced["outputs"][program.output]
    exact = bool(np.array_equal(spliced_final, ref_final))
    match = bool(np.allclose(spliced_final, ref_final, rtol=1e-4, atol=1e-4))
    return {**base, "status": "pass" if match else "fail", "exact": exact, "match": match,
            "note": "single co-scheduled whole-model run: matmul layers executed on the real mesh oracle, "
                    "scalar/vector lane inline, activations handed between lanes; gated vs the whole-model "
                    "engine reference. Residual: host-driven multi-kernel, not yet one fused address-space "
                    "image."}


def _int8_chain_reference(A0, weights: list, acc_scale: float):
    """Host int8 matmul-chain reference matching the mesh's per-layer requant EXACTLY: i32 accumulate,
    gemmini-faithful ``acc_scale`` (round-half-even of the f32 product), saturating i8 cast; each layer's
    i8 output is the next layer's activation. This is the golden the on-mesh chain is gated against."""
    import numpy as np
    x = np.asarray(A0, dtype=np.int64)
    for w in weights:
        acc = x @ np.asarray(w, dtype=np.int64)
        x = np.clip(np.rint(acc.astype(np.float32) * np.float32(acc_scale)), -128, 127).astype(np.int64)
    return x


def run_int8_chain_on_mesh(target: str, A0: list, weights: list, *, acc_scale: float,
                           operand_dtype: str = "i8", accum_dtype: str = "i32",
                           simulator: str | None = None, package: str | None = None,
                           timeout: int = 900) -> dict:
    """Run an int8 matmul CHAIN on the target mesh with the per-layer REQUANT HANDOFF — the shape of a real
    quantized model's linear stack. For each layer ``Y_l = requant_i8(A_l @ W_l)`` executes on the REAL
    oracle (``run_matmul_on_mesh`` with an ``acc_scale`` epilogue that commits the i32 accumulator back to
    i8), and ``Y_l`` becomes ``A_{l+1}`` — so the activation stays device-native i8 across the whole chain
    rather than round-tripping through a wider host dtype. The final tensor (and every layer) is gated
    bit-exact vs :func:`_int8_chain_reference`.

    This closes the int8 inter-layer handoff that a single independent matmul does not exercise: a real
    model is a CHAIN, and each mesh layer's output must be requantized to the operand dtype to feed the
    next mesh layer. FAIL-CLOSED: a layer with no reachable oracle returns ``oracle_unavailable`` (never a
    fabricated activation). TARGET-AGNOSTIC: the requant + narrow output dtype are the target's own
    (``run_matmul_on_mesh``); this only threads activations layer to layer."""
    import os

    import numpy as np

    a = np.asarray(A0, dtype=np.int64)           # device activation, threaded layer to layer
    r = np.asarray(A0, dtype=np.int64)           # independent host reference, advanced in lockstep
    per_layer: list = []
    for i, w in enumerate(weights):
        wl = np.asarray(w, dtype=np.int64)
        # advance the reference prefix (i32 matmul, gemmini-faithful acc_scale requant, i8 saturate)
        r = np.clip(np.rint((r @ wl).astype(np.float32) * np.float32(acc_scale)), -128, 127).astype(np.int64)
        out = run_matmul_on_mesh(target, a.tolist(), wl.tolist(), operand_dtype=operand_dtype,
                                 accum_dtype=accum_dtype, simulator=simulator, package=package,
                                 epilogue=["acc_scale"], acc_scale=acc_scale, timeout=timeout)
        if out is None:
            return {"target": target, "status": "oracle_unavailable", "failed_layer": i,
                    "n_layers": len(weights), "per_layer": per_layer,
                    "reason": f"mesh layer {i} ({a.shape[0]}x{a.shape[1]}x{wl.shape[1]}) has no reachable "
                              f"oracle in this env"}
        a = np.asarray(out, dtype=np.int64)
        per_layer.append({"layer": i, "m": int(a.shape[0]), "n": int(wl.shape[1]),
                          "matches_ref": bool(np.array_equal(a, r))})

    exact = bool(all(p["matches_ref"] for p in per_layer))
    return {"target": target, "status": "pass" if exact else "fail", "exact": exact,
            "n_layers": len(weights), "acc_scale": acc_scale, "per_layer": per_layer,
            "final_shape": [int(d) for d in a.shape],
            "simulator": simulator or os.environ.get("MERLIN_MESH_SIM", "verilator"),
            "note": "int8 matmul chain on the real mesh oracle with the per-layer acc_scale requant handoff "
                    "(each layer's i8 output feeds the next mesh layer); gated bit-exact vs the host int8 "
                    "chain reference at EVERY layer."}


def _summarize_route_plan(plan: dict) -> dict:
    """Collapse a routing.route_plan into per-op-family counts for the report, plus the REAL per-matmul
    extents each mesh contraction carries (threaded from the linalg by ``model_op_demands``) so the plan
    reports each layer's true M x K x N rather than only the op family. A mesh matmul whose extents could
    not be read from the linalg (``m``/``k``/``n`` None) is surfaced with None extents, never dropped."""
    def _counts(results):
        c: dict[str, int] = {}
        for r in results:
            c[r.demand.op] = c.get(r.demand.op, 0) + 1
        return c

    def _extents(results):
        out = []
        for r in results:
            d = r.demand
            if d.m is None and d.k is None and d.n is None:
                continue                                     # not a contraction / no extents attached
            out.append({"op": d.op, "site": d.site, "m": d.m, "k": d.k, "n": d.n})
        return out

    return {
        "on_mesh": _counts(plan["mesh"]),
        "in_contract_vector_scalar": _counts(plan["fallback"]),
        "scalar_rvv_lane": _counts(plan["scalar_rvv"]),
        "n_mesh_ops": len(plan["mesh"]),
        "n_scalar_ops": len(plan["fallback"]) + len(plan["scalar_rvv"]),
        "mesh_matmul_extents": _extents(plan["mesh"]),
        "note": "mesh ops execute on the target's systolic/spatial/simt unit (accelerator OOT path); the "
                "rest run on the vector/scalar (RVV) lane. Each mesh matmul carries its real MxKxN extent "
                "(mesh_matmul_extents) so a whole-model layer is compiled at its true shape. The functional "
                "gate below is the scalar/RVV whole-model reference (numerically correct across ALL ops).",
    }


def compile_model(workload: str, dtype: str, *, target: str | None, run: str, verify: bool,
                  package: str | None, auto_capture: bool, timeout: int,
                  linalg_mlir: str | None = None, mesh_verify: bool = False,
                  mesh_package: str | None = None) -> dict:
    """Target-aware whole-model compile. Routes each op across the target's compute units (matmul/systolic
    tiles -> the mesh, norms/activations/elementwise -> the vector/scalar lane) via
    ``routing.route_plan``, then compiles the functional whole model (the scalar/RVV reference, numerically
    correct across every op) and attaches the per-op mesh-routing plan. An op that no unit supports is an
    honest scalar/RVV fallback, never a silent drop. ``target=None`` degrades to the plain RVV flow.

    ``mesh_verify=True`` goes one step past the PLAN and actually EXECUTES the matmul layers on the mesh:
    for each mesh-routed matmul it synthesizes a single ``DxD`` systolic-tile ``merlin_iface`` capsule and
    runs it through the existing ``oot_runner.certify`` accelerator path on the target's real mesh oracle,
    gated bit-exact (int) / within-tolerance (float). The aggregate lands in ``out["mesh_execution"]``
    (``n_tiles``/``n_passed``/``n_unavailable``/``per_tile``); an unavailable oracle is reported honestly,
    never a fake pass. This proves the matmul layers RUN correctly on the mesh.

    WHOLE-MODEL ON MESH (``run_whole_model_on_mesh``): past per-tile certification, that entrypoint runs a
    whole model co-scheduled across the lanes on the REAL oracle — each matmul LAYER executes on the target
    mesh with its operands injected, the scalar/vector ops run inline, and every layer's on-device output
    is handed to the op that consumes it, gated bit-exact vs the whole-model engine reference. Of the three
    pieces this used to name: (a) per-op real shapes are now threaded from the module's def-use edges
    (``mesh_program_run.demands_from_module`` + ``mesh_matmul_extents``); (b) each layer compiles at its
    real MxKxN and the OOT backend tiles it. RESIDUAL: (c) it is still host-driven multi-kernel — one
    program dispatching several mesh kernels + the scalar lane, not yet ONE fused kernel in a single device
    address space. That last slice is the OOT backend emitting the whole loop nest inline."""
    # run=="mesh": execute the model's matmul layers on the target accelerator mesh (host dispatch runtime
    # with mesh routing); otherwise the plain RVV/scalar reference (host/spike/...).
    if run == "mesh":
        out = compile_rvv(workload, dtype, run="host", verify=verify, package=package,
                          auto_capture=auto_capture, timeout=timeout,
                          kernel_backend="mesh", mesh_target=target, mesh_package=mesh_package)
    else:
        out = compile_rvv(workload, dtype, run=run, verify=verify, package=package,
                          auto_capture=auto_capture, timeout=timeout)
    out["requested_target"] = target
    if target and linalg_mlir:
        try:
            from .targetgen import capsule_source as CSRC
            from .targetgen import routing as _routing
            demands = CSRC.model_op_demands(linalg_mlir, dtype)
            plan = _routing.route_plan(demands, target)
            out["routing_plan"] = _summarize_route_plan(plan)
            try:
                from .targetgen import coverage_certificate as _cert
                # ARR coverage certificate: the compiler's routing decisions (numerator) scored against
                # the target's INDEPENDENT eligibility oracle (denominator). Empty capability map (target
                # declares no semantic_capabilities yet) yields an honest all-ineligible certificate.
                out["coverage_certificate"] = _cert.for_target(plan, target)
            except Exception as e:  # noqa: BLE001 — certificate is advisory; never mask routing/functional
                out["coverage_certificate"] = {"error": f"{type(e).__name__}: {e}"}
            if mesh_verify:
                out["mesh_execution"] = _mesh_verify(plan, target=target, package=mesh_package,
                                                     timeout=timeout)
        except Exception as e:  # noqa: BLE001 — a routing failure must not mask the functional result
            out["routing_plan"] = {"error": f"{type(e).__name__}: {e}"}
    return out


def compile_oot(workload: str, *, target: str, run: str, verify: bool, package: str | None,
                timeout: int) -> dict:
    """OOT target: build the backend package and run a capsule through it, three-way gated.

    Serves any registered out-of-tree target (gemmini and beyond). ``--workload`` names a capsule
    (e.g. A2_single_tile_matmul) and ``--package`` the OOT backend. Accelerators run capsules/kernels,
    not whole VLA models."""
    from .targetgen import oot_runner, capsule_common
    from .benchharness import runs_root
    from .common.paths import repo_root

    # Default to the resolved target's conventional OOT backend package
    # (``out/artifacts/targets/<target>/agent_spec_v1_mlir_oot``, artifact_type mlir_oot_target_backend,
    # which oot_runner.certify requires) when it exists on disk — so any target that ships one resolves
    # its default. A target without that package (or with a bespoke layout) must name it via --package.
    pkg_dir = package or _default_oot_package(target)
    corpus = repo_root() / "merlin/contract/capsules/isa"
    out: dict = {"tool": "merlin-compile", "target": target, "workload": workload,
                 "package": pkg_dir, "run": run}
    if pkg_dir is None:
        out["status"] = "not_run"
        out["reason"] = f"--package required for target {target!r} (no default OOT package)"
        return out
    rr = runs_root(target, "compile")

    # compile-only: build the OOT package (board-free, needs the OOT/clang toolchain).
    pkg = oot_runner.load_package(pkg_dir)
    oot_runner.build_package(pkg, timeout=timeout)
    out["status"] = "compiled"
    if run == "none":
        return out

    sim = "spike" if run in ("spike", "k1", "run") else run  # gemmini runs on sim, not the K1 SoC
    cap_dir = corpus / workload
    if not cap_dir.is_dir():
        out["status"] = "not_run"; out["reason"] = f"no capsule {workload} under {corpus} (use a capsule name)"
        return out
    iface = cap_dir / "capsule.interface.mlir"
    res = oot_runner.certify(pkg_dir, iface, runs_root=str(rr), run_id=f"{workload}_{sim}",
                             simulator=sim, timeout=timeout)
    out["status"] = "verified" if res.get("status") == "pass" else "run_mismatch"
    out["verify"] = {"gate_ok": res.get("status") == "pass", **{k: res.get(k) for k in ("status", "cycles")}}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="merlin-compile",
        description="Compile (and optionally build/run/verify) a workload with Merlin. One command; "
                    "everything else handled in the background.")
    ap.add_argument("--workload", required=True,
                    help="rvv: a captured model name (bitvla, openvla, rdt2, …); "
                         "gemmini: a capsule name (A2_single_tile_matmul, …)")
    # --target choices = rvv (whole-model) + every registered OOT target, auto-discovered via the
    # target registry (in-tree references + MERLIN_TARGET_PATH). Registering a dialect package makes
    # `--target=<name>` work with no code change here.
    try:
        from .targetgen.target_registry import all_targets
        _oot_targets = sorted(set(all_targets()) | {"gemmini"})
    except Exception:  # noqa: BLE001 — registry unreadable → fall back to the in-tree reference
        _oot_targets = ["gemmini"]
    ap.add_argument("--target", choices=["rvv", *_oot_targets], default="rvv",
                    help="rvv (whole-model) or any registered OOT target (auto-discovered)")
    ap.add_argument("--dtype", choices=list(_RVV_DTYPES), default="fp32", help="rvv only")
    ap.add_argument("--harts", type=int, default=1,
                    help="rvv+zephyr: harts to fan the model across (>1 builds the multicore "
                         "OpenMP image; needs a matching SoC/sim)")
    ap.add_argument("--iters", type=int, default=1,
                    help="rvv: timed inference iterations (sustained mode; k1/zephyr/spike/verilator)")
    ap.add_argument("--warmup", type=int, default=0,
                    help="rvv: untimed warmup iterations before the timed ones")
    ap.add_argument("--run", choices=["none", "host", "k1", "spike", "zephyr", "verilator"],
                    default=None,
                    help="where to run after compiling (default: rvv→k1, gemmini→spike; 'none' = compile only)")
    ap.add_argument("--verify", dest="verify", action="store_true", default=True,
                    help="gate the run output vs the golden (default on)")
    ap.add_argument("--no-verify", dest="verify", action="store_false")
    ap.add_argument("--no-capture", dest="capture", action="store_false", default=True,
                    help="rvv: do NOT auto-capture a missing bundle (fail with the capture command instead)")
    ap.add_argument("--package", default=None, help="override the codegen/OOT package dir")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--json", action="store_true", help="emit the result dict as JSON")
    a = ap.parse_args(argv)

    run = a.run or ("k1" if a.target == "rvv" else "spike")
    try:
        if a.target == "rvv":
            res = compile_rvv(a.workload, a.dtype, run=run, verify=a.verify, package=a.package,
                              auto_capture=a.capture, timeout=a.timeout,
                              harts=a.harts, iters=a.iters, warmup=a.warmup)
        else:
            res = compile_oot(a.workload, target=a.target, run=run, verify=a.verify,
                              package=a.package, timeout=a.timeout)
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001 — surface any pipeline error honestly, don't fake a pass
        res = {"tool": "merlin-compile", "target": a.target, "workload": a.workload,
               "status": "error", "error": f"{type(e).__name__}: {e}"}

    if a.json:
        print(json.dumps(res, indent=2, default=str))
    else:
        print(f"\n[merlin-compile] {a.target}:{a.workload}"
              f"{':'+a.dtype if a.target=='rvv' else ''} → status={res.get('status')}"
              + (f"  gate_ok={res['verify'].get('gate_ok')}" if res.get("verify") else "")
              + (f"  reason={res.get('reason') or res.get('error')}" if res.get('reason') or res.get('error') else ""))
        for k in ("binary", "cycles", "vlen", "bundle", "package"):
            if res.get(k) is not None:
                print(f"    {k}: {res[k]}")
    return 0 if res.get("status") in ("compiled", "ran", "verified") else 1


if __name__ == "__main__":
    raise SystemExit(main())
