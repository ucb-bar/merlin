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


def ir_scalar_dtype(bundle: "Path") -> str | None:
    """The dtype the bundle's IR ACTUALLY carries, read off ``model.mlir`` — the fact that decides which
    scalar/RVV datapath is correct.

    A bundle's NAME states how the model was quantized; it does not state what the compiled IR contains.
    A weight-only fake-quant capture stores fp8/int8 weights but emits f32 tensors end to end, so the
    correct scalar package for it is the f32 one — choosing by the name instead asks for a datapath the IR
    has no operands for. Returns ``None`` when no known element type dominates, so the caller fails closed
    rather than guessing.

    Selection is by PRESENCE of the narrowest datapath, not by majority. Every weight-only capture is
    mostly f32 by count -- an int8 bundle measures 46 i8 tensors against 1190 f32 ones -- so a majority vote
    hands back f32 and silently drops the int8 datapath the i8 operands require. One tensor of a narrow type
    means the narrow datapath is needed; only a bundle with none of them is an f32 model.

    Structural, not pattern-matched: it counts occurrences of each known tensor element-type spelling.
    (No regex — see the repo's no-regex rule.)"""
    mlir = bundle / "model.mlir"
    if not mlir.is_file():
        return None
    text = mlir.read_text(encoding="utf-8", errors="replace")
    # spelling in the IR -> the --dtype token that selects its datapath, narrowest datapath first
    present = [tok for tok in _IR_ELEMENT_ORDER
               if text.count(f"x{_IR_ELEMENT_SPELLING[tok]}>")]
    return present[0] if present else None


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
#: NOTE ``fp8`` is deliberately ABSENT. A dtype only belongs here when the SCALAR/RVV lane has a
#: datapath for it, and it does not for fp8: the widening rewrite that gives narrow floats an f32
#: accumulator (``passes_xdsl.lower_bf16_matmul_f32acc``) is typed on the xDSL builtin float types,
#: which have no fp8 member at all, and every fp8 recapture's ``model.mlir`` is 100% f32 (the bundles
#: are weight-only fake-quant, so the fp8 never reaches the IR). fp8 IS a real MESH operand format --
#: it travels as ``operation.attributes.dtype`` through ``routing_dtype`` and executes on the target's
#: matrix unit -- but that is a different lane from the one this map configures. Mapping it to a
#: strategy string no package can legally declare produced "no package declares dtype_strategy='fp8'",
#: which reads as a missing artifact and sent readers off to mint a package that cannot exist.
_DTYPE_STRATEGY = {"int8": "int8_w8a8", "fp32": "fp32", "fp16": "fp16_f32acc",
                   "bf16": "bf16_f32acc"}

#: MLIR tensor element-type spelling for each ``--dtype`` token, used to read a bundle's ACTUAL IR dtype
#: back off ``model.mlir`` (:func:`ir_scalar_dtype`). Keys must stay a subset of ``_DTYPE_STRATEGY``.
_IR_ELEMENT_SPELLING = {"int8": "i8", "fp32": "f32", "fp16": "f16", "bf16": "bf16"}
#: narrowest datapath first — the first spelling PRESENT in the IR decides (see :func:`ir_scalar_dtype`).
_IR_ELEMENT_ORDER = ("int8", "fp16", "bf16", "fp32")


def default_package(dtype: str, *, bundle: "Path | None" = None) -> str:
    """The package `merlin-compile` uses when `--package` is not given.

    Resolves the CERTIFIED CHAMPION for this datatype via ``targetgen.publish.select_champion``
    rather than hard-coding a name. The previous default was ``hand_v0``/``hand_v0_int8`` — the
    FROZEN, hand-authored, UNOPTIMIZED control that exists to be the before/after baseline. Every
    default invocation therefore shipped the slowest package in the repo while the tuned ones sat
    unused. Falls back to the hand baseline only when no package of this dtype is certified, and
    says so.
    """
    from .rvvgen.tuning_agent import _DTYPE_STRATEGIES
    from .targetgen.publish import PublishError, select_champion
    strategy = _DTYPE_STRATEGY.get(dtype)
    if strategy is None and bundle is not None:
        # The requested dtype names how the model was QUANTIZED; it does not name what the compiled IR
        # carries, and only the latter decides which scalar datapath is correct. A weight-only fp8 capture
        # emits f32 tensors end to end, so the f32 package IS its datapath -- not a cross-datatype
        # substitution but the derived one. Read it off the bundle rather than refusing.
        derived = ir_scalar_dtype(bundle)
        if derived is not None and derived in _DTYPE_STRATEGY:
            print(f"[merlin-compile] --dtype {dtype} has no scalar/RVV datapath; the bundle's IR carries "
                  f"{derived}, so the scalar lane uses the {derived} package. ({dtype} remains the MESH "
                  f"operand format, routed and executed on the matrix unit.)", flush=True)
            dtype, strategy = derived, _DTYPE_STRATEGY[derived]
    if strategy is None:
        raise SystemExit(
            f"[merlin-compile] --dtype {dtype} has no scalar/RVV datapath (known: "
            f"{', '.join(sorted(_DTYPE_STRATEGY))}). If {dtype} is a MESH operand format, it belongs in "
            f"the capsule's operation.attributes.dtype (threaded as routing_dtype and executed on the "
            f"matrix unit), and the scalar lane should declare the dtype its IR actually carries. Only "
            f"add it here alongside a lowering that gives {dtype} a scalar datapath.")
    if strategy not in _DTYPE_STRATEGIES:
        # A map entry naming a strategy the knob validator rejects can never be satisfied by ANY package.
        # Diagnose it as the configuration error it is rather than as a missing artifact.
        raise SystemExit(
            f"[merlin-compile] _DTYPE_STRATEGY maps --dtype {dtype} to dtype_strategy {strategy!r}, which "
            f"is not a strategy packages may declare ({', '.join(sorted(_DTYPE_STRATEGIES))}); no package "
            f"can ever satisfy it. Fix the map, or add {strategy!r} to rvvgen.tuning_agent.")
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
                mesh_target: str | None = None) -> dict:
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
    pkg_dir = package or default_package(dtype, bundle=bundle)
    pkg = load_rvv_package(pkg_dir)
    work = Path(tempfile.mkdtemp(prefix=f"merlin_compile_{workload}_{dtype}_"))
    out: dict = {"tool": "merlin-compile", "target": "rvv", "workload": workload, "dtype": dtype,
                 "bundle": str(bundle), "package": pkg_dir, "run": run,
                 "harts": harts, "iters": iters}
    from .runtime.dispatch_runtime import mesh_datapath

    refs: dict = {}
    if verify:
        refs["fp32"] = np.load(bundle / "golden.npy")
        w8 = bundle / "golden_w8a8.npy"
        # WHO quantizes the activations decides which golden is the right yardstick, and it is not
        # always the scalar package. Routing the contractions to a mesh whose operands are 8-bit
        # quantizes them there instead, so a run with an f32 scalar package (an fp8 capture's IR is
        # f32 end to end) still computes W8A8 arithmetic. Gating this on pkg.is_int8 alone left every
        # fp8 mesh run graded against a weight-only f32 reference its datapath could never reproduce.
        _act_quant = bool(pkg.is_int8)
        if not _act_quant and kernel_backend == "mesh" and mesh_target:
            try:
                _op_dt = mesh_datapath(mesh_target).operand_dtype
                from .common import quant_formats as _qf
                _act_quant = int(_qf.get(_op_dt).element_bits or 32) <= 8
            except Exception:                        # noqa: BLE001 — undecidable: keep the f32 golden
                _act_quant = False
        if _act_quant and w8.is_file():
            refs["w8a8"] = np.load(w8)
        elif _act_quant:
            # `golden.npy` in a weight-only bundle is an f32-ACTIVATION reference; this run computes
            # W8A8. Grading one against the other measures activation-quantization error, not
            # correctness, and reads as a large cos drop. Say so rather than letting the run be
            # silently judged by the wrong yardstick.
            out["reference_warning"] = (
                f"{bundle.name} has no golden_w8a8.npy — a W8A8 run graded only against "
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
                        kernel_backend=kernel_backend, mesh_target=mesh_target)
        if kernel_backend == "mesh":
            from .runtime import dispatch_runtime as _dr
            # THIS MODEL's own layers: how many of its matmuls reached the accelerator and how many
            # fell back to the host kernel. Distinct from the synthetic tile certification below, which
            # proves that tiles OF THESE SHAPES run -- a much weaker claim about a different artifact.
            out["mesh_execution"] = {"target": mesh_target,
                                     "matmul_layers_routed": getattr(_dr.execute, "last_mesh_routed", None),
                                     "matmul_layers_on_mesh": getattr(_dr.execute, "mesh_ran", None),
                                     "matmul_layers_host_fallback": getattr(_dr.execute, "mesh_fell_back", None),
                                     # WHICH layers fell back and why -- a count alone fails the
                                     # must_accelerate gate without saying what to fix.
                                     "host_fallback_detail": getattr(_dr.execute, "mesh_fallbacks", None),
                                     # How much of what we handed the mesh its operand format could hold.
                                     # After the boundary's power-of-two scaling this should read zero
                                     # flushed and zero saturating; a nonzero count is the run telling you
                                     # the numbers below it are worth less than they look.
                                     "operand_representability": getattr(_dr.execute, "mesh_operand_repr",
                                                                         None)}
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
    # Selected by KIND, not merely by the presence of a manifest. ``out/artifacts/targets/<target>/`` also
    # holds CODEGEN packages (schedules/knobs/dialects, e.g. a hand-curated ``hand_v0``), whose manifest is
    # a different artifact entirely. Returning one of those as an OOT backend does not fail cleanly: it
    # gets as far as the package loader and dies on "manifest schema violation: 'artifact_type' is a
    # required property", which reads like a corrupt backend rather than a directory that was never one.
    for pkg_id in ("agent_spec_v1_mlir_oot", "reference_v0"):
        cand = base / pkg_id
        mf = cand / "manifest.yaml"
        if not mf.is_file():
            continue
        try:
            from .common.yaml import load_yaml
            if not str((load_yaml(mf) or {}).get("artifact_type", "")).strip():
                continue                     # not an OOT backend manifest — keep looking
        except Exception:                    # noqa: BLE001 — unreadable manifest: let the loader report it
            pass
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
    layer's accumulator back to i8 so it can feed the next mesh layer.

    The caller's pins OVERLAY the target's declared NUMERIC ``datapath`` facts; they do not replace them.
    This function used to build the dict from nothing but its own arguments, which meant every numeric
    fact the target declares and this caller does not pin — tolerances, inapplicable tiers,
    ``subnormal_operand_flush`` — silently took a dataclass default instead of the declared value. The
    corpus generator got the real block and the mesh path got defaults, from the same deriver.

    ``numeric_only`` because the profile block also holds corpus-AUTHORING choices (which oracle tiers the
    graded suite demands, its ``must_accelerate`` posture, the requant epilogue's output dtype), and this
    caller is describing hardware, not generating a corpus. Handing it those too would quietly give every
    gemmini tile an i8 requant handoff its caller never asked for."""
    from types import SimpleNamespace
    from .targetgen import corpus_spec as CS
    from .targetgen.capsule_runner import _bespoke_sim_via
    te = SimpleNamespace(target=target, sim_via=_bespoke_sim_via(target))
    datapath: dict = CS.profile_datapath(target, numeric_only=True)
    if operand_dtype:
        datapath["operand_dtype"] = operand_dtype
    if accum_dtype:
        datapath["accum_dtype"] = accum_dtype
    if requant_output_dtype:
        datapath["requant_output_dtype"] = requant_output_dtype
    return CS.derive_binding(te, datapath)


def _accum_rel_tolerance(accum_dtype: str, k: int) -> float | None:
    """Relative tolerance for a K-deep accumulation in ``accum_dtype``, DERIVED from that format's mantissa
    width in the quant-format registry rather than picked. A bf16 accumulator carries 7 mantissa bits, so a
    352-deep reduction CANNOT be bit-exact against an f32 reference and a bit-exact gate there would report
    a correct mesh as broken. An integer accumulator does not round: it gets 0.0, i.e. bit-exact.

    The bound is the format's unit roundoff ``2**-(mant_bits+1)`` grown by ``sqrt(k)`` — the random-walk
    growth of a k-deep sequential sum — with a small safety factor. CALIBRATED, not guessed: a 32x352x128
    fp8xbf16 layer measured 6.7% max relative error on the large elements against an f32 reference, versus
    the 7.3% this predicts before the safety factor. An earlier version used ``2**-mant_bits`` and a factor
    of 8, which returned 117% for that same layer — a gate that wide accepts anything.

    Returns ``None`` when the format cannot be resolved — the caller must fail closed rather than pick a
    tolerance, because both defaults are wrong in one direction (too tight condemns a good mesh, too loose
    passes a broken one)."""
    from .common import quant_formats as QF
    try:
        f = QF.get(accum_dtype)
    except KeyError:
        # Not a registry format. An MLIR integer spelling (iN) is still unambiguous: integer accumulation
        # is exact, so it gates bit-exact. Anything else is genuinely unresolved.
        body = accum_dtype[1:] if accum_dtype[:1] == "i" else ""
        return 0.0 if body.isdigit() else None
    if f.kind == "int_affine":
        return 0.0
    mant = int(f.mant_bits or 0)
    if not mant:
        return None
    return (2.0 ** -(mant + 1)) * max(1.0, float(k) ** 0.5) * 1.5


def _reference_on_datapath(A, W, binding):
    """``A @ W`` as the TARGET's datapath computes it — operands decoded the way its compute unit reads
    them, then accumulated in its declared accumulator format. The reference a mesh should be gated
    against, rather than an f32 one it was never going to reproduce.

    Takes the whole ``CorpusBinding`` rather than the accumulator dtype alone, and that is deliberate.
    Modelled on the accumulator only, this function was right about the half of the datapath it had been
    handed and silently wrong about the other half: atlas's MXU sees a signed zero wherever an operand's
    exponent field is zero (``E4M3Mul.scala``: ``aZero := aExp === 0.U``, declared as
    ``subnormal_operand_flush`` in the target's profile), and a reference that reads the operand at full
    precision grades that hardware as broken. One object carries the whole datapath, so a field added to
    it later reaches this model without a signature change.

    Two halves, both measured. A narrow-float accumulator rounds every partial sum, so an f32 reference
    disagrees with a perfectly correct device by design: a 32x352x128 fp8xbf16 layer measured 796 of 4096
    elements differing, max absolute error 22, purely from bf16 rounding — and rounding the running sum
    after each MAC reproduced that device output BIT-FOR-BIT on all 4096. On the operand side, one atlas
    capsule's 30 divergent elements were exactly its 30 subnormal codes.

    LIMIT, stated rather than implied: operands are flushed but NOT otherwise rounded into the operand
    format. Callers feed exactly-representable values (that is what makes a bit-exact gate possible), so
    rounding would be a no-op here; modelling it would mean writing a round-to-nearest-even encoder whose
    own correctness nothing checks, and a reference is only worth what its weakest step is worth.

    Returns ``None`` when the accumulator format cannot be resolved, or for an integer accumulator (exact:
    the plain product is already the right reference). Assumes sequential k-order accumulation, which is
    why the caller treats a mismatch as "not bit-exact" and falls back to the tolerance gate rather than a
    failure -- another device may reduce in a different order and still be correct."""
    import numpy as np

    from .common import quant_formats as QF
    try:
        f = QF.get(binding.accum_dtype)
    except KeyError:
        return None
    if getattr(binding, "subnormal_operand_flush", False):
        from .runtime import fp8_formats as FF
        try:
            min_normal, _max_finite = FF.normal_range(binding.operand_dtype)
        except KeyError:                                 # operand format unresolvable: fail closed
            return None
        A = np.where(np.abs(A) < min_normal, np.float32(0.0), A).astype(np.float32)
        W = np.where(np.abs(W) < min_normal, np.float32(0.0), W).astype(np.float32)
    if f.kind == "int_affine":
        return None
    mant, exp = int(f.mant_bits or 0), int(f.exp_bits or 0)
    if mant == 10 and exp == 5:                      # IEEE half
        rnd = lambda x: x.astype("<f2").astype(np.float32)          # noqa: E731
    elif mant == 7 and exp == 8:                     # bfloat16: top half of the f32 word, RNE
        def rnd(x):
            u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
            return (((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint32) << 16
                    ).astype(np.uint32).view(np.float32)
    elif mant == 23 and exp == 8:                    # f32: the product already is the reference
        return None
    else:
        return None
    acc = np.zeros((A.shape[0], W.shape[1]), dtype=np.float32)
    for i in range(A.shape[1]):
        acc = rnd(acc + np.outer(A[:, i], W[i, :]))
    return acc


def _certify_tile_via_executor(target, mlir, *, m, k, n, binding, timeout) -> dict:
    """Certify one mesh tile on a target whose endpoint is NOT the OOT-cert path, by EXECUTING it through
    the shared endpoint-aware executor (``run_matmul_on_mesh``) with injected operands and gating the
    device's output against the mathematical reference.

    ``_mesh_verify`` used to call ``oot_runner.certify`` for every target regardless of endpoint kind, so a
    self-hosted-ISA target (``external_backend``) was graded on the RoCC/OOT path it has no backend for and
    every tile came back ``oracle_unavailable`` with "no registered backend" -- while the very same target's
    whole-model run executed fine through the endpoint-aware dispatcher. The oracle choice is now made in
    ONE place for both.

    Returns a ``certify``-shaped record so the caller's loop is unchanged. Honest about strength: this is a
    TWO-way gate (reference == oracle), not the OOT path's three-way (reference == simulate == oracle)."""
    import numpy as np

    # The stimulus has to be exactly representable in the operand format -- that is what makes a bit-exact
    # gate possible at all -- but "exactly representable" does not have to mean "small integers", and while
    # it did, this gate was structurally incapable of finding the defect it was supposed to find. Small
    # integers are normal numbers in every format here, so no operand ever landed in the subnormal band,
    # so a datapath that flushes subnormals to zero certified bit-exact on all 15 synthesized tiles while
    # all 15 REAL layers of the same model diverged. The blind spot was in the seeds, not in the gate.
    #
    # A float datapath now draws from its own format's representable set instead -- a spread from the
    # smallest subnormal to near the cap, both signs, with distinct rows, distinct columns and A != A^T so
    # stride/transpose bugs stay visible (corpus_operands, the same synthesis the graded corpus uses). An
    # integer datapath keeps small integers: they ARE its range, and it has no subnormal band to miss.
    if binding.integer:
        rng = np.random.default_rng(0xA71A5)
        A = np.rint(rng.standard_normal((m, k)) * 3).clip(-8, 7).astype(np.float32)
        W = np.rint(rng.standard_normal((k, n)) * 3).clip(-8, 7).astype(np.float32)
        stimulus = "exactly-representable small integers"
    else:
        from .targetgen import corpus_operands as CO
        A = np.asarray(CO.operand_values((m, k), binding.operand_dtype, salt=0xA7),
                       dtype=np.float32).reshape(m, k)
        W = np.asarray(CO.operand_values((k, n), binding.operand_dtype, salt=0x5E),
                       dtype=np.float32).reshape(k, n)
        stimulus = (f"{binding.operand_dtype} representable spread (subnormal..near-cap, both signs), "
                    f"distinct rows/cols, asymmetric")
    obs: dict = {}
    try:
        got = run_matmul_on_mesh(target, A.tolist(), W.tolist(),
                                 operand_dtype=binding.operand_dtype, accum_dtype=binding.accum_dtype,
                                 package=None, timeout=timeout, observed=obs)
    except Exception as e:  # noqa: BLE001 — an executor failure is recorded, never a fake pass
        return {"status": "fail", "oracle": {"kind": obs.get("path"), "result": "error"},
                "failure": {"detail": f"{type(e).__name__}: {str(e)[-300:]}"}}
    if got is None:
        return {"status": "fail", "oracle": {"kind": obs.get("path"), "result": "skipped"},
                "failure": {"detail": f"no reachable mesh oracle for endpoint path {obs.get('path')!r}"}}
    ref = A @ W
    dev = np.asarray(got, dtype=np.float32)
    if dev.shape != ref.shape:
        return {"status": "fail", "oracle": {"kind": obs.get("oracle"), "result": "ran"},
                "failure": {"detail": f"device returned {dev.shape}, reference is {ref.shape}"}}
    # STRONGEST GATE FIRST: bit-exact against the target's own declared accumulator. Only if the device
    # reduces in a different order (or the format is unresolvable) do we fall back to a tolerance against
    # the f32 product, and the record says which gate carried the verdict.
    acc_ref = _reference_on_datapath(A, W, binding)
    if acc_ref is not None and np.array_equal(dev, acc_ref):
        return {"status": "pass",
                "oracle": {"kind": obs.get("oracle"), "result": "ran", "cycles": None},
                "gate": {"kind": f"bit-exact vs {binding.accum_dtype} accumulation", "rtol": 0.0,
                         "exact": True,
                         "f32_max_abs_err": float(np.abs(dev - ref).max()),
                         "stimulus": stimulus,
                         "does_not_cover": ("operand-precision error on realistic value distributions"
                                            if binding.integer else
                                            "operand values not exactly representable in "
                                            f"{binding.operand_dtype} (the format's own rounding)")}}
    rtol = _accum_rel_tolerance(binding.accum_dtype, k)
    if rtol is None:
        return {"status": "fail", "oracle": {"kind": obs.get("oracle"), "result": "ran"},
                "failure": {"detail": f"cannot derive a numeric gate for accumulator dtype "
                                      f"{binding.accum_dtype!r}: refusing to pick a tolerance"}}
    ok = (np.array_equal(dev, ref) if rtol == 0.0
          else bool(np.allclose(dev, ref, rtol=rtol, atol=rtol * float(np.abs(ref).max() or 1.0))))
    rec = {"status": "pass" if ok else "fail",
           "oracle": {"kind": obs.get("oracle"), "result": "ran", "cycles": None},
           "gate": {"kind": "reference==oracle (f32, tolerance)", "rtol": rtol,
                    "exact": bool(np.array_equal(dev, ref)),
                    "stimulus": stimulus,
                    "accum_exact": False if acc_ref is not None else None}}
    if not ok:
        rec["failure"] = {"detail": f"tile {m}x{k}x{n} diverged from the reference: "
                                    f"max abs err {float(np.abs(dev - ref).max())} (rtol {rtol})"}
    return rec


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
    from .targetgen.capsule_runner import _bespoke_sim_via, _endpoint_of, _SIM_ORACLES
    from .benchharness import runs_root

    pkg_dir = package or _default_oot_package(target)
    out: dict = {"n_tiles": 0, "n_passed": 0, "n_failed": 0, "n_unavailable": 0,
                 "n_unsynthesizable": 0, "package": pkg_dir, "per_tile": []}
    if pkg_dir is None:
        out["status"] = "not_run"
        out["reason"] = (f"no default OOT backend package for target {target!r}; pass --mesh-package "
                         f"to name one")
        return out
    # WHICH oracle certifies a tile follows the target's DERIVED endpoint, the same decision
    # ``run_matmul_on_mesh`` makes -- not an assumption that every target is the RoCC/OOT one. A
    # self-hosted-ISA target graded on the OOT path reports "no registered backend" for every tile
    # while its whole-model run executes fine, which is the same fact stated as an unavailability.
    _so = _SIM_ORACLES.get(_bespoke_sim_via(target))
    _endpoint, _ = _endpoint_of(target)
    _via_oot = not (_so is not None and _so.exclusive) and _endpoint in (
        None, "inline_asm_insn", "upstream_target")
    out["certified_via"] = "oot_cert" if _via_oot else "endpoint_executor"
    if _via_oot:
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
        # ``sim`` is only the simulator the OOT path was ASKED for; the endpoint executor runs on the
        # target's own oracle and ignores it, so record it only where it is the truth and let the
        # executor stamp ``oracle_kind`` with what actually ran.
        rec.update(M=M, K=K, N=N, layer_extent=layer_extent, n_subtiles=n_subtiles,
                   operand_dtype=binding.cap_dtype(binding.operand_dtype),
                   output_dtype=binding.cap_dtype(binding.accum_dtype))
        if _via_oot:
            rec["sim"] = sim
        if _via_oot:
            with tempfile.TemporaryDirectory(prefix="mesh_tile_") as td:
                iface = Path(td) / f"{entry['name']}.interface.mlir"
                iface.write_text(mlir, encoding="utf-8")
                res = oot_runner.certify(pkg_dir, iface, runs_root=str(rr), run_id=entry["name"],
                                         simulator=sim, target=target, timeout=timeout)
        else:
            res = _certify_tile_via_executor(target, mlir, m=M, k=K, n=N, binding=binding,
                                             timeout=timeout)
            rec["gate"] = res.get("gate")
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
    _gate_note = ("the emitted kernel's output is gated (bit-exact int / tolerance float) against the "
                  "command buffer's reference == simulate == oracle three-way"
                  if _via_oot else
                  "the device's output is gated against the mathematical reference with the operands "
                  "INJECTED (reference == oracle, a two-way gate: this endpoint has no separate simulate "
                  "leg), at a tolerance derived from the accumulator format's mantissa width")
    out["note"] = (
        "each mesh-routed matmul LAYER is verified by EXECUTING its capacity-fit tile on the target mesh "
        f"oracle ({'simulator=' + sim if _via_oot else 'via the endpoint-derived executor'}): the tile is "
        "the largest D-aligned unit whose weight+activation working "
        "set fits the target's DERIVED scratchpad capacity (n_subtiles = how many tile it into the layer); "
        f"{_gate_note}. A fit tile that certifies proves the layer runs on the "
        "mesh once tiled. The remaining gap is the single SPLICED image (all layers' mesh kernels + the "
        "scalar/RVV remainder co-scheduled in one binary with activations handed between layers) — see "
        "compile_model's docstring.")
    return out


def run_matmul_on_mesh(target: str, A: list, W: list, *, operand_dtype: str | None = None,
                       accum_dtype: str | None = None, simulator: str | None = None,
                       package: str | None = None, epilogue: list | None = None,
                       acc_scale: float | None = None, timeout: int = 900,
                       observed: dict | None = None) -> list | None:
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

    # PAD TO THE MESH TILE EDGE. A generated package is entitled to reject a sub-tile extent, and a real
    # model is full of them: every matmul layer of an 8-token sequence has M=8 against a tile edge of 32
    # (atlas) or 16 (gemmini), so the mesh refused all 15 layers and the dispatch runtime silently fell
    # back to the host kernel for every one -- a whole model that reported "on mesh" while running
    # entirely on the CPU. Zero-padding is EXACT for a contraction: the padded rows of A and columns of W
    # contribute 0 to every retained output element, and the K padding multiplies zeros against zeros.
    # The result is sliced back to the true extent, so callers see the shape they asked for.
    _D = int(binding.tile_dim or 1)

    def _up(x):
        return ((int(x) + _D - 1) // _D) * _D if _D > 1 else int(x)

    _m_true, _n_true = M, N                      # the extent the caller asked for, restored on the way out
    _Mp, _Kp, _Np = _up(M), _up(K), _up(N)
    _padded = (_Mp, _Kp, _Np) != (M, K, N)
    if _padded:
        import numpy as _np
        _A = _np.zeros((_Mp, _Kp), dtype=_np.float64)
        _A[:M, :K] = _np.asarray(A, dtype=_np.float64)
        _W = _np.zeros((_Kp, _Np), dtype=_np.float64)
        _W[:K, :N] = _np.asarray(W, dtype=_np.float64)
        A, W = _A.tolist(), _W.tolist()
        M, K, N = _Mp, _Kp, _Np
    def _build(m: int, k: int, n: int) -> str:
        e = {"name": "mesh_layer", "op": "matmul", "kind": "op",
             "source_role": "mesh_layer_real_operands",
             "source_reference": f"whole-model matmul layer {m}x{k}x{n} on the mesh with real operands",
             "M": m, "K": k, "N": n, "lhs": "A0", "weight": "W", "out": "Y0"}
        if epilogue:
            e["epilogue"] = list(epilogue)
            if acc_scale is not None:
                e["acc_scale"] = float(acc_scale)
        return CS.build(e, binding)[1]

    mlir = _build(M, K, N)

    # DISPATCH ORDER is deliberate (mirrors capsule_runner.oracle_adapters): a target whose contract
    # DECLARES an EXCLUSIVE bespoke sim (a self-hosted SIMT core graded on its own emitted kernel by its
    # own oracle, e.g. a cyclotron/muon backend) is routed FIRST — its endpoint is ALSO external_backend,
    # but the arc command-buffer program oracle grades the WRONG artifact for a SIMT kernel, so the bespoke
    # executor must take precedence. Only when no exclusive sim is declared does the endpoint kind pick the
    # path. Derived from the contract's sim_via + the _SIM_ORACLES registry — never a target-name branch.
    def _unpad(out):
        """Slice the padded mesh result back to the extent the caller asked for."""
        if out is None or not _padded:
            return out
        return [row[:_n_true] for row in out[:_m_true]]

    so = _SIM_ORACLES.get(_bespoke_sim_via(target))

    def _dispatch(_mlir, _A, _W) -> list | None:
        """One mesh call at whatever extent it is given."""
        if so is not None and so.exclusive:
            if observed is not None:
                observed["path"] = "bespoke_sim"
            return _matmul_via_bespoke_sim(target, _mlir, _A, _W, package=package, timeout=timeout,
                                           observed=observed)
        if endpoint in (None, "inline_asm_insn", "upstream_target"):
            if observed is not None:
                observed["path"] = "oot_cert"
            return _matmul_via_oot_cert(target, _mlir, _A, _W, simulator=simulator, package=package,
                                        timeout=timeout, observed=observed)
        if endpoint == "external_backend":
            if observed is not None:
                observed["path"] = "program_oracle"
            return _matmul_via_program_oracle(target, _mlir, _A, _W, model_ext=model_ext,
                                              package=package, timeout=timeout,
                                              operand_dtype=binding.operand_dtype, observed=observed)
        return None                              # no mesh-execution path derived for this endpoint kind

    # BLOCK A LAYER THAT DOES NOT FIT ON CHIP, rather than declining it. The working set of a matmul is
    # the weight tile K*N plus the activation tile M*K; past the target's scratchpad the mesh returns
    # nothing and the whole layer falls back to the host. Measured on lstmnetvit/gemmini: K and N are each
    # fine alone (1x16x512 and 1x512x16 both run) and together they are not (1x512x512 declines), so two
    # of 37 layers fell back and the model failed its must_accelerate gate at 35/37.
    #
    # `_capacity_fit_tile` already computes the fitting extent from RTL-derived facts, and `_mesh_verify`
    # already uses it to shrink the tile it CERTIFIES. Only the execution path did not, so the tile record
    # said "runs at this shape" about a shape the model never got to run. Same tiler, both paths.
    #
    # Splitting is EXACT on an integer datapath: partial products over a K split sum in the i32
    # accumulator, and an N or M split is independent columns/rows. It is NOT applied when an epilogue is
    # declared -- an acc_scale requant must see the whole accumulation, not each K block -- so such a
    # layer still declines, now with that as the stated reason.
    _cap = _scratchpad_capacity_elems(target, _dtype_bytes(binding.operand_dtype))
    _mt, _kt, _nt = M, K, N
    if _cap:
        _mt, _kt, _nt, _n_sub = _capacity_fit_tile(M, K, N, max(1, _D), _cap)
    if (_mt, _kt, _nt) != (M, K, N):
        if epilogue:
            if observed is not None:
                observed["decline"] = (f"{M}x{K}x{N} exceeds the on-chip working set ({_cap} elems) and "
                                       f"declares epilogue {list(epilogue)}; an accumulator epilogue "
                                       f"cannot be split across K blocks")
            return None
        import numpy as _np
        _An, _Wn = _np.asarray(A, dtype=_np.float64), _np.asarray(W, dtype=_np.float64)
        _acc = _np.zeros((M, N), dtype=_np.float64)
        _sub_mlir: dict[tuple[int, int, int], str] = {}
        if observed is not None:
            observed["blocked"] = {"tile": [_mt, _kt, _nt], "n_subtiles": _n_sub,
                                   "capacity_elems": _cap}
        for m0 in range(0, M, _mt):
            for n0 in range(0, N, _nt):
                for k0 in range(0, K, _kt):
                    a = _An[m0:m0 + _mt, k0:k0 + _kt]
                    w = _Wn[k0:k0 + _kt, n0:n0 + _nt]
                    shp = (a.shape[0], a.shape[1], w.shape[1])
                    if shp not in _sub_mlir:
                        _sub_mlir[shp] = _build(*shp)
                    part = _dispatch(_sub_mlir[shp], a.tolist(), w.tolist())
                    if part is None:
                        return None              # fail closed: a partial sum is not a result
                    _acc[m0:m0 + a.shape[0], n0:n0 + w.shape[1]] += _np.asarray(part, dtype=_np.float64)
        return _unpad(_acc.tolist())
    return _unpad(_dispatch(mlir, A, W))


def _matmul_via_oot_cert(target, mlir, A, W, *, simulator, package, timeout,
                         observed: dict | None = None) -> list | None:
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
    if observed is not None:
        # the cert reports its oracle as a record ({kind, derived_from_rtl, ...}); the simulator that ran
        # it is the other half of the identity, so label with both when the record is present.
        _orc = res.get("oracle")
        _kind = _orc.get("kind") if isinstance(_orc, dict) else _orc
        observed["oracle"] = f"{target}-{_kind}-{sim}" if _kind else f"{target}-{sim}"
    if res.get("status") != "pass":
        # KEEP THE REASON. Returning a bare None here is why a whole-model fallback could only be
        # described as "unsynthesizable at this shape, or the oracle was unreachable" -- a guess covering
        # two very different causes, offered because the cert's own verdict was discarded one frame down.
        if observed is not None:
            observed["decline_status"] = res.get("status")
            observed["decline"] = (res.get("failure") or {}).get("detail") or res.get("failure")
        return None
    return (res.get("oracle_outputs") or {}).get("Y0")


def _matmul_via_program_oracle(target, mlir, A, W, *, model_ext, package, timeout,
                               operand_dtype=None, observed: dict | None = None) -> list | None:
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
        timeout=timeout, dtype_hint=operand_dtype, observed=observed)


def _matmul_via_bespoke_sim(target, mlir, A, W, *, package, timeout,
                            observed: dict | None = None) -> list | None:
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
        paths = make_run_paths(runs_root(target, "mesh_bsim"), "mesh_layer", suite="mesh",
                               target=target, dtype="prog", benchmark="mesh_layer")
        try:
            _pkg, cb, kernel_text = CC.run_entrypoints(None, pkg, capsule, paths, contract=None,
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
        if observed is not None:
            observed["oracle"] = res.get("oracle") or CR._bespoke_sim_via(target)
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
        # ``obs`` records the executor that ACTUALLY ran this layer. The requested ``simulator`` is only a
        # preference and two of the three dispatch paths ignore it (a self-hosted-ISA target runs on its
        # mlc-derived cosim, an exclusive bespoke sim on its own engine) -- so reporting the request as
        # though it were the device named a simulator that never ran. Report what executed.
        obs: dict = {}
        got = run_matmul_on_mesh(target, la.tolist(), ra.tolist(),
                                 operand_dtype=operand_dtype, accum_dtype=accum_dtype,
                                 simulator=simulator, package=package, timeout=timeout, observed=obs)
        # extents read from the actual operands (a fused op's matmul sub-ops carry the sub-op shapes, not
        # the enclosing step's), so the per-layer log is honest for attention/geglu as well as plain matmuls.
        per_layer.append({"index": step.index, "m": int(la.shape[0]), "k": int(la.shape[1]),
                          "n": int(ra.shape[1]), "unit": step.unit,
                          "oracle": "ok" if got is not None else "unavailable",
                          "executed_on": obs.get("oracle"), "path": obs.get("path")})
        return got

    base = {"target": target, "ref_target": ref_target, "ref_kind": ref_kind,
            "n_steps": len(program.steps), "n_mesh": program.n_mesh(),
            "n_scalar": program.n_scalar(), "output_id": program.output, "per_layer": per_layer,
            "simulator_requested": simulator or os.environ.get("MERLIN_MESH_SIM", "verilator")}
    def _executors() -> list:
        """The DISTINCT executors the mesh layers actually ran on (empty when no layer reached one)."""
        return sorted({e for e in (lay.get("executed_on") for lay in per_layer) if e})

    try:
        spliced = mp.run_whole_model_program(program, seed_leaves, mesh_exec=_mesh_exec)
    except mp.MeshLayerUnavailable as e:
        return {**base, "status": "oracle_unavailable", "reason": str(e),
                "mesh_executors": _executors()}

    spliced_final = spliced["outputs"][program.output]
    exact = bool(np.array_equal(spliced_final, ref_final))
    match = bool(np.allclose(spliced_final, ref_final, rtol=1e-4, atol=1e-4))
    return {**base, "status": "pass" if match else "fail", "exact": exact, "match": match,
            "mesh_executors": _executors(),
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
                  mesh_package: str | None = None, routing_dtype: str | None = None) -> dict:
    """Target-aware whole-model compile. Routes each op across the target's compute units (matmul/systolic
    tiles -> the mesh, norms/activations/elementwise -> the vector/scalar lane) via
    ``routing.route_plan``, then compiles the functional whole model (the scalar/RVV reference, numerically
    correct across every op) and attaches the per-op mesh-routing plan. An op that no unit supports is an
    honest scalar/RVV fallback, never a silent drop. ``target=None`` degrades to the plain RVV flow.

    ``mesh_verify=True`` goes one step past the PLAN and certifies a SYNTHESIZED tile per mesh-routed
    matmul: a single ``DxD`` systolic-tile ``merlin_iface`` capsule run on the target's real mesh oracle,
    gated bit-exact against the declared accumulator (tolerance fallback). The aggregate lands in
    ``out["mesh_tile_verification"]`` (``n_tiles``/``n_passed``/``n_unavailable``/``per_tile``); an
    unavailable oracle is reported honestly, never a fake pass.

    READ THESE TWO KEYS AS DIFFERENT CLAIMS. ``mesh_tile_verification`` says "a tile of this layer's shape
    executes correctly on the mesh". ``out["mesh_execution"]`` says what happened to THIS MODEL: how many
    of its matmul layers the dispatch runtime actually got onto the accelerator
    (``matmul_layers_on_mesh``) versus fell back to the host kernel for (``matmul_layers_host_fallback``).
    The tile record was previously written to the same key and clobbered the model record, so a capstone
    could report "15 tiles passed" over a model that ran 0 of its 15 layers on the mesh.

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
                          kernel_backend="mesh", mesh_target=target)
    else:
        out = compile_rvv(workload, dtype, run=run, verify=verify, package=package,
                          auto_capture=auto_capture, timeout=timeout)
    out["requested_target"] = target
    if target and linalg_mlir:
        try:
            from .targetgen import capsule_source as CSRC
            from .targetgen import routing as _routing
            # Route on the EXACT registry format name, not the compile-mode token. `dtype` here is a
            # compile mode (one of _RVV_DTYPES: "int8", "fp8", ...) chosen for the RVV lowering; a
            # target declares its datapath with the precise format ("fp8_e4m3"). Feeding the compile
            # token to the router made every fp8 demand carry in_fmt="fp8" while the unit declared
            # "fp8_e4m3", so a whole model routed 0 of its 15 contractions to a mesh that supports every
            # one of them. Threading the exact name is also the SAFE fix: an "fp8" -> "fp8_e4m3" alias
            # would route e5m2 data onto an e4m3 unit, which is why the registry omits that alias.
            demands = CSRC.model_op_demands(linalg_mlir, routing_dtype or dtype)
            plan = _routing.route_plan(demands, target)
            out["routing_plan"] = _summarize_route_plan(plan)
            try:
                from .targetgen import coverage_certificate as _cert
                # ARR coverage certificate: the compiler's routing decisions (numerator) scored against
                # the target's INDEPENDENT eligibility oracle (denominator). Empty capability map (target
                # declares no semantic_capabilities yet) yields an honest all-ineligible certificate.
                #
                # The module goes in too, because both sides of that ratio are built from `demands` and a
                # contraction the matcher never matched is in neither -- it is absent, and its absence
                # raises the recall. With the module the certificate also prices what the demands missed
                # and states a recall FLOOR beside the headline figure.
                out["coverage_certificate"] = _cert.for_target(plan, target, linalg_mlir=linalg_mlir)
            except Exception as e:  # noqa: BLE001 — certificate is advisory; never mask routing/functional
                out["coverage_certificate"] = {"error": f"{type(e).__name__}: {e}"}
            if mesh_verify:
                # SEPARATE KEY. This used to overwrite out["mesh_execution"] -- the record of what
                # happened to the MODEL -- with the synthetic tile certification, so a capstone reported
                # "15 tiles passed" while the model itself ran 0 of its 15 matmul layers on the mesh and
                # fell back to the host for every one. Two different claims cannot share one key.
                out["mesh_tile_verification"] = _mesh_verify(plan, target=target, package=mesh_package,
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
