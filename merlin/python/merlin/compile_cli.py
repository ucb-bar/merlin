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


def _ensure_bundle(workload: str, dtype: str, *, auto_capture: bool) -> Path:
    """Resolve the RVV capture bundle, auto-capturing via model2MLIR if absent (and allowed)."""
    bundle = _bundle_dir(workload, dtype)
    if (bundle / "model.mlir").is_file():
        return bundle
    if not auto_capture:
        raise SystemExit(f"[merlin-compile] no bundle {bundle} (and --no-capture). Capture it with:\n"
                         f"  <model venv>/bin/python $MERLIN_M2M_DIR/workloads/capture_consistent.py "
                         f"{workload} {dtype} {bundle}")
    # Auto-capture: model2MLIR's consistent-capture worker, run inside the model's own venv. The venv +
    # per-model env live in model2MLIR; we resolve MERLIN_M2M_DIR from .env and shell to capture.py
    # (which manages the per-model venv). Honest: a fresh machine without the model's upstream venv
    # cannot capture — we surface that clearly rather than fake a bundle.
    import subprocess
    from .common.paths import env as _env
    m2m = _env("MERLIN_M2M_DIR") or _env("MERLIN_MODEL2MLIR")
    if not m2m or not (Path(m2m) / "workloads" / "capture.py").is_file():
        raise SystemExit(f"[merlin-compile] bundle {bundle} missing and MERLIN_M2M_DIR unset/invalid — "
                         f"cannot auto-capture. Set MERLIN_M2M_DIR in .env or pre-capture the bundle.")
    print(f"[merlin-compile] bundle absent → auto-capturing {workload} ({dtype}) via model2MLIR…", flush=True)
    r = subprocess.run([sys.executable, str(Path(m2m) / "workloads" / "capture.py"),
                        workload, "--formats", dtype], capture_output=True, text=True)
    if (bundle / "model.mlir").is_file():
        return bundle
    raise SystemExit(f"[merlin-compile] auto-capture did not produce {bundle}.\n{r.stdout[-800:]}\n{r.stderr[-800:]}")


#: knobs.yaml ``dtype_strategy`` for each ``--dtype``, used to pick a champion package of the
#: RIGHT datatype. An fp32 schedule applied to an int8 workload builds a silently wrong
#: datapath rather than failing, so this must never fall back across datatypes.
_DTYPE_STRATEGY = {"int8": "int8_w8a8", "fp32": "fp32", "fp16": "fp16", "fp8": "fp8"}


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
    strategy = _DTYPE_STRATEGY.get(dtype, "fp32")
    try:
        sel = select_champion("rvv", dtype_strategy=strategy)
        return str(sel.package_dir)
    except PublishError:
        fallback = "hand_v0_int8" if dtype == "int8" else "hand_v0"
        print(f"[merlin-compile] no certified {strategy} package; falling back to the frozen "
              f"baseline {fallback} (this is the UNOPTIMIZED control)", flush=True)
        from .common.artifacts import artifacts_dir
        return str(artifacts_dir() / "targets" / "rvv" / fallback)


def compile_rvv(workload: str, dtype: str, *, run: str, verify: bool, package: str | None,
                auto_capture: bool, timeout: int, harts: int = 1, iters: int = 1,
                warmup: int = 0) -> dict:
    """RVV whole-model: resolve/capture → lower → build → (run) → (gate vs golden)."""
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

        res = run_model(bundle, work, int8_compute=pkg.is_int8)
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
        b = zm.build_app(bundle, work, board=board, backend="rvv", rvv_hart=0,
                         int8_compute=pkg.is_int8, rvv_schedule=pkg.schedule_text,
                         cflags_override=pkg.cflags + zm._CFLAGS_COMMON,
                         features=frozenset(pkg.compiler_features) or None,
                         n_harts=harts, iters=iters, warmup=warmup,
                         cpus=max(2, harts))
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


def compile_oot(workload: str, *, target: str, run: str, verify: bool, package: str | None,
                timeout: int) -> dict:
    """OOT target: build the backend package and run a capsule through it, three-way gated.

    Serves any registered out-of-tree target (gemmini and beyond). ``--workload`` names a capsule
    (e.g. A2_single_tile_matmul) and ``--package`` the OOT backend. Accelerators run capsules/kernels,
    not whole VLA models."""
    from .targetgen import oot_runner, capsule_common
    from .benchharness import runs_root
    from .common.paths import repo_root

    # Default to the certified gemmini OOT package (has artifact_type: mlir_oot_target_backend, which
    # oot_runner.certify requires); any other target must name its OOT package via --package.
    default_pkg = "out/artifacts/targets/gemmini/agent_spec_v1_mlir_oot" if target == "gemmini" else None
    pkg_dir = package or default_pkg
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
