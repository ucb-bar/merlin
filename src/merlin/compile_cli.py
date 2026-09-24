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
(``llvmlower.lower``, ``mining.k1``, ``mining.registry``, ``runtime.backends.zephyr_model``,
``targetgen.oot_runner``) — it adds no new compile logic.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

# Everything below `main`'s own steps lives in `merlin.compile` and is re-exported here, so
# `merlin.compile_cli.<name>` keeps resolving for callers. A re-export is a second binding: patching it
# does not reach callers in the defining module, so tests patch a name where it is DEFINED
# (enforced by merlin/tests/infra/test_compile_cli_patch_targets.py).
from .compile.bundles import (  # noqa: F401 -- re-exported
    _IR_ELEMENT_ORDER,
    _IR_ELEMENT_SPELLING,
    _bundle_dir,
    _capture_python,
    ir_scalar_dtype,
)
from .compile.capacity import (  # noqa: F401 -- re-exported
    _accumulator_capacity_elems,
    _capacity_fit_tile,
    _dtype_bits,
    _dtype_bytes,
    _operand_store_bytes,
    _operand_store_capacity_elems,
    capacity_fit,
    declared_primitive_tile,
)
from .compile.host_lane import (  # noqa: F401 -- re-exported
    _DTYPE_STRATEGY,
    _verified_against_the_pinned_lane,
    default_package,
    host_lane_identity,
    host_lane_pin_name,
)
from .compile.mesh import (  # noqa: F401 -- re-exported
    _MESH_NTILE_WIDTH,
    _certify_tile_via_executor,
    _default_oot_package,
    _matmul_via_bespoke_sim,
    _matmul_via_oot_cert,
    _matmul_via_program_oracle,
    _mesh_rows,
    _mesh_tile_binding,
    _mesh_verify,
    run_matmul_on_mesh,
    tile_builder_op,
)
from .compile.mesh_backend import (  # noqa: F401 -- re-exported
    _MESH_PKG_CACHE,
    _MESH_PKG_LOCK,
    _MESH_REFUSAL,
    _MESH_RUN_SEQ,
    _built_mesh_package,
    _mesh_invocation_id,
    _mesh_layer_id,
    _refuse,
    _requested_mesh_simulator,
    _resolve_oot_mesh_simulator,
)
from .compile.mesh_model import (  # noqa: F401 -- re-exported
    _int8_chain_reference,
    run_int8_chain_on_mesh,
    run_whole_model_on_mesh,
)
from .compile.mesh_reference import (  # noqa: F401 -- re-exported
    _accum_rel_tolerance,
    _reference_on_datapath,
)

# Workloads that ship as model2MLIR capture bundles (RVV whole-model path). Not exhaustive — any
# workloads/<name> with a loader can be captured; this is the "known-good" convenience set for --list.
_RVV_DTYPES = ("fp32", "int8", "fp16", "fp8")


def _ensure_bundle(workload: str, dtype: str, *, auto_capture: bool) -> Path:
    """Resolve the RVV capture bundle, auto-capturing via model2MLIR if absent (and allowed)."""
    bundle = _bundle_dir(workload, dtype)
    if (bundle / "model.mlir").is_file():
        return bundle
    if not auto_capture:
        raise SystemExit(
            f"[merlin-compile] no bundle {bundle} (and --no-capture). Capture it with:\n"
            f"  <model venv>/bin/python $MERLIN_M2M_DIR/workloads/capture_consistent.py "
            f"{workload} {dtype} {bundle}"
        )
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
        raise SystemExit(
            f"[merlin-compile] bundle {bundle} missing and MERLIN_M2M_DIR unset/invalid — "
            f"cannot auto-capture. Set MERLIN_M2M_DIR in .env or pre-capture the bundle."
        )
    py = _capture_python(Path(m2m), workload)
    print(
        f"[merlin-compile] bundle absent → auto-capturing {workload} ({dtype}) via {worker.name} in {py}…", flush=True
    )
    r = subprocess.run(
        [str(py), str(worker), workload, dtype, str(bundle)], capture_output=True, text=True, cwd=str(m2m)
    )
    if (bundle / "model.mlir").is_file():
        return bundle
    raise SystemExit(f"[merlin-compile] auto-capture did not produce {bundle}.\n{r.stdout[-800:]}\n{r.stderr[-800:]}")


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
        from .mining.apply import blocking_risks, shape_adapted_features

        # A package whose block lives in its SCHEDULE TEXT (no feature) cannot be re-resolved. Say so
        # rather than letting the resulting ~34x fp32 degradation read as "this model is slow".
        for risk in blocking_risks(pkg, bundle):
            out.setdefault("blocking_risks", []).append(risk)
            print(f"[merlin-compile] WARNING: {risk}", file=sys.stderr, flush=True)
        feats = shape_adapted_features(pkg, bundle, harts=harts)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[merlin-compile] shape adaptation unavailable ({type(exc).__name__}: {exc}); "
            f"using the package block as pinned",
            file=sys.stderr,
        )
        return frozen
    if sorted(feats) != sorted(frozen):
        # A re-derived point may leave an op class UNCLAIMED: no block wider than one lane is legal
        # for its extents, so its contractions go through convert-linalg-to-loops (scalar). Correct,
        # but a perf fact a reader must not have to decode from a feature name -- so say it.
        from .llvmlower.impr_features import unclaimed_op_classes

        unclaimed = sorted({cls for f in feats for cls in unclaimed_op_classes(f)})
        out["features_shape_adapted"] = {
            "pinned": frozen,
            "used": feats,
            **({"unclaimed_op_classes": unclaimed} if unclaimed else {}),
        }
        print(
            f"[merlin-compile] the package register block does not lower for this workload's "
            f"contraction extents; re-derived per op class: {frozen} -> {feats}",
            flush=True,
        )
        if unclaimed:
            print(
                f"[merlin-compile] NOT VECTORIZED (no multi-lane block is legal for their "
                f"extents, so they run scalar): {', '.join(unclaimed)}",
                file=sys.stderr,
                flush=True,
            )
    return feats


def _session_correctness_gate(trajectory: object, expected_steps: int) -> dict:
    """Gate compiled output against the same-precision eager trajectory.

    This is distinct from the paper's FP32 model-quality gate. The fixed thresholds catch codegen
    faults while allowing ordinary floating-point reassociation in the lowered implementation.
    """
    value = dict(trajectory or {}) if isinstance(trajectory, dict) else {}
    steps = int(value.get("steps") or 0)
    cosine = value.get("min_cosine")
    relative = value.get("max_relative_error")
    top1 = int(value.get("top1_matches") or 0)
    ok = (
        steps == int(expected_steps)
        and cosine is not None
        and float(cosine) >= 0.999
        and relative is not None
        and float(relative) <= 0.01
        and top1 == steps
    )
    return {
        "gate_ok": ok,
        "scope": value.get("scope"),
        "steps": steps,
        "min_cosine": cosine,
        "max_relative_error": relative,
        "top1_matches": top1,
        "top1_agreement": value.get("top1_agreement"),
        "reference": "eager_same_precision",
        "thresholds": {"cosine_min": 0.999, "max_relative_error": 0.01, "top1_agreement": 1.0},
    }


def compile_rvv(
    workload: str,
    dtype: str,
    *,
    run: str,
    verify: bool,
    package: str | None,
    auto_capture: bool,
    timeout: int,
    harts: int = 1,
    iters: int = 1,
    warmup: int = 0,
    session_repeats: int | None = None,
    kernel_backend: str | None = None,
    fallback_policy: str = "allow",
    mesh_target: str | None = None,
    mesh_package: str | None = None,
    numeric_policy: dict | None = None,
    bundle_path: str | Path | None = None,
    capture_bundle: str | Path | None = None,
    deadline_ns: int | None = None,
) -> dict:
    """RVV whole-model: resolve/capture → lower → build → (run) → (gate vs golden).

    ``kernel_backend='mesh'`` + ``mesh_target`` runs the model's matmul LAYERS on that target's
    accelerator mesh (host dispatch runtime, each matmul injected onto the mesh oracle)."""
    import os

    import numpy as np

    from .mining import k1
    from .mining.registry import load_rvv_package
    from .runtime.backends import zephyr_model as zm

    # ``timeout`` is the budget for the complete subprocess session.  The generated harness owns
    # observations, warmups, and measurement repeats *inside that one invocation*.  Multiplying
    # this value here double-counts the session dimensions (32 observations x 5 repeats used to
    # turn a four-hour paper-cell budget into several weeks) and defeats fail-closed scheduling.
    timeout = int(timeout)
    if timeout < 1:
        raise ValueError("timeout must be a positive whole-session budget in seconds")

    # TWO ways a caller pins the capture rather than letting the mutable registry resolve the workload
    # name again -- with DIFFERENT completeness contracts, which is why they stay separate parameters
    # instead of one that quietly means two things:
    #   `capture_bundle` -- a capsule grade's short-lived, read-only bundle whose content identity the
    #       grader already bound. It must be COMPLETE; a partial or symlinked directory is rejected
    #       before any compiler can interpret it.
    #   `bundle_path`    -- a frozen paper study's content-addressed capture, which may legitimately be
    #       a version-2 multi-program session carrying no top-level `model.mlir`.
    if bundle_path is not None and capture_bundle is not None:
        raise ValueError(
            "pass either bundle_path or capture_bundle, never both: they pin the same "
            "bundle under different completeness contracts"
        )
    if capture_bundle is not None:
        lexical = Path(capture_bundle)
        if lexical.is_symlink() or not lexical.is_dir():
            raise SystemExit("[merlin-compile] explicit capture bundle is missing or a symlink")
        bundle = lexical.resolve(strict=True)
        required = (
            "model.mlir",
            "weights.safetensors",
            "weights.safetensors.manifest.json",
            "inputs.npz",
            "input_order.json",
            "golden.npy",
        )
        unsafe = [name for name in required if not (bundle / name).is_file() or (bundle / name).is_symlink()]
        if unsafe:
            raise SystemExit(f"[merlin-compile] explicit capture bundle is incomplete/unsafe: {unsafe}")
    else:
        bundle = (
            Path(bundle_path).resolve()
            if bundle_path is not None
            else _ensure_bundle(workload, dtype, auto_capture=auto_capture)
        )
    # The session contract is read for EVERY bundle, however it was resolved: whether a capture is a
    # version-2 multi-program session is a property of the bundle, not of which caller named it.
    multi_program = False
    session_contract_version = 0
    session_path = bundle / "session_contract.yaml"
    if session_path.is_file():
        from .common.yaml import load_yaml

        session_value = load_yaml(session_path)
        session_contract_version = int(session_value.get("version", 0)) if isinstance(session_value, dict) else 0
        multi_program = session_contract_version == 2
    if not (bundle / "model.mlir").is_file() and not multi_program:
        raise FileNotFoundError(f"explicit capture bundle has neither model.mlir nor a version-2 session: {bundle}")
    pkg_dir = package or default_package(dtype, bundle=bundle)
    pkg = load_rvv_package(pkg_dir)
    work = Path(tempfile.mkdtemp(prefix=f"merlin_compile_{workload}_{dtype}_"))
    out: dict = {
        "tool": "merlin-compile",
        "target": "rvv",
        "workload": workload,
        "dtype": dtype,
        "bundle": str(bundle),
        "package": pkg_dir,
        "run": run,
        "harts": harts,
        "iters": iters,
        "session_repeats": session_repeats,
        # WHICH HOST COMPILER THIS RESULT WAS BUILT WITH, under the same key and subkeys the
        # grading path records, so a graded capsule and an ordinary compile of the same model
        # produce comparable records. Without it a result built against an unpinned lane was
        # indistinguishable afterwards from one built against the pinned lane.
        "host_lane": host_lane_identity(pkg_dir),
    }
    from .runtime.dispatch_runtime import mesh_datapath

    refs: dict = {}
    if verify and not multi_program:
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
                _op_dt = mesh_datapath(mesh_target, numeric_policy=numeric_policy).operand_dtype
                from .common import quant_formats as _qf

                _act_quant = int(_qf.get(_op_dt).element_bits or 32) <= 8
            except Exception:  # noqa: BLE001 — undecidable: keep the f32 golden
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
                f"divergence, NOT evidence of a defect. Generate the W8A8 reference first."
            )
            print(f"[merlin-compile] WARNING: {out['reference_warning']}", file=sys.stderr)

    # `host` runs the model through the x86 dispatch runtime (per-kernel JIT via the SAME
    # MLIR lowering, minus the RVV/cross-compile tail). It needs no board, simulator or
    # cross-toolchain, so it is the first correctness stage on a fresh machine — and the
    # discriminator when a board run disagrees: host-vs-board separates a quantization-math
    # bug (both wrong) from a codegen/RVV bug (only the board wrong).
    if run == "host":
        from .runtime.dispatch_runtime import run_model

        res = run_model(
            bundle,
            work,
            int8_compute=pkg.is_int8,
            kernel_backend=kernel_backend,
            mesh_target=mesh_target,
            mesh_package=mesh_package,
            numeric_policy=numeric_policy,
        )
        if kernel_backend == "mesh":
            # THIS MODEL's own layers: how many of its matmuls reached the accelerator and how many
            # fell back to the host kernel. Distinct from the synthetic tile certification below, which
            # proves that tiles OF THESE SHAPES run -- a much weaker claim about a different artifact.
            #
            # Read the counters off THIS run's result. They used to be function attributes on a
            # module-global (`_dr.execute.mesh_ran`), which concurrent grades clobber -- and a model
            # verdict now depends on them. `UNKNOWN` rather than None when absent: None reads as "zero
            # layers ran", when what it means is "nobody could tell".
            from .common.provenance import UNKNOWN as _UNKNOWN

            out["mesh_execution"] = {
                "target": mesh_target,
                # Trusted formal grading overwrites MERLIN_MESH_SIM from its selected/pinned L3 engine.
                # Carry the request beside each call's observed engine so the ledger checker can reject
                # a fallback or a mixed-engine result rather than inferring provenance from fidelity.
                "simulator_requested": (
                    os.environ.get("MERLIN_MESH_SIM") or os.environ.get("MERLIN_REQUIRED_RTL_ENGINE")
                ),
                "matmul_layers_routed": res.get("mesh_routed", _UNKNOWN),
                "matmul_layers_on_mesh": res.get("mesh_ran", _UNKNOWN),
                "matmul_layers_host_fallback": res.get("mesh_fell_back", _UNKNOWN),
                # coverage, NOT a gate: matmuls the classifier never routed (e.g. batched attention
                # generics), so a row can say "15 of 19 layers on the mesh" instead of implying 19.
                "matmul_layers_unrouted": res.get("mesh_unrouted_matmuls", _UNKNOWN),
                # which layers the mesh could not take, by extent -- a count alone cannot distinguish
                # "one shape the backend refuses" from "four unrelated capability gaps".
                "fallback_shapes": res.get("mesh_fallback_shapes", []),
                # WHICH layers fell back and why -- a count alone fails the must_accelerate gate without
                # saying what to fix.
                "host_fallback_detail": res.get("mesh_fallbacks"),
                # Layers the ORACLE could not measure (timed-out / unreachable simulator). Counted
                # apart from a fallback because it is not evidence about the backend: the mesh may
                # well run these.
                "matmul_layers_oracle_unavailable": res.get("mesh_unavailable", _UNKNOWN),
                "oracle_unavailable_detail": res.get("mesh_unavailable_detail"),
                # Ordered, runner-owned proof of which lane EACH dynamic call actually used.  Static
                # routing is deliberately separate: it cannot prove that a host island or mesh call ran.
                "mesh_route_symbols": res.get("mesh_route_symbols"),
                "dispatch_ledger": res.get("dispatch_ledger"),
                # Layers whose capacity_fit obligation the RUNTIME discharged for the backend; non-empty
                # means this result is evidence about runtime+backend together, not about the backend
                # alone. Each entry names the tiler that chose its extent (`tile_source`).
                "capacity_fit_delegated_to_runtime": res.get("mesh_capacity_fit_delegated"),
                # How much of what we handed the mesh its operand format could hold. After the boundary's
                # power-of-two scaling this should read zero flushed and zero saturating; a nonzero count
                # is the run telling you the numbers below it are worth less than they look.
                "operand_representability": res.get("mesh_operand_repr"),
            }
        # THE HOST LANE'S OWN EXECUTION RECORD, emitted whatever the kernel backend, because a
        # host-only capsule never enters the mesh branch above and would otherwise have no evidence at
        # all. Same `_UNKNOWN` discipline: absent means "nobody could tell", not "nothing ran".
        #
        # `contractions_ran` is reported apart from `kernels_ran` because the host lane is populated
        # overwhelmingly by norms and activations -- a bare kernel count is satisfied by any model and
        # says nothing about whether real work crossed to the host.
        from .common.provenance import UNKNOWN as _UNKNOWN_HOST

        out["host_execution"] = {
            "kernels_ran": res.get("host_kernels_ran", _UNKNOWN_HOST),
            "contractions_ran": res.get("host_contractions_ran", _UNKNOWN_HOST),
            "host_lane": out.get("host_lane"),
        }
        out["status"] = "ran"
        out["n_kernels"] = res.get("n_kernels")
        if refs:
            outputs = list(res.get("outputs") or [res["output"]])
            multi_path = bundle / "goldens.npz"
            order_path = bundle / "output_order.json"
            if len(outputs) > 1 and multi_path.is_file() and order_path.is_file():
                names = [str(x) for x in json.loads(order_path.read_text(encoding="utf-8"))]
                archive = np.load(multi_path, allow_pickle=False)
                if len(names) != len(outputs) or set(names) != set(archive.files):
                    raise ValueError("multi-output golden manifest disagrees with runtime results")
                per_output = []
                for name, value in zip(names, outputs, strict=True):
                    check = zm._gate(value, {"fp32": archive[name]})
                    per_output.append({"name": name, **check})
                g = {
                    "ok": all(bool(x.get("ok")) for x in per_output),
                    "per_output": per_output,
                    "fp32_cos": min(float(x.get("fp32_cos", 0.0)) for x in per_output),
                    "fp32_rel": max(float(x.get("fp32_rel", float("inf"))) for x in per_output),
                }
            else:
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
            return zm.build_app(
                bundle,
                work,
                board=board,
                backend="rvv",
                rvv_hart=0,
                int8_compute=pkg.is_int8,
                rvv_schedule=pkg.schedule_text,
                cflags_override=pkg.cflags + zm._CFLAGS_COMMON,
                features=frozenset(fs) or None,
                n_harts=harts,
                iters=iters,
                warmup=warmup,
                cpus=max(2, harts),
            )

        try:
            b = _build(feats)
        except Exception as exc:  # noqa: BLE001
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
            print(
                f"[merlin-compile] the register block does not lower once each matmul is split "
                f"across {harts} harts; re-deriving against the per-hart tile: "
                f"{feats} -> {narrowed}",
                file=sys.stderr,
                flush=True,
            )
            out["harts_split_block_retry"] = {"tried": list(feats), "used": list(narrowed)}
            b = _build(narrowed)
            feats = narrowed
        out["binary"] = str(b["elf"])
        out["status"] = "compiled"
        if run == "verilator":
            sim = zm.verilator_sim()
            if sim is None:
                out["status"] = "not_run"
                out["reason"] = "no multicore Saturn Verilator sim built (see docs/guides/tinyllama_int8_rvv_zephyr.md)"
                return out
            res = zm.run_on_verilator(b["elf"], timeout=timeout, references=refs or None)
        else:
            res = zm.run_on_spike(b["elf"], harts=max(2, harts), mem_bytes=b["ram_bytes"], timeout=timeout)
            if refs:
                res.update(zm._gate(res["prefix"], refs))
        out["status"] = "ran"
        out["cycles"] = res.get("metrics", {}).get("cycles")
        if res.get("sustained"):
            out["sustained"] = res["sustained"]
            out["iter_cycles"] = list(res.get("iter_cycles", ()))
        if res.get("sustained_wall_ns"):
            out["sustained_wall_ns"] = res["sustained_wall_ns"]
            out["iter_wall_ns"] = list(res.get("iter_wall_ns", ()))
        if refs:
            out["verify"] = {
                "gate_ok": bool(res.get("ok")),
                **{k: v for k, v in res.items() if k.endswith(("_cos", "_rel", "_max_rel"))},
            }
            out["status"] = "verified" if res.get("ok") else "run_mismatch"
        return out

    # K1 / compile-only. `run_on_k1` does its own build (into <work>/v), so building here too
    # would compile the whole model TWICE — for TinyLlama int8 that is ~40 min of clang thrown
    # away, enough to push the run past its own timeout. Build directly only when nothing else
    # will.
    if run != "k1":
        # `--harts N` reaches THIS build too. It used not to: the flag was plumbed only to the
        # run-on-hardware routes, so `--run none --harts 8` produced a single-core binary and
        # reported success -- a compile-only multicore A/B silently compared two identical
        # single-core images. Same expression as the `--run k1` call below, so the two agree.
        binary = k1.build_k1_binary(
            bundle, work, pkg, inputs_npz=bundle / "inputs.npz", parallel_harts=(harts if harts > 1 else None)
        )
        out["binary"] = str(binary)
        out["parallel_harts"] = harts if harts > 1 else None
    out["status"] = "compiled"
    if run == "none":
        return out
    if run == "k1":
        board_available = k1.available(deadline_ns=deadline_ns) if deadline_ns is not None else k1.available()
        if not board_available:
            out["status"] = "not_run"
            out["reason"] = "K1 board unreachable (see MERLIN_K1_HOST/port 2222)"
            return out
        res = k1.run_on_k1(
            bundle,
            work,
            pkg,
            timeout=timeout,
            iters=iters,
            warmup=warmup,
            deadline_ns=deadline_ns,
            session_repeats=session_repeats,
            kernel_backend=kernel_backend,
            parallel_harts=(harts if harts > 1 else None),
            fallback_policy=fallback_policy,
            require_csr_vlen=(fallback_policy == "forbid"),
        )
        out["binary"] = res.get("local_binary")
        out["status"] = "ran"
        out["cycles"] = res.get("metrics", {}).get("cycles")
        out["wall_ns"] = res.get("metrics", {}).get("wall_ns")
        out["vlen"] = res.get("vlen")
        out["vlen_source"] = res.get("vlen_source")
        out["peak_rss_bytes"] = (res.get("metrics", {}).get("peak_rss_kb") or 0) * 1024 or None
        out["memory_policy"] = res.get("memory_policy")
        out["board_conditions"] = res.get("board_conditions")
        out["trajectory_quality"] = res.get("trajectory_quality")
        out["trajectory_correctness"] = res.get("trajectory_correctness")
        out["stage_wall_ns"] = res.get("stage_wall_ns")
        routed_key = {
            "xnnpack": "n_xnn_routed",
            "openblas": "n_openblas_routed",
            "ours": "n_ours_routed",
            "outlined_int8": "n_outlined_int8_routed",
        }.get(kernel_backend or "")
        eligible_key = {"xnnpack": "n_xnn_eligible", "openblas": "n_openblas_eligible"}.get(kernel_backend or "")
        candidates_key = {"xnnpack": "n_xnn_candidates", "openblas": "n_openblas_candidates"}.get(kernel_backend or "")
        out["execution"] = {
            "mode": res.get("execution_mode"),
            "requested_mode": res.get("requested_execution_mode"),
            "fallback_used": res.get("fallback_used"),
            "core_count": res.get("core_count"),
            "requested_core_count": res.get("requested_core_count"),
            "affinity_source": res.get("affinity_source"),
            # These describe what the generated harness actually executed, not merely what the
            # paper spec requested.  A capture session advances its stream/carried state within a
            # complete semantic session; a non-session sustained benchmark repeats one input.
            "semantic_session": session_contract_version == 2,
            "same_input_repetition": (
                session_contract_version != 2 and (session_repeats is not None or int(iters) > 1 or int(warmup) > 0)
            ),
            "kernel_backend": kernel_backend,
            "n_routed": res.get(routed_key) if routed_key else None,
            "n_eligible": res.get(eligible_key) if eligible_key else None,
            "n_candidates": res.get(candidates_key) if candidates_key else None,
        }
        if res.get("sustained"):
            out["sustained"] = res["sustained"]
            out["iter_cycles"] = list(res.get("iter_cycles", ()))
        if res.get("sustained_wall_ns"):
            out["sustained_wall_ns"] = res["sustained_wall_ns"]
            out["iter_wall_ns"] = list(res.get("iter_wall_ns", ()))
        if verify:
            if res.get("trajectory_correctness") is not None:
                correctness = dict(res["trajectory_correctness"])
                g = _session_correctness_gate(correctness, int(iters))
            elif multi_program:
                g = _session_correctness_gate(None, 1)
            else:
                g0 = zm._gate(res["prefix"], refs)
                g = {"gate_ok": bool(g0.get("ok")), **g0}
            out["verify"] = g
            out["status"] = "verified" if out["verify"]["gate_ok"] else "run_mismatch"
        return out
    out["status"] = "not_run"
    out["reason"] = f"run mode {run!r} not supported for rvv (use none|k1|spike|zephyr|verilator)"
    return out


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
                continue  # not a contraction / no extents attached
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


def compile_model(
    workload: str,
    dtype: str,
    *,
    target: str | None,
    run: str,
    verify: bool,
    package: str | None,
    auto_capture: bool,
    timeout: int,
    linalg_mlir: str | None = None,
    mesh_verify: bool = False,
    mesh_package: str | None = None,
    numeric_policy: dict | None = None,
    routing_dtype: str | None = None,
    capture_bundle: str | Path | None = None,
) -> dict:
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
        out = compile_rvv(
            workload,
            dtype,
            run="host",
            verify=verify,
            package=package,
            auto_capture=auto_capture,
            timeout=timeout,
            kernel_backend="mesh",
            mesh_target=target,
            mesh_package=mesh_package,
            numeric_policy=numeric_policy,
            capture_bundle=capture_bundle,
        )
    else:
        out = compile_rvv(
            workload,
            dtype,
            run=run,
            verify=verify,
            package=package,
            auto_capture=auto_capture,
            timeout=timeout,
            capture_bundle=capture_bundle,
        )
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
            # THE PLACEMENT DECIDES; THE ROUTER IS ITS LEGALITY ORACLE AND ITS CROSS-CHECK.
            #
            # `place` used to run in shadow while `route_plan` decided, because the two surfaces had to
            # be shown equal on real models before either could be trusted with the decision. They now
            # are: over the 22 whole-model captures under `recaptures/` crossed with the four targets
            # that resolve a contract, all 88 pairs agree lane-for-lane AND accumulate-token-for-
            # accumulate-token. That is what makes the flip a change of authority rather than a change
            # of behaviour -- they share `_legal_on`, so agreement is structural, not coincidental.
            #
            # What the flip buys is the thing routing cannot express. `route_plan` partitions by whether
            # a DEVICE unit was legal, so the host is an absence: an op nobody could take and an op the
            # host was chosen for land in the same bucket with the same silence. A Placement decides over
            # the host's units too, so every op carries a device, a unit, a lane and a reason -- and
            # `emulated` (an op the host took whose format the host cannot natively carry, so the
            # lowering has to emulate it) becomes statable for the first time.
            #
            # The cost model is passed rather than defaulted. `measured_cost_for` returns None on every
            # target today because no unit's contract records a measured rate, so pricing is inert here
            # and placement stays declaration-order -- but a rate landing in a contract now changes the
            # decision without changing this call, which is the whole point of `select` taking a cost.
            #
            # FAIL SOFT, AND SAY SO: if the system cannot be derived, the router decides and the record
            # names it as the authority. A compile is the wrong place to discover a modelling gap.
            _shadow = _routing.route_plan(demands, target)
            plan, _authority = _shadow, "routing.route_plan"
            try:
                from .system.derive import system_for_experiment as _sysfor
                from .system.place import measured_cost_for as _cost_for
                from .system.place import place as _place

                _system, _host_why = _sysfor(target)
                _placement = _place(demands, _system, cost=_cost_for(_system))
                _proj = _placement.as_route_plan()
                _divergence = {
                    k: {"placement": len(_proj[k]), "route_plan": len(_shadow[k])}
                    for k in ("mesh", "fallback", "scalar_rvv")
                    if len(_proj[k]) != len(_shadow[k])
                }
                plan = _proj
                _authority = "system.place (routing.route_plan is the cross-check)"
                out["placement"] = {
                    **_placement.to_dict(),
                    "host": _host_why,
                    "authority": _authority,
                    "divergence": _divergence or None,
                }
            except Exception as _exc:  # noqa: BLE001 -- a modelling gap must not fail a compile
                out["placement"] = {
                    "status": "unavailable",
                    "authority": "routing.route_plan (placement unavailable)",
                    "why": f"{type(_exc).__name__}: {_exc}",
                }
            out["routing_plan"] = {**_summarize_route_plan(plan), "authority": _authority}
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
                # ...and the execution record when this run produced one, so a certificate built from
                # the PLAN carries the run that either bore it out or did not, instead of being quoted
                # as though the two were the same fact.
                out["coverage_certificate"] = _cert.for_target(
                    plan, target, linalg_mlir=linalg_mlir, execution=out.get("mesh_execution")
                )
            except Exception as e:  # noqa: BLE001 — certificate is advisory; never mask routing/functional
                out["coverage_certificate"] = {"error": f"{type(e).__name__}: {e}"}
            try:
                # THE POPULATION IS THE MODEL, NOT THE DEMANDS. Placement above decides over the
                # contraction demands; a requantize, a clamp, a residual add and a pool never become
                # demands, so they are not "placed on the host", they are absent -- and they are
                # where a whole model spends its time. The census walks every region of the module
                # and gives each a unit, or a refusal with its owner, or names it a silent fallback.
                from .common import mlir_query as _mq
                from .perf import placement_census as _census

                # AT THE DATAPATH THE ROUTE WAS DECIDED AT. The demands above carry
                # `routing_dtype or dtype` for the reason `model_op_demands` states: a capture is
                # routed under the datapath the compiler will lower it to. The census used to read
                # element types off the same capture instead, so its denominator answered a
                # different question than its numerator -- on a dynamically quantized ResNet-50 it
                # refused all 53 matmuls for `input_dtype fp32` while the route placed 54/54, and
                # reported offload_of_eligible 1.0 over the one region that carried an i8 type.
                out["placement_census"] = _census.census_of_module(
                    _mq.parse(linalg_mlir), target, datapath=routing_dtype or dtype
                )
            except Exception as e:  # noqa: BLE001 -- the census reports on a compile, it never fails one
                out["placement_census"] = {"error": f"{type(e).__name__}: {e}"}
            else:
                # A census that could not be TAKEN never fails a compile; one that was taken and found
                # work the target could run with none of it on a unit does. That outcome used to be
                # `compiled`, exit 0, and a program that spends its whole life on the host.
                try:
                    _census.require_offload(out["placement_census"])
                except _census.ZeroOffloadError as refusal:
                    out["offload_refusal"] = str(refusal)
                    if out.get("status") in ("compiled", "ran", "verified"):
                        out["status"], out["reason"] = "zero_offload", str(refusal)
            try:
                # THE DECISION, NOT THE AUDIT. The census above asks each region on its own; this
                # forms the groups a unit actually takes -- a contraction with the stages its
                # readout absorbs -- and every other operation lands in a host region that carries
                # the refusal that put it there. It is what an emission should be built from.
                from .xdsl_dialects.lowering import compute_groups as _groups

                _plan = _groups.plan(_mq.parse(linalg_mlir), target)
                out["compute_groups"] = {
                    "schema": _plan["schema"],
                    "summary": _plan["summary"],
                    "absorption_refusals": _plan["absorption_refusals"],
                    "groups": _plan["groups"],
                }
            except Exception as e:  # noqa: BLE001 -- a planning gap must not fail a compile
                out["compute_groups"] = {"error": f"{type(e).__name__}: {e}"}
            try:
                # WHAT LIES BETWEEN THE GROUPS: buffers that cross the host/device boundary, work
                # that is constant and need not run per inference, the waits the program truly
                # owes, and host work that could run while the device does.
                from .xdsl_dialects.lowering import compute_groups as _groups
                from .xdsl_dialects.lowering import stream_plan as _stream
                from .xdsl_dialects.lowering.dispatch_program import build_dispatch_program
                from .xdsl_dialects.lowering.outline import outline_dispatches

                _module = _mq.parse(linalg_mlir)
                _outlined = outline_dispatches(_module, groups=_groups.form_groups(_module, target))
                out["stream_plan"] = _stream.plan(build_dispatch_program(_outlined), _outlined.dispatches)
            except Exception as e:  # noqa: BLE001 -- analysis only; never fails a compile
                out["stream_plan"] = {"error": f"{type(e).__name__}: {e}"}
            try:
                # WHAT THE MODEL ASKS OF A BACKEND, in the form a unit runs it: each accelerator
                # group restated as a device program (stored tensor on the right, a gathered
                # convolution as the unit's own, the stages in readout order, the real multiplier).
                # These are the entries the capsule generator builds from, so the layer a model
                # needs and the capsule that certifies it are one statement. A group that cannot be
                # restated is counted with its reason.
                from .targetgen import group_capsule_entries as _group_capsules
                from .xdsl_dialects.lowering import stream_plan as _stream_plan

                _stated = _group_capsules.entries(
                    target,
                    _mq.parse(linalg_mlir),
                    weight_args=_stream_plan.weight_args_beside(linalg_mlir),
                    with_raw=False,
                )
                out["group_programs"] = {
                    key: _stated[key]
                    for key in ("schema", "accelerator_groups", "stated", "distinct", "unstated", "entries")
                }
            except Exception as e:  # noqa: BLE001 -- analysis only; never fails a compile
                out["group_programs"] = {"error": f"{type(e).__name__}: {e}"}
            if mesh_verify:
                # SEPARATE KEY. This used to overwrite out["mesh_execution"] -- the record of what
                # happened to the MODEL -- with the synthetic tile certification, so a capstone reported
                # "15 tiles passed" while the model itself ran 0 of its 15 matmul layers on the mesh and
                # fell back to the host for every one. Two different claims cannot share one key.
                out["mesh_tile_verification"] = _mesh_verify(
                    plan, target=target, package=mesh_package, timeout=timeout, numeric_policy=numeric_policy
                )
        except Exception as e:  # noqa: BLE001 — a routing failure must not mask the functional result
            out["routing_plan"] = {"error": f"{type(e).__name__}: {e}"}
    return out


def compile_oot(workload: str, *, target: str, run: str, verify: bool, package: str | None, timeout: int) -> dict:
    """OOT target: build the backend package and run a capsule through it, three-way gated.

    Serves any registered out-of-tree target (gemmini and beyond). ``--workload`` names a capsule
    (e.g. A2_single_tile_matmul) and ``--package`` the OOT backend. Accelerators run capsules/kernels,
    not whole VLA models."""
    from .common.paths import repo_root, runs_root
    from .targetgen import oot_runner

    corpus = repo_root() / "merlin/contract/capsules/isa"
    out: dict = {"tool": "merlin-compile", "target": target, "workload": workload, "package": package, "run": run}

    # compile-only: build the OOT package (board-free, needs the OOT/clang toolchain).
    # VALIDATE THE WORKLOAD BEFORE ANYTHING REPORTS SUCCESS. This check used to live below the
    # `run == "none"` early return, so a compile-only invocation never read `workload` at all: it
    # built the BACKEND PACKAGE, wrote `status: compiled`, and returned. MEASURED 2026-09-10:
    # `--workload definitely_not_a_real_workload_xyz --target gemmini --run none` reported
    # `compiled`, and so did `--workload tiny_llama`, which is a whole MODEL and never was a capsule
    # (see this function's own docstring). A caller reading that status believed a model had been
    # compiled for gemmini when nothing of the kind had happened.
    cap_dir = corpus / workload
    if not cap_dir.is_dir():
        out["status"] = "not_run"
        out["reason"] = (
            f"no capsule {workload!r} under {corpus} (use a capsule name). This target "
            "command compiles capsules, not target-native whole models. "
            "Use --model-preflight with an explicit capture bundle and deployment dtype "
            "to inspect model readiness; preflight does not compile a target binary."
        )
        return out

    # Resolve the backend only after checking that this interface can compile the
    # requested workload. A missing package must not conceal a whole-model request
    # behind a generic package error.
    pkg_dir = package or _default_oot_package(target)
    out["package"] = pkg_dir
    if pkg_dir is None:
        out["status"] = "not_run"
        out["reason"] = f"--package required for target {target!r} (no default OOT package)"
        return out
    rr = runs_root(target, "compile")

    pkg = oot_runner.load_package(pkg_dir)
    oot_runner.build_package(pkg, timeout=timeout)
    out["status"] = "compiled"
    if run == "none":
        return out

    sim = "spike" if run in ("spike", "k1", "run") else run  # gemmini runs on sim, not the K1 SoC
    iface = cap_dir / "capsule.interface.mlir"
    res = oot_runner.certify(
        pkg_dir, iface, runs_root=str(rr), run_id=f"{workload}_{sim}", simulator=sim, timeout=timeout
    )
    out["status"] = "verified" if res.get("status") == "pass" else "run_mismatch"
    out["verify"] = {"gate_ok": res.get("status") == "pass", **{k: res.get(k) for k in ("status", "cycles")}}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="merlin-compile",
        description="Compile RVV models or OOT capsules; inspect OOT model readiness without claiming a binary.",
    )
    ap.add_argument(
        "--workload",
        required=False,
        help="rvv: a captured model name (bitvla, openvla, rdt2, …); "
        "an OOT target: a capsule name (A2_single_tile_matmul, …); "
        "not needed for --model-preflight (which selects an explicit bundle)",
    )
    # --target choices = rvv (whole-model) + every registered OOT target, auto-discovered via the
    # target registry (in-tree references + MERLIN_TARGET_PATH). Registering a dialect package makes
    # `--target=<name>` work with no code change here.
    try:
        from .targetgen.target_registry import all_targets

        _oot_targets = sorted(all_targets())
    except Exception as exc:  # noqa: BLE001 — an unreadable registry offers no OOT target, and says so
        print(
            f"[merlin-compile] target registry unreadable ({type(exc).__name__}: {exc}); only "
            f"--target rvv is available",
            file=sys.stderr,
        )
        _oot_targets = []
    ap.add_argument(
        "--target",
        choices=["rvv", *_oot_targets],
        default="rvv",
        help="rvv (whole-model) or any registered OOT target (auto-discovered)",
    )
    ap.add_argument("--dtype", choices=list(_RVV_DTYPES), default="fp32", help="rvv only")
    ap.add_argument(
        "--harts",
        type=int,
        default=1,
        help="rvv+zephyr: harts to fan the model across (>1 builds the multicore "
        "OpenMP image; needs a matching SoC/sim)",
    )
    ap.add_argument(
        "--iters",
        type=int,
        default=1,
        help="rvv: timed inference iterations (sustained mode; k1/zephyr/spike/verilator)",
    )
    ap.add_argument("--warmup", type=int, default=0, help="rvv: untimed warmup iterations before the timed ones")
    ap.add_argument(
        "--run",
        choices=["none", "host", "k1", "spike", "zephyr", "verilator"],
        default=None,
        help="where to run after compiling (default: rvv→k1, an OOT target→spike; 'none' = compile only)",
    )
    ap.add_argument(
        "--verify",
        dest="verify",
        action="store_true",
        default=True,
        help="gate the run output vs the golden (default on)",
    )
    ap.add_argument("--no-verify", dest="verify", action="store_false")
    ap.add_argument(
        "--no-capture",
        dest="capture",
        action="store_false",
        default=True,
        help="rvv: do NOT auto-capture a missing bundle (fail with the capture command instead)",
    )
    ap.add_argument("--package", default=None, help="override the codegen/OOT package dir")
    ap.add_argument(
        "--model-preflight",
        action="store_true",
        help="read-only OOT model analysis: compare declared routes with groups in a captured program",
    )
    ap.add_argument("--capture-bundle", help="explicit model2MLIR capture directory for --model-preflight")
    ap.add_argument(
        "--deployment-dtype",
        help="exact target operand format for --model-preflight (e.g. int8, bf16, fp8_e4m3)",
    )
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--json", action="store_true", help="emit the result dict as JSON")
    a = ap.parse_args(argv)

    if a.model_preflight and (a.target == "rvv" or not a.capture_bundle or not a.deployment_dtype):
        ap.error("--model-preflight requires an OOT --target, --capture-bundle, and --deployment-dtype")
    if not a.model_preflight and not a.workload:
        ap.error("--workload is required for compilation")
    if a.model_preflight:
        # The explicit bundle selects the model. An optional --workload supplied
        # out of habit must not become a second, possibly conflicting selector.
        a.workload = Path(a.capture_bundle).name
    run = a.run or ("k1" if a.target == "rvv" else "spike")
    try:
        if a.model_preflight:
            from .compile.model_preflight import preflight_model

            res = preflight_model(a.capture_bundle, target=a.target, deployment_dtype=a.deployment_dtype)
        elif a.target == "rvv":
            res = compile_rvv(
                a.workload,
                a.dtype,
                run=run,
                verify=a.verify,
                package=a.package,
                auto_capture=a.capture,
                timeout=a.timeout,
                harts=a.harts,
                iters=a.iters,
                warmup=a.warmup,
            )
        else:
            res = compile_oot(
                a.workload, target=a.target, run=run, verify=a.verify, package=a.package, timeout=a.timeout
            )
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001 — surface any pipeline error honestly, don't fake a pass
        res = {
            "tool": "merlin-compile",
            "target": a.target,
            "workload": a.workload,
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
        }

    if a.json:
        print(json.dumps(res, indent=2, default=str))
    else:
        print(
            f"\n[merlin-compile] {a.target}:{a.workload}"
            f"{':' + a.dtype if a.target == 'rvv' else ''} → status={res.get('status')}"
            + (f"  gate_ok={res['verify'].get('gate_ok')}" if res.get("verify") else "")
            + (f"  reason={res.get('reason') or res.get('error')}" if res.get("reason") or res.get("error") else "")
        )
        for k in ("binary", "cycles", "vlen", "bundle", "package"):
            if res.get(k) is not None:
                print(f"    {k}: {res[k]}")
    return 0 if res.get("status") in ("compiled", "ran", "verified") else 1


if __name__ == "__main__":
    raise SystemExit(main())
