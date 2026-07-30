"""aet recording substrate for Gemmini conformance runs.

Records each run (one conformance rung on one oracle) as an aet run directory:
a manifest (with oracle provenance + toolchain SHAs), metrics/events via EvalRunLogger,
content-addressed artifacts (command buffer / generated kernel / console) tagged by origin
(who produced them — GENERATED / COMPILER_GENERATED / ORACLE_OUTPUT), and a FailureRecord on
mismatch. This is the anti-blur ledger: every artifact and every cycle number is attributable.

`record_gemmini_run` is the core (used by tests and the conformance dispatcher); a `result`
may be injected to record without running a simulator (fast, deterministic test of the
recording layer itself).
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from aet.core.artifact_store import ArtifactOrigin, ArtifactStore
from aet.core.failures import FailureCategory, FailureRecord
from aet.core.run_paths import RunPaths
from aet.core.run_spec import RunSpec
from aet.tracking import EvalRunLogger

from merlin.targetgen.eval.gemmini_conformance import build
from merlin.runtime.backends import gemmini
from merlin.runtime.backends.gemmini_codegen import generate_driver

SUITE = "gemmini-conformance"
# This is the gemmini reference conformance suite (it drives the gemmini runtime backend + codegen). Its
# target identity is declared ONCE here — co-located with SUITE — and referenced by the run record +
# logger below, rather than repeated as bare "gemmini" literals.
TARGET = "gemmini"


def _git_sha(path: str) -> str:
    try:
        return subprocess.run(["git", "-C", path, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:  # pragma: no cover
        return "unknown"


def toolchain_shas() -> dict[str, str]:
    cy = str(gemmini.chipyard_root())
    return {
        "merlin": _git_sha("."),
        "chipyard": _git_sha(cy),
        "gemmini": _git_sha(cy + "/generators/gemmini"),
    }


CODEGEN_BACKENDS = ("legacy_c", "mlir_inline_asm_rocc")


def record_gemmini_run(rung: str, simulator: str, *, runs_root: str | Path,
                       codegen_backend: str = "mlir_inline_asm_rocc",
                       result: dict | None = None, seed: int = 0,
                       timeout: int = 600) -> dict[str, Any]:
    """Run (or accept an injected ``result``) for one rung+oracle+backend and record an aet run dir.

    ``codegen_backend`` selects the codegen (NOT the oracle): ``mlir_inline_asm`` is the
    MLIR-faithful path (compute via merlin's compiler, custom instrs as ``.insn`` RoCC);
    ``legacy_c`` is the libgemmini C prototype. Returns a summary dict.
    """
    if codegen_backend not in CODEGEN_BACKENDS:
        raise ValueError(f"unknown codegen_backend {codegen_backend!r}; have {CODEGEN_BACKENDS}")
    cb = build(rung)
    run_id = f"{rung}_{simulator}_{codegen_backend}_seed{seed:03d}"
    shas = toolchain_shas()
    spec = RunSpec(project="merlin", suite=SUITE, method=f"{rung}_{simulator}_{codegen_backend}",
                   seed=seed, run_id=run_id, project_root=Path(runs_root), tracking_mode="local",
                   target=TARGET, dtype="i8xi8_i32", benchmark=rung,
                   repo_initial_commit=shas["merlin"])
    paths = RunPaths.from_spec(spec, run_id)
    for d in (paths.run_path, paths.logs, paths.artifacts_dir, paths.generated, paths.contracts):
        d.mkdir(parents=True, exist_ok=True)

    # Run the selected codegen path (the MLIR path writes its artifacts into paths.generated).
    if result is None:
        if codegen_backend == "mlir_inline_asm_rocc":
            from merlin.runtime.backends import gemmini_codegen_mlir as gmlir
            result = gmlir.run_on_spike(cb, workdir=paths.generated, simulator=simulator,
                                        timeout=timeout)
        else:
            result = gemmini.run_command_buffer(cb, simulator=simulator, timeout=timeout)

    oracle = dict(result.get("oracle", gemmini.ORACLE.get(simulator, {})))
    metrics = dict(result.get("metrics", {}))
    metrics.setdefault("cycle_source", "rdcycle" if metrics.get("cycles") else "unknown")
    metrics.setdefault("cycle_window", "gemmini_region")
    metrics.setdefault("memory_model", "functional_model" if simulator == "spike" else "unknown")
    cycle_accurate = simulator == "verilator"      # spike is functional, NOT cycle-accurate
    correct = bool(result.get("correct"))

    # Manifest (oracle provenance + SHAs in metadata) — the certification record.
    manifest = {
        "schema_version": "1.0", "project": "merlin", "suite": SUITE, "method": spec.method,
        "seed": seed, "run_id": run_id, "target": TARGET, "benchmark": rung,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "status": "pass" if correct else "fail",
        "codegen_backend": codegen_backend,
        "metadata": {
            "oracle": oracle, "toolchain_shas": shas,
            "codegen_backend": codegen_backend,
            "cycles": metrics.get("cycles"),
            "cycle_source": metrics.get("cycle_source"),
            "cycle_window": metrics.get("cycle_window"),
            "memory_model": metrics.get("memory_model"),
            "cycle_accurate": cycle_accurate,
        },
    }
    (paths.run_path / "run_manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    logger = EvalRunLogger.start(project="merlin", suite=SUITE, target=TARGET,
                                 method=spec.method, seed=seed, run_id=run_id,
                                 run_path=paths.run_path, tracking_mode="local")
    logger.log_params({"rung": rung, "simulator": simulator, "codegen_backend": codegen_backend,
                       "oracle_kind": oracle.get("kind"),
                       "derived_from_rtl": oracle.get("derived_from_rtl"),
                       "cycle_accurate": cycle_accurate,
                       **{f"sha.{k}": v for k, v in shas.items()}})
    logger.log_metrics({"correct": int(correct), "cycles": int(metrics.get("cycles", 0))})
    logger.log_event("gemmini.run", {"rung": rung, "simulator": simulator,
                                     "codegen_backend": codegen_backend,
                                     "correct": correct, "oracle": oracle})

    # Artifacts, tagged by who produced them (the anti-blur ledger).
    store = ArtifactStore(paths.run_path, run_id)
    cb_path = paths.generated / "command_buffer.json"
    cb_path.write_text(json.dumps(cb, indent=2), encoding="utf-8")
    store.record(cb_path, ArtifactOrigin.GENERATED, kind="command_buffer")
    if codegen_backend == "mlir_inline_asm_rocc":
        # The MLIR-faithful compiler artifacts: emitted MLIR is THE recorded kernel; the LLVM
        # IR / object / harness (already materialized in paths.generated by run_on_spike) too.
        from merlin.runtime.backends import gemmini_codegen_mlir as gmlir
        try:
            mlir_path = paths.generated / "kernel.mlir"
            mlir_path.write_text(gmlir.emit_kernel_mlir(cb)[0], encoding="utf-8")
            store.record(mlir_path, ArtifactOrigin.COMPILER_GENERATED, kind="mlir")
        except Exception:
            pass
        for fname, kind in (("gemmini_kernel.ll", "llvm_ir"),
                            ("gemmini_kernel.o", "object"), ("harness.c", "harness")):
            p = paths.generated / fname
            if p.exists():
                store.record(p, ArtifactOrigin.COMPILER_GENERATED, kind=kind)
    else:
        kernel_path = paths.generated / "main.c"
        try:
            kernel_path.write_text(generate_driver(cb), encoding="utf-8")
            store.record(kernel_path, ArtifactOrigin.COMPILER_GENERATED, kind="kernel")
        except Exception:  # codegen may not support a rung yet; record what we have
            pass
    if result.get("console") is not None:
        console_path = paths.artifacts_dir / "console.log"
        console_path.write_text(result["console"], encoding="utf-8")
        store.record(console_path, ArtifactOrigin.ORACLE_OUTPUT, kind="log")

    # FailureRecord on mismatch (the repair-diff seed), routed by oracle level → responsible plane.
    if not correct:
        if simulator == "spike":
            plane = "codegen_or_spike_invocation"   # L0 already passed (else we'd not be here)
            candidates = "MLIR/RoCC encoding, operand packing, spike invocation/flags"
        else:                                        # L1 passed, L2 fails: stricter-RTL interaction
            plane = "runtime_kernel_rtl_plane"
            candidates = ("config ordering, fences, block_stride, mvin/mvout layout, accumulator "
                          "addressing, stationary transpose, RoCC state, alignment, sim invocation")
        fr = FailureRecord(
            category=FailureCategory.FUNCTIONAL_MISMATCH,
            detail=f"{rung} on {simulator} ({codegen_backend}): oracle output != reference. "
                   f"Likely causes: {candidates}",
            failure_id=f"{run_id}-mismatch", likely_cause=plane,
            observed=str(result.get("outputs")), artifact_refs=[str(cb_path)])
        (paths.logs / "failures.jsonl").write_text(
            json.dumps(dataclasses.asdict(fr), default=str) + "\n", encoding="utf-8")

    logger.finish(status="pass" if correct else "fail")
    return {"run_id": run_id, "run_path": str(paths.run_path), "rung": rung,
            "simulator": simulator, "codegen_backend": codegen_backend,
            "cycle_accurate": cycle_accurate, "correct": correct,
            "oracle": oracle, "metrics": metrics}
