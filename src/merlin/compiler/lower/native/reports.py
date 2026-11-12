"""Check completed native reports and bind original artifacts to the recipe."""

from __future__ import annotations

import subprocess
from pathlib import Path

from merlin.compiler.lower.model import fusion_guard
from merlin.compiler.lower.model.masked_contraction import FEATURE as _MASKED_FEATURE
from merlin.compiler.lower.model.source_observation_stage import FEATURE as _SOURCE_OBSERVATION_FEATURE
from merlin.compiler.lower.native.arguments import RunnerCommand
from merlin.compiler.lower.native.invoke import _harvest_native
from merlin.compiler.lower.native.policy import SINK_DEALLOC_PASS
from merlin.compiler.lower.native.recipe import LoweringRecipe
from merlin.compiler.lower.native.stages import DEALLOC_CHECK_TOKEN
from merlin.compiler.passes.contract import PipelineError


def check_reports(
    proc: subprocess.CompletedProcess[str],
    *,
    work: Path,
    mlir_text: str,
    feats: frozenset[str],
    arguments: RunnerCommand,
    recipe: LoweringRecipe,
    lowering_selection: dict | None,
    audit,
    retention,
    stage_out: Path,
    omp: bool,
    traced,
    runner: Path,
    borrowed_buffer_calls=None,
    scalar_stage: Path | None = None,
    source_observation_effects=None,
):
    checked = None
    from merlin.compiler.lower.model.automatic_host_features import FEATURE as automatic_host_feature

    if automatic_host_feature in feats:
        from merlin.common.jsonio import write_pretty_json
        from merlin.compiler.lower.model.automatic_host_features import require_report

        automatic_report = require_report(proc.stdout, features=feats)
        report_path = work / "automatic-host-optimizations.json"
        write_pretty_json(report_path, automatic_report)
        recipe.bind_source("automatic_host_optimizations", report_path)
        if lowering_selection is not None:
            lowering_selection["automatic_host_optimizations"] = automatic_report
    if audit is not None:
        audit.collect_views()
    _harvest_native(traced, stage_out)
    if borrowed_buffer_calls is not None:
        from merlin.compiler.lower.model.borrowed_buffer_calls import verify_native

        checked = verify_native(borrowed_buffer_calls, mlir_text, work, native_stdout=proc.stdout, native_runner=runner)
        from merlin.compiler.lower.model.borrowed_buffer_calls import CHECKPOINT, REPORT

        recipe.bind_source("borrowed_lifetime_checkpoint", work / CHECKPOINT)
        recipe.bind_source("borrowed_lifetime_report", work / REPORT)
        if lowering_selection is not None:
            lowering_selection["borrowed_buffer_lifetime"] = checked
    if scalar_stage is not None:
        import hashlib as _hashlib
        import json as _json

        source_digest = _hashlib.sha256((scalar_stage / "current.mlir").read_bytes()).hexdigest()
        response_digest = _hashlib.sha256((scalar_stage / "response.json").read_bytes()).hexdigest()
        expected = f"OK current_scalar_leaf_stage {source_digest} {response_digest}"
        if [line for line in proc.stdout.splitlines() if line.startswith("OK current_scalar_leaf_stage")] != [expected]:
            raise PipelineError("native scalar rewrite completion is absent or inconsistent")
        for name in ("current.mlir", "request.json", "response.json", "binding.json"):
            recipe.bind_source("scalar_carrier_" + name.replace(".", "_"), scalar_stage / name)
        binding = _json.loads((scalar_stage / "binding.json").read_text())
        binding.update(status="CURRENT_TYPED_EDITS_NATIVE_VERIFIED", native_response_sha256=response_digest)
        if lowering_selection is not None:
            lowering_selection["source_scalar_carrier"] = binding
    if _SOURCE_OBSERVATION_FEATURE in feats:
        from merlin.compiler.lower.model.source_observation_stage import CHECKPOINT
        from merlin.compiler.lower.model.source_observation_stage import (
            require_report as _require_source_observation_report,
        )

        observed_source = _require_source_observation_report(proc.stdout, work, effects=source_observation_effects)
        recipe.bind_source("source_observation", work / CHECKPOINT)
        if lowering_selection is not None:
            lowering_selection["source_observation"] = observed_source
    if audit is not None:
        if retention is not None:
            audit.stage("llvm-dialect", retention.module.read_text(encoding="utf-8"))
        audit.stage(
            "llvm-dialect" if omp else "llvm-translated",
            stage_out.read_text(encoding="utf-8"),
            format="mlir" if omp else "llvm-ir",
        )
    if _MASKED_FEATURE in feats:
        from merlin.compiler.lower.model.masked_contraction import require_report as _require_masked_report

        try:
            _require_masked_report(proc.stdout, work)
        except ValueError as exc:
            raise PipelineError(str(exc) + f"\n{proc.stdout}") from exc
    if "specialize_contiguous_copy" in feats:
        from merlin.compiler.lower.model.contiguous_suffix_copy import require_report as _require_contiguous_copy_report

        try:
            _require_contiguous_copy_report(proc.stdout, work)
        except ValueError as exc:
            raise PipelineError(str(exc) + f"\n{proc.stdout}") from exc
    if "fold_uniform_fill_copy" in feats:
        from merlin.compiler.lower.model.uniform_fill_copy import require_report as _require_uniform_fill_report

        try:
            _require_uniform_fill_report(proc.stdout, work)
        except ValueError as exc:
            raise PipelineError(str(exc) + f"\n{proc.stdout}") from exc
    if arguments.fold_broadcast == "1":
        from merlin.compiler.lower.model.broadcast_fold import require_report as _require_broadcast_fold_report

        try:
            _folded = _require_broadcast_fold_report(proc.stdout, work)
            print(f"[broadcast-fold] folded {_folded} materialized broadcasts")
        except ValueError as exc:
            raise PipelineError(str(exc) + f"\n{proc.stdout}") from exc
    if arguments.named_broadcast == "1":
        from merlin.compiler.lower.model.named_broadcast_fold import require_report as _require_named_broadcast_report

        try:
            _named = _require_named_broadcast_report(proc.stdout, work)
            print(f"[named-broadcast-fold] {_named}")
        except ValueError as exc:
            raise PipelineError(str(exc) + f"\n{proc.stdout}") from exc
    if arguments.alloca_scope == "1":
        from merlin.compiler.lower.model.alloca_scope_lower import require_report as _require_alloca_scope_report

        try:
            _lowered_scopes = _require_alloca_scope_report(proc.stdout, work)
            print(f"[alloca-scope] lowered {_lowered_scopes} structured scope(s) before CFG")
        except ValueError as exc:
            raise PipelineError(str(exc) + f"\n{proc.stdout}") from exc
    if arguments.panel_parallel == "1":
        from merlin.compiler.lower.model.panel_parallel import require_complete_report as _require_panel_report

        try:
            _require_panel_report(proc.stdout, work)
        except ValueError as exc:
            raise PipelineError(str(exc) + f"\n{proc.stdout}") from exc
    if arguments.coarsen == "1":
        from merlin.compiler.lower.model.parallel_coarsen import require_report as _require_coarsen_report

        try:
            _require_coarsen_report(proc.stdout)
        except ValueError as exc:
            raise PipelineError(str(exc) + f"\n{proc.stdout}") from exc
    if arguments.int_softmax == "1":
        from merlin.compiler.lower.model.int_softmax_table import require_report as _require_int_softmax_report

        try:
            _ist = _require_int_softmax_report(proc.stdout, work)
        except ValueError as exc:
            raise PipelineError(str(exc) + f"\n{proc.stdout}") from exc
        print(
            f"[int-softmax-table] {_ist['softmax']} softmax(es), {_ist['int32_sums']} int32 row sum(s), "
            f"{_ist['row_quantizations']} per-row quantization(s), {_ist['scales_moved']} scale(s) moved"
            + (f"; left alone: {'; '.join(_ist['refused'])}" if _ist["refused"] else "")
        )
    if (
        fusion_guard.FUSE_PASS in arguments.pipeline
        and fusion_guard.enabled()
        and fusion_guard.TOKEN not in proc.stdout
    ):
        # The pass ran and the broadcast control function did not: a runner that reached the native
        # PassManager without the guard would silently re-evaluate per-row values per element.
        raise PipelineError(
            f"elementwise fusion ran without its broadcast guard ({fusion_guard.TOKEN!r} absent from "
            f"the runner's output)\n{proc.stdout}"
        )
    if SINK_DEALLOC_PASS in arguments.pipeline and DEALLOC_CHECK_TOKEN not in proc.stdout:
        # The sinking stage ran and the use-after-free check did NOT. That is only reachable from a
        # runner variant that drives the PassManager itself instead of going through `_run_stages`
        # (the accumulator-resident v3 runner does, for its first stage), and it would sink every
        # free with nothing verifying the placement -- silently wrong numbers, not a failed build.
        # Refuse the artifact rather than ship an unchecked one.
        raise PipelineError(
            "the dealloc sinking stage ran but its use-after-free check did not report "
            f"({DEALLOC_CHECK_TOKEN!r} absent from the runner's output). This runner variant "
            "bypasses `_run_stages`; do not combine MERLIN_SINK_DEALLOCS with it until the check "
            f"is wired there too.\n{proc.stdout}"
        )
    return checked
