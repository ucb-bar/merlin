"""Original checked-source L2 diagnostic consumer, separate from final grading."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from types import MethodType

from merlin_experiments.phase0 import connected_integer_qualification as SOURCE

from merlin.common.cpu_pool import shared_compute_pool
from merlin.targetgen import native_model_execution as NATIVE
from merlin.targetgen.checked_source_model_diagnostic import CheckedSourceModelDiagnostic


def execute_native_model(**kwargs):
    """Use the original whole-program route unless the live host opted into V2.

    Only sealed original-child restoration can recreate a selected diagnostic.
    A command-buffer field or a saved successful report cannot choose this path.
    Both routes retain the independent complete source/effects check in finish.
    """
    from .checked_source_model_diagnostic import (
        POINTER_PROTOCOL,
        active_checked_source_diagnostic,
    )

    selection = active_checked_source_diagnostic()
    if selection is None:
        return NATIVE.execute_candidate_model(**kwargs)
    if type(selection) is not CheckedSourceModelDiagnostic:
        raise ValueError("native source execution requires its original live selector")
    selection.revalidate()
    row = selection.record()
    if row.get("native_execution_protocol") != POINTER_PROTOCOL:
        return NATIVE.execute_candidate_model(**kwargs)
    from .native_pointer_source_execution import execute_candidate_pointer_source

    try:
        return execute_candidate_pointer_source(**kwargs)
    finally:
        selection.revalidate()


def grade_from_context(runner, capsule, *, target, timeout, package_dir, context, pkg):
    from . import capsule_runner as original
    from .checked_source_model_diagnostic import selection_from_context

    if runner is not original:
        raise ValueError("checked source diagnostic requires its original runner")
    selection = selection_from_context(
        context["checked_source_diagnostic"], capsule=capsule, target=target, package_dir=package_dir)
    return runner._grade_public_program_capsule_inline(
        capsule, target=target, timeout=timeout, package_dir=package_dir, context=context,
        pkg=pkg, diagnostic_selection=selection)


def finish_unavailable(selection, result, capsule, *, target, deadline):
    return finish(selection, result, capsule, target=target, native_result=None,
                  work_root=None, remaining_s=max(0, deadline - time.monotonic()))


def execution_kwargs(selection, capsule, *, target):
    if type(selection) is not CheckedSourceModelDiagnostic:
        raise ValueError("checked source execution needs the actual original host selector")
    row = selection.verify_for(capsule, target=target)
    return dict(simulator="spike", oracle_tier_ceiling="L2", complete_call_timing=False,
                rtl_facts=row["rtl_facts"]["path"], board_config=row["board_config"])


def _live_realization(owner):
    for name in ("revalidate", "source_rule", "verify_source_execution"):
        callback = getattr(owner, name, None)
        if type(callback) is not MethodType or callback.__self__ is not owner:
            raise ValueError("checked source diagnostic needs actual original bound source/effect methods")
    owner.revalidate()


def _audit_execution(selection, emission, certificate, native, *, target, entry_symbol):
    from .checked_source_model_diagnostic import POINTER_PROTOCOL

    if selection.record().get("native_execution_protocol") == POINTER_PROTOCOL:
        from .native_pointer_source_audit import audit_pointer_source_tiers

        return audit_pointer_source_tiers(
            emission, native, target=target, entry_symbol=entry_symbol,
        )
    return NATIVE.audit_candidate_tiers(
        emission, certificate, native, target=target, entry_symbol=entry_symbol,
        completed_dispatch=None, oracle_tier_ceiling="L2",
    )


def finish(selection, result, capsule, *, target, native_result, work_root, remaining_s):
    """Recheck actual emission/roles, then invoke the live target effect owner.

    No copied numeric, source or packet verdict issues a pass. This consumes the
    exact original native record, unchanged all-output comparison and selected
    live SourceExecutionOwner. Required deeper tiers remain pending.
    """
    started = time.monotonic()
    if type(remaining_s) not in (int, float) or not math.isfinite(remaining_s) or remaining_s <= 0:
        raise TimeoutError("checked source diagnostic exhausted its original model deadline")
    deadline = started + remaining_s
    if type(selection) is not CheckedSourceModelDiagnostic:
        raise ValueError("checked source completion requires the issued original selection")
    selection.verify_for(capsule, target=target)
    base = dict(result)
    for key in ("complete_source_qualification", "checked_source_status", "checked_source_failure",
                "original_tier_audit", "tiers"):
        base.pop(key, None)
    base.update(status="incomplete", schema="merlin.checked_source_model_diagnostic_result.v1",
                formal_admission=False, promotion_authority="NONE", timing_cycles=None,
                checked_source_diagnostic=selection.record(),
                diagnostic_scope="original complete source/compile/ISA/functional effects; no RTL or final grade")
    native = base.get("candidate_native_execution")
    if type(native) is not dict or type(base.get("candidate_emission")) is not dict:
        base["checked_source_failure"] = "original candidate emission or complete native result unavailable"
        return base
    eligible = (base.get("candidate_source_eligibility") or {}).get("n_eligible")
    if type(eligible) is not int or eligible <= 0:
        base["checked_source_failure"] = "original complete source has no independently eligible accelerator work"
        return base
    native_path = Path(native_result)
    if native_path.is_symlink() or not native_path.is_file() or json.loads(native_path.read_bytes()) != native:
        raise ValueError("checked source diagnostic must consume the exact original native result")
    emission = base["candidate_emission"]
    from merlin.runtime.backends import base as backends

    entry_symbol = backends.harness_build_recipe(target).require_kernel_stack_frame().entry_symbol
    costs = {}
    tick = time.monotonic()
    with shared_compute_pool(timeout_s=max(0, deadline - time.monotonic())):
        costs["initial_audit_queue_seconds"] = time.monotonic() - tick
        tick = time.monotonic()
        tiers = _audit_execution(
            selection, emission, base.get("candidate_source_eligibility"), native,
            target=target, entry_symbol=entry_symbol,
        )
        costs["initial_audit_verification_seconds"] = time.monotonic() - tick
    base["tiers"] = tiers["tiers"]
    base["original_tier_audit"] = tiers
    if any((tiers["tiers"].get(tier) or {}).get("status") != "pass" for tier in ("L0", "L1", "L2")):
        base["checked_source_failure"] = "original L0/L1/L2 obligations remain incomplete or failed"
        return base
    capability = selection.revalidate()
    tick = time.monotonic()
    realization = capability.owner.create_source_execution_owner(
        native_result=native_path,
        command_buffer=NATIVE._pinned_native_file(emission, "command_buffer"),
        lowered=NATIVE._pinned_native_file(emission, "lowered_mlir"),
        work_root=Path(work_root), remaining_s=max(0, deadline - time.monotonic()),
    )
    costs["effect_owner_preparation_wall_seconds"] = time.monotonic() - tick
    costs["effect_owner_queue_seconds"] = costs["effect_trace_execution_seconds"] = None
    costs["effect_owner_cost_scope"] = "target issuer owns queue/trace subdivision; unavailable here"
    _live_realization(realization)
    selection.revalidate()
    tick = time.monotonic()
    with (shared_compute_pool(timeout_s=max(0, deadline - time.monotonic())),
          SOURCE.scoped_realization_owner(realization)):
        costs["source_verification_queue_seconds"] = time.monotonic() - tick
        tick = time.monotonic()
        checked = SOURCE.qualify({"source_integer_proof": SOURCE.SELECTION}, capsule, Path(capsule["__dir__"]))
        costs["source_verification_seconds"] = time.monotonic() - tick
    _live_realization(realization)
    if checked.get("status") != "qualified":
        raise ValueError("original complete source and actual execution correspondence is unqualified")
    # A new target-effect check cannot repair or replace the original native
    # role audit, and all those exact bytes must remain available afterwards.
    tick = time.monotonic()
    with shared_compute_pool(timeout_s=max(0, deadline - time.monotonic())):
        costs["final_audit_queue_seconds"] = time.monotonic() - tick
        tick = time.monotonic()
        final_tiers = _audit_execution(
            selection, emission, base.get("candidate_source_eligibility"), native,
            target=target, entry_symbol=entry_symbol,
        )
        costs["final_audit_verification_seconds"] = time.monotonic() - tick
    if final_tiers != tiers or json.loads(native_path.read_bytes()) != native:
        raise ValueError("checked source diagnostic original tiers or native products changed")
    selection.verify_for(capsule, target=target)
    base["complete_source_qualification"] = checked
    base["checked_source_status"] = "complete_source_L0_L1_L2_diagnostic_pass"
    base["native_execution_basis"] = (
        "exact original candidate native record and independently rechecked process products")
    base["diagnostic_verification_seconds"] = time.monotonic() - started
    base["diagnostic_costs"] = costs
    base["timing_cycles"] = None
    return base
