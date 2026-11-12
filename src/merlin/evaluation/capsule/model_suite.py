"""Capsule evaluation: model suite."""

from __future__ import annotations

import json
from contextlib import contextmanager

from merlin.evaluation.capsule.eligibility import _gate_counts


def _model_gate_operation_passed(result: dict, capsule: dict, oracle_adapters) -> bool:
    """An explicit L2 screen may open only the L2 capstone, never full certification."""
    if result.get("status") == "pass":
        return True
    if oracle_adapters is None or set(oracle_adapters) != {"L2"}:
        return False
    if result.get("status") != "incomplete" or "L2" not in capsule.get("required_oracle_tiers", []):
        return False
    failure = result.get("failure") or {}
    if (
        failure.get("plane") != "oracle_unavailable"
        or failure.get("category") != "NOT_RUN_IS_NOT_PASS"
        or failure.get("tier") != "L3"
        or failure.get("tier_status") != "unavailable"
    ):
        return False
    tiers = result.get("tiers") or {}
    if (tiers.get("L2") or {}).get("status") != "pass" or (result.get("numeric") or {}).get("status") != "pass":
        return False
    for tier in capsule.get("required_oracle_tiers", []):
        if tier == "L3":
            continue
        record = tiers.get(tier) or {}
        if record.get("status") != "pass" and not record.get("not_applicable"):
            return False
    return True


def _run_streaming_model_gate(
    op_caps,
    model_caps,
    run_one,
    *,
    max_workers,
    target,
    oracle_adapters,
    on_complete=None,
    calibrate=None,
    calibration_cap=0,
    model_gate_continuations=(),
    defer_pending=None,
    defer_running=True,
    boundary_readiness=None,
    model_gate_prerequisites=None,
):
    """Open capstones on a fixed selected-operator denominator in the same worker pool."""
    from merlin.development.targetgen.capsule_scheduling import run_pipelined
    from merlin.experiments.corpus.model_sources import model_gate_satisfied

    caps = op_caps + model_caps
    model_names = {str(cap["name"]) for cap in model_caps}
    if any(str(cap.get("extends") or "") in model_names for cap in op_caps):
        raise ValueError("operator capsule extends a gated model; dependency conflicts with the model gate")
    # Determine eligibility from the complete selected roster, never completed rows.
    eligible = [c for c in op_caps if _gate_counts({"capsule": c["name"]}, caps, target)]
    denominator = eligible or op_caps
    completed = {}

    def fraction(cap=None):
        if cap is not None:
            from merlin.development.targetgen.model_gate_continuation import prerequisite_fraction

            prior = prerequisite_fraction(cap, model_gate_prerequisites, target=target)
            if prior is not None:
                return prior
        return (
            sum(_model_gate_operation_passed(completed.get(c["name"], {}), c, oracle_adapters) for c in denominator)
            / len(denominator)
            if denominator
            else 0.0
        )

    def finished(result):
        completed[str(result.get("capsule"))] = result
        if on_complete is not None:
            on_complete(result)

    def blocked(cap):
        return {
            "capsule": cap["name"],
            "kind": cap.get("kind"),
            "label": cap.get("label"),
            "status": "gated",
            "failure": {
                "plane": "gate",
                "category": "GATED",
                "detail": f"whole-model capsule deferred: op pass fraction {fraction(cap):.2f} "
                f"< gate {(cap.get('gate') or {}).get('after_op_pass_fraction')}",
            },
        }

    def admitted(cap):
        if cap.get("kind") != "model":
            return True
        from merlin.development.targetgen.model_gate_continuation import admits, prerequisite_fraction

        prior = prerequisite_fraction(cap, model_gate_prerequisites, target=target)
        continuation = (
            prior is None
            and not op_caps
            and admits(cap, model_gate_continuations, target=target, adapters=oracle_adapters)
        )
        value = fraction(cap)
        passed = continuation or model_gate_satisfied(cap, value)
        if passed:
            cap["_driver_model_gate_admission"] = {
                "passed": True,
                "fraction": value,
                "stage": "L2" if set(oracle_adapters or {}) == {"L2"} else "continuation" if continuation else "full",
            }
        return passed

    return run_pipelined(
        caps,
        run_one,
        max_workers=max_workers,
        calibrate=calibrate,
        calibration_cap=calibration_cap,
        boundary_readiness=boundary_readiness,
        on_complete=finished,
        can_start=admitted,
        blocked_result=blocked,
        calibration_eligible=lambda cap: cap.get("kind") != "model",
        priority=lambda cap: 0 if cap.get("kind") == "model" else 1,
        **({"defer_pending": defer_pending, "defer_running": defer_running} if defer_pending is not None else {}),
    )


@contextmanager
def _original_capsule_worker_scope():
    """Rebind the original executor's guarded entry on this actual worker thread.

    A copied parent context alone deliberately grants no thread-local snapshot
    reuse. Reuse requires the executor's existing live concurrency capability;
    its scope checks owner/selection identity at entry and exit. Legacy owners
    retain ordinary fresh verification, and every command keeps its final read.
    """
    from merlin.development.targetgen.package_runtime import active_package_executor

    owner = active_package_executor()
    capability = getattr(owner, "concurrent_entrypoints_v1", None)
    if not callable(capability) or capability() is not True:
        yield
        return
    scope = getattr(owner, "concurrent_entrypoint_scope_v1", None)
    if not callable(scope):
        raise ValueError("concurrent capsule execution requires its original worker scope")
    with scope():
        yield


def _candidate_boundary_readiness(capsules, pkg, *, oracle_adapters):
    """Order one existing public L2 operator witness for this exact native owner.

    This is scoped to the immutable suite invocation. Models/slices use distinct
    host graph routes and remain unchanged. Component model_slice wrappers that
    reach the ordinary operator ladder share this route; raw connected source
    slices and Merlin-owned host leaves do not. No error message or case-local
    structural category can prove the whole compiler route invalid.
    """
    if set(oracle_adapters or {}) != {"L2"}:
        return None
    from merlin.development.backend import joint_control
    from merlin.development.backend.feedback.compiler_profile_runtime import active_original_execution
    from merlin.development.targetgen.capsule_scheduling import BoundaryReadiness
    from merlin.development.targetgen.expected_refusal import expects_refusal
    from merlin.development.targetgen.owned_host_leaf_execution import selected_host_leaf
    from merlin.development.targetgen.owned_host_slice_execution import selected_source_slice

    if joint_control.selected() is None:
        return None
    owner = active_original_execution()
    if owner is None:
        return None
    snapshot = getattr(owner.owner, "snapshot_digest", None)
    profile = getattr(owner.owner, "profile", None)
    if not snapshot or not isinstance(profile, dict) or not profile.get("sha256"):
        return None
    import hashlib

    identity = {
        "compiler_sha256": owner.source_sha256,
        "bundle_snapshot_sha256": snapshot,
        "profile_sha256": profile["sha256"],
        "manifest": pkg.manifest,
        "route": "native_operator_lower_interface",
        "requested_tiers": ["L2"],
    }
    route = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return BoundaryReadiness(
        {
            str(cap["name"]): route
            for cap in capsules
            if cap.get("kind") in {"isa", "layer", "model_slice"}
            and cap.get("label") in {"public", "dev"}
            and not expects_refusal(cap)
            and not selected_host_leaf(cap)
            and not selected_source_slice(cap)
        }
    )
