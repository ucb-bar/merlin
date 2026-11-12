"""Original owned-model input selections, frozen policy and independent engine bindings."""

from __future__ import annotations

import json
from pathlib import Path

from merlin.development.targetgen.owned_host_dispatch_checks import CandidateDispatchRefusal
from merlin.runtime.modelinputs import file_sha256


def _selected_file(selection, role):
    record = selection.get(role)
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError(f"owned host build needs exact {role} selection")
    path = Path(record["path"])
    if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f"owned host build {role} is indirect")
    if not path.is_file() or file_sha256(path) != record["sha256"]:
        raise ValueError(f"owned host build {role} changed")
    return path


def _composition_preferences(candidate_package):
    """Read bounded optimization requests, never source or effect permissions."""
    from merlin.development.targetgen.source_composition_preferences import read

    return read(candidate_package)


def _resolve_weight_preparation(bundle, *, requested):
    from merlin.development.targetgen.source_composition_preferences import weight_preparation

    return weight_preparation(bundle, requested=requested)


def _independently_eligible_group_ids(placement, groups):
    """Keep every group containing proven eligible work, regardless of its root."""
    rows = placement.get("eligible_source_regions")
    if not isinstance(rows, list) or not rows:
        raise ValueError("owned source lacks independently eligible group roster")
    selected = set()
    for row in rows:
        index = row.get("source_group") if isinstance(row, dict) else None
        if type(index) is not int or index < 0:
            raise ValueError("owned source eligible group identity is invalid")
        selected.add(index)
    available = {group.index for group in groups}
    if not selected <= available:
        raise ValueError("owned source eligible group is absent from original groups")
    return selected


def _untagged_assert_comparisons(ops, frozen, groups):
    """Scalar support for tagged assertions is not another provenance demand.

    Keep these operations in the complete frozen placement/build inventory. Only
    exact source-bound host admission permits this distinction; arbitrary untagged
    arithmetic still fails the demand join.
    """
    from merlin.common import mlir_query as mq
    from merlin.compiler.dialects.lowering import compute_groups as groups_api

    host = frozen["host_operations"]
    owner = {id(member): group for group in groups for member in group.members}
    support = set()
    assertions = set()
    for op in ops:
        if (
            mq.op_name(op) != "arith.cmpi"
            or "prov.op" in op.attributes
            or op.regions
            or op.successors
            or len(op.operands) != 2
            or [str(value.type) for value in op.operands] != ["index", "index"]
            or len(op.results) != 1
            or str(op.results[0].type) != "i1"
        ):
            continue
        uses = tuple(op.results[0].uses)
        if not uses or any(
            use.operation not in ops
            or mq.op_name(use.operation) != "cf.assert"
            or mq.attr_str(use.operation, "prov.op") != "shape_guard"
            or len(use.operation.operands) != 1
            or use.operation.results
            or use.operation.regions
            or use.operation.successors
            for use in uses
        ):
            continue
        members = [op, *(use.operation for use in uses)]
        if any(id(member) in owner and owner[id(member)].placement != groups_api.HOST for member in members):
            continue
        support.add(id(op))
        assertions.update(id(use.operation) for use in uses)
    if len(support) > host.get("arith.cmpi", 0) or len(assertions) > host.get("cf.assert", 0):
        raise ValueError("scalar assertion support lacks complete frozen host admission")
    return support


def _pinned(record, role):
    from merlin.development.targetgen.native_model_execution import _pinned_native_file, _read_pinned_payload

    _read_pinned_payload(record, role)
    return _pinned_native_file(record, role)


def _require_instruction_classes(required, observed):
    """Report declared structural obligations, never private reference values."""
    missing = set(required) - set(observed)
    if missing:
        raise ValueError(
            "owned device kernels omit frozen required instruction classes: "
            f"missing={sorted(missing)!r}; observed={sorted(set(observed))!r}; "
            f"required={sorted(set(required))!r}"
        )


def _selected_facts(path, *, target):
    from merlin.runtime.modelinputs import selected_execution_firrtl

    return selected_execution_firrtl(path, target=target)


def _native_engine_for_facts(target, engine, facts):
    from merlin.development.targetgen.rtl.facts import observed_facts
    from merlin.runtime.modelinputs import native_engine, selected_firrtl

    if selected_firrtl(facts["path"], target=target, config=facts["config"]) != facts:
        raise ValueError("protected owned-model RTL facts changed")
    with observed_facts(target, json.loads(Path(facts["path"]).read_text()), path=Path(facts["path"])):
        return native_engine(target, engine, facts)


def _functional_engine_for_facts(target, facts):
    from merlin.development.targetgen.native_model_execution import _functional_engine
    from merlin.development.targetgen.rtl.facts import observed_facts

    with observed_facts(target, json.loads(Path(facts["path"]).read_text()), path=Path(facts["path"])):
        return _functional_engine(target)


def _same_owned_frozen_policy(recorded, fresh) -> bool:
    """Join a retained capsule copy by verified bytes, preserving policy owners."""
    from merlin.development.targetgen.native_model_execution import _read_pinned_payload

    local_roles = {"capsule_declaration", "instruction_coverage"}
    if not isinstance(recorded, dict) or not isinstance(fresh, dict) or set(recorded) != set(fresh):
        return False
    if any(recorded[key] != fresh[key] for key in fresh if key not in local_roles):
        return False
    for role in local_roles:
        # Reopen both ordinary files under their exact stored identities;
        # dropping path strings alone would let a missing or mutated copy pass.
        original = _read_pinned_payload(recorded, role)
        selected = _read_pinned_payload(fresh, role)
        old_pin, new_pin = recorded[role], fresh[role]
        if (
            set(old_pin) != {"path", "sha256", "size_bytes"}
            or set(new_pin) != {"path", "sha256", "size_bytes"}
            or old_pin["sha256"] != new_pin["sha256"]
            or old_pin["size_bytes"] != new_pin["size_bytes"]
            or original != selected
        ):
            return False
    return True


def _stopped_before_receipt(stopped):
    """The audit verdict for a run that stopped before producing its receipt: its root cause first.

    The absent receipt is only a consequence. Reporting it ("receipt ... is absent") hid the actual
    reason, and when that reason is the candidate's own dispatch refusal, it turned a compiler failure
    into what reads as unmeasured tooling. A proven emitted-device host-compute veto is also a
    candidate failure. Any other stop stays incomplete (NOT_RUN_IS_NOT_PASS) and
    names its stage and exception; host paths are scrubbed from the quoted message.
    """
    from merlin.common.path_scrub import scrub_host_paths

    stage, kind = str(stopped.get("stage") or "unknown stage"), str(stopped.get("type") or "error")
    message = scrub_host_paths(str(stopped.get("detail") or ""))[:2000]
    if kind == "HostComputeVeto" and stage == "owned_host_execution":
        return {
            "status": "fail",
            "kind": "owned_host_model_execution",
            "violations": ["candidate_host_tensor_compute_violation"],
            "detail": message,
            "failure": {
                "plane": "host_compute",
                "category": "host_compute_in_accelerator_group",
                "detail": message,
                "stage": stage,
            },
        }
    if kind == CandidateDispatchRefusal.__name__ and stage == "owned_host_execution":
        return {
            "status": "fail",
            "kind": "owned_host_model_execution",
            "violations": ["candidate_dispatch_attribution_refused"],
            "detail": message,
            "failure": {
                "plane": "candidate_compilation",
                "category": "CANDIDATE_EMISSION_FAILED",
                "detail": message,
                "stage": stage,
            },
        }
    detail = f"the model run stopped at {stage} with {kind} before producing its receipt: {message}"
    return {
        "status": "incomplete",
        "kind": "owned_host_model_execution",
        "violations": ["candidate_completed_dispatch_unverified", "candidate_required_tiers_unverified"],
        "detail": detail,
        "failure": {"plane": "candidate_model_execution", "category": "NOT_RUN_IS_NOT_PASS", "detail": detail},
    }


# Preserve historical callable and record pickle lookup identities.
_selected_file.__module__ = "merlin.development.targetgen.owned_host_model_execution"
_composition_preferences.__module__ = "merlin.development.targetgen.owned_host_model_execution"
_resolve_weight_preparation.__module__ = "merlin.development.targetgen.owned_host_model_execution"
_independently_eligible_group_ids.__module__ = "merlin.development.targetgen.owned_host_model_execution"
_untagged_assert_comparisons.__module__ = "merlin.development.targetgen.owned_host_model_execution"
_pinned.__module__ = "merlin.development.targetgen.owned_host_model_execution"
_require_instruction_classes.__module__ = "merlin.development.targetgen.owned_host_model_execution"
_selected_facts.__module__ = "merlin.development.targetgen.owned_host_model_execution"
_native_engine_for_facts.__module__ = "merlin.development.targetgen.owned_host_model_execution"
_functional_engine_for_facts.__module__ = "merlin.development.targetgen.owned_host_model_execution"
_same_owned_frozen_policy.__module__ = "merlin.development.targetgen.owned_host_model_execution"
_stopped_before_receipt.__module__ = "merlin.development.targetgen.owned_host_model_execution"
