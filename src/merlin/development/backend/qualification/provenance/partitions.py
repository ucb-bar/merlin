"""Join protected canonical model workers without reusing diagnostic images."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from merlin.common.process import invocation_records as invocation_record
from merlin.compiler.ir import mlir_query as mq
from merlin.development.backend.qualification.full_models import execution as _merlin_execution_1
from merlin.development.backend.qualification.full_models import execution_engine as execution
from merlin.development.backend.qualification.full_models import private_full_models as models
from merlin.development.backend.qualification.full_models.private_source_member_hardware import observe
from merlin.development.backend.qualification.provenance import private_instruction_declaration as declaration
from merlin.runtime.execution.model_inputs import file_sha256

# ---- private_model_partitions ------------------------------------------------------------------

SCHEMA = "merlin.phase1.private_model_partition.v1"

BINDINGS = (
    "run_dir",
    "candidate_tree_sha256",
    "private_spec_sha256",
    "authored_source_freeze",
    "compiler_profile",
    "fact_reader_binding",
    "instruction_declaration_sha256",
)


def _ordinary(path):
    path = Path(path)
    if not path.is_absolute() or not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("model partition has no ordinary protected artifact")
    return path


def _selection(record, *, target, instruction_path):
    """Reopen each worker's actual issuance; fresh output paths are never equated."""
    selected = record["instruction_selection"]
    original = declaration.read(instruction_path, target=target)
    original.verify()
    if selected.get("declaration_sha256") != file_sha256(instruction_path):
        raise ValueError("model worker selected another original instruction declaration")
    receipt = _ordinary(record["instruction_selection_receipt"])
    if json.loads(receipt.read_bytes()) != selected:
        raise ValueError("model worker instruction selection receipt changed")
    issued = invocation_record.verify(_ordinary(record["instruction_selection_invocation"]))
    if (
        issued.get("kind") != "python_call"
        or issued.get("stage") != "fresh_instruction_coordinator_selection"
        or issued.get("callable")
        != "merlin.development.backend.qualification.provenance.private_instruction_coordinator._issue"
        or issued.get("arguments") != {"target": target, "declaration_sha256": file_sha256(instruction_path)}
        or not any(p.get("path") == str(receipt) and p.get("sha256") == file_sha256(receipt) for p in issued["outputs"])
    ):
        raise ValueError("model worker has no completed original instruction issuer")
    pins = record.get("instruction_source_pins")
    if (
        not isinstance(pins, list)
        or not pins
        or any(not isinstance(pin, list) or len(pin) != 2 or file_sha256(_ordinary(pin[0])) != pin[1] for pin in pins)
    ):
        raise ValueError("model worker instruction source selection changed")


def join(
    records: Mapping[str, str | Path],
    invocations: Mapping[str, str | Path],
    *,
    expected: Mapping,
    required_models: Sequence[str],
    required_programs: Mapping[str, Sequence[str]],
    gate: Mapping,
    target: str,
    instruction_path: str | Path,
) -> dict:
    """Join one completed host-issued worker per full declared model.

    The normal coordinator supplies freshly verified original run/profile/source
    bindings. Worker observations reopen actual reports and inputs; the existing
    full static validator reopens all compilation recipes, tools, objects and ELFs.
    Each instruction issuer retains its own paths and proofs under the same
    original declaration. No individual worker constitutes the full roster gate.
    """
    roster = list(required_models)
    programs = {name: list(required_programs[name]) for name in roster}
    if (
        not roster
        or len(set(roster)) != len(roster)
        or set(records) != set(roster)
        or set(invocations) != set(roster)
        or set(required_programs) != set(roster)
        or any(key not in expected for key in BINDINGS)
        or programs != {name: list(stages) for name, stages in gate["programs"].items()}
        or expected["instruction_declaration_sha256"] != file_sha256(_ordinary(instruction_path))
    ):
        raise ValueError("model partition join differs from the complete original declaration")
    workers, static_rows, execution_rows = [], [], []
    for name in roster:
        path = _ordinary(records[name])
        observation = invocation_record.verify(_ordinary(invocations[name]))
        if observation.get("kind") != "subprocess" or not any(
            pin.get("path") == str(path) and pin.get("sha256") == file_sha256(path) for pin in observation["outputs"]
        ):
            raise ValueError("model partition has no completed protected host worker")
        record = json.loads(path.read_bytes())
        if (
            record.get("schema") != SCHEMA
            or record.get("selected_model") != name
            or any(record.get(key) != expected[key] for key in BINDINGS)
        ):
            raise ValueError("model partition changed its original owner, source, facts or declaration")
        _selection(record, target=target, instruction_path=Path(instruction_path))
        static = record.get("private_full_models") or {}
        static_report = _ordinary(record["static_partition_report"])
        if (
            file_sha256(static_report) != record.get("static_partition_report_sha256")
            or json.loads(static_report.read_bytes()) != static
        ):
            raise ValueError("model worker canonical static producer record changed")
        rows = static.get("models")
        if (
            static.get("schema") != models.RESULT_SCHEMA
            or static.get("mode") != "canonical_model_partition"
            or static.get("selected_model") != name
            or static.get("passed") is not False
            or static.get("required_models") != roster
            or static.get("required_programs") != programs
            or static.get("target") != target
            or any(
                static.get(key) != expected[key]
                for key in ("candidate_tree_sha256", "private_spec_sha256", "authored_source_freeze")
            )
            or not isinstance(rows, list)
            or len(rows) != 1
            or rows[0].get("model") != name
            or static.get("partition_passed") is not (rows[0].get("status") == "pass")
        ):
            raise ValueError("model worker omitted or upgraded its canonical static partition")
        numerical = record.get("private_full_model_execution") or {}
        if numerical.get("mode") == "canonical_model_partition":
            required = [(model, program) for model, program, _ in execution.roster_of(gate) if model == name]
            observed = [(row.get("model"), row.get("program")) for row in numerical.get("programs", [])]
            if (
                numerical.get("schema") != execution.SCHEMA
                or numerical.get("selected_model") != name
                or numerical.get("passed") is not False
                or observed != required
                or numerical.get("required_models") != roster
                or numerical.get("required_programs") != programs
                or numerical.get("candidate_tree_sha256") != expected["candidate_tree_sha256"]
            ):
                raise ValueError("model worker omitted or upgraded its canonical numerical partition")
            execution_rows.extend(numerical["programs"])
        elif static.get("partition_passed") is True or numerical.get("passed") is not False:
            raise ValueError("successful static partition has no canonical numerical partition")
        static_rows.extend(rows)
        workers.append(record)
    full = {
        "schema": models.RESULT_SCHEMA,
        "target": target,
        "candidate_tree_sha256": expected["candidate_tree_sha256"],
        "private_spec_sha256": expected["private_spec_sha256"],
        "authored_source_freeze": expected["authored_source_freeze"],
        "required_models": roster,
        "required_programs": programs,
        "models": static_rows,
        "passed": all(row["status"] == "pass" for row in static_rows),
        "full_model_numerical_equivalence": "not_run",
        "paper_accuracy": "not_claimed",
        "scope": "complete captured networks, canonical per-model workers; full-model execution separate",
        "worker_records": [str(records[name]) for name in roster],
        "instruction_selections_by_model": {row["selected_model"]: row["instruction_selection"] for row in workers},
    }
    full["passed"] = models.complete(
        full,
        required_models=roster,
        required_programs=required_programs,
        candidate_sha256=expected["candidate_tree_sha256"],
    )
    numerical = {
        "schema": execution.SCHEMA,
        "target": target,
        "required": gate["required"],
        "candidate_tree_sha256": expected["candidate_tree_sha256"],
        "programs": execution_rows,
        "deferred": [
            f"{row['model']}:{row['program']}" for row in execution_rows if row["status"] == execution.DEFERRED
        ],
        "passed": full["passed"],
        "per_group_exactness": execution.PER_GROUP_SCOPE,
        "numerical_scope": "complete original program roster joined from canonical model workers",
    }
    numerical["passed"] = full["passed"] and _merlin_execution_1.complete(
        numerical,
        gate,
        static=full,
        candidate_sha256=expected["candidate_tree_sha256"],
    )
    return {
        "private_full_models": full,
        "private_full_model_execution": numerical,
        "passed": full["passed"] and numerical["passed"],
    }


# ---- private_placement_feedback ----------------------------------------------------------------
# Diagnostic placement projection over native source groups; grants no admission.


def _bytes(values):
    total = 0
    for value in values:
        typ = value.type
        if not hasattr(typ, "get_shape"):
            return None
        shape = typ.get_shape()
        from xdsl.dialects.builtin import AnyFloat, IntegerType

        element = typ.element_type
        if not isinstance(element, (IntegerType, AnyFloat)) or any(dim < 0 for dim in shape):
            return None
        bits = element.bitwidth
        total += (math.prod(shape) * bits + 7) // 8
    return total


def describe(groups, capability, *, demanded, unknown_host, missing_direct):
    """Explain the existing demand decision without reclassifying or relaxing it."""
    rows = []
    for group in groups:
        members = {id(op) for op in group.members}
        inputs, outputs = {}, {}
        identities, observations = [], []
        for op, stage in zip(group.members, group.stages, strict=True):
            attrs = op.attributes
            region = attrs.get("prov.region_id")
            nodes = attrs.get("prov.source_node_ids")
            identities.append(
                {
                    "operation": mq.op_name(op),
                    "region_id": getattr(region, "data", None),
                    "source_node_ids": [getattr(node, "data", None) for node in getattr(nodes, "data", ())],
                }
            )
            try:
                observations.append(observe(op, stage, capability))
            except ValueError:
                # This projection cannot introduce an admission rule for a
                # source grouping form the independent member observer lacks.
                observations.append(
                    {"status": "unknown", "refusal": "typed_member_observation_refused", "basis": "diagnostic_only"}
                )
            for value in op.operands:
                if id(value.owner) not in members:
                    inputs[id(value)] = value
            for value in op.results:
                if any(id(use.operation) not in members for use in value.uses):
                    outputs[id(value)] = value
        required = group.index in demanded
        unresolved = group.index in unknown_host
        admitted_member = any(row.get("status") == "admitted" for row in observations)
        reason = (
            "required_device_region"
            if required
            else "eligible_member_without_region_lowering"
            if unresolved and admitted_member
            else "unresolved_host_placement"
            if unresolved
            else "source_checked_host_placement"
        )
        rows.append(
            {
                "source_group": group.index,
                "source_identities": identities,
                "stages": list(group.stages),
                "placement": group.placement,
                "required_device": required,
                "reason": reason,
                "compiler_refusal": group.refusal,
                "compiler_gap": group.gap,
                "member_hardware_observations": observations,
                "mapping_scope": "composed_region" if len(group.members) > 1 else "standalone_region",
                "standalone_equivalence": "not_proved_by_composed_group_admission",
                "logical_input_bytes": _bytes(inputs.values()),
                "logical_output_bytes": _bytes(outputs.values()),
                "transferred_bytes": None,
                "host_time_cycles": None,
            }
        )
    return {
        "schema": "merlin.source_placement_feedback.v1",
        "authority": "diagnostic_only",
        "counts": dict(Counter(row["reason"] for row in rows)),
        "source_groups": len(groups),
        "missing_direct_operations": missing_direct,
        "byte_scope": "distinct static SSA boundary payloads per group; not physical traffic or cross-group unique bytes",
        "timing_scope": "not_measured",
        "rows": rows,
    }
