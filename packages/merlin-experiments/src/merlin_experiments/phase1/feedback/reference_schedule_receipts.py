"""Read source-bound construction receipts without upgrading them to execution.

The scheduling library records its consumed decision IR in command buffers.
These records are useful even when the original emission owner is unavailable,
but supplied records cannot authenticate their consumer, LLVM, or machine code.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter

MAX_NODES = 16384
MAX_DEPTH = 32
MAX_RECEIPTS = 32
MAX_TRANSFORMS = 64
MAX_TEXT_BYTES = 65536
_CONSUMED = frozenset(("block_tiles", "outer_order", "row_bases", "panel_pipeline",
                       "pipeline_window_offsets", "loop_transforms"))
_HASH_FIELDS = frozenset(("kernel_sha256", "actual_kernel_sha256", "source_sha256", "actual_sha256",
                          "checking_kernel_sha256", "original_kernel_sha256", "expanded_kernel_sha256",
                          "checking_expansion_sha256", "phase_kernel_sha256"))


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _hash(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _attribute(value):
    from xdsl.dialects.builtin import ArrayAttr, DenseArrayBase, IntegerAttr, StringAttr

    if isinstance(value, StringAttr):
        return value.data
    if isinstance(value, IntegerAttr):
        data = value.value.data
        return bool(data) if value.type.width.data == 1 else data
    if isinstance(value, DenseArrayBase):
        return list(value.get_values())
    if isinstance(value, ArrayAttr):
        return [_attribute(item) for item in value]
    raise ValueError("unsupported_schedule_attribute")


def _positive(values, size):
    return isinstance(values, list) and len(values) == size and all(type(v) is int and v > 0 for v in values)


def _receipt(record, *, source_sha256, pointer):
    from merlin.common.ir_lock import IR_LOCK
    from merlin.compile.scheduling.schedule_ir import parse_dense_schedule_plan

    result = dict(pointer=pointer, status="invalid", consumption="unknown")
    text = record.get("text")
    if type(text) is not str or len(text.encode()) > MAX_TEXT_BYTES:
        return dict(result, reason="schedule_text_missing_or_over_budget")
    raw = text.encode()
    result.update(schedule_sha256=_sha(raw), source_sha256=record.get("source_identity"))
    if not _hash(record.get("sha256")) or _sha(raw) != record["sha256"]:
        return dict(result, reason="schedule_text_digest_mismatch")
    if record.get("source_identity") != source_sha256:
        return dict(result, reason="schedule_source_identity_mismatch")
    consumed = record.get("decisions_consumed")
    if (not isinstance(consumed, (list, tuple)) or not consumed
            or any(type(v) is not str or v not in _CONSUMED for v in consumed)
            or len(set(consumed)) != len(consumed)):
        return dict(result, reason="consumed_decision_roster_unknown")
    try:
        with IR_LOCK:
            module = parse_dense_schedule_plan(text)
            module.verify()
            operations = list(module.walk())
            if len(operations) > 1024:
                return dict(result, reason="schedule_operation_budget_exhausted")
            binds = [op for op in operations if op.name == "schedule.bind"]
            blocks = [op for op in operations if op.name == "schedule.block_contraction"]
            if len(binds) != 1 or len(blocks) != 1:
                return dict(result, reason="dense_decision_scope_not_unique")
            target = _attribute(binds[0].properties["target"])
            fields = target.split(":")
            if len(fields) != 3 or fields[:2] != ["dense", source_sha256] or not _hash(fields[2]):
                return dict(result, reason="schedule_bind_source_mismatch")
            block = {key: _attribute(value) for key, value in blocks[0].attributes.items()}
            if (any(not _positive(block.get(key), 3) for key in ("problem", "primitive_tile", "block_tiles"))
                    or any(not _positive(block.get(key), 2) for key in ("storage_bytes", "storage_capacity"))
                    or any(used > capacity for used, capacity in
                           zip(block["storage_bytes"], block["storage_capacity"], strict=True))):
                return dict(result, reason="recorded_geometry_or_capacity_invalid")
            pipelines = [{key: _attribute(value) for key, value in op.properties.items()}
                         for op in operations if op.name == "schedule.panel_pipeline"]
            transforms = [{key: _attribute(value) for key, value in op.properties.items()}
                          for op in operations if op.name == "schedule.loop_transform"]
            if len(pipelines) > 1:
                return dict(result, reason="pipeline_decision_scope_not_unique")
            pipeline = pipelines[0] if pipelines else {}
            for external, internal in (("pipeline_kind", "kind"), ("prefetch", "prefetch"),
                                       ("pipeline_issue_order", "issue_order"),
                                       ("pipeline_window_offsets", "window_offsets")):
                if record.get(external) is not None and record[external] != pipeline.get(internal):
                    return dict(result, reason="receipt_pipeline_differs_from_decision_ir")
            if ("panel_pipeline" in consumed and not pipeline
                    or "loop_transforms" in consumed and not transforms):
                return dict(result, reason="consumed_decision_absent_from_ir")
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        return dict(result, reason="schedule_decision_parse_or_verification_refused", error=type(error).__name__)
    except Exception as error:  # xDSL parse/verify exception classes vary across selected releases.
        from xdsl.utils.exceptions import ParseError, VerifyException

        if not isinstance(error, (ParseError, VerifyException)):
            raise
        return dict(result, reason="schedule_decision_parse_or_verification_refused", error=type(error).__name__)
    return dict(result, status="source_bound_recorded_consumption", consumption="recorded_consumer_claim",
                context_sha256=fields[2], consumed=list(consumed), choices=dict(block=block, pipeline=pipeline,
                loop_transforms=transforms), capacity_condition=dict(status="within_recorded_private_capacity",
                used_bytes=block["storage_bytes"], capacity_bytes=block["storage_capacity"],
                selected_hardware_and_simultaneous_lifetimes="unknown"),
                source_shape_correspondence="unknown", emitted_correspondence="unknown", target_execution="unknown")


def inspect_schedule_receipts(*, source_mlir, command_buffer, emitted_mlir, producer_binding=None):
    """Reopen bounded decision text and hash links in an actual supplied CB.

    No aggregate loads, traffic, fences, timing or CCA axes are filled from
    construction estimates. The original compiler/build owner must separately
    establish that these exact CB/LLVM bytes were its consumed products.
    """
    pins = {role: _sha(raw) for role, raw in
            (("source", source_mlir), ("command_buffer", command_buffer), ("emitted", emitted_mlir))}
    document = json.loads(command_buffer)
    receipts, transformations, omitted = [], [], []
    stack = [(document.get("params", {}), "/params", 0)]
    visited = 0
    while stack:
        value, pointer, depth = stack.pop()
        visited += 1
        if visited > MAX_NODES or depth > MAX_DEPTH:
            omitted.append("metadata_node_or_depth_budget_exhausted")
            break
        if isinstance(value, dict):
            if isinstance(value.get("schedule_ir"), dict):
                if len(receipts) < MAX_RECEIPTS:
                    receipts.append(_receipt(value["schedule_ir"], source_sha256=pins["source"],
                                             pointer=pointer + "/schedule_ir"))
                else:
                    omitted.append("schedule_receipt_budget_exhausted")
            schema = value.get("schema")
            if type(schema) is str and schema.startswith("merlin."):
                links = {}
                for key in sorted(_HASH_FIELDS & value.keys()):
                    item = value[key]
                    if (_hash(item) or isinstance(item, dict) and 0 < len(item) <= 32
                            and all(type(k) is str and len(k) <= 128 and _hash(v) for k, v in item.items())):
                        links[key] = item
                if links:
                    if len(transformations) < MAX_TRANSFORMS:
                        transformations.append(dict(pointer=pointer, schema=schema, hash_links=links,
                            status="recorded_construction_links", executable_correspondence="unknown"))
                    else:
                        omitted.append("transformation_record_budget_exhausted")
            stack.extend((item, pointer + "/" + str(key).replace("~", "~0").replace("/", "~1"), depth + 1)
                         for key, item in reversed(list(value.items())) if key != "schedule_ir")
        elif isinstance(value, list):
            stack.extend((item, pointer + "/" + str(index), depth + 1)
                         for index, item in reversed(list(enumerate(value))))
    valid = [row for row in receipts if row["status"] == "source_bound_recorded_consumption"]
    result = dict(schema="merlin.reference_schedule_receipts.v1",
        status="partial" if omitted else "observed_recorded_consumption" if valid else "unknown",
        inputs=pins, records=receipts, valid_records=len(valid), invalid_records=len(receipts) - len(valid),
        transformation_records=transformations, budget_status="partial" if omitted else "complete",
        missing=sorted(set(omitted)) + ["original_compiler_command_buffer_emission_owner",
            "instruction_kernel_to_emitted_function_correspondence", "selected_hardware_lifetime_legality",
            "complete_reference_candidate_semantic_alignment"],
        schedule_consumption="recorded_only" if valid else "unknown",
        compiler_and_emission_binding="unknown", semantic_equivalence="unknown", target_execution="unknown",
        next_action="Bind the original immutable compiler operation and its exact CB/LLVM products; "
                    "inspect recorded blocking, capacity and pipeline choices without treating them as executed work.")
    if producer_binding is not None:
        # This argument is supplied only by the original host's protected build
        # index after its ownership checks; author artifact manifests have no
        # producer-binding field. Hash equality here guards this adapter's join,
        # and cannot by itself issue an original build or semantic witness.
        required = ("compiler_sha256", "object_sha256", "linked_elf_sha256", "function_view_sha256")
        matches = (isinstance(producer_binding, dict)
            and producer_binding.get("status") == "observed_original_build_consumption"
            and all(producer_binding.get(role + "_sha256") == value for role, value in pins.items())
            and all(_hash(producer_binding.get(key)) for key in required))
        if matches:
            result["compiler_and_emission_binding"] = "observed_original_build_consumption"
            result["producer_binding"] = producer_binding
            result["missing"].remove("original_compiler_command_buffer_emission_owner")
            if valid:
                result["schedule_consumption"] = "recorded_in_observed_compiler_products"
        else:
            result["producer_binding"] = dict(status="unknown", reason="original_build_product_pins_do_not_match")
    return result


def compare_schedule_receipts(reference, candidate):
    """Compare recorded decision populations, never pair arbitrary source regions.

    A shared whole-source hash does not align each group's virtual rows. Keep
    population differences distinct from a transfer or mechanism application.
    """
    def profiles(report):
        values = {}
        for record in report.get("records", []):
            if record.get("status") != "source_bound_recorded_consumption":
                continue
            for facet, fields in record["choices"].items():
                if facet == "loop_transforms":
                    fields = {"requests": fields}
                for key, value in fields.items():
                    values.setdefault(facet + "." + key, Counter())[json.dumps(value, sort_keys=True)] += 1
        return {key: [dict(value=json.loads(value), records=count) for value, count in sorted(counts.items())]
                for key, counts in sorted(values.items())}

    left, right = profiles(reference), profiles(candidate)
    source = reference.get("inputs", {}).get("source")
    comparable = (bool(left) and bool(right) and _hash(source)
        and source == candidate.get("inputs", {}).get("source")
        and reference.get("budget_status") == candidate.get("budget_status") == "complete"
        and not reference.get("invalid_records") and not candidate.get("invalid_records"))
    return dict(status="compared_recorded_choice_populations" if comparable else "independent_recorded_choices",
        source_identity="identical_bytes" if comparable else "unknown_or_different",
        choices=[dict(axis="schedule." + axis, reference=left.get(axis), candidate=right.get(axis),
                      differs=left.get(axis) != right.get(axis) if comparable else None,
                      status="compared_recorded_population" if comparable else "unknown")
                 for axis in sorted(left.keys() | right.keys())],
        scope="recorded decision populations; group/source-operation alignment unknown",
        canonical_cca_axes_added=0, semantic_equivalence="unknown", optimization_application="unknown",
        performance="unknown")
