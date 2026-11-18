"""Replay an operator-selected original issue capture through its pinned observer.

The private environment selects immutable analysis tools and captured bytes.
Requests cannot choose Python code, execute C, or supply a semantic summary.
Numerical execution, device completion and elapsed timing retain their owners.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import types
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from merlin.analysis.mining.references.evidence import EvidenceFact, cca_projection

# ---- reference_capture_views -------------------------------------------------------------------

SELECTION = "MERLIN_REFERENCE_CAPTURE_SELECTION"

SCHEMA = "merlin.reference_capture_selection.v1"

MAX_CAPTURE_COMMANDS = 1_000_000


def _dispatch(model, *, op, fences, configurations, target, source="reference", facts=()):
    """Existing CCA axes, witnessed at the complete invocation's issue boundary."""
    commands = model.event_counts.get("command")
    witness = (f"schedule:{model.identity}",)
    additions = [EvidenceFact("dispatch.n_fences", fences, "observed", witness)]
    if commands and configurations is not None:
        additions.append(EvidenceFact("dispatch.config_fraction", configurations / commands, "observed", witness))
    cca = cca_projection(model, op=op, source=source, backend=(target,), facts=tuple(facts) + tuple(additions))
    for axis, row in cca.provenance["axis_evidence"].items():
        if axis.startswith("dispatch.") or axis in ("memory.onchip_resident", "memory.panel_reuse"):
            row["scope"] = "complete_invocation"
    return cca


def lift_selected(raw, source_pin, *, symbol, workspace, reference_root, facts, contract, target, descriptor, limit):
    """Return a fresh typed lift only for a source in the original host selection."""
    from merlin.analysis.mining.references.capture import source_function
    from merlin.development.backend.feedback.references import analysis as A

    selected = os.environ.get(SELECTION)
    if not selected:
        return None
    selection = json.loads(selected)
    selected_path = Path(selection["path"])
    if (
        set(selection) != {"path", "sha256"}
        or not selected_path.is_absolute()
        or selected_path != selected_path.resolve()
        or not selected_path.is_relative_to(workspace / "cca-evidence")
    ):
        raise ValueError("analysis_capture_selection_invalid")
    manifest_raw, manifest_pin = A._read(selected_path, "cca-evidence/" + selected_path.name)
    if manifest_pin["sha256"] != selection["sha256"] or len(manifest_raw) > 4 << 20:
        raise ValueError("analysis_capture_selection_invalid")
    manifest = json.loads(manifest_raw)
    entries = manifest.get("entries")
    if manifest.get("schema") != SCHEMA or not isinstance(entries, list) or not 1 <= len(entries) <= 16:
        raise ValueError("analysis_capture_selection_invalid")
    matching = [row for row in entries if row.get("source", {}).get("path") == source_pin["path"]]
    if not matching:
        return None
    if symbol:
        matching = [row for row in matching if row.get("symbol") == symbol]
    if len(matching) != 1:
        raise ValueError("analysis_capture_source_mismatch")
    (entry,) = matching
    required = {
        "source",
        "symbol",
        "function_body_sha256",
        "function_source_sha256",
        "capture",
        "facts",
        "operand_protocol",
        "contract",
        "observer",
    }
    if set(entry) != required or entry["source"] != {k: source_pin[k] for k in ("path", "sha256")}:
        raise ValueError("analysis_capture_source_mismatch")
    function = source_function(raw, entry["symbol"])
    if (
        function["body_sha256"] != entry["function_body_sha256"]
        or hashlib.sha256(function["source"].encode()).hexdigest() != entry["function_source_sha256"]
    ):
        raise ValueError("analysis_capture_source_mismatch")
    payloads, pins = {}, {}
    for role in ("capture", "facts", "operand_protocol", "contract", "observer"):
        selected = entry[role]
        if set(selected) != ({"path", "sha256", "factory"} if role == "observer" else {"path", "sha256"}):
            raise ValueError("analysis_capture_selection_invalid")
        if not selected["path"].startswith(("cca-evidence/", "reference-kernels/")):
            raise ValueError("analysis_capture_selection_invalid")
        path, label = A._path(workspace, selected["path"], reference_root=reference_root)
        payloads[role], pins[role] = A._read(path, label)
        if pins[role]["sha256"] != selected["sha256"]:
            raise ValueError("analysis_capture_artifact_mismatch")
    decoder = A._operand_decoder(contract, facts, target, descriptor=descriptor)
    original = {row["path"]: row["sha256"] for row in decoder.get("decoder_inputs", [])}
    if (
        original.get("selected_rtl_facts") != pins["facts"]["sha256"]
        or original.get("selected_target_contract") != pins["contract"]["sha256"]
        or json.loads(payloads["operand_protocol"]).get("target") != target
    ):
        raise ValueError("analysis_capture_decoder_mismatch")
    records = [json.loads(line) for line in payloads["capture"].splitlines() if line.strip()]
    kinds = Counter(row.get("kind") for row in records if isinstance(row, dict))
    if (
        not records
        or len(records) > MAX_CAPTURE_COMMANDS + 1024
        or sum(kinds.values()) != len(records)
        or records[-1] != {"kind": "complete"}
        or kinds["complete"] != 1
        or set(kinds) - {"buffer", "probe", "command", "fence", "complete"}
        or kinds["command"] + kinds["fence"] > MAX_CAPTURE_COMMANDS
    ):
        raise ValueError("analysis_capture_records_invalid")
    arrays = {name: declaration for name, declaration in function["declarations"].items() if declaration["shape"]}
    buffers = [row for row in records if row["kind"] == "buffer"]
    widths = dict(i8=1, i16=2, i32=4, f32=4, bool=1)
    if len(buffers) != len(arrays) or {row.get("name") for row in buffers} != set(arrays):
        raise ValueError("analysis_capture_source_mismatch")
    for row in buffers:
        declared = arrays[row["name"]]
        if (
            type(row.get("address")) is not int
            or row["address"] <= 0
            or row.get("bytes") != math.prod(declared["shape"]) * widths[declared["dtype"]]
        ):
            raise ValueError("analysis_capture_source_mismatch")
    factory = entry["observer"]["factory"]
    if not isinstance(factory, str) or not factory.isidentifier() or len(payloads["observer"]) > 1 << 20:
        raise ValueError("analysis_capture_selection_invalid")
    # Only the original environment's hash-pinned tool is loaded. Neither the
    # request nor candidate bytes can select or replace this analysis function.
    module = types.ModuleType("merlin_selected_reference_capture_observer")
    module.__file__ = pins["observer"]["path"]
    exec(compile(payloads["observer"], module.__file__, "exec"), module.__dict__)
    observer = getattr(module, factory)
    if not callable(observer) or getattr(observer, "__module__", None) != module.__name__:
        raise ValueError("analysis_capture_selection_invalid")
    temporary_root = os.environ.get("TMPDIR")
    if not temporary_root or not Path(temporary_root).is_absolute():
        raise ValueError("analysis_requires_owner_task_temporary_root")
    with tempfile.TemporaryDirectory(prefix="reference-replay-", dir=temporary_root) as directory:
        directory = Path(directory)
        for role, filename in (
            ("facts", "facts.json"),
            ("operand_protocol", "operand-protocol.json"),
            ("capture", "capture.jsonl"),
        ):
            (directory / filename).write_bytes(payloads[role])
        capture = dict(
            function=function,
            records=records,
            capture_path=str(directory / "capture.jsonl"),
            artifacts=[
                dict(kind=role, uri=pin["path"], sha256=pin["sha256"], size_bytes=pin["size_bytes"])
                for role, pin in (
                    ("source", source_pin),
                    ("capture", pins["capture"]),
                    ("observer_source", pins["observer"]),
                )
            ],
        )
        try:
            observed = observer(
                capture,
                settings={"backend_root": str(directory), "max_commands": MAX_CAPTURE_COMMANDS},
                workspace=directory / "observed",
            )
        except ValueError as error:
            raise A._AnalysisGap("analysis_capture_replay_incomplete", witness=str(error)[:256]) from error
    region, evidence, model = (observed[key] for key in ("semantic_region", "evidence", "schedule_model"))
    if (
        not region.complete
        or not model.complete
        or evidence.capture_kind != "native_issue"
        or model.command_count != kinds["command"] + kinds["fence"]
    ):
        raise ValueError("analysis_capture_replay_incomplete")
    cca = _dispatch(
        model,
        op="matmul" if observed["problem"] else "projection_group",
        fences=kinds["fence"],
        configurations=sum(count for role, count in model.role_counts.items() if role == "configure"),
        target=target,
        facts=evidence.facts,
    )
    # Rebind temporary reader paths to their already checked public inputs.
    aliases = {pin["sha256"]: pin["path"] for pin in pins.values()}
    clean_evidence = evidence.to_dict()
    clean_evidence["artifacts"] = [
        dict(
            pin,
            uri=aliases.get(
                pin["sha256"],
                pin["uri"] if not Path(pin["uri"]).is_absolute() else "analysis_tool:" + Path(pin["uri"]).name,
            ),
        )
        for pin in clean_evidence["artifacts"]
    ]
    dataflow = observed["tile_dataflow"]
    from merlin.analysis.kernels.mechanisms.inference import repeated_read_mechanisms

    report = dict(
        kind="captured_reference_analysis",
        input=source_pin,
        symbol=entry["symbol"],
        selection=manifest_pin,
        capture=pins["capture"],
        capture_records=len(records),
        count_basis="complete_original_native_issue_records_replayed_with_selected_operand_protocol",
        count_scope="selected_function_body; caller preparation and wrapper commands excluded",
        semantic_region=region.to_dict(),
        mechanism_observations=repeated_read_mechanisms(model),
        problems=observed["problems"],
        tile_dataflow={key: value for key, value in dataflow.items() if key != "output_tiles"},
        output_tiles=len(dataflow["output_tiles"]),
        schedule_model=dict(
            identity=model.identity,
            source_kind=model.source_kind,
            command_count=model.command_count,
            complete=model.complete,
            role_counts=dict(model.role_counts),
            event_counts=dict(model.event_counts),
            instruction_roles={stage.identity: stage.roles for stage in model.stages},
            loops=[asdict(loop) for loop in model.loops[:limit]],
            loops_total=len(model.loops),
            dependencies=len(model.dependencies),
            storage_lifetimes=len(model.storage),
            unknowns=model.unknowns,
        ),
        evidence=clean_evidence,
        cca=cca.to_dict(),
        source_binding="selected_byte_preserved_function",
        completion="unknown",
        overlap="unknown",
        numerical_status="not_run",
        timing_status="unknown",
        capture_execution="not_rerun; replay_only",
        generalized_loop_proof="unknown",
        next_action="Compare a source-bound emitted artifact view; absent operand/residency witnesses remain UNKNOWN.",
    )
    return report, cca


def candidate_view(inputs, *, entry_symbol, decoder):
    """Count actual emitted IR through existing decoder and closed-CFG owners."""
    from xdsl.parser import Parser

    from merlin.analysis.observation.schedule_structure import ScheduleModel
    from merlin.analysis.programs.emitted_work import summarize_emitted_work
    from merlin.compiler.ir.llvm_context import make_llvm_context
    from merlin.compiler.ir.xdsl_io import IR_LOCK
    from merlin.development.backend.feedback.references.views import source_view
    from merlin.target.isa.rocc.decode import decode_module

    def gap(reason, *, required_adapter, **details):
        return dict(
            status="unknown",
            reason=reason,
            required_adapter=required_adapter,
            semantic_alignment="unknown",
            numerical_equivalence="unknown",
            execution="not_established",
            **details,
        ), None

    if not decoder.get("semantics"):
        return gap("candidate_selected_operand_semantics_unavailable", required_adapter="selected_operand_decoder")
    try:
        source = source_view(inputs["source"])
    except (ValueError, TypeError, KeyError, AttributeError):
        return gap(
            "candidate_source_to_command_buffer_adapter_unavailable",
            required_adapter="typed_source_to_command_buffer",
            supported_source_contract="closed_merlin_interface",
            required_obligations=["typed_product_operand_and_output_views", "ordered_numeric_contract"],
        )
    try:
        cb = source["command_buffer"]
        packs = {
            row["operands"]["dst"]: row["operands"]["src"] for row in cb["commands"] if row["opcode"] == "RES_PACK"
        }
        products = [row for row in cb["commands"] if row["opcode"] == "MATMUL_RESIDENT"]
        commits = [row for row in cb["commands"] if row["opcode"] == "COMMIT"]
        if len(products) != 1 or len(commits) != 1:
            return gap(
                "candidate_product_readout_cardinality_unsupported",
                required_adapter="typed_multi_product_view"
                if len(products) > 1 or len(commits) > 1
                else "complete_single_product_readout",
                source_counts=dict(products=len(products), commits=len(commits)),
                supported_counts=dict(products=1, commits=1),
                required_obligations=[
                    "per_product_operand_subviews",
                    "complete_output_coordinates",
                    "ordered_output_composition",
                    "per_product_numeric_contract",
                ],
            )
        product, commit = products[0]["operands"], commits[0]
        a, b = cb["tensors"][product["lhs"]], cb["tensors"][packs[product["rhs"]]]
        output = cb["tensors"][commit["operands"]["dst"]]
        attrs = commit["attributes"]
        if commit["operands"]["src"] != product["dst"]:
            return gap("candidate_product_readout_join_unavailable", required_adapter="typed_product_readout_view")
        if attrs.get("epilogue") != []:
            return gap(
                "candidate_numeric_epilogue_adapter_unavailable",
                required_adapter="ordered_numeric_epilogue",
                ordered_stages=attrs.get("epilogue"),
                required_obligations=[
                    "stage_order",
                    "scale_and_bias_operands",
                    "rounding_and_saturation",
                    "activation_and_readout_domain",
                ],
            )
        problem = dict(
            m=a["shape"][0],
            k=a["shape"][1],
            n=b["shape"][1],
            dtypes=[a["dtype"], b["dtype"], output["dtype"]],
            readout="raw",
        )
        with IR_LOCK:
            module = Parser(make_llvm_context(), inputs["emitted"].decode()).parse_module()
            entries = [op for op in module.walk() if op.name == "llvm.func" and op.sym_name.data == entry_symbol]
            if len(entries) != 1:
                return gap(
                    "candidate_emitted_entry_unavailable",
                    required_adapter="unique_emitted_function_entry",
                    matching_entries=len(entries),
                )
            (entry,) = entries
            trace = decode_module(module, target=decoder["target"], semantics=decoder["semantics"])
            sites = [op for op in module.walk() if op.name == "llvm.inline_asm"]
            classes = {id(op): row["class"] for op, row in zip(sites, trace["instructions"], strict=True)}
            counted = summarize_emitted_work(entry, command_kind=lambda op: classes.get(id(op)))
        counts = counted["command_counts"]
        if counts.get("UNKNOWN") or not counted["command_classification_complete"]:
            return gap(
                "candidate_emitted_command_classification_incomplete",
                required_adapter="complete_selected_instruction_decoder",
            )
        fences = sum(count for kind, count in counts.items() if kind == "FENCE")
        commands = sum(counts.values()) - fences
        roles = (decoder.get("structure") or {}).get("instruction_roles") or {}
        configurations = (
            sum(count for name, count in counts.items() if "config" in roles.get(name, ()))
            if all(roles.get(name) for name in counts if name != "FENCE")
            else None
        )
        model = ScheduleModel(
            source_kind="counted_emitted_ir",
            complete=True,
            command_count=commands + fences,
            event_counts={"command": commands, "fence": fences},
            evidence_id="emitted:" + hashlib.sha256(inputs["emitted"]).hexdigest(),
            unknowns=("hardware_completion_and_timing_not_established", "operand_and_storage_dataflow_not_recovered"),
        )
        cca = _dispatch(
            model,
            op="matmul",
            fences=fences,
            configurations=configurations,
            target=decoder["target"],
            source="candidate",
        )
        return dict(
            status="observed_counted_emission",
            problem=problem,
            command_counts=counts,
            scope="complete_invocation",
            basis=counted["domain"],
            cca=cca.to_dict(),
            count_scope="selected_function_body; caller preparation and wrapper commands excluded",
            residency="unknown",
            traffic="unknown",
            completion="unknown",
            timing="unknown",
        ), cca
    except (ValueError, TypeError, KeyError, AttributeError):
        return gap("candidate_emitted_count_analysis_incomplete", required_adapter="complete_emitted_ir_counter")


def compare_candidate(reference, candidate, expert, ours):
    from merlin.analysis.kernels.cca.comparison import compare_evidence

    projected = candidate.get("captured_reference_axes") or {}
    problems = reference["problems"]
    actual = projected.get("problem", {})
    projection_gap = (
        {
            key: projected[key]
            for key in (
                "reason",
                "required_adapter",
                "source_counts",
                "supported_counts",
                "ordered_stages",
                "required_obligations",
            )
            if key in projected
        }
        if projected.get("status") == "unknown"
        else None
    )
    compatible = (
        len(problems) == 1
        and all(problems[0].get(key) == actual.get(key) for key in ("m", "k", "n", "dtypes"))
        and actual.get("readout") == "raw"
        and problems[0]["numeric"]
        == dict(accumulator_dtype="i32", bias="none", scale=1, activation="none", rounding="none", saturation="none")
        and not candidate.get("mismatches")
    )
    comparison = (
        compare_evidence(expert, ours).to_dict()
        if compatible and ours is not None
        else dict(
            status="unknown",
            denominator=0,
            compared_axes=[],
            divergences=[],
            unknowns=[projection_gap or dict(reason="declared_contraction_or_candidate_count_not_aligned")],
        )
    )
    roles = reference["schedule_model"].get("instruction_roles", {})
    candidate_roles = Counter()
    unmatched = {}
    for instruction, count in projected.get("command_counts", {}).items():
        selected_roles = roles.get(instruction, roles.get("fence", ()) if instruction == "FENCE" else ())
        if not selected_roles:
            unmatched[instruction] = count
        for role in selected_roles:
            candidate_roles[role] += count
    role_counts = reference["schedule_model"]["role_counts"]
    return dict(
        kind="captured_reference_candidate_comparison",
        reference=reference,
        candidate=candidate,
        cca=comparison,
        declared_contraction_alignment="matches" if compatible else "unknown_or_mismatch",
        command_role_comparison={
            role: dict(
                reference=count,
                candidate=sum(count for name, count in candidate_roles.items() if name == role)
                if ours is not None
                else None,
                status="compared_function_body_counts" if compatible and ours is not None else "unknown",
            )
            for role, count in role_counts.items()
        },
        unmatched_candidate_commands=unmatched,
        command_role_basis="Observed reference instruction roles only; unmatched candidate classes remain separate.",
        source_layout_adapter="unknown",
        semantic_equivalence="unknown",
        numerical_status="unknown",
        completion="unknown",
        overlap="unknown",
        performance="unknown",
        schedule_consumption="unknown",
        next_action="Use CCA-supported differences to propose ordinary shared-pass alternatives; verify every output "
        "and compare complete matched cycles. Missing candidate traffic/residency is an explicit gap.",
    )


# ---- reference_schedule_receipts ---------------------------------------------------------------
# Read source-bound construction receipts without upgrading them to execution.
#
# The scheduling library records its consumed decision IR in command buffers.
# These records are useful even when the original emission owner is unavailable,
# but supplied records cannot authenticate their consumer, LLVM, or machine code.

MAX_NODES = 16384

MAX_DEPTH = 32

MAX_RECEIPTS = 32

MAX_TRANSFORMS = 64

MAX_TEXT_BYTES = 65536

_CONSUMED = frozenset(
    ("block_tiles", "outer_order", "row_bases", "panel_pipeline", "pipeline_window_offsets", "loop_transforms")
)

_HASH_FIELDS = frozenset(
    (
        "kernel_sha256",
        "actual_kernel_sha256",
        "source_sha256",
        "actual_sha256",
        "checking_kernel_sha256",
        "original_kernel_sha256",
        "expanded_kernel_sha256",
        "checking_expansion_sha256",
        "phase_kernel_sha256",
    )
)


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
    from merlin.compiler.ir.xdsl_io import IR_LOCK
    from merlin.compiler.schedule.dense.plans import parse_dense_schedule_plan

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
    if (
        not isinstance(consumed, (list, tuple))
        or not consumed
        or any(type(v) is not str or v not in _CONSUMED for v in consumed)
        or len(set(consumed)) != len(consumed)
    ):
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
            if (
                any(not _positive(block.get(key), 3) for key in ("problem", "primitive_tile", "block_tiles"))
                or any(not _positive(block.get(key), 2) for key in ("storage_bytes", "storage_capacity"))
                or any(
                    used > capacity
                    for used, capacity in zip(block["storage_bytes"], block["storage_capacity"], strict=True)
                )
            ):
                return dict(result, reason="recorded_geometry_or_capacity_invalid")
            pipelines = [
                {key: _attribute(value) for key, value in op.properties.items()}
                for op in operations
                if op.name == "schedule.panel_pipeline"
            ]
            transforms = [
                {key: _attribute(value) for key, value in op.properties.items()}
                for op in operations
                if op.name == "schedule.loop_transform"
            ]
            if len(pipelines) > 1:
                return dict(result, reason="pipeline_decision_scope_not_unique")
            pipeline = pipelines[0] if pipelines else {}
            for external, internal in (
                ("pipeline_kind", "kind"),
                ("prefetch", "prefetch"),
                ("pipeline_issue_order", "issue_order"),
                ("pipeline_window_offsets", "window_offsets"),
            ):
                if record.get(external) is not None and record[external] != pipeline.get(internal):
                    return dict(result, reason="receipt_pipeline_differs_from_decision_ir")
            if "panel_pipeline" in consumed and not pipeline or "loop_transforms" in consumed and not transforms:
                return dict(result, reason="consumed_decision_absent_from_ir")
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        return dict(result, reason="schedule_decision_parse_or_verification_refused", error=type(error).__name__)
    except Exception as error:  # xDSL parse/verify exception classes vary across selected releases.
        from xdsl.utils.exceptions import ParseError, VerifyException

        if not isinstance(error, (ParseError, VerifyException)):
            raise
        return dict(result, reason="schedule_decision_parse_or_verification_refused", error=type(error).__name__)
    return dict(
        result,
        status="source_bound_recorded_consumption",
        consumption="recorded_consumer_claim",
        context_sha256=fields[2],
        consumed=list(consumed),
        choices=dict(block=block, pipeline=pipeline, loop_transforms=transforms),
        capacity_condition=dict(
            status="within_recorded_private_capacity",
            used_bytes=block["storage_bytes"],
            capacity_bytes=block["storage_capacity"],
            selected_hardware_and_simultaneous_lifetimes="unknown",
        ),
        source_shape_correspondence="unknown",
        emitted_correspondence="unknown",
        target_execution="unknown",
    )


def inspect_schedule_receipts(*, source_mlir, command_buffer, emitted_mlir, producer_binding=None):
    """Reopen bounded decision text and hash links in an actual supplied CB.

    No aggregate loads, traffic, fences, timing or CCA axes are filled from
    construction estimates. The original compiler/build owner must separately
    establish that these exact CB/LLVM bytes were its consumed products.
    """
    pins = {
        role: _sha(raw)
        for role, raw in (("source", source_mlir), ("command_buffer", command_buffer), ("emitted", emitted_mlir))
    }
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
                    receipts.append(
                        _receipt(value["schedule_ir"], source_sha256=pins["source"], pointer=pointer + "/schedule_ir")
                    )
                else:
                    omitted.append("schedule_receipt_budget_exhausted")
            schema = value.get("schema")
            if type(schema) is str and schema.startswith("merlin."):
                links = {}
                for key in sorted(_HASH_FIELDS & value.keys()):
                    item = value[key]
                    if (
                        _hash(item)
                        or isinstance(item, dict)
                        and 0 < len(item) <= 32
                        and all(type(k) is str and len(k) <= 128 and _hash(v) for k, v in item.items())
                    ):
                        links[key] = item
                if links:
                    if len(transformations) < MAX_TRANSFORMS:
                        transformations.append(
                            dict(
                                pointer=pointer,
                                schema=schema,
                                hash_links=links,
                                status="recorded_construction_links",
                                executable_correspondence="unknown",
                            )
                        )
                    else:
                        omitted.append("transformation_record_budget_exhausted")
            stack.extend(
                (item, pointer + "/" + str(key).replace("~", "~0").replace("/", "~1"), depth + 1)
                for key, item in reversed(list(value.items()))
                if key != "schedule_ir"
            )
        elif isinstance(value, list):
            stack.extend(
                (item, pointer + "/" + str(index), depth + 1) for index, item in reversed(list(enumerate(value)))
            )
    valid = [row for row in receipts if row["status"] == "source_bound_recorded_consumption"]
    result = dict(
        schema="merlin.reference_schedule_receipts.v1",
        status="partial" if omitted else "observed_recorded_consumption" if valid else "unknown",
        inputs=pins,
        records=receipts,
        valid_records=len(valid),
        invalid_records=len(receipts) - len(valid),
        transformation_records=transformations,
        budget_status="partial" if omitted else "complete",
        missing=sorted(set(omitted))
        + [
            "original_compiler_command_buffer_emission_owner",
            "instruction_kernel_to_emitted_function_correspondence",
            "selected_hardware_lifetime_legality",
            "complete_reference_candidate_semantic_alignment",
        ],
        schedule_consumption="recorded_only" if valid else "unknown",
        compiler_and_emission_binding="unknown",
        semantic_equivalence="unknown",
        target_execution="unknown",
        next_action="Bind the original immutable compiler operation and its exact CB/LLVM products; "
        "inspect recorded blocking, capacity and pipeline choices without treating them as executed work.",
    )
    if producer_binding is not None:
        # This argument is supplied only by the original host's protected build
        # index after its ownership checks; author artifact manifests have no
        # producer-binding field. Hash equality here guards this adapter's join,
        # and cannot by itself issue an original build or semantic witness.
        required = ("compiler_sha256", "object_sha256", "linked_elf_sha256", "function_view_sha256")
        matches = (
            isinstance(producer_binding, dict)
            and producer_binding.get("status") == "observed_original_build_consumption"
            and all(producer_binding.get(role + "_sha256") == value for role, value in pins.items())
            and all(_hash(producer_binding.get(key)) for key in required)
        )
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
        return {
            key: [dict(value=json.loads(value), records=count) for value, count in sorted(counts.items())]
            for key, counts in sorted(values.items())
        }

    left, right = profiles(reference), profiles(candidate)
    source = reference.get("inputs", {}).get("source")
    comparable = (
        bool(left)
        and bool(right)
        and _hash(source)
        and source == candidate.get("inputs", {}).get("source")
        and reference.get("budget_status") == candidate.get("budget_status") == "complete"
        and not reference.get("invalid_records")
        and not candidate.get("invalid_records")
    )
    return dict(
        status="compared_recorded_choice_populations" if comparable else "independent_recorded_choices",
        source_identity="identical_bytes" if comparable else "unknown_or_different",
        choices=[
            dict(
                axis="schedule." + axis,
                reference=left.get(axis),
                candidate=right.get(axis),
                differs=left.get(axis) != right.get(axis) if comparable else None,
                status="compared_recorded_population" if comparable else "unknown",
            )
            for axis in sorted(left.keys() | right.keys())
        ],
        scope="recorded decision populations; group/source-operation alignment unknown",
        canonical_cca_axes_added=0,
        semantic_equivalence="unknown",
        optimization_application="unknown",
        performance="unknown",
    )
