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

from merlin.kernels.kernel_evidence import EvidenceFact, cca_projection

SELECTION = "MERLIN_REFERENCE_CAPTURE_SELECTION"
SCHEMA = "merlin.reference_capture_selection.v1"
# Match the native issue owner's maximum finite capture; replay never truncates it.
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
    from merlin.kernels.reference_capture import source_function

    from . import reference_analysis as A

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
        configurations=model.role_counts.get("configure", 0),
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
    from merlin.kernels.mechanism_inference import repeated_read_mechanisms

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

    from merlin.common.ir_lock import IR_LOCK
    from merlin.compile.analysis.schedule_structure import ScheduleModel
    from merlin.perf.emitted_work import summarize_emitted_work
    from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context
    from merlin.targetgen.rocc.decode import decode_module

    from .reference_views import source_view

    def gap(reason, *, required_adapter, **details):
        return dict(status="unknown", reason=reason, required_adapter=required_adapter,
                    semantic_alignment="unknown", numerical_equivalence="unknown",
                    execution="not_established", **details), None

    if not decoder.get("semantics"):
        return gap("candidate_selected_operand_semantics_unavailable", required_adapter="selected_operand_decoder")
    try:
        source = source_view(inputs["source"])
    except (ValueError, TypeError, KeyError, AttributeError):
        return gap("candidate_source_to_command_buffer_adapter_unavailable",
                   required_adapter="typed_source_to_command_buffer",
                   supported_source_contract="closed_merlin_interface",
                   required_obligations=["typed_product_operand_and_output_views", "ordered_numeric_contract"])
    try:
        cb = source["command_buffer"]
        packs = {
            row["operands"]["dst"]: row["operands"]["src"] for row in cb["commands"] if row["opcode"] == "RES_PACK"
        }
        products = [row for row in cb["commands"] if row["opcode"] == "MATMUL_RESIDENT"]
        commits = [row for row in cb["commands"] if row["opcode"] == "COMMIT"]
        if len(products) != 1 or len(commits) != 1:
            return gap("candidate_product_readout_cardinality_unsupported",
                       required_adapter="typed_multi_product_view" if len(products) > 1 or len(commits) > 1
                       else "complete_single_product_readout",
                       source_counts=dict(products=len(products), commits=len(commits)),
                       supported_counts=dict(products=1, commits=1),
                       required_obligations=["per_product_operand_subviews", "complete_output_coordinates",
                                             "ordered_output_composition", "per_product_numeric_contract"])
        product, commit = products[0]["operands"], commits[0]
        a, b = cb["tensors"][product["lhs"]], cb["tensors"][packs[product["rhs"]]]
        output = cb["tensors"][commit["operands"]["dst"]]
        attrs = commit["attributes"]
        if commit["operands"]["src"] != product["dst"]:
            return gap("candidate_product_readout_join_unavailable", required_adapter="typed_product_readout_view")
        if attrs.get("epilogue") != []:
            return gap("candidate_numeric_epilogue_adapter_unavailable", required_adapter="ordered_numeric_epilogue",
                       ordered_stages=attrs.get("epilogue"),
                       required_obligations=["stage_order", "scale_and_bias_operands", "rounding_and_saturation",
                                             "activation_and_readout_domain"])
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
                return gap("candidate_emitted_entry_unavailable", required_adapter="unique_emitted_function_entry",
                           matching_entries=len(entries))
            (entry,) = entries
            trace = decode_module(module, target=decoder["target"], semantics=decoder["semantics"])
            sites = [op for op in module.walk() if op.name == "llvm.inline_asm"]
            classes = {id(op): row["class"] for op, row in zip(sites, trace["instructions"], strict=True)}
            counted = summarize_emitted_work(entry, command_kind=lambda op: classes.get(id(op)))
        counts = counted["command_counts"]
        if counts.get("UNKNOWN") or not counted["command_classification_complete"]:
            return gap("candidate_emitted_command_classification_incomplete",
                       required_adapter="complete_selected_instruction_decoder")
        fences = counts.get("FENCE", 0)
        commands = sum(counts.values()) - fences
        roles = (decoder.get("structure") or {}).get("instruction_roles") or {}
        configurations = (sum(count for name, count in counts.items() if "config" in roles.get(name, ()))
                          if all(roles.get(name) for name in counts if name != "FENCE") else None)
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
    from merlin.kernels.cca_compare import compare_evidence

    projected = candidate.get("captured_reference_axes") or {}
    problems = reference["problems"]
    actual = projected.get("problem", {})
    projection_gap = ({key: projected[key] for key in (
        "reason", "required_adapter", "source_counts", "supported_counts", "ordered_stages", "required_obligations"
    ) if key in projected} if projected.get("status") == "unknown" else None)
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
                candidate=candidate_roles.get(role, 0) if ours is not None else None,
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
