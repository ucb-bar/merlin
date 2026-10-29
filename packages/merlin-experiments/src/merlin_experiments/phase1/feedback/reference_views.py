"""Cross-check original source, declared commands and emitted artifacts.

This is diagnostic evidence, not a source proof, execution owner or cost model.
Native event records retain their recorded origin; a content pin cannot issue
an execution, numerical or timing verdict.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict

from merlin.verify.lean_contract import document_digest

MAX_BYTES = 8 << 20
MAX_OPERATIONS = 100_000


def _pin(raw):
    if type(raw) is not bytes or len(raw) > MAX_BYTES:
        raise ValueError("artifact view requires bounded original bytes")
    return dict(sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def source_view(source_mlir, *, reduction_lanes=None):
    """Observe SSA operand sharing and every original declared result type.

    Reuse the closed interface grammar, including its shape tokenizer. The
    command parser omits COMMIT result tensors, so those types must be read
    independently from the original declaration before joining a candidate CB.
    """
    from merlin.compile.logical_abi import (
        HarnessRenderError,
        commit_output_shape,
        kernel_abi_from_commands,
        kernel_arg_order,
    )
    from merlin.perf.candidate_source import source_observation
    from merlin.runtime.commandbuffer import EPILOGUE_STAGE_SET
    from merlin.targetgen.contract import interface_emit as grammar

    pin = _pin(source_mlir)
    text = source_mlir.decode("utf-8")
    observed = source_observation(text)
    cb = grammar.parse_interface_mlir(text)
    outputs = {}
    for line in text.splitlines():
        op = grammar._op_line(line.strip())
        if op is None or op["mnemonic"] != "commit":
            continue
        attrs = grammar._parse_attr_block(op["attrs_body"])
        name = attrs.get("name", op["result"])
        shape, dtype = grammar._shape_dtype(grammar._last_type(op["tail"]))
        if name in outputs or not name or any(type(d) is not int or d <= 0 for d in shape):
            raise ValueError("source readouts require unique static typed results")
        outputs[name] = dict(shape=shape, dtype=dtype, role="output")
    # Whole interface operations declare their named results directly. The
    # original parser retains these types; COMMIT is only one output form.
    for name, spec in cb["tensors"].items():
        if spec.get("role") == "output" and name not in outputs:
            if any(type(d) is not int or d <= 0 for d in spec.get("shape", ())):
                raise ValueError("source readouts require unique static typed results")
            outputs[name] = dict(spec)
    cb["tensors"] = {**cb["tensors"], **outputs}
    original_outputs = kernel_abi_from_commands(cb)["outputs"]
    outputs = {name: outputs[name] for name in original_outputs}
    products, groups, packs = {}, defaultdict(list), {}
    for ordinal, command in enumerate(cb["commands"]):
        operands = command.get("operands", {})
        if command["opcode"] == "RES_PACK":
            if operands["dst"] in packs:
                raise ValueError("source resident SSA handle is defined more than once")
            packs[operands["dst"]] = operands["src"]
        elif command["opcode"] == "EVICT":
            if packs.pop(operands["handle"], None) is None:
                raise ValueError("source evicts an unknown or already evicted handle")
        elif command["opcode"] == "MATMUL_RESIDENT":
            lhs, rhs = operands["lhs"], packs.get(operands["rhs"])
            if lhs not in cb["tensors"] or rhs not in cb["tensors"]:
                raise ValueError("source product operands are unresolved")
            a, b = cb["tensors"][lhs], cb["tensors"][rhs]
            if len(a["shape"]) != 2 or len(b["shape"]) != 2 or a["shape"][1] != b["shape"][0]:
                raise ValueError("source product operand shapes disagree")
            if operands["dst"] in products:
                raise ValueError("source product SSA result is defined more than once")
            products[operands["dst"]] = dict(ordinal=ordinal, lhs=lhs, rhs=rhs,
                shape=[a["shape"][0], a["shape"][1], b["shape"][1]],
                dtypes=[a["dtype"], b["dtype"]])
        elif command["opcode"] == "COMMIT":
            product = products.get(operands["src"])
            result = outputs.get(operands["dst"])
            attrs = command.get("attributes", {})
            if product is None or result is None:
                raise ValueError("source readout is not bound to its original product")
            if result["dtype"] != attrs.get("output_dtype") or "epilogue" not in attrs:
                raise ValueError("source readout numeric contract is incomplete or inconsistent")
            stages = attrs["epilogue"]
            if not isinstance(stages, list) or any(stage not in EPILOGUE_STAGE_SET for stage in stages):
                raise ValueError("source readout epilogue is outside the original contract")
            # Bias/scale/ReLU/requantization preserve the output domain. The
            # existing ABI owner also derives a fully specified pooled domain;
            # nonempty epilogues cannot excuse an arbitrary result shape.
            try:
                expected_shape = list(commit_output_shape(cb, operands["src"], attrs))
            except (ValueError, HarnessRenderError) as error:
                raise ValueError("analysis_source_readout_geometry_unknown") from error
            if result["shape"] != expected_shape:
                raise ValueError("analysis_source_readout_geometry_mismatch")
            groups[product["rhs"]].append(dict(**product, output=operands["dst"],
                result=result, readout_attributes=attrs))
    grouped = [dict(rhs=rhs, members=members) for rhs, members in groups.items() if len(members) > 1]
    from .reference_mechanisms import interface_sharing

    return dict(status="observed_source", input=pin, observation=observed,
        command_buffer=cb, groups=grouped, products=len(products), outputs=outputs,
        mechanism_observations=interface_sharing(products.values(), pin=pin, command_buffer=cb, lanes=reduction_lanes),
        argument_order=kernel_arg_order(cb), source_semantics="declared_interface_contract",
        numerical_equivalence="unknown", emitted_mechanism="unknown")


def artifact_views(*, source_mlir, command_buffer, emitted_mlir, entry_symbol="kernel_entry",
                   native_capture=None, isa=None, native_decoder=None, semantics=None, target=None,
                   decoder_inputs=(), reduction_lanes=None, producer_binding=None):
    """Reopen all supplied byte domains and diagnose mismatches before proposing work.

    ``native_capture`` is host-supplied ``{raw: bytes, evidence: KernelEvidence,
    abi_arity: int}``, never a summary-count document. Even a fully matching
    record remains recorded evidence without an original producer owner.
    """
    from xdsl.dialects import llvm
    from xdsl.parser import Parser
    from xdsl.utils.exceptions import ParseError, VerifyException

    from merlin.common.ir_lock import IR_LOCK
    from merlin.perf.command_buffer_diagnostics import representation_activity
    from merlin.perf.emitted_work import assembly_site_kind, summarize_emitted_work
    from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context

    from .reference_schedule_receipts import inspect_schedule_receipts

    pins = dict(source=_pin(source_mlir), command_buffer=_pin(command_buffer), emitted=_pin(emitted_mlir))
    if b"merlin_iface." in source_mlir:
        source = source_view(source_mlir, reduction_lanes=reduction_lanes)
    else:
        from .reference_source_structure import source_structure

        source = source_structure(source_mlir, pin=pins["source"], reduction_lanes=reduction_lanes)
    cb = json.loads(command_buffer)
    if (not isinstance(cb, dict) or not isinstance(cb.get("tensors"), dict)
            or not isinstance(cb.get("commands"), list)
            or any(not isinstance(row, dict) for row in cb["tensors"].values())):
        raise ValueError("command buffer must be an object")
    expected = source.get("command_buffer")
    mismatches = []
    if expected is not None:
        for key in ("abi_version", "target", "commands"):
            if cb.get(key) != expected.get(key):
                mismatches.append(dict(axis="source." + key, status="mismatch"))
        for name, spec in expected["tensors"].items():
            actual = cb.get("tensors", {}).get(name, {})
            if any(actual.get(key) != spec.get(key) for key in ("shape", "dtype", "role")):
                mismatches.append(dict(axis="source.tensor", tensor=name, status="mismatch"))
        if set(cb.get("tensors", {})) != set(expected["tensors"]):
            mismatches.append(dict(axis="source.tensor_roster", status="mismatch"))
    abi = cb.get("kernel_abi")
    if abi is not None:
        if (not isinstance(abi, dict) or not isinstance(abi.get("args"), list)
                or any(not isinstance(row, dict) for row in abi["args"])):
            raise ValueError("command buffer ABI must declare typed argument records")
        args = [row.get("tensor") for row in abi.get("args", [])]
        if expected is not None and (args != source["argument_order"] or abi.get("outputs") != list(source["outputs"])):
            mismatches.append(dict(axis="source.abi_roster", status="mismatch"))
    try:
        with IR_LOCK:
            module = Parser(make_llvm_context(), emitted_mlir.decode("utf-8")).parse_module()
        module.verify()
    except (ParseError, VerifyException) as error:
        raise ValueError("analysis_emitted_invalid") from error
    entries = [op for op in module.walk() if isinstance(op, llvm.FuncOp)
               and op.sym_name.data == entry_symbol and op.body.blocks]
    if len(entries) != 1 or sum(1 for _ in module.walk()) > MAX_OPERATIONS:
        raise ValueError("emitted view requires one bounded defined entry")
    entry, = entries
    from merlin.kernels.consumer_structure import consumer_structure

    entry_block = next(iter(entry.body.blocks))
    arity = len(entry_block.args)
    if expected is not None and (arity != len(source["argument_order"]) or any(
            not isinstance(arg.type, llvm.LLVMPointerType) for arg in entry_block.args)):
        mismatches.append(dict(axis="emitted.abi_arity", expected=len(source["argument_order"]),
                               actual=arity, status="mismatch"))
    static = Counter(assembly_site_kind(op) for op in entry.walk() if op.name == "llvm.inline_asm")
    decoded_static = dict(status="unknown", reason="original_selected_operand_decoder_absent",
                          command_counts=None, decoded_operands="unknown")
    if semantics is not None:
        from merlin.targetgen.rocc.decode import decode_module

        trace = decode_module(module, target=target, semantics=semantics, source=pins["emitted"]["sha256"])
        entry_sites = {op for op in entry.walk() if op.name == "llvm.inline_asm"}
        positions = {i for i, op in enumerate(op for op in module.walk() if op.name == "llvm.inline_asm")
                     if op in entry_sites}
        rows = [row for row in trace["instructions"] if row["index"] in positions]
        decoded_static = dict(status="observed_static_sites", summary=dict(
            entry_sites=len(rows), unknown_sites=sum(row["class"] == "UNKNOWN" for row in rows),
            unknown_operand_fields=sum(value is None for row in rows for value in row["decoded"].values())),
            command_counts=dict(Counter(row["class"] for row in rows)),
            decoded_operands="partial", scope="actual entry static sites; unknown SSA operands stay unknown",
            dynamic_multiplicity="unknown", source_tensor_binding="unknown", inputs=list(decoder_inputs))
        if isa is None:
            isa = semantics.isa_constants(target)
        if native_decoder is None:
            native_decoder = semantics.decode_instruction
    try:
        counted = dict(status="observed_static_cfg", **summarize_emitted_work(
            entry, command_kind=assembly_site_kind, max_operations=MAX_OPERATIONS))
    except ValueError as error:
        counted = dict(status="unknown", reason=str(error), command_counts=None)
    native = _native(native_capture, pins, arity, isa=isa, decoder=native_decoder) if native_capture else dict(
        status="unknown", reason="original_native_capture_absent", command_counts=None, fence_count=None)
    construction = inspect_schedule_receipts(source_mlir=source_mlir, command_buffer=command_buffer,
                                             emitted_mlir=emitted_mlir, producer_binding=producer_binding)
    gaps = [dict(kind="evidence_or_adapter_gap", goals=[row["axis"]], reason="cross-view mismatch",
                 witnesses=[row], next_action="repair_original_source_command_or_emitted_binding")
            for row in mismatches]
    if native["status"] == "unknown":
        gaps.append(dict(kind="adapter_gap", goals=["evidence.native_issue"], reason=native["reason"],
            next_action="capture_full_original_abi_then_decode_actual_operands_with_selected_isa"))
    if semantics is None:
        gaps.append(dict(kind="adapter_gap", goals=["evidence.selected_operand_decoder"],
            reason=decoded_static["reason"], next_action="bind_original_contract_facts_operand_observer"))
    return dict(kind="artifact_cross_view_analysis", inputs=pins,
        source={k: v for k, v in source.items() if k not in ("command_buffer", "observation")},
        declared_representations=representation_activity(cb),
        source_command_join=("mismatch" if mismatches else
                             "matches_declared_source" if expected is not None else "unknown"),
        emitted=dict(entry=entry_symbol, abi_arity=arity, static_assembly_sites=dict(static), counted_work=counted,
                     consumer_structure=consumer_structure(module, symbol=entry_symbol),
                     selected_decode=decoded_static),
        native=native, gaps=gaps, mismatches=mismatches, construction_receipts=construction,
        semantic_equivalence="unknown", mechanism_status="unknown",
        schedule_consumption=construction["schedule_consumption"],
        numerical_status="unknown", timing_status="unknown", heldout_status="unknown",
        next_action="Compare complete-region alternatives; retain load, configuration, fence and alias costs "
                    "separately.")


def _native(capture, pins, arity, *, isa, decoder):
    from merlin.kernels.kernel_evidence import KernelEvidence, analyze_native_commands

    raw, evidence = capture.get("raw"), capture.get("evidence")
    def unknown(reason):
        return dict(status="unknown", reason=reason, command_counts=None, fence_count=None)
    if type(evidence) is not KernelEvidence or type(raw) is not bytes or len(raw) > MAX_BYTES:
        return unknown("original_native_capture_or_evidence_missing")
    kinds = defaultdict(set)
    for pin in evidence.artifacts:
        kinds[pin.kind].add(pin.sha256)
    required = {"source": pins["source"]["sha256"], "command_buffer": pins["command_buffer"]["sha256"],
                "emitted": pins["emitted"]["sha256"], "capture": hashlib.sha256(raw).hexdigest()}
    if any(kinds[kind] != {value} for kind, value in required.items()):
        return unknown("native_capture_artifact_binding_mismatch")
    if type(capture.get("abi_arity")) is not int or capture["abi_arity"] != arity:
        return unknown("native_capture_abi_arity_mismatch")
    if not isinstance(isa, dict) or kinds["isa"] != {document_digest(isa)}:
        return unknown("native_capture_selected_isa_unbound")
    counts, rectangles, fields = Counter(), Counter(), defaultdict(set)
    def project(row):
        if row.get("class") == "FENCE":
            counts["FENCE"] += 1
            return dict(kind="fence", identity="FENCE", parameters={})
        cls, decoded = (decoder(row["funct"], dict(raw=row["rs1"], kind="const"),
                               dict(raw=row["rs2"], kind="const"), isa)
                        if decoder else (row["class"], row["decoded"]))
        counts[cls] += 1
        fields[cls].update(decoded)
        if type(decoded.get("rows")) is int and type(decoded.get("cols")) is int:
            rectangles[(cls, decoded["rows"], decoded["cols"])] += 1
        return dict(identity=cls, parameters=dict(decoded=decoded))
    try:
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
        if not records or len(records) > MAX_OPERATIONS:
            return unknown("native_capture_event_roster_empty_or_exceeds_bound")
        model = analyze_native_commands(records, isa=isa, evidence=evidence, max_commands=MAX_OPERATIONS,
                                       project_command=project)
        if counts.get("UNKNOWN") or model.command_count != len(records):
            return unknown("native_capture_operand_decode_incomplete")
    except (ValueError, TypeError, KeyError):
        return unknown("native_capture_operand_decode_failed")
    return dict(status="recorded_unverified_execution", evidence=evidence.to_dict(),
        count_status="observed_raw_issue_records", capture_complete=evidence.complete,
        isa_sha256=document_digest(isa),
        model_identity=model.identity, command_counts={k: v for k, v in counts.items() if k != "FENCE"},
        fence_count=counts.get("FENCE", 0),
        count_basis="decoded_original_raw_issue_records", executed_region="unknown",
        operand_projection="host_selected_decoder" if decoder else "unknown",
        decoded_fields={key: sorted(value) for key, value in fields.items()},
        transfer_rectangles=[dict(instruction=key[0], rows=key[1], cols=key[2], count=value,
                                  source_tensor_binding="unknown") for key, value in sorted(rectangles.items())],
        traffic_bytes=None, physical_cycles=None,
        numerical_execution="unknown", original_producer_custody="unverified",
        residency="unknown", dependency_safety="unknown", storage_lifetimes="unknown")


def compare_views(reference, candidate):
    """Report supported tradeoffs; never reward retention or fewer commands alone."""
    from .reference_mechanisms import align_questions
    from .reference_schedule_receipts import compare_schedule_receipts

    compatible = (reference["inputs"]["source"] == candidate["inputs"]["source"]
                  and reference["source_command_join"] == candidate["source_command_join"] == "matches_declared_source"
                  and not reference["mismatches"] and not candidate["mismatches"])
    counts = {}
    complete_scope = (reference["native"].get("capture_complete") is True
                      and candidate["native"].get("capture_complete") is True
                      and reference["native"].get("isa_sha256") == candidate["native"].get("isa_sha256"))
    for name in ("command_counts", "fence_count"):
        values = [row["native"].get(name) for row in (reference, candidate)]
        counts[name] = dict(reference=values[0], candidate=values[1],
            status=("recorded_complete_issue_scope" if complete_scope else "recorded_partial_or_incompatible_scope")
                   if compatible and all(value is not None for value in values) else "unknown")
    gaps = [*reference["gaps"], *candidate["gaps"]]
    if compatible and reference["source"]["groups"]:
        known = all(row["status"] == "recorded_complete_issue_scope" for row in counts.values())
        differs = known and any(row["reference"] != row["candidate"] for row in counts.values())
        gaps.append(dict(kind="scoped_mechanism_difference" if differs else "adapter_gap",
            goals=["dispatch.complete_region_tradeoff"],
            reason=("Recorded issue mixes differ; compare all costs rather than rewarding retention"
                    if differs else "Shared RHS source needs bound complete-region schedule/issue evidence"),
            next_action="compare_declared_shared_rhs_and_epoch_alternatives_then_reanalyze_all_issue_costs",
            witnesses=[dict(status="observed_source", source=reference["inputs"]["source"],
                            issue_comparison=counts, execution_origin="unverified")]))
    return dict(kind="artifact_cross_view_comparison", reference=reference, candidate=candidate, gaps=gaps,
        construction_comparison=compare_schedule_receipts(reference.get("construction_receipts", {}),
                                                         candidate.get("construction_receipts", {})),
        mechanism_alignment=align_questions(
            dict(views=dict(source=reference["source"], emitted=reference["emitted"])),
            dict(views=dict(source=candidate["source"], emitted=candidate["emitted"]))),
        declared_source_join="identical_bytes" if compatible else "unknown_or_mismatch",
        issue_comparison=counts, mechanism_status="unknown", semantic_equivalence="unknown",
        numerical_status="unknown", timing_status="unknown", heldout_status="unknown",
        next_action="Use ordinary A/B selfchecks and matched complete-region timing; compare synchronization, "
                    "configuration, operand traffic and alias-guard overhead together.")


def experiment(report, *, packet, compiler_sha256):
    """Use existing finite experiment routing; stale or unbound contexts stay blocked."""
    try:
        from merlin.mining.reconstruction_experiments import build_experiment_packet
    except ImportError:
        return dict(status="unknown", reason="existing_experiment_package_unavailable")
    context = packet.get("context", {})
    if compiler_sha256 != context.get("compiler_sha256"):
        return dict(status="prior_only", reason="current_compiler_context_requires_rebinding",
                    requests=[], gaps=report.get("gaps", []))
    return build_experiment_packet(report, context=context)
