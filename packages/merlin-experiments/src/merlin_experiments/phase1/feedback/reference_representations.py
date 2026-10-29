"""Independent upstream, schedule and emitted views with explicit join limits."""

from __future__ import annotations

from collections import Counter


def emitted_view(raw, pin, *, decoder, symbol=""):
    from xdsl.dialects import llvm
    from xdsl.parser import Parser
    from xdsl.utils.exceptions import ParseError, VerifyException

    from merlin.common.ir_lock import IR_LOCK
    from merlin.perf.emitted_work import assembly_site_kind, summarize_emitted_work
    from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context

    if len(raw) > 32 << 20:
        raise ValueError("analysis_emitted_invalid")
    try:
        with IR_LOCK:
            module = Parser(make_llvm_context(), raw.decode()).parse_module()
            module.verify()
    except (ParseError, VerifyException) as error:
        raise ValueError("analysis_emitted_invalid") from error
    functions = [op for op in module.walk() if isinstance(op, llvm.FuncOp) and op.body.blocks]
    if not functions or sum(1 for _ in module.walk()) > 100_000:
        raise ValueError("analysis_emitted_invalid")
    subject = module
    if symbol:
        functions = [op for op in functions if op.sym_name.data == symbol]
        if len(functions) != 1:
            raise ValueError("analysis_symbol_not_unique")
        # The existing counted observer consumes a module, including when the
        # selected function itself has several CFG blocks. Keep original
        # globals/declarations but remove other function bodies from its scope.
        subject = module.clone()
        for operation in tuple(subject.body.block.ops):
            if isinstance(operation, llvm.FuncOp) and operation.sym_name.data != symbol:
                subject.body.block.erase_op(operation)
    sites = Counter(assembly_site_kind(op) for op in subject.walk() if op.name == "llvm.inline_asm")
    from merlin.kernels.consumer_structure import consumer_structure

    report = dict(
        kind="emitted_ir_analysis",
        input=pin,
        functions=[op.sym_name.data for op in functions],
        scope="selected_function" if symbol else "complete_emitted_artifact",
        instruction_sites=dict(sites),
        consumer_structure=consumer_structure(module, symbol=symbol),
        source_correspondence="unknown",
        numerical_equivalence="unknown",
        performance="unknown",
    )
    if decoder.get("semantics") is not None:
        from merlin.kernels.instruction_comparison import OperandSummary
        from merlin.kernels.machine_evidence import _finite
        from merlin.targetgen.rocc.decode import decode_module

        trace = _finite(decode_module(subject, target=decoder["target"], semantics=decoder["semantics"]))
        operands = OperandSummary()
        for row in trace["instructions"]:
            operands.observe(
                row["class"], _finite(row.get("decoded", {})), _finite({name: row.get(name) for name in ("rs1", "rs2")})
            )
        report["operand_summary"] = operands.report()
        report["selected_instruction_decode"] = dict(
            classes=trace["summary"]["class_histogram"],
            witnesses=trace["instructions"][:4],
            omitted=max(0, len(trace["instructions"]) - 4),
        )
    if len(functions) == 1:
        try:
            report["counted_work"] = dict(
                status="observed_counted_cfg", **summarize_emitted_work(functions[0], command_kind=assembly_site_kind)
            )
        except ValueError as error:
            report["counted_work"] = dict(status="unknown", reason=str(error), command_counts=None)
    return report, None


def compare_bundles(reference, candidate, expert, ours, *, compare_machine):
    from merlin.kernels.kernel_evidence import SemanticRegion, compare_regions

    from .reference_mechanisms import align_questions

    left, right = reference["views"], candidate["views"]
    a, b = left.get("source", {}).get("semantic_region"), right.get("source", {}).get("semantic_region")
    alignment = (
        compare_regions(SemanticRegion.from_record(a), SemanticRegion.from_record(b)).to_dict()
        if a and b
        else dict(status="unknown", reason="complete_upstream_semantic_regions_required")
    )
    machine = None
    if left.get("binary", {}).get("kind") == right.get("binary", {}).get("kind") == "static_instruction_analysis":
        machine = compare_machine(left["binary"], right["binary"], expert, ours)
    emitted = None
    if left.get("emitted", {}).get("kind") == right.get("emitted", {}).get("kind") == "emitted_ir_analysis":
        emitted = compare_machine(left["emitted"], right["emitted"], None, None)
    # Bundle pins identify artifacts. They do not attest that either binary
    # implements the supplied source or schedule.
    return dict(
        kind="representation_comparison",
        reference=reference,
        candidate=candidate,
        upstream_semantics=alignment,
        machine_comparison=machine,
        emitted_comparison=emitted,
        mechanism_alignment=align_questions(reference, candidate),
        semantic_compatibility="unknown",
        producer_correspondence="unknown",
        performance="unknown",
        next_action="Bind source-to-emission producer records or original capture correspondence; "
        "then test supported differences through the ordinary experiment owner.",
    )
