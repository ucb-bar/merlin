"""Cheap work accounting over actual emitted LLVM control flow.

Counted regions use the existing independent CFG/domain analysis. No dynamic
command trace, tensor values, target latency or instruction semantics are
invented. A classifier labels static command sites; it does not evaluate their
operands or authorize a schedule. Traffic and elapsed cycles need other tools.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256

from .constant_control_cfg import constant_control_counts


def assembly_site_kind(command):
    """Label actual assembly text, without interpreting its target semantics."""
    if command.name != "llvm.inline_asm":
        return None
    text = command.asm_string.data
    if len(text) <= 120 and "/" not in text and "\\" not in text:
        return "asm:" + text
    return "asm_sha256:" + sha256(text.encode()).hexdigest()


def summarize_emitted_work(function, *, command_kind=None, max_operations=200_000, max_blocks=20_000):
    """Return exact dynamic IR/site counts, without unrolling rectangular loops.

    ``command_kind(op)`` may label an original inline-assembly or call site with
    a nonempty string, or return None when the site is unclassified. Counts are
    scoped to this function; callee bodies, encoded transfer sizes and physical
    activity are not expanded. Unsupported control or exhausted analysis fails
    the whole observation rather than returning an accepted partial total.
    """
    if (function.name != "llvm.func" or not function.body.blocks
            or command_kind is not None and not callable(command_kind)
            or type(max_operations) is not int or not 0 < max_operations <= 1_000_000
            or type(max_blocks) is not int or not 0 < max_blocks <= 100_000):
        raise ValueError("emitted work requires an original function, classifier and bounded analysis")
    function.verify()
    original = sha256(str(function).encode()).hexdigest()
    blocks = tuple(function.body.blocks)
    sites = {}
    for block in blocks:
        for op in block.ops:
            if op.name not in ("llvm.inline_asm", "llvm.call", "llvm.invoke"):
                continue
            kind = command_kind(op) if command_kind is not None else None
            if kind is not None and (type(kind) is not str or not kind or len(kind) > 256):
                raise ValueError("command classification needs a bounded nonempty name or None")
            sites[op] = kind
    # An empty watched roster requests count-only closed-region summaries.
    # Result-free assembly cannot supply a scalar branch value. This grants
    # no effects, memory, instruction or numerical permission to that assembly.
    counts = constant_control_counts(
        function, max_operations=max_operations, max_blocks=max_blocks,
        observe=lambda op, args: None, observed_operations=(),
        summarizable_assembly=tuple(op for op in sites if op.name == "llvm.inline_asm" and not op.results),
    )
    operations, commands = Counter(), Counter()
    unclassified = 0
    for block in blocks:
        for op in block.ops:
            operations[op.name] += counts[block]
            if op in sites:
                if sites[op] is None:
                    unclassified += counts[block]
                else:
                    commands[sites[op]] += counts[block]
    if sha256(str(function).encode()).hexdigest() != original:
        raise ValueError("emitted function changed during work observation")
    return {
        "schema": "merlin.emitted_work.v1",
        "basis": "static_emitted_ir",
        "domain": "original_function_dynamic_ir_sites",
        "function_sha256": original,
        "operation_counts": dict(sorted(operations.items())),
        "command_counts": dict(sorted(commands.items())),
        "unclassified_command_instances": unclassified,
        "command_classification_complete": unclassified == 0,
        "dynamic_ir_operations": sum(operations.values()),
        "static_ir_operations": sum(len(tuple(block.ops)) for block in blocks),
        "blocks": len(blocks),
        "block_visits": [counts[block] for block in blocks] if len(blocks) <= 256 else None,
        "callee_bodies_counted": False,
        "requested_transfer_bytes": None,
        "physical_cycles": None,
        "physical_utilization": None,
        "numerical_execution": False,
        "performance_qualification": False,
    }


def main(argv=None):
    """Observe an own emitted artifact through the ordinary compiler context."""
    import argparse
    import json
    from pathlib import Path

    from xdsl.parser import Parser

    from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("llvm_mlir", type=Path)
    parser.add_argument("--entry", default="kernel_entry")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-operations", type=int, default=200_000)
    parser.add_argument("--max-blocks", type=int, default=20_000)
    args = parser.parse_args(argv)
    payload = args.llvm_mlir.read_bytes()
    try:
        module = Parser(make_llvm_context(), payload.decode()).parse_module()
        entries = [op for op in module.walk() if op.name == "llvm.func" and op.sym_name.data == args.entry]
        if len(entries) != 1:
            raise ValueError("one original entry function required")
        report = {"status": "observed", **summarize_emitted_work(
            entries[0], command_kind=assembly_site_kind,
            max_operations=args.max_operations, max_blocks=args.max_blocks,
        )}
    except ValueError as error:
        report = {"status": "UNKNOWN", "reason": str(error), "performance_qualification": False}
    report.update(artifact_sha256=sha256(payload).hexdigest(), entry_symbol=args.entry)
    text = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(text)
    else:
        print(text, end="")
    return 0 if report["status"] == "observed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
