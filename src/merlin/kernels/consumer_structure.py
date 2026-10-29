"""Actual LLVM call reachability, independent of source grouping or policy intent."""

from __future__ import annotations

from collections import defaultdict


def consumer_structure(module, *, symbol="", max_operations=100_000, max_witnesses=16):
    """Expose reachable selected consumers and shared call arguments without inlining.

    Static sites have no dynamic multiplicity. Definitions outside the selected
    root's call closure cannot witness a realized optimization. An indirect call
    or external body remains unknown, rather than inheriting an advertised policy.
    """
    from xdsl.dialects import llvm

    if max_operations < 1 or max_witnesses < 1:
        raise ValueError("consumer analysis requires finite positive budgets")
    operations = []
    for index, operation in enumerate(module.walk()):
        if index >= max_operations:
            return dict(status="unknown", reason="consumer_operation_budget_exhausted", functions=[])
        operations.append(operation)
    functions = {}
    for operation in operations:
        if isinstance(operation, llvm.FuncOp):
            name = operation.sym_name.data
            if name in functions:
                return dict(status="unknown", reason="consumer_function_symbol_not_unique", functions=[])
            functions[name] = operation
    roots = [symbol] if symbol else [name for name, op in functions.items() if op.body.blocks]
    if symbol and (symbol not in functions or not functions[symbol].body.blocks):
        raise ValueError("consumer selected function is not defined")
    summaries = {}
    edges = {}
    for name, function in functions.items():
        calls, shared, names, assembly = [], defaultdict(list), {}, 0
        for index, operation in enumerate(function.walk()):
            assembly += operation.name == "llvm.inline_asm"
            if not isinstance(operation, llvm.CallOp):
                continue
            callee = operation.callee
            direct = callee is not None and not tuple(callee.nested_references)
            target = callee.root_reference.data if direct else None
            calls.append(dict(site=index, callee=target, status="direct" if direct else "unknown_indirect_call"))
            for argument, value in enumerate(operation.args):
                if not isinstance(value.type, llvm.LLVMPointerType):
                    continue
                names.setdefault(value, f"pointer:{len(names)}")
                shared[value].append(dict(site=index, argument=argument, callee=target))
        edges[name] = {call["callee"] for call in calls if call["callee"] in functions}
        groups = [dict(value=names[value], uses=uses[:max_witnesses], uses_total=len(uses),
                       reuse="unknown", alias_and_effects="unknown")
                  for value, uses in shared.items() if len({row["site"] for row in uses}) > 1]
        summaries[name] = dict(symbol=name, body="defined" if function.body.blocks else "external_unknown",
                               call_sites=calls[:max_witnesses], call_sites_total=len(calls),
                               call_sites_omitted=max(0, len(calls) - max_witnesses),
                               assembly_sites=assembly, shared_pointer_arguments=groups[:max_witnesses],
                               shared_pointer_arguments_total=len(groups))
    reachable, pending = set(), list(roots)
    while pending:
        name = pending.pop()
        if name in reachable:
            continue
        reachable.add(name)
        pending.extend(edges[name] - reachable)
    return dict(status="observed_static_call_structure", roots=roots[:max_witnesses], roots_total=len(roots),
                functions=[summaries[name] for name in sorted(reachable)[:max_witnesses]],
                reachable_symbols=sorted(reachable)[:max_witnesses], reachable_total=len(reachable),
                functions_omitted=max(0, len(reachable) - max_witnesses),
                unselected_definitions=sum(bool(op.body.blocks) and name not in reachable
                                           for name, op in functions.items()),
                scope="selected actual emitted call closure; static sites only",
                source_correspondence="unknown", optimization_application="unknown",
                numerical_equivalence="unknown", execution_multiplicity="unknown", performance="unknown")
