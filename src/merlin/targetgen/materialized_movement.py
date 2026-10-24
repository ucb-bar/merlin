"""Borrowed result descriptors backed by source-bound byte-selector copies.

The frontend storage contract describes the same converted program as the
selector IR. It supplies alias identity, never device legality or payloads.
Only complete F1 lineage plus a static, read-only storage envelope licenses
copying the materialized value back into that borrowed descriptor.
"""

import hashlib
import json


def expose_alias_results(module, report):
    """Mark observable static borrowed outputs for the result-layout copy-out."""
    from xdsl.dialects.builtin import StringAttr

    from merlin.llvmlower.input_layout import static_storage_span
    from merlin.llvmlower.result_layout import ATTRIBUTE

    attr = module.attributes.get(ATTRIBUTE)
    if attr is None:
        return []
    layout = json.loads(attr.data)
    functions = [op for op in module.walk() if op.name == "func.func" and op.sym_name.data == "forward"]
    if len(functions) != 1:
        return []
    function = functions[0]
    returns = [op for op in function.body.block.ops if op.name == "func.return"]
    if len(returns) != 1 or len(layout["results"]) != len(returns[0].operands):
        return []
    if len(layout["inputs"]) != len(function.body.block.args):
        return []
    records = [r for r in report["cases"] if r.get("status") == "bundled"]
    if any((r.get("case") or {}).get("mutated_arguments") for r in records):
        return []
    if any(r and r.get("role") == "user_input_mutation" for r in layout["results"]):
        return []
    indices = []
    for index in sorted({i for r in records for i in r.get("output_indices", [])}):
        record = layout["results"][index]
        if not record or not record.get("alias_inputs"):
            continue
        aliases = record["alias_inputs"]
        if len(aliases) != 1 or type(aliases[0]) is not int or not 0 <= aliases[0] < len(layout["inputs"]):
            return []
        base = aliases[0]
        backing = layout["inputs"][base]
        result_type = returns[0].operands[index].type
        input_type = function.body.block.args[base].type
        need, capacity = static_storage_span(record), static_storage_span(backing)
        if (
            need is None
            or capacity is None
            or need > capacity
            or list(result_type.get_shape()) != record["shape"]
            or list(input_type.get_shape()) != backing["shape"]
            or result_type.element_type != input_type.element_type
        ):
            return []
        indices.append(index)
    for index in indices:
        layout["results"][index]["materialized_byte_selection"] = True
        value = returns[0].operands[index]
        if value in function.body.block.args:
            # An observable identity may have no tensor operation left after
            # frontend folding. Expose real movement at the return boundary;
            # internal aliases and the post-input observations remain borrowed.
            from merlin.llvmlower._ir import copy

            operations = []
            copied = copy(value, operations)
            function.body.block.insert_ops_before(operations, returns[0])
            returns[0].operands = [copied if i == index else v for i, v in enumerate(returns[0].operands)]
    if indices:
        module.attributes[ATTRIBUTE] = StringAttr(json.dumps(layout, sort_keys=True))
    return indices


def alias_materialization_valid(directory, report, plan):
    """Reopen the actual prepared source and its complete selector binding."""
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.targetgen.core_aten_exact_dtype import bitwise_catalog_coverage

    proof = plan.get("materialized_output_aliases")
    if not proof or plan.get("numerical_family") != "F1" or not plan.get("exact_dtype_sites"):
        return False
    source = (directory / "model.mlir").read_text().rstrip("\n")
    if proof.get("source_sha256") != hashlib.sha256(source.encode()).hexdigest():
        return False
    try:
        module = parse_mlir_text(source)
        indices = expose_alias_results(module, report)
        if not indices or indices != proof.get("output_indices") or str(module) != source:
            return False
        bitwise_catalog_coverage(module, plan["provider_catalog"])
    except (KeyError, ValueError, TypeError, IndexError):
        return False
    return True
