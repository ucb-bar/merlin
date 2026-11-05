"""Share exact immutable tensor views and exact integer-to-float casts.

Only provenance differs in a match. Complete source metadata is retained on
each surviving operation; no target placement, floating reassociation, opaque
call purity or memory alias premise is inferred here.
"""

from __future__ import annotations

import io

FEATURE = "share_exact_tensor_preparations"


def _supported(op):
    from xdsl.dialects.builtin import Float32Type, IntegerType, NoneAttr, TensorType

    if any("strictfp" in k or "strictfp" in str(v) for p in op.walk()
           for k, v in (*p.attributes.items(), *p.properties.items())):
        return False
    for value in (*op.operands, *op.results):
        if isinstance(value.type, TensorType) and (
            not isinstance(value.type.encoding, NoneAttr) or any(d < 0 for d in value.type.get_shape())
        ):
            return False
    if op.name == "arith.constant":
        return not op.operands and not op.regions and len(op.results) == 1
    if op.name in {"tensor.extract_slice", "tensor.collapse_shape", "tensor.expand_shape", "tensor.cast",
                   "tensor.splat"}:
        return not op.regions and len(op.results) == 1 and isinstance(op.results[0].type, TensorType)
    if (op.name not in {"linalg.generic", "linalg.transpose"} or len(op.operands) != 2
            or len(op.results) != 1 or len(op.regions) != 1 or len(op.regions[0].blocks) != 1
            or any(not isinstance(v.type, TensorType) for v in (*op.operands, *op.results))
            or op.results[0].type != op.operands[1].type):
        return False
    allowed = {"indexing_maps", "iterator_types", "operandSegmentSizes", "permutation"}
    if any(k not in allowed and not k.startswith("prov.") for k in (*op.attributes, *op.properties)):
        return False
    block = op.regions[0].block
    if len(block.args) != 2 or block.args[1].uses:
        return False
    body = list(block.ops)
    copy = len(body) == 1 and body[0].name == "linalg.yield" and tuple(body[0].operands) == (block.args[0],)
    if op.name == "linalg.transpose":
        return copy
    if copy:
        return _parallel_output(op)
    if not (len(body) == 2 and body[0].name == "arith.sitofp" and body[1].name == "linalg.yield"
            and tuple(body[0].operands) == (block.args[0],)
            and tuple(body[1].operands) == (body[0].results[0],) and not body[0].regions
            and all(k.startswith("prov.") for k in (*body[0].attributes, *body[0].properties))):
        return False
    integer = block.args[0].type
    # These casts are exact under every rounding mode and raise no FP flags.
    if not isinstance(integer, IntegerType) or integer.width.data not in (8, 16):
        return False
    if not isinstance(body[0].results[0].type, Float32Type):
        return False
    return _parallel_output(op)


def _parallel_output(op):
    from xdsl.dialects.linalg.attrs import IteratorType
    from xdsl.ir.affine import AffineMap

    maps = op.indexing_maps
    rank = len(op.results[0].type.get_shape())
    return (all(i.data == IteratorType.PARALLEL for i in op.iterator_types)
            and len(op.iterator_types) == rank and len(maps) == 2
            and maps.data[-1].data == AffineMap.identity(rank))


def _stripped(op):
    clone = op.clone()
    for inner in clone.walk():
        for attrs in (inner.attributes, inner.properties):
            for name in tuple(attrs):
                if name.startswith("prov."):
                    del attrs[name]
    return clone


def _retain_provenance(kept, removed):
    from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr

    for left, right in zip(kept.walk(), removed.walk(), strict=True):
        records = []
        for operation in (left, right):
            previous = operation.attributes.get("prov.cse_sources")
            if isinstance(previous, ArrayAttr):
                records.extend(previous.data)
            else:
                metadata = {k: v for k, v in operation.attributes.items() if k.startswith("prov.")}
                if metadata:
                    records.append(DictionaryAttr(metadata))
        if records:
            left.attributes["prov.cse_sources"] = ArrayAttr(tuple(dict.fromkeys(records)))
        for name in ("prov.source_node_ids", "prov.origin_node_ids"):
            arrays = [o.attributes.get(name) for o in (left, right)]
            if all(a is None or isinstance(a, ArrayAttr) for a in arrays):
                values = tuple(dict.fromkeys(v for a in arrays if a is not None for v in a.data))
                if values:
                    left.attributes[name] = ArrayAttr(values)


def _returned_preparation(op):
    # The selected upstream output conversion cannot materialize duplicated
    # aliased tensor returns. Preserve independent return preparation cones.
    pending, seen = list(op.results), set()
    while pending:
        value = pending.pop()
        if value in seen:
            continue
        seen.add(value)
        for use in value.uses:
            if use.operation.name == "func.return":
                return True
            if _supported(use.operation):
                pending.extend(use.operation.results)
    return False


def deduplicate_tensor_preparation(module) -> dict:
    """CSE in each sequential block; candidates never cross a region boundary."""
    from collections import Counter

    from xdsl.dialects import tensor

    counts = Counter()
    for function in tuple(module.walk()):
        if function.name != "func.func" or not function.regions or any(
            "strictfp" in k or "strictfp" in str(v)
            for k, v in (*function.attributes.items(), *function.properties.items())
        ):
            continue
        for block in function.regions[0].blocks:
            candidates = {}
            for op in tuple(block.ops):
                if not _supported(op) or _returned_preparation(op):
                    continue
                operands = tuple(op.operands)
                destination = None
                if op.name.startswith("linalg."):
                    if not isinstance(operands[-1].owner, tensor.EmptyOp):
                        continue
                    operands, destination = operands[:-1], operands[-1]
                key = (op.name, operands, tuple(v.type for v in op.results))
                normalized = _stripped(op)
                match = None
                for previous in candidates.get(key, ()):
                    other = _stripped(previous)
                    context = {} if destination is None else {destination: previous.operands[-1]}
                    equivalent = normalized.is_structurally_equivalent(other, context)
                    other.drop_all_references()
                    if equivalent:
                        match = previous
                        break
                normalized.drop_all_references()
                if match is None:
                    candidates.setdefault(key, []).append(op)
                    continue
                _retain_provenance(match, op)
                for before, after in zip(op.results, match.results, strict=True):
                    before.replace_all_uses_with(after)
                block.erase_op(op)
                counts[op.name] += 1
    module.verify()
    return dict(sorted(counts.items()))


def rewrite_text(source: str) -> tuple[str, dict]:
    from xdsl.printer import Printer

    from merlin.frontends.linalg_mlir import parse_mlir_text

    module = parse_mlir_text(source)
    report = deduplicate_tensor_preparation(module)
    if not report:
        return source, report
    output = io.StringIO()
    Printer(stream=output, print_generic_format=True).print_op(module)
    return output.getvalue(), report


def ensure_registered():
    from .impr_features import ImprFeature, known, register

    if FEATURE not in known():
        register(ImprFeature(
            name=FEATURE,
            action_class="PASS",
            description=(
                "Share identical static immutable tensor views and exact i8/i16-to-f32 preparations, "
                "retaining all merged provenance; explicit and default off."
            ),
        ))
    return FEATURE
