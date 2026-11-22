"""Exact, replayable source BF16 bias separation for prequantized contractions."""
from dataclasses import replace
from math import prod
import json

from merlin.compiler.semantic.model import IndexMap, SemanticNode, pointwise_identity_maps
from merlin.compiler.quantization.contraction.scaled_fp8 import ContractionNumerics

NUMERICS_ATTRIBUTE = "numerical.contraction"


def _readout_attrs(attrs):
    if NUMERICS_ATTRIBUTE not in dict(attrs):
        return attrs
    value = dict(attrs)[NUMERICS_ATTRIBUTE]
    try:
        numerics = ContractionNumerics.from_record(json.loads(value))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid prequantized bias numerical declaration") from exc
    stage = "source-bias-after-output-cast"
    if (numerics.bias_rounding != "bf16-rne" or numerics.activation != "none"
            or len(numerics.stages) < 2 or numerics.stages[-1] != stage
            or numerics.stages.count(stage) != 1):
        raise ValueError("prequantized bias requires a final BF16 RNE source bias stage")
    readout = replace(numerics, stages=numerics.stages[:-1], bias_rounding="none")
    value = json.dumps(readout.record(), sort_keys=True, separators=(",", ":"))
    return tuple((key, value if key == NUMERICS_ATTRIBUTE else item) for key, item in attrs)


_BIAS_SPLIT_SCHEMA = "merlin.prequantized_bias.v1"


def is_prequantized_bias_add(source, node):
    """Recognize split structure; the replay receipt authenticates its origin."""
    if (node.effect != "pure" or node.op != "bf16_add"
            or node.inputs != (node.id + ".readout", node.id + ".bias")
            or node.type.dtype != "bf16" or not node.type.shape
            or node.type.numerical_policy != "raw-bit-pattern"
            or dict(node.attrs) != {"numeric_semantics": "ieee754-bf16-add-rne-v1"}):
        return False
    try:
        readout, bias = (source.node(key) for key in node.inputs)
        original_bias = source.node(bias.inputs[0]) if len(bias.inputs) == 1 else None
        rank = len(node.type.shape)
        column = IndexMap(rank, ((0,) * (rank - 1) + (1,),), (0,))
        if not (readout.effect == "pure"
                and dict(readout.attrs).get("semantic_kind") == "prequantized_contraction"
                and dict(readout.attrs).get("bias_order") == "after_output_cast"
                and len(readout.inputs) == 4 and not readout.index_maps
                and readout.type == bias.type == node.type
                and node.index_maps == pointwise_identity_maps(rank, 2)):
            return False
        if bias.effect == "constant":
            # Exact source preparation can fold the column broadcast. Its
            # dense bytes must still be declared by one complete BF16 binding;
            # the source-preparation receipt authenticates the fold's origin.
            bindings = [binding for binding in source.constants if binding.node_id == bias.id]
            if (bias.op != "constant" or bias.inputs or bias.attrs or bias.index_maps
                    or len(bindings) != 1 or bindings[0].encoding != "bf16-le-raw"
                    or len(bytes.fromhex(bindings[0].data_hex)) != prod(node.type.shape) * 2):
                return False
        elif not (bias.effect == "pure" and bias.op == "gather" and not bias.attrs
                  and original_bias is not None
                  and original_bias.type.shape == (node.type.shape[-1],)
                  and original_bias.type.dtype == "bf16"
                  and original_bias.type.numerical_policy == "raw-bit-pattern"
                  and bias.index_maps == (column, pointwise_identity_maps(rank, 1)[-1])):
            return False
        if NUMERICS_ATTRIBUTE in dict(readout.attrs):
            numerics = ContractionNumerics.from_record(
                json.loads(dict(readout.attrs)[NUMERICS_ATTRIBUTE]))
            if (numerics.bias_rounding != "none" or numerics.activation != "none"
                    or "source-bias-after-output-cast" in numerics.stages):
                return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def split_prequantized_bias(source):
    """Preserve output identities and round bias only after the BF16 readout.

    The broadcast is an exact logical gather. The separate addition uses the
    IEEE BF16 RNE contract for host lowering; device fusion is not authorized.
    """
    nodes, changes = [], []
    used = {node.id for node in source.nodes}
    for node in source.nodes:
        attrs = dict(node.attrs)
        if (node.effect != "pure" or attrs.get("semantic_kind") != "prequantized_contraction"
                or len(node.inputs) != 5 or node.type.dtype != "bf16"):
            nodes.append(node)
            continue
        bias = source.node(node.inputs[4])
        if (attrs.get("bias_order") != "after_output_cast" or node.index_maps
                or not node.type.shape or bias.type.shape != (node.type.shape[-1],)
                or bias.type.dtype != "bf16" or bias.type.numerical_policy != "raw-bit-pattern"
                or node.type.numerical_policy != "raw-bit-pattern"):
            raise ValueError("prequantized BF16 bias requires a column vector after the output cast")
        readout_id, broadcast_id = node.id + ".readout", node.id + ".bias"
        if readout_id in used or broadcast_id in used:
            raise ValueError("prequantized bias expansion identity collision")
        used.update((readout_id, broadcast_id))
        rank = len(node.type.shape)
        identity = pointwise_identity_maps(rank, 1)[-1]
        column = IndexMap(rank, ((0,) * (rank - 1) + (1,),), (0,))
        readout = replace(node, id=readout_id, inputs=node.inputs[:4], attrs=_readout_attrs(node.attrs))
        broadcast = SemanticNode(broadcast_id, "gather", (bias.id,), node.type,
                                 index_maps=(column, identity))
        addition = SemanticNode(node.id, "bf16_add", (readout_id, broadcast_id), node.type,
                                (("numeric_semantics", "ieee754-bf16-add-rne-v1"),),
                                index_maps=pointwise_identity_maps(rank, 2))
        changes.append({"original": node.record()})
        nodes.extend((readout, broadcast, addition))
    if changes and (source.physical_extents or source.boundary_layouts or source.output_layouts):
        raise ValueError("prequantized bias expansion requires a logical source")
    result = replace(source, nodes=tuple(nodes)) if changes else source
    return result, dict(schema=_BIAS_SPLIT_SCHEMA, before_digest=source.digest(), after_digest=result.digest(), changes=changes)


def verify_prequantized_bias(source, record):
    if (not isinstance(record, dict)
            or set(record) != {"schema", "before_digest", "after_digest", "changes"}
            or record["schema"] != _BIAS_SPLIT_SCHEMA or record["after_digest"] != source.digest()
            or not isinstance(record["changes"], list)):
        raise ValueError("prequantized bias receipt identity or fields changed")
    originals = {}
    for row in record["changes"]:
        if not isinstance(row, dict) or set(row) != {"original"}:
            raise ValueError("invalid prequantized bias change")
        original = SemanticNode.from_record(row["original"])
        if original.id in originals:
            raise ValueError("duplicate prequantized bias change")
        originals[original.id] = original
    erased = {key + suffix for key in originals for suffix in (".readout", ".bias")}
    restored = replace(source, nodes=tuple(originals.get(node.id, node) for node in source.nodes if node.id not in erased))
    if restored.digest() != record["before_digest"]:
        raise ValueError("prequantized bias original changed")
    replayed, expected = split_prequantized_bias(restored)
    if replayed.digest() != source.digest() or expected != record:
        raise ValueError("prequantized bias differs from exact replay")
    return restored
