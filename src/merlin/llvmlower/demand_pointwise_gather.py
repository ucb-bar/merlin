"""Explicit demand-driven pointwise tensor evaluation at unchanged gathers.

No quantizer, table specialization, indexing policy or backend is selected.
The original in-bounds tensor.extract precondition remains authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DemandGatherEffects:
    """Caller obligations for omitting ungathered pure floating evaluations."""

    immutable_inputs: bool
    nontrapping_floating_point: bool
    floating_flags_unobserved: bool

    def validate(self):
        if any(
            value is not True
            for value in (self.immutable_inputs, self.nontrapping_floating_point, self.floating_flags_unobserved)
        ):
            raise ValueError("explicit immutable/nontrapping/unobserved-effects contract required")


def _attributes(op):
    from xdsl.dialects.builtin import ArrayAttr, StringAttr

    arrays = {"prov.source_node_ids", "prov.origin_node_ids"}
    strings = {
        "prov.region_id",
        "prov._pattern_hint",
        "prov.op",
        "prov.family",
        "prov.trace_role",
        "prov.aten",
        "prov.orig_dtype",
        "prov.module",
        "prov.fqn",
    }
    for key, value in op.attributes.items():
        if key in arrays:
            if not isinstance(value, ArrayAttr) or any(not isinstance(v, StringAttr) for v in value):
                return False
        elif key not in strings or not isinstance(value, StringAttr):
            return False
    return True


def _shape(value):
    from xdsl.dialects.builtin import NoneAttr, TensorType

    dtype = value.type
    if not isinstance(dtype, TensorType) or not isinstance(dtype.encoding, NoneAttr):
        raise ValueError("dense static tensor storage required")
    shape = dtype.get_shape()
    if any(type(d) is not int or d <= 0 for d in shape):
        raise ValueError("positive static tensor dimensions required")
    return shape


def _maps(op):
    from xdsl.dialects.linalg.attrs import IteratorType
    from xdsl.dialects.linalg.ops import GenericOp
    from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr, AffineMap

    if (
        not isinstance(op, GenericOp)
        or len(op.outputs) != 1
        or len(op.results) != 1
        or len(op.body.blocks) != 1
        or not _attributes(op)
        or set(op.properties) != {"indexing_maps", "iterator_types", "operandSegmentSizes"}
    ):
        raise ValueError("one pure closed generic tensor result required")
    shape = _shape(op.results[0])
    if op.outputs[0].type != op.results[0].type or len(op.iterator_types) != len(shape):
        raise ValueError("complete output coverage required")
    if any(v.data is not IteratorType.PARALLEL for v in op.iterator_types):
        raise ValueError("pointwise parallel iterators required")
    maps = [v.data for v in op.indexing_maps]
    if len(maps) != len(op.inputs) + 1 or maps[-1] != AffineMap.identity(len(shape)):
        raise ValueError("identity output map required")
    if len(op.body.block.args) != len(op.inputs) + 1 or op.body.block.args[-1].uses:
        raise ValueError("output-init reads are unsupported")
    for value, mapping in zip(op.inputs, maps[:-1], strict=True):
        source = _shape(value)
        if mapping.num_dims != len(shape) or mapping.num_symbols or len(mapping.results) != len(source):
            raise ValueError("closed static input maps required")
        dimensions = set()
        for extent, expr in zip(source, mapping.results, strict=True):
            if isinstance(expr, AffineDimExpr):
                if expr.position in dimensions or not 0 <= expr.position < len(shape) or extent != shape[expr.position]:
                    raise ValueError("ambiguous or mismatched input projection")
                dimensions.add(expr.position)
            elif not isinstance(expr, AffineConstantExpr) or expr.value != 0 or extent != 1:
                raise ValueError("only proved singleton broadcast is supported")
    return maps[:-1]


def _scalar_body(op):
    from xdsl.dialects.linalg.ops import YieldOp

    allowed = {
        "arith.sitofp",
        "arith.uitofp",
        "arith.extf",
        "arith.truncf",
        "arith.extsi",
        "arith.extui",
        "arith.trunci",
        "arith.subf",
        "arith.mulf",
        "arith.addf",
        "arith.constant",
    }
    available = set(op.body.block.args[:-1])
    body = tuple(op.body.block.ops)
    if not body or not isinstance(body[-1], YieldOp) or len(body[-1].arguments) != 1:
        raise ValueError("one scalar pointwise yield required")
    for scalar in body[:-1]:
        if (
            scalar.name not in allowed
            or scalar.regions
            or not _attributes(scalar)
            or any(value not in available for value in scalar.operands)
        ):
            raise ValueError("unknown scalar arithmetic, capture or effects")
        permitted = {"value"} if scalar.name == "arith.constant" else {"fastmath"}
        if set(scalar.properties) - permitted:
            raise ValueError("unknown scalar numerical contract")
        flags = scalar.properties.get("fastmath")
        if flags is not None and flags.data:
            raise ValueError("flagged floating arithmetic is unsupported")
        available.update(scalar.results)
    if body[-1].arguments[0] not in available or body[-1].properties or not _attributes(body[-1]):
        raise ValueError("unknown yield contract")


def rewrite_module(module, effects: DemandGatherEffects | None = None, *, max_scalar_ops=256):
    """Return an owned clone and explicit accepted/refused gather observations."""
    if effects is None:
        return module, {"selected": False, "rewritten": [], "refused": []}
    if type(effects) is not DemandGatherEffects or type(max_scalar_ops) is not int or not 0 < max_scalar_ops <= 4096:
        raise ValueError("explicit effects and finite scalar expansion budget required")
    effects.validate()
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, IndexType, StringAttr
    from xdsl.dialects.func import FuncOp
    from xdsl.dialects.linalg.ops import GenericOp
    from xdsl.ir import BlockArgument, OpResult
    from xdsl.ir.affine import AffineDimExpr
    from xdsl.rewriter import Rewriter

    cloned = module.clone()
    cloned.verify()
    report = {
        "selected": True,
        "rewritten": [],
        "refused": [],
        "index_bounds": "unchanged original tensor.extract in-bounds precondition; no index remapping",
        "specialized_input_values": False,
    }
    candidates = [
        op for op in cloned.walk() if isinstance(op, tensor.ExtractOp) and isinstance(op.tensor.owner, GenericOp)
    ]
    for extract in candidates:
        try:
            gather = extract.parent_op()
            if not isinstance(gather, GenericOp) or not _attributes(extract) or extract.properties:
                raise ValueError("gather must retain a closed scalar tensor.extract")
            _maps(gather)
            if gather.body.block.args[-1].uses:
                raise ValueError("gather output-init reads are unsupported")
            for op in gather.body.block.ops:
                if (
                    op.name
                    not in {"arith.index_cast", "arith.constant", "linalg.index", "tensor.extract", "linalg.yield"}
                    or op.regions
                    or not _attributes(op)
                ):
                    raise ValueError("unknown gather coordinate/effect body")
                expected = {"dim"} if op.name == "linalg.index" else {"value"} if op.name == "arith.constant" else set()
                if set(op.properties) != expected:
                    raise ValueError("unknown gather operation properties")
            producers = []
            mapping = {}
            active = set()

            def visit(value):
                if isinstance(value, BlockArgument):
                    if not isinstance(value.owner.parent_op(), FuncOp) or value.owner is not gather.parent:
                        raise ValueError("source must be an immutable enclosing tensor argument")
                    _shape(value)
                    return
                if not isinstance(value, OpResult) or not isinstance(value.owner, GenericOp):
                    raise ValueError("producer DAG has an unsupported tensor source")
                op = value.owner
                if op in active or value is not op.results[0] or op.parent is not gather.parent:
                    raise ValueError("producer must dominate in the same closed block")
                if op in mapping:
                    return
                active.add(op)
                maps = _maps(op)
                _scalar_body(op)
                seed = op.outputs[0].owner
                if isinstance(seed, tensor.EmptyOp) and (seed.operands or seed.properties or not _attributes(seed)):
                    raise ValueError("unknown output-init tensor contract")
                for source in op.inputs:
                    visit(source)
                active.remove(op)
                mapping[op] = maps
                producers.append(op)

            visit(extract.tensor)
            costs = {}
            for op in producers:
                cost = len(tuple(op.body.block.ops)) - 1
                for value, amap in zip(op.inputs, mapping[op], strict=True):
                    cost += 1 if isinstance(value, BlockArgument) else costs[value.owner]
                    cost += sum(not isinstance(expr, AffineDimExpr) for expr in amap.results)
                if cost > max_scalar_ops:
                    raise ValueError("pointwise scalar expansion exceeds declared budget")
                costs[op] = cost
            if not producers:
                raise ValueError("no closed pointwise producer")
            for producer in producers:
                if any(
                    use.operation not in mapping and use.operation is not extract for use in producer.results[0].uses
                ):
                    raise ValueError("producer has other unclosed users; original uses retained")
            positions = {op: i for i, op in enumerate(gather.parent.ops)}
            if any(positions[op] >= positions[gather] for op in producers):
                raise ValueError("producer must precede gather")
            between = min(producers, key=lambda op: positions[op])
            while between is not gather:
                if between is None:
                    raise ValueError("producer does not dominate gather")
                if between not in mapping:
                    if (
                        not isinstance(between, tensor.EmptyOp | arith.ConstantOp)
                        or not _attributes(between)
                        or (isinstance(between, tensor.EmptyOp) and (between.operands or between.properties))
                    ):
                        raise ValueError("interposed guard, unknown contract or effect boundary")
                between = between.next_op
            inserted = []

            def evaluate(value, coordinates):
                owner = value.owner
                if isinstance(value, BlockArgument):
                    leaf = tensor.ExtractOp(value, coordinates, value.type.element_type)
                    leaf.attributes.update(extract.attributes)
                    inserted.append(leaf)
                    return leaf.result
                values = {}
                for scalar_arg, source, amap in zip(
                    owner.body.block.args[:-1], owner.inputs, mapping[owner], strict=True
                ):
                    projected = []
                    for expression in amap.results:
                        if isinstance(expression, AffineDimExpr):
                            projected.append(coordinates[expression.position])
                        else:
                            zero = arith.ConstantOp.from_int_and_width(0, IndexType())
                            inserted.append(zero)
                            projected.append(zero.result)
                    values[scalar_arg] = evaluate(source, projected)
                for scalar in tuple(owner.body.block.ops)[:-1]:
                    copied = scalar.clone(value_mapper=values)
                    inserted.append(copied)
                    values.update(zip(scalar.results, copied.results, strict=True))
                return values[owner.body.block.last_op.arguments[0]]

            result = evaluate(extract.tensor, list(extract.indices))
            if len(inserted) != costs[extract.tensor.owner]:
                raise RuntimeError("scalar expansion changed after bounded preflight")
            # Preserve removed region/operator metadata as typed provenance;
            # none of these attributes supplied admission or numerical permission.
            provenance = ArrayAttr(
                [
                    DictionaryAttr(
                        {
                            "attributes": DictionaryAttr(dict(op.attributes)),
                            "properties": DictionaryAttr(dict(op.properties)),
                            "yield_attributes": DictionaryAttr(dict(op.body.block.last_op.attributes)),
                            "location": op.location,
                            "seed_attributes": DictionaryAttr(dict(op.outputs[0].owner.attributes))
                            if isinstance(op.outputs[0].owner, tensor.EmptyOp)
                            else DictionaryAttr({}),
                        }
                    )
                    for op in producers
                ]
            )
            Rewriter.replace_op(extract, inserted, new_results=[result])
            gather.attributes["prov.demand_pointwise_sources"] = provenance
            gather.attributes["prov.demand_pointwise_indices"] = StringAttr("original gather coordinates unchanged")
            removed = []
            for producer in reversed(producers):
                if producer.results[0].uses:
                    raise RuntimeError("internal use closure changed during rewrite")
                seed = producer.outputs[0].owner
                removed.append(str(producer.results[0].type))
                Rewriter.erase_op(producer)
                if isinstance(seed, tensor.EmptyOp) and not seed.tensor.uses:
                    Rewriter.erase_op(seed)
            report["rewritten"].append(
                {
                    "producer_tensors_removed": removed,
                    "scalar_ops_inserted": len(inserted),
                    "arithmetic_recreated": False,
                    "original_indices_retained": True,
                }
            )
        except ValueError as ex:
            report["refused"].append({"reason": str(ex)})
    cloned.verify()
    effects.validate()
    return cloned, report


def transform_file(source: Path, output: Path, effects: DemandGatherEffects | None = None, *, max_scalar_ops=256):
    """Explicit prepared-model-transform seam; default None performs no I/O."""
    if effects is None:
        return source
    from merlin.common.digest import sha256_file
    from merlin.common.jsonio import write_pretty_json
    from merlin.common.proc import run_checked
    from merlin.frontends.linalg_mlir import parse_mlir_text

    from . import stock_arith_compat, toolchain
    from .compilation_recipe import CompilationRecipe
    from .deployable_ir import checked_generic_text

    if type(effects) is not DemandGatherEffects:
        raise ValueError("explicit DemandGatherEffects required")
    effects.validate()
    source, output = Path(source), Path(output)
    original = source.read_bytes()
    output.mkdir(parents=True, exist_ok=False)
    snapshot, normalized = output / "input.mlir", output / "normalized.mlir"
    snapshot.write_bytes(original)
    tool = toolchain.llvm_install() / "bin/mlir-opt"
    recipe = CompilationRecipe(output, producer=Path(__file__))
    compatibility_source = Path(stock_arith_compat.__file__)
    compatibility_sha256 = sha256_file(compatibility_source)
    recipe.bind_preparation("stock_integer_compatibility", compatibility_source)
    recipe.run(
        [tool, snapshot, "--mlir-print-op-generic", "-o", normalized],
        runner=lambda args: run_checked(args, timeout=180),
        inputs=[snapshot],
        output=normalized,
    )
    parsed = parse_mlir_text(normalized.read_text())
    compatibility = stock_arith_compat.unflagged_truncations(parsed)
    module, report = rewrite_module(parsed, effects, max_scalar_ops=max_scalar_ops)
    result = output / "model.mlir"
    if (source.read_bytes() != original or snapshot.read_bytes() != original
            or sha256_file(compatibility_source) != compatibility_sha256):
        raise ValueError("source changed during pointwise gather fusion")
    effects.validate()
    result.write_text(checked_generic_text(module))
    report.update(
        schema="merlin.demand_pointwise_gather.v1",
        source_sha256=sha256_file(source),
        output_sha256=sha256_file(result),
        normalizer_sha256=sha256_file(tool),
        stock_integer_compatibility=compatibility,
        stock_integer_compatibility_input_sha256={str(compatibility_source): compatibility_sha256},
    )
    write_pretty_json(output / "report.json", report)
    return result
