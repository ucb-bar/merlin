"""Source-faithful, stock-compiler fusion of a closed clamp/encoder pair.

Only complete same-shape identity pointwise operations with one producer use
are eligible. No guard/effect boundary is crossed. The isolated stock fusion is
checked against the original typed scalar DAG; original scalar attributes and
both original operation provenance dictionaries are retained in the result.
"""

from __future__ import annotations

from pathlib import Path

from merlin.common.digest import sha256_file
from merlin.common.jsonio import write_pretty_json
from merlin.common.proc import run_checked


def _expression(value, arguments, allowed, seen, active=frozenset()):
    from xdsl.dialects.arith import ConstantOp
    from xdsl.ir import OpResult

    if value in arguments:
        return ("argument", arguments[value], str(value.type))
    if value in active or not isinstance(value, OpResult):
        raise ValueError("fusion has an unknown/cyclic scalar capture")
    op = value.owner
    if op not in allowed and not isinstance(op, ConstantOp):
        raise ValueError("fusion hoisted unproved scalar arithmetic")
    if op.name not in {"arith.constant", "arith.cmpf", "arith.select", "arith.truncf"} or op.regions:
        raise ValueError("fusion changed the closed scalar operation set")
    seen.add(op)
    return (
        op.name,
        value.index,
        str(value.type),
        tuple((key, str(attr)) for key, attr in sorted(op.properties.items())),
        tuple((key, str(attr)) for key, attr in sorted(op.attributes.items())),
        tuple(_expression(v, arguments, allowed, seen, active | {value}) for v in op.operands),
    )


def fuse_clamped_encoders(source: Path, output: Path, policy):
    """Keep a source-proved clamp local to its independently compiled encoder.

    No selection returns without source/tool discovery. Unknown/multi-use or
    effect-crossing pairs remain unchanged, so later codec admission still
    refuses an unproved outlined argument. This supplies no range by metadata.
    """
    if policy is None:
        return source
    from xdsl.dialects.arith import TruncFOp
    from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, FunctionType, ModuleOp
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.dialects.linalg.ops import GenericOp, YieldOp
    from xdsl.ir import Block, Region
    from xdsl.pattern_rewriter import Rewriter

    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.xdsl_dialects.fp8 import Float8E4M3FNType

    from . import toolchain
    from .compilation_recipe import CompilationRecipe
    from .deployable_ir import checked_generic_text
    from .fp8_scalar_codec import BoundedE4M3FNCodec, _complete_elementwise, _ordered_clamp_body
    from .outlined_guards import discardable_driver_op

    if type(policy) is not BoundedE4M3FNCodec:
        raise ValueError("explicit bounded scalar codec required")
    bindings = policy.validate()
    source, output = Path(source), Path(output)
    original = source.read_bytes()
    output.mkdir(parents=True, exist_ok=False)
    snapshot, normalized = output / "input.mlir", output / "normalized.mlir"
    snapshot.write_bytes(original)
    normalizer = toolchain.llvm_install() / "bin" / "mlir-opt"
    recipe = CompilationRecipe(output, producer=Path(__file__))
    recipe.run(
        [normalizer, snapshot, "--mlir-print-op-generic", "-o", normalized],
        runner=lambda args: run_checked(args, timeout=180),
        inputs=[snapshot],
        output=normalized,
    )
    module = parse_mlir_text(normalized.read_text())
    module.verify()
    records = []
    for consumer in list(module.walk()):
        if not isinstance(consumer, GenericOp) or not _complete_elementwise(consumer):
            continue
        body = tuple(consumer.body.block.ops)
        if (
            len(body) != 2
            or not isinstance(body[0], TruncFOp)
            or not isinstance(body[1], YieldOp)
            or body[0].input is not consumer.body.block.args[0]
            or body[1].arguments[0] is not body[0].result
            or not isinstance(body[0].result.type, Float8E4M3FNType)
            or body[0].properties
            or consumer.body.block.args[1].uses
        ):
            continue
        producer = consumer.inputs[0].owner
        if (
            not _complete_elementwise(producer)
            or not _ordered_clamp_body(producer)
            or producer.body.block.args[1].uses
            or len(tuple(producer.results[0].uses)) != 1
            or producer.parent is not consumer.parent
        ):
            continue
        current = producer.next_op
        while current is not None and current is not consumer and discardable_driver_op(current):
            current = current.next_op
        if current is not consumer:
            continue
        index = len(records)
        work = output / ("pair-" + str(index))
        work.mkdir()
        external = list(dict.fromkeys([*producer.inputs, *producer.outputs, *consumer.outputs]))
        block = Block(arg_types=[value.type for value in external])
        mapping = dict(zip(external, block.args, strict=True))
        p = producer.clone(value_mapper=mapping)
        mapping[producer.results[0]] = p.results[0]
        c = consumer.clone(value_mapper=mapping)
        block.add_ops([p, c, ReturnOp(c.results[0])])
        fragment = ModuleOp(
            [
                FuncOp(
                    "merlin_codec_pair",
                    FunctionType.from_lists([v.type for v in external], [consumer.results[0].type]),
                    Region(block),
                )
            ]
        )
        before, after = work / "input.mlir", work / "fused.mlir"
        before.write_text(checked_generic_text(fragment))
        recipe.run(
            [
                normalizer,
                before,
                "--pass-pipeline=builtin.module(func.func(linalg-fuse-elementwise-ops))",
                "--mlir-print-op-generic",
                "-o",
                after,
            ],
            runner=lambda args: run_checked(args, timeout=180),
            inputs=[before],
            output=after,
        )
        fused_module = parse_mlir_text(after.read_text())
        fused_module.verify()
        candidates = [op for op in fused_module.walk() if isinstance(op, GenericOp)]
        if len(candidates) != 1:
            raise ValueError("stock compiler did not produce one proved fused encoder")
        fused = candidates[0]
        if not _complete_elementwise(fused) or len(fused.body.block.args) != 2:
            raise ValueError("stock fusion changed complete pointwise coverage or contracts")
        source_ops = set(producer.body.block.ops) | set(consumer.body.block.ops)
        # Expected source graph: substitute only the producer's yielded value
        # into the original cast, preserving every scalar op/property/attribute.
        expected_map = {producer.body.block.args[0]: 0}
        source_seen = set()
        producer_expr = _expression(producer.body.block.last_op.arguments[0], expected_map, source_ops, source_seen)
        # One temporary symbolic argument represents the exact producer yield.
        expected_seen = set()
        expected = _expression(body[0].result, {consumer.body.block.args[0]: producer_expr}, source_ops, expected_seen)
        expected = _substitute_argument(expected)
        actual_seen = set()
        actual = _expression(
            fused.body.block.last_op.arguments[0], {fused.body.block.args[0]: 0}, set(fused.body.block.ops), actual_seen
        )
        if actual != expected or any(op not in actual_seen for op in tuple(fused.body.block.ops)[:-1]):
            raise ValueError("stock fusion changed scalar arithmetic, attributes or added operations")
        # Reify the verified fusion with the original source scalar operations.
        # This restores unchanged local constants and terminator provenance that
        # the stock pass may hoist/drop; it does not add a clamp or quantizer.
        new_body = Block(arg_types=[producer.body.block.args[0].type, consumer.body.block.args[1].type])
        values = {producer.body.block.args[0]: new_body.args[0]}
        for op in tuple(producer.body.block.ops)[:-1]:
            clone = op.clone(value_mapper=values)
            new_body.add_op(clone)
            values.update(zip(op.results, clone.results, strict=True))
        values[consumer.body.block.args[0]] = values[producer.body.block.last_op.arguments[0]]
        for op in consumer.body.block.ops:
            clone = op.clone(value_mapper=values)
            new_body.add_op(clone)
            values.update(zip(op.results, clone.results, strict=True))
        replacement = consumer.clone()
        replacement.detach_region(0)
        replacement.add_region(Region(new_body))
        replacement.operands = [producer.inputs[0], consumer.outputs[0]]
        replacement.attributes["prov.fused_elementwise_sources"] = ArrayAttr(
            [DictionaryAttr(dict(producer.attributes)), DictionaryAttr(dict(consumer.attributes))]
        )
        replacement.verify()
        Rewriter.replace_op(consumer, replacement)
        Rewriter.erase_op(producer)
        records.append(
            {
                "before_sha256": sha256_file(before),
                "compiler_fused_sha256": sha256_file(after),
                "scalar_DAG_and_attributes_equal": True,
                "no_guard_or_effect_crossed": True,
                "source_provenance_retained": True,
            }
        )
    module.verify()
    if source.read_bytes() != original or snapshot.read_bytes() != original or policy.validate() != bindings:
        raise ValueError("encoder fusion changed source or selected bindings")
    result = output / "model.mlir"
    result.write_text(checked_generic_text(module))
    recipe.completed_product(result, kind="mlir")
    write_pretty_json(
        output / "report.json",
        {
            "schema": "merlin.proved_clamp_encoder_fusion.v1",
            "source_sha256": sha256_file(source),
            "output_sha256": sha256_file(result),
            "pairs": records,
            "source_quantizer_replaced": False,
        },
    )
    return result


def _substitute_argument(expression):
    if expression[0] == "argument":
        return expression[1] if isinstance(expression[1], tuple) else expression
    return (*expression[:-1], tuple(_substitute_argument(value) for value in expression[-1]))
