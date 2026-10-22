"""Extract typed source windows while retaining literal-dependent source proofs.

The result is a source projection, not host admission or numerical qualification.
Reverification reconstructs it from the exact original source; caller proof records
must be reproved on the returned operations, never copied by ordinal.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import StringIO

from xdsl.dialects import arith, tensor
from xdsl.dialects.builtin import FunctionType, ModuleOp
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.ir import Block, Operation, Region
from xdsl.printer import Printer

from merlin.common import mlir_query as mq


@dataclass(frozen=True)
class SourceWindow:
    module: ModuleOp
    inputs: tuple
    outputs: tuple
    operation_pairs: tuple
    start: int
    stop: int


def text(module) -> str:
    stream = StringIO()
    Printer(stream=stream).print_op(module)
    return stream.getvalue() + "\n"


def extract_window(module, start: int, stop: int) -> SourceWindow:
    """Clone an inclusive single-block window, all live outputs and constant closure.

    External runtime tensors remain explicit typed inputs. Literal tensors, scalar
    constants, splats and empty initializers remain internal, including operands
    captured by nested regions. No other computation is pulled across the cut.
    """
    functions = [op for op in module.walk() if isinstance(op, FuncOp)]
    if len(functions) != 1 or len(functions[0].body.blocks) != 1:
        raise ValueError("source window requires one single-block function")
    original = functions[0]
    if tuple(module.body.block.ops) != (original,):
        raise ValueError("source window requires a self-contained single-function module")
    if set(original.properties) - {"sym_name", "function_type", "sym_visibility"}:
        raise ValueError("source window cannot remap per-argument or unknown function properties")
    block = original.body.blocks[0]
    ops = tuple(block.ops)
    if type(start) is not int or type(stop) is not int or not 0 <= start <= stop < len(ops):
        raise ValueError("source window bounds are invalid")
    kept = set(ops[start:stop + 1])
    if any(isinstance(op, ReturnOp) or op.successors for op in kept):
        raise ValueError("source window cannot contain function return or CFG edges")
    pending = list(kept)
    while pending:
        for nested in pending.pop().walk():
            for operand in nested.operands:
                owner = operand.owner
                if (isinstance(owner, Operation) and owner.parent is block
                        and type(owner) in (arith.ConstantOp, tensor.SplatOp, tensor.EmptyOp)
                        and owner not in kept):
                    kept.add(owner)
                    pending.append(owner)
    ordered = [op for op in ops if op in kept]
    inside = {nested for op in ordered for nested in op.walk()}
    boundary, seen = [], set()
    for op in ordered:
        for nested in op.walk():
            for operand in nested.operands:
                owner = operand.owner
                if isinstance(owner, Operation) and owner in inside:
                    continue
                if isinstance(owner, Block) and owner is not block:
                    continue
                if operand not in seen:
                    boundary.append(operand)
                    seen.add(operand)
    # Constants pulled in for source premises are not new observable results.
    outputs = [value for op in ops[start:stop + 1] for value in op.results
               if any(use.operation not in inside for use in value.uses)]
    if not outputs:
        raise ValueError("source window has no externally used outputs")
    target_block = Block(arg_types=[value.type for value in boundary])
    mapping = dict(zip(boundary, target_block.args, strict=True))
    pairs = []
    for op in ordered:
        clone = op.clone(value_mapper=mapping)
        target_block.add_op(clone)
        pairs.extend(zip(op.walk(), clone.walk(), strict=True))
    target_block.add_op(ReturnOp(*(mapping[value] for value in outputs)))
    function = FuncOp("forward", FunctionType.from_lists(
        [value.type for value in boundary], [value.type for value in outputs]), Region([target_block]))
    function.attributes.update(original.attributes)
    if "sym_visibility" in original.properties:
        function.properties["sym_visibility"] = original.properties["sym_visibility"]
    extracted = ModuleOp([function])
    extracted.attributes.update(module.attributes)
    extracted.verify()
    return SourceWindow(extracted, tuple(boundary), tuple(outputs), tuple(pairs), start, stop)


def verify_window(source: str, projected: str, *, source_sha256: str,
                  projected_sha256: str, start: int, stop: int) -> SourceWindow:
    """Reopen exact bytes and reconstruct before exposing source/clone associations.

    Provenance attributes, typed arithmetic, literal bits, nested captures and all
    live outputs must match. This grants no inherited trace or host proof: callers
    re-run the original proof on these operations and retain its separate premises.
    """
    if sha256(source.encode()).hexdigest() != source_sha256:
        raise ValueError("source window original content pin differs")
    if sha256(projected.encode()).hexdigest() != projected_sha256:
        raise ValueError("source window projection content pin differs")
    result = extract_window(mq.parse(source), start, stop)
    if text(result.module) != text(mq.parse(projected)):
        raise ValueError("source window does not equal exact source reconstruction")
    return result
