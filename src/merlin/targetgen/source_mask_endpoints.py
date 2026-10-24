"""Fold masks that a literal comparison decides over a provably bounded value.

A mask generic compares each element of a tensor against a closed literal.
When the closed interval of every value that tensor can hold (for example a
uniform draw built from a bit-masked integer scaled by a literal) lies
entirely on one side of the literal, the mask is uniform for every input and
every generator state. The mask is replaced by that uniform literal, the
remaining arithmetic is folded where it becomes closed, and private generator
state that no longer reaches an observable output is removed. Everything is
read from the IR: no case record, overload name, provenance label or tensor
payload participates. Unknown effects, external state and undecided masks
refuse and leave the program unchanged.
"""

import hashlib
import json
from fractions import Fraction

from merlin.targetgen.native_payload_carriers import op_name


def closed_private_state(module, function):
    from xdsl.dialects import arith
    from xdsl.rewriter import Rewriter

    from merlin.targetgen.source_primitive_effects import pure_tensor_source

    globals_ = {}
    for op in module.body.block.ops:
        if op is function:
            continue
        if (
            op_name(op) != "ml_program.global"
            or getattr(op.properties.get("sym_visibility"), "data", None) != "private"
        ):
            return False
        symbol = getattr(op.properties.get("sym_name"), "data", None)
        if not symbol or symbol in globals_ or "value" not in op.properties:
            return False
        globals_[symbol] = op
    clone = function.clone()
    for op in list(clone.walk()):
        if op_name(op) not in {"ml_program.global_load", "ml_program.global_store"}:
            continue
        ref = op.properties.get("global")
        symbol = getattr(getattr(ref, "root_reference", None), "data", None)
        if symbol not in globals_ or ref.nested_references:
            return False
        declared = globals_[symbol].properties
        if op_name(op) == "ml_program.global_load":
            if op.operands or len(op.results) != 1 or op.results[0].type != declared["type"]:
                return False
            Rewriter.replace_op(op, arith.ConstantOp(declared["value"]))
        else:
            if op.results or len(op.operands) != 1 or op.operands[0].type != declared["type"]:
                return False
            Rewriter.erase_op(op)
    return pure_tensor_source(clone)


def _decided(predicate, bounds, constant):
    """The comparison result shared by every value in bounds, or None."""
    from merlin.targetgen.closed_scalar import predicate_holds

    c = Fraction(constant)
    relations = set()
    if bounds.lo < c:
        relations.add(-1)
    if bounds.hi > c:
        relations.add(1)
    if bounds.lo <= c <= bounds.hi:
        relations.add(0)
    outcomes = {predicate_holds(predicate, r, False, floating=True) for r in relations}
    return outcomes.pop() if len(outcomes) == 1 else None


def decided_masks(function):
    """[(generic, bit, evidence)] for every elementwise mask decided by a closed interval."""
    import math

    from merlin.targetgen.closed_scalar import interval, literal

    found = []
    memo = {}
    for op in function.body.block.ops:
        if op.name != "linalg.generic" or len(op.results) != 1 or str(op.results[0].type.element_type) != "i1":
            continue
        if any(it.data.value != "parallel" for it in op.get_iterator_types()):
            continue
        yielded = op.body.block.last_op.operands[0]
        compare = yielded.owner
        if getattr(compare, "name", None) != "arith.cmpf" or compare.fastmath.data:
            continue
        predicate = compare.predicate.value.data
        a, b = compare.operands
        swapped = False
        constant = literal(b)
        tested = a
        if constant is None:
            constant, tested, swapped = literal(a), b, True
        if not isinstance(constant, float) or not math.isfinite(constant):
            continue
        args = op.body.block.args
        if tested not in args or args.index(tested) >= len(op.inputs):
            continue
        bounds = interval(op.inputs[args.index(tested)], memo)
        if bounds is None or bounds.empty:
            continue
        if swapped:
            # c PRED u  ==  u PRED' c with the relation reversed.
            reverse = {2: 4, 3: 5, 4: 2, 5: 3, 9: 11, 10: 12, 11: 9, 12: 10}
            predicate = reverse.get(predicate, predicate)
        bit = _decided(predicate, bounds, constant)
        if bit is None:
            continue
        found.append(
            (
                op,
                bool(bit),
                dict(
                    predicate=predicate,
                    literal=constant,
                    interval=[str(bounds.lo), str(bounds.hi)],
                ),
            )
        )
    return found


def fold_closed_scalars(function):
    """Replace generic-body scalars whose value is closed by literals; return the count."""
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import FloatAttr, IntegerAttr, IntegerType
    from xdsl.rewriter import Rewriter

    from merlin.targetgen.closed_scalar import UNKNOWN, evaluate

    count = 0
    for generic in [op for op in function.walk() if op.name == "linalg.generic"]:
        for op in list(generic.body.block.ops):
            if op.name in {"arith.constant", "linalg.yield", "linalg.index"} or len(op.results) != 1:
                continue
            typ = op.results[0].type
            value = evaluate(op.results[0], {})
            if value is UNKNOWN or value is None:
                continue
            if isinstance(typ, IntegerType):
                attr = IntegerAttr(int(value), typ)
            elif str(typ) in {"f16", "bf16", "f32", "f64"}:
                attr = FloatAttr(float(value), typ)
            else:
                continue
            Rewriter.replace_op(op, arith.ConstantOp(attr))
            count += 1
    return count


def _erase_dead(function):
    """Erase unused, effect-free top-level operations, including dead loops."""
    from merlin.llvmlower.entry_weight_projection import _pure_dce

    def effect_free(op):
        # Pure reads of private state (closed_private_state) and value-only
        # arith/math/tensor/scf/linalg operations on tensors.
        for child in op.walk():
            if op_name(child) == "ml_program.global_load":
                continue
            if type(child).__module__.startswith(
                (
                    "xdsl.dialects.arith",
                    "xdsl.dialects.math",
                    "xdsl.dialects.tensor",
                    "xdsl.dialects.scf",
                    "xdsl.dialects.linalg",
                )
            ):
                continue
            return False
        return True

    while True:
        _pure_dce(function)
        changed = False
        for op in reversed(tuple(function.body.block.ops)):
            if op.results and all(not r.uses for r in op.results) and effect_free(op):
                op.parent.erase_op(op)
                changed = True
        if not changed:
            return


def specialize(directory, report):
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import IntegerAttr, i1
    from xdsl.rewriter import InsertPoint, Rewriter

    from merlin.llvmlower.structural_folds import prune_unused_tensor_inputs
    from merlin.targetgen.source_primitive_effects import pure_tensor_source, single_forward

    found = single_forward(directory, report)
    if found is None:
        return None
    record, original, module, function = found
    if len(function.body.blocks) != 1:
        return None
    stateful = any(op is not function for op in module.body.block.ops)
    if stateful and not closed_private_state(module, function):
        return None
    if not stateful and not pure_tensor_source(function):
        return None
    masks = decided_masks(function)
    if not masks:
        return None
    entry = function.body.block
    evidence = []
    for generic, bit, detail in masks:
        constant = arith.ConstantOp(IntegerAttr(int(bit), i1))
        filled = tensor.SplatOp(constant.result, [], generic.results[0].type)
        Rewriter.insert_op([constant, filled], InsertPoint.at_start(entry))
        generic.results[0].replace_all_uses_with(filled.result)
        evidence.append(dict(mask=int(bit), **detail))

    # Private generator state is unobservable (closed_private_state): its
    # updates are dropped, after which unused generator work is dead code.
    for op in list(function.walk()):
        if op_name(op) == "ml_program.global_store":
            Rewriter.erase_op(op)
    _erase_dead(function)
    folded = fold_closed_scalars(function)
    _erase_dead(function)
    prune_unused_tensor_inputs(module)
    _erase_dead(function)
    if any(op_name(op).startswith("ml_program.") for op in function.walk()):
        return None  # Generator state still reaches an observable result.
    for op in list(module.body.block.ops):
        if op is not function:
            Rewriter.erase_op(op)
    if not pure_tensor_source(function):
        return None
    module.verify()
    prepared = str(module) + "\n"
    (directory / "model.mlir").write_text(prepared)
    proof = dict(
        method="ir_closed_interval_mask_endpoint",
        masks=evidence,
        folded_closed_scalars=folded,
        state_scope="capsule_private; no observable output depends on generator state",
        retained_arithmetic="the IR's own arithmetic with the decided mask literal and its typed IEEE boundary",
        source_sha256=hashlib.sha256(original).hexdigest(),
        prepared_sha256=hashlib.sha256(prepared.encode()).hexdigest(),
        parameters="live IR operations, literals and closed value intervals only; no case record or tensor payload",
    )
    record["source_specialization"] = proof
    (directory / "native-source-specialization.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof
