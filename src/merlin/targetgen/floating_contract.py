"""Which declared numeric contract discharges a program's floating work.

The selected software spec (numerical_semantics.component_domain.
floating_storage_arithmetic) declares that IEEE arithmetic on floating storage
is discharged by the declared quantized model, while native eager-bit proofs
discharge movement, exact conversions out of integer/bool storage, literal
fills and identities. This module decides that from the IR alone: a program
needs the quantized model exactly when, after the closed identity
normalizations, some live floating arithmetic, float-to-float conversion or
data-dependent floating selection still consumes data that originates in a
floating function argument.

A literal scale or offset is affine arithmetic, except the exact identities
x * 1 and x / 1, which the closed normalization (fold_unit_scales) removes:
they are movement, graded on eager bits (ruling "exact identities are
bit-exact"). A widening followed by a narrowing back to the original format
is an identity too. No source overload, provenance label, case record or
tensor payload participates.
"""

from merlin.targetgen.native_payload_carriers import op_name

FLOAT_TYPES = {"f16", "bf16", "f32", "f64"}
FLOAT_ARITHMETIC = {
    "arith.addf",
    "arith.subf",
    "arith.mulf",
    "arith.divf",
    "arith.remf",
    "arith.negf",
    "arith.maximumf",
    "arith.minimumf",
    "arith.maxnumf",
    "arith.minnumf",
    "arith.extf",
    "arith.truncf",
}
CONTRACTIONS = {
    "linalg.matmul",
    "linalg.batch_matmul",
    "linalg.matvec",
    "linalg.vecmat",
    "linalg.dot",
}
TERMINATORS = {"func.return", "linalg.yield", "scf.yield", "tensor.yield", "scf.condition"}


def _element(value):
    typ = value.type
    return str(getattr(typ, "element_type", typ))


def _is_identity(op):
    """Only an exact widening-then-narrowing round trip is an identity operation."""
    name = op_name(op)
    if name == "arith.truncf":
        producer = _through_identities(op.operands[0]).owner
        return op_name(producer) == "arith.extf" and str(producer.operands[0].type) == str(
            op.results[0].type
        )
    return False


def _carried_source(value):
    """Step back through one elementwise generic boundary (block argument or result)."""
    owner = value.owner
    if op_name(owner) == "linalg.generic":
        return owner.body.block.last_op.operands[value.index]
    if not hasattr(owner, "results") and hasattr(owner, "parent_op"):
        parent = owner.parent_op()
        if op_name(parent) == "linalg.generic" and value.index < len(parent.inputs):
            return parent.inputs[value.index]
    return None


def _through_identities(value):
    """Step back through elementwise generic boundaries to the reproduced value."""
    seen = set()
    while value not in seen:
        seen.add(value)
        carried = _carried_source(value)
        if carried is not None:
            value = carried
            continue
        return value
    return value


def _ends_in_narrowing(value, seen=None):
    """Does every use of value reach an exact narrowing back through identities only?"""
    seen = set() if seen is None else seen
    if value in seen:
        return True
    seen.add(value)
    uses = list(value.uses)
    if not uses:
        return False
    for use in uses:
        op = use.operation
        name = op_name(op)
        if name == "arith.truncf" and _is_identity(op):
            continue
        if name == "linalg.yield" and op_name(op.parent_op()) == "linalg.generic":
            if _ends_in_narrowing(op.parent_op().results[use.index], seen):
                continue
        if name == "linalg.generic" and use.index < len(op.inputs):
            if _ends_in_narrowing(op.body.block.args[use.index], seen):
                continue
        return False
    return True


def _nan_class_test(condition):
    """A self-comparison that only distinguishes NaN (cmpf uno/ord x, x).

    Selecting on it re-encodes NaN payloads (quieting, canonical NaN); every
    non-NaN value passes unchanged, so it is not arithmetic on finite data.
    """
    owner = condition.owner
    if op_name(owner) != "arith.cmpf":
        return False
    first, second = owner.operands
    return first is second and owner.predicate.value.data in {7, 14}


def _live_operations(function):
    """Operations whose results reach the function's results."""
    live = set()
    pending = [op for op in function.walk() if op.name == "func.return"]
    while pending:
        op = pending.pop()
        if op in live:
            continue
        live.add(op)
        for value in op.operands:
            owner = value.owner
            if not hasattr(owner, "results"):
                # Block argument: its structured parent and the matching operand are live.
                parent = owner.parent_op() if hasattr(owner, "parent_op") else None
                if parent is not None and parent is not function:
                    pending.append(parent)
                    operands = list(parent.operands)
                    if value.index < len(operands) and hasattr(operands[value.index].owner, "results"):
                        pending.append(operands[value.index].owner)
                continue
            pending.append(owner)
            for region in owner.regions:
                for block in region.blocks:
                    if block.ops and block.last_op.name in TERMINATORS:
                        pending.append(block.last_op)
        parent = op.parent_op()
        if op.name in TERMINATORS and parent is not None and parent is not function:
            pending.append(parent)
    return live


def floating_arithmetic(module):
    functions = [op for op in module.walk() if op.name == "func.func" and op.sym_name.data == "forward"]
    if len(functions) != 1:
        return dict(quantized=False, ops=[], reason="no unique forward function")
    function = functions[0]
    tainted = {arg for arg in function.body.block.args if _element(arg) in FLOAT_TYPES}
    changed = True
    while changed:
        changed = False
        for op in function.walk():
            if op is function:
                continue
            operands = list(op.operands)
            if not any(value in tainted for value in operands):
                continue
            for region in op.regions:
                for block in region.blocks:
                    for index, arg in enumerate(block.args):
                        if index < len(operands) and operands[index] in tainted and arg not in tainted:
                            tainted.add(arg)
                            changed = True
            if op.name in TERMINATORS:
                parent = op.parent_op()
                for index, value in enumerate(op.operands):
                    if value in tainted and index < len(parent.results) and parent.results[index] not in tainted:
                        tainted.add(parent.results[index])
                        changed = True
            for result in op.results:
                if result not in tainted:
                    tainted.add(result)
                    changed = True
    live = _live_operations(function)
    found = set()
    for op in function.walk():
        if op not in live or not any(value in tainted for value in op.operands):
            continue
        name = op_name(op)
        floating_result = any(_element(r) in FLOAT_TYPES for r in op.results)
        if name == "arith.select" and floating_result:
            # A data-dependent choice between floating data and something else
            # (a rectifier or clamp) is floating arithmetic; a choice between two
            # literals or two copies of the same value is not.
            condition, first, second = op.operands
            if (
                condition in tainted
                and first is not second
                and (first in tainted or second in tainted)
                and not _nan_class_test(condition)
            ):
                found.add(name)
        elif (name in FLOAT_ARITHMETIC or name.startswith("math.")) and floating_result:
            if not _is_identity(op):
                found.add(name)
        elif name in CONTRACTIONS or name.startswith("linalg.conv"):
            found.add(name)
    if "arith.extf" in found:
        # A widening consumed only by an exact narrowing back is part of that identity.
        widenings = [
            op for op in function.walk() if op.name == "arith.extf" and op in live and op.operands[0] in tainted
        ]
        if all(_ends_in_narrowing(op.results[0]) for op in widenings):
            found.discard("arith.extf")
    return dict(quantized=bool(found), ops=sorted(found))


def requires_quantized_model(module):
    """Decide on a clone normalized by the closed identity folds."""
    from merlin.llvmlower.entry_weight_projection import _pure_dce
    from merlin.targetgen.core_aten_numeric import normalize_closed_identities

    clone = module.clone()
    try:
        normalize_closed_identities(clone)
    except ValueError:  # a fold that refuses the program keeps its raw arithmetic
        clone = module.clone()
    for function in clone.walk():
        if function.name == "func.func":
            _pure_dce(function)
    return floating_arithmetic(clone)
