"""Original-only bounded source selection for operational loop compiler gates.

This predicate never executes a producer and never inspects a proposed target.
A returned ID selects a proof model, never verification/delivery authority.
"""

from __future__ import annotations

from xdsl.dialects import builtin, func, scf

from merlin.common.ir_lock import IR_LOCK
from merlin.frontends.linalg_mlir import make_context
from merlin.transforms.normalization import constant_integer

from .compiler_profile import SourceOutsideProfile
from .memory_scalar_nan_admission import _parse_bounded, admit_loop_scalar_attributes

WORD_PROFILE = "compiler-static-ordered-scf-index64-v1"
GENERIC_PROFILE = "compiler-static-ordered-scf-generic-value-index64-v1"
UNROLL = "ordered_loop_unrolling"
TILE = "ordered_loop_segmentation"
MAX_RAW_BYTES = 4 * 1024 * 1024
MAX_WORD_SYNTAX = 2048
MAX_WORK = 65536


def _outside(code, reason):
    return SourceOutsideProfile(WORD_PROFILE, code, reason)


def _depth(loop):
    count = 1
    owner = loop.parent_op()
    while type(owner) is scf.ForOp:
        count += 1
        owner = owner.parent_op()
    if type(owner) is not func.FuncOp:
        raise _outside("structured-ancestry", "selected loops require function/SCF lexical ancestry")
    return count


def _word(module, *, kind, factor, tile, depth):
    from .lean_graph_pipeline_gate_profile_prepare import _admit
    from .lean_graph_state_pipeline_capture import index_operation
    from .lean_graph_state_scalar_capture import project

    operations = tuple(module.walk())
    work = sum(
        1
        + len(op.operands)
        + len(op.results)
        + sum(len(block.args) for region in op.regions for block in region.blocks)
        for op in operations
    )
    if work > MAX_WORD_SYNTAX:
        raise _outside("word-syntax-bound", "complete word model has weighted syntax bound2048")
    # These are actual source-only model checks; the identity pair grants no
    # theorem and no claim that any proposed transformation is correct.
    try:
        admit_loop_scalar_attributes(module)
        _admit([module])
        project(module, module)
    except ValueError as error:
        raise _outside("word-source-model", str(error)) from error
    for op in operations:
        if type(op) is scf.ForOp:
            lower = constant_integer(op.lb)
            if type(lower) is not int or lower < 0:
                raise _outside(
                    "all-loop-nonnegative-lower",
                    "every modeled SCF loop requires an established nonnegative literal lower bound",
                )
        control = index_operation(op)
        if control is not None:
            if control["kind"] != "index-constant":
                raise _outside("source-index-arithmetic", "ordinary index add/min needs another model")
            if not -(2**63) <= control["value"] < 2**63:
                raise _outside("source-index-literal", "actual index literal is outside signed64")
    selected = [op for op in operations if type(op) is scf.ForOp and _depth(op) == depth]
    if not selected:
        raise _outside("selected-depth-absent", "no actual source loop occurs at the selected depth")
    predicted = work
    for loop in selected:
        lower, upper, step = (constant_integer(value) for value in (loop.lb, loop.ub, loop.step))
        if any(type(value) is not int for value in (lower, upper, step)) or not 0 <= lower <= upper or step < 1:
            raise _outside(
                "selected-static-controls", "selected controls need nonnegative static bounds/positive stride"
            )
        if any(type(value.type) is not builtin.IntegerType for value in loop.iter_args):
            raise _outside("selected-word-carries", "selected loop carried state must be signless scalar words")
        chosen = factor if kind == UNROLL else tile
        if upper + step * chosen >= 2**63 or upper + step >= 2**63:
            raise _outside("selected-control-bounds", "selected/generated loop controls do not fit signed64")
        trips = (upper - lower + step - 1) // step
        body_work = sum(
            1
            + len(op.operands)
            + len(op.results)
            + sum(len(block.args) for region in op.regions for block in region.blocks)
            for op in loop.body.walk()
        )
        if kind == UNROLL:
            lanes = min(trips, factor)
            if lanes * max(1, body_work) > MAX_WORK:
                raise _outside("predicted-proof-work", "source-only expansion estimate exceeds65536 work")
            # Existing generated query/decoder profiles also have2048 syntax.
            # Charge clones plus inserted IV controls before producer execution.
            predicted += max(0, lanes - 1) * body_work + 4 * lanes + 12
            bulk_upper = lower + (trips // factor) * factor * step
            if (
                trips > factor
                and trips % factor == 0
                and (bulk_upper + step * factor * tile >= 2**63 or bulk_upper + step * factor >= 2**63)
            ):
                raise _outside("auxiliary-control-bounds", "existing bulk proof scaffold controls do not fit signed64")
        else:
            predicted += 32 + body_work
    if predicted > MAX_WORD_SYNTAX:
        raise _outside("predicted-target-syntax", "source-only generated-syntax estimate exceeds2048 word profile")
    return WORD_PROFILE


def admit_source(raw, *, kind, factor=2, tile=3, depth=1, index_bits=64):
    if kind not in (UNROLL, TILE):
        raise _outside("transform-selection", "one explicit ordered loop transform is required")
    if type(index_bits) is not int or index_bits != 64:
        raise _outside("index-width", "operational compiler loop gates support signed-index64")
    if type(factor) is not int or factor < 2 or type(tile) is not int or tile < 2:
        raise _outside("transform-parameter", "integer loop factor/tile at least2 required")
    if type(depth) is not int or depth < 1:
        raise _outside("selected-depth", "selected loop depth must be a positive one-based integer")
    if type(raw) is not str or len(raw.encode()) > MAX_RAW_BYTES:
        raise _outside("raw-source-bound", "complete original source must fit the declared4MiB model bound")
    # Parser/resource/custody errors are not converted into source exclusions.
    # Passing a context adds no global parser/producer normalization policy.
    with IR_LOCK:
        module = _parse_bounded(raw, maximum_bytes=MAX_RAW_BYTES, memory=False, context=make_context())
    try:
        return _word(module, kind=kind, factor=factor, tile=tile, depth=depth)
    except SourceOutsideProfile as word_error:
        # The bounded generic profile is an extension of source coverage, not a
        # target-side fallback. Each bounded generic adjacent pass is checked
        # independently; a sibling-loop middle is outside its singleton model.
        from .compiler_loop_generic_profile import OutsideProfile, source_profile

        try:
            source_profile(raw, kind=kind, factor=factor, tile=tile, depth=depth, index_bits=index_bits)
        except OutsideProfile as generic_error:
            raise SourceOutsideProfile(
                WORD_PROFILE + "|" + GENERIC_PROFILE,
                "outside-word-and-generic-model",
                str(word_error) + "; " + str(generic_error),
            ) from generic_error
        return GENERIC_PROFILE
