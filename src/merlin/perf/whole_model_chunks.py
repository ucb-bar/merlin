"""Cut a whole model's flat ``@forward`` into bounded, sequentially-called functions.

LLVM's compile cost on one huge function is superlinear in its size, so an open-model build may
split the host program into chunks before lowering it (``build(chunk_ops=...)``). The cut moves
only which function's text an op's clone sits in; nothing a value computes, or its order, changes.

``chunk_ops="auto"`` derives the size from the program itself (:func:`resolve_chunk_ops`): a forward
body larger than :data:`DEFAULT_CHUNK_OPS` is cut at that size, and one that fits is left unchunked,
byte for byte, and recorded as unchunked so its reference arm is the unchunked one too.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

__all__ = [
    "AUTO",
    "CHUNK_PREFIX",
    "DEFAULT_CHUNK_OPS",
    "chunk_forward",
    "chunk_symbols",
    "forward_body_size",
    "resolve_chunk_ops",
]

#: The ``chunk_ops`` value that derives the chunk size from the program (see :func:`resolve_chunk_ops`).
AUTO = "auto"
#: The chunk size ``"auto"`` cuts a large forward at. Measured on SmolVLA's open-model build (a
#: 1,551-call ``forward``): unchunked, its host program took over two hours to compile; cut at 1,000
#: ops, the whole build took about eleven minutes.
DEFAULT_CHUNK_OPS = 1000

#: The symbol of chunk ``i`` is ``CHUNK_PREFIX + str(i)``, for ``i`` in ``range(chunk_forward(...))``.
CHUNK_PREFIX = "merlin_forward_chunk_"

#: A cheap producer (compute_groups.classify's own ``_PRODUCERS``): never left as the LAST op of a
#: chunk, because the pipeline's one-shot-bufferize does not handle one of these -- a destination a
#: later op writes into, in the out-params/destination-passing sense -- crossing a function-call
#: boundary from its own write the way a dispatch's single ``committed`` value crosses one. Measured:
#: every chunk_ops tried on a real capture (14 through 835 resulting chunks) segfaulted the upstream
#: lowering pipeline identically, which is what pointed at the CUT'S POSITION, not its size -- a
#: 53k-op quantized model is dense enough with tensor.empty()-then-write pairs that almost any cut
#: lands between one.
_CHEAP_PRODUCERS = frozenset({"tensor.empty", "arith.constant", "tensor.splat", "linalg.fill"})

#: Ops whose result is never RETURNED from a chunk: a later reader gets a copy of the op re-created in
#: the caller from the values it reads. They compute nothing (a reshape view, a constant, an empty
#: destination), and returned they can make one chunk return the SAME buffer twice once the lowering
#: canonicalizes: a row scale ``s`` and ``expand_shape(collapse_shape(s))`` both leaving a chunk fold
#: to ``s``, and buffer-results-to-out-params{modify-public-functions hoist-static-allocs} erases that
#: allocation once per returned copy -- a use-after-free that segfaulted SmolVLA int8full's lowering.
_REMATERIALIZED = frozenset(
    {"tensor.expand_shape", "tensor.collapse_shape", "arith.constant", "tensor.splat", "tensor.empty"}
)


def _dynamically_shaped(value) -> bool:
    """A value whose type has a dynamic extent (or no rank): it cannot leave a chunk as a result,
    because buffer-results-to-out-params can only turn a statically shaped result into an out-param.
    Measured: a SmolVLA int8 capture chunked at op counts reached ``memref<?xi64>`` live-outs and
    failed there, after the cheap-producer rule had cleared the earlier segfault."""
    from xdsl.dialects.builtin import UnrankedMemRefType, UnrankedTensorType

    kind = value.type
    if isinstance(kind, UnrankedTensorType | UnrankedMemRefType):
        return True
    shape = getattr(kind, "get_shape", None)
    return shape is not None and any(extent < 0 for extent in shape())


def _top_level(op, block):
    """The op of ``block`` that ``op`` is, or is nested inside (``None`` when it is in neither)."""
    while op is not None and op.parent_block() is not block:
        op = op.parent_op()
    return op


def _dynamic_reach(body_ops: Sequence[Any]) -> list[int]:
    """``reach[c]``: the furthest position (index into ``body_ops``; ``len(body_ops)`` for the
    terminator) at which a dynamically shaped value defined before position ``c`` is still read.
    A cut at ``c`` would make that value a chunk's live-out exactly when ``reach[c] >= c``."""
    position = {id(op): i for i, op in enumerate(body_ops)}
    n = len(body_ops)
    block = body_ops[0].parent_block() if body_ops else None
    last = [-1] * n
    for i, op in enumerate(body_ops):
        for result in op.results:
            if not _dynamically_shaped(result):
                continue
            for use in result.uses:
                user = _top_level(use.operation, block)
                last[i] = max(last[i], position.get(id(user), n) if user is not None else n)
    reach = [-1] * (n + 1)
    for c in range(1, n + 1):
        reach[c] = max(reach[c - 1], last[c - 1])
    return reach


def _chunk_bounds(body_ops: Sequence[Any], chunk_ops: int) -> list[int]:
    """Cut points for ``body_ops`` near every ``chunk_ops``'th op, each at a LEGAL position: not
    right after a cheap producer (see ``_CHEAP_PRODUCERS``), and not where a dynamically shaped value
    would cross (see ``_dynamically_shaped``) -- the producer and every reader of such a value stay in
    one chunk. A cut is pulled back to the nearest legal position after the previous bound, else
    pushed forward to the nearest one after its target; with neither, the rest is one chunk."""
    n = len(body_ops)
    reach = _dynamic_reach(body_ops)

    def legal(cut: int) -> bool:
        return body_ops[cut - 1].name not in _CHEAP_PRODUCERS and reach[cut] < cut

    bounds = [0]
    target = chunk_ops
    while target < n:
        cut = next((c for c in range(target, bounds[-1], -1) if legal(c)), None)
        if cut is None:
            cut = next((c for c in range(target + 1, n) if legal(c)), None)
        if cut is None:
            break
        bounds.append(cut)
        target = max(target + chunk_ops, cut + 1)
    return bounds


def resolve_chunk_ops(requested: int | str | None, *, forward_ops: int | None = None) -> int | None:
    """The chunk size a build cuts its forward at, for the requested ``chunk_ops``.

    ``None`` keeps the unchunked program. A positive integer (or its decimal spelling) is itself.
    :data:`AUTO` is :data:`DEFAULT_CHUNK_OPS` when the forward body has more ops than that, and
    ``None`` when it fits in one chunk (``forward_ops``, from :func:`forward_body_size`; unknown counts
    as large). Anything else is refused: a misspelled size must not silently build unchunked."""
    if requested is None:
        return None
    if isinstance(requested, str):
        spelled = requested.strip().lower()
        if spelled == AUTO:
            return None if forward_ops is not None and forward_ops <= DEFAULT_CHUNK_OPS else DEFAULT_CHUNK_OPS
        if not spelled.isdigit():
            raise ValueError(f"chunk_ops is a positive op count or {AUTO!r}, not {requested!r}")
        requested = int(spelled)
    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
        raise ValueError(f"chunk_ops is a positive op count or {AUTO!r}, not {requested!r}")
    return requested


def _forward_block(module, function: str):
    from merlin.perf.whole_model_open import OpenModelError

    functions = [op for op in module.walk() if op.name == "func.func" and op.body.blocks]
    functions = [f for f in functions if f.sym_name.data == function] or functions[:1]
    if not functions:
        raise OpenModelError(f"the module has no function @{function} with a body")
    return functions[0].body.blocks[0]


def forward_body_size(module, *, function: str = "forward") -> int:
    """How many top-level ops ``@function``'s body holds before its terminator."""
    return max(len(list(_forward_block(module, function).ops)) - 1, 0)


def chunk_forward(module, *, chunk_ops: int, function: str = "forward") -> int:
    """Split ``@function``'s single flat block into ``chunk_ops``-bounded, sequentially-called
    functions, in place. Returns how many chunk functions were made (0 when the block already fit
    in one).

    LLVM's own compile cost on one huge function is superlinear in its size -- measured directly: a
    105,185-line, 1,551-call ``forward`` (SmolVLA's) took 76 minutes to reach one object, single-
    threaded, dwarfing every other build stage. This is the SAME cut ``externalize_dispatches`` makes
    for a device group (a bounded run of ops, its free values threaded in as arguments, replaced by
    one call at the same position), applied to plain consecutive RUNS of ops instead of to one
    group's own members, and returning every value a LATER op still reads (there can be several) in
    place of the single ``committed`` value a group cut always has exactly one of. Nothing a value
    computes, or the order it runs in, changes: only which function's text an op's clone sits in.
    """
    from xdsl.dialects import func
    from xdsl.ir import Block, Region

    from merlin.perf.whole_model_open import OpenModelError
    from merlin.xdsl_dialects.lowering import outline as OL

    block = _forward_block(module, function)
    ops = list(block.ops)
    if not ops or ops[-1].name != "func.return":
        raise OpenModelError(f"@{function}'s block does not end in func.return")
    body_ops, terminator = ops[:-1], ops[-1]
    if len(body_ops) <= chunk_ops:
        return 0
    bounds = _chunk_bounds(body_ops, chunk_ops) + [len(body_ops)]
    chunks = [body_ops[bounds[i] : bounds[i + 1]] for i in range(len(bounds) - 1)]
    chunks = [c for c in chunks if c]  # a boundary pulled all the way back can empty one
    for index, chunk in enumerate(chunks):
        chunk_ids = {id(op) for op in chunk}
        live_out: list = []
        seen: set[int] = set()

        # A use counts by the top-level op it sits in: a reader nested in a later op's region (a
        # tensor.extract in a generic's body) is outside the chunk, one nested in the chunk is not.
        # Judged by the nested op itself, every value a chunk read only from inside its own regions
        # became a spurious live-out -- among them SmolVLA's dynamically shaped compaction result.
        def outside(use, chunk_ids=chunk_ids) -> bool:
            return id(_top_level(use.operation, block)) not in chunk_ids

        # A value a later op reads leaves the chunk as a result, unless its op is rematerialized (see
        # ``_REMATERIALIZED``): then the op is re-created after the call from what it reads -- the
        # chunk's results, or values the caller already has -- and every value it reads from inside
        # the chunk is handled the same way, transitively.
        remat_ids: set[int] = set()

        def leave(result, chunk_ids=chunk_ids, remat_ids=remat_ids, seen=seen, live_out=live_out) -> None:
            owner = result.owner
            if id(result) in seen:
                return
            seen.add(id(result))
            if id(owner) in chunk_ids and owner.name in _REMATERIALIZED and not owner.regions:
                remat_ids.add(id(owner))
                for operand in owner.operands:
                    if id(getattr(operand, "owner", None)) in chunk_ids:
                        leave(operand)
            else:
                live_out.append(result)

        for op in chunk:
            for result in op.results:
                if any(outside(use) for use in result.uses):
                    leave(result)
        params = OL._free_values(chunk)
        symbol = f"{CHUNK_PREFIX}{index}"

        kblock = Block(arg_types=[p.type for p in params])
        mapping = dict(zip(params, kblock.args, strict=True))
        for op in chunk:
            clone = op.clone(value_mapper=mapping)
            for old, new in zip(op.results, clone.results, strict=True):
                mapping[old] = new
            kblock.add_op(clone)
        kblock.add_op(func.ReturnOp(*[mapping[v] for v in live_out]))
        chunk_fn = func.FuncOp(symbol, ([p.type for p in params], [v.type for v in live_out]), Region([kblock]))
        module.body.block.add_op(chunk_fn)

        call = func.CallOp(symbol, list(params), [v.type for v in live_out])
        block.insert_op_before(call, chunk[0])
        outer = dict(zip(live_out, call.results, strict=True))
        for op in chunk:
            if id(op) in remat_ids:
                clone = op.clone(value_mapper=outer)
                block.insert_op_before(clone, chunk[0])
                outer.update(zip(op.results, clone.results, strict=True))
        for value, result in outer.items():
            value.replace_uses_with_if(result, outside)
        for op in reversed(chunk):
            op.detach()
            op.erase()
    del terminator  # kept in the outer block untouched; named only to state the invariant above
    return len(chunks)


def chunk_symbols(count: int) -> list[str]:
    """The symbols of the ``count`` chunk functions :func:`chunk_forward` made, in call order."""
    return [f"{CHUNK_PREFIX}{index}" for index in range(count)]
