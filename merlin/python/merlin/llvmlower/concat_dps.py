"""Let a `tensor.concat`'s operands be COMPUTED INTO the concatenated buffer (default-off).

READ THE VERDICT FIRST: on this repo's shipped whole-model config the STATIC evidence says this
feature COSTS more than it saves, and the wall is UNMEASURED. It is kept, default-off and fully
documented, because the numbers below are the answer to "is the concat cost still worth attacking?"
and nobody should have to re-derive them. See "MEASURED ON THE EMITTED CODE", below.

WHAT A CONCAT COSTS, AND WHAT IT COSTS *TODAY*
----------------------------------------------
`tensor.concat` contains no arithmetic. It exists to say "these values sit next to each other", and
the compiler pays for that statement in bytes: bufferization allocates the result, gives each
operand a `memref.subview` of it as a destination, and emits a `memref.copy` per operand. The
operand was just written into a buffer of its own; every element is then read back and written
again.

On ``out/artifacts/recaptures/small_llama_int8_consistent`` (whole model, int8) there are 12 such
concats, and they used to be the ONLY reason ``@forward`` called ``@memrefCopy`` at all -- 24 call
sites, two per concat -- because a strided subview destination is exactly the case
``finalize-memref-to-llvm`` declines to emit code for. The K1 per-op profile priced
``tensor.concat`` at 8.41 % of whole-model runtime (60,805 of 722,604 attributed ticks, joined on
``mlir_op``) for 6,144 elements of pure data movement.

That 8.41 % is the PRE-fix price and must not be quoted as the remaining one. MEASURED on the
emitted LLVM IR with the shipped ``expand_memref_copy`` enabled: ``@memrefCopy`` call sites 24 -> 0
and ``llvm.memcpy`` 38 -> 0. The runtime CALL is gone; what is left is a load/store loop. Measured
at the post-bufferization point of the RVV pipeline, the whole model then still runs 50
``memref.copy`` moving 25,928 elements, of which 12 copies / 4,608 elements have a STRIDED
destination -- the shape no vectorizer helps with, and the residue of the concats.

WHAT THE 12 CONCATS ARE FOR
---------------------------
Eight are the rotary tables duplicated to full width, ``cat(cos(f), cos(f))`` and
``cat(sin(f), sin(f))``; four are ``rotate_half``, ``cat(-x2, x1)``. With
``cse_through_provenance`` the two halves of each table are ONE value, so those concats arrive as
``cat(%x, %x)``: one placement is a genuine second location for the same bytes, not redundant work.
The ``rotate_half`` concats have one operand from a ``linalg.generic`` (the negation) and one from a
``tensor.extract_slice`` of another tensor.

WHAT THIS FEATURE DOES
----------------------
The concat is rewritten into the destination-passing chain that denotes the same value, and each
operand whose producer already takes a destination is retargeted to write into the result's slice::

    %e0 = tensor.empty() : tensor<8x16xf32>
    %a  = linalg.generic ... outs(%e0) -> tensor<8x16xf32>
    %r  = tensor.concat dim(1) %a, %b : (...) -> tensor<8x32xf32>

    =>  %d  = tensor.empty() : tensor<8x32xf32>
        %s0 = tensor.extract_slice %d[0, 0] [8, 16] [1, 1]
        %a  = linalg.generic ... outs(%s0) -> tensor<8x16xf32>
        %c0 = tensor.insert_slice %a into %d[0, 0]  [8, 16] [1, 1]
        %c1 = tensor.insert_slice %b into %c0[0, 16] [8, 16] [1, 1]

Nothing here knows what a rotary embedding is. The rule is "a concatenation is a statement about
placement, not a reason to move bytes", keyed on IR STRUCTURE alone -- a static concat, a producer
that declares an ins/outs split, a ``tensor.empty`` destination -- never on an op name, a model, a
shape, a dtype or a target.

WHY IT IS VALUE-PRESERVING
--------------------------
Tensors are values. ``tensor.concat dim(d)`` is DEFINED as the tensor in which operand i occupies
``[offset_i, offset_i + size_i)`` along ``d``, and the chain writes exactly those disjoint,
covering slices in that order, so every element keeps its definition. The only other change is
which uninitialized buffer a producer is handed, and that is sound precisely because the buffer it
had was a ``tensor.empty`` -- a value whose contents were already undefined. Reading it was
undefined before and is undefined now. (Same argument as upstream's empty-tensor elimination.)
MEASURED: the model's output is BIT-IDENTICAL to the baseline in every arm, sha256
``cc60e8a9...``, gate ``ok=True`` on ``tiers=['fp32', 'w8a8']``.

WHAT IT REFUSES TO DO (fail closed, and count every refusal)
------------------------------------------------------------
1. **Static shapes only** -- the result and every operand, so the slice offsets are constants.
2. **A real destination-passing producer** -- an ins/outs split in ``operandSegmentSizes`` with one
   outs entry per result, and that entry a ``tensor.empty`` of the operand's own type. An op that
   does not say where it writes is not told where to write.
3. **All uses inside this concat** -- a value read elsewhere must still exist on its own. A value
   used TWICE by the same concat (what ``cse_through_provenance`` leaves for ``cat(x, x)``) is
   retargeted at its FIRST position only; the second placement is real data movement.
4. **Program order** -- the producer must sit after the chain value it appends to, so the rewritten
   IR still dominates. An operand that cannot be ordered keeps a plain ``insert_slice``.
5. **At least one retarget** -- a concat where nothing can be retargeted is LEFT ALONE rather than
   decomposed into an equivalent pile of ``insert_slice`` ops, which would move the same bytes under
   a different op name and make the feature look like it fired.

WHY IT RUNS AFTER ``canonicalize,cse``, AND WHY IT ``implies`` ``erase_self_copy``
----------------------------------------------------------------------------------
Both orderings are MEASURED, not assumed. The rewrite runs the pipeline's own opening
``canonicalize,cse`` first (idempotent -- the pipeline runs them again immediately after). Run
BEFORE them instead, it gives two identical ``math.cos``/``math.sin`` generics different
destinations and so blocks the CSE that ``cse_through_provenance`` exists to enable: MEASURED,
``linalg.generic`` 242 -> 244 and transcendental ops 8 -> 10, i.e. 256 libm calls per inference
bought back to save 384 copied elements. Trading one shipped feature's win for another's is not a
win.

The ``insert_slice`` then bufferizes to a ``memref.copy`` whose source and destination are the SAME
subview named by two SSA values; they collapse into a literal self-copy only after the
``canonicalize,cse`` that ``erase_self_copy`` inserts post-bufferization, and only then is the copy
deleted. Counting ``memref.copy`` and the elements they move, at that point, on the RVV pipeline
with ``expand_memref_copy`` on:

    feature set                                  copies  elements  strided-dest  strided elements
    baseline                                         50    25,928            12             4,608
    this rewrite WITHOUT erase_self_copy              69    26,536            31             5,216
    this rewrite WITH erase_self_copy (shipped)      46    23,880             8             2,560
    upstream eliminate-empty-tensors, no rewrite     50    25,928            12             4,608

So without the implied feature the lever is not merely inert, it is WORSE than baseline; and the
upstream pass alone is exactly inert, which is what makes the concat rewrite the load-bearing half.
The strided-destination copies -- the ones that were ``@memrefCopy`` before ``expand_memref_copy``
-- lose 44 % of their elements.

MEASURED ON THE EMITTED CODE -- AND THIS IS WHY IT STAYS OFF
-------------------------------------------------------------
A cross-compiled K1 build (``hand_v0_int8`` + ``perop_register_block``,
``promote_buffers_to_stack``, ``expand_memref_copy``, ``cse_through_provenance``), audited on the
LINKED ELF, baseline vs the same build with this feature:

    forward instructions      36,463 -> 37,255   (+792)
    forward vector            14,094 -> 13,843   (-251)
    forward scalar compute    16,072 -> 16,772   (+700)
    forward vector coverage   0.4672 -> 0.4522
    model.o                  194,544 -> 195,952 bytes
    cosf32 / sinf32 calls         16 -> 16       (the CSE win is preserved)
    @memrefCopy call sites          0 -> 0       (already zero: expand_memref_copy)

The four copies it removes are small loops; the price is that the retargeted producers now write
into a STRIDED destination, and some of those loops stop vectorizing. On the host build a callgrind
count inside ``forward`` moves 36,843,158 -> 36,847,639 instructions (+0.012 %), i.e. no dynamic win
there either -- though the host build is scalar x86 and cannot see the RVV effect.

**THE K1 WALL IS UNMEASURED.** Both numbers above are static/host proxies, and this repo has been
burned in both directions by exactly that kind of evidence. What is certain: the output is
bit-identical, the frozen baseline is byte-identical, and the concat's remaining cost after
``expand_memref_copy`` is far smaller than the 8.41 % the pre-fix profile shows.

WHAT WOULD ACTUALLY REMOVE THE REST
------------------------------------
The 2,560 elements that survive are the duplicated rotary table's second placement and the
un-negated ``rotate_half`` half, which arrives as a ``tensor.extract_slice`` rather than from a
producer with a destination. Removing either needs the concat FOLDED INTO ITS CONSUMER -- a
half-width table addressed twice is ``d3 mod 16``, a non-projected-permutation indexing map that
would cost the consumer its vectorization. Left undone rather than done badly.

Default OFF, so with ``features == frozenset()`` the emitted IR is byte-identical to the frozen
baseline (asserted on the .ll bytes: ``e041fdb3...``, 427,790 bytes, pre- and post-change).
"""
from __future__ import annotations

#: Feature name, as it appears in a package's ``compiler_features``.
FEATURE = "concat_dps"

#: Spliced into every lowering runner (they execute in the m2m venv, which owns the MLIR Python
#: bindings). Defines ``_concat_dps(module, ctx) -> (n, report)`` and reads its argv gate.
RUNNER_PRELUDE = r'''
_CONCAT_DPS = len(sys.argv) > 8 and sys.argv[8] == '1'


def _cd_static(t):
    """The `ir.RankedTensorType` when `t` is ranked, rank >= 1 and fully static, else None.

    Fail-closed: a shape this cannot prove static has no constant slice offsets, so the concat is
    left alone rather than rewritten on an assumption."""
    from torch_mlir import ir as _cdir
    try:
        tt = _cdir.RankedTensorType(t)
    except (ValueError, TypeError):
        return None
    if tt.rank < 1:
        return None
    for d in range(tt.rank):
        if tt.is_dynamic_dim(d):
            return None
    return tt


def _cd_out_operand_index(op, value):
    """Index in ``op.operands`` of the destination-passing ``outs`` entry backing ``value``.

    Structural: a DPS op declares its operand split in ``operandSegmentSizes`` and writes result i
    through outs entry i. Anything without that split, or whose arities disagree, returns None --
    an op that does not say where it writes is not told where to write."""
    try:
        seg = op.operation.attributes["operandSegmentSizes"]
    except KeyError:
        return None
    sizes = [int(seg[i]) for i in range(len(seg))]
    if len(sizes) != 2:
        return None
    n_in, n_out = sizes
    if n_in + n_out != len(op.operands) or n_out != len(op.results):
        return None
    for i, r in enumerate(op.results):
        if r == value:
            return n_in + i
    return None


def _cd_writes_into_empty(op, idx, want_type):
    """True when ``op``'s operand ``idx`` is a `tensor.empty` of ``want_type``.

    The empty may have OTHER users: it is a value with no contents, so several ops may name it as
    their destination and retargeting one use changes nothing for the rest. Requiring a private
    empty instead would make this refuse everything on a module that has run `cse`, which merges
    identical empties."""
    dest = op.operands[idx]
    if dest.type != want_type:
        return False
    try:
        return dest.owner.operation.name == "tensor.empty"
    except AttributeError:              # block argument: not an op at all
        return False


def _cd_uses_only_in(value, op):
    """True when every use of ``value`` is an operand of ``op``."""
    for use in value.uses:
        if use.owner.operation != op.operation:
            return False
    return True


def _cd_next_concat(module, index):
    """The ``index``-th `tensor.concat` in walk order, with its block and that block's ops, or None.

    Re-walked per rewrite ON PURPOSE: erasing an operation invalidates the Python handles to the
    other operations in the context (MEASURED -- the second concat's position lookup raised "the
    operation has been invalidated"), so no handle may be carried across a rewrite."""
    found = []

    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                ops = list(block.operations)
                for inner in ops:
                    walk(inner)
                    if inner.operation.name == "tensor.concat":
                        found.append((inner, ops))

    walk(module.operation)
    return found[index] if index < len(found) else None


def _concat_dps(module, ctx):
    """Rewrite each eligible `tensor.concat` into a destination-passing insert_slice chain.

    Returns ``(n_rewritten, report)``; ``report`` is a list of ``(kind, detail)`` lines the runner
    prints, so a concat left alone is visible in the build log instead of passing for rewritten."""
    from torch_mlir import ir as _cdir
    from torch_mlir.dialects import tensor as _cdt
    from torch_mlir.passmanager import PassManager as _cdpm

    # Canonicalize and CSE FIRST -- the same two passes the pipeline itself opens with, so this is
    # idempotent, not an extra transformation. It is required: run before them, this rewrite gives
    # two identical transcendental generics different destinations and blocks the CSE that
    # `cse_through_provenance` exists to enable (MEASURED: linalg.generic 242 -> 244, transcendental
    # ops 8 -> 10 = 256 libm calls per inference bought back to save 384 copied elements).
    _cdpm.parse("builtin.module(canonicalize,cse)", ctx).run(module.operation)

    n = retargeted = 0
    report = []
    index = 0                       # how many leading concats were REFUSED (and so are still there)
    while True:
        hit = _cd_next_concat(module, index)
        if hit is None:
            break
        cat, ops = hit
        result = _cd_static(cat.results[0].type)
        shapes = [_cd_static(o.type) for o in cat.operands]
        if result is None or any(o is None for o in shapes):
            report.append(("skip_dynamic", str(cat.results[0].type)))
            index += 1
            continue
        pos = {}
        for i, o in enumerate(ops):
            pos[o.operation] = i
        if cat.operation not in pos:
            report.append(("skip_unplaced", str(cat.results[0].type)))
            index += 1
            continue
        dim = int(_cdir.IntegerAttr(cat.operation.attributes["dim"]))

        # Plan, in chain order: which operands can be produced straight into the destination.
        plan = []
        after = -1                  # program position at which the chain value is defined
        placed = []                 # operand values already placed (a repeat is a 2nd placement)
        for value, shape in zip(cat.operands, shapes):
            producer = None
            owner = value.owner
            try:
                owner_op = owner.operation
            except AttributeError:
                owner_op = None
            repeat = any(value == p for p in placed)
            if owner_op is not None and not repeat and pos.get(owner_op, -1) > after:
                oi = _cd_out_operand_index(owner, value)
                if oi is not None and _cd_uses_only_in(value, cat) \
                        and _cd_writes_into_empty(owner, oi, value.type):
                    producer = (owner, oi, pos[owner_op])
                    after = pos[owner_op]
            placed.append(value)
            plan.append((value, shape, producer))
        if not any(p is not None for _, _, p in plan):
            report.append(("skip_no_dps_producer", str(cat.results[0].type)))
            index += 1
            continue

        with ctx, _cdir.Location.unknown():
            first = min(p[2] for _, _, p in plan if p is not None)
            with _cdir.InsertionPoint(ops[first]):
                cur = _cdt.EmptyOp(list(result.shape), result.element_type).result
            offset = 0
            for value, shape, producer in plan:
                offsets = [0] * result.rank
                offsets[dim] = offset
                sizes = list(shape.shape)
                strides = [1] * result.rank
                if producer is not None:
                    owner, oi, k = producer
                    with _cdir.InsertionPoint(ops[k]):
                        sl = _cdt.ExtractSliceOp(value.type, cur, [], [], [],
                                                 offsets, sizes, strides).result
                    ip = _cdir.InsertionPoint(ops[k + 1])   # resolved BEFORE the operand is set:
                    owner.operation.operands[oi] = sl       # setting one invalidates the position
                    retargeted += 1                         # lookup for that operation
                else:
                    ip = _cdir.InsertionPoint(cat)
                with ip:
                    ins = _cdt.InsertSliceOp(value, cur, [], [], [], offsets, sizes, strides)
                for i in range(len(cat.operation.attributes)):
                    a = cat.operation.attributes[i]
                    if a.name != "dim":          # carry provenance onto the replacement ops
                        ins.operation.attributes[a.name] = a.attr
                cur = ins.result
                offset += shape.shape[dim]
        cat.results[0].replace_all_uses_with(cur)
        cat.operation.erase()
        n += 1
    report.append(("retargeted", str(retargeted)))
    return n, report
'''


def ensure_registered() -> str:
    """Register the feature (idempotent) and return its name.

    EAGER, and called from every entry point that normalizes a feature set, because ``normalize``
    REJECTS an unregistered name and ``wholemodel_proposer._composes`` swallows that KeyError and
    returns False -- an unregistered feature is not an error anyone sees, it is a feature that is
    silently never proposed.
    """
    from .impr_features import ImprFeature, _REGISTRY, register

    if FEATURE in _REGISTRY:
        return FEATURE
    from .selfcopy import FEATURE as _SELF_COPY_FEATURE

    register(ImprFeature(
        name=FEATURE,
        action_class="PASS",
        description=(
            "Decompose tensor.concat into a tensor.insert_slice chain and run "
            "eliminate-empty-tensors before bufferization, so each operand's producer writes "
            "straight into the concatenated buffer instead of into a private one that is then "
            "copied. Implies erase_self_copy, which deletes the self-copy the in-place "
            "insert_slice bufferizes to -- without it the rewrite measures WORSE than baseline."),
        implies=frozenset({_SELF_COPY_FEATURE}),
    ))
    return FEATURE
