"""Fold a loop-invariant weight `linalg.transpose` into its consumers' indexing maps.

WHY THE EXISTING FOLD MISSES IT
-------------------------------
``llvmlower/transpose_fuse.py`` (``fuse_transpose_b``) folds a transpose into the B operand of a
``linalg.matmul``. That match is the reason it captures almost nothing on a QUANTIZED model:
``passes_quant_int`` replaces every captured contraction with a ``linalg.generic`` carrying an
``i8 x i8 -> i32`` body, so the lowered int8 module has **no ``linalg.matmul`` at all** and the
existing fold fires zero times.

MEASURED on ``out/artifacts/recaptures/small_llama_int8_consistent`` (whole model, prepared with
``int8_compute=True``): 25 ``linalg.transpose``; 0 ``linalg.matmul``; 280 ``linalg.generic``. A
per-op board profile of that model (join on ``mlir_op``, not on ``prov.family`` -- the quant and
blocking rewrites drop that tag, so the tool's own family summary reports a false
``contraction = 0.0 ms``) puts ``linalg.transpose`` at **45.9 % of the attributed op time** -- the
single largest bucket, larger than every contraction combined -- of which **45.3 points** are the 15
i8 transposes whose input is a function ARGUMENT (a weight), and 0.6 points the 10 f32 4-D
attention-head permutes. An offline bundle rewrite that pre-transposes those same 15 weights
measures 1.61x faster on the K1; the compiler was capturing none of it.

WHAT THIS DOES
--------------
A ``linalg.transpose`` reading a value the function does not compute is not arithmetic: it is the
model paying, on every inference, to convert stored bytes into the layout its consumer wanted. Every
consumer that is a ``linalg`` op already states its access as an ``indexing_map``, so the conversion
can be expressed in the MAP instead of in memory::

    %T = linalg.transpose ins(%W) outs(...) permutation = P      # T[i] = W[f], f[P[t]] = i[t]
    ... linalg.generic ins(%T, ...) {indexing_maps = [M, ...]}   # reads T at e(d) = M(d)

    =>  ... linalg.generic ins(%W, ...) {indexing_maps = [M', ...]}   with  M'[P[t]] = M[t]

and the transpose dies. No data moves, no buffer is materialized, and the composed map is
value-identical by construction: the consumer reads exactly the element it read before.

On the int8 whole model each weight transpose has TWO consumers -- the (dead) dequantize generic
left behind by the integer datapath, and the i8 contraction. The contraction's B map is
``(m, n, k) -> (k, n)``; composing it with ``[1, 0]`` gives ``(m, n, k) -> (n, k)``, so after the
fold the reduction reads the weight CONTIGUOUSLY along k in the row-major ``[N, K]`` blob, where
before it strided by N through a buffer that a scalar transpose loop had just written. The fold
therefore removes a whole pass over the weight AND improves the access that remains -- the same
end state the offline pre-transposed bundle reaches, without touching the stored bytes.

WHAT IT REFUSES TO DO (fail closed)
-----------------------------------
Four structural preconditions, each checked and each counted when it fails; nothing is guessed:

1. **Loop-invariant source.** The transpose's input must be a ``func.func`` entry-block argument or
   a constant. This is a COST criterion, not a soundness one (SSA tensors are immutable, so the map
   composition is sound for any source): a transpose of a value the function computes is not a
   stored-layout conversion.
2. **Every consumer states its access as a map.** All uses of the transpose result must be operands
   of ops carrying ``indexing_maps`` with one map per operand. A use this pass cannot rewrite (a
   ``tensor.extract_slice``, a ``func.return``, a copy) means the transposed value is genuinely
   needed, so the transpose stays -- rewriting only SOME uses would leave it alive and buy nothing.
3. **Input operands only.** A DPS ``outs`` operand is WRITTEN. Permuting its map would change where
   results land, so an ``outs`` use is a refusal, not a rewrite.
4. **The vectorized axis may not get worse.** See below -- the one the board taught us.

THE HOT AXIS, AND WHY THE FIRST VERSION OF THIS PASS REGRESSED THE WALL
-----------------------------------------------------------------------
Removing the transpose is not free: the permutation has to go somewhere, and where it goes is the
consumer's access stride. The consumer wanted ``[K, N]``; after the fold the only buffer left is the
argument in ``[N, K]``, so the map MUST become ``(n, k)``. There is no third option -- keeping the
consumer reading ``(k, n)`` with no transpose op would need the bytes to actually BE in ``[K, N]``,
which means either materializing them (the transpose just deleted) or storing them that way (the
offline bundle rewrite, which changes the stored weights and is out of reach for an immutable
function argument). A map fold can only ever flip which axis is contiguous.

That flip is a win when it lands on the reduction and a loss when it lands on the vectorized axis.
MEASURED on the K1, interleaved same-session on top of the search's own winner
(``perop_register_block, promote_buffers_to_stack, expand_memref_copy, cse_through_provenance``):
3,594,824 ns without this feature, 3,994,718 ns with it -- **1.09x SLOWER**, at bit-identical output.

The mechanism, read off the IR after the transform schedule (which tiles the parallel dims 4x16 and
vectorizes the innermost OUTPUT dim, n)::

    baseline  %s = tensor.extract_slice %transposed[%k, 0] [1, 16] [1, 1]  ->  tensor<1x16xi8>
              vector.transfer_read %s : tensor<1x16xi8>, vector<1x16xi8>    # 16 CONSECUTIVE n

    folded    %s = tensor.extract_slice %arg2[0, %k] [16, 1] [1, 1]        ->  tensor<16x1xi8>
              vector.transfer_read %s : tensor<16x1xi8>, vector<16x1xi8>    # 16 n, 128 B APART

The fold turned the contraction's B read from a ROW of the vectorized axis into a COLUMN of it. No
static count could show this: op count fell, transposes fell 25 -> 10, the object SHRANK 241,872 ->
221,392 bytes, and the whole-object vector-load census got *better* (40 -> 102 ``vle8.v``, 15 -> 13
``vlse8.v``, 223 -> 96 scalar byte loads), with bit-identical output. Only the wall moved, and only
on the in-order board -- an interleaved host A/B measured 1.008x, inside noise.

So precondition 4 PRICES the axis instead of hoping. For each consumer the pass derives the loop dim
that is fastest-varying in the OUTPUT map -- the axis the vectorizer makes contiguous -- and computes
the operand's row-major stride along it before and after the fold. A fold that would INCREASE that
stride is refused, with both strides reported. Derived from the maps and the static shapes; it needs
no knowledge of the schedule, the target or the model.

WHAT THAT LEAVES
----------------
On ``small_llama_int8_consistent`` the guard refuses all 15 weight relayouts (``d1`` stride 1 -> 128,
or 1 -> 344 for the down-projection), so the feature folds **0 of 25** and the lowering is once again
byte-identical to the baseline. It is not vacuous in general: a permutation that leaves the
fastest-varying axis alone -- ``[0, 2, 1, 3]`` on a 4-D tensor, the shape of an attention head permute
-- still folds, because it moves no data onto the hot axis.

For a transposed weight feeding an n-vectorized contraction this transform is a DEAD END, and
structurally so rather than for a missing case: the only map it can produce is the one that strides
the vectorized axis. Closing that cost in the compiler needs a different lever -- a micro-kernel that
vectorizes along k and reduces horizontally (the NT-GEMM form a transpose-b BLAS kernel uses), for
which ``(n, k)`` is the RIGHT layout. That is a schedule change, not a map fold, and it is unmeasured.

Structure-keyed throughout: no op name beyond ``linalg.transpose``, no shape, no model, no target,
no provenance tag. Default OFF, so the frozen baseline (empty feature set) lowers byte-identically.

MEASURED (small_llama int8 capture, whole model):

  * unguarded, this folded 15 of 25 -- exactly the 15 the offline ``hoist_weight_transposes`` bundle
    rewrite pre-applies -- with bit-identical output and both goldens gating ok=True on
    ``tiers=['fp32', 'w8a8']``, and it still cost 1.09x on the board (numbers above).
  * guarded, it folds 0 of 25 here and the emitted object is byte-identical to the baseline.
"""
from __future__ import annotations

FEATURE = "fold_weight_transpose"

#: Spliced into every lowering runner (they execute in the m2m venv, which owns the MLIR Python
#: bindings). Defines ``_fold_weight_transposes(module, ctx) -> (folded, report)`` and reads its gate
#: from ``sys.argv[7]`` so a baseline lowering never runs it.
RUNNER_PRELUDE = r'''
def _wt_permutation(top):
    """The `permutation = array<i64: ...>` of a `linalg.transpose`, parsed structurally.

    Returns None (never a guess) when the attribute is missing or not a list of ints."""
    try:
        attr = top.attributes["permutation"]
    except (KeyError, IndexError, TypeError):
        return None
    text = str(attr)
    colon = text.find(":")
    close = text.rfind(">")
    if colon < 0 or close < colon:
        return None
    body = text[colon + 1:close].strip()
    if not body:
        return []
    perm = []
    for token in body.split(","):
        token = token.strip()
        try:
            perm.append(int(token))
        except ValueError:
            return None
    if sorted(perm) != list(range(len(perm))):
        return None                             # not a permutation -- refuse rather than reinterpret
    return perm


def _wt_op_name(op):
    """The OPERATION name of `op`, whether it arrives as an Operation or as a typed OpView.

    An OpView's own `.name` is the symbol name (`FuncOp.name` is "forward"), not "func.func", so
    reading `.name` off whatever the bindings hand back silently classifies every function argument
    as non-invariant -- measured: 15 weight transposes skipped, 0 folded."""
    if op is None:
        return None
    return getattr(getattr(op, "operation", op), "name", None)


def _wt_loop_invariant(value):
    """Is `value` stored data rather than something the function computes?

    True for a `func.func` entry-block argument (a weight/parameter the caller hands in) and for a
    materialized constant. Anything this cannot positively identify is False."""
    from torch_mlir import ir as _wtir
    try:
        arg = _wtir.BlockArgument(value)
    except (ValueError, TypeError):
        arg = None
    if arg is not None:
        try:
            parent = arg.owner.owner              # Block -> the op owning the block's region
            return _wt_op_name(parent) == "func.func"
        except (AttributeError, ValueError, TypeError):
            return False
    return _wt_op_name(getattr(value, "owner", None)) in ("arith.constant", "memref.get_global")


def _wt_dim_position(expr):
    """The loop-dim index of `expr` when it is a bare dim (`d3`), else None.

    A map result that is anything else -- a constant (`(d0, 0, d2)`), a sum, a floordiv -- is not a
    dimension this pass can price a stride along, so it reports None and the caller fails closed."""
    from torch_mlir import ir as _wtir
    try:
        return _wtir.AffineDimExpr(expr).position
    except (ValueError, TypeError):
        return None


def _wt_static_shape(value):
    """The operand's static shape, or None if any extent is dynamic/unranked (fail closed)."""
    from torch_mlir import ir as _wtir
    try:
        st = _wtir.ShapedType(value.type)
        if not st.has_static_shape:
            return None
        return list(st.shape)
    except (ValueError, TypeError):
        return None


def _wt_stride(results, shape, dim):
    """Element stride of a row-major operand of `shape`, read through `results`, along loop `dim`.

    Varying loop dim `dim` by one moves the linear offset by the row-major stride of whichever
    operand axis that dim indexes. A dim the map never mentions leaves the operand invariant -> 0.
    Returns None when the access is not a plain permutation of bare dims (fail closed)."""
    for j, e in enumerate(results):
        pos = _wt_dim_position(e)
        if pos is None:
            return None
        if pos == dim:
            stride = 1
            for extent in shape[j + 1:]:
                stride *= int(extent)
            return stride
    return 0


def _wt_hot_dim(op, maps):
    """The loop dim the vectorizer will make contiguous: the FASTEST-VARYING axis of the output.

    MEASURED, and the reason this guard exists. The frozen RVV schedule tiles the contraction's
    parallel dims and vectorizes the innermost OUTPUT dim -- on small_llama int8 the B operand is
    read as `tensor<1x16xi8>`, a row of 16 consecutive n. Folding a `[1, 0]` weight transpose into
    that map turns the same read into `tensor<16x1xi8>`: 16 elements 128 bytes apart, a strided
    read on exactly the axis being vectorized. Statically that is invisible -- same op count, same
    vector shape count, fewer instructions -- and on the K1 it cost 1.09x.

    Returns None when the output map's last result is not a bare dim, so the caller refuses."""
    n_outs = len(op.results)
    if n_outs < 1 or len(maps) <= n_outs:
        return None
    out = maps[len(maps) - n_outs].value          # first output operand's map
    results = list(out.results)
    if not results:
        return None
    return _wt_dim_position(results[-1])


def _wt_indexing_maps(op):
    """The op's `indexing_maps` array when it has one map per operand, else None."""
    try:
        maps = op.attributes["indexing_maps"]
    except (KeyError, IndexError, TypeError):
        return None
    try:
        if len(maps) != len(op.operands):
            return None                         # one map per operand, or this pass does not know
    except TypeError:
        return None
    return maps


def _fold_weight_transposes(module, ctx):
    """Fold every loop-invariant `linalg.transpose` into its consumers' indexing maps.

    Returns (folded, report) where `report` is a list of (kind, detail) lines: one `fold` per
    transpose removed and one `skip` per transpose left in place WITH the reason it was left."""
    from torch_mlir.ir import AffineMap, AffineMapAttr, ArrayAttr

    transposes = []

    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for inner in list(block.operations):
                    walk(inner)
                    if inner.operation.name == "linalg.transpose":
                        transposes.append(inner.operation)

    walk(module.operation)

    folded = 0
    report = []
    for top in transposes:
        if not top.operands or not top.results:
            report.append(("skip", "transpose with no operand/result"))
            continue
        src = top.operands[0]
        shape = str(top.results[0].type)
        if not _wt_loop_invariant(src):
            report.append(("skip", shape + ": source is computed, not stored (not loop-invariant)"))
            continue
        perm = _wt_permutation(top)
        if perm is None:
            report.append(("skip", shape + ": permutation attribute not a static permutation"))
            continue
        uses = list(top.results[0].uses)
        if not uses:
            report.append(("skip", shape + ": result is dead; left for DCE"))
            continue

        # PLAN first, rewrite only if EVERY use can be rewritten: a partial fold leaves the
        # transpose alive and buys nothing.
        plan = []
        reason = ""
        for use in uses:
            owner = getattr(use.owner, "operation", use.owner)
            idx = use.operand_number
            maps = _wt_indexing_maps(owner)
            if maps is None:
                reason = "consumer %s states no per-operand indexing_maps" % owner.name
                break
            n_outs = len(owner.results)
            if n_outs and idx >= len(maps) - n_outs:
                reason = "used as an `outs` operand of %s (written, not read)" % owner.name
                break
            m = maps[idx].value
            old_results = list(m.results)
            if len(old_results) != len(perm):
                reason = ("consumer %s map has %d results, permutation has %d"
                          % (owner.name, len(old_results), len(perm)))
                break
            hot = _wt_hot_dim(owner, maps)
            if hot is None:
                reason = ("consumer %s has no bare fastest-varying output dim, so the axis the "
                          "vectorizer makes contiguous cannot be derived" % owner.name)
                break
            t_shape = _wt_static_shape(top.results[0])
            w_shape = _wt_static_shape(src)
            if t_shape is None or w_shape is None:
                reason = "operand shape is dynamic or unranked, so the access stride is unpriceable"
                break
            new_results = list(old_results)
            for t, dst in enumerate(perm):
                new_results[dst] = old_results[t]
            before = _wt_stride(old_results, t_shape, hot)
            after = _wt_stride(new_results, w_shape, hot)
            if before is None or after is None:
                reason = ("consumer %s reads a non-permutation access this pass cannot price"
                          % owner.name)
                break
            if after > before:
                reason = ("folding would move the vectorized axis d%d of %s from stride %d to "
                          "stride %d -- the fold removes a pass over the weight but makes the "
                          "contiguous vector read a strided one, which MEASURED 1.09x SLOWER on "
                          "the K1" % (hot, owner.name, before, after))
                break
            plan.append((owner, idx))
        if reason:
            report.append(("skip", shape + ": " + reason))
            continue

        with ctx:
            for owner, idx in plan:
                maps = owner.attributes["indexing_maps"]     # re-read: an op may hold several uses
                m = maps[idx].value
                old = list(m.results)
                new = list(old)
                for t, dst in enumerate(perm):               # T[i] = W[f] with f[perm[t]] = i[t]
                    new[dst] = old[t]
                entries = [maps[j] for j in range(len(maps))]
                entries[idx] = AffineMapAttr.get(
                    AffineMap.get(m.n_dims, m.n_symbols, new))
                owner.attributes["indexing_maps"] = ArrayAttr.get(entries)
                owner.operands[idx] = src
        top.erase()
        folded += 1
        report.append(("fold", "%s into %d consumer operand(s)" % (shape, len(plan))))

    return folded, report


_FOLD_WEIGHT_TRANSPOSE = len(sys.argv) > 7 and sys.argv[7] == '1'
'''


def run_source() -> str:
    """A standalone m2m-venv script: parse argv[1], fold, print the report, write argv[2].

    This is the SAME prelude the lowering runners splice, driven directly — so a test measures the
    shipped rewrite rather than a copy of it.
    """
    return (
        "import sys\n"
        "from torch_mlir import ir\n"
        + RUNNER_PRELUDE
        + "src_path, out_path = sys.argv[1], sys.argv[2]\n"
        "ctx = ir.Context()\n"
        "with open(src_path) as f:\n"
        "    module = ir.Module.parse(f.read(), ctx)\n"
        "n, report = _fold_weight_transposes(module, ctx)\n"
        "for kind, detail in report:\n"
        "    print(kind.upper(), detail)\n"
        "print('FOLDED', n)\n"
        "with open(out_path, 'w') as f:\n"
        "    f.write(str(module.operation))\n"
    )


def _feature():
    from .impr_features import ImprFeature
    return ImprFeature(
        name=FEATURE,
        action_class="PASS",
        description=(
            "fold a loop-invariant weight `linalg.transpose` into the indexing_maps of EVERY linalg "
            "consumer that reads it, then erase it -- so the re-layout costs no op and no buffer. "
            "Generalizes `fuse_transpose_b`, which matches `linalg.matmul` and therefore fires ZERO "
            "times on a quantized model (measured: small_llama int8 has 25 linalg.transpose, 0 "
            "linalg.matmul, 280 linalg.generic). GUARDED on the vectorized axis: a map fold can only "
            "flip which axis is contiguous, and flipping the one the schedule vectorizes MEASURED "
            "1.09x SLOWER on the K1 (3,594,824 -> 3,994,718 ns interleaved, bit-identical output) -- "
            "the contraction's B read went from tensor<1x16xi8> to tensor<16x1xi8>. The pass derives "
            "each consumer's fastest-varying output dim and refuses any fold that increases the "
            "operand's stride along it, reporting both strides. With that guard it folds 0 of 25 on "
            "small_llama int8 and the object is byte-identical; a permutation leaving the hot axis "
            "alone still folds. Fails closed and counts the reason for a computed source, a consumer "
            "stating no per-operand map, an `outs` use, and an unpriceable or dynamic access. "
            "Default-off, baseline byte-identical."
        ),
    )


def ensure_registered() -> str:
    """Register the feature if it is not already. Idempotent, so importing from several entry
    points is safe. Returns the feature name."""
    from .impr_features import known, register
    if FEATURE not in known():
        register(_feature())
    return FEATURE
