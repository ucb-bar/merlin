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
Three structural preconditions, each checked and each counted when it fails; nothing is guessed:

1. **Loop-invariant source.** The transpose's input must be a ``func.func`` entry-block argument or
   a constant. This is a COST criterion, not a soundness one (SSA tensors are immutable, so the map
   composition is sound for any source): a transpose of a value the function computes is not a
   stored-layout conversion, and folding it would trade one materialization for strided reads inside
   the consumer with no evidence that wins. Measured here: the 10 activation permutes are excluded
   by this rule and would be excluded by rule 2 anyway (``tensor.extract_slice`` consumers).
2. **Every consumer states its access as a map.** All uses of the transpose result must be operands
   of ops carrying ``indexing_maps`` with one map per operand. A use this pass cannot rewrite (a
   ``tensor.extract_slice``, a ``func.return``, a copy) means the transposed value is genuinely
   needed, so the transpose stays -- rewriting only SOME uses would leave the transpose alive and
   buy nothing.
3. **Input operands only.** A DPS ``outs`` operand is WRITTEN. Permuting its map would change where
   results land, so an ``outs`` use is a refusal, not a rewrite.

Structure-keyed throughout: no op name beyond ``linalg.transpose``, no shape, no model, no target,
no provenance tag. Default OFF, so the frozen baseline (empty feature set) lowers byte-identically.

MEASURED (small_llama int8 capture, whole model, HOST lowering + execution):

  * 25 ``linalg.transpose`` before, 10 after -- the 15 weight relayouts fold, the 10 activation
    head-permutes are refused (their sources are computed, and two of the three uses of each are
    ``tensor.extract_slice``, which states no map). Those same 15 are exactly what the offline
    ``hoist_weight_transposes`` bundle rewrite pre-applies to the stored weights, reached here
    without touching a single stored byte.
  * emitted host object 241,872 -> 221,392 bytes; ``model.ll`` 452,612 -> 429,736; sha256 of the
    object changes (``b3e7ba50d4ccca67`` -> ``4f25241fb4ca263c``, first 16).
  * the model's f32 output is BIT-IDENTICAL to the baseline's, and both arms gate ``ok=True`` on
    ``tiers=['fp32', 'w8a8']`` with the same ``tier_ok``.

The BOARD runtime effect is UNMEASURED here (this session had no access to the K1). What is
established statically is that a whole pass over each weight disappears and the access that remains
is contiguous rather than N-strided; the offline pre-transposed bundle, which reaches a strictly
weaker end state (it removes the pass but leaves the contraction reading B ``(k, n)``), measures
1.61x on that board.
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
            if len(m.results) != len(perm):
                reason = ("consumer %s map has %d results, permutation has %d"
                          % (owner.name, len(m.results), len(perm)))
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
            "consumer that reads it, then erase it -- so the re-layout costs no op, no buffer and no "
            "data movement. Generalizes `fuse_transpose_b`, which matches `linalg.matmul` and "
            "therefore fires ZERO times on a quantized model: the integer datapath emits its "
            "contraction as a `linalg.generic` (measured on the small_llama int8 capture: 25 "
            "linalg.transpose, 0 linalg.matmul, 280 linalg.generic, with transpose at 45.9% of the "
            "board profile). Composing the contraction's B map (m,n,k)->(k,n) with [1,0] also turns "
            "the strided weight read into a contiguous one. Fails closed and counts the reason when "
            "the source is computed rather than stored, when any consumer does not state its access "
            "as a per-operand map, or when the value feeds an `outs` operand. Structure-keyed "
            "(loop-invariance, all-uses-foldable, static permutation). MEASURED on that capture: 25 "
            "transposes -> 10, host object 241,872 -> 221,392 bytes, output BIT-IDENTICAL and both "
            "arms gate ok on fp32+w8a8; board runtime UNMEASURED. Default-off, baseline "
            "byte-identical."
        ),
    )


def ensure_registered() -> str:
    """Register the feature if it is not already. Idempotent, so importing from several entry
    points is safe. Returns the feature name."""
    from .impr_features import known, register
    if FEATURE not in known():
        register(_feature())
    return FEATURE
