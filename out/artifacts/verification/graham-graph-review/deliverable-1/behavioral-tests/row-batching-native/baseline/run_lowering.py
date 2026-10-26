
_FG_ENABLED = True

import torch_mlir.passmanager as _fg_pm_module
from torch_mlir import ir as _fg_ir

_FG_NATIVE = _fg_pm_module.PassManager
_FG_PASS = 'func.func(linalg-fuse-elementwise-ops)'
_FG_PIN = 'merlin.fusion_pin'
_FG_RESHAPES = ('tensor.expand_shape', 'tensor.collapse_shape', 'tensor.cast')


def _fg_walk(op):
    for region in op.regions:
        for block in region.blocks:
            for inner in list(block.operations):
                yield inner.operation
                yield from _fg_walk(inner.operation)


def _fg_split(text):
    """Top-level entries of a pass-pipeline string (commas inside (), {} are not separators)."""
    out, depth, cur = [], 0, []
    for ch in text:
        if ch in '({':
            depth += 1
        elif ch in ')}':
            depth -= 1
        if ch == ',' and depth == 0:
            out.append(''.join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        out.append(''.join(cur))
    return [p.strip() for p in out if p.strip()]


def _fg_elements(value):
    t = value.type
    if not isinstance(t, _fg_ir.RankedTensorType) or not t.has_static_shape:
        return None
    n = 1
    for d in t.shape:
        n *= d
    return n


def _fg_iterations(op, maps):
    """Static iteration count of a linalg op, read off its operands' shapes through its maps."""
    n_dims = maps[0].n_dims
    ranges = [None] * n_dims
    for value, amap in zip(op.operands, maps):
        t = value.type
        if not isinstance(t, _fg_ir.RankedTensorType):
            continue
        for j, expr in enumerate(amap.results):
            if isinstance(expr, _fg_ir.AffineDimExpr) and j < t.rank and not t.is_dynamic_dim(j):
                ranges[expr.position] = t.shape[j]
    if any(r is None for r in ranges):
        return None
    n = 1
    for r in ranges:
        n *= r
    return n


def _fg_computes(producer):
    """Whether the producer's body does any work beyond yielding arguments and constants."""
    body = producer.regions[0].blocks[0]
    return any(o.operation.name not in ('linalg.yield', 'arith.constant') for o in body.operations)


def _fg_all_parallel(op):
    try:
        iters = op.attributes['iterator_types']
    except KeyError:
        return False
    return all('parallel' in str(it) for it in iters)


def _fg_refused(module):
    """The producers the control function refuses: ``[(producer value, consumer op)]``."""
    out = []
    for consumer in _fg_walk(module.operation):
        if consumer.name != 'linalg.generic':
            continue
        try:
            maps = [_fg_ir.AffineMapAttr(a).value for a in consumer.attributes['indexing_maps']]
            n_in = int(_fg_ir.DenseI32ArrayAttr(consumer.attributes['operandSegmentSizes'])[0])
        except Exception:
            continue
        if len(maps) != len(consumer.operands):
            continue
        iterations = None
        for k in range(n_in):
            value = consumer.operands[k]
            # The value the consumer reads, followed back through single-use reshapes to its producer.
            source = value
            while (isinstance(source, _fg_ir.OpResult) and source.owner.name in _FG_RESHAPES
                   and len(list(source.uses)) == 1):
                source = source.owner.operands[0]
            if not isinstance(source, _fg_ir.OpResult):
                continue
            producer = source.owner.operation
            if (producer.name != 'linalg.generic' or len(producer.results) != 1
                    or len(list(source.uses)) != 1 or not _fg_all_parallel(producer)
                    or not _fg_computes(producer)):
                continue  # not a fusion candidate, or fusing it is free
            if iterations is None:
                iterations = _fg_iterations(consumer, maps)
            elements = _fg_elements(value)
            if iterations is not None and elements is not None:
                multiplies = iterations > elements
            else:
                multiplies = not maps[k].is_permutation
            if multiplies:  # one use, so each producer is met once
                out.append((source, consumer))
    return out


def _fg_pin(ctx, module):
    refused = _fg_refused(module)
    with ctx, _fg_ir.Location.unknown():
        for value, consumer in refused:
            _fg_ir.Operation.create(_FG_PIN, results=[], operands=[value],
                                    ip=_fg_ir.InsertionPoint(consumer))
    return len(refused)


def _fg_unpin(module):
    pins = [op for op in _fg_walk(module.operation) if op.name == _FG_PIN]
    for op in pins:
        op.erase()
    return len(pins)


class _FGPassManager:
    """Stands in for torch_mlir's PassManager: runs the pipeline natively, pinning the refused
    producers around each top-level run of the elementwise fusion pass."""

    def __init__(self, text, context):
        self._text, self._ctx, self._config = text, context, []

    @classmethod
    def parse(cls, pipeline, context=None):
        return cls(pipeline, context)

    def __getattr__(self, name):
        # Configuration calls (enable_ir_printing, enable_verifier, ...) are replayed on every native
        # manager this one runs.
        def record(*args, **kwargs):
            self._config.append((name, args, kwargs))
        return record

    def _native(self, text):
        manager = _FG_NATIVE.parse(text, self._ctx) if self._ctx is not None else _FG_NATIVE.parse(text)
        for name, args, kwargs in self._config:
            getattr(manager, name)(*args, **kwargs)
        return manager

    def run(self, operation):
        text = self._text
        if not _FG_ENABLED or 'linalg-fuse-elementwise-ops' not in text:
            return self._native(text).run(operation)
        head = 'builtin.module('
        if not (text.startswith(head) and text.endswith(')')):
            raise RuntimeError('fusion guard: cannot place the fusion pass in ' + text[:200])
        passes = _fg_split(text[len(head):-1])
        stray = [p for p in passes if 'linalg-fuse-elementwise-ops' in p and p != _FG_PASS]
        if stray:
            raise RuntimeError('fusion guard: the fusion pass is nested in ' + stray[0])
        ctx = self._ctx if self._ctx is not None else operation.context
        pending = []
        for p in passes + [None]:
            if p is not None and p != _FG_PASS:
                pending.append(p)
                continue
            if pending:
                self._native(head + ','.join(pending) + ')').run(operation)
                pending = []
            if p is None:
                break
            allowed = ctx.allow_unregistered_dialects
            ctx.allow_unregistered_dialects = True
            try:
                pinned = _fg_pin(ctx, operation)
                self._native(head + p + ')').run(operation)
                unpinned = _fg_unpin(operation)
            finally:
                ctx.allow_unregistered_dialects = allowed
            if unpinned != pinned:
                raise RuntimeError('fusion guard: pinned %d producers but found %d pins after fusion'
                                   % (pinned, unpinned))
            print('OK fusion_guard', pinned, 'pinned', flush=True)


_fg_pm_module.PassManager = _FGPassManager
PassManager = _FGPassManager

import sys

from torch_mlir import ir
from torch_mlir.passmanager import PassManager
from torch_mlir.dialects import llvm

def _erase_self_copies(module):
    """Erase `memref.copy %x, %x` -- copying a buffer onto itself is a no-op. Returns the count."""
    n = 0

    def walk(op):
        nonlocal n
        for region in op.regions:
            for block in region.blocks:
                for inner in list(block.operations):
                    walk(inner)
                    if inner.operation.name == "memref.copy" and inner.operands[0] == inner.operands[1]:
                        inner.operation.erase()
                        n += 1

    walk(module.operation)
    return n


def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(),
                pre_generalize=()):
    """Run `pipeline`; when `erase` (or any `mid` rewrite is requested), split it after the
    post-bufferization canonicalize/cse and run the rewrites in between. Splitting on
    buffer-loop-hoisting (not a fixed index) so the hook stays put if the pass list moves.

    `mid` is a sequence of `(label, fn(ctx, module) -> int)` rewrites that need the SAME window as
    the erase: after bufferization has created the buffer ops, before finalize-memref-to-llvm turns
    them into opaque runtime calls. Each reports its count as `OK <label> <n>` so a rewrite that
    matched nothing is visible in the build log instead of passing for applied.

    `late` is the same shape, in a DIFFERENT window: after the forall/linalg -> `scf.parallel`
    conversions and before `convert-scf-to-openmp` turns each `scf.parallel` into a fork. It is a
    separate list rather than more `mid` entries because at the `mid` point no `scf.parallel` exists
    yet -- a grain decision made there would price loops that have not been formed. Empty `late`
    (the default) leaves the pass string split exactly as before, so the lowering is byte-identical.

    `post_openmp` runs immediately AFTER `convert-scf-to-openmp`.  It exists for structural edits
    that need to reason about `scf.parallel` before conversion and then annotate the corresponding
    `omp.parallel` afterwards.  When empty, conversion and its following passes stay in the same
    PassManager invocation as before.
    """
    from torch_mlir.passmanager import PassManager

    def _run(sub):
        if sub:
            PassManager.parse('builtin.module(' + ','.join(sub) + ')', ctx).run(module.operation)

    def _late_split(sub):
        """Run `sub`, pausing before `convert-scf-to-openmp` to run the `late` rewrites."""
        if not late and not post_openmp:
            _run(sub)
            return
        j = next((i for i, p in enumerate(sub) if 'convert-scf-to-openmp' in p), -1)
        if j < 0:
            # No OpenMP conversion in this pass list: run the rewrites at the END, where they still
            # see whatever `scf.parallel` survives, rather than dropping them silently.
            _run(sub)
            for label, fn in late:
                print('OK ' + label, fn(ctx, module))
            for label, fn in post_openmp:
                print('OK ' + label, fn(ctx, module))
            return
        _run(sub[:j])
        for label, fn in late:
            print('OK ' + label, fn(ctx, module))
        if not post_openmp:
            _run(sub[j:])
            return
        _run(sub[j:j + 1])
        for label, fn in post_openmp:
            print('OK ' + label, fn(ctx, module))
        _run(sub[j + 1:])

    passes = [p for p in pipeline.split(',') if p]
    if not passes:
        return
    if pre_generalize:
        marker = '__merlin_targeted_named_broadcast_fold__'
        mark = next((i for i, p in enumerate(passes) if p == marker), -1)
        if mark < 0:
            raise RuntimeError('pre-generalize rewrite requested but its pipeline marker is absent')
        _run(passes[:mark])
        for label, fn in pre_generalize:
            print('OK ' + label, fn(ctx, module))
        passes = passes[mark + 1:]
    elif '__merlin_targeted_named_broadcast_fold__' in passes:
        raise RuntimeError('pre-generalize pipeline marker present without a requested rewrite')
    want_split = bool(erase) or bool(mid)
    k = next((i for i, p in enumerate(passes) if 'buffer-loop-hoisting' in p), -1) if want_split else -1
    if k < 0:
        _late_split(passes)
        return
    # ...hoisting, canonicalize, cse -- but NEVER past the pass that lowers linalg to loops. A mid
    # rewrite may EMIT linalg (expand_memref_copy rewrites a copy to `linalg.copy` and relies on
    # convert-linalg-to-loops to turn it into an scf nest), and in the scalar pipeline that pass sits
    # at k+1, so a fixed k+3 window put the rewrite AFTER its own lowering: the linalg op survived to
    # LLVM conversion as an unrealized_conversion_cast and the whole build failed. Clamping keeps the
    # RVV window (where the pass is at k+7) exactly where it was.
    end = k + 3
    for i, p in enumerate(passes):
        if 'convert-linalg-to-loops' in p or 'convert-linalg-to-parallel-loops' in p:
            end = min(end, i)
            break
    head, tail = passes[:end], passes[end:]
    _run(head)
    if erase:
        print('OK erase_self_copy', _erase_self_copies(module))
    for label, fn in mid:
        print('OK ' + label, fn(ctx, module))
    _late_split(tail)


_ERASE_SELF_COPY = len(sys.argv) > 4 and sys.argv[4] == '1'

def _fuse_transpose_b(module, ctx):
    """Fold `matmul(A, transpose(B, perm))` into a transpose-b `linalg.matmul` (operand + B
    indexing_map rewrite), erasing the now-dead transpose. Returns the number of matmuls fused."""
    from torch_mlir.ir import AffineMap, AffineMapAttr, ArrayAttr

    def _walk(op, fn):
        for region in op.regions:
            for block in region.blocks:
                for inner in list(block.operations):
                    _walk(inner, fn)
                    fn(inner)

    def _perm(top):
        # linalg.transpose carries `permutation = array<i64: ...>`; parse it structurally.
        try:
            a = top.attributes["permutation"]
        except Exception:
            return None
        s = str(a)
        i = s.find(":")
        if i < 0:
            return None
        try:
            return [int(t.strip()) for t in s[i + 1:s.rfind(">")].split(",")]
        except ValueError:
            return None

    fused = 0
    dead = []
    def _fuse(o):
        nonlocal fused
        op = o.operation
        if op.name != "linalg.matmul" or len(op.operands) < 2:
            return
        b = op.operands[1]
        prod = b.owner
        if not hasattr(prod, "name") or prod.name != "linalg.transpose":
            return
        perm = _perm(prod)
        if perm is None:
            return
        try:
            maps = op.attributes["indexing_maps"]
        except KeyError:
            return                                  # need explicit maps to rewrite B's access
        m_b = maps[1].value                         # AffineMapAttr -> AffineMap
        results = list(m_b.results)
        if len(perm) != len(results):
            return                                  # perm must match B's map arity
        with ctx:
            new_b = AffineMap.get(m_b.n_dims, m_b.n_symbols,
                                  [results[perm[j]] for j in range(len(perm))])
            new_maps = ArrayAttr.get([maps[0], AffineMapAttr.get(new_b), maps[2]])
        op.attributes["indexing_maps"] = new_maps
        op.operands[1] = prod.operands[0]           # read the un-transposed source weight
        dead.append(prod)
        fused += 1

    _walk(module.operation, _fuse)
    for t in dead:
        if len(list(t.results[0].uses)) == 0:
            t.operation.erase()
    return fused


_FUSE_TRANSPOSE_B = len(sys.argv) > 5 and sys.argv[5] == '1'

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

def _bf_walk(op):
    for region in op.regions:
        for block in region.blocks:
            for inner in list(block.operations):
                yield inner.operation
                yield from _bf_walk(inner.operation)


def _bf_rank(value):
    try:
        shaped = ir.ShapedType(value.type)
        return shaped.rank if shaped.has_rank else None
    except Exception:
        return None


def _bf_maps(op):
    try:
        maps = op.attributes['indexing_maps']
        if len(maps) != len(op.operands):
            return None
        return maps
    except (KeyError, TypeError, ValueError):
        return None


def _bf_all_parallel(op):
    try:
        iters = op.attributes['iterator_types']
    except KeyError:
        return False
    return bool(iters) and all('parallel' in str(iterator) for iterator in iters)


def _fold_broadcasts(module, ctx, required_attr=None):
    """Return ``(folded, report)`` after applying only mechanically proven map folds."""
    from torch_mlir.ir import AffineMap, AffineMapAttr, ArrayAttr

    broadcasts = [op for op in _bf_walk(module.operation) if op.name == 'linalg.broadcast']
    folded = 0
    report = []
    for broadcast in broadcasts:
        if required_attr is not None and required_attr not in broadcast.attributes:
            continue
        shape = str(broadcast.results[0].type) if broadcast.results else '<no result>'
        if len(broadcast.operands) != 2 or len(broadcast.results) != 1:
            report.append(('skip', shape + ': expected one input, one init and one result'))
            continue
        src, init = broadcast.operands
        result = broadcast.results[0]
        uses = list(result.uses)
        if len(uses) != 1:
            report.append(('skip', shape + ': broadcast has %d readers, not one' % len(uses)))
            continue
        use = uses[0]
        consumer = use.owner
        operand_index = int(use.operand_number)
        if consumer.name != 'linalg.generic':
            report.append(('skip', shape + ': sole reader is %s, not linalg.generic'
                           % consumer.name))
            continue
        # This guard is the boundary from the refuted blanket fusion lever: a contraction carries a
        # reduction iterator.  Do not change contraction input maps or its scheduled access pattern.
        if not _bf_all_parallel(consumer):
            report.append(('skip', shape + ': consumer has a non-parallel iterator'))
            continue
        maps = _bf_maps(consumer)
        if maps is None:
            report.append(('skip', shape + ': consumer states no map for every operand'))
            continue
        n_outs = len(consumer.results)
        n_inputs = len(consumer.operands) - n_outs
        if n_outs < 1 or operand_index >= n_inputs:
            report.append(('skip', shape + ': broadcast is not a tensor input of its consumer'))
            continue
        try:
            dimensions = [int(dim) for dim in broadcast.attributes['dimensions']]
        except (KeyError, TypeError, ValueError):
            report.append(('skip', shape + ': dimensions are not a static integer list'))
            continue
        src_rank = _bf_rank(src)
        out_rank = _bf_rank(result)
        if src_rank is None or out_rank is None:
            report.append(('skip', shape + ': source/result is unranked'))
            continue
        if (dimensions != sorted(set(dimensions))
                or any(dim < 0 or dim >= out_rank for dim in dimensions)
                or src_rank + len(dimensions) != out_rank):
            report.append(('skip', shape + ': dimensions do not define a rank projection'))
            continue
        old_map = maps[operand_index].value
        old_results = list(old_map.results)
        if len(old_results) != out_rank:
            report.append(('skip', shape + ': consumer map/result rank disagree'))
            continue
        keep = [position for position in range(out_rank) if position not in dimensions]
        new_results = [old_results[position] for position in keep]

        # Changing the shaped operand must not change the scalar type of the corresponding generic
        # block argument.  The linalg verifier checks this too, but refusing before mutation keeps
        # the transformation atomic and makes the reason visible.
        try:
            if ir.ShapedType(src.type).element_type != ir.ShapedType(result.type).element_type:
                report.append(('skip', shape + ': source/result element types disagree'))
                continue
        except Exception:
            report.append(('skip', shape + ': source/result is not shaped'))
            continue

        with ctx:
            entries = [maps[index] for index in range(len(maps))]
            entries[operand_index] = AffineMapAttr.get(
                AffineMap.get(old_map.n_dims, old_map.n_symbols, new_results))
            consumer.attributes['indexing_maps'] = ArrayAttr.get(entries)
            consumer.operands[operand_index] = src

        # Result is now dead by construction.  Its destination is normally a tensor.empty used only
        # by the broadcast; erase it too, otherwise bufferization can still leave a pointless alloc.
        init_owner = init.owner
        broadcast.erase()
        if (init_owner is not None and init_owner.name == 'tensor.empty'
                and all(not list(value.uses) for value in init_owner.results)):
            init_owner.erase()
        folded += 1
        report.append(('fold', '%s operand %d -> rank-%d projection'
                       % (shape, operand_index, src_rank)))
    return folded, report


_FOLD_BROADCAST = len(sys.argv) > 14 and sys.argv[14] == '1'

def _fold_targeted_generalized_broadcasts(module, ctx, target_attr):
    """Fold only generic broadcasts carrying ``target_attr`` into parallel generics."""
    from torch_mlir.ir import AffineMap, AffineMapAttr, ArrayAttr

    producers = [op for op in _bf_walk(module.operation)
                 if op.name == 'linalg.generic' and target_attr in op.attributes]
    folded = 0
    report = []
    for producer in producers:
        shape = str(producer.results[0].type) if producer.results else '<no result>'
        # Stock named-op generalization turns a tensor broadcast into one source, one init, one
        # result and two maps.  The marker proves its origin, but keep the full structural check so
        # an upstream representation change fails closed.
        maps = _bf_maps(producer)
        if (len(producer.operands) != 2 or len(producer.results) != 1
                or maps is None or len(maps) != 2 or not _bf_all_parallel(producer)):
            report.append(('skip', shape + ': marked producer is not a generalized broadcast'))
            continue
        src, init = producer.operands
        result = producer.results[0]
        uses = list(result.uses)
        if len(uses) != 1:
            report.append(('skip', shape + ': generalized broadcast has %d readers' % len(uses)))
            continue
        use = uses[0]
        consumer = use.owner
        operand_index = int(use.operand_number)
        consumer_maps = _bf_maps(consumer)
        if (consumer.name != 'linalg.generic' or not _bf_all_parallel(consumer)
                or consumer_maps is None):
            report.append(('skip', shape + ': generalized consumer is not all-parallel generic'))
            continue
        n_outs = len(consumer.results)
        n_inputs = len(consumer.operands) - n_outs
        if n_outs != 1 or operand_index >= n_inputs:
            report.append(('skip', shape + ': generalized broadcast is not a consumer input'))
            continue
        source_map = maps[0].value
        output_map = maps[1].value
        consumer_map = consumer_maps[operand_index].value
        src_rank = _bf_rank(src)
        out_rank = _bf_rank(result)
        if (src_rank is None or out_rank is None
                or len(source_map.results) != src_rank
                or len(output_map.results) != out_rank
                or not source_map.is_projected_permutation
                or not output_map.is_permutation
                or len(consumer_map.results) != source_map.n_dims
                or source_map.n_symbols != 0 or consumer_map.n_symbols != 0):
            report.append(('skip', shape + ': generalized maps do not prove broadcast semantics'))
            continue
        try:
            if ir.ShapedType(src.type).element_type != ir.ShapedType(result.type).element_type:
                report.append(('skip', shape + ': source/result element types disagree'))
                continue
        except Exception:
            report.append(('skip', shape + ': source/result is not shaped'))
            continue

        # source_map maps producer points to source indices; consumer_map maps consumer points to
        # producer-result indices. Compose them to address the source directly.
        with ctx:
            composed = [expr.compose(consumer_map) for expr in source_map.results]
            entries = [consumer_maps[index] for index in range(len(consumer_maps))]
            entries[operand_index] = AffineMapAttr.get(
                AffineMap.get(consumer_map.n_dims, 0, composed))
            consumer.attributes['indexing_maps'] = ArrayAttr.get(entries)
            consumer.operands[operand_index] = src

        init_owner = init.owner
        producer.erase()
        if (init_owner is not None and init_owner.name == 'tensor.empty'
                and all(not list(value.uses) for value in init_owner.results)):
            init_owner.erase()
        folded += 1
        report.append(('fold', shape + ' operand %d -> rank-%d projection'
                       % (operand_index, src_rank)))
    return folded, report


def _targeted_named_broadcast_fold(ctx, module):
    """Generalize and fold only sole-use broadcasts feeding named add/mul operations."""
    target_attr = 'merlin.targeted_named_broadcast_fold'
    targets = []
    counts = {'linalg.add': 0, 'linalg.mul': 0}
    for broadcast in _bf_walk(module.operation):
        if broadcast.name != 'linalg.broadcast' or len(broadcast.results) != 1:
            continue
        uses = list(broadcast.results[0].uses)
        if len(uses) != 1 or uses[0].owner.name not in counts:
            continue
        # Stock generalization preserves this discardable attribute when rebuilding the producer,
        # giving the post-pass rewrite a mechanically exact provenance token.
        with ctx:
            broadcast.attributes[target_attr] = ir.UnitAttr.get()
        targets.append(broadcast)
        counts[uses[0].owner.name] += 1

    print('CENSUS targeted_named_broadcast_fold add', counts['linalg.add'],
          'mul', counts['linalg.mul'])
    if not targets:
        return 0

    # This is the same generalization the surrounding pipeline is about to run. Contractions have
    # already been scheduled and lowered before this stage; its later invocation is a no-op.
    PassManager.parse('builtin.module(func.func(linalg-generalize-named-ops))', ctx).run(
        module.operation)
    folded, report = _fold_targeted_generalized_broadcasts(module, ctx, target_attr)
    if folded != len(targets):
        details = '; '.join(kind + ': ' + detail for kind, detail in report)
        raise RuntimeError('targeted named broadcast fold planned %d but folded %d: %s'
                           % (len(targets), folded, details))
    return folded


_TARGETED_NAMED_BROADCAST_FOLD = len(sys.argv) > 15 and sys.argv[15] == '1'
_PRE_GENERALIZE_STAGES = ([('targeted_named_broadcast_fold',
                            _targeted_named_broadcast_fold)]
                          if _TARGETED_NAMED_BROADCAST_FOLD else [])

def _mc_static_memref(t):
    """The `ir.MemRefType` when `t` is a ranked memref of static shape and rank >= 1, else None.

    Fail-closed: anything this cannot prove static (unranked, a dynamic dim, rank 0) returns None
    and the copy is left for the existing lowering rather than expanded on an assumption."""
    from torch_mlir import ir as _mcir
    try:
        mt = _mcir.MemRefType(t)
    except (ValueError, TypeError):
        return None
    if mt.rank < 1:
        return None
    for d in range(mt.rank):
        if mt.is_dynamic_dim(d):
            return None
    return mt


def _expand_memref_copies(ctx, module):
    """Rewrite every static `memref.copy` to a `linalg.copy` so it lowers to an emitted loop nest.

    Returns the number expanded; prints the number SKIPPED (never zero-filled) so a copy that could
    not be proven static is surfaced instead of silently remaining a runtime call."""
    from torch_mlir import ir as _mcir
    from torch_mlir.dialects import linalg as _mclinalg

    todo = []

    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for inner in list(block.operations):
                    walk(inner)
                    if inner.operation.name == "memref.copy":
                        todo.append(inner)

    walk(module.operation)
    n = skipped = 0
    with ctx, _mcir.Location.unknown():
        for cp in todo:
            src, dst = cp.operands[0], cp.operands[1]
            s_t, d_t = _mc_static_memref(src.type), _mc_static_memref(dst.type)
            if s_t is None or d_t is None or list(s_t.shape) != list(d_t.shape):
                skipped += 1
                continue
            with _mcir.InsertionPoint(cp):
                op = _mclinalg.CopyOp([], [src], [dst])
                body = op.regions[0].blocks.append(s_t.element_type, d_t.element_type)
                with _mcir.InsertionPoint(body):
                    _mclinalg.YieldOp([body.arguments[0]])
            cp.operation.erase()
            n += 1
    print("OK expand_memref_copy skipped", skipped)
    return n

def _fold_uniform_fill_copies(ctx, module):
    from torch_mlir import ir

    allocations = []
    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for child in list(block.operations):
                    walk(child.operation)
                    if child.operation.name == "memref.alloc":
                        allocations.append(child.operation)
    walk(module.operation)
    plans = []
    for alloc in allocations:
        if len(alloc.results) != 1 or len(alloc.operands):
            continue
        source = alloc.results[0]
        try:
            ty = ir.MemRefType(source.type)
        except (ValueError, TypeError):
            continue
        if ty.rank < 1 or any(n <= 0 for n in ty.shape):
            continue
        block = alloc.block
        positions = {op.operation: i for i, op in enumerate(block.operations)}
        users = list(source.uses)
        writers, copies, frees = [], [], []
        valid = True
        for use in users:
            op = use.owner.operation
            if op.block != block:
                valid = False
                break
            if op.name == "linalg.generic" and len(op.operands) == 1 and use.operand_number == 0:
                writers.append(op)
            elif op.name == "memref.copy" and use.operand_number == 0:
                copies.append(op)
            elif op.name == "memref.dealloc" and use.operand_number == 0:
                frees.append(op)
            else:
                valid = False
                break
        if not valid or len(writers) != 1 or not copies or len(frees) > 1:
            continue
        writer = writers[0]
        if len(writer.results) or len(writer.regions) != 1 or len(writer.regions[0].blocks) != 1:
            continue
        body = writer.regions[0].blocks[0]
        ops = list(body.operations)
        if len(body.arguments) != 1 or list(body.arguments[0].uses) or len(ops) != 1:
            continue
        term = ops[0].operation
        if term.name != "linalg.yield" or len(term.operands) != 1:
            continue
        maps = ir.ArrayAttr(writer.attributes["indexing_maps"])
        kinds = ir.ArrayAttr(writer.attributes["iterator_types"])
        if len(maps) != 1 or ir.AffineMapAttr(maps[0]).value != ir.AffineMap.get_identity(ty.rank, context=ctx):
            continue
        if len(kinds) != ty.rank or any(str(kind) != '#linalg.iterator_type<parallel>' for kind in kinds):
            continue
        if any(positions[copy] <= positions[writer] for copy in copies):
            continue
        if frees and any(positions[copy] >= positions[frees[0]] for copy in copies):
            continue
        for copy in copies:
            try:
                dest = ir.MemRefType(copy.operands[1].type)
            except (ValueError, TypeError):
                valid = False
                break
            if list(dest.shape) != list(ty.shape) or dest.element_type != ty.element_type or dest.memory_space != ty.memory_space:
                valid = False
                break
        if valid:
            plans.append((alloc, writer, copies, frees))
    count = 0
    with ctx:
        for alloc, writer, copies, frees in plans:
            for copy in copies:
                with ir.InsertionPoint(copy):
                    cloned = writer.clone()
                cloned.operands[0] = copy.operands[1]
                copy.erase()
                count += 1
            writer.erase()
            for free in frees:
                free.erase()
            alloc.erase()
    return count

def _copy_allocation_root(value):
    seen = set()
    views = {"memref.subview", "memref.cast", "memref.collapse_shape",
             "memref.expand_shape", "memref.reinterpret_cast", "memref.view"}
    while value not in seen:
        seen.add(value)
        owner = value.owner
        if not hasattr(owner, "operation"):
            return None
        op = owner.operation
        if op.name == "memref.alloc":
            return op
        if op.name not in views or not op.operands:
            return None
        value = op.operands[0]
    return None


def _specialize_contiguous_copies(ctx, module):
    from torch_mlir import ir
    from torch_mlir.dialects import arith, memref, scf
    plans = []
    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for child in list(block.operations):
                    walk(child.operation)
                    cp = child.operation
                    if cp.name != "memref.copy":
                        continue
                    src, dst = cp.operands
                    roots = (_copy_allocation_root(src), _copy_allocation_root(dst))
                    if roots[0] is None or roots[1] is None or roots[0] == roots[1]:
                        continue
                    try:
                        st, dt = ir.MemRefType(src.type), ir.MemRefType(dst.type)
                        ss, _ = st.get_strides_and_offset()
                        ds, _ = dt.get_strides_and_offset()
                    except (ValueError, TypeError):
                        continue
                    shape = list(st.shape)
                    if (len(shape) < 2 or shape != list(dt.shape)
                        or st.element_type != dt.element_type or st.memory_space != dt.memory_space
                        or any(d <= 0 for d in shape) or any(s <= 0 for s in [*ss, *ds])):
                        continue
                    axis, span = len(shape), 1
                    for i in reversed(range(len(shape))):
                        if shape[i] != 1 and (ss[i] != span or ds[i] != span):
                            break
                        span *= shape[i]
                        axis = i
                    if axis == 0 or axis == len(shape) or span <= 1 or span >= (1 << 63):
                        continue
                    total = span
                    for d in shape[:axis]:
                        total *= d
                    if total >= (1 << 63):
                        continue
                    plans.append((cp, st, dt, shape, list(ss), list(ds), axis))
    walk(module.operation)
    with ctx, ir.Location.unknown():
        for cp, st, dt, shape, ss, ds, axis in plans:
            source, dest = cp.operands
            rank = len(shape)
            dynamic = ir.ShapedType.get_dynamic_size()
            index = ir.IndexType.get()
            with ir.InsertionPoint(cp):
                zero = arith.ConstantOp(index, 0).result
                one = arith.ConstantOp(index, 1).result
                bounds = [arith.ConstantOp(index, n).result for n in shape[:axis]]
                def emit(level, offsets):
                    if level < axis:
                        if shape[level] == 1:
                            emit(level + 1, [*offsets, zero])
                            return
                        loop = scf.ForOp(zero, bounds[level], one)
                        with ir.InsertionPoint(loop.body):
                            emit(level + 1, [*offsets, loop.induction_variable])
                            scf.YieldOp([])
                        return
                    static_offsets = [dynamic] * axis + [0] * (rank - axis)
                    static_sizes = [1] * axis + shape[axis:]
                    rows = []
                    for value, ty, strides in [(source, st, ss), (dest, dt, ds)]:
                        layout = ir.StridedLayoutAttr.get(dynamic, strides[axis:])
                        rowty = ir.MemRefType.get(shape[axis:], ty.element_type, layout=layout, memory_space=ty.memory_space)
                        row = memref.SubViewOp(rowty, value, offsets, [], [], static_offsets, static_sizes, [1] * rank)
                        rows.append(row.result)
                    memref.CopyOp(rows[0], rows[1])
                emit(0, [])
            cp.erase()
    return len(plans)

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

_PG_TERMINATORS = frozenset((
    "scf.yield", "scf.reduce", "scf.reduce.return", "scf.condition",
    "memref.alloca_scope.return", "omp.yield", "omp.terminator", "func.return", "cf.br"))


def _pg_const_index(value):
    """The integer of an index-typed `arith.constant` defining `value`, else None (fail closed)."""
    from torch_mlir import ir as _pgir
    try:
        owner = value.owner.operation
    except AttributeError:                      # a block argument has no defining operation
        return None
    if owner.name != "arith.constant":
        return None
    try:
        return int(_pgir.IntegerAttr(owner.attributes["value"]))
    except Exception:                           # noqa: BLE001 - not an integer attribute
        return None


def _pg_trip(lb, ub, step):
    """ceil((ub - lb) / step) when all three are index constants and step > 0, else None."""
    l, u, s = _pg_const_index(lb), _pg_const_index(ub), _pg_const_index(step)
    if l is None or u is None or s is None or s <= 0:
        return None
    return max(0, -(-(u - l) // s))


def _pg_lanes(op):
    """Lane count of the widest vector result of `op` (1 for a scalar op).

    A vector operation retires one instruction but does that many lanes of work; pricing it as 1
    would make a vectorized region look 16x cheaper than the scalar spelling of the same work."""
    from torch_mlir import ir as _pgir
    n = 1
    for res in op.results:
        try:
            vt = _pgir.VectorType(res.type)
        except Exception:                       # noqa: BLE001 - not a vector type
            continue
        c = 1
        for d in vt.shape:
            c *= int(d)
        n = max(n, c)
    return n


def _pg_body_cost(block):
    """Static lane-operation cost of `block`, or None when something in it cannot be priced."""
    total = 0
    for handle in block.operations:
        op = handle.operation
        name = op.name
        if name in _PG_TERMINATORS:
            continue
        if name == "scf.parallel":
            return None                          # a nested parallel: leave the outer one alone
        if name == "scf.for":
            trip = _pg_trip(op.operands[0], op.operands[1], op.operands[2])
            if trip is None:
                return None
            inner = _pg_body_cost(op.regions[0].blocks[0])
            if inner is None:
                return None
            total += trip * inner
            continue
        if len(op.regions):
            sub = 0
            for region in op.regions:
                for blk in region.blocks:
                    cost = _pg_body_cost(blk)
                    if cost is None:
                        return None
                    # `scf.if` takes ONE of its regions; everything else (alloca_scope, ...) all.
                    sub = max(sub, cost) if name == "scf.if" else sub + cost
            total += sub + 1
            continue
        total += _pg_lanes(op)
    return total


def _pg_cost(op):
    """Static cost of one `scf.parallel`: parallel trip product * body cost. None => unpriceable."""
    block = op.regions[0].blocks[0]
    rank = len(block.arguments)
    operands = list(op.operands)
    if rank == 0 or len(operands) < 3 * rank:
        return None
    trip = 1
    for i in range(rank):
        t = _pg_trip(operands[i], operands[rank + i], operands[2 * rank + i])
        if t is None:
            return None
        trip *= t
    body = _pg_body_cost(block)
    if body is None:
        return None
    return trip * body


def _pg_serialize(op, ctx):
    """Rewrite one reduction-free `scf.parallel` into an equivalent serial `scf.for` nest."""
    from torch_mlir import ir as _pgir
    from torch_mlir.dialects import scf as _pgscf
    block = op.regions[0].blocks[0]
    rank = len(block.arguments)
    operands = list(op.operands)
    with ctx, _pgir.Location.unknown():
        point = _pgir.InsertionPoint(op)
        ivs = []
        terminator = None
        for i in range(rank):
            with point:
                loop = _pgscf.ForOp(operands[i], operands[rank + i], operands[2 * rank + i])
            ivs.append(loop.induction_variable)
            with _pgir.InsertionPoint(loop.regions[0].blocks[0]):
                terminator = _pgscf.YieldOp([])
            point = _pgir.InsertionPoint(terminator)
    for i in range(rank):
        block.arguments[i].replace_all_uses_with(ivs[i])
    # Everything but the `scf.reduce` terminator moves, in order, into the innermost body.
    for handle in [o.operation for o in block.operations][:-1]:
        handle.move_before(terminator.operation)
    op.erase()


def _parallel_grain(ctx, module):
    """Serialize every `scf.parallel` cheaper than `_PARALLEL_GRAIN` lane-operations.

    Returns the number serialized; prints the kept / unpriceable / reduction-carrying counts so a
    threshold that matched nothing is visible instead of passing for applied."""
    if not _PARALLEL_GRAIN:
        return 0
    found = []

    def _walk(op):
        for region in op.regions:
            for blk in region.blocks:
                for handle in blk.operations:
                    inner = handle.operation
                    if inner.name == "scf.parallel":
                        found.append(inner)
                    _walk(inner)

    _walk(module.operation)
    serialized = kept = unpriceable = reducing = 0
    for op in found:
        if len(op.results):                      # carries a reduction: refuse rather than rewrite
            reducing += 1
            continue
        cost = _pg_cost(op)
        if cost is None:
            unpriceable += 1
            continue
        if cost < _PARALLEL_GRAIN:
            _pg_serialize(op, ctx)
            serialized += 1
        else:
            kept += 1
    print("OK parallel_grain threshold", _PARALLEL_GRAIN, "regions", len(found),
          "serialized", serialized, "kept", kept, "unpriceable", unpriceable,
          "reducing", reducing)
    return serialized

_PT_PLAN = []


def _pt_walk_named(module, wanted):
    found = []

    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for handle in block.operations:
                    inner = handle.operation
                    if inner.name == wanted:
                        found.append(inner)
                    walk(inner)

    walk(module.operation)
    return found


def _pt_width(cost, work, cap):
    needed = max(1, (cost + work - 1) // work)
    width = 1
    while width < needed and width < cap:
        width = min(width * 2, cap)
    return width


def _parallel_team_plan(ctx, module):
    """Record widths in structural order and serialize regions assigned one worker."""
    global _PT_PLAN
    found = _pt_walk_named(module, "scf.parallel")
    _PT_PLAN = []
    serialized = unpriceable = 0
    histogram = {}
    for op in found:
        cost = _pg_cost(op)
        if cost is None or len(op.results):
            width = _PARALLEL_TEAM_CAP
            unpriceable += 1
        else:
            width = _pt_width(cost, _PARALLEL_TEAM_WORK, _PARALLEL_TEAM_CAP)
        if width == 1 and not len(op.results):
            _pg_serialize(op, ctx)
            serialized += 1
            continue
        _PT_PLAN.append(width)
        histogram[width] = histogram.get(width, 0) + 1
    print("OK parallel_team_plan regions", len(found), "serialized", serialized,
          "remaining", len(_PT_PLAN), "unpriceable", unpriceable,
          "widths", ",".join(str(k) + ":" + str(histogram[k]) for k in sorted(histogram)))
    return len(_PT_PLAN)


def _pt_set_num_threads(ctx, old, width):
    """Replace one result-free omp.parallel with an equivalent op carrying num_threads."""
    from torch_mlir import ir as _ptir
    if len(old.results):
        raise RuntimeError("parallel-team policy cannot replace result-carrying omp.parallel")
    block = old.regions[0].blocks[0]
    if len(block.arguments):
        raise RuntimeError("parallel-team policy found omp.parallel block arguments")
    body = [handle.operation for handle in block.operations]
    if not body or body[-1].name != "omp.terminator":
        raise RuntimeError("parallel-team policy found malformed omp.parallel region")
    with ctx, _ptir.Location.unknown(), _ptir.InsertionPoint(old):
        i32 = _ptir.IntegerType.get_signless(32)
        constant = _ptir.Operation.create(
            "arith.constant", results=[i32],
            attributes={"value": _ptir.IntegerAttr.get(i32, int(width))})
        replacement = _ptir.Operation.create(
            "omp.parallel", operands=[constant.results[0]], regions=1,
            attributes={"operandSegmentSizes":
                        _ptir.DenseI32ArrayAttr.get([0, 0, 0, 1, 0, 0])})
        target = replacement.regions[0].blocks.append()
        with _ptir.InsertionPoint(target):
            terminator = _ptir.Operation.create("omp.terminator")
    for inner in body[:-1]:
        inner.move_before(terminator)
    old.erase()


def _parallel_team_apply(ctx, module):
    found = _pt_walk_named(module, "omp.parallel")
    if len(found) != len(_PT_PLAN):
        raise RuntimeError("parallel-team pre/post OpenMP region census differs: planned "
                           + str(len(_PT_PLAN)) + ", converted " + str(len(found)))
    for op, width in zip(found, _PT_PLAN):
        _pt_set_num_threads(ctx, op, width)
    module.operation.verify()
    print("OK parallel_team_apply regions", len(found))
    return len(found)

def _pc_canonical(op):
    """Whether ``op`` is a canonical result/operand-free OpenMP team region."""
    if op.name != "omp.parallel" or len(op.results) or len(op.operands):
        return False
    if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
        return False
    block = op.regions[0].blocks[0]
    body = [handle.operation for handle in block.operations]
    return not len(block.arguments) and bool(body) and body[-1].name == "omp.terminator"


def _parallel_coarsen(ctx, module):
    """Merge maximal straight-line runs of canonical ``omp.parallel`` operations."""
    # Conversion materializes duplicate bound constants immediately before individual regions.
    # The normal pipeline removes them during later cleanup, but by then OpenMP has been lowered to
    # LLVM and its team boundaries are no longer editable.  Run that same cleanup here so only real
    # intervening computation separates runs; the pipeline's later canonicalize/CSE is idempotent.
    from torch_mlir.passmanager import PassManager as _pcPassManager
    _pcPassManager.parse("builtin.module(canonicalize,cse)", ctx).run(module.operation)
    merged = groups = original = 0

    def walk(op):
        nonlocal merged, groups, original
        for region in op.regions:
            for block in region.blocks:
                children = [handle.operation for handle in block.operations]
                # Process nested blocks before editing this block, so no recursion observes an
                # operation invalidated by erasing a later member of a run.
                for child in children:
                    walk(child)
                i = 0
                while i < len(children):
                    if not _pc_canonical(children[i]):
                        i += 1
                        continue
                    run = [children[i]]
                    j = i + 1
                    while j < len(children) and _pc_canonical(children[j]):
                        run.append(children[j])
                        j += 1
                    original += len(run)
                    if len(run) > 1:
                        groups += 1
                        target = run[0].regions[0].blocks[0]
                        terminator = [h.operation for h in target.operations][-1]
                        for later in run[1:]:
                            body = [h.operation for h in later.regions[0].blocks[0].operations]
                            for inner in body[:-1]:
                                inner.move_before(terminator)
                            later.erase()
                            merged += 1
                    i = j

    walk(module.operation)
    module.operation.verify()
    print("OK parallel_coarsen original", original, "merged", merged,
          "groups", groups, "remaining", original - merged)
    return merged

_MERLIN_PANEL_MARKER_SYMBOL = "__merlin_parallel_panel_marker"


def _pp_has_nested_parallel(op):
    """Whether ``op`` contains a parallel region below its own body."""
    def walk(inner):
        for region in inner.regions:
            for block in region.blocks:
                for handle in block.operations:
                    child = handle.operation
                    if child.name in ("scf.parallel", "scf.forall", "omp.parallel"):
                        return True
                    if walk(child):
                        return True
        return False
    return walk(op)


def _pp_marker_calls(op):
    """Marker calls below ``op``, preserved through bufferization by their side effect."""
    found = []
    def walk(inner):
        for region in inner.regions:
            for block in region.blocks:
                for handle in block.operations:
                    child = handle.operation
                    if child.name == "func.call":
                        try:
                            if str(child.attributes["callee"]) == "@" + _MERLIN_PANEL_MARKER_SYMBOL:
                                found.append(child)
                        except Exception:
                            pass
                    walk(child)
    walk(op)
    return found


def _parallelize_panel_loops(ctx, module):
    """Rewrite verified bufferized panel carriers and return the number changed."""
    from torch_mlir import ir as _ppir
    from torch_mlir.dialects import scf as _ppscf

    found = []
    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for handle in block.operations:
                    child = handle.operation
                    if child.name == "scf.for" and _pp_marker_calls(child):
                        found.append(child)
                    walk(child)
    walk(module.operation)

    rewritten = refused = nested = 0
    tensor_carrier = nonidentity_carrier = operand_arity = body_arg_arity = 0
    marker_arity = marker_not_direct = bad_yield = 0
    for op in found:
        body = op.regions[0].blocks[0]
        terms = [h.operation for h in body.operations]
        markers = _pp_marker_calls(op)
        # This is the post-bufferization proof boundary.  One-shot bufferization usually erases the
        # tensor carrier completely.  Some accumulator schedules instead leave identity-carried
        # memref/status values: each yield is the unchanged matching block argument.  Those are safe
        # to replace with their initial operands because mutations already happen through the memref.
        # A tensor carrier or any non-identity yield is still refused.
        n_iter = max(0, len(op.operands) - 3)
        _operand_arity = len(op.operands) < 3 or len(op.results) != n_iter
        _body_arg_arity = len(body.arguments) != 1 + n_iter
        _marker_arity = len(markers) != 1
        _marker_not_direct = len(markers) == 1 and markers[0] not in terms
        _bad_yield = not terms or terms[-1].name != "scf.yield"
        yielded = list(terms[-1].operands) if not _bad_yield else []
        iter_args = list(body.arguments)[1:] if not _body_arg_arity else []
        _nonidentity_carrier = (len(yielded) != n_iter or len(iter_args) != n_iter or
                                any(y != arg for y, arg in zip(yielded, iter_args)))
        carrier_types = ([str(x.type) for x in op.results]
                         + [str(x.type) for x in list(op.operands)[3:]]
                         + [str(x.type) for x in iter_args]
                         + [str(x.type) for x in yielded])
        _tensor_carrier = any(t.startswith("tensor<") for t in carrier_types)
        tensor_carrier += int(_tensor_carrier)
        nonidentity_carrier += int(_nonidentity_carrier)
        operand_arity += int(_operand_arity)
        body_arg_arity += int(_body_arg_arity)
        marker_arity += int(_marker_arity)
        marker_not_direct += int(_marker_not_direct)
        bad_yield += int(_bad_yield)
        if (_tensor_carrier or _nonidentity_carrier or _operand_arity or _body_arg_arity or
                _marker_arity or _marker_not_direct or _bad_yield):
            refused += 1
            continue
        if _pp_has_nested_parallel(op):
            refused += 1
            nested += 1
            continue

        operands = list(op.operands)
        lb, ub, step = operands[:3]
        initial = operands[3:]
        old_iv = body.arguments[0]
        with ctx, _ppir.Location.unknown():
            par = _ppscf.ParallelOp([], [lb], [ub], [step], [], ip=_ppir.InsertionPoint(op))
            par.regions[0].blocks.append(lb.type)
            pbody = par.regions[0].blocks[0]
            with _ppir.InsertionPoint(pbody):
                reduce = _ppscf.ReduceOp([], 0)
        old_iv.replace_all_uses_with(pbody.arguments[0])
        for old_arg, init in zip(list(body.arguments)[1:], initial):
            old_arg.replace_all_uses_with(init)
        for result, init in zip(list(op.results), initial):
            result.replace_all_uses_with(init)
        for inner in terms[:-1]:
            if inner == markers[0]:
                inner.erase()
                continue
            inner.move_before(reduce.operation)
        terms[-1].erase()
        op.erase()
        rewritten += 1

    print("OK panel_parallel regions", len(found), "rewritten", rewritten,
          "refused", refused, "nested_parallel", nested,
          "tensor_carrier", tensor_carrier, "nonidentity_carrier", nonidentity_carrier,
          "operand_arity", operand_arity,
          "body_arg_arity", body_arg_arity, "marker_arity", marker_arity,
          "marker_not_direct", marker_not_direct, "bad_yield", bad_yield)
    return rewritten

_EXPAND_MEMREF_COPY = len(sys.argv) > 6 and sys.argv[6] == '1'
_MID_STAGES = []
if _EXPAND_MEMREF_COPY:
    _MID_STAGES.append(("expand_memref_copy", _expand_memref_copies))

if len(sys.argv) > 19 and sys.argv[19] == "1":
    _MID_STAGES.insert(0, ("specialize_contiguous_copy", _specialize_contiguous_copies))

if len(sys.argv) > 18 and sys.argv[18] == "1":
    _MID_STAGES.insert(0, ("fold_uniform_fill_copy", _fold_uniform_fill_copies))

_PANEL_PARALLEL = len(sys.argv) > 10 and sys.argv[10] == "1"
if _PANEL_PARALLEL:
    _MID_STAGES.insert(0, ("panel_parallel", _parallelize_panel_loops))

_PARALLEL_GRAIN = int(sys.argv[9]) if len(sys.argv) > 9 else 0
_LATE_STAGES = [("parallel_grain", _parallel_grain)] if _PARALLEL_GRAIN else []

_PARALLEL_TEAM_WORK = int(sys.argv[11]) if len(sys.argv) > 11 else 0
_PARALLEL_TEAM_CAP = int(sys.argv[12]) if len(sys.argv) > 12 else 0
if bool(_PARALLEL_TEAM_WORK) != bool(_PARALLEL_TEAM_CAP):
    raise RuntimeError("parallel-team work and cap must be supplied together")
if _PARALLEL_TEAM_WORK < 0 or _PARALLEL_TEAM_CAP < 0:
    raise RuntimeError("parallel-team work and cap must be non-negative")
if _PARALLEL_TEAM_WORK:
    _LATE_STAGES = [*_LATE_STAGES, ("parallel_team_plan", _parallel_team_plan)]
    _POST_OPENMP_STAGES = [("parallel_team_apply", _parallel_team_apply)]
else:
    _POST_OPENMP_STAGES = []

_PARALLEL_COARSEN = len(sys.argv) > 13 and sys.argv[13] == "1"
if _PARALLEL_COARSEN:
    _POST_OPENMP_STAGES = [*_POST_OPENMP_STAGES,
                           ("parallel_coarsen", _parallel_coarsen)]

def _lower_structured_alloca_scopes(ctx, module):
    """Replace zero-result one-block alloca scopes by explicit LLVM stack lifetime intrinsics."""
    scopes = []

    def visit(op):
        # Post-order makes nested scopes independent: lower the inner lifetime first.
        for region in op.regions:
            for block in region.blocks:
                for child in list(block.operations):
                    visit(child)
        if op.operation.name == 'memref.alloca_scope':
            scopes.append(op.operation)

    for top in list(module.body.operations):
        visit(top)

    lowered = 0
    ptr = ir.Type.parse('!llvm.ptr', ctx)
    for scope in scopes:
        if len(list(scope.results)) != 0 or len(list(scope.regions)) != 1:
            raise RuntimeError('alloca-scope pre-CFG lowering only accepts zero-result scopes')
        blocks = list(scope.regions[0].blocks)
        if len(blocks) != 1:
            raise RuntimeError('alloca-scope reached pre-CFG lowering with ' + str(len(blocks)) +
                               ' blocks; expected exactly one structured block')
        body = blocks[0]
        operations = list(body.operations)
        if not operations or operations[-1].operation.name != 'memref.alloca_scope.return':
            raise RuntimeError('alloca-scope has no recognizable return terminator')
        terminator = operations[-1].operation
        if len(list(terminator.operands)) != 0:
            raise RuntimeError('zero-result alloca-scope return unexpectedly carries operands')

        with ir.InsertionPoint(scope):
            saved = ir.Operation.create(
                'llvm.intr.stacksave', results=[ptr], loc=scope.location).results[0]
        for child in operations[:-1]:
            child.operation.move_before(scope)
        with ir.InsertionPoint(scope):
            ir.Operation.create(
                'llvm.intr.stackrestore', operands=[saved], loc=scope.location)
        terminator.erase()
        scope.erase()
        lowered += 1
    return lowered


_LOWER_ALLOCA_SCOPES = len(sys.argv) > 16 and sys.argv[16] == '1'
if _LOWER_ALLOCA_SCOPES:
    _POST_OPENMP_STAGES = [*_POST_OPENMP_STAGES,
                           ('alloca_scope_pre_cfg', _lower_structured_alloca_scopes)]

def _lower_math_intrinsics(ctx, module, mapping, libm_nan=False):
    """Replace every op named in ``mapping`` with its LLVM intrinsic; return the exact count.

    ``libm_nan`` returns ``x + x`` for a NaN ``x`` from every rounding op, as the libm functions do
    (newlib's ``floorf``/``ceilf``/``truncf``/``roundf``/``rintf`` and their doubles), where the bare
    intrinsic returns the NaN unchanged: the two then agree on the bits of every input, NaN payloads
    and signs included. ``fabs`` is the same sign-bit clear either way and keeps the intrinsic alone."""
    from torch_mlir import ir as _ri_ir
    todo = []

    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for inner in list(block.operations):
                    if inner.operation.name in mapping:
                        todo.append(inner)
                    walk(inner.operation)

    walk(module.operation)
    with ctx, _ri_ir.Location.unknown():
        for old in todo:
            ty, x = old.results[0].type, old.operands[0]
            with _ri_ir.InsertionPoint(old):
                value = _ri_ir.Operation.create(
                    mapping[old.operation.name], results=[ty], operands=[x]).results[0]
                if libm_nan and old.operation.name != "math.absf":
                    i1 = _ri_ir.IntegerType.get_signless(1)
                    flag = (_ri_ir.VectorType.get(_ri_ir.VectorType(ty).shape, i1)
                            if str(ty).startswith("vector<") else i1)
                    unordered = _ri_ir.Operation.create(
                        "arith.cmpf", results=[flag], operands=[x, x],
                        attributes={"predicate": _ri_ir.IntegerAttr.get(
                            _ri_ir.IntegerType.get_signless(64), 14)}).results[0]  # uno: x is NaN
                    doubled = _ri_ir.Operation.create("arith.addf", results=[ty], operands=[x, x]).results[0]
                    value = _ri_ir.Operation.create(
                        "arith.select", results=[ty], operands=[unordered, doubled, value]).results[0]
            old.results[0].replace_all_uses_with(value)
            old.operation.erase()
    module.operation.verify()
    return len(todo)


def _lower_roundeven_intrinsics(ctx, module):
    """Replace every math.roundeven with llvm.intr.roundeven; return the exact count."""
    return _lower_math_intrinsics(ctx, module, {"math.roundeven": "llvm.intr.roundeven"})


_EXACT_ROUNDING = {'math.floor': 'llvm.intr.floor', 'math.ceil': 'llvm.intr.ceil', 'math.trunc': 'llvm.intr.trunc', 'math.round': 'llvm.intr.round', 'math.roundeven': 'llvm.intr.roundeven', 'math.absf': 'llvm.intr.fabs'}

#: (exponent bias, smallest normal exponent, largest finite exponent, mantissa bits) per float width.
_POW2_FORMAT = {64: (1023, -1022, 1023, 52), 32: (127, -126, 127, 23)}


def _lower_pow2_of_integer(ctx, module):
    """Replace every ``math.powf(2.0, sitofp(i))`` by ``2**e1 * 2**e2`` built from exponent bits.

    ``e1 = clamp(i, lo, hi)`` and ``e2 = clamp(i - e1, lo, hi)``: in range the second factor is 1; past
    the top the product overflows to infinity; below the normal range it is the exact subnormal, or
    rounds to zero where 2**i is below the smallest one -- the correctly rounded value of 2**i in
    every case, which is what the libm call returns for a power of two. Returns the count."""
    from torch_mlir import ir as _ri_ir
    todo = []

    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for inner in list(block.operations):
                    if inner.operation.name == "math.powf":
                        todo.append(inner)
                    walk(inner.operation)

    def width_of(t):
        text = str(t)
        return {"f64": 64, "f32": 32}.get(text)

    def int_width(t):
        text = str(t)
        return int(text[1:]) if text.startswith("i") and text[1:].isdigit() else None

    def is_two(value):
        owner = getattr(value, "owner", None)
        if owner is None or getattr(owner, "name", "") != "arith.constant":
            return False
        attrs = owner.attributes
        if "value" not in attrs:
            return False
        try:
            return _ri_ir.FloatAttr(attrs["value"]).value == 2.0
        except Exception:
            return False

    walk(module.operation)
    count = 0
    with ctx, _ri_ir.Location.unknown():
        i64 = _ri_ir.IntegerType.get_signless(64)
        for op in todo:
            width = width_of(op.results[0].type)
            base, exponent = op.operands[0], op.operands[1]
            source = getattr(exponent, "owner", None)
            if width is None or not is_two(base) or getattr(source, "name", "") != "arith.sitofp":
                continue
            integer = source.operands[0]
            bits = int_width(integer.type)
            if bits is None or bits > 64:
                continue
            bias, lo, hi, mantissa = _POW2_FORMAT[width]
            with _ri_ir.InsertionPoint(op):
                def make(name, operands, result=i64, **attrs):
                    return _ri_ir.Operation.create(name, results=[result], operands=operands,
                                                    attributes=attrs).results[0]

                def const(value):
                    return make("arith.constant", [], value=_ri_ir.IntegerAttr.get(i64, value))

                wide = integer if bits == 64 else make("arith.extsi", [integer])
                low, high = const(lo), const(hi)

                def clamp(v):
                    return make("arith.maxsi", [make("arith.minsi", [v, high]), low])

                def power(e):
                    field = make("arith.shli", [make("arith.addi", [e, const(bias)]), const(mantissa)])
                    if width == 32:
                        field = make("arith.trunci", [field], result=_ri_ir.IntegerType.get_signless(32))
                    return make("arith.bitcast", [field], result=op.results[0].type)

                e1 = clamp(wide)
                e2 = clamp(make("arith.subi", [wide, e1]))
                value = make("arith.mulf", [power(e1), power(e2)], result=op.results[0].type)
            op.results[0].replace_all_uses_with(value)
            op.operation.erase()
            count += 1
    module.operation.verify()
    return count


_BF16_PROMOTED = ('arith.addf', 'arith.subf', 'arith.mulf', 'arith.divf', 'arith.maximumf', 'arith.minimumf', 'arith.maxnumf', 'arith.minnumf', 'arith.sitofp', 'arith.uitofp', 'math.floor', 'math.ceil', 'math.trunc', 'math.round', 'math.roundeven')


def _inline_bf16_rounding(ctx, module):
    """Compute every scalar bf16 op in f32 and round to bf16 in integer arithmetic; return the count.

    The same promotion LLVM applies on a target without bf16 arithmetic, with the f32 -> bf16 rounding
    spelled as the runtime helper ``__truncsfbf2`` computes it, so the result bits are the helper's and
    no call is left in the loop."""
    from torch_mlir import ir as _ri_ir
    names = set(_BF16_PROMOTED) | {"arith.truncf", "arith.extf", "arith.fptosi", "arith.fptoui",
                                     "arith.cmpf", "arith.negf"}
    todo = []

    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for inner in list(block.operations):
                    if inner.operation.name in names:
                        todo.append(inner)
                    walk(inner.operation)

    walk(module.operation)
    count = 0
    with ctx, _ri_ir.Location.unknown():
        bf16, f32 = _ri_ir.BF16Type.get(), _ri_ir.F32Type.get()
        i16, i32 = _ri_ir.IntegerType.get_signless(16), _ri_ir.IntegerType.get_signless(32)

        def is_bf16(t):
            return str(t) == "bf16"

        def is_f32(t):
            return str(t) == "f32"

        for op in todo:
            name = op.operation.name
            operands = list(op.operands)
            result = op.results[0]
            touches = any(is_bf16(v.type) for v in operands) or is_bf16(result.type)
            if not touches:
                continue
            with _ri_ir.InsertionPoint(op):
                def make(kind, args, rtype, **attrs):
                    return _ri_ir.Operation.create(kind, results=[rtype], operands=args,
                                                    attributes=attrs).results[0]

                def const(value, itype=i32):
                    return make("arith.constant", [], itype, value=_ri_ir.IntegerAttr.get(itype, value))

                def widen(v):
                    """bf16 -> f32: the bf16 bits are the f32's top half (exact)."""
                    bits = make("arith.extui", [make("arith.bitcast", [v], i16)], i32)
                    return make("arith.bitcast", [make("arith.shli", [bits, const(16)], i32)], f32)

                def narrow(v):
                    """f32 -> bf16 exactly as the runtime's ``__truncsfbf2``."""
                    bits = make("arith.bitcast", [v], i32)
                    top = make("arith.shrui", [bits, const(16)], i32)
                    special = make("arith.cmpi", [make("arith.andi", [bits, const(0x7F800000)], i32),
                                                  const(0x7F800000)], _ri_ir.IntegerType.get_signless(1),
                                   predicate=_ri_ir.IntegerAttr.get(_ri_ir.IntegerType.get_signless(64), 0))
                    payload = make("arith.cmpi", [make("arith.andi", [bits, const(0x007FFFFF)], i32), const(0)],
                                   _ri_ir.IntegerType.get_signless(1),
                                   predicate=_ri_ir.IntegerAttr.get(_ri_ir.IntegerType.get_signless(64), 1))
                    quiet = make("arith.select", [payload, const(0x40), const(0)], i32)
                    nan = make("arith.ori", [top, quiet], i32)
                    bias = make("arith.addi", [make("arith.andi", [top, const(1)], i32), const(0x7FFF)], i32)
                    rounded = make("arith.shrui", [make("arith.addi", [bits, bias], i32), const(16)], i32)
                    chosen = make("arith.select", [special, nan, rounded], i32)
                    return make("arith.bitcast", [make("arith.trunci", [chosen], i16)], bf16)

                held = op.operation.attributes  # iterates names or NamedAttributes by binding version
                attrs = {a: held[a] for a in held} if all(isinstance(a, str) for a in held) else {
                    a.name: a.attr for a in held}
                if name == "arith.extf":
                    if not (is_bf16(operands[0].type) and is_f32(result.type)):
                        continue
                    value = widen(operands[0])
                elif name == "arith.truncf":
                    if not (is_f32(operands[0].type) and is_bf16(result.type)):
                        continue
                    value = narrow(operands[0])
                elif name == "arith.negf":
                    if not is_bf16(result.type):
                        continue
                    bits = make("arith.bitcast", [operands[0]], i16)
                    value = make("arith.bitcast", [make("arith.xori", [bits, const(-0x8000, i16)], i16)], bf16)
                elif name in ("arith.fptosi", "arith.fptoui", "arith.cmpf"):
                    if not all(is_bf16(v.type) for v in operands):
                        continue
                    value = make(name, [widen(v) for v in operands], result.type, **attrs)
                else:
                    # An integer-to-bf16 conversion converts to f32 and rounds; every other op here
                    # takes bf16 operands only.
                    converts = name in ("arith.sitofp", "arith.uitofp")
                    if not is_bf16(result.type) or not (converts or all(is_bf16(v.type) for v in operands)):
                        continue
                    args = operands if converts else [widen(v) for v in operands]
                    value = narrow(make(name, args, f32, **attrs))
            result.replace_all_uses_with(value)
            op.operation.erase()
            count += 1
    module.operation.verify()
    return count


_RI_MARKER = "__merlin_lower_roundeven_to_intrinsic__"
_RI_EXACT_MARKER = "__merlin_lower_exact_math_inline__"
_RI_ORIG_RUN_STAGES = _run_stages


def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(),
                pre_generalize=()):
    passes = [p for p in pipeline.split(',') if p]
    markers = [p for p in passes if p in (_RI_MARKER, _RI_EXACT_MARKER)]
    if not markers:
        return _RI_ORIG_RUN_STAGES(ctx, module, pipeline, erase, mid, late, post_openmp,
                                   pre_generalize)
    # Both markers sit right after linalg-to-loops: run the head, every requested rewrite, then the
    # tail. mid/pre-generalize belong to the head, while parallel grain/team/coarsening stages still
    # belong to the tail; keeping that routing explicit avoids silently dropping another requested
    # rewrite when this lever is composed with it.
    i = min(passes.index(m) for m in markers)
    _RI_ORIG_RUN_STAGES(ctx, module, ','.join(passes[:i]), erase, mid, (), (),
                        pre_generalize)
    if _RI_MARKER in markers:
        print('OK roundeven_intrinsic', _lower_roundeven_intrinsics(ctx, module))
    if _RI_EXACT_MARKER in markers:
        # bf16 first: a bf16 round-even becomes an f32 one, which the intrinsic mapping then takes.
        bf16 = _inline_bf16_rounding(ctx, module)
        print('OK exact_math_inline', _lower_math_intrinsics(ctx, module, _EXACT_ROUNDING, libm_nan=True),
              _lower_pow2_of_integer(ctx, module), bf16)
    _RI_ORIG_RUN_STAGES(ctx, module, ','.join(p for p in passes[i + 1:] if p not in markers), 0, (),
                        late, post_openmp, ())

"""The integer softmax, restructured over its captured IR: the runner half.

This file is SOURCE for the lowering runner, not a module Merlin imports: :mod:`.int_softmax_table`
reads it and splices it into every runner variant, which executes it in the compiler's Python (the
MLIR bindings live there). It is therefore self-contained -- only ``torch_mlir`` and the standard
library, imported inside the functions -- and every top-level name starts with ``_ist_`` so it cannot
collide with another rewrite spliced into the same script.

What it matches is the STRUCTURE an integer softmax is captured as, never a model or a target:

    m  = rowmax(x)                      linalg.reduce maximumf over the last dimension
    q  = fptosi(clamp(roundeven((x - m) / S), lo, hi))     S > 0, lo <= 0 <= hi, lo integral
    p  = f(q)                           any pure integer elementwise DAG of q and constants
    T  = rowsum(p)                      linalg.reduce addi over the last dimension

and, when it follows, the per-row symmetric int8 quantization of ``P = p / T`` that the next
contraction runs. Each rewrite is the same value as what it replaces, and why is stated where it
is done:

* ``p = f(q)`` becomes a read of a table holding ``f`` at every ``q`` in ``[lo, 0]``, computed here
  by evaluating the captured integer ops with the IR's own integer semantics (a domain point where
  one would be poison refuses the match). ``q <= 0`` because ``m`` is the row's maximum, so the
  clamp's upper bound never binds and is dropped; its lower bound becomes a compare-and-select,
  which agrees with the clamp on every non-NaN value and also sends NaN (a row with no finite
  maximum, where the original converts NaN to an integer, which is poison) to the bound -- so every
  table read is in bounds.
* the row sum accumulates in i32 when the table's largest magnitude times the row length cannot
  overflow it, then is widened back.
* ``P``'s per-row quantization depends on the row only through ``T``: ``P = k / T`` for ``k`` in the
  table's range, the row's largest ``P`` is ``max / T`` (the row maximum's ``q`` is 0, and the table
  is largest there -- checked), the smallest is clamped with 0 by the quantization (checked), and
  every per-element step reads ``P`` and per-row values only (checked). So the quantization runs once
  per row on the ``K`` candidate values and each element reads its int8 at ``p``.

Anything that does not match leaves the IR untouched and is reported with the reason.
"""

_IST_TOKEN = "OK int_softmax_table "
#: The largest grid a table may cover (entries). The integer softmax's grid is ~11k.
_IST_TABLE_LIMIT = 1 << 16
#: How far the forward search for the row sum may walk from ``q`` (elementwise ops).
_IST_SEARCH_LIMIT = 256
#: Integer ops the table evaluator implements, with the IR's semantics.
_IST_INT_OPS = (
    "arith.constant",
    "arith.addi",
    "arith.subi",
    "arith.muli",
    "arith.andi",
    "arith.ori",
    "arith.xori",
    "arith.shli",
    "arith.shrsi",
    "arith.shrui",
    "arith.floordivsi",
    "arith.ceildivsi",
    "arith.divsi",
    "arith.divui",
    "arith.remsi",
    "arith.remui",
    "arith.minsi",
    "arith.maxsi",
    "arith.minui",
    "arith.maxui",
    "arith.extsi",
    "arith.extui",
    "arith.trunci",
    "arith.cmpi",
    "arith.select",
)
#: Ops whose results may be erased when nothing reads them any more.
_IST_PURE = (
    "linalg.generic",
    "linalg.reduce",
    "tensor.empty",
    "tensor.splat",
    "tensor.collapse_shape",
    "tensor.expand_shape",
    "arith.constant",
)


class _IstRefusal(Exception):
    """A matched prefix whose remainder is not the integer softmax; the message says what differs."""


def _ist_ir():
    from torch_mlir import ir

    return ir


# ---- IR inspection ---------------------------------------------------------------------------


def _ist_operation(x):
    return getattr(x, "operation", x)


def _ist_producer(value):
    ir = _ist_ir()
    owner = value.owner
    if isinstance(owner, ir.Block):
        return None
    return _ist_operation(owner)


def _ist_uses(value):
    return [(_ist_operation(use.owner), int(use.operand_number)) for use in value.uses]


def _ist_users(value):
    return [op for op, _ in _ist_uses(value)]


def _ist_tensor(value):
    """``(shape, element type text)`` of a statically shaped ranked tensor, else None."""
    ir = _ist_ir()
    try:
        t = ir.RankedTensorType(value.type)
    except (TypeError, ValueError):
        return None
    if not t.has_static_shape:
        return None
    return [int(d) for d in t.shape], str(t.element_type)


def _ist_width(type_text):
    """The bit width of a signless integer type's text (``i64`` -> 64), else None."""
    if len(type_text) > 1 and type_text[0] == "i" and type_text[1:].isdecimal():
        return int(type_text[1:])
    return None


def _ist_attr_number(attr):
    ir = _ist_ir()
    for cls in (ir.IntegerAttr, ir.FloatAttr):
        try:
            return cls(attr).value
        except (TypeError, ValueError):
            pass
    try:
        dense = ir.DenseElementsAttr(attr)
    except (TypeError, ValueError):
        return None
    if not dense.is_splat:
        return None
    return _ist_attr_number(dense.get_splat_value())


def _ist_constant(value):
    """The number a scalar constant, or a splat tensor of one, holds; None for anything else."""
    op = _ist_producer(value)
    if op is None:
        return None
    if op.name == "tensor.splat" and len(op.operands) == 1:
        return _ist_constant(op.operands[0])
    if op.name != "arith.constant":
        return None
    return _ist_attr_number(op.attributes["value"])


class _IstGeneric:
    """A ``linalg.generic``'s operands, maps and body, read once."""

    def __init__(self, op):
        ir = _ist_ir()
        self.op = op
        n_out = len(op.results)
        operands = list(op.operands)
        self.inputs = operands[: len(operands) - n_out]
        self.outputs = operands[len(operands) - n_out :]
        self.maps = [ir.AffineMapAttr(a).value for a in op.attributes["indexing_maps"]]
        self.parallel = all("parallel" in str(a) for a in op.attributes["iterator_types"])
        self.block = op.regions[0].blocks[0]
        self.args = list(self.block.arguments)
        self.body = [_ist_operation(o) for o in self.block.operations]
        self.names = [o.name for o in self.body[:-1]]
        self.yielded = list(self.body[-1].operands)

    def arg_index(self, value):
        for i, arg in enumerate(self.args):
            if arg == value:
                return i
        return None

    def body_value(self, value):
        """A body operand as ``("arg", input index)`` or ``("const", number)``; None otherwise."""
        i = self.arg_index(value)
        if i is not None:
            if i >= len(self.inputs):
                return None
            c = _ist_constant(self.inputs[i])
            return ("const", c) if c is not None else ("arg", i)
        c = _ist_constant(value)
        return ("const", c) if c is not None else None

    def outputs_unread(self):
        return all(not list(self.args[len(self.inputs) + i].uses) for i in range(len(self.outputs)))


def _ist_generic(op):
    if op is None or op.name != "linalg.generic" or len(op.results) != 1:
        return None
    return _IstGeneric(op)


def _ist_identity(amap, rank):
    ir = _ist_ir()
    return amap == ir.AffineMap.get_identity(rank)


def _ist_rowbcast(amap, rank):
    """``(d0, ..., d{r-2}, 0)``: one value per row, read by every element of the row."""
    ir = _ist_ir()
    exprs = [ir.AffineDimExpr.get(i) for i in range(rank - 1)] + [ir.AffineConstantExpr.get(0)]
    return amap == ir.AffineMap.get(rank, 0, exprs)


def _ist_elementwise(g, rank):
    """All parallel, one result, every map the identity over ``rank`` dimensions."""
    return g is not None and g.parallel and all(_ist_identity(m, rank) for m in g.maps)


def _ist_single_op(g, name):
    """The generic's body is exactly one ``name`` op (plus constants) yielding its result."""
    real = [o for o in g.body[:-1] if o.name != "arith.constant"]
    if len(real) != 1 or real[0].name != name or len(g.yielded) != 1:
        return None
    if g.yielded[0] != real[0].results[0]:
        return None
    return real[0]


def _ist_reduce(op, rank, combiner):
    """``op`` is a ``linalg.reduce`` over the last of ``rank`` dims whose body is ``combiner``."""
    ir = _ist_ir()
    if op is None or op.name != "linalg.reduce" or len(op.results) != 1 or len(op.operands) != 2:
        return False
    dims = list(ir.DenseI64ArrayAttr(op.attributes["dimensions"]))
    if dims != [rank - 1]:
        return False
    body = [_ist_operation(o) for o in op.regions[0].blocks[0].operations]
    if [o.name for o in body] != [combiner, "linalg.yield"]:
        return False
    args = list(op.regions[0].blocks[0].arguments)
    operands = list(body[0].operands)
    return sorted(i for i, a in enumerate(args) if any(a == v for v in operands)) == [0, 1]


def _ist_back_through_reshapes(value, limit=8):
    for _ in range(limit):
        op = _ist_producer(value)
        if op is None or op.name not in ("tensor.collapse_shape", "tensor.expand_shape"):
            return value
        value = op.operands[0]
    return value


# ---- the table evaluator -----------------------------------------------------------------------


def _ist_signed(v, w):
    v &= (1 << w) - 1
    return v - (1 << w) if v >> (w - 1) else v


def _ist_unsigned(v, w):
    return v & ((1 << w) - 1)


def _ist_overflow_checked(op):
    try:
        flags = str(op.attributes["overflowFlags"])
    except KeyError:
        return False
    return "nsw" in flags or "nuw" in flags


def _ist_compile_slice(generics, root):
    """Compile the integer DAG ``generics`` (topological) into steps over integer slots.

    Slot 0 is the root's value. Returns ``(steps, slot of the last generic's result)``. A body op the
    evaluator does not implement, a non-integer value, or a body that reads its output refuses.
    """
    slots = {}
    steps = []
    slots[root] = 0
    count = 1

    def constant_slot(value, what):
        """A slot holding ``value``'s integer constant, read at its own width."""
        nonlocal count
        c = _ist_constant(value)
        tv = _ist_tensor(value)
        w = _ist_width(tv[1] if tv is not None else str(value.type))
        if c is None or isinstance(c, float) or w is None:
            raise _IstRefusal(what)
        steps.append(("const", count, (), 0, _ist_signed(int(c), w)))
        count += 1
        return count - 1

    def slot_of(value):
        for known, slot in slots.items():
            if known == value:
                return slot
        return constant_slot(value, "an operand of the integer DAG is neither computed in it nor an integer constant")

    last = None
    for g in generics:
        if not g.outputs_unread():
            raise _IstRefusal("an integer op reads its output buffer")
        local = []
        for i, arg in enumerate(g.args[: len(g.inputs)]):
            local.append((arg, slot_of(g.inputs[i])))
        for op in g.body[:-1]:
            if op.name not in _IST_INT_OPS:
                raise _IstRefusal(f"{op.name} in the integer DAG has no evaluator")
            ins = []
            for operand in op.operands:
                found = next((s for a, s in local if a == operand), None)
                if found is None:
                    found = constant_slot(operand, f"{op.name} reads a value the evaluator cannot resolve")
                ins.append(found)
            widths = []
            for v in [*op.operands, *op.results]:
                w = _ist_width(str(v.type))
                if w is None:
                    raise _IstRefusal(f"{op.name} on {v.type} is not a signless integer op")
                widths.append(w)
            extra = None
            if op.name == "arith.constant":
                extra = _ist_attr_number(op.attributes["value"])
                if extra is None or isinstance(extra, float):
                    raise _IstRefusal("a non-integer constant in the integer DAG")
                extra = _ist_signed(int(extra), widths[-1])
            elif op.name == "arith.cmpi":
                extra = int(_ist_ir().IntegerAttr(op.attributes["predicate"]).value)
            elif _ist_overflow_checked(op):
                extra = "checked"
            steps.append((op.name, count, tuple(ins), tuple(widths), extra))
            local.append((op.results[0], count))
            count += 1
        out = next((s for a, s in local if a == g.yielded[0]), None)
        if out is None:
            raise _IstRefusal("an integer op yields a value it does not compute")
        slots[g.op.results[0]] = out
        last = out
    return steps, last, count


def _ist_run(steps, size, q):
    v = [0] * size
    v[0] = q
    for name, out, ins, widths, extra in steps:
        if name == "const":
            v[out] = extra
            continue
        w = widths[-1]
        a = v[ins[0]] if ins else None
        b = v[ins[1]] if len(ins) > 1 else None
        if name == "arith.constant":
            r = extra
        elif name in ("arith.addi", "arith.subi", "arith.muli", "arith.shli"):
            if name == "arith.shli":
                if _ist_unsigned(b, w) >= w:
                    raise _IstRefusal("a shift amount reaches the width (poison) inside the domain")
                exact = a << _ist_unsigned(b, w)
            else:
                exact = a + b if name == "arith.addi" else a - b if name == "arith.subi" else a * b
            r = _ist_signed(exact, w)
            if extra == "checked" and r != exact:
                raise _IstRefusal(f"{name} overflows a no-wrap flag (poison) inside the domain")
        elif name in ("arith.andi", "arith.ori", "arith.xori"):
            ua, ub = _ist_unsigned(a, w), _ist_unsigned(b, w)
            r = _ist_signed(ua & ub if name == "arith.andi" else ua | ub if name == "arith.ori" else ua ^ ub, w)
        elif name in ("arith.shrsi", "arith.shrui"):
            amount = _ist_unsigned(b, w)
            if amount >= w:
                raise _IstRefusal("a shift amount reaches the width (poison) inside the domain")
            r = a >> amount if name == "arith.shrsi" else _ist_signed(_ist_unsigned(a, w) >> amount, w)
        elif name in ("arith.floordivsi", "arith.ceildivsi", "arith.divsi", "arith.remsi"):
            if b == 0 or (a == -(1 << (w - 1)) and b == -1):
                raise _IstRefusal(f"{name} divides by zero or overflows (poison) inside the domain")
            if name == "arith.floordivsi":
                r = a // b
            elif name == "arith.ceildivsi":
                r = -((-a) // b)
            else:
                t = abs(a) // abs(b)
                t = t if (a >= 0) == (b >= 0) else -t
                r = t if name == "arith.divsi" else a - b * t
            r = _ist_signed(r, w)
        elif name in ("arith.divui", "arith.remui"):
            ua, ub = _ist_unsigned(a, w), _ist_unsigned(b, w)
            if ub == 0:
                raise _IstRefusal(f"{name} divides by zero (poison) inside the domain")
            r = _ist_signed(ua // ub if name == "arith.divui" else ua % ub, w)
        elif name in ("arith.minsi", "arith.maxsi"):
            r = min(a, b) if name == "arith.minsi" else max(a, b)
        elif name in ("arith.minui", "arith.maxui"):
            pick_a = _ist_unsigned(a, w) <= _ist_unsigned(b, w)
            r = (a if pick_a else b) if name == "arith.minui" else (b if pick_a else a)
        elif name == "arith.extsi":
            r = a
        elif name == "arith.extui":
            r = _ist_signed(_ist_unsigned(a, widths[0]), w)
        elif name == "arith.trunci":
            r = _ist_signed(a, w)
        elif name == "arith.cmpi":
            wi = widths[0]
            sa, sb, ua, ub = a, b, _ist_unsigned(a, wi), _ist_unsigned(b, wi)
            r = (sa == sb, sa != sb, sa < sb, sa <= sb, sa > sb, sa >= sb, ua < ub, ua <= ub, ua > ub, ua >= ub)[extra]
            r = -1 if r else 0
        elif name == "arith.select":
            r = v[ins[1]] if a != 0 else v[ins[2]]
        else:
            raise _IstRefusal(f"{name} has no evaluator")
        v[out] = r
    return v


# ---- matching ----------------------------------------------------------------------------------


def _ist_match(sub_op):
    """Match one integer softmax from its ``x - rowmax(x)`` op; None when that prefix is not there.

    Returns the plan dict; raises :class:`_IstRefusal` for a softmax-shaped prefix whose remainder
    differs."""
    g_sub = _ist_generic(sub_op)
    if g_sub is None or len(g_sub.inputs) != 2 or not g_sub.parallel:
        return None
    sub = _ist_single_op(g_sub, "arith.subf")
    if sub is None or not g_sub.outputs_unread():
        return None
    x, mb = g_sub.inputs
    tx, tm = _ist_tensor(x), _ist_tensor(mb)
    if tx is None or tm is None:
        return None
    shape, ftype = tx
    rank = len(shape)
    if rank < 1 or tm[0] != shape[:-1] + [1]:
        return None
    if not (_ist_identity(g_sub.maps[0], rank) and _ist_rowbcast(g_sub.maps[1], rank)):
        return None
    if not (_ist_identity(g_sub.maps[2], rank) and list(sub.operands) == g_sub.args[:2]):
        return None
    red_max = _ist_producer(_ist_back_through_reshapes(mb))
    if not _ist_reduce(red_max, rank, "arith.maximumf") or red_max.operands[0] != x:
        return None
    # From here on the prefix IS ``x - rowmax(x)``: anything else differing is reported.
    plan = {"x": x, "shape": shape, "red_max": red_max}
    plan["max_attained"] = _ist_constant(red_max.operands[1]) == float("-inf")

    def single_user(value, what):
        users = _ist_users(value)
        if len(users) != 1:
            raise _IstRefusal(f"{what} has {len(users)} readers, not one")
        return users[0]

    def elementwise_one(op, name, what):
        g = _ist_generic(op)
        if not _ist_elementwise(g, rank) or not g.outputs_unread():
            raise _IstRefusal(f"{what} is not one elementwise op")
        inner = _ist_single_op(g, name)
        if inner is None:
            raise _IstRefusal(f"{what} is not {name}")
        return g, inner

    g_div, div = elementwise_one(single_user(g_sub.op.results[0], "x - max"), "arith.divf", "(x - max) / S")
    lhs, rhs = (g_div.body_value(v) for v in div.operands)
    if lhs != ("arg", 0) or rhs is None or rhs[0] != "const" or not (0 < rhs[1] < float("inf")):
        raise _IstRefusal("the exponent grid step is not a positive constant divisor")
    g_round, rnd = elementwise_one(single_user(g_div.op.results[0], "(x - max) / S"), "math.roundeven", "round")
    clamp_op = single_user(g_round.op.results[0], "the rounded grid index")
    g_clamp = _ist_generic(clamp_op)
    if not _ist_elementwise(g_clamp, rank) or len(g_clamp.inputs) != 1:
        raise _IstRefusal("the grid index clamp is not one elementwise op")
    real = [o for o in g_clamp.body[:-1] if o.name != "arith.constant"]
    if [o.name for o in real] != ["arith.maximumf", "arith.minimumf"] or g_clamp.yielded[0] != real[1].results[0]:
        raise _IstRefusal("the grid index clamp is not max-then-min")
    lo_v = [g_clamp.body_value(v) for v in real[0].operands]
    hi_v = [g_clamp.body_value(v) for v in real[1].operands]
    if lo_v[0] != ("arg", 0) or lo_v[1] is None or lo_v[1][0] != "const":
        raise _IstRefusal("the clamp's lower bound is not a constant")
    if real[1].operands[0] != real[0].results[0] or hi_v[1] is None or hi_v[1][0] != "const":
        raise _IstRefusal("the clamp's upper bound is not a constant")
    lo, hi = float(lo_v[1][1]), float(hi_v[1][1])
    if not (lo <= 0.0 <= hi) or lo != int(lo) or -lo + 1 > _IST_TABLE_LIMIT:
        raise _IstRefusal(f"the clamp [{lo}, {hi}] does not bound a table-sized grid at or below 0")
    g_cast, _ = elementwise_one(single_user(g_clamp.op.results[0], "the clamped grid index"), "arith.fptosi", "cast")
    qi = g_cast.op.results[0]
    qw = _ist_width(_ist_tensor(qi)[1])
    if qw is None or int(lo) < -(1 << (qw - 1)):
        raise _IstRefusal("the grid index type cannot hold the clamp's bound")
    plan.update(g_clamp=g_clamp, g_cast=g_cast, qi=qi, lo=int(lo), ftype=ftype)

    # The numerator: the nearest value downstream of q that a row sum reads.
    p = red_sum = None
    seen, frontier = [], [qi]
    while frontier and p is None and len(seen) < _IST_SEARCH_LIMIT:
        value = frontier.pop(0)
        for op in _ist_users(value):
            if _ist_reduce(op, rank, "arith.addi") and op.operands[0] == value:
                p, red_sum = value, op
                break
            g = _ist_generic(op)
            if _ist_elementwise(g, rank) and _ist_width(_ist_tensor(g.op.results[0])[1]) is not None:
                if all(other != g.op for other in seen):
                    seen.append(g.op)
                    frontier.append(g.op.results[0])
    if p is None:
        raise _IstRefusal("no row sum reads an integer function of the grid index")
    # Its backward slice must close on q and integer constants.
    order, visiting = [], []

    def visit(value):
        if value == qi or _ist_constant(value) is not None:
            return
        op = _ist_producer(value)
        if any(done.op == op for done in order):
            return
        if op is None or any(v == op for v in visiting):
            raise _IstRefusal("the numerator is not a function of the grid index alone")
        g = _ist_generic(op)
        if not _ist_elementwise(g, rank):
            raise _IstRefusal(f"the numerator reads {op.name}, not an elementwise integer op")
        visiting.append(op)
        for inp in g.inputs:
            visit(inp)
        visiting.pop()
        order.append(g)

    visit(p)
    if not order:
        raise _IstRefusal("the row sum reads the grid index itself")
    steps, out_slot, size = _ist_compile_slice(order, qi)
    table = [_ist_run(steps, size, q)[out_slot] for q in range(int(lo), 1)]
    plan.update(p=p, red_sum=red_sum, slice=order, table=table)
    return plan


def _ist_match_quantization(plan):
    """The per-row int8 quantization of ``p / T`` that follows; returns its plan or raises."""
    shape, p, red_sum, table = plan["shape"], plan["p"], plan["red_sum"], plan["table"]
    rank, n = len(shape), shape[-1]
    if not plan["max_attained"]:
        raise _IstRefusal("the row maximum's initial value is not -inf, so it need not be an element")
    if min(table) < 0 or max(table) <= 0 or table[-1] != max(table):
        raise _IstRefusal("the numerator table is not non-negative with its maximum at q = 0")
    # T as a float, one per row.
    tf = None
    for op in _ist_users(red_sum.results[0]):
        value = op.results[0] if op.name in ("tensor.collapse_shape", "tensor.expand_shape") else None
        while value is not None and _ist_tensor(value)[0] != shape[:-1] + [1]:
            nxt = [u for u in _ist_users(value) if u.name in ("tensor.collapse_shape", "tensor.expand_shape")]
            value = nxt[0].results[0] if len(nxt) == 1 else None
        if value is None:
            continue
        for user in _ist_users(value):
            g = _ist_generic(user)
            if _ist_elementwise(g, rank) and _ist_single_op(g, "arith.sitofp") is not None:
                tf = g.op.results[0]
    if tf is None:
        raise _IstRefusal("the row sum is not converted to a per-row float")
    probs = g_pf = g_pr = None
    for op in _ist_users(p):
        g = _ist_generic(op)
        if not _ist_elementwise(g, rank) or _ist_single_op(g, "arith.sitofp") is None:
            continue
        for user in _ist_users(g.op.results[0]):
            gd = _ist_generic(user)
            if gd is None or len(gd.inputs) != 2 or not gd.parallel or not gd.outputs_unread():
                continue
            div = _ist_single_op(gd, "arith.divf")
            if div is None or gd.inputs[0] != g.op.results[0] or gd.inputs[1] != tf:
                continue
            if not (_ist_identity(gd.maps[0], rank) and _ist_rowbcast(gd.maps[1], rank)):
                continue
            if _ist_identity(gd.maps[2], rank) and list(div.operands) == gd.args[:2]:
                probs, g_pf, g_pr = gd.op.results[0], g, gd
    if probs is None:
        raise _IstRefusal("the probabilities are not p / T")
    # Monotone value steps and row-preserving reshapes to the quantized operand.
    chain, value = [], probs
    while True:
        users = _ist_users(value)
        if len(users) != 1:
            break
        op = users[0]
        if op.name in ("tensor.collapse_shape", "tensor.expand_shape"):
            nxt = op.results[0]
        else:
            g = _ist_generic(op)
            vrank = len(_ist_tensor(value)[0])
            if not _ist_elementwise(g, vrank) or len(g.inputs) != 1 or not g.outputs_unread():
                break
            real = [o for o in g.body[:-1]]
            if real and not (len(real) == 1 and real[0].name in ("arith.truncf", "arith.extf")):
                break
            if not real and g.yielded[0] != g.args[0]:
                break
            nxt = g.op.results[0]
        tv = _ist_tensor(nxt)
        if tv is None or tv[0][-1] % n:
            raise _IstRefusal("a reshape of the probabilities moves the key dimension")
        chain.append(op)
        value = nxt
    xq = value
    sq = _ist_tensor(xq)[0]
    if sq[-1] != n:
        raise _IstRefusal("the quantized operand's last dimension is not the softmax row")
    rq = len(sq)
    red_min = red_max = head = None
    for op in _ist_users(xq):
        if _ist_reduce(op, rq, "arith.minimumf") and op.operands[0] == xq and red_min is None:
            red_min = op
        elif _ist_reduce(op, rq, "arith.maximumf") and op.operands[0] == xq and red_max is None:
            red_max = op
        elif head is None and _ist_generic(op) is not None:
            head = op
        else:
            raise _IstRefusal(f"the quantized operand is also read by {op.name}")
    if red_min is None or red_max is None or head is None:
        raise _IstRefusal("the probabilities are not quantized per row (min, max and a per-element step)")
    for op in _ist_users(red_min.results[0]):
        g = _ist_generic(op)
        mn = _ist_single_op(g, "arith.minimumf") if g is not None and g.parallel else None
        kinds = [g.body_value(v) for v in mn.operands] if mn is not None else []
        consts = [k[1] for k in kinds if k is not None and k[0] == "const"]
        if mn is None or len(consts) != 1 or not (consts[0] <= 0.0) or not g.outputs_unread():
            raise _IstRefusal("the row minimum is read other than as min(row minimum, c <= 0)")
    # The per-element steps: each reads the previous value (identity) and per-row values or constants.
    steps, value = [], xq
    op = head
    while True:
        g = _ist_generic(op)
        if g is None or not g.parallel or not g.outputs_unread() or not _ist_identity(g.maps[-1], rq):
            break
        ok, saw = True, False
        for inp, amap in zip(g.inputs, g.maps):
            tv = _ist_tensor(inp)
            if inp == value and _ist_identity(amap, rq):
                saw = True
            elif tv is not None and tv[0] == sq[:-1] + [1] and _ist_rowbcast(amap, rq):
                pass
            elif _ist_constant(inp) is not None and _ist_identity(amap, rq):
                pass
            else:
                ok = False
        if (
            not ok
            or not saw
            or any(not (o.name.startswith("arith.") or o.name.startswith("math.")) for o in g.body[:-1])
        ):
            break
        steps.append(g)
        value = g.op.results[0]
        users = _ist_users(value)
        if len(users) != 1:
            break
        op = users[0]
    if not steps or steps[0].op != head:
        raise _IstRefusal("the per-element quantization step does not read the operand and per-row values only")
    for g in steps[:-1]:
        if len(_ist_users(g.op.results[0])) != 1:
            raise _IstRefusal("an intermediate quantization value has other readers")
    return {
        "tf": tf,
        "g_pf": g_pf,
        "g_pr": g_pr,
        "chain": chain,
        "xq": xq,
        "sq": sq,
        "red_min": red_min,
        "red_max": red_max,
        "steps": steps,
        "k": max(table) + 1,
    }


# ---- rewriting ---------------------------------------------------------------------------------


def _ist_splice(ctx, anchor, args, body, result_types, *, after=False):
    """Parse ``body`` (MLIR, reading ``%a0..``) as a function, move its ops before ``anchor`` (after it,
    with ``after``) with its arguments bound to ``args``, and return the values it returns."""
    ir = _ist_ir()
    sig = ", ".join(f"%a{i}: {v.type}" for i, v in enumerate(args))
    rets = ", ".join(str(t) for t in result_types)
    text = "module {\nfunc.func @__ist_splice(" + sig + ") -> (" + rets + ") {\n" + body + "\n}\n}\n"
    tmp = ir.Module.parse(text, ctx)
    func = _ist_operation(tmp.body.operations[0])
    block = func.regions[0].blocks[0]
    ops = [_ist_operation(o) for o in block.operations]
    results = list(ops[-1].operands)
    ops[-1].erase()
    if after:
        for op in reversed(ops[:-1]):
            op.move_after(anchor)
    else:
        for op in ops[:-1]:
            op.move_before(anchor)
    for arg, value in zip(block.arguments, args):
        arg.replace_all_uses_with(value)
    return results


def _ist_tensor_type(shape, elem):
    return "tensor<" + "x".join(str(d) for d in shape) + ("x" if shape else "") + elem + ">"


def _ist_maps(rank, count):
    dims = ", ".join(f"d{i}" for i in range(rank))
    ident = f"affine_map<({dims}) -> ({dims})>"
    return "[" + ", ".join([ident] * count) + "]", "[" + ", ".join(['"parallel"'] * rank) + "]"


def _ist_head(maps, iters, ins, outs):
    """``linalg.generic {...} ins(...) outs(...) {`` for ``ins``/``outs`` lists of ``(name, type)``."""
    text = f"linalg.generic {{indexing_maps = {maps}, iterator_types = {iters}}}"
    for word, operands in (("ins", ins), ("outs", outs)):
        if operands:
            text += f" {word}(" + ", ".join(n for n, _ in operands) + " : " + ", ".join(t for _, t in operands) + ")"
    return text + " {"


def _ist_carry(src, dst):
    """The replaced op's provenance on its replacement, marked as this rewrite's."""
    ir = _ist_ir()
    for key in list(src.attributes):
        name = getattr(key, "name", key)
        if name.startswith("prov."):
            dst.attributes[name] = src.attributes[name]
    dst.attributes["prov.rewrite"] = ir.StringAttr.get("int_softmax_table")


def _ist_narrow_type(values):
    lo, hi = min(values), max(values)
    for w in (8, 16, 32, 64):
        if -(1 << (w - 1)) <= lo and hi < (1 << (w - 1)):
            return w
    raise _IstRefusal("the table's values do not fit 64 bits")


def _ist_float_text(v):
    return repr(float(v)) if v == v and v not in (float("inf"), float("-inf")) else None


def _ist_rewrite(ctx, plan, quant):
    ir = _ist_ir()
    shape, rank = plan["shape"], len(plan["shape"])
    lo, table = plan["lo"], plan["table"]
    g_clamp, g_cast, qi, p, red_sum = plan["g_clamp"], plan["g_cast"], plan["qi"], plan["p"], plan["red_sum"]
    ptype = _ist_tensor(p)[1]
    pw = _ist_width(ptype)
    qtype = _ist_tensor(qi)[1]
    n = shape[-1]
    init = _ist_constant(red_sum.operands[1])
    bound = max(abs(min(table)), abs(max(table))) * n + abs(int(init or 0))
    narrow = init is not None and not isinstance(init, float) and bound < (1 << 31) and pw >= 32
    gw = 32 if narrow else pw
    gtype = f"i{gw}"
    kw = _ist_narrow_type(table)
    maps1, iters = _ist_maps(rank, 2)
    report = {"table_entries": len(table), "table_bits": kw, "int32_sum": narrow, "row_quantization": False}

    # 1. the clamp as compare-and-select; the cast reads it.
    lo_text = _ist_float_text(lo)
    ft = plan["ftype"]
    (sel,) = _ist_splice(
        ctx,
        g_cast.op,
        [g_clamp.inputs[0]],
        f"""  %lo = arith.constant {lo_text} : {ft}
  %e = tensor.empty() : {_ist_tensor_type(shape, ft)}
  %r = {_ist_head(maps1, iters, [("%a0", _ist_tensor_type(shape, ft))], [("%e", _ist_tensor_type(shape, ft))])}
  ^bb0(%v: {ft}, %o: {ft}):
    %c = arith.cmpf oge, %v, %lo : {ft}
    %s = arith.select %c, %v, %lo : {ft}
    linalg.yield %s : {ft}
  }} -> {_ist_tensor_type(shape, ft)}
  return %r : {_ist_tensor_type(shape, ft)}""",
        [g_clamp.op.results[0].type],
    )
    _ist_carry(g_clamp.op, _ist_producer(sel))
    g_cast.op.operands[0] = sel

    # 2. the table read: tab[q - lo], at the gather's width.
    values = ", ".join(str(v) for v in table)
    widen = f"%w = arith.extsi %t : i{kw} to {gtype}" if kw < gw else ""
    out = "%w" if kw < gw else "%t"
    (pg,) = _ist_splice(
        ctx,
        g_cast.op,
        [qi],
        f"""  %tab = arith.constant dense<[{values}]> : tensor<{len(table)}xi{kw}>
  %lo = arith.constant {lo} : {qtype}
  %e = tensor.empty() : {_ist_tensor_type(shape, gtype)}
  %r = {_ist_head(maps1, iters, [("%a0", _ist_tensor_type(shape, qtype))], [("%e", _ist_tensor_type(shape, gtype))])}
  ^bb0(%q: {qtype}, %o: {gtype}):
    %d = arith.subi %q, %lo : {qtype}
    %i = arith.index_cast %d : {qtype} to index
    %t = tensor.extract %tab[%i] : tensor<{len(table)}xi{kw}>
    {widen}
    linalg.yield {out} : {gtype}
  }} -> {_ist_tensor_type(shape, gtype)}
  return %r : {_ist_tensor_type(shape, gtype)}""",
        [ir.RankedTensorType.get(shape, ir.IntegerType.get_signless(gw))],
        after=True,
    )
    _ist_carry(p.owner.operation if hasattr(p.owner, "operation") else p.owner, _ist_producer(pg))
    # 3. p at its own type for every other reader.
    if gw != pw:
        (p_wide,) = _ist_splice(
            ctx,
            _ist_producer(pg),
            [pg],
            f"""  %e = tensor.empty() : {_ist_tensor_type(shape, ptype)}
  %r = {_ist_head(maps1, iters, [("%a0", _ist_tensor_type(shape, gtype))], [("%e", _ist_tensor_type(shape, ptype))])}
  ^bb0(%v: {gtype}, %o: {ptype}):
    %w = arith.extsi %v : {gtype} to {ptype}
    linalg.yield %w : {ptype}
  }} -> {_ist_tensor_type(shape, ptype)}
  return %r : {_ist_tensor_type(shape, ptype)}""",
            [p.type],
            after=True,
        )
    else:
        p_wide = pg
    # 4. the row sum, at the gather's width, widened back.
    rows = shape[:-1]
    if narrow:
        maps_r, iters_r = _ist_maps(len(rows), 2)
        (t_wide,) = _ist_splice(
            ctx,
            red_sum,
            [pg],
            f"""  %c = arith.constant {int(init)} : {gtype}
  %init = tensor.splat %c : {_ist_tensor_type(rows, gtype)}
  %s = linalg.reduce ins(%a0 : {_ist_tensor_type(shape, gtype)})
    outs(%init : {_ist_tensor_type(rows, gtype)}) dimensions = [{rank - 1}]
    (%x: {gtype}, %y: {gtype}) {{
      %z = arith.addi %x, %y : {gtype}
      linalg.yield %z : {gtype}
    }}
  %e = tensor.empty() : {_ist_tensor_type(rows, ptype)}
  %r = {_ist_head(maps_r, iters_r, [("%s", _ist_tensor_type(rows, gtype))], [("%e", _ist_tensor_type(rows, ptype))])}
  ^bb0(%v: {gtype}, %o: {ptype}):
    %w = arith.extsi %v : {gtype} to {ptype}
    linalg.yield %w : {ptype}
  }} -> {_ist_tensor_type(rows, ptype)}
  return %r : {_ist_tensor_type(rows, ptype)}""",
            [red_sum.results[0].type],
        )
        _ist_carry(red_sum, _ist_producer(t_wide))
        red_sum.results[0].replace_all_uses_with(t_wide)
    for op, index in _ist_uses(p):
        if op != red_sum:
            op.operands[index] = p_wide
    if not narrow:
        red_sum.operands[0] = p_wide

    if quant is not None:
        _ist_rewrite_quantization(ctx, plan, quant, pg)
        report["row_quantization"] = True
    return report


def _ist_rewrite_quantization(ctx, plan, quant, pg):
    ir = _ist_ir()
    shape, rank, n = plan["shape"], len(plan["shape"]), plan["shape"][-1]
    k, sq, rq = quant["k"], quant["sq"], len(quant["sq"])
    ptype = _ist_tensor(plan["p"])[1]
    first = quant["red_min"]
    for op in (quant["red_max"], quant["steps"][0].op):
        if op.is_before_in_block(first):
            first = op

    def widened(t, elem=None):
        tv = _ist_tensor(t)
        return ir.RankedTensorType.get(
            tv[0][:-1] + [tv[0][-1] // n * k],
            ir.RankedTensorType(t.type).element_type if elem is None else elem,
        )

    def empty_like(anchor, rtype):
        (e,) = _ist_splice(ctx, anchor, [], f"  %e = tensor.empty() : {rtype}\n  return %e : {rtype}", [rtype])
        return e

    def clone_onto(op, anchor, replacements, rtype):
        new = op.clone(ip=ir.InsertionPoint(anchor))
        for index, value in replacements.items():
            new.operands[index] = value
        new.results[0].set_type(rtype)
        if new.name == "tensor.expand_shape":
            static = list(ir.DenseI64ArrayAttr(new.attributes["static_output_shape"]))
            new.attributes["static_output_shape"] = ir.DenseI64ArrayAttr.get(static[:-1] + [static[-1] // n * k])
        if new.name in ("linalg.generic",):
            new.operands[len(new.operands) - 1] = empty_like(new, rtype)
        _ist_carry(op, new)
        return new

    # The K candidates k / T per row, by the probabilities' own ops on k.
    kshape = shape[:-1] + [k]
    maps1, iters = _ist_maps(rank, 1)
    (kidx,) = _ist_splice(
        ctx,
        first,
        [],
        f"""  %e = tensor.empty() : {_ist_tensor_type(kshape, ptype)}
  %r = {_ist_head(maps1, iters, [], [("%e", _ist_tensor_type(kshape, ptype))])}
  ^bb0(%o: {ptype}):
    %i = linalg.index {rank - 1} : index
    %c = arith.index_cast %i : index to {ptype}
    linalg.yield %c : {ptype}
  }} -> {_ist_tensor_type(kshape, ptype)}
  return %r : {_ist_tensor_type(kshape, ptype)}""",
        [ir.RankedTensorType.get(kshape, ir.RankedTensorType(plan["p"].type).element_type)],
    )
    g_pf, g_pr = quant["g_pf"], quant["g_pr"]
    pf = clone_onto(g_pf.op, first, {0: kidx}, widened(g_pf.op.results[0]))
    cand = clone_onto(g_pr.op, first, {0: pf.results[0], 1: quant["tf"]}, widened(g_pr.op.results[0])).results[0]
    index_view = pg
    for op in quant["chain"]:
        rtype = widened(op.results[0])
        cand = clone_onto(op, first, {0: cand}, rtype).results[0]
    quant["red_min"].operands[0] = cand
    quant["red_max"].operands[0] = cand
    # The per-element steps run on the candidates.
    steps = quant["steps"]
    last = steps[-1].op
    old_uses = _ist_uses(last.results[0])
    value_old, value_new = quant["xq"], cand
    for g in steps:
        op = g.op
        rtype = widened(op.results[0])
        for index, inp in enumerate(g.inputs):
            if inp == value_old:
                op.operands[index] = value_new
            elif _ist_tensor(inp)[0] == quant["sq"]:
                c = _ist_producer(inp)
                splat_t = ir.RankedTensorType.get(quant["sq"][:-1] + [k], ir.RankedTensorType(inp.type).element_type)
                (s,) = _ist_splice(
                    ctx,
                    op,
                    [c.operands[0]] if c.name == "tensor.splat" else [],
                    (f"  %s = tensor.splat %a0 : {splat_t}\n  return %s : {splat_t}")
                    if c.name == "tensor.splat"
                    else f"  %s = arith.constant {c.attributes['value']}\n  return %s : {splat_t}",
                    [splat_t],
                )
                quant.setdefault("dead", []).append(c)
                op.operands[index] = s
        quant.setdefault("dead", []).append(_ist_producer(op.operands[len(op.operands) - 1]))
        op.operands[len(op.operands) - 1] = empty_like(op, rtype)
        value_old = op.results[0]
        op.results[0].set_type(rtype)
        value_new = op.results[0]
    qc = last.results[0]
    # The element's int8 at its numerator: the numerator viewed as the operand is.
    anchor = last
    for op in quant["chain"]:
        if op.name in ("tensor.collapse_shape", "tensor.expand_shape"):
            new = op.clone(ip=ir.InsertionPoint(last))  # a detached op cannot be moved
            new.move_after(anchor)
            anchor = new
            new.operands[0] = index_view
            tv = _ist_tensor(op.results[0])
            new.results[0].set_type(ir.RankedTensorType.get(tv[0], ir.RankedTensorType(pg.type).element_type))
            index_view = new.results[0]
    gtype = _ist_tensor(pg)[1]
    dtype = str(ir.RankedTensorType(qc.type).element_type)
    idx = "\n    ".join(f"%i{j} = linalg.index {j} : index" for j in range(rq - 1))
    coords = ", ".join([f"%i{j}" for j in range(rq - 1)] + ["%k"])
    maps2, iters2 = _ist_maps(rq, 2)
    (gathered,) = _ist_splice(
        ctx,
        anchor,
        [index_view, qc],
        f"""  %e = tensor.empty() : {_ist_tensor_type(sq, dtype)}
  %r = {_ist_head(maps2, iters2, [("%a0", _ist_tensor_type(sq, gtype))], [("%e", _ist_tensor_type(sq, dtype))])}
  ^bb0(%p: {gtype}, %o: {dtype}):
    {idx}
    %k = arith.index_cast %p : {gtype} to index
    %v = tensor.extract %a1[{coords}] : {_ist_tensor_type(sq[:-1] + [k], dtype)}
    linalg.yield %v : {dtype}
  }} -> {_ist_tensor_type(sq, dtype)}
  return %r : {_ist_tensor_type(sq, dtype)}""",
        [ir.RankedTensorType.get(sq, ir.RankedTensorType(qc.type).element_type)],
        after=True,
    )
    _ist_carry(last, _ist_producer(gathered))
    for op, index in old_uses:
        op.operands[index] = gathered


def _ist_match_scale(plan):
    """A constant elementwise step on the scores (the attention scale) applied BEHIND a reshape.

    Walks up from the softmax input through elementwise ops' first operand. The step found must read
    a pure reshape of a ``linalg.generic`` result and constants only; applied before the reshape it is
    the same op on the same values, and it can then fuse into the op that produced them. Returns
    ``(step, reshapes, source)`` or None."""
    value = plan["x"]
    for _ in range(4):
        op = _ist_producer(value)
        g = _ist_generic(op)
        if g is None or not g.parallel or not g.outputs_unread() or not g.inputs:
            return None
        rank = len(_ist_tensor(value)[0])
        if not all(_ist_identity(m, rank) for m in g.maps[:1] + g.maps[-1:]):
            return None
        first = g.inputs[0]
        others_const = all(_ist_constant(v) is not None for v in g.inputs[1:])
        reshapes, src = [], first
        while True:
            p = _ist_producer(src)
            if p is None or p.name not in ("tensor.collapse_shape", "tensor.expand_shape"):
                break
            if len(_ist_users(src)) != 1:
                return None
            reshapes.insert(0, p)
            src = p.operands[0]
        if (
            others_const
            and reshapes
            and all(_ist_identity(m, rank) for m in g.maps)
            and len(_ist_users(first)) == 1
            and _ist_generic(_ist_producer(src)) is not None
            and all(o.name.startswith("arith.") or o.name.startswith("math.") for o in g.body[:-1])
        ):
            return g, reshapes, src
        value = first
    return None


def _ist_rewrite_scale(ctx, match):
    """Apply the constant step to the reshape's source and reshape its result instead."""
    ir = _ist_ir()
    g, reshapes, src = match
    op = g.op
    shape = _ist_tensor(src)[0]
    elem = ir.RankedTensorType(op.results[0].type).element_type
    rtype = ir.RankedTensorType.get(shape, elem)
    old_uses = _ist_uses(op.results[0])
    new = op.clone(ip=ir.InsertionPoint(op))
    new.operands[0] = src
    for index, inp in enumerate(g.inputs[1:], start=1):
        c = _ist_producer(inp)
        stype = ir.RankedTensorType.get(shape, ir.RankedTensorType(inp.type).element_type)
        if c is not None and c.name == "tensor.splat":
            (s,) = _ist_splice(
                ctx, new, [c.operands[0]], f"  %s = tensor.splat %a0 : {stype}\n  return %s : {stype}", [stype]
            )
        else:
            (s,) = _ist_splice(
                ctx, new, [], f"  %s = arith.constant {c.attributes['value']}\n  return %s : {stype}", [stype]
            )
        new.operands[index] = s
    (e,) = _ist_splice(ctx, new, [], f"  %e = tensor.empty() : {rtype}\n  return %e : {rtype}", [rtype])
    new.operands[len(new.operands) - 1] = e
    new.results[0].set_type(rtype)
    # Every map was the identity (checked), so the step is the identity over the source's rank.
    rank = len(shape)
    new.attributes["indexing_maps"] = ir.ArrayAttr.get(
        [ir.AffineMapAttr.get(ir.AffineMap.get_identity(rank))] * len(new.operands)
    )
    new.attributes["iterator_types"] = ir.ArrayAttr.get([ir.Attribute.parse("#linalg.iterator_type<parallel>")] * rank)
    _ist_carry(op, new)
    value = new.results[0]
    for r in reshapes:
        moved = r.clone(ip=ir.InsertionPoint(op))
        moved.operands[0] = value
        moved.results[0].set_type(ir.RankedTensorType.get(_ist_tensor(r.results[0])[0], elem))
        value = moved.results[0]
    for user, index in old_uses:
        user.operands[index] = value
    return [op, *reversed(reshapes)]


def _ist_erase_dead(ops):
    """Erase every op of ``ops`` (and what only they read) that nothing reads any more."""
    pending = list(ops)
    while pending:
        progressed = False
        for op in list(pending):
            if op.name not in _IST_PURE:
                pending.remove(op)
                continue
            if any(list(r.uses) for r in op.results):
                continue
            feeders = [_ist_producer(v) for v in op.operands]
            pending.remove(op)
            op.erase()
            pending.extend(f for f in feeders if f is not None and all(f != q for q in pending))
            progressed = True
        if not progressed:
            break


def _int_softmax_table(ctx, module):
    """Rewrite every integer softmax in ``module``; returns the report dict the runner prints."""
    with ctx, _ist_ir().Location.unknown():
        report = {"softmax": 0, "int32_sums": 0, "row_quantizations": 0, "scales_moved": 0, "tables": [], "refused": []}
        # Every match is made before anything is edited, and nothing is erased until every rewrite is
        # done: an erased op's binding object is invalid, so no plan may still hold one.
        plans = []
        for op in [op for op in _ist_walk(module.operation) if op.name == "linalg.generic"]:
            try:
                plan = _ist_match(op)
            except _IstRefusal as why:
                report["refused"].append(str(why))
                continue
            if plan is None:
                continue
            quant = None
            try:
                quant = _ist_match_quantization(plan)
            except _IstRefusal as why:
                report["refused"].append("row quantization: " + str(why))
            plans.append((plan, quant, _ist_match_scale(plan)))
        dead = []
        for plan, quant, scale in plans:
            done = _ist_rewrite(ctx, plan, quant)
            report["softmax"] += 1
            report["int32_sums"] += int(done["int32_sum"])
            report["row_quantizations"] += int(done["row_quantization"])
            report["tables"].append([done["table_entries"], done["table_bits"]])
            dead += [plan["red_sum"], plan["g_clamp"].op, *(g.op for g in reversed(plan["slice"]))]
            if quant is not None:
                dead += [quant["g_pr"].op, *quant["chain"], *(op for op in quant.get("dead", ()) if op is not None)]
            if scale is not None:
                dead += _ist_rewrite_scale(ctx, scale)
                report["scales_moved"] += 1
        _ist_erase_dead(dead)
        module.operation.verify()
    return report


def _ist_walk(op):
    for region in op.regions:
        for block in region.blocks:
            for inner in list(block.operations):
                inner = _ist_operation(inner)
                yield inner
                yield from _ist_walk(inner)


_INT_SOFTMAX_TABLE = len(sys.argv) > 20 and sys.argv[20] == '1'


def _ist_run_and_report(ctx, module):
    import json as _ist_json
    print(_IST_TOKEN + _ist_json.dumps(_int_softmax_table(ctx, module), sort_keys=True), flush=True)

def _lower_fma_intrinsics(ctx, module):
    from torch_mlir import ir as _fi_ir
    todo = []

    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for inner in list(block.operations):
                    if inner.operation.name == "math.fma":
                        element = inner.results[0].type
                        if isinstance(element, _fi_ir.VectorType):
                            element = element.element_type
                        if isinstance(element, (_fi_ir.F32Type, _fi_ir.F64Type)):
                            todo.append(inner)
                    walk(inner.operation)

    walk(module.operation)
    with ctx:
        for old in todo:
            with _fi_ir.InsertionPoint(old):
                new = _fi_ir.Operation.create(
                    "llvm.intr.fma", results=[old.results[0].type],
                    operands=list(old.operands), loc=old.location)
                # Retain nonsemantic attribution. Dropping optional fastmath
                # permissions selects the stricter fused operation.
                for name in old.attributes:
                    if name != "fastmath":
                        new.attributes[name] = old.attributes[name]
            old.results[0].replace_all_uses_with(new.results[0])
            old.operation.erase()
    module.operation.verify()
    return len(todo)


_FI_MARKER = "__merlin_lower_fma_to_intrinsic__"
_FI_ORIG_RUN_STAGES = _run_stages


def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(),
                pre_generalize=()):
    passes = [p for p in pipeline.split(',') if p]
    if _FI_MARKER not in passes:
        return _FI_ORIG_RUN_STAGES(ctx, module, pipeline, erase, mid, late,
                                   post_openmp, pre_generalize)
    i = passes.index(_FI_MARKER)
    _FI_ORIG_RUN_STAGES(ctx, module, ','.join(passes[:i]), erase, mid, (), (),
                        pre_generalize)
    print('OK fma_intrinsic', _lower_fma_intrinsics(ctx, module))
    _FI_ORIG_RUN_STAGES(ctx, module, ','.join(passes[i + 1:]), 0, (), late,
                        post_openmp, ())

def _scalar_contraction_spec(op):
    from torch_mlir import ir as _sc_ir
    if (op.operation.name != "linalg.generic" or len(op.operands) != 3
            or len(op.results) != 1 or len(op.regions) != 1
            or len(op.regions[0].blocks) != 1):
        return None
    types = list(v.type for v in op.operands)
    if any(not isinstance(t, _sc_ir.RankedTensorType) for t in types):
        return None
    types = [_sc_ir.RankedTensorType(t) for t in types]
    if (any(not isinstance(t.element_type, _sc_ir.F32Type) for t in types)
            or op.results[0].type != op.operands[2].type
            or any(d < 0 for t in types for d in t.shape)):
        return None
    block = op.regions[0].blocks[0]
    body = list(block.operations)
    if (len(block.arguments) != 3 or len(body) != 3
            or [x.operation.name for x in body] != ["arith.mulf", "arith.addf", "linalg.yield"]
            or any("fastmath" in x.attributes
                   and str(x.attributes["fastmath"]) != "#arith.fastmath<none>" for x in body)
            or any(name != "fastmath" and not name.startswith("prov.")
                   for x in body[:2] for name in x.attributes)):
        return None
    mul, add, yield_op = body
    args = list(block.arguments)
    if list(mul.operands) == args[:2]:
        mul_order = [0, 1]
    elif list(mul.operands) == args[1::-1]:
        mul_order = [1, 0]
    else:
        return None
    if list(add.operands) == [mul.results[0], args[2]]:
        add_order = [0, 1]
    elif list(add.operands) == [args[2], mul.results[0]]:
        add_order = [1, 0]
    else:
        return None
    if list(yield_op.operands) != [add.results[0]]:
        return None
    maps = [_sc_ir.AffineMapAttr(a).value for a in op.attributes["indexing_maps"]]
    if len(maps) != 3:
        return None
    dims = maps[0].n_dims
    if dims < 2 or any(m.n_dims != dims or m.n_symbols for m in maps):
        return None
    iters = list(op.attributes["iterator_types"])
    # Enum attribute printer is the upstream binding's stable spelling.
    if ([str(x) for x in iters] !=
            ["#linalg.iterator_type<parallel>"] * (dims - 1) + ["#linalg.iterator_type<reduction>"]):
        return None
    positions = []
    extents = [None] * dims
    for m, t in zip(maps, types):
        if len(m.results) != len(t.shape):
            return None
        pos = []
        for expr, extent in zip(m.results, t.shape):
            if not isinstance(expr, _sc_ir.AffineDimExpr):
                return None
            d = _sc_ir.AffineDimExpr(expr).position
            if d in pos or (extents[d] is not None and extents[d] != extent):
                return None
            pos.append(d)
            extents[d] = extent
        positions.append(pos)
    if (sorted(positions[2]) != list(range(dims - 1))
            or any(dims - 1 not in pos for pos in positions[:2])
            or any(d is None for d in extents)):
        return None
    return positions, extents, mul_order, add_order


def _scalarize_tensor_contractions(
        ctx, module, outputs=1, reduction_unroll=0, rows=1, selection=None, output_guard=None,
        product_guard=None, reduction_limit=None, reduction_setup=None):
    from torch_mlir import ir as _sc_ir
    todo = []

    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for inner in list(block.operations):
                    spec = _scalar_contraction_spec(inner)
                    eligible = (spec is not None and spec[1][-2] % outputs == 0
                                and (selection is None or any(inner.operation == item for item in selection)))
                    if eligible and rows != 1:
                        positions, extents, _, _ = spec
                        row_axis, col_axis = len(extents) - 3, len(extents) - 2
                        # A rectangular tile is a matrix schedule only when the
                        # exact projected maps prove complementary row/column
                        # reuse. No names, shapes alone or pointer facts suffice.
                        eligible = (row_axis >= 0 and extents[row_axis] % rows == 0
                                    and row_axis in positions[0] and col_axis not in positions[0]
                                    and col_axis in positions[1] and row_axis not in positions[1])
                        owner = inner.operation
                        while eligible and owner is not None:
                            if any(a in owner.attributes for a in ("strictfp", "llvm.strictfp")):
                                eligible = False
                            owner = owner.parent
                    if eligible:
                        todo.append((inner, spec))
                    else:
                        walk(inner.operation)

    walk(module.operation)
    with ctx:
        for old, (positions, extents, mul_order, add_order) in todo:
            with old.location, _sc_ir.InsertionPoint(old):
                index = _sc_ir.IndexType.get()
                scalar = old.regions[0].blocks[0].arguments[0].type
                tensor = old.results[0].type
                def create(name, operands=(), results=(), attributes=None, regions=0):
                    return _sc_ir.Operation.create(name, operands=list(operands), results=list(results),
                                                   attributes=attributes or {}, regions=regions)
                def constant(value):
                    return create("arith.constant", results=[index], attributes={
                        "value": _sc_ir.IntegerAttr.get(index, value)}).results[0]
                zero, one = constant(0), constant(1)
                limits = [constant(e) for e in extents]
                step = one if outputs == 1 else constant(outputs)
                offsets = [zero, *[constant(i) for i in range(1, outputs)]]
                row_offsets = [zero, *[constant(i) for i in range(1, rows)]]
                row_step = one if rows == 1 else constant(rows)
                workspace = (reduction_setup(old.operation, positions, extents, create, constant)
                             if reduction_setup is not None else None)

                def output_loop(depth, current, indices, bound=None):
                    if depth < len(extents) - 1:
                        if reduction_limit is not None and depth == len(extents) - 2:
                            bound = reduction_limit(old.operation, indices, positions, extents,
                                                    rows, create, constant, workspace)
                        stride = (step if depth == len(extents) - 2 else
                                  row_step if rows != 1 and depth == len(extents) - 3 else one)
                        loop = create("scf.for", [zero, limits[depth], stride, current], [tensor], regions=1)
                        block = _sc_ir.Block.create_at_start(loop.regions[0], [index, tensor])
                        with _sc_ir.InsertionPoint(block):
                            updated = output_loop(depth + 1, block.arguments[1],
                                                  [*indices, block.arguments[0]], bound)
                            create("scf.yield", [updated])
                        return loop.results[0]
                    if rows == 1:
                        lanes = [indices]
                        for offset in offsets[1:]:
                            column = create("arith.addi", [indices[-1], offset], [index]).results[0]
                            lanes.append([*indices[:-1], column])
                    else:
                        lanes = []
                        for row_offset in row_offsets:
                            row = (indices[-2] if row_offset == zero else
                                   create("arith.addi", [indices[-2], row_offset], [index]).results[0])
                            for offset in offsets:
                                column = (indices[-1] if offset == zero else
                                          create("arith.addi", [indices[-1], offset], [index]).results[0])
                                lanes.append([*indices[:-2], row, column])
                    out_indices = [[lane[d] for d in positions[2]] for lane in lanes]
                    def reduction(ordered=True):
                        initial = [create("tensor.extract", [current, *idx], [scalar]).results[0]
                                   for idx in out_indices]
                        # SCF -> CF forwards LLVM-typed values; CF -> LLVM keeps the key.
                        # Use the final branch property's spelling: a retained namespaced
                        # llvm.loop_annotation is ignored by the installed translator.
                        attributes = ({"loop_annotation": _sc_ir.Attribute.parse(
                            "#llvm.loop_annotation<unroll = <count = " + str(reduction_unroll) + " : i32>>")}
                            if reduction_unroll else {})
                        lane_count = rows * outputs
                        order = bound[1] if ordered and isinstance(bound, tuple) else None
                        limit = (limits[-1] if not ordered or bound is None else
                                 bound[0] if order is not None else bound)
                        loop = create("scf.for", [zero, limit, one, *initial], [scalar] * lane_count,
                                      attributes=attributes, regions=1)
                        block = _sc_ir.Block.create_at_start(loop.regions[0], [index, *([scalar] * lane_count)])
                        with _sc_ir.InsertionPoint(block):
                            load = ("memref.load" if order is not None and
                                    isinstance(order.type, _sc_ir.MemRefType) else "tensor.extract")
                            k = block.arguments[0]
                            if order is not None:
                                storage = (_sc_ir.MemRefType(order.type).element_type if load == "memref.load"
                                           else _sc_ir.RankedTensorType(order.type).element_type)
                                k = create(load, [order, k], [storage]).results[0]
                                if storage != index:
                                    k = create("arith.index_cast", [k], [index]).results[0]
                            accumulated, shared, pending = [], {}, []
                            def update(values, prior):
                                product = create("arith.mulf", [values[d] for d in mul_order], [scalar]).results[0]
                                operands = [product, prior]
                                return create("arith.addf", [operands[d] for d in add_order], [scalar]).results[0]
                            for lane_number, lane in enumerate(lanes):
                                all_indices = [*lane, k]
                                values = []
                                for i in range(2):
                                    if rows == 1:
                                        key = i if len(extents) - 2 not in positions[i] else None
                                    else:
                                        key = (i,
                                               lane_number // outputs if len(extents) - 3 in positions[i] else 0,
                                               lane_number % outputs if len(extents) - 2 in positions[i] else 0)
                                    if key is not None and key in shared:
                                        value = shared[key]
                                    else:
                                        value = create(
                                            "tensor.extract",
                                            [old.operands[i], *[all_indices[d] for d in positions[i]]],
                                            [scalar]).results[0]
                                        if key is not None:
                                            shared[key] = value
                                    values.append(value)
                                prior = block.arguments[lane_number + 1]
                                if product_guard is None:
                                    accumulated.append(update(values, prior))
                                else:
                                    pending.append((values, prior))
                            if product_guard is not None:
                                for start in range(0, lane_count, outputs):
                                    group = pending[start:start + outputs]
                                    condition = product_guard(old.operation, group[0][0], create)
                                    choice = create("scf.if", [condition], [scalar] * outputs, regions=2)
                                    for branch in range(2):
                                        child = _sc_ir.Block.create_at_start(choice.regions[branch], [])
                                        with _sc_ir.InsertionPoint(child):
                                            create("scf.yield", [prior if branch == 0 else update(values, prior)
                                                                 for values, prior in group])
                                    accumulated.extend(choice.results)
                            create("scf.yield", accumulated)
                        destination = current
                        for value, idx in zip(loop.results, out_indices):
                            destination = create("tensor.insert", [value, destination, *idx], [tensor]).results[0]
                        return destination
                    def selected_reduction():
                        if not isinstance(bound, tuple) or len(bound) < 3:
                            return reduction()
                        guarded = create("scf.if", [bound[2]], [tensor], regions=2)
                        for branch in range(2):
                            block = _sc_ir.Block.create_at_start(guarded.regions[branch], [])
                            with _sc_ir.InsertionPoint(block):
                                create("scf.yield", [reduction(ordered=branch == 0)])
                        return guarded.results[0]
                    if output_guard is None:
                        return selected_reduction()
                    condition = output_guard(old.operation, lanes, positions[2], create, constant)
                    guarded = create("scf.if", [condition], [tensor], regions=2)
                    for branch in range(2):
                        block = _sc_ir.Block.create_at_start(guarded.regions[branch], [])
                        with _sc_ir.InsertionPoint(block):
                            create("scf.yield", [selected_reduction() if branch == 0 else current])
                    return guarded.results[0]

                result = output_loop(0, old.operands[2], [])
                replacement = result.owner
                for name in old.attributes:
                    if name.startswith("prov."):
                        replacement.attributes[name] = old.attributes[name]
                previous = old.attributes["prov.transforms"] if "prov.transforms" in old.attributes else None
                text = _sc_ir.StringAttr(previous).value + "," if previous is not None else ""
                suffix = (f"_{rows}x{outputs}_outputs" if rows != 1 else
                          "" if outputs == 1 else "_" + str(outputs) + "_outputs")
                transforms = text + "scalar_contraction_accumulator" + suffix
                if reduction_unroll:
                    transforms += ",unroll_scalar_contraction_reduction_by_" + str(reduction_unroll)
                replacement.attributes["prov.transforms"] = _sc_ir.StringAttr.get(transforms)
            old.results[0].replace_all_uses_with(result)
            old.operation.erase()
    module.operation.verify()
    return len(todo)


_SC_MARKER = "__merlin_scalar_contraction_accumulator__"
_SC2_MARKER = "__merlin_scalar_contraction_accumulator_2_outputs__"
_SC4_MARKER = "__merlin_scalar_contraction_accumulator_4_outputs__"
_SC8_MARKER = "__merlin_scalar_contraction_accumulator_8_outputs__"
_SC2X4_MARKER = "__merlin_scalar_contraction_accumulator_2x4_outputs__"
_SC4X4_MARKER = "__merlin_scalar_contraction_accumulator_4x4_outputs__"
_SC2X8_MARKER = "__merlin_scalar_contraction_accumulator_2x8_outputs__"
_SC_MARKERS = (_SC_MARKER, _SC2_MARKER, _SC4_MARKER, _SC8_MARKER, _SC2X4_MARKER, _SC4X4_MARKER, _SC2X8_MARKER)
_SC_UNROLL2_MARKER = "__merlin_scalar_contraction_reduction_unroll_2__"
_SC_UNROLL4_MARKER = "__merlin_scalar_contraction_reduction_unroll_4__"
_SC_UNROLL_MARKERS = {_SC_UNROLL2_MARKER: 2, _SC_UNROLL4_MARKER: 4}
_SC_ORIG_RUN_STAGES = _run_stages


def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(), pre_generalize=()):
    passes = [p for p in pipeline.split(',') if p]
    hints = [m for m in _SC_UNROLL_MARKERS if m in passes]
    if len(hints) > 1:
        raise ValueError("scalar contraction reduction unroll counts are alternatives")
    unroll = _SC_UNROLL_MARKERS[hints[0]] if hints else 0
    passes = [p for p in passes if p not in _SC_UNROLL_MARKERS]
    selected = [m for m in _SC_MARKERS if m in passes]
    if not selected:
        return _SC_ORIG_RUN_STAGES(ctx, module, pipeline, erase, mid, late, post_openmp, pre_generalize)
    if len(selected) != 1:
        raise ValueError("scalar contraction accumulator schedules are alternatives")
    i = passes.index(selected[0])
    head_generalizes = any('linalg-generalize-named-ops' in p for p in passes[:i])
    _SC_ORIG_RUN_STAGES(ctx, module, ','.join(passes[:i]), 0, (), (), (),
                        pre_generalize if head_generalizes else ())
    if not head_generalizes:
        _SC_ORIG_RUN_STAGES(ctx, module, 'func.func(linalg-generalize-named-ops)', 0, (), (), (),
                            pre_generalize)
    rows = {_SC2X4_MARKER: 2, _SC4X4_MARKER: 4, _SC2X8_MARKER: 2}.get(selected[0], 1)
    outputs = {_SC_MARKER: 1, _SC2_MARKER: 2, _SC4_MARKER: 4, _SC8_MARKER: 8,
               _SC2X4_MARKER: 4, _SC4X4_MARKER: 4, _SC2X8_MARKER: 8}[selected[0]]
    name = 'scalar_contraction_accumulator' + (
        '_' + str(rows) + 'x' + str(outputs) + '_outputs' if rows != 1
        else '' if outputs == 1 else '_' + str(outputs) + '_outputs')
    count = _scalarize_tensor_contractions(ctx, module, outputs, unroll, rows)
    print('OK', name, count)
    if unroll:
        print('OK unroll_scalar_contraction_reduction_by_' + str(unroll), count)
    _SC_ORIG_RUN_STAGES(ctx, module, ','.join(passes[i + 1:]), erase, mid, late, post_openmp, ())

_ZC_INDEX_CAPACITY = 4096
_ZC_INDEX_TOTAL_CAPACITY = 8192


def _zc_owner(value):
    owner = value.owner
    return owner.operation if hasattr(owner, "operation") else None


def _zc_zero_scalar(value):
    import struct
    from torch_mlir import ir

    owner = _zc_owner(value)
    return (
        owner is not None
        and owner.name == "arith.constant"
        and "value" in owner.attributes
        and isinstance(owner.attributes["value"], ir.FloatAttr)
        and struct.pack(">f", ir.FloatAttr(owner.attributes["value"]).value) == b"\0\0\0\0"
    )


def _zc_zero_seed(value):
    from torch_mlir import ir

    owner = _zc_owner(value)
    if owner is None:
        return False
    if owner.name in ("linalg.fill", "tensor.splat"):
        return _zc_zero_scalar(owner.operands[0])
    if owner.name == "arith.constant" and "value" in owner.attributes:
        attr = owner.attributes["value"]
        if isinstance(attr, ir.DenseFPElementsAttr) and attr.is_splat:
            return _zc_zero_scalar_attr(attr.get_splat_value())
    return False


def _zc_zero_scalar_attr(attr):
    import struct
    from torch_mlir import ir

    return isinstance(attr, ir.FloatAttr) and struct.pack(">f", ir.FloatAttr(attr).value) == b"\0\0\0\0"


def _zc_finite_integer_rhs(value, depth=0):
    from torch_mlir import ir

    if depth > 16:
        return False
    owner = _zc_owner(value)
    if owner is None:
        return False
    if owner.name in (
        "tensor.extract_slice",
        "tensor.collapse_shape",
        "tensor.expand_shape",
        "tensor.cast",
        "linalg.transpose",
        "linalg.copy",
    ):
        return _zc_finite_integer_rhs(owner.operands[0], depth + 1)
    if owner.name != "linalg.generic" or len(owner.operands) != 2 or len(owner.regions) != 1:
        return False
    block = owner.regions[0].blocks[0]
    body = list(block.operations)
    if [str(x) for x in owner.attributes["iterator_types"]] != ["#linalg.iterator_type<parallel>"] * len(
        ir.RankedTensorType(value.type).shape
    ):
        return False
    maps = [ir.AffineMapAttr(x).value for x in owner.attributes["indexing_maps"]]
    out = maps[-1]
    if (
        out.n_symbols
        or len(out.results) != out.n_dims
        or sorted(ir.AffineDimExpr(x).position for x in out.results if isinstance(x, ir.AffineDimExpr))
        != list(range(out.n_dims))
    ):
        return False
    if len(body) == 1 and body[0].operation.name == "linalg.yield" and list(body[0].operands) == [block.arguments[0]]:
        return _zc_finite_integer_rhs(owner.operands[0], depth + 1)
    if (
        len(body) != 2
        or [x.operation.name for x in body] != ["arith.sitofp", "linalg.yield"]
        or list(body[0].operands) != [block.arguments[0]]
        or list(body[1].operands) != [body[0].results[0]]
        or not isinstance(block.arguments[0].type, ir.IntegerType)
        or not 1 <= ir.IntegerType(block.arguments[0].type).width <= 64
    ):
        return False
    return str(body[0].results[0].type) == "f32"


def _zc_scalarize(ctx, module, outputs=1, reduction_unroll=0, rows=1, selection=None, output_guard=None):
    from torch_mlir import ir

    if output_guard is not None or selection is not None:
        raise ValueError("guarded zero products do not compose with another observation guard")
    plans = []

    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for child in list(block.operations):
                    spec = _scalar_contraction_spec(child)
                    if (
                        spec is not None
                        and len(spec[1]) >= 3
                        and len(spec[1]) - 2 not in spec[0][0]
                        and len(spec[1]) - 2 in spec[0][1]
                        and spec[1][-2] % outputs == 0
                        and (rows == 1 or len(spec[1]) >= 3 and spec[1][-3] % rows == 0)
                        and 0 <= spec[1][-1] <= (_ZC_INDEX_CAPACITY if _ZC_INDICES else 1 << 20)
                        and _zc_zero_seed(child.operands[2])
                        and _zc_finite_integer_rhs(child.operands[1])
                    ):
                        ancestor = child.operation
                        safe = True
                        while ancestor is not None:
                            if any(
                                "strictfp" in name or "strictfp" in str(ancestor.attributes[name])
                                for name in ancestor.attributes
                            ):
                                safe = False
                            ancestor = ancestor.parent
                        if safe:
                            plans.append(child.operation)
                    else:
                        walk(child.operation)

    walk(module.operation)
    if _ZC_INDICES:
        # Bound all private index workspaces in this source module. Every
        # retained fallback remains a complete original contraction.
        admitted, used = [], 0
        for operation in plans:
            count = _scalar_contraction_spec(operation)[1][-1]
            if used + count <= _ZC_INDEX_TOTAL_CAPACITY:
                admitted.append(operation)
                used += count
        plans = admitted
    with ctx, ir.Location.unknown():
        i1, i32, index = ir.IntegerType.get_signless(1), ir.IntegerType.get_signless(32), ir.IndexType.get()

        def create(name, operands=(), results=(), attributes=None, regions=0):
            return ir.Operation.create(
                name, operands=list(operands), results=list(results), attributes=attributes or {}, regions=regions
            )

        symbols = [
            o
            for o in module.body.operations
            if "sym_name" in o.attributes
            and ir.StringAttr(o.attributes["sym_name"]).value == "__merlin_contraction_round_is_nearest"
        ]
        if plans and symbols:
            raise ValueError("guarded zero runtime helper symbol collision")
        if plans:
            with ir.InsertionPoint(module.body):
                create(
                    "func.func",
                    attributes={
                        "sym_name": ir.StringAttr.get("__merlin_contraction_round_is_nearest"),
                        "sym_visibility": ir.StringAttr.get("private"),
                        "function_type": ir.TypeAttr.get(ir.FunctionType.get([], [i32])),
                    },
                    regions=1,
                )
        fast = []
        for old in plans:
            with old.location, ir.InsertionPoint(old):

                def const(value, typ):
                    return create(
                        "arith.constant", results=[typ], attributes={"value": ir.IntegerAttr.get(typ, value)}
                    ).results[0]

                zero, one, yes = const(0, index), const(1, index), const(1, i1)
                magnitude_mask, limit = const(0x7FFFFFFF, i32), const(0x3F800000, i32)

                def scan(depth, coordinates, admitted):
                    shape = list(ir.RankedTensorType(old.operands[0].type).shape)
                    if depth == len(shape):
                        x = create("tensor.extract", [old.operands[0], *coordinates], [ir.F32Type.get()]).results[0]
                        word = create("arith.bitcast", [x], [i32]).results[0]
                        word = create("arith.andi", [word, magnitude_mask], [i32]).results[0]
                        ok = create(
                            "arith.cmpi",
                            [word, limit],
                            [i1],
                            {"predicate": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 7)},
                        ).results[0]
                        return create("arith.andi", [admitted, ok], [i1]).results[0]
                    loop = create("scf.for", [zero, const(shape[depth], index), one, admitted], [i1], regions=1)
                    block = ir.Block.create_at_start(loop.regions[0], [index, i1])
                    with ir.InsertionPoint(block):
                        create("scf.yield", [scan(depth + 1, [*coordinates, block.arguments[0]], block.arguments[1])])
                    return loop.results[0]

                nearest = create(
                    "func.call",
                    results=[i32],
                    attributes={"callee": ir.FlatSymbolRefAttr.get("__merlin_contraction_round_is_nearest")},
                ).results[0]
                nearest = create(
                    "arith.cmpi",
                    [nearest, const(0, i32)],
                    [i1],
                    {"predicate": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 1)},
                ).results[0]
                admitted = (nearest if _ZC_INDICES else
                            create("arith.andi", [nearest, scan(0, [], yes)], [i1]).results[0])
                guarded = create("scf.if", [admitted], [old.results[0].type], regions=2)
                for branch in range(2):
                    block = ir.Block.create_at_start(guarded.regions[branch], [])
                    with ir.InsertionPoint(block):
                        clone = old.clone(ip=ir.InsertionPoint(block))
                        if branch == 0:
                            fast.append(clone)
                        create("scf.yield", [clone.results[0]])
                old.results[0].replace_all_uses_with(guarded.results[0])
                old.erase()

        def skip(operation, values, create):
            word = create("arith.bitcast", [values[0]], [i32]).results[0]
            mask = create(
                "arith.constant", results=[i32], attributes={"value": ir.IntegerAttr.get(i32, 0x7FFFFFFF)}
            ).results[0]
            zero_word = create(
                "arith.constant", results=[i32], attributes={"value": ir.IntegerAttr.get(i32, 0)}
            ).results[0]
            word = create("arith.andi", [word, mask], [i32]).results[0]
            return create(
                "arith.cmpi",
                [word, zero_word],
                [i1],
                {"predicate": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 0)},
            ).results[0]

        def suffix_bound(operation, indices, positions, extents, row_count, create, constant, workspace=None):
            # One scan per row block, outside the output-column loop. The maximum
            # last-nonzero position covers every row in that block, in source K order.
            zero, one = constant(0), constant(1)
            state, types = ([zero, const(1, i1)], [index, i1]) if _ZC_INDICES else ([zero], [index])
            loop = create("scf.for", [zero, constant(extents[-1]), one, *state], types, regions=1)
            block = ir.Block.create_at_start(loop.regions[0], [index, *types])
            with ir.InsertionPoint(block):
                nonzero = const(0, i1)
                admitted = block.arguments[2] if _ZC_INDICES else None
                for row in range(row_count):
                    coordinates = [*indices, zero, block.arguments[0]]
                    if row:
                        coordinates[-3] = create("arith.addi", [coordinates[-3], constant(row)], [index]).results[0]
                    value = create(
                        "tensor.extract",
                        [operation.operands[0], *[coordinates[d] for d in positions[0]]],
                        [ir.F32Type.get()],
                    ).results[0]
                    word = create("arith.bitcast", [value], [i32]).results[0]
                    word = create("arith.andi", [word, const(0x7FFFFFFF, i32)], [i32]).results[0]
                    present = create(
                        "arith.cmpi",
                        [word, const(0, i32)],
                        [i1],
                        {"predicate": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 1)},
                    ).results[0]
                    nonzero = create("arith.ori", [nonzero, present], [i1]).results[0]
                    if _ZC_INDICES:
                        bounded = create(
                            "arith.cmpi", [word, const(0x3f800000, i32)], [i1],
                            {"predicate": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 7)},
                        ).results[0]
                        admitted = create("arith.andi", [admitted, bounded], [i1]).results[0]
                if _ZC_INDICES:
                    selected = create("scf.if", [nonzero], [index], regions=2)
                    for branch in range(2):
                        child = ir.Block.create_at_start(selected.regions[branch], [])
                        with ir.InsertionPoint(child):
                            count = block.arguments[1]
                            if branch == 0:
                                storage = ir.MemRefType(workspace.type).element_type
                                stored_index = block.arguments[0]
                                if storage != index:
                                    stored_index = create("arith.index_cast", [stored_index], [storage]).results[0]
                                create("memref.store", [stored_index, workspace, count])
                                count = create("arith.addi", [count, one], [index]).results[0]
                            create("scf.yield", [count])
                    create("scf.yield", [selected.results[0], admitted])
                else:
                    following = create("arith.addi", [block.arguments[0], one], [index]).results[0]
                    bound = create("arith.select", [nonzero, following, block.arguments[1]], [index]).results[0]
                    create("scf.yield", [bound])
            return (loop.results[0], workspace, loop.results[1]) if _ZC_INDICES else loop.results[0]

        def setup(operation, positions, extents, create, constant):
            # The existing K<=4096 guard admits only indices in [0,4095].
            # Retain wide storage if a future internal producer has larger K.
            storage = ir.IntegerType.get_signless(16) if extents[-1] <= 1 << 15 else index
            typ = ir.MemRefType.get([extents[-1]], storage)
            return create(
                "memref.alloca", results=[typ], attributes={"operandSegmentSizes": ir.DenseI32ArrayAttr.get([0, 0])}
            ).results[0]

        options = {"reduction_limit": suffix_bound} if _ZC_PREFIX or _ZC_INDICES else {"product_guard": skip}
        if _ZC_INDICES:
            options["reduction_setup"] = setup
        count = (
            _ZC_ORIGINAL_SCALARIZE(ctx, module, outputs, reduction_unroll, rows, selection=fast, **options)
            if fast
            else 0
        )
        count += _ZC_ORIGINAL_SCALARIZE(ctx, module, outputs, reduction_unroll, rows)
        print("OK scalar_contraction_guarded_zero_products", len(plans))
        return count


_ZC_ORIGINAL_SCALARIZE = _scalarize_tensor_contractions
_ZC_ORIGINAL_RUN_STAGES = _run_stages
_ZC_PREFIX = False
_ZC_INDICES = False


def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(), pre_generalize=()):
    global _scalarize_tensor_contractions, _ZC_PREFIX, _ZC_INDICES
    passes = [p for p in pipeline.split(",") if p]
    markers = (
        "__merlin_scalar_contraction_guarded_zero_products__",
        "__merlin_scalar_contraction_guarded_zero_suffix__",
        "__merlin_scalar_contraction_guarded_nonzero_reduction__",
    )
    if not any(marker in passes for marker in markers):
        return _ZC_ORIGINAL_RUN_STAGES(ctx, module, pipeline, erase, mid, late, post_openmp, pre_generalize)
    saved = _scalarize_tensor_contractions
    prior_prefix = _ZC_PREFIX
    prior_indices = _ZC_INDICES
    _ZC_PREFIX = markers[1] in passes
    _ZC_INDICES = markers[2] in passes
    _scalarize_tensor_contractions = _zc_scalarize
    try:
        return _ZC_ORIGINAL_RUN_STAGES(
            ctx, module, ",".join(p for p in passes if p not in markers), erase, mid, late, post_openmp, pre_generalize
        )
    finally:
        _scalarize_tensor_contractions = saved
        _ZC_PREFIX = prior_prefix
        _ZC_INDICES = prior_indices

def _ae_spec(operation):
    from torch_mlir import ir

    if (
        operation.name != "linalg.generic"
        or len(operation.results) != 1
        or len(operation.regions) != 1
        or len(operation.regions[0].blocks) != 1
        or len(operation.operands) < 2
    ):
        return None
    ancestor = operation
    while ancestor is not None:
        if any("strictfp" in name or "strictfp" in str(ancestor.attributes[name]) for name in ancestor.attributes):
            return None
        ancestor = ancestor.parent
    allowed = {"indexing_maps", "iterator_types", "operandSegmentSizes"}
    if any(name not in allowed and not name.startswith("prov.") for name in operation.attributes):
        return None
    types = [value.type for value in operation.operands]
    if any(
        not isinstance(typ, ir.RankedTensorType) or any(x < 0 for x in ir.RankedTensorType(typ).shape) for typ in types
    ):
        return None
    output = ir.RankedTensorType(types[-1])
    rank = len(output.shape)
    if not rank or str(output.element_type) != "f32" or operation.results[0].type != types[-1]:
        return None
    if [str(x) for x in operation.attributes["iterator_types"]] != ["#linalg.iterator_type<parallel>"] * rank:
        return None
    maps = [ir.AffineMapAttr(x).value for x in operation.attributes["indexing_maps"]]
    if len(maps) != len(types) or any(m.n_dims != rank or m.n_symbols for m in maps):
        return None
    output_map = maps[-1]
    if (
        len(output_map.results) != rank
        or any(not isinstance(x, ir.AffineDimExpr) for x in output_map.results)
        or sorted(ir.AffineDimExpr(x).position for x in output_map.results) != list(range(rank))
    ):
        return None
    extents = [None] * rank
    for expr, extent in zip(output_map.results, output.shape):
        extents[ir.AffineDimExpr(expr).position] = extent
    positions = []
    for affine, typ in zip(maps, types):
        shape = ir.RankedTensorType(typ).shape
        if len(affine.results) != len(shape):
            return None
        indices = []
        for expr, extent in zip(affine.results, shape):
            if isinstance(expr, ir.AffineDimExpr):
                axis = ir.AffineDimExpr(expr).position
                if extent != extents[axis]:
                    return None
                indices.append(axis)
            elif isinstance(expr, ir.AffineConstantExpr) and ir.AffineConstantExpr(expr).value == 0 and extent == 1:
                indices.append(None)
            else:
                return None
        positions.append(indices)
    block = operation.regions[0].blocks[0]
    body = list(block.operations)
    exps = [op for op in body if op.operation.name == "math.exp"]
    arithmetic = {
        "arith.constant",
        "arith.sitofp",
        "arith.uitofp",
        "arith.extf",
        "arith.truncf",
        "arith.addf",
        "arith.subf",
        "arith.mulf",
        "arith.divf",
        "arith.cmpf",
        "arith.cmpi",
        "arith.select",
        "arith.bitcast",
        "arith.extsi",
        "arith.extui",
        "arith.trunci",
        "arith.addi",
        "arith.subi",
        "arith.muli",
        "arith.andi",
        "arith.ori",
        "arith.xori",
    }
    if (
        len(exps) != 1
        or len(exps[0].operands) != 1
        or str(exps[0].operands[0].type) != "f32"
        or str(exps[0].results[0].type) != "f32"
        or body[-1].operation.name != "linalg.yield"
        or list(body[-1].operands) != [exps[0].results[0]]
        or list(block.arguments[-1].uses)
        or any(op.regions or op.operation.name not in arithmetic | {"math.exp"} for op in body[:-1])
        or any("fastmath" in op.attributes and str(op.attributes["fastmath"]) != "#arith.fastmath<none>" for op in body)
    ):
        return None
    return positions, extents


def _memoize_adjacent_tensor_exp(
    ctx, module, cache_entries=0, errno_checked=False, underflow_class=None,
    invocation_shared=False, memoize_inputs=False, inner_unroll=0
):
    from torch_mlir import ir

    plans = []
    groups = []

    def walk(parent):
        for region in parent.regions:
            for block in region.blocks:
                group = None
                if invocation_shared and parent.name == "func.func" and len(parent.regions[0].blocks) == 1:
                    group = len(groups)
                    groups.append(block)
                for child in list(block.operations):
                    spec = _ae_spec(child.operation)
                    if spec is not None:
                        if invocation_shared and group is None:
                            continue
                        plans.append((child.operation, spec, group))
                    else:
                        walk(child.operation)

    walk(module.operation)
    with ctx, ir.Location.unknown():
        index, i1, i32, f32 = (
            ir.IndexType.get(),
            ir.IntegerType.get_signless(1),
            ir.IntegerType.get_signless(32),
            ir.F32Type.get(),
        )
        if plans and errno_checked:
            functions = {
                "__merlin_exp_errno_location": ([], [ir.Type.parse("!llvm.ptr")]),
                "__merlin_exp_observe_errno": ([f32], [f32]),
            }
            if invocation_shared:
                for name in ("__merlin_exp_environment_matches", "__merlin_exp_environment_capture"):
                    functions[name] = ([ir.MemRefType.get([4096], ir.IntegerType.get_signless(8))], [i32])
            if any(
                "sym_name" in op.attributes and ir.StringAttr(op.attributes["sym_name"]).value in functions
                for op in module.body.operations
            ):
                raise ValueError("exp errno helper symbol collision")
            with ir.InsertionPoint(module.body):
                for symbol, (arguments, results) in functions.items():
                    ir.Operation.create(
                        "func.func",
                        attributes={
                            "sym_name": ir.StringAttr.get(symbol),
                            "sym_visibility": ir.StringAttr.get("private"),
                            "function_type": ir.TypeAttr.get(ir.FunctionType.get(arguments, results)),
                        },
                        regions=1,
                    )
        cache_budget = 8192
        shared = {}
        for old, (positions, extents), group in plans:
            previous = shared.get(group) if group is not None else None
            entries = previous[0] if previous else cache_entries if cache_budget >= cache_entries else 0
            if invocation_shared and group is None:
                entries = 0
            if previous is None:
                cache_budget -= entries
            input_key = None
            if memoize_inputs:
                varying = [i for i, axes in enumerate(positions[:-1]) if len(extents) - 1 in axes]
                if len(varying) == 1 and str(old.regions[0].blocks[0].arguments[varying[0]].type) == "f32":
                    input_key = varying[0]
            with old.location, ir.InsertionPoint(old):

                def create(name, operands=(), results=(), attributes=None, regions=0):
                    return ir.Operation.create(
                        name,
                        operands=list(operands),
                        results=list(results),
                        attributes=attributes or {},
                        regions=regions,
                    )

                def constant(value, typ=index):
                    return create(
                        "arith.constant",
                        results=[typ],
                        attributes={"value": ir.IntegerAttr.get(typ, value)},
                    ).results[0]

                zero, one = constant(0), constant(1)
                scalar_zero = create(
                    "arith.constant",
                    results=[f32],
                    attributes={"value": ir.FloatAttr.get(f32, 0.0)},
                ).results[0]
                state = [
                    old.operands[-1],
                    constant(0, i1),
                    constant(0, i32),
                    scalar_zero,
                ]
                state_types = [old.results[0].type, i1, i32, f32]
                original_block = old.regions[0].blocks[0]
                keys = values = errnos = errno_pointer = environment = None
                if previous is not None:
                    _, keys, values, errnos, errno_pointer, environment, environment_valid = previous
                if entries and errno_checked and previous is None:
                    if not invocation_shared:
                        errnos = create("memref.alloca", results=[ir.MemRefType.get([entries], i32)]).results[0]
                    errno_pointer = create(
                        "func.call",
                        results=[ir.Type.parse("!llvm.ptr")],
                        attributes={
                            "callee": ir.FlatSymbolRefAttr.get("__merlin_exp_errno_location"),
                        },
                    ).results[0]
                if entries:
                    if previous is None:
                        if invocation_shared:
                            storage = create(
                                "memref.alloca",
                                results=[ir.MemRefType.get([entries * 16], ir.IntegerType.get_signless(8))],
                                attributes={"alignment": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 16)},
                            ).results[0]
                            keys = create(
                                "memref.view", [storage, zero], [ir.MemRefType.get([entries, 4], i32)]
                            ).results[0]
                            values = create(
                                "memref.view", [storage, zero], [ir.MemRefType.get([entries, 4], f32)]
                            ).results[0]
                            errnos = keys
                        else:
                            keys = create("memref.alloca", results=[ir.MemRefType.get([entries], i32)]).results[0]
                            values = create("memref.alloca", results=[ir.MemRefType.get([entries], f32)]).results[0]
                        if group is not None:
                            environment = create(
                                "memref.alloca", results=[ir.MemRefType.get([4096], ir.IntegerType.get_signless(8))]
                            ).results[0]
                    def cache_indices(slot, field):
                        return [slot, constant(field)] if invocation_shared else [slot]
                    def initialize_keys():
                        initialize = create("scf.for", [zero, constant(entries), one], regions=1)
                        initialize_block = ir.Block.create_at_start(initialize.regions[0], [index])
                        with ir.InsertionPoint(initialize_block):
                            create(
                                "memref.store",
                                [constant(0x7FC00000, i32), keys, *cache_indices(initialize_block.arguments[0], 0)],
                            )
                            create("scf.yield")
                    if previous is None:
                        initialize_keys()
                    else:
                        checked = create("scf.if", [environment_valid], [i1], regions=2)
                        for branch in range(2):
                            block = ir.Block.create_at_start(checked.regions[branch], [])
                            with ir.InsertionPoint(block):
                                valid = constant(0, i1)
                                if branch == 0:
                                    status = create(
                                        "func.call", [environment], [i32],
                                        {"callee": ir.FlatSymbolRefAttr.get("__merlin_exp_environment_matches")},
                                    ).results[0]
                                    valid = create(
                                        "arith.cmpi", [status, constant(0, i32)], [i1],
                                        {"predicate": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 1)},
                                    ).results[0]
                                create("scf.yield", [valid])
                        clear = create("scf.if", [checked.results[0]], regions=2)
                        for branch in range(2):
                            block = ir.Block.create_at_start(clear.regions[branch], [])
                            with ir.InsertionPoint(block):
                                if branch:
                                    initialize_keys()
                                create("scf.yield")

                def original_exp(operands, attributes):
                    if entries and errno_checked:
                        return create(
                            "func.call",
                            operands,
                            [f32],
                            {**attributes, "callee": ir.FlatSymbolRefAttr.get("__merlin_exp_observe_errno")},
                        ).results[0]
                    return create("math.exp", operands, [f32], attributes).results[0]

                def lookup_exp(word, operands, attributes):
                    if not entries:
                        return original_exp(operands, attributes)
                    key_word = word
                    if underflow_class is not None:
                        minimum, maximum = underflow_class
                        lower = create(
                            "arith.cmpi",
                            [word, constant(minimum, i32)],
                            [i1],
                            {"predicate": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 9)},
                        ).results[0]
                        upper = create(
                            "arith.cmpi",
                            [word, constant(maximum, i32)],
                            [i1],
                            {"predicate": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 7)},
                        ).results[0]
                        member = create("arith.andi", [lower, upper], [i1]).results[0]
                        key_word = create("arith.select", [member, constant(minimum, i32), word], [i32]).results[0]
                    absolute = create("arith.andi", [word, constant(0x7FFFFFFF, i32)], [i32]).results[0]
                    bounded = create(
                        "arith.cmpi",
                        [absolute, constant(0x7F800000 if errno_checked else 0x42800000, i32)],
                        [i1],
                        {"predicate": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 7)},
                    ).results[0]
                    selected = create("scf.if", [bounded], [f32], regions=2)
                    for branch in range(2):
                        block = ir.Block.create_at_start(selected.regions[branch], [])
                        with ir.InsertionPoint(block):
                            if branch:
                                result = original_exp(operands, attributes)
                            else:
                                # Fold upper binary32 fraction bits with sign/exponent for the shared table.
                                fraction_shift = 23 - (entries.bit_length() - 1) if invocation_shared else 8
                                low = create(
                                    "arith.shrui", [key_word, constant(fraction_shift, i32)], [i32]
                                ).results[0]
                                high = create(
                                    "arith.shrui", [key_word, constant(23 if invocation_shared else 16, i32)], [i32]
                                ).results[0]
                                mixed = create("arith.xori", [low, high], [i32]).results[0]
                                slot_word = create(
                                    "arith.andi",
                                    [mixed, constant(entries - 1, i32)],
                                    [i32],
                                ).results[0]
                                slot = create("arith.index_cast", [slot_word], [index]).results[0]
                                stored = create("memref.load", [keys, *cache_indices(slot, 0)], [i32]).results[0]
                                hit = create(
                                    "arith.cmpi",
                                    [key_word, stored],
                                    [i1],
                                    {"predicate": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 0)},
                                ).results[0]
                                if errno_checked:
                                    matched = create("scf.if", [hit], [i1], regions=2)
                                    for effect_branch in range(2):
                                        effect_block = ir.Block.create_at_start(matched.regions[effect_branch], [])
                                        with ir.InsertionPoint(effect_block):
                                            effect = constant(0, i1)
                                            if effect_branch == 0:
                                                previous_errno = create(
                                                    "memref.load", [errnos, *cache_indices(slot, 2)], [i32]
                                                ).results[0]
                                                current_errno = create("llvm.load", [errno_pointer], [i32]).results[0]
                                                effect = create(
                                                    "arith.cmpi",
                                                    [previous_errno, current_errno],
                                                    [i1],
                                                    {
                                                        "predicate": ir.IntegerAttr.get(
                                                            ir.IntegerType.get_signless(64), 0
                                                        )
                                                    },
                                                ).results[0]
                                            create("scf.yield", [effect])
                                    hit = matched.results[0]
                                cached_exp = create("scf.if", [hit], [f32], regions=2)
                                for cache_branch in range(2):
                                    cache_block = ir.Block.create_at_start(cached_exp.regions[cache_branch], [])
                                    with ir.InsertionPoint(cache_block):
                                        if cache_branch:
                                            value = original_exp(operands, attributes)
                                            create("memref.store", [value, values, *cache_indices(slot, 1)])
                                            if errno_checked:
                                                error = create("llvm.load", [errno_pointer], [i32]).results[0]
                                                create("memref.store", [error, errnos, *cache_indices(slot, 2)])
                                            create("memref.store", [key_word, keys, *cache_indices(slot, 0)])
                                        else:
                                            value = create(
                                                "memref.load", [values, *cache_indices(slot, 1)], [f32]
                                            ).results[0]
                                        create("scf.yield", [value])
                                result = cached_exp.results[0]
                            create("scf.yield", [result])
                    return selected.results[0]

                def loop(depth, coordinates, state):
                    if depth < len(extents):
                        if input_key is not None and depth == len(extents) - 1:
                            state = [state[0], constant(0, i1), state[2], state[3]]
                        body = create(
                            "scf.for",
                            [zero, constant(extents[depth]), one, *state],
                            state_types,
                            attributes=({"loop_annotation": ir.Attribute.parse(
                                "#llvm.loop_annotation<unroll = <count = " + str(inner_unroll) + " : i32>>")}
                                if inner_unroll and depth == len(extents) - 1 else {}),
                            regions=1,
                        )
                        block = ir.Block.create_at_start(body.regions[0], [index, *state_types])
                        with ir.InsertionPoint(block):
                            create(
                                "scf.yield",
                                loop(
                                    depth + 1,
                                    [*coordinates, block.arguments[0]],
                                    list(block.arguments[1:]),
                                ),
                            )
                        return list(body.results)
                    mapping = {}
                    for argument, value, axes in zip(original_block.arguments[:-1], old.operands[:-1], positions[:-1]):
                        mapping[argument] = create(
                            "tensor.extract",
                            [
                                value,
                                *[zero if axis is None else coordinates[axis] for axis in axes],
                            ],
                            [argument.type],
                        ).results[0]
                    destination, valid, key, cached = state
                    if input_key is not None:
                        word = create("arith.bitcast", [mapping[original_block.arguments[input_key]]], [i32]).results[0]
                        same = create("arith.cmpi", [word, key], [i1],
                            {"predicate": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 0)}).results[0]
                        hit = create("arith.andi", [valid, same], [i1]).results[0]
                        repeated = create("scf.if", [hit], [f32], regions=2)
                        for branch in range(2):
                            block = ir.Block.create_at_start(repeated.regions[branch], [])
                            with ir.InsertionPoint(block):
                                value = cached
                                if branch:
                                    body_mapping = dict(mapping)
                                    for scalar in list(original_block.operations)[:-1]:
                                        operands = [body_mapping.get(v, v) for v in scalar.operands]
                                        attributes = {name: scalar.attributes[name] for name in scalar.attributes}
                                        if scalar.operation.name == "math.exp":
                                            exp_word = create("arith.bitcast", operands, [i32]).results[0]
                                            value = lookup_exp(exp_word, operands, attributes)
                                            body_mapping[scalar.results[0]] = value
                                        else:
                                            clone = create(scalar.operation.name, operands,
                                                [v.type for v in scalar.results], attributes)
                                            body_mapping.update(zip(scalar.results, clone.results))
                                    value = body_mapping[list(original_block.operations)[-1].operands[0]]
                                create("scf.yield", [value])
                        result = repeated.results[0]
                        destination = create("tensor.insert",
                            [result, destination, *[coordinates[x] for x in positions[-1]]],
                            [state_types[0]]).results[0]
                        return [destination, constant(1, i1), word, result]
                    for scalar in list(original_block.operations)[:-1]:
                        operands = [mapping.get(value, value) for value in scalar.operands]
                        attributes = {name: scalar.attributes[name] for name in scalar.attributes}
                        if scalar.operation.name != "math.exp":
                            clone = create(
                                scalar.operation.name,
                                operands,
                                [v.type for v in scalar.results],
                                attributes,
                            )
                        else:
                            word = create("arith.bitcast", operands, [i32]).results[0]
                            same = create(
                                "arith.cmpi",
                                [word, key],
                                [i1],
                                {"predicate": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 0)},
                            ).results[0]
                            hit = create("arith.andi", [valid, same], [i1]).results[0]
                            clone = create("scf.if", [hit], [f32], regions=2)
                            for branch in range(2):
                                block = ir.Block.create_at_start(clone.regions[branch], [])
                                with ir.InsertionPoint(block):
                                    result = cached if branch == 0 else lookup_exp(word, operands, attributes)
                                    create("scf.yield", [result])
                            key, cached, valid = word, clone.results[0], constant(1, i1)
                        mapping.update(zip(scalar.results, clone.results))
                    value = mapping[list(original_block.operations)[-1].operands[0]]
                    destination = create(
                        "tensor.insert",
                        [value, destination, *[coordinates[x] for x in positions[-1]]],
                        [state_types[0]],
                    ).results[0]
                    return [destination, valid, key, cached]

                result = loop(0, [], state)[0]
                for name in old.attributes:
                    if name.startswith("prov."):
                        result.owner.attributes[name] = old.attributes[name]
                previous = (
                    ir.StringAttr(old.attributes["prov.transforms"]).value + ","
                    if "prov.transforms" in old.attributes
                    else ""
                )
                result.owner.attributes["prov.transforms"] = ir.StringAttr.get(
                    previous
                    + (
                        "memoize_invocation_exp_input_words"
                        if input_key is not None
                        else "memoize_invocation_selected_expf"
                        if entries and group is not None
                        else "memoize_selected_expf_underflow_class"
                        if entries and underflow_class is not None
                        else "memoize_errno_checked_exp_arguments"
                        if entries and errno_checked
                        else "memoize_bounded_exp_arguments_128"
                        if entries == 128
                        else "memoize_bounded_exp_arguments"
                        if entries
                        else "memoize_adjacent_exp_arguments"
                    )
                )
                if entries and group is not None:
                    captured = create(
                        "func.call", [environment], [i32],
                        {"callee": ir.FlatSymbolRefAttr.get("__merlin_exp_environment_capture")},
                    ).results[0]
                    valid = create(
                        "arith.cmpi", [captured, constant(0, i32)], [i1],
                        {"predicate": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 1)},
                    ).results[0]
                    shared[group] = (entries, keys, values, errnos, errno_pointer, environment, valid)
                old.results[0].replace_all_uses_with(result)
                old.erase()
    module.operation.verify()
    return len(plans)


_AE_ORIGINAL_RUN_STAGES = _run_stages


def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(), pre_generalize=()):
    passes = [p for p in pipeline.split(",") if p]
    markers = {
        "__merlin_memoize_adjacent_exp_arguments__": (0, False),
        "__merlin_memoize_bounded_exp_arguments__": (64, False),
        "__merlin_memoize_bounded_exp_arguments_128__": (128, False),
        "__merlin_memoize_errno_checked_exp_arguments__": (128, True),
        "__merlin_memoize_selected_expf_underflow_class__": (128, True),
        "__merlin_memoize_invocation_selected_expf__": (1024, True),
        "__merlin_memoize_invocation_exp_input_words__": (1024, True),
    }
    unroll_counts = {"__merlin_unroll_exp_input_cache_2__": 2, "__merlin_unroll_exp_input_cache_4__": 4}
    unroll_hints = [p for p in passes if p in unroll_counts]
    if len(unroll_hints) > 1:
        raise ValueError("exact exp input unrolling must be unique")
    passes = [p for p in passes if p not in unroll_counts]
    selected = [p for p in passes if p in markers]
    if unroll_hints and selected != ["__merlin_memoize_invocation_exp_input_words__"]:
        raise ValueError("exact exp unrolling requires its input-word cache stage")
    if not selected:
        return _AE_ORIGINAL_RUN_STAGES(ctx, module, pipeline, erase, mid, late, post_openmp, pre_generalize)
    if len(selected) != 1:
        raise ValueError("exp memoization stage must be unique")
    marker = selected[0]
    entries, errno_checked = markers[marker]
    class_bounds = None
    if marker in ("__merlin_memoize_selected_expf_underflow_class__",
                  "__merlin_memoize_invocation_selected_expf__",
                  "__merlin_memoize_invocation_exp_input_words__"):
        if "_AE_SELECTED_EXPF_CLASS" not in globals():
            raise ValueError("missing compiler-bound selected expf equivalence")
        class_bounds = _AE_SELECTED_EXPF_CLASS
    i = passes.index(marker)
    _AE_ORIGINAL_RUN_STAGES(ctx, module, ",".join(passes[:i]), 0, (), (), (), pre_generalize)
    print("OK exact_exp_memoization", _memoize_adjacent_tensor_exp(
        ctx, module, entries, errno_checked, class_bounds,
        marker in ("__merlin_memoize_invocation_selected_expf__", "__merlin_memoize_invocation_exp_input_words__"),
        marker == "__merlin_memoize_invocation_exp_input_words__", unroll_counts[unroll_hints[0]] if unroll_hints else 0
    ))
    return _AE_ORIGINAL_RUN_STAGES(ctx, module, ",".join(passes[i + 1 :]), erase, mid, late, post_openmp, ())

def _mc_shape(value, element="f32"):
    from torch_mlir import ir
    if not isinstance(value.type, ir.RankedTensorType):
        return None
    tensor = ir.RankedTensorType(value.type)
    shape = list(tensor.shape)
    if str(tensor.element_type) != element or any(d < 0 for d in shape):
        return None
    return shape

def _mc_attrs(op, allowed):
    return all(name in allowed or name.startswith("prov.") for name in op.attributes)

def _mc_single(value):
    uses = list(value.uses)
    return uses[0].owner.operation if len(uses) == 1 else None

def _mc_maps(op):
    from torch_mlir import ir
    if "indexing_maps" not in op.attributes:
        return None
    maps = [ir.AffineMapAttr(a).value for a in op.attributes["indexing_maps"]]
    if not maps or any(m.n_symbols or m.n_dims != maps[0].n_dims for m in maps):
        return None
    result = []
    for affine in maps:
        row = []
        for expr in affine.results:
            if isinstance(expr, ir.AffineDimExpr):
                row.append(("dim", ir.AffineDimExpr(expr).position))
            elif isinstance(expr, ir.AffineConstantExpr) and ir.AffineConstantExpr(expr).value == 0:
                row.append(("zero", 0))
            else:
                return None
        result.append(row)
    return maps[0].n_dims, result

def _mc_parallel(op):
    from torch_mlir import ir
    if (op.name != "linalg.generic" or len(op.results) != 1 or len(op.regions) != 1
            or len(op.regions[0].blocks) != 1 or not _mc_attrs(op,
                {"indexing_maps", "iterator_types", "operandSegmentSizes"})):
        return None
    maps = _mc_maps(op)
    if maps is None:
        return None
    dims, positions = maps
    if [str(i) for i in op.attributes["iterator_types"]] != ["#linalg.iterator_type<parallel>"] * dims:
        return None
    if positions[-1] != [("dim", d) for d in range(dims)]:
        return None
    if list(op.regions[0].blocks[0].arguments[-1].uses):
        return None
    return positions

def _mc_before(value, operation):
    from torch_mlir import ir
    if isinstance(value, ir.BlockArgument):
        # The exact enclosing block's argument dominates; nested capture is
        # deliberately unsupported rather than inventing dominance.
        return value.owner == operation.block
    owner = value.owner.operation
    if owner.block != operation.block:
        return False
    for current in operation.block.operations:
        if current.operation == owner:
            return True
        if current.operation == operation:
            return False
    return False

def _mc_mask_source(value, operation):
    from torch_mlir import ir
    if _mc_before(value, operation):
        return value, False
    if isinstance(value, ir.BlockArgument):
        return None
    invert = value.owner.operation
    if not isinstance(invert, ir.Operation) or invert.block != operation.block:
        return None
    positions = _mc_parallel(invert)
    if (positions is None or len(invert.operands) != 2 or positions[0] != positions[1]
            or _mc_shape(value, "i1") != _mc_shape(invert.operands[0], "i1")):
        return None
    block = invert.regions[0].blocks[0]
    body = list(block.operations)
    if ([o.operation.name for o in body] != ["arith.constant", "arith.xori", "linalg.yield"]
            or any(not _mc_attrs(o.operation, {"value"}) for o in body[:1])
            or any(not _mc_attrs(o.operation, set()) for o in body[1:])):
        return None
    literal, xor, yield_op = body
    if (str(literal.results[0].type) != "i1" or ir.IntegerAttr(literal.attributes["value"]).value == 0
            or list(xor.operands) not in ([block.arguments[0], literal.results[0]],
                                        [literal.results[0], block.arguments[0]])
            or list(yield_op.operands) != [xor.results[0]]
            or not _mc_before(invert.operands[0], operation)):
        return None
    return invert.operands[0], True

def _mc_spec(op):
    from torch_mlir import ir
    source = _scalar_contraction_spec(op)
    if source is None or not _mc_attrs(op, {"indexing_maps", "iterator_types", "operandSegmentSizes"}):
        return None
    owner = op
    while owner is not None:
        if any("strictfp" in name or "strictfp" in str(owner.attributes[name]) for name in owner.attributes):
            return None
        owner = owner.parent
    value = op.results[0]
    original_shape = _mc_shape(value)
    if original_shape is None or any(d == 0 for d in original_shape):
        return None
    current_shape = original_shape
    consumer = _mc_single(value)
    # Support exact full row-major flatten/unflatten. General reassociation or
    # permutation refuses; total element equality alone is insufficient.
    while consumer is not None and consumer.name in ("tensor.collapse_shape", "tensor.expand_shape"):
        if (len(consumer.operands) != 1 or len(consumer.results) != 1
                or consumer.block != op.block
                or not _mc_attrs(consumer, {"reassociation", "static_output_shape"})):
            return None
        next_shape = _mc_shape(consumer.results[0])
        if next_shape is None or any(d == 0 for d in next_shape):
            return None
        reassociation = [[ir.IntegerAttr(i).value for i in group]
                         for group in consumer.attributes["reassociation"]]
        larger = len(current_shape) if consumer.name == "tensor.collapse_shape" else len(next_shape)
        smaller = len(next_shape) if consumer.name == "tensor.collapse_shape" else len(current_shape)
        if smaller != 1 or reassociation != [list(range(larger))]:
            return None
        product = lambda s: __import__("math").prod(s)
        if product(current_shape) != product(next_shape):
            return None
        value, current_shape = consumer.results[0], next_shape
        consumer = _mc_single(value)
    positions = _mc_parallel(consumer) if consumer is not None else None
    if positions is None or len(consumer.operands) != 3 or _mc_shape(consumer.results[0]) != current_shape:
        return None
    arguments = list(consumer.regions[0].blocks[0].arguments)
    body = list(consumer.regions[0].blocks[0].operations)
    if ([o.operation.name for o in body] != ["arith.mulf", "linalg.yield"]
            or any(not _mc_attrs(o.operation, {"fastmath"}) for o in body)
            or any("fastmath" in o.attributes and str(o.attributes["fastmath"]) != "#arith.fastmath<none>" for o in body)
            or list(body[1].operands) != [body[0].results[0]]):
        return None
    lane = next((i for i in range(2) if consumer.operands[i] == value), None)
    if (lane is None or positions[lane] != positions[-1]
            or list(body[0].operands) not in ([arguments[0], arguments[1]], [arguments[1], arguments[0]])):
        return None
    # Source coefficient is a rank-zero tensor. It remains computed at its
    # original position; no division/scale rewrite or assumption is granted.
    if _mc_shape(consumer.operands[1-lane]) != [] or positions[1-lane] != []:
        return None
    scaled = consumer.results[0]
    select = _mc_single(scaled)
    positions = _mc_parallel(select) if select is not None else None
    if (positions is None or select.block != op.block or len(select.operands) != 4
            or _mc_shape(select.results[0]) != current_shape):
        return None
    block = select.regions[0].blocks[0]
    body = list(block.operations)
    if ([o.operation.name for o in body] != ["arith.select", "linalg.yield"]
            or any(not _mc_attrs(o.operation, set()) for o in body)
            or list(body[1].operands) != [body[0].results[0]]):
        return None
    choices = list(body[0].operands)
    if any(v not in list(block.arguments)[:3] for v in choices) or choices[1] == choices[2]:
        return None
    condition = list(block.arguments).index(choices[0])
    chosen = next((i for i in (1,2) if select.operands[list(block.arguments).index(choices[i])] == scaled), None)
    if condition is None or chosen is None:
        return None
    scaled_position = list(block.arguments).index(choices[chosen])
    if positions[scaled_position] != positions[-1]:
        return None
    mask = select.operands[condition]
    mask_shape = _mc_shape(mask, "i1")
    if mask_shape is None or len(mask_shape) != len(positions[condition]):
        return None
    for mapping, extent in zip(positions[condition], mask_shape):
        if (mapping[0] == "zero" and extent != 1
                or mapping[0] == "dim" and extent != current_shape[mapping[1]]):
            return None
    resolved = _mc_mask_source(mask, op)
    if resolved is None:
        return None
    raw, inverted = resolved
    return raw, positions[condition], current_shape, (chosen == 1) != inverted

def _scalarize_closed_mask_contractions(ctx, module, *, outputs=1, rows=1, reduction_unroll=0):
    from torch_mlir import ir
    plans = []
    for op in module.body.operations:
        def walk(operation):
            for region in operation.regions:
                for block in region.blocks:
                    for child in list(block.operations):
                        spec = _mc_spec(child.operation)
                        if spec is not None:
                            plans.append((child.operation, spec))
                        else:
                            walk(child.operation)
        walk(op.operation)
    def guard(operation, lanes, output_positions, create, constant):
        raw, mask_positions, observed_shape, true_observed = next(spec for old, spec in plans if old == operation)
        source_shape = _mc_shape(operation.results[0])
        conditions = []
        for lane in lanes:
            source_indices = [lane[d] for d in output_positions]
            flat = source_indices[0]
            for coordinate, extent in zip(source_indices[1:], source_shape[1:]):
                flat = create("arith.addi", [create("arith.muli", [flat, constant(extent)],
                    [ir.IndexType.get()]).results[0], coordinate], [ir.IndexType.get()]).results[0]
            observed_indices = []
            for number, extent in enumerate(observed_shape):
                stride = __import__("math").prod(observed_shape[number+1:])
                coordinate = flat if stride == 1 else create("arith.divui", [flat, constant(stride)], [ir.IndexType.get()]).results[0]
                if number != 0:
                    coordinate = create("arith.remui", [coordinate, constant(extent)], [ir.IndexType.get()]).results[0]
                observed_indices.append(coordinate)
            coordinates = [constant(0) if kind == "zero" else observed_indices[position] for kind, position in mask_positions]
            condition = create("tensor.extract", [raw, *coordinates], [ir.IntegerType.get_signless(1)]).results[0]
            if not true_observed:
                one = create("arith.constant", results=[ir.IntegerType.get_signless(1)],
                    attributes={"value": ir.IntegerAttr.get(ir.IntegerType.get_signless(1), 1)}).results[0]
                condition = create("arith.xori", [condition, one], [ir.IntegerType.get_signless(1)]).results[0]
            conditions.append(condition)
        combined = conditions[0]
        for condition in conditions[1:]:
            combined = create("arith.ori", [combined, condition], [ir.IntegerType.get_signless(1)]).results[0]
        return combined
    return _scalarize_tensor_contractions(ctx, module, outputs=outputs, rows=rows,
        reduction_unroll=reduction_unroll, selection=[op for op,_ in plans], output_guard=guard)

_MC_MARKER = "__merlin_scalar_contraction_masked_observation__"
_MC_ORIG_RUN_STAGES = _run_stages

def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(), pre_generalize=()):
    passes = [p for p in pipeline.split(',') if p]
    if _MC_MARKER not in passes:
        return _MC_ORIG_RUN_STAGES(ctx, module, pipeline, erase, mid, late, post_openmp, pre_generalize)
    if len(sys.argv) <= 21 or sys.argv[21] != '1':
        raise ValueError("closed-mask scheduling requires explicit arithmetic effect permission")
    selected = [m for m in _SC_MARKERS if m in passes]
    if len(selected) != 1 or passes.count(_MC_MARKER) != 1:
        raise ValueError("closed-mask scheduling requires one scalar schedule")
    outputs = {_SC_MARKER: 1, _SC2_MARKER: 2, _SC4_MARKER: 4, _SC8_MARKER: 8, _SC2X4_MARKER: 4}[selected[0]]
    rows = 2 if selected[0] == _SC2X4_MARKER else 1
    hints = [m for m in _SC_UNROLL_MARKERS if m in passes]
    if len(hints) > 1:
        raise ValueError("scalar contraction reduction unroll counts are alternatives")
    count = _scalarize_closed_mask_contractions(ctx, module, outputs=outputs, rows=rows,
        reduction_unroll=_SC_UNROLL_MARKERS[hints[0]] if hints else 0)
    print('OK scalar_contraction_masked_observation', count)
    return _MC_ORIG_RUN_STAGES(ctx, module, ','.join(p for p in passes if p != _MC_MARKER),
        erase, mid, late, post_openmp, pre_generalize)

def _sqs_spec(op):
    from torch_mlir import ir as _sqs_ir
    if (op.operation.name != "linalg.generic" or len(op.operands) != 2
            or len(op.results) != 1 or len(op.regions) != 1
            or len(op.regions[0].blocks) != 1):
        return None
    ancestor = op.operation
    while ancestor is not None:
        if any("strictfp" in name or "strictfp" in str(ancestor.attributes[name])
               for name in ancestor.attributes):
            return None
        ancestor = ancestor.parent
    types = []
    for value in op.operands:
        if not isinstance(value.type, _sqs_ir.RankedTensorType):
            return None
        tensor = _sqs_ir.RankedTensorType(value.type)
        if (not isinstance(tensor.element_type, _sqs_ir.F32Type)
                or any(extent < 0 for extent in tensor.shape)):
            return None
        types.append(tensor)
    if op.results[0].type != op.operands[1].type:
        return None
    allowed_attributes = {"indexing_maps", "iterator_types", "operandSegmentSizes"}
    if any(name not in allowed_attributes and not name.startswith("prov.")
           for name in op.attributes):
        return None
    block = op.regions[0].blocks[0]
    body = list(block.operations)
    if (len(block.arguments) != 2 or len(body) != 3
            or [x.operation.name for x in body]
                != ["arith.mulf", "arith.addf", "linalg.yield"]
            or any(x.regions for x in body)
            or any(name != "fastmath" and not name.startswith("prov.")
                   for scalar in body for name in scalar.attributes)
            or any("fastmath" in x.attributes
                   and str(x.attributes["fastmath"]) != "#arith.fastmath<none>"
                   for x in body)):
        return None
    mul, add, terminator = body
    value, seed = block.arguments
    if list(mul.operands) != [value, value]:
        return None
    if list(add.operands) == [mul.results[0], seed]:
        order = [0, 1]
    elif list(add.operands) == [seed, mul.results[0]]:
        order = [1, 0]
    else:
        return None
    if list(terminator.operands) != [add.results[0]]:
        return None
    maps = [_sqs_ir.AffineMapAttr(a).value for a in op.attributes["indexing_maps"]]
    if len(maps) != 2:
        return None
    dims = maps[0].n_dims
    if (dims < 1 or any(m.n_dims != dims or m.n_symbols for m in maps)
            or [str(x) for x in op.attributes["iterator_types"]]
                != ["#linalg.iterator_type<parallel>"] * (dims - 1)
                   + ["#linalg.iterator_type<reduction>"]):
        return None
    extents, positions = [None] * dims, []
    for affine, tensor in zip(maps, types):
        if len(affine.results) != len(tensor.shape):
            return None
        used = []
        for expr, extent in zip(affine.results, tensor.shape):
            if not isinstance(expr, _sqs_ir.AffineDimExpr):
                return None
            position = _sqs_ir.AffineDimExpr(expr).position
            if (position in used or position >= dims
                    or (extents[position] is not None and extents[position] != extent)):
                return None
            used.append(position)
            extents[position] = extent
        positions.append(used)
    if (sorted(positions[0]) != list(range(dims))
            or sorted(positions[1]) != list(range(dims - 1))
            or any(extent is None for extent in extents)):
        return None
    return positions, extents, order


def _scalarize_squared_sums(ctx, module):
    from torch_mlir import ir as _sqs_ir
    todo = []
    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for child in list(block.operations):
                    spec = _sqs_spec(child)
                    if spec is not None:
                        todo.append((child, spec))
                    else:
                        walk(child.operation)
    walk(module.operation)
    with ctx:
        for old, (positions, extents, add_order) in todo:
            with old.location, _sqs_ir.InsertionPoint(old):
                index = _sqs_ir.IndexType.get()
                scalar = old.regions[0].blocks[0].arguments[0].type
                tensor = old.results[0].type
                def create(name, operands=(), results=(), attributes=None, regions=0):
                    return _sqs_ir.Operation.create(name, operands=list(operands),
                        results=list(results), attributes=attributes or {}, regions=regions)
                def constant(value):
                    return create("arith.constant", results=[index], attributes={
                        "value": _sqs_ir.IntegerAttr.get(index, value)}).results[0]
                zero, one = constant(0), constant(1)
                limits = [constant(extent) for extent in extents]
                source_mul, source_add = list(old.regions[0].blocks[0].operations)[:2]
                def output_loop(depth, current, coordinates):
                    if depth < len(extents) - 1:
                        loop = create("scf.for", [zero, limits[depth], one, current],
                                      [tensor], regions=1)
                        block = _sqs_ir.Block.create_at_start(loop.regions[0], [index, tensor])
                        with _sqs_ir.InsertionPoint(block):
                            updated = output_loop(depth + 1, block.arguments[1],
                                                  [*coordinates, block.arguments[0]])
                            create("scf.yield", [updated])
                        return loop.results[0]
                    out_indices = [coordinates[d] for d in positions[1]]
                    initial = create("tensor.extract", [current, *out_indices], [scalar]).results[0]
                    loop = create("scf.for", [zero, limits[-1], one, initial],
                                  [scalar], regions=1)
                    block = _sqs_ir.Block.create_at_start(loop.regions[0], [index, scalar])
                    with _sqs_ir.InsertionPoint(block):
                        all_indices = [*coordinates, block.arguments[0]]
                        value = create("tensor.extract", [old.operands[0],
                            *[all_indices[d] for d in positions[0]]], [scalar]).results[0]
                        product = create("arith.mulf", [value, value], [scalar],
                            {name: source_mul.attributes[name] for name in source_mul.attributes}).results[0]
                        values = [product, block.arguments[1]]
                        total = create("arith.addf", [values[d] for d in add_order], [scalar],
                            {name: source_add.attributes[name] for name in source_add.attributes}).results[0]
                        create("scf.yield", [total])
                    return create("tensor.insert", [loop.results[0], current, *out_indices],
                                  [tensor]).results[0]
                result = output_loop(0, old.operands[1], [])
                for name in old.attributes:
                    if name.startswith("prov."):
                        result.owner.attributes[name] = old.attributes[name]
                previous = old.attributes["prov.transforms"] if "prov.transforms" in old.attributes else None
                prefix = _sqs_ir.StringAttr(previous).value + "," if previous is not None else ""
                result.owner.attributes["prov.transforms"] = _sqs_ir.StringAttr.get(
                    prefix + "scalar_squared_sum_accumulator")
            old.results[0].replace_all_uses_with(result)
            old.operation.erase()
    module.operation.verify()
    return len(todo)

_SQS_MARKER = "__merlin_scalar_squared_sum_accumulator__"
_SQS_ORIG_RUN_STAGES = _run_stages

def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(), pre_generalize=()):
    passes = [p for p in pipeline.split(',') if p]
    if _SQS_MARKER not in passes:
        return _SQS_ORIG_RUN_STAGES(ctx, module, pipeline, erase, mid, late, post_openmp, pre_generalize)
    if passes.count(_SQS_MARKER) != 1:
        raise ValueError("scalar squared sum marker must occur once")
    at = passes.index(_SQS_MARKER)
    _SQS_ORIG_RUN_STAGES(ctx, module, ','.join(passes[:at]), 0, (), (), (), pre_generalize)
    print('OK scalar_squared_sum_accumulator', _scalarize_squared_sums(ctx, module))
    _SQS_ORIG_RUN_STAGES(ctx, module, ','.join(passes[at + 1:]), erase, mid, late, post_openmp, ())

class _OrderedReductionDomains:
    def __init__(self):
        self.memo = {}

    def ordinary(self, op, allowed=()):
        return all(
            name in allowed
            or name.startswith("prov.")
            or name == "fastmath"
            and str(op.attributes[name]) == "#arith.fastmath<none>"
            for name in op.attributes
        )

    def constant(self, attr):
        from torch_mlir import ir
        import math

        if not isinstance(attr, ir.FloatAttr):
            return None
        value = ir.FloatAttr(attr).value
        if math.isnan(value):
            return None
        return (math.isfinite(value), value >= 0, abs(value), value)

    def scalar(self, value, bindings, depth):
        from torch_mlir import ir

        if value in bindings:
            return bindings[value]
        if depth > 64 or not isinstance(value.type, ir.F32Type):
            return None
        op = value.owner
        if isinstance(op, ir.OpView):
            op = op.operation
        if not isinstance(op, ir.Operation):
            return None
        if op.name == "arith.constant" and self.ordinary(op, {"value"}):
            return self.constant(op.attributes["value"])
        if not self.ordinary(op):
            return None
        if op.name in {"arith.sitofp", "arith.uitofp"} and len(op.operands) == 1:
            typ = op.operands[0].type
            if isinstance(typ, ir.IntegerType) and 1 <= ir.IntegerType(typ).width <= 64:
                return (
                    True,
                    op.name == "arith.uitofp",
                    float(2 ** ir.IntegerType(typ).width),
                    None,
                )
            return None
        if op.name == "arith.select" and len(op.operands) == 3:
            a, b = [self.scalar(v, bindings, depth + 1) for v in op.operands[1:]]
            return (
                None
                if a is None or b is None
                else (a[0] and b[0], a[1] and b[1], max(a[2], b[2]), None)
            )
        if op.name == "math.exp" and len(op.operands) == 1:
            a = self.scalar(op.operands[0], bindings, depth + 1)
            return None if a is None else (False, True, float("inf"), None)
        if len(op.operands) != 2 or op.name not in {
            "arith.mulf",
            "arith.divf",
            "arith.addf",
            "arith.subf",
        }:
            return None
        a, b = [self.scalar(v, bindings, depth + 1) for v in op.operands]
        if a is None or b is None:
            return None
        if op.name == "arith.addf" and a[1] and b[1]:
            bound = a[2] + b[2]
            return (
                a[0] and b[0] and bound < 1.7e38,
                True,
                max(2 * bound, 2.0**-149),
                None,
            )
        if not a[0] or not b[0]:
            return None
        if op.name == "arith.divf":
            if b[3] is None or b[3] <= 0:
                return None
            bound, nonnegative = a[2] / b[3], a[1]
        elif op.name == "arith.mulf":
            bound, nonnegative = a[2] * b[2], a[1] and b[1]
        else:
            bound, nonnegative = a[2] + b[2], op.name == "arith.addf" and a[1] and b[1]
        return (bound < 1.7e38, nonnegative, max(2 * bound, 2.0**-149), None)

    def tensor(self, value, depth=0):
        from torch_mlir import ir

        if value in self.memo:
            return self.memo[value]
        if depth > 64 or len(self.memo) > 8192:
            return None
        self.memo[value] = None
        op = value.owner
        if isinstance(op, ir.OpView):
            op = op.operation
        if not isinstance(op, ir.Operation) or not isinstance(
            value.type, ir.RankedTensorType
        ):
            return None
        typ = ir.RankedTensorType(value.type)
        if not isinstance(typ.element_type, ir.F32Type):
            return None
        result = None
        if op.name == "arith.constant" and self.ordinary(op, {"value"}):
            attr = op.attributes["value"]
            if isinstance(attr, ir.DenseFPElementsAttr) and attr.is_splat:
                result = self.constant(attr.get_splat_value())
        elif op.name == "tensor.splat" and self.ordinary(op):
            result = self.scalar(op.operands[0], {}, depth + 1)
        elif op.name == "linalg.fill" and self.ordinary(op, {"operandSegmentSizes"}):
            result = self.scalar(op.operands[0], {}, depth + 1)
        elif op.name in {
            "tensor.cast",
            "tensor.extract_slice",
            "tensor.collapse_shape",
            "tensor.expand_shape",
        }:
            if self.ordinary(
                op,
                {
                    "static_offsets",
                    "static_sizes",
                    "static_strides",
                    "operandSegmentSizes",
                    "reassociation",
                    "static_output_shape",
                },
            ):
                result = self.tensor(op.operands[0], depth + 1)
        elif (
            op.name in {"linalg.generic", "linalg.reduce"}
            and len(op.results) == 1
            and len(op.regions) == 1
            and len(op.regions[0].blocks) == 1
        ):
            block = op.regions[0].blocks[0]
            body = list(block.operations)
            allowed = (
                {"dimensions"}
                if op.name == "linalg.reduce"
                else {"indexing_maps", "iterator_types", "operandSegmentSizes"}
            )
            if (
                not self.ordinary(op, allowed)
                or not body
                or body[-1].operation.name != "linalg.yield"
            ):
                return None
            iterators = (
                ["reduction"]
                if op.name == "linalg.reduce"
                else [str(x) for x in op.attributes["iterator_types"]]
            )
            pure = {
                "arith.constant",
                "arith.sitofp",
                "arith.uitofp",
                "arith.select",
                "arith.addf",
                "arith.subf",
                "arith.mulf",
                "arith.divf",
                "arith.cmpi",
                "arith.cmpf",
                "linalg.index",
                "math.exp",
            }
            if all(x == "#linalg.iterator_type<parallel>" for x in iterators):
                if not list(block.arguments[-1].uses) and all(
                    not child.regions and child.operation.name in pure
                    for child in body[:-1]
                ):
                    bindings = {
                        arg: self.tensor(v, depth + 1)
                        for arg, v in zip(block.arguments, op.operands[:-1])
                    }
                    result = self.scalar(body[-1].operands[0], bindings, depth + 1)
            elif (
                len(op.operands) == 2
                and len(block.arguments) == 2
                and len(body) == 2
                and body[0].operation.name in {"arith.maximumf", "arith.maxnumf"}
                and list(body[0].operands)
                in [list(block.arguments), list(block.arguments)[::-1]]
                and list(body[-1].operands) == list(body[0].results)
                and self.ordinary(body[0])
            ):
                a, b = [self.tensor(v, depth + 1) for v in op.operands]
                if a is not None and b is not None:
                    finite, positive, bound = (
                        a[0] and b[0],
                        a[1] and b[1],
                        max(a[2], b[2]),
                    )
                    if (
                        a[0]
                        and b[3] == -float("inf")
                        and all(
                            x > 0
                            for x in ir.RankedTensorType(op.operands[0].type).shape
                        )
                    ):
                        finite, positive, bound = True, a[1], a[2]
                    result = (finite, positive, bound, None)
        self.memo[value] = result
        return result

def _ordered_reduction_spec(op):
    from torch_mlir import ir as _or_ir

    if (
        op.operation.name not in {"linalg.generic", "linalg.reduce"}
        or len(op.operands) != 2
        or len(op.results) != 1
        or len(op.regions) != 1
        or len(op.regions[0].blocks) != 1
    ):
        return None
    ancestor = op.operation
    while ancestor is not None:
        if any(
            "strictfp" in name or "strictfp" in str(ancestor.attributes[name])
            for name in ancestor.attributes
        ):
            return None
        ancestor = ancestor.parent
    types = []
    for value in op.operands:
        if not isinstance(value.type, _or_ir.RankedTensorType):
            return None
        tensor = _or_ir.RankedTensorType(value.type)
        if not isinstance(tensor.element_type, _or_ir.F32Type) or any(
            extent < 0 for extent in tensor.shape
        ):
            return None
        types.append(tensor)
    if op.results[0].type != op.operands[1].type:
        return None
    allowed = (
        {"dimensions"}
        if op.operation.name == "linalg.reduce"
        else {"indexing_maps", "iterator_types", "operandSegmentSizes"}
    )
    if any(
        name not in allowed and not name.startswith("prov.") for name in op.attributes
    ):
        return None
    block = op.regions[0].blocks[0]
    body = list(block.operations)
    if (
        len(block.arguments) != 2
        or len(body) != 2
        or body[0].operation.name
        not in {"arith.addf", "arith.maximumf", "arith.maxnumf"}
        or body[1].operation.name != "linalg.yield"
        or any(x.regions for x in body)
        or any(
            name != "fastmath" and not name.startswith("prov.")
            for scalar in body
            for name in scalar.attributes
        )
        or any(
            "fastmath" in scalar.attributes
            and str(scalar.attributes["fastmath"]) != "#arith.fastmath<none>"
            for scalar in body
        )
    ):
        return None
    scalar, terminator = body
    arguments = list(block.arguments)
    if list(scalar.operands) == arguments:
        order = [0, 1]
    elif list(scalar.operands) == arguments[::-1]:
        order = [1, 0]
    else:
        return None
    if list(terminator.operands) != [scalar.results[0]]:
        return None
    if op.operation.name == "linalg.reduce":
        dims = len(types[0].shape)
        reduced = list(_or_ir.DenseI64ArrayAttr(op.attributes["dimensions"]))
        if (
            dims < 1
            or reduced != [dims - 1]
            or list(types[1].shape) != list(types[0].shape[:-1])
        ):
            return None
        positions, extents = (
            [list(range(dims)), list(range(dims - 1))],
            list(types[0].shape),
        )
    else:
        maps = [_or_ir.AffineMapAttr(a).value for a in op.attributes["indexing_maps"]]
        if len(maps) != 2:
            return None
        dims = maps[0].n_dims
        if (
            dims < 1
            or any(m.n_dims != dims or m.n_symbols for m in maps)
            or [str(x) for x in op.attributes["iterator_types"]]
            != ["#linalg.iterator_type<parallel>"] * (dims - 1)
            + ["#linalg.iterator_type<reduction>"]
        ):
            return None
        extents, positions = [None] * dims, []
        for affine, tensor in zip(maps, types):
            if len(affine.results) != len(tensor.shape):
                return None
            used = []
            for expr, extent in zip(affine.results, tensor.shape):
                if not isinstance(expr, _or_ir.AffineDimExpr):
                    return None
                position = _or_ir.AffineDimExpr(expr).position
                if (
                    position in used
                    or position >= dims
                    or (extents[position] is not None and extents[position] != extent)
                ):
                    return None
                used.append(position)
                extents[position] = extent
            positions.append(used)
        if (
            sorted(positions[0]) != list(range(dims))
            or sorted(positions[1]) != list(range(dims - 1))
            or any(extent is None for extent in extents)
        ):
            return None
    proof = _OrderedReductionDomains()
    a, b = [proof.tensor(value) for value in op.operands]
    if a is None or b is None:
        return None
    if scalar.operation.name == "arith.addf" and not (
        (a[0] and b[0]) or (a[1] and b[1])
    ):
        return None
    return positions, extents, order


def _scalarize_ordered_reductions(ctx, module):
    from torch_mlir import ir as _or_ir

    todo = []

    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for child in list(block.operations):
                    spec = _ordered_reduction_spec(child)
                    if spec is not None:
                        todo.append((child, spec))
                    else:
                        walk(child.operation)

    walk(module.operation)
    with ctx:
        for old, (positions, extents, operand_order) in todo:
            with old.location, _or_ir.InsertionPoint(old):
                index = _or_ir.IndexType.get()
                scalar = old.regions[0].blocks[0].arguments[0].type
                tensor = old.results[0].type

                def create(name, operands=(), results=(), attributes=None, regions=0):
                    return _or_ir.Operation.create(
                        name,
                        operands=list(operands),
                        results=list(results),
                        attributes=attributes or {},
                        regions=regions,
                    )

                def constant(value):
                    return create(
                        "arith.constant",
                        results=[index],
                        attributes={"value": _or_ir.IntegerAttr.get(index, value)},
                    ).results[0]

                zero, one = constant(0), constant(1)
                limits = [constant(extent) for extent in extents]
                source = list(old.regions[0].blocks[0].operations)[0]

                def output_loop(depth, current, coordinates):
                    if depth < len(extents) - 1:
                        loop = create(
                            "scf.for",
                            [zero, limits[depth], one, current],
                            [tensor],
                            regions=1,
                        )
                        block = _or_ir.Block.create_at_start(
                            loop.regions[0], [index, tensor]
                        )
                        with _or_ir.InsertionPoint(block):
                            updated = output_loop(
                                depth + 1,
                                block.arguments[1],
                                [*coordinates, block.arguments[0]],
                            )
                            create("scf.yield", [updated])
                        return loop.results[0]
                    indices = [coordinates[d] for d in positions[1]]
                    initial = create(
                        "tensor.extract", [current, *indices], [scalar]
                    ).results[0]
                    loop = create(
                        "scf.for", [zero, limits[-1], one, initial], [scalar], regions=1
                    )
                    block = _or_ir.Block.create_at_start(
                        loop.regions[0], [index, scalar]
                    )
                    with _or_ir.InsertionPoint(block):
                        all_indices = [*coordinates, block.arguments[0]]
                        value = create(
                            "tensor.extract",
                            [old.operands[0], *[all_indices[d] for d in positions[0]]],
                            [scalar],
                        ).results[0]
                        values = [value, block.arguments[1]]
                        result = create(
                            source.operation.name,
                            [values[d] for d in operand_order],
                            [scalar],
                            {
                                name: source.attributes[name]
                                for name in source.attributes
                            },
                        ).results[0]
                        create("scf.yield", [result])
                    return create(
                        "tensor.insert", [loop.results[0], current, *indices], [tensor]
                    ).results[0]

                result = output_loop(0, old.operands[1], [])
                for name in old.attributes:
                    if name.startswith("prov."):
                        result.owner.attributes[name] = old.attributes[name]
                previous = (
                    old.attributes["prov.transforms"]
                    if "prov.transforms" in old.attributes
                    else None
                )
                prefix = (
                    _or_ir.StringAttr(previous).value + ","
                    if previous is not None
                    else ""
                )
                result.owner.attributes["prov.transforms"] = _or_ir.StringAttr.get(
                    prefix + "scalar_ordered_tensor_reductions"
                )
            old.results[0].replace_all_uses_with(result)
            old.operation.erase()
    module.operation.verify()
    return len(todo)

_ORDERED_REDUCTION_ORIGINAL = _run_stages

def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(), pre_generalize=()):
    passes = [p for p in pipeline.split(',') if p]
    marker = '__merlin_scalar_ordered_tensor_reductions__'
    if marker not in passes:
        return _ORDERED_REDUCTION_ORIGINAL(ctx, module, pipeline, erase, mid, late, post_openmp, pre_generalize)
    if passes.count(marker) != 1:
        raise ValueError('ordered reduction marker must occur exactly once')
    at = passes.index(marker)
    head_has_generalize = any('linalg-generalize-named-ops' in stage for stage in passes[:at])
    _ORDERED_REDUCTION_ORIGINAL(ctx, module, ','.join(passes[:at]), 0, (), (), (),
                                pre_generalize if head_has_generalize else ())
    print('OK scalar_ordered_tensor_reductions', _scalarize_ordered_reductions(ctx, module))
    return _ORDERED_REDUCTION_ORIGINAL(ctx, module, ','.join(passes[at + 1:]), erase, mid, late, post_openmp,
                                     () if head_has_generalize else pre_generalize)

def _rc_preparation_anchor(operation, rhs):
    # Only a single-block function with the exact SSA value defined before the
    # containing root operation admits cross-consumer sharing. Never use pointer
    # identity, alias annotations, equal shapes or equal source names as a join.
    anchor = operation
    while anchor.parent is not None and anchor.parent.name != "func.func":
        anchor = anchor.parent
    function = anchor.parent
    if (
        function is None
        or len(function.regions) != 1
        or len(function.regions[0].blocks) != 1
    ):
        return None
    block = function.regions[0].blocks[0]
    roots = list(block.operations)
    if not any(anchor == root.operation for root in roots):
        return None
    if any(rhs == argument for argument in block.arguments):
        return anchor
    owner = _rc_owner(rhs)
    before = []
    for root in roots:
        if root.operation == anchor:
            break
        before.append(root.operation)
    return (
        anchor if owner is not None and any(owner == root for root in before) else None
    )


def _rc_prepare_columns(rhs, extents, anchor=None):
    from torch_mlir import ir
    import contextlib

    with ir.InsertionPoint(anchor) if anchor is not None else contextlib.nullcontext():
        index = ir.IndexType.get()
        i1 = ir.IntegerType.get_signless(1)
        i32 = ir.IntegerType.get_signless(32)
        f32 = ir.F32Type.get()

        def create(name, operands=(), results=(), attrs=None, regions=0):
            return ir.Operation.create(
                name,
                operands=list(operands),
                results=list(results),
                attributes=attrs or {},
                regions=regions,
            )

        def const(n, typ=None):
            typ = typ or index
            return create(
                "arith.constant",
                results=[typ],
                attrs={"value": ir.IntegerAttr.get(typ, n)},
            ).results[0]

        zero, one, yes = const(0), const(1), const(1, i1)
        K, N = const(extents[2]), const(extents[1])

        def cmp(a, b, p):
            return create(
                "arith.cmpi",
                [a, b],
                [i1],
                {"predicate": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), p)},
            ).results[0]

        def loop(stop, initial, body):
            op = create(
                "scf.for",
                [zero, stop, one, *initial],
                [x.type for x in initial],
                regions=1,
            )
            block = ir.Block.create_at_start(
                op.regions[0], [index, *[x.type for x in initial]]
            )
            with ir.InsertionPoint(block):
                create("scf.yield", body(block.arguments[0], list(block.arguments[1:])))
            return list(op.results)

        def choose(test, types, then, otherwise):
            op = create("scf.if", [test], types, regions=2)
            for n, fn in enumerate((then, otherwise)):
                block = ir.Block.create_at_start(op.regions[n], [])
                with ir.InsertionPoint(block):
                    create("scf.yield", fn())
            return list(op.results)

        def alloc(shape, typ):
            return create(
                "memref.alloca",
                results=[ir.MemRefType.get(shape, typ)],
                attrs={"operandSegmentSizes": ir.DenseI32ArrayAttr.get([0, 0])},
            ).results[0]

        representatives, mapping, hashes = (
            alloc([extents[1]], index),
            alloc([extents[1]], index),
            alloc([extents[1]], i32),
        )

        def load(mem, pos):
            return create(
                "memref.load", [mem, pos], [ir.MemRefType(mem.type).element_type]
            ).results[0]

        def store(value, mem, pos):
            create("memref.store", [value, mem, pos])

        def word(k, n):
            return create(
                "arith.bitcast",
                [create("tensor.extract", [rhs, k, n], [f32]).results[0]],
                [i32],
            ).results[0]

        offset, prime = const(2166136261, i32), const(16777619, i32)

        def column(n, state):
            count = state[0]

            def hash_k(k, state):
                mixed = create("arith.xori", [state[0], word(k, n)], [i32]).results[0]
                return [create("arith.muli", [mixed, prime], [i32]).results[0]]

            signature = loop(K, [offset], hash_k)[0]

            def probe(slot, state):
                prior = load(representatives, slot)

                def exact():
                    def equal_k(k, state):
                        return [
                            create(
                                "arith.andi",
                                [state[0], cmp(word(k, n), word(k, prior), 0)],
                                [i1],
                            ).results[0]
                        ]

                    return loop(K, [yes], equal_k)

                equal = choose(
                    cmp(signature, load(hashes, slot), 0),
                    [i1],
                    exact,
                    lambda: [const(0, i1)],
                )[0]
                return [
                    create("arith.select", [equal, prior, state[0]], [index]).results[0]
                ]

            representative = loop(count, [n], probe)[0]
            store(representative, mapping, n)

            def append():
                store(n, representatives, count)
                store(signature, hashes, count)
                return [create("arith.addi", [count, one], [index]).results[0]]

            return choose(cmp(representative, n, 0), [index], append, lambda: [count])

        count = loop(N, [zero], column)[0]
        packed = alloc([extents[2], extents[1]], f32)

        def pack():
            def pack_column(slot, state):
                n = load(representatives, slot)

                def pack_k(k, state):
                    x = create("tensor.extract", [rhs, k, n], [f32]).results[0]
                    create("memref.store", [x, packed, k, slot])
                    return []

                loop(K, [], pack_k)
                return []

            return loop(count, [], pack_column)

        choose(cmp(count, N, 6), [], pack, lambda: [])
        return count, representatives, mapping, packed

def _rc_owner(value):
    owner = value.owner
    return owner.operation if hasattr(owner, "operation") else None


def _rc_zero_seed(value):
    import struct
    from torch_mlir import ir

    owner = _rc_owner(value)
    if owner is None:
        return False
    if (
        owner.name == "linalg.generic"
        and len(owner.operands) == 2
        and len(owner.regions) == 1
    ):
        block = owner.regions[0].blocks[0]
        body = list(block.operations)
        rank = len(ir.RankedTensorType(value.type).shape)
        maps = [ir.AffineMapAttr(a).value for a in owner.attributes["indexing_maps"]]
        if (
            len(body) == 1
            and body[0].operation.name == "linalg.yield"
            and list(body[0].operands) == [block.arguments[0]]
            and len(maps) == 2
            and maps[0].n_dims == rank
            and not maps[0].n_symbols
            and not maps[0].results
            and maps[1].n_dims == rank
            and not maps[1].n_symbols
            and len(maps[1].results) == rank
            and sorted(
                ir.AffineDimExpr(x).position
                for x in maps[1].results
                if isinstance(x, ir.AffineDimExpr)
            )
            == list(range(rank))
            and [str(x) for x in owner.attributes["iterator_types"]]
            == ["#linalg.iterator_type<parallel>"] * rank
        ):
            scalar = _rc_owner(owner.operands[0])
            attr = (
                scalar.attributes["value"]
                if scalar is not None and scalar.name == "arith.constant"
                else None
            )
            return isinstance(attr, ir.FloatAttr) and struct.pack(
                ">f", ir.FloatAttr(attr).value
            ) == bytes(4)
    if owner.name in ("tensor.splat", "linalg.fill"):
        scalar = _rc_owner(owner.operands[0])
        attr = (
            scalar.attributes["value"]
            if scalar is not None and scalar.name == "arith.constant"
            else None
        )
    else:
        attr = (
            owner.attributes["value"]
            if owner.name == "arith.constant" and "value" in owner.attributes
            else None
        )
        if isinstance(attr, ir.DenseFPElementsAttr) and attr.is_splat:
            attr = attr.get_splat_value()
    return isinstance(attr, ir.FloatAttr) and struct.pack(
        ">f", ir.FloatAttr(attr).value
    ) == bytes(4)


def _rc_integer_rhs(value, depth=0):
    from torch_mlir import ir

    owner = _rc_owner(value)
    if owner is None or depth > 16:
        return False
    if owner.name in (
        "tensor.extract_slice",
        "tensor.expand_shape",
        "tensor.collapse_shape",
        "tensor.cast",
        "linalg.copy",
        "linalg.transpose",
    ):
        return _rc_integer_rhs(owner.operands[0], depth + 1)
    if (
        owner.name != "linalg.generic"
        or len(owner.operands) != 2
        or len(owner.regions) != 1
    ):
        return False
    block = owner.regions[0].blocks[0]
    body = list(block.operations)
    if len(body) != 2 or [x.operation.name for x in body] != [
        "arith.sitofp",
        "linalg.yield",
    ]:
        return False
    if list(body[0].operands) != [block.arguments[0]] or list(body[1].operands) != [
        body[0].results[0]
    ]:
        return False
    if (
        not isinstance(block.arguments[0].type, ir.IntegerType)
        or not 1 <= ir.IntegerType(block.arguments[0].type).width <= 16
    ):
        return False
    if any(
        name != "fastmath" and not name.startswith("prov.")
        for name in body[0].attributes
    ):
        return False
    if (
        "fastmath" in body[0].attributes
        and str(body[0].attributes["fastmath"]) != "#arith.fastmath<none>"
    ):
        return False
    rank = len(ir.RankedTensorType(value.type).shape)
    maps = [ir.AffineMapAttr(a).value for a in owner.attributes["indexing_maps"]]
    return (
        str(body[0].results[0].type) == "f32"
        and len(maps) == 2
        and [str(x) for x in owner.attributes["iterator_types"]]
        == ["#linalg.iterator_type<parallel>"] * rank
        and all(
            m.n_dims == rank
            and not m.n_symbols
            and len(m.results) == rank
            and sorted(
                ir.AffineDimExpr(x).position
                for x in m.results
                if isinstance(x, ir.AffineDimExpr)
            )
            == list(range(rank))
            for m in maps
        )
    )


def _reuse_rhs_columns(
    ctx,
    module,
    unroll=0,
    selection=None,
    reduction_limit=None,
    reduction_setup=None,
    remaining_selection=None,
    fresh_result=False,
    overlap_tail=0,
):
    from torch_mlir import ir

    if overlap_tail not in {0, 2}:
        raise ValueError("unsupported overlapping representative group")
    todo, used, packed_used = [], 0, 0
    prepared = []

    def walk(op):
        nonlocal used, packed_used
        for region in op.regions:
            for block in region.blocks:
                for child in list(block.operations):
                    spec = _scalar_contraction_spec(child)
                    if spec is not None:
                        positions, extents, _, _ = spec
                        eligible = (
                            (
                                selection is None
                                or any(
                                    child.operation == operation
                                    for operation in selection
                                )
                            )
                            and positions == [[0, 2], [2, 1], [0, 1]]
                            and len(extents) == 3
                            and 2 <= extents[0]
                            and 2 <= extents[1] <= 128
                            and 1 <= extents[2] <= 4096
                            and extents[0] * extents[2] <= 1 << 20
                            and used + 3 * extents[1] <= 8192
                            and extents[1] * extents[2] <= 8192
                            and packed_used + extents[1] * extents[2] <= 131072
                            and _rc_zero_seed(child.operands[2])
                            and _rc_integer_rhs(child.operands[1])
                            and all(
                                name
                                in {
                                    "indexing_maps",
                                    "iterator_types",
                                    "operandSegmentSizes",
                                }
                                or name.startswith("prov.")
                                for name in child.attributes
                            )
                        )
                        ancestor = child.operation
                        while eligible and ancestor is not None:
                            eligible = not any(
                                "strictfp" in name
                                or "strictfp" in str(ancestor.attributes[name])
                                for name in ancestor.attributes
                            )
                            ancestor = ancestor.parent
                        if eligible:
                            todo.append((child, spec))
                            used += 3 * extents[1]
                            packed_used += extents[1] * extents[2]
                            continue
                    walk(child.operation)

    walk(module.operation)
    if remaining_selection is not None:
        # Join live operation identities before erasing anything. Hashes alone
        # never select a source operation or restore an invalidated handle.
        remaining_selection.extend(
            operation
            for operation in selection
            if not any(operation == old.operation for old, _ in todo)
        )
    with ctx:
        for old, (positions, extents, mul_order, add_order) in todo:
            with old.location, ir.InsertionPoint(old):
                index, i1, i32, f32 = (
                    ir.IndexType.get(),
                    ir.IntegerType.get_signless(1),
                    ir.IntegerType.get_signless(32),
                    ir.F32Type.get(),
                )
                tensor = old.results[0].type

                def create(
                    name,
                    operands=(),
                    results=(),
                    attrs=None,
                    regions=0,
                    attributes=None,
                ):
                    if attrs is not None and attributes is not None:
                        raise ValueError(
                            "ambiguous RHS reuse scalar operation attributes"
                        )
                    return ir.Operation.create(
                        name,
                        operands=list(operands),
                        results=list(results),
                        attributes=attrs or attributes or {},
                        regions=regions,
                    )

                def const(n, typ=None):
                    typ = typ or index
                    return create(
                        "arith.constant",
                        results=[typ],
                        attrs={"value": ir.IntegerAttr.get(typ, n)},
                    ).results[0]

                def cmp(a, b, p):
                    return create(
                        "arith.cmpi",
                        [a, b],
                        [i1],
                        {
                            "predicate": ir.IntegerAttr.get(
                                ir.IntegerType.get_signless(64), p
                            )
                        },
                    ).results[0]

                zero, one, yes = const(0), const(1), const(1, i1)
                M, N, K = [const(x) for x in extents]

                def for_loop(stop, initial, body, start=None, step=None):
                    loop = create(
                        "scf.for",
                        [
                            zero if start is None else start,
                            stop,
                            one if step is None else step,
                            *initial,
                        ],
                        [v.type for v in initial],
                        regions=1,
                    )
                    block = ir.Block.create_at_start(
                        loop.regions[0], [index, *[v.type for v in initial]]
                    )
                    with ir.InsertionPoint(block):
                        create(
                            "scf.yield",
                            body(block.arguments[0], list(block.arguments[1:])),
                        )
                    return list(loop.results)

                def choose(test, types, then, otherwise):
                    op = create("scf.if", [test], types, regions=2)
                    for n, fn in enumerate((then, otherwise)):
                        block = ir.Block.create_at_start(op.regions[n], [])
                        with ir.InsertionPoint(block):
                            create("scf.yield", fn())
                    return list(op.results)

                def source():
                    seed = None
                    if fresh_result:
                        # The accepted seed producer is a pure exact +0 fill,
                        # splat, or dense constant. Clone it only in fallback.
                        seed = _rc_owner(old.operands[2]).clone(ip=ir.InsertionPoint.current).results[0]
                    clone = old.clone(ip=ir.InsertionPoint.current)
                    if seed is not None:
                        clone.operands[2] = seed
                    return [clone.results[0]]

                def word(value):
                    return create("arith.bitcast", [value], [i32]).results[0]

                mask, bound = const(0x7FFFFFFF, i32), const(0x3F800000, i32)

                def scan_row(m, state):
                    def scan_k(k, state):
                        x = create(
                            "tensor.extract", [old.operands[0], m, k], [f32]
                        ).results[0]
                        absolute = create("arith.andi", [word(x), mask], [i32]).results[
                            0
                        ]
                        return [
                            create(
                                "arith.andi", [state[0], cmp(absolute, bound, 7)], [i1]
                            ).results[0]
                        ]

                    return for_loop(K, state, scan_k)

                admitted = for_loop(M, [yes], scan_row)[0]

                def optimized():
                    workspace = (
                        reduction_setup(
                            old.operation, positions, extents, create, const
                        )
                        if reduction_setup is not None
                        else None
                    )
                    anchor = _rc_preparation_anchor(old.operation, old.operands[1])
                    if anchor == old.operation:
                        # The replacement guard is already before the original
                        # operation. Hoist before its first fresh constant so
                        # preparation values dominate that replacement too.
                        anchor = _rc_owner(zero)
                    preparation = None
                    if anchor is not None:
                        for rhs, function, values in prepared:
                            if rhs == old.operands[1] and function == anchor.parent:
                                preparation = values
                                break
                    if preparation is None:
                        preparation = _rc_prepare_columns(
                            old.operands[1], extents, anchor
                        )
                        if anchor is not None:
                            prepared.append(
                                (old.operands[1], anchor.parent, preparation)
                            )
                    count, representatives, mapping, packed_rhs = preparation

                    def load(mem, pos):
                        return create(
                            "memref.load",
                            [mem, pos],
                            [ir.MemRefType(mem.type).element_type],
                        ).results[0]

                    def packed():
                        positive_zero = create(
                            "arith.constant",
                            results=[f32],
                            attrs={"value": ir.FloatAttr.get(f32, 0.0)},
                        ).results[0]
                        offsets = [const(i) for i in range(8)]
                        remainder = create(
                            "arith.remui", [count, const(8)], [index]
                        ).results[0]
                        full = create(
                            "arith.subi", [count, remainder], [index]
                        ).results[0]
                        mul, add = list(old.regions[0].blocks[0].operations)[:2]

                        def row(m, state):
                            bound = (
                                reduction_limit(
                                    old.operation,
                                    [m],
                                    positions,
                                    extents,
                                    1,
                                    create,
                                    const,
                                    workspace,
                                )
                                if reduction_limit is not None
                                else None
                            )
                            if bound is not None and (
                                not isinstance(bound, tuple) or len(bound) != 3
                            ):
                                raise ValueError(
                                    "RHS reuse requires the original guarded support tuple"
                                )
                            # The complete outer |LHS|<=1 word scan implies the
                            # original callback's per-row validity check. Its
                            # ordered support and initialized-count proof remain
                            # original; no second sparse-support algorithm exists.
                            limit = K if bound is None else bound[0]

                            def source_k(k):
                                if bound is None:
                                    return k
                                order = bound[1]
                                storage = ir.MemRefType(order.type).element_type
                                k = create(
                                    "memref.load", [order, k], [storage]
                                ).results[0]
                                return (
                                    create("arith.index_cast", [k], [index]).results[0]
                                    if storage != index
                                    else k
                                )

                            def group(g, state, width=8):
                                group_offsets = (
                                    offsets if width == 8 else [const(i) for i in range(width)]
                                )
                                slots = [
                                    create("arith.addi", [g, o], [index]).results[0]
                                    for o in group_offsets
                                ]
                                seeds = [positive_zero] * width
                                attrs = (
                                    {
                                        "loop_annotation": ir.Attribute.parse(
                                            "#llvm.loop_annotation<unroll = <count = "
                                            + str(unroll)
                                            + " : i32>>"
                                        )
                                    }
                                    if unroll
                                    else {}
                                )
                                loop = create(
                                    "scf.for",
                                    [zero, limit, one, *seeds],
                                    [f32] * width,
                                    attrs,
                                    regions=1,
                                )
                                block = ir.Block.create_at_start(
                                    loop.regions[0], [index, *([f32] * width)]
                                )
                                with ir.InsertionPoint(block):
                                    k = source_k(block.arguments[0])
                                    lhs = create(
                                        "tensor.extract", [old.operands[0], m, k], [f32]
                                    ).results[0]
                                    values = []
                                    for lane, slot in enumerate(slots):
                                        rhs = create(
                                            "memref.load",
                                            [packed_rhs, k, slot],
                                            [f32],
                                        ).results[0]
                                        operands = [lhs, rhs]
                                        product = create(
                                            "arith.mulf",
                                            [operands[d] for d in mul_order],
                                            [f32],
                                            {
                                                a: mul.attributes[a]
                                                for a in mul.attributes
                                            },
                                        ).results[0]
                                        operands = [product, block.arguments[lane + 1]]
                                        values.append(
                                            create(
                                                "arith.addf",
                                                [operands[d] for d in add_order],
                                                [f32],
                                                {
                                                    a: add.attributes[a]
                                                    for a in add.attributes
                                                },
                                            ).results[0]
                                        )
                                    create("scf.yield", values)
                                current = state[0]
                                columns = [load(representatives, s) for s in slots]
                                for n, value in zip(columns, loop.results):
                                    current = create(
                                        "tensor.insert",
                                        [value, current, m, n],
                                        [tensor],
                                    ).results[0]
                                return [current]

                            current = for_loop(full, state, group, step=const(8))[0]

                            def tail(slot, state):
                                seed = positive_zero

                                def k_body(k, state):
                                    k = source_k(k)
                                    lhs = create(
                                        "tensor.extract", [old.operands[0], m, k], [f32]
                                    ).results[0]
                                    rhs = create(
                                        "memref.load", [packed_rhs, k, slot], [f32]
                                    ).results[0]
                                    operands = [lhs, rhs]
                                    product = create(
                                        "arith.mulf",
                                        [operands[d] for d in mul_order],
                                        [f32],
                                        {a: mul.attributes[a] for a in mul.attributes},
                                    ).results[0]
                                    operands = [product, state[0]]
                                    return [
                                        create(
                                            "arith.addf",
                                            [operands[d] for d in add_order],
                                            [f32],
                                            {
                                                a: add.attributes[a]
                                                for a in add.attributes
                                            },
                                        ).results[0]
                                    ]

                                value = for_loop(limit, [seed], k_body)[0]
                                n = load(representatives, slot)
                                return [
                                    create(
                                        "tensor.insert",
                                        [value, state[0], m, n],
                                        [tensor],
                                    ).results[0]
                                ]

                            if overlap_tail:
                                fits = create(
                                    "arith.andi",
                                    [cmp(count, const(8), 9), cmp(remainder, const(overlap_tail), 7)],
                                    [i1],
                                ).results[0]
                                fits = create(
                                    "arith.andi", [fits, cmp(remainder, zero, 1)], [i1]
                                ).results[0]
                                def final_group():
                                    start = create(
                                        "arith.subi", [count, const(overlap_tail)], [index]
                                    ).results[0]
                                    return group(start, [current], overlap_tail)
                                current = choose(
                                    fits, [tensor], final_group,
                                    lambda: for_loop(count, [current], tail, start=full),
                                )[0]
                            else:
                                current = for_loop(count, [current], tail, start=full)[0]

                            def scatter(n, state):
                                representative = load(mapping, n)
                                x = create(
                                    "tensor.extract",
                                    [state[0], m, representative],
                                    [f32],
                                ).results[0]
                                return [
                                    create(
                                        "tensor.insert", [x, state[0], m, n], [tensor]
                                    ).results[0]
                                ]

                            return for_loop(N, [current], scatter)

                        initial = (
                            create("tensor.empty", results=[tensor]).results[0]
                            if fresh_result
                            else old.operands[2]
                        )
                        return for_loop(M, [initial], row)

                    return choose(cmp(count, N, 6), [tensor], packed, source)

                result = choose(admitted, [tensor], optimized, source)[0]
                for name in old.attributes:
                    if name.startswith("prov."):
                        result.owner.attributes[name] = old.attributes[name]
                previous = (
                    ir.StringAttr(old.attributes["prov.transforms"]).value + ","
                    if "prov.transforms" in old.attributes
                    else ""
                )
                result.owner.attributes["prov.transforms"] = ir.StringAttr.get(
                    previous + "reuse_exact_rhs_columns"
                )
            old.results[0].replace_all_uses_with(result)
            old.erase()
    module.operation.verify()
    return len(todo)

_RHS_REUSE_ORIGINAL = _run_stages


def _run_stages(
    ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(), pre_generalize=()
):
    passes = [p for p in pipeline.split(",") if p]
    marker = "__merlin_reuse_exact_rhs_columns__"
    nonzero_marker = "__merlin_reuse_exact_rhs_columns_nonzero__"
    empty_marker = "__merlin_reuse_exact_rhs_columns_nonzero_fresh_result__"
    tails = {"__merlin_reuse_exact_rhs_columns_nonzero_tail_2_outputs__": 2}
    selected = [p for p in (nonzero_marker, empty_marker, *tails) if p in passes]
    if selected:
        if len(selected) != 1 or passes.count(selected[0]) != 1 or marker in passes:
            raise ValueError("RHS reuse strategies are explicit alternatives")
        selected_marker = selected[0]
        if "__merlin_scalar_contraction_guarded_nonzero_reduction__" not in passes:
            raise ValueError(
                "RHS support composition requires the original nonzero strategy"
            )
        original_scalarize = globals().get("_ZC_ORIGINAL_SCALARIZE")
        if original_scalarize is None:
            raise ValueError("original guarded support dispatcher is unavailable")

        def scalarize(
            ctx,
            module,
            outputs=1,
            reduction_unroll=0,
            rows=1,
            selection=None,
            output_guard=None,
            product_guard=None,
            reduction_limit=None,
            reduction_setup=None,
        ):
            options = dict(
                selection=selection,
                output_guard=output_guard,
                product_guard=product_guard,
                reduction_limit=reduction_limit,
                reduction_setup=reduction_setup,
            )
            if (
                selection is None
                or output_guard is not None
                or product_guard is not None
                or rows != 1
                or reduction_limit is None
                or reduction_setup is None
            ):
                return original_scalarize(
                    ctx, module, outputs, reduction_unroll, rows, **options
                )
            remaining = []
            count = _reuse_rhs_columns(
                ctx,
                module,
                reduction_unroll,
                selection,
                reduction_limit,
                reduction_setup,
                remaining,
                **({"fresh_result": True} if selected_marker == empty_marker or selected_marker in tails else {}),
                **({"overlap_tail": tails[selected_marker]} if selected_marker in tails else {}),
            )
            options["selection"] = remaining
            return count + (
                original_scalarize(
                    ctx, module, outputs, reduction_unroll, rows, **options
                )
                if options["selection"]
                else 0
            )

        globals()["_ZC_ORIGINAL_SCALARIZE"] = scalarize
        try:
            return _RHS_REUSE_ORIGINAL(
                ctx,
                module,
                ",".join(p for p in passes if p != selected_marker),
                erase,
                mid,
                late,
                post_openmp,
                pre_generalize,
            )
        finally:
            globals()["_ZC_ORIGINAL_SCALARIZE"] = original_scalarize
    if marker not in passes:
        return _RHS_REUSE_ORIGINAL(
            ctx, module, pipeline, erase, mid, late, post_openmp, pre_generalize
        )
    if passes.count(marker) != 1:
        raise ValueError("RHS reuse marker must occur once")
    at = passes.index(marker)
    generalized = any("linalg-generalize-named-ops" in p for p in passes[:at])
    _RHS_REUSE_ORIGINAL(
        ctx,
        module,
        ",".join(passes[:at]),
        0,
        (),
        (),
        (),
        pre_generalize if generalized else (),
    )
    if not generalized:
        _RHS_REUSE_ORIGINAL(
            ctx,
            module,
            "func.func(linalg-generalize-named-ops)",
            0,
            (),
            (),
            (),
            pre_generalize,
        )
    unroll = (
        4
        if "__merlin_scalar_contraction_reduction_unroll_4__" in passes
        else 2
        if "__merlin_scalar_contraction_reduction_unroll_2__" in passes
        else 0
    )
    print("OK reuse_exact_rhs_columns", _reuse_rhs_columns(ctx, module, unroll))
    return _RHS_REUSE_ORIGINAL(
        ctx, module, ",".join(passes[at + 1 :]), erase, mid, late, post_openmp, ()
    )

def _pointwise_unroll_candidate(op):
    from torch_mlir import ir as _pu_ir
    if (op.operation.name != "scf.for" or len(op.operands) != 3
            or len(op.results) or len(op.regions) != 1
            or len(op.regions[0].blocks) != 1
            or "loop_annotation" in op.attributes
            or "llvm.loop_annotation" in op.attributes):
        return False
    bounds = []
    for operand in op.operands:
        owner = operand.owner
        owner = owner.operation if isinstance(owner, _pu_ir.OpView) else owner
        if not isinstance(owner, _pu_ir.Operation) or owner.name != "arith.constant":
            return False
        try:
            bounds.append(_pu_ir.IntegerAttr(owner.attributes["value"]).value)
        except (TypeError, ValueError):
            return False
    lower, upper, step = bounds
    if step != 1 or upper - lower < 2:
        return False
    block = op.regions[0].blocks[0]
    if len(block.arguments) != 1:
        return False
    fmas = 0
    division = False
    load = store = False
    for inner in block.operations:
        name = inner.operation.name
        if inner.regions or any(isinstance(v.type, _pu_ir.VectorType)
                                for v in [*inner.operands, *inner.results]):
            return False
        if "fastmath" in inner.attributes and str(inner.attributes["fastmath"]) not in (
                "#arith.fastmath<none>", "#llvm.fastmath<none>"):
            return False
        if name in ("math.fma", "llvm.intr.fma"):
            if len(inner.results) != 1 or not isinstance(inner.results[0].type, _pu_ir.F32Type):
                return False
            fmas += 1
        elif name == "arith.divf":
            if not isinstance(inner.results[0].type, _pu_ir.F32Type):
                return False
            division = True
        elif name == "memref.load":
            load = True
        elif name == "memref.store":
            store = True
        elif not (name.startswith("arith.") or name == "affine.apply"
                  or (name == "scf.yield" and not inner.operands)):
            return False
    # Existing FMA chains are selected by scalar operation cost, not provenance.
    return fmas >= 4 and division and load and store


def _annotate_pointwise_unroll(ctx, module):
    from torch_mlir import ir as _pu_ir
    todo = []

    def walk(op):
        if any("strictfp" in str(op.attributes[name]) for name in op.attributes):
            return
        for region in op.regions:
            for block in region.blocks:
                for inner in list(block.operations):
                    if _pointwise_unroll_candidate(inner):
                        todo.append(inner)
                    else:
                        walk(inner.operation)

    walk(module.operation)
    with ctx:
        annotation = _pu_ir.Attribute.parse("#llvm.loop_annotation<unroll = <count = 2 : i32>>")
        for op in todo:
            op.attributes["loop_annotation"] = annotation
    module.operation.verify()
    return len(todo)


_PU_MARKER = "__merlin_scalar_pointwise_fma_division_unroll_2__"
_PU_ORIG_RUN_STAGES = _run_stages


def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(), pre_generalize=()):
    passes = [p for p in pipeline.split(',') if p]
    if _PU_MARKER not in passes:
        return _PU_ORIG_RUN_STAGES(ctx, module, pipeline, erase, mid, late, post_openmp, pre_generalize)
    if passes.count(_PU_MARKER) != 1:
        raise ValueError("scalar pointwise unroll stage occurs more than once")
    i = passes.index(_PU_MARKER)
    _PU_ORIG_RUN_STAGES(ctx, module, ','.join(passes[:i]), erase, mid, (), (), pre_generalize)
    print('OK scalar_pointwise_fma_division_unroll', _annotate_pointwise_unroll(ctx, module))
    _PU_ORIG_RUN_STAGES(ctx, module, ','.join(passes[i + 1:]), 0, (), late, post_openmp, ())

def _pointwise_packet_spec(op, multiplication=False, two_products=False):
    from torch_mlir import ir as _pp_ir
    if (op.operation.name != "linalg.generic" or len(op.results) != 1
            or len(op.regions) != 1 or len(op.regions[0].blocks) != 1):
        return None
    types = list(v.type for v in op.operands)
    if any(not isinstance(t, _pp_ir.RankedTensorType) or any(d < 0 for d in t.shape) for t in types):
        return None
    if len(op.operands) < 2:
        return None
    maps = [_pp_ir.AffineMapAttr(a).value for a in op.attributes["indexing_maps"]]
    dims = maps[-1].n_dims
    if (dims < 1 or len(maps) != len(types)
            or any(m.n_dims != dims or m.n_symbols for m in maps)
            or [str(x) for x in op.attributes["iterator_types"]] != ["#linalg.iterator_type<parallel>"] * dims):
        return None
    positions, extents = [], [None] * dims
    for m, t in zip(maps, types):
        if len(m.results) != len(t.shape):
            return None
        pos = []
        for expr, extent in zip(m.results, t.shape):
            if (multiplication and isinstance(expr, _pp_ir.AffineConstantExpr)
                    and _pp_ir.AffineConstantExpr(expr).value == 0 and extent == 1):
                pos.append(None)
                continue
            if not isinstance(expr, _pp_ir.AffineDimExpr):
                return None
            d = _pp_ir.AffineDimExpr(expr).position
            if d in pos or (extents[d] is not None and extents[d] != extent):
                return None
            pos.append(d)
            extents[d] = extent
        positions.append(pos)
    if (None in positions[-1] or sorted(positions[-1]) != list(range(dims))
            or any(d is None or d <= 0 for d in extents)):
        return None
    body = _pointwise_packet_body_spec(op, types, multiplication, two_products)
    return (positions, extents, body) if body is not None else None


def _pointwise_packet_body_spec(op, types, multiplication=False, two_products=False):
    from torch_mlir import ir as _pp_ir
    if any("strictfp" in name or "strictfp" in str(op.attributes[name]) for name in op.attributes):
        return None
    block = op.regions[0].blocks[0]
    body = list(block.operations)
    if len(block.arguments) != len(types) or not body or body[-1].operation.name != "linalg.yield" or len(body[-1].operands) != 1:
        return None
    if list(block.arguments[-1].uses):
        return None
    if two_products:
        # A separate closed arithmetic family. The existing >=3 selector keeps
        # its original eligibility and is never silently broadened.
        if (not multiplication or not isinstance(types[-1].element_type, _pp_ir.F32Type)
                or any(inner.operation.name not in {"arith.constant", "arith.sitofp",
                       "arith.uitofp", "arith.mulf", "arith.addf", "linalg.yield"}
                       for inner in body)):
            return None
    allowed = {"arith.constant", "arith.addf", "arith.subf", "arith.mulf", "arith.divf",
               "arith.negf", "arith.maximumf", "arith.minimumf", "arith.cmpf", "arith.cmpi",
               "arith.select", "arith.andi", "arith.ori", "arith.xori", "arith.addi", "arith.subi",
               "arith.muli", "arith.shli", "arith.shrsi", "arith.shrui", "arith.bitcast",
               "arith.sitofp", "arith.uitofp", "arith.fptosi", "arith.fptoui", "arith.extsi",
               "arith.extui", "arith.trunci", "arith.extf", "arith.truncf", "math.fma"}
    fmas = division = multiplies = 0
    for inner in body[:-1]:
        if any("strictfp" in name or "strictfp" in str(inner.attributes[name]) for name in inner.attributes):
            return None
        if inner.operation.name not in allowed or inner.regions or len(inner.results) != 1:
            return None
        if "fastmath" in inner.attributes and str(inner.attributes["fastmath"]) != "#arith.fastmath<none>":
            return None
        if isinstance(inner.results[0].type, _pp_ir.VectorType):
            return None
        if multiplication:
            if any("strictfp" in name or "strictfp" in str(inner.attributes[name])
                   for name in inner.attributes):
                return None
            if any(not isinstance(v.type, (_pp_ir.F32Type, _pp_ir.IntegerType, _pp_ir.IndexType))
                   for v in list(inner.operands) + list(inner.results)):
                return None
            if inner.operation.name in {"math.fma", "arith.divf", "arith.extf", "arith.truncf"}:
                return None
            if inner.operation.name == "arith.mulf" and isinstance(inner.results[0].type, _pp_ir.F32Type):
                multiplies += 1
        if inner.operation.name == "math.fma" and isinstance(inner.results[0].type, _pp_ir.F32Type):
            fmas += 1
        if inner.operation.name == "arith.divf" and isinstance(inner.results[0].type, _pp_ir.F32Type):
            division += 1
    eligible = (multiplies == 2 if two_products else multiplies >= 3) if multiplication else fmas >= 4 and division
    return body if eligible else None


def _pointwise_broadcast_axis(op, positions, extents, lanes):
    # Scalar inputs are already loop invariant and do not justify an interchange.
    # Every shared load comes from the exact same immutable tensor and coordinates.
    args = list(op.regions[0].blocks[0].arguments)
    choices = []
    for axis, extent in enumerate(extents):
        if extent < lanes:
            continue
        shared = sum(bool(pos) and axis not in pos and bool(list(arg.uses))
                     for arg, pos in zip(args[:-1], positions[:-1]))
        if shared:
            choices.append((shared, extent, -axis))
    return -max(choices)[2] if choices else None


def _packetize_pointwise(ctx, module, lanes=2, broadcast=False, multiplication=False,
                         two_products=False):
    from torch_mlir import ir as _pp_ir
    todo = []
    def walk(op):
        if any("strictfp" in name or "strictfp" in str(op.attributes[name]) for name in op.attributes):
            return
        for region in op.regions:
            for block in region.blocks:
                for inner in list(block.operations):
                    spec = _pointwise_packet_spec(inner, multiplication=multiplication,
                                                   two_products=two_products)
                    if spec is not None:
                        axis = _pointwise_broadcast_axis(inner, spec[0], spec[1], lanes) if broadcast else len(spec[1]) - 1
                        if axis is not None:
                            todo.append((inner, spec, axis))
                    else:
                        walk(inner.operation)
    walk(module.operation)
    with ctx:
        for old, (positions, extents, scalar_body), axis in todo:
            with old.location, _pp_ir.InsertionPoint(old):
                index = _pp_ir.IndexType.get()
                tensor_type = old.results[0].type
                def create(name, operands=(), results=(), attributes=None, regions=0):
                    return _pp_ir.Operation.create(name, operands=list(operands), results=list(results),
                                                  attributes=attributes or {}, regions=regions)
                constants = {}
                def c(n):
                    if n not in constants:
                        constants[n] = create("arith.constant", results=[index], attributes={"value": _pp_ir.IntegerAttr.get(index, n)}).results[0]
                    return constants[n]
                # Constants dominate all subsequently constructed loops.
                for n in [0, 1, lanes, *extents, *range(lanes), extents[axis] // lanes * lanes]:
                    c(n)
                def packet(current, indices, width):
                    lane_indices = [indices]
                    for lane in range(1, width):
                        coordinate = create("arith.addi", [indices[axis], c(lane)], [index]).results[0]
                        ids = list(indices)
                        ids[axis] = coordinate
                        lane_indices.append(ids)
                    mappings = []
                    shared_extracts = {}
                    args = list(old.regions[0].blocks[0].arguments)
                    for ids in lane_indices:
                        mapped = {}
                        for arg, operand, pos in zip(args[:-1], old.operands[:-1], positions[:-1]):
                            key = (operand, *[c(0) if d is None else ids[d] for d in pos])
                            if broadcast and key in shared_extracts:
                                mapped[arg] = shared_extracts[key]
                            else:
                                mapped[arg] = create("tensor.extract", [operand, *key[1:]], [arg.type]).results[0]
                                if broadcast:
                                    shared_extracts[key] = mapped[arg]
                        mappings.append(mapped)
                    # Breadth-first scalar operation order exposes independent chains.
                    for scalar_op in scalar_body[:-1]:
                        for mapped in mappings:
                            cloned = create(scalar_op.operation.name,
                                [mapped.get(v, v) for v in scalar_op.operands],
                                [r.type for r in scalar_op.results],
                                {name: scalar_op.attributes[name] for name in scalar_op.attributes})
                            mapped[scalar_op.results[0]] = cloned.results[0]
                    for ids, mapped in zip(lane_indices, mappings):
                        value = mapped.get(scalar_body[-1].operands[0], scalar_body[-1].operands[0])
                        current = create("tensor.insert", [value, current, *[ids[d] for d in positions[-1]]], [tensor_type]).results[0]
                    return current
                def nest(depth, current, indices):
                    if depth == len(extents) - 1:
                        end = extents[-1] // lanes * lanes
                        if end:
                            loop = create("scf.for", [c(0), c(end), c(lanes), current], [tensor_type], regions=1)
                            block = _pp_ir.Block.create_at_start(loop.regions[0], [index, tensor_type])
                            with _pp_ir.InsertionPoint(block):
                                value = packet(block.arguments[1], [*indices, block.arguments[0]], lanes)
                                create("scf.yield", [value])
                            current = loop.results[0]
                        tail = extents[-1] - end
                        if tail:
                            current = packet(current, [*indices, c(end)], tail)
                        return current
                    loop = create("scf.for", [c(0), c(extents[depth]), c(1), current], [tensor_type], regions=1)
                    block = _pp_ir.Block.create_at_start(loop.regions[0], [index, tensor_type])
                    with _pp_ir.InsertionPoint(block):
                        value = nest(depth + 1, block.arguments[1], [*indices, block.arguments[0]])
                        create("scf.yield", [value])
                    return loop.results[0]
                def broadcast_group(depth, current, coordinates, width):
                    order = [d for d in range(len(extents)) if d != axis]
                    if depth == len(order):
                        return packet(current, [coordinates[d] for d in range(len(extents))], width)
                    d = order[depth]
                    loop = create("scf.for", [c(0), c(extents[d]), c(1), current], [tensor_type], regions=1)
                    block = _pp_ir.Block.create_at_start(loop.regions[0], [index, tensor_type])
                    with _pp_ir.InsertionPoint(block):
                        value = broadcast_group(depth + 1, block.arguments[1],
                                                {**coordinates, d: block.arguments[0]}, width)
                        create("scf.yield", [value])
                    return loop.results[0]

                if broadcast:
                    current = old.operands[-1]
                    end = extents[axis] // lanes * lanes
                    loop = create("scf.for", [c(0), c(end), c(lanes), current], [tensor_type], regions=1)
                    block = _pp_ir.Block.create_at_start(loop.regions[0], [index, tensor_type])
                    with _pp_ir.InsertionPoint(block):
                        value = broadcast_group(0, block.arguments[1], {axis: block.arguments[0]}, lanes)
                        create("scf.yield", [value])
                    result = loop.results[0]
                    tail = extents[axis] - end
                    if tail:
                        result = broadcast_group(0, result, {axis: c(end)}, tail)
                else:
                    result = nest(0, old.operands[-1], [])
                replacement = result.owner
                for name in old.attributes:
                    if name.startswith("prov."):
                        replacement.attributes[name] = old.attributes[name]
                previous = old.attributes["prov.transforms"] if "prov.transforms" in old.attributes else None
                text = _pp_ir.StringAttr(previous).value + "," if previous is not None else ""
                transform = ("scalar_pointwise_two_multiplications_packet_" if two_products else
                             "scalar_pointwise_multiplication_packet_" if multiplication else
                             "scalar_pointwise_broadcast_packet_" if broadcast else "scalar_pointwise_packet_")
                replacement.attributes["prov.transforms"] = _pp_ir.StringAttr.get(text + transform + str(lanes))
            old.results[0].replace_all_uses_with(result)
            old.operation.erase()
    module.operation.verify()
    return len(todo)


def _borrowed_pointwise_packet_spec(op):
    from torch_mlir import ir as _pp_ir
    contract_name = "merlin.borrowed_pointwise_effects"
    permissions = {"immutable_inputs", "fresh_disjoint_output", "stable_rounding",
                   "nontrapping_arithmetic", "exception_flags_unobserved"}
    if (op.operation.name != "linalg.generic" or op.results
            or len(op.regions) != 1 or len(op.regions[0].blocks) != 1
            or contract_name not in op.attributes or len(op.operands) < 2):
        return None
    contract = op.attributes[contract_name]
    if not isinstance(contract, _pp_ir.DictAttr):
        return None
    if ({entry.name for entry in contract} != permissions
            or any(not isinstance(contract[key], _pp_ir.UnitAttr) for key in permissions)):
        return None
    types = [value.type for value in op.operands]
    if any(not isinstance(t, _pp_ir.MemRefType) or any(d <= 0 for d in t.shape) for t in types):
        return None
    maps = [_pp_ir.AffineMapAttr(a).value for a in op.attributes["indexing_maps"]]
    dims = maps[-1].n_dims
    if (dims < 1 or len(maps) != len(types)
            or any(m.n_dims != dims or m.n_symbols for m in maps)
            or [str(x) for x in op.attributes["iterator_types"]] != ["#linalg.iterator_type<parallel>"] * dims):
        return None
    positions, extents = [], [None] * dims
    for mapping, typ in zip(maps, types):
        if len(mapping.results) != len(typ.shape):
            return None
        try:
            strides, offset = typ.get_strides_and_offset()
        except (ValueError, TypeError, RuntimeError):
            return None
        if any(stride <= 0 for stride in strides) or offset < 0:
            return None
        pos = []
        for expr, extent in zip(mapping.results, typ.shape):
            if not isinstance(expr, _pp_ir.AffineDimExpr):
                return None
            dim = _pp_ir.AffineDimExpr(expr).position
            if dim in pos or (extents[dim] is not None and extents[dim] != extent):
                return None
            pos.append(dim)
            extents[dim] = extent
        positions.append(pos)
    # A private span does not prove that distinct logical output coordinates
    # have distinct addresses. Require a separable positive-stride layout;
    # arbitrary affine layouts and overlapping rows conservatively refuse.
    output_strides, _ = types[-1].get_strides_and_offset()
    covered = 1
    for stride, extent in sorted(zip(output_strides, types[-1].shape)):
        if extent > 1:
            if stride < covered:
                return None
            covered += (extent - 1) * stride
    # Exactly one output argument and a one-to-one full output map. Linalg
    # operand segment sizes are authoritative; scalar init reads still refuse.
    if (len(op.attributes["operandSegmentSizes"]) != 2
            or list(op.attributes["operandSegmentSizes"])[-1] != 1
            or sorted(positions[-1]) != list(range(dims))
            or any(extent is None for extent in extents)):
        return None
    body = _pointwise_packet_body_spec(op, types)
    return (positions, extents, body) if body is not None else None


def _packetize_borrowed_pointwise(ctx, module):
    from torch_mlir import ir as _pp_ir
    todo = []
    def walk(op):
        if any("strictfp" in name or "strictfp" in str(op.attributes[name]) for name in op.attributes):
            return
        for region in op.regions:
            for block in region.blocks:
                for inner in list(block.operations):
                    spec = _borrowed_pointwise_packet_spec(inner)
                    if spec is None:
                        walk(inner.operation)
                    else:
                        todo.append((inner, spec))
    walk(module.operation)
    with ctx:
        for old, (positions, extents, scalar_body) in todo:
            with old.location, _pp_ir.InsertionPoint(old):
                index = _pp_ir.IndexType.get()
                def create(name, operands=(), results=(), attributes=None, regions=0):
                    return _pp_ir.Operation.create(name, operands=list(operands), results=list(results),
                                                   attributes=attributes or {}, regions=regions)
                constants = {}
                def c(value):
                    if value not in constants:
                        constants[value] = create("arith.constant", results=[index],
                            attributes={"value": _pp_ir.IntegerAttr.get(index, value)}).results[0]
                    return constants[value]
                for value in [0, 1, 2, *extents, extents[-1] // 2 * 2]:
                    c(value)
                def packet(indices, width):
                    mappings, lane_indices = [], []
                    args = list(old.regions[0].blocks[0].arguments)
                    for lane in range(width):
                        ids = list(indices)
                        if lane:
                            ids[-1] = create("arith.addi", [ids[-1], c(lane)], [index]).results[0]
                        lane_indices.append(ids)
                        mapped = {}
                        for arg, operand, pos in zip(args[:-1], old.operands[:-1], positions[:-1]):
                            mapped[arg] = create(
                                "memref.load", [operand, *[ids[d] for d in pos]], [arg.type]
                            ).results[0]
                        mappings.append(mapped)
                    for scalar_op in scalar_body[:-1]:
                        for mapped in mappings:
                            cloned = create(scalar_op.operation.name,
                                [mapped.get(v, v) for v in scalar_op.operands],
                                [r.type for r in scalar_op.results],
                                {name: scalar_op.attributes[name] for name in scalar_op.attributes})
                            mapped[scalar_op.results[0]] = cloned.results[0]
                    stores = []
                    for ids, mapped in zip(lane_indices, mappings):
                        value = mapped.get(scalar_body[-1].operands[0], scalar_body[-1].operands[0])
                        stores.append(create(
                            "memref.store", [value, old.operands[-1], *[ids[d] for d in positions[-1]]]
                        ))
                    return stores
                def nest(depth, indices):
                    if depth == len(extents) - 1:
                        end = extents[-1] // 2 * 2
                        replacements = []
                        if end:
                            loop = create("scf.for", [c(0), c(end), c(2)], regions=1)
                            block = _pp_ir.Block.create_at_start(loop.regions[0], [index])
                            with _pp_ir.InsertionPoint(block):
                                packet([*indices, block.arguments[0]], 2)
                                create("scf.yield")
                            replacements.append(loop)
                        if extents[-1] != end:
                            replacements.extend(packet([*indices, c(end)], 1))
                        return replacements
                    loop = create("scf.for", [c(0), c(extents[depth]), c(1)], regions=1)
                    block = _pp_ir.Block.create_at_start(loop.regions[0], [index])
                    with _pp_ir.InsertionPoint(block):
                        nest(depth + 1, [*indices, block.arguments[0]])
                        create("scf.yield")
                    return [loop]
                replacements = nest(0, [])
                # A buffer writer may have both a packet loop and a scalar tail
                # at the top level. Each retains the enclosing source trace;
                # nested stores are represented by their outer loop.
                previous = old.attributes["prov.transforms"] if "prov.transforms" in old.attributes else None
                transforms = _pp_ir.StringAttr(previous).value + "," if previous is not None else ""
                for replacement in replacements:
                    for name in old.attributes:
                        if name.startswith("prov."):
                            replacement.attributes[name] = old.attributes[name]
                    replacement.attributes["prov.transforms"] = _pp_ir.StringAttr.get(
                        transforms + "borrowed_scalar_pointwise_packet_2")
                old.operation.erase()
    module.operation.verify()
    return len(todo)

_PP_MARKERS = {"__merlin_borrowed_pointwise_packet_2__": 2,
               "__merlin_pointwise_packet_2__": 2, "__merlin_pointwise_packet_4__": 4,
               "__merlin_pointwise_broadcast_packet_2__": 2,
               "__merlin_pointwise_multiplication_packet_4__": 4,
               "__merlin_pointwise_two_multiplications_packet_4__": 4}
_PP_ORIG_RUN_STAGES = _run_stages

def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(), pre_generalize=()):
    passes = [p for p in pipeline.split(',') if p]
    selected = [m for m in passes if m in _PP_MARKERS]
    if not selected:
        return _PP_ORIG_RUN_STAGES(ctx, module, pipeline, erase, mid, late, post_openmp, pre_generalize)
    multiplication_marker = "__merlin_pointwise_multiplication_packet_4__"
    two_products_marker = "__merlin_pointwise_two_multiplications_packet_4__"
    multiplication_markers = {multiplication_marker, two_products_marker}
    borrowed_marker = "__merlin_borrowed_pointwise_packet_2__"
    if (len([m for m in selected if m not in multiplication_markers and m != borrowed_marker]) > 1
            or any(selected.count(m) > 1 for m in _PP_MARKERS)):
        raise ValueError("pointwise packet schedules are alternatives")
    begin = 0
    for marker in selected:
        i = passes.index(marker, begin)
        _PP_ORIG_RUN_STAGES(ctx, module, ','.join(passes[begin:i]), 0, (), (), (),
                           pre_generalize if begin == 0 else ())
        if marker == borrowed_marker:
            print('OK borrowed_pointwise_packet', 2, _packetize_borrowed_pointwise(ctx, module))
            begin = i + 1
            continue
        print('OK scalar_pointwise_packet', _PP_MARKERS[marker], _packetize_pointwise(
            ctx, module, _PP_MARKERS[marker],
            broadcast=marker == "__merlin_pointwise_broadcast_packet_2__",
            multiplication=marker in multiplication_markers,
            two_products=marker == two_products_marker))
        begin = i + 1
    _PP_ORIG_RUN_STAGES(ctx, module, ','.join(passes[begin:]), erase, mid, late, post_openmp, ())

def _broadcast_math_spec(op):
    from torch_mlir import ir as _bh_ir
    if any("strictfp" in name or "strictfp" in str(op.attributes[name])
           for name in op.attributes):
        return None
    if (op.operation.name != "linalg.generic" or len(op.results) != 1
            or len(op.regions) != 1 or len(op.regions[0].blocks) != 1
            or len(op.operands) < 2):
        return None
    types = [value.type for value in op.operands]
    if any(not isinstance(t, _bh_ir.RankedTensorType)
           or any(extent < 0 for extent in t.shape) for t in types):
        return None
    maps = [_bh_ir.AffineMapAttr(a).value for a in op.attributes["indexing_maps"]]
    dims = maps[-1].n_dims
    if (dims < 1 or len(maps) != len(types)
            or any(m.n_dims != dims or m.n_symbols for m in maps)
            or [str(x) for x in op.attributes["iterator_types"]]
               != ["#linalg.iterator_type<parallel>"] * dims):
        return None
    # Every domain dimension is fixed by the complete output permutation.
    out_dims = []
    extents = [None] * dims
    for expr, extent in zip(maps[-1].results, types[-1].shape):
        if not isinstance(expr, _bh_ir.AffineDimExpr):
            return None
        dimension = _bh_ir.AffineDimExpr(expr).position
        if dimension in out_dims:
            return None
        out_dims.append(dimension)
        extents[dimension] = extent
    if (sorted(out_dims) != list(range(dims))
            or any(extent is None or extent <= 0 for extent in extents)):
        # A producer must never evaluate an operation absent from an empty consumer.
        return None
    dependencies = []
    for affine, tensor in zip(maps, types):
        if len(affine.results) != len(tensor.shape):
            return None
        used = set()
        for expr, extent in zip(affine.results, tensor.shape):
            if isinstance(expr, _bh_ir.AffineDimExpr):
                dimension = _bh_ir.AffineDimExpr(expr).position
                if dimension in used or extents[dimension] != extent:
                    return None
                used.add(dimension)
            elif isinstance(expr, _bh_ir.AffineConstantExpr):
                if _bh_ir.AffineConstantExpr(expr).value != 0 or extent != 1:
                    return None
            else:
                return None
        dependencies.append(frozenset(used))
    block = op.regions[0].blocks[0]
    body = list(block.operations)
    if (len(block.arguments) != len(types) or not body
            or body[-1].operation.name != "linalg.yield"
            or len(body[-1].operands) != 1):
        return None
    # These registered source operations are scalar and have no observable effects.
    # Reject an entire body containing a call/index/memory/unknown operation: moving
    # a pure chain across it could change an observed floating environment.
    allowed = {"arith.constant", "arith.addf", "arith.subf", "arith.mulf", "arith.divf",
               "arith.negf", "arith.maximumf", "arith.minimumf", "arith.cmpf", "arith.cmpi",
               "arith.select", "arith.andi", "arith.ori", "arith.xori", "arith.addi", "arith.subi",
               "arith.muli", "arith.shli", "arith.shrsi", "arith.shrui", "arith.bitcast",
               "arith.sitofp", "arith.uitofp", "arith.fptosi", "arith.fptoui", "arith.extsi",
               "arith.extui", "arith.trunci", "arith.extf", "arith.truncf", "math.rsqrt"}
    scalar_deps = dict(zip(block.arguments, dependencies))
    defining = {}
    candidates = []
    for scalar in body[:-1]:
        if any("strictfp" in name or "strictfp" in str(scalar.attributes[name])
               for name in scalar.attributes):
            return None
        if (scalar.operation.name not in allowed or scalar.regions
                or len(scalar.results) != 1
                or isinstance(scalar.results[0].type, (_bh_ir.VectorType, _bh_ir.ShapedType))):
            return None
        if ("fastmath" in scalar.attributes
                and str(scalar.attributes["fastmath"]) != "#arith.fastmath<none>"):
            return None
        deps = frozenset().union(*(scalar_deps.get(value, frozenset())
                                  for value in scalar.operands))
        scalar_deps[scalar.results[0]] = deps
        defining[scalar.results[0]] = scalar
        if (scalar.operation.name == "math.rsqrt"
                and isinstance(scalar.results[0].type,
                               (_bh_ir.F16Type, _bh_ir.F32Type, _bh_ir.F64Type))
                and any(d not in deps and extents[d] > 1 for d in range(dims))):
            candidates.append(scalar)
    if not candidates:
        return None
    target = candidates[0]
    needed = set()
    def need(value):
        if value in needed:
            return
        needed.add(value)
        scalar = defining.get(value)
        if scalar is not None:
            for operand in scalar.operands:
                need(operand)
    need(target.results[0])
    # A dependency on an output-init block argument cannot be read by a fresh producer.
    if block.arguments[-1] in needed:
        return None
    inputs = [i for i, value in enumerate(block.arguments[:-1]) if value in needed]
    chain = [scalar for scalar in body[:-1] if scalar.results[0] in needed]
    keep = sorted(scalar_deps[target.results[0]])
    return maps, extents, body, target, inputs, chain, keep


def _hoist_broadcast_source_math(ctx, module):
    from torch_mlir import ir as _bh_ir
    todo = []
    def walk(operation):
        if any("strictfp" in name or "strictfp" in str(operation.attributes[name])
               for name in operation.attributes):
            return
        for region in operation.regions:
            for block in region.blocks:
                for inner in list(block.operations):
                    spec = _broadcast_math_spec(inner)
                    if spec is not None:
                        todo.append((inner, spec))
                    else:
                        walk(inner.operation)
    walk(module.operation)
    with ctx:
        for old, (maps, extents, body, target, inputs, chain, keep) in todo:
            old_args = list(old.regions[0].blocks[0].arguments)
            with old.location:
                reduced_type = _bh_ir.RankedTensorType.get(
                    [extents[d] for d in keep], target.results[0].type)
            def create(name, operands=(), results=(), attributes=None, regions=0):
                return _bh_ir.Operation.create(name, operands=list(operands), results=list(results),
                                              attributes=attributes or {}, regions=regions)
            def attrs(operation):
                return {name: operation.attributes[name] for name in operation.attributes}
            def clone(scalar, values):
                with scalar.location:
                    copy = create(scalar.operation.name,
                                  [values.get(v, v) for v in scalar.operands],
                                  [v.type for v in scalar.results], attrs(scalar))
                values[scalar.results[0]] = copy.results[0]
            def map_on_reduced(affine):
                results = []
                for expr in affine.results:
                    if isinstance(expr, _bh_ir.AffineDimExpr):
                        position = _bh_ir.AffineDimExpr(expr).position
                        results.append(_bh_ir.AffineDimExpr.get(keep.index(position)))
                    else:
                        results.append(_bh_ir.AffineConstantExpr.get(0))
                return _bh_ir.AffineMapAttr.get(_bh_ir.AffineMap.get(len(keep), 0, results))
            with old.location, _bh_ir.InsertionPoint(old):
                empty = create("tensor.empty", results=[reduced_type])
                producer_attrs = {
                    "indexing_maps": _bh_ir.ArrayAttr.get(
                        [map_on_reduced(maps[i]) for i in inputs]
                        + [_bh_ir.AffineMapAttr.get(_bh_ir.AffineMap.get_identity(len(keep)))]),
                    "iterator_types": _bh_ir.ArrayAttr.get(
                        [_bh_ir.Attribute.parse("#linalg.iterator_type<parallel>")] * len(keep)),
                    "operandSegmentSizes": _bh_ir.DenseI32ArrayAttr.get([len(inputs), 1]),
                }
                producer = create("linalg.generic", [*[old.operands[i] for i in inputs], empty.results[0]],
                                  [reduced_type], producer_attrs, 1)
                producer_block = _bh_ir.Block.create_at_start(
                    producer.regions[0], [*[old_args[i].type for i in inputs], target.results[0].type])
                with _bh_ir.InsertionPoint(producer_block):
                    values = {old_args[i]: producer_block.arguments[j] for j, i in enumerate(inputs)}
                    for scalar in chain:
                        clone(scalar, values)
                    create("linalg.yield", [values[target.results[0]]])
                consumer_attrs = attrs(old)
                consumer_attrs["indexing_maps"] = _bh_ir.ArrayAttr.get(
                    [*[_bh_ir.AffineMapAttr.get(m) for m in maps[:-1]],
                     _bh_ir.AffineMapAttr.get(_bh_ir.AffineMap.get(maps[-1].n_dims, 0,
                                               [_bh_ir.AffineDimExpr.get(d) for d in keep])),
                     _bh_ir.AffineMapAttr.get(maps[-1])])
                consumer_attrs["operandSegmentSizes"] = _bh_ir.DenseI32ArrayAttr.get([len(old.operands), 1])
                consumer = create("linalg.generic", [*old.operands[:-1], producer.results[0], old.operands[-1]],
                                  [old.results[0].type], consumer_attrs, 1)
                consumer_block = _bh_ir.Block.create_at_start(
                    consumer.regions[0], [*[v.type for v in old_args[:-1]],
                                         target.results[0].type, old_args[-1].type])
                with _bh_ir.InsertionPoint(consumer_block):
                    values = {old_args[i]: consumer_block.arguments[i] for i in range(len(old_args) - 1)}
                    values[old_args[-1]] = consumer_block.arguments[-1]
                    values[target.results[0]] = consumer_block.arguments[-2]
                    # Backward liveness preserves every other use of a chain intermediate.
                    live = set(body[-1].operands)
                    retained = []
                    for scalar in reversed(body[:-1]):
                        if scalar is target:
                            continue
                        if scalar.results[0] in live:
                            retained.append(scalar)
                            live.update(scalar.operands)
                    for scalar in reversed(retained):
                        clone(scalar, values)
                    create("linalg.yield", [values.get(v, v) for v in body[-1].operands])
            old.results[0].replace_all_uses_with(consumer.results[0])
            old.operation.erase()
    module.operation.verify()
    return len(todo)

_BH_MARKER = "__merlin_broadcast_source_rsqrt_hoist__"
_BH_ORIG_RUN_STAGES = _run_stages

def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(), pre_generalize=()):
    passes = [p for p in pipeline.split(',') if p]
    if _BH_MARKER not in passes:
        return _BH_ORIG_RUN_STAGES(ctx, module, pipeline, erase, mid, late, post_openmp, pre_generalize)
    if passes.count(_BH_MARKER) != 1:
        raise ValueError("broadcast source math hoist marker must occur once")
    at = passes.index(_BH_MARKER)
    _BH_ORIG_RUN_STAGES(ctx, module, ','.join(passes[:at]), 0, (), (), (), pre_generalize)
    print('OK broadcast_source_rsqrt_hoist', _hoist_broadcast_source_math(ctx, module))
    _BH_ORIG_RUN_STAGES(ctx, module, ','.join(passes[at + 1:]), erase, mid, late, post_openmp, ())

def _scalar_contraction_spec(op):
    from torch_mlir import ir as _sc_ir
    if (op.operation.name != "linalg.generic" or len(op.operands) != 3
            or len(op.results) != 1 or len(op.regions) != 1
            or len(op.regions[0].blocks) != 1):
        return None
    types = list(v.type for v in op.operands)
    if any(not isinstance(t, _sc_ir.RankedTensorType) for t in types):
        return None
    types = [_sc_ir.RankedTensorType(t) for t in types]
    if (any(not isinstance(t.element_type, _sc_ir.F32Type) for t in types)
            or op.results[0].type != op.operands[2].type
            or any(d < 0 for t in types for d in t.shape)):
        return None
    block = op.regions[0].blocks[0]
    body = list(block.operations)
    if (len(block.arguments) != 3 or len(body) != 3
            or [x.operation.name for x in body] != ["arith.mulf", "arith.addf", "linalg.yield"]
            or any("fastmath" in x.attributes
                   and str(x.attributes["fastmath"]) != "#arith.fastmath<none>" for x in body)
            or any(name != "fastmath" and not name.startswith("prov.")
                   for x in body[:2] for name in x.attributes)):
        return None
    mul, add, yield_op = body
    args = list(block.arguments)
    if list(mul.operands) == args[:2]:
        mul_order = [0, 1]
    elif list(mul.operands) == args[1::-1]:
        mul_order = [1, 0]
    else:
        return None
    if list(add.operands) == [mul.results[0], args[2]]:
        add_order = [0, 1]
    elif list(add.operands) == [args[2], mul.results[0]]:
        add_order = [1, 0]
    else:
        return None
    if list(yield_op.operands) != [add.results[0]]:
        return None
    maps = [_sc_ir.AffineMapAttr(a).value for a in op.attributes["indexing_maps"]]
    if len(maps) != 3:
        return None
    dims = maps[0].n_dims
    if dims < 2 or any(m.n_dims != dims or m.n_symbols for m in maps):
        return None
    iters = list(op.attributes["iterator_types"])
    # Enum attribute printer is the upstream binding's stable spelling.
    if ([str(x) for x in iters] !=
            ["#linalg.iterator_type<parallel>"] * (dims - 1) + ["#linalg.iterator_type<reduction>"]):
        return None
    positions = []
    extents = [None] * dims
    for m, t in zip(maps, types):
        if len(m.results) != len(t.shape):
            return None
        pos = []
        for expr, extent in zip(m.results, t.shape):
            if not isinstance(expr, _sc_ir.AffineDimExpr):
                return None
            d = _sc_ir.AffineDimExpr(expr).position
            if d in pos or (extents[d] is not None and extents[d] != extent):
                return None
            pos.append(d)
            extents[d] = extent
        positions.append(pos)
    if (sorted(positions[2]) != list(range(dims - 1))
            or any(dims - 1 not in pos for pos in positions[:2])
            or any(d is None for d in extents)):
        return None
    return positions, extents, mul_order, add_order


def _scalarize_tensor_contractions(
        ctx, module, outputs=1, reduction_unroll=0, rows=1, selection=None, output_guard=None,
        product_guard=None, reduction_limit=None, reduction_setup=None):
    from torch_mlir import ir as _sc_ir
    todo = []

    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for inner in list(block.operations):
                    spec = _scalar_contraction_spec(inner)
                    eligible = (spec is not None and spec[1][-2] % outputs == 0
                                and (selection is None or any(inner.operation == item for item in selection)))
                    if eligible and rows != 1:
                        positions, extents, _, _ = spec
                        row_axis, col_axis = len(extents) - 3, len(extents) - 2
                        # A rectangular tile is a matrix schedule only when the
                        # exact projected maps prove complementary row/column
                        # reuse. No names, shapes alone or pointer facts suffice.
                        eligible = (row_axis >= 0 and extents[row_axis] % rows == 0
                                    and row_axis in positions[0] and col_axis not in positions[0]
                                    and col_axis in positions[1] and row_axis not in positions[1])
                        owner = inner.operation
                        while eligible and owner is not None:
                            if any(a in owner.attributes for a in ("strictfp", "llvm.strictfp")):
                                eligible = False
                            owner = owner.parent
                    if eligible:
                        todo.append((inner, spec))
                    else:
                        walk(inner.operation)

    walk(module.operation)
    with ctx:
        for old, (positions, extents, mul_order, add_order) in todo:
            with old.location, _sc_ir.InsertionPoint(old):
                index = _sc_ir.IndexType.get()
                scalar = old.regions[0].blocks[0].arguments[0].type
                tensor = old.results[0].type
                def create(name, operands=(), results=(), attributes=None, regions=0):
                    return _sc_ir.Operation.create(name, operands=list(operands), results=list(results),
                                                   attributes=attributes or {}, regions=regions)
                def constant(value):
                    return create("arith.constant", results=[index], attributes={
                        "value": _sc_ir.IntegerAttr.get(index, value)}).results[0]
                zero, one = constant(0), constant(1)
                limits = [constant(e) for e in extents]
                step = one if outputs == 1 else constant(outputs)
                offsets = [zero, *[constant(i) for i in range(1, outputs)]]
                row_offsets = [zero, *[constant(i) for i in range(1, rows)]]
                row_step = one if rows == 1 else constant(rows)
                workspace = (reduction_setup(old.operation, positions, extents, create, constant)
                             if reduction_setup is not None else None)

                def output_loop(depth, current, indices, bound=None):
                    if depth < len(extents) - 1:
                        if reduction_limit is not None and depth == len(extents) - 2:
                            bound = reduction_limit(old.operation, indices, positions, extents,
                                                    rows, create, constant, workspace)
                        stride = (step if depth == len(extents) - 2 else
                                  row_step if rows != 1 and depth == len(extents) - 3 else one)
                        loop = create("scf.for", [zero, limits[depth], stride, current], [tensor], regions=1)
                        block = _sc_ir.Block.create_at_start(loop.regions[0], [index, tensor])
                        with _sc_ir.InsertionPoint(block):
                            updated = output_loop(depth + 1, block.arguments[1],
                                                  [*indices, block.arguments[0]], bound)
                            create("scf.yield", [updated])
                        return loop.results[0]
                    if rows == 1:
                        lanes = [indices]
                        for offset in offsets[1:]:
                            column = create("arith.addi", [indices[-1], offset], [index]).results[0]
                            lanes.append([*indices[:-1], column])
                    else:
                        lanes = []
                        for row_offset in row_offsets:
                            row = (indices[-2] if row_offset == zero else
                                   create("arith.addi", [indices[-2], row_offset], [index]).results[0])
                            for offset in offsets:
                                column = (indices[-1] if offset == zero else
                                          create("arith.addi", [indices[-1], offset], [index]).results[0])
                                lanes.append([*indices[:-2], row, column])
                    out_indices = [[lane[d] for d in positions[2]] for lane in lanes]
                    def reduction(ordered=True):
                        initial = [create("tensor.extract", [current, *idx], [scalar]).results[0]
                                   for idx in out_indices]
                        # SCF -> CF forwards LLVM-typed values; CF -> LLVM keeps the key.
                        # Use the final branch property's spelling: a retained namespaced
                        # llvm.loop_annotation is ignored by the installed translator.
                        attributes = ({"loop_annotation": _sc_ir.Attribute.parse(
                            "#llvm.loop_annotation<unroll = <count = " + str(reduction_unroll) + " : i32>>")}
                            if reduction_unroll else {})
                        lane_count = rows * outputs
                        order = bound[1] if ordered and isinstance(bound, tuple) else None
                        limit = (limits[-1] if not ordered or bound is None else
                                 bound[0] if order is not None else bound)
                        loop = create("scf.for", [zero, limit, one, *initial], [scalar] * lane_count,
                                      attributes=attributes, regions=1)
                        block = _sc_ir.Block.create_at_start(loop.regions[0], [index, *([scalar] * lane_count)])
                        with _sc_ir.InsertionPoint(block):
                            load = ("memref.load" if order is not None and
                                    isinstance(order.type, _sc_ir.MemRefType) else "tensor.extract")
                            k = block.arguments[0]
                            if order is not None:
                                storage = (_sc_ir.MemRefType(order.type).element_type if load == "memref.load"
                                           else _sc_ir.RankedTensorType(order.type).element_type)
                                k = create(load, [order, k], [storage]).results[0]
                                if storage != index:
                                    k = create("arith.index_cast", [k], [index]).results[0]
                            accumulated, shared, pending = [], {}, []
                            def update(values, prior):
                                product = create("arith.mulf", [values[d] for d in mul_order], [scalar]).results[0]
                                operands = [product, prior]
                                return create("arith.addf", [operands[d] for d in add_order], [scalar]).results[0]
                            for lane_number, lane in enumerate(lanes):
                                all_indices = [*lane, k]
                                values = []
                                for i in range(2):
                                    if rows == 1:
                                        key = i if len(extents) - 2 not in positions[i] else None
                                    else:
                                        key = (i,
                                               lane_number // outputs if len(extents) - 3 in positions[i] else 0,
                                               lane_number % outputs if len(extents) - 2 in positions[i] else 0)
                                    if key is not None and key in shared:
                                        value = shared[key]
                                    else:
                                        value = create(
                                            "tensor.extract",
                                            [old.operands[i], *[all_indices[d] for d in positions[i]]],
                                            [scalar]).results[0]
                                        if key is not None:
                                            shared[key] = value
                                    values.append(value)
                                prior = block.arguments[lane_number + 1]
                                if product_guard is None:
                                    accumulated.append(update(values, prior))
                                else:
                                    pending.append((values, prior))
                            if product_guard is not None:
                                for start in range(0, lane_count, outputs):
                                    group = pending[start:start + outputs]
                                    condition = product_guard(old.operation, group[0][0], create)
                                    choice = create("scf.if", [condition], [scalar] * outputs, regions=2)
                                    for branch in range(2):
                                        child = _sc_ir.Block.create_at_start(choice.regions[branch], [])
                                        with _sc_ir.InsertionPoint(child):
                                            create("scf.yield", [prior if branch == 0 else update(values, prior)
                                                                 for values, prior in group])
                                    accumulated.extend(choice.results)
                            create("scf.yield", accumulated)
                        destination = current
                        for value, idx in zip(loop.results, out_indices):
                            destination = create("tensor.insert", [value, destination, *idx], [tensor]).results[0]
                        return destination
                    def selected_reduction():
                        if not isinstance(bound, tuple) or len(bound) < 3:
                            return reduction()
                        guarded = create("scf.if", [bound[2]], [tensor], regions=2)
                        for branch in range(2):
                            block = _sc_ir.Block.create_at_start(guarded.regions[branch], [])
                            with _sc_ir.InsertionPoint(block):
                                create("scf.yield", [reduction(ordered=branch == 0)])
                        return guarded.results[0]
                    if output_guard is None:
                        return selected_reduction()
                    condition = output_guard(old.operation, lanes, positions[2], create, constant)
                    guarded = create("scf.if", [condition], [tensor], regions=2)
                    for branch in range(2):
                        block = _sc_ir.Block.create_at_start(guarded.regions[branch], [])
                        with _sc_ir.InsertionPoint(block):
                            create("scf.yield", [selected_reduction() if branch == 0 else current])
                    return guarded.results[0]

                result = output_loop(0, old.operands[2], [])
                replacement = result.owner
                for name in old.attributes:
                    if name.startswith("prov."):
                        replacement.attributes[name] = old.attributes[name]
                previous = old.attributes["prov.transforms"] if "prov.transforms" in old.attributes else None
                text = _sc_ir.StringAttr(previous).value + "," if previous is not None else ""
                suffix = (f"_{rows}x{outputs}_outputs" if rows != 1 else
                          "" if outputs == 1 else "_" + str(outputs) + "_outputs")
                transforms = text + "scalar_contraction_accumulator" + suffix
                if reduction_unroll:
                    transforms += ",unroll_scalar_contraction_reduction_by_" + str(reduction_unroll)
                replacement.attributes["prov.transforms"] = _sc_ir.StringAttr.get(transforms)
            old.results[0].replace_all_uses_with(result)
            old.operation.erase()
    module.operation.verify()
    return len(todo)



def _sqs_spec(op):
    from torch_mlir import ir as _sqs_ir
    if (op.operation.name != "linalg.generic" or len(op.operands) != 2
            or len(op.results) != 1 or len(op.regions) != 1
            or len(op.regions[0].blocks) != 1):
        return None
    ancestor = op.operation
    while ancestor is not None:
        if any("strictfp" in name or "strictfp" in str(ancestor.attributes[name])
               for name in ancestor.attributes):
            return None
        ancestor = ancestor.parent
    types = []
    for value in op.operands:
        if not isinstance(value.type, _sqs_ir.RankedTensorType):
            return None
        tensor = _sqs_ir.RankedTensorType(value.type)
        if (not isinstance(tensor.element_type, _sqs_ir.F32Type)
                or any(extent < 0 for extent in tensor.shape)):
            return None
        types.append(tensor)
    if op.results[0].type != op.operands[1].type:
        return None
    allowed_attributes = {"indexing_maps", "iterator_types", "operandSegmentSizes"}
    if any(name not in allowed_attributes and not name.startswith("prov.")
           for name in op.attributes):
        return None
    block = op.regions[0].blocks[0]
    body = list(block.operations)
    if (len(block.arguments) != 2 or len(body) != 3
            or [x.operation.name for x in body]
                != ["arith.mulf", "arith.addf", "linalg.yield"]
            or any(x.regions for x in body)
            or any(name != "fastmath" and not name.startswith("prov.")
                   for scalar in body for name in scalar.attributes)
            or any("fastmath" in x.attributes
                   and str(x.attributes["fastmath"]) != "#arith.fastmath<none>"
                   for x in body)):
        return None
    mul, add, terminator = body
    value, seed = block.arguments
    if list(mul.operands) != [value, value]:
        return None
    if list(add.operands) == [mul.results[0], seed]:
        order = [0, 1]
    elif list(add.operands) == [seed, mul.results[0]]:
        order = [1, 0]
    else:
        return None
    if list(terminator.operands) != [add.results[0]]:
        return None
    maps = [_sqs_ir.AffineMapAttr(a).value for a in op.attributes["indexing_maps"]]
    if len(maps) != 2:
        return None
    dims = maps[0].n_dims
    if (dims < 1 or any(m.n_dims != dims or m.n_symbols for m in maps)
            or [str(x) for x in op.attributes["iterator_types"]]
                != ["#linalg.iterator_type<parallel>"] * (dims - 1)
                   + ["#linalg.iterator_type<reduction>"]):
        return None
    extents, positions = [None] * dims, []
    for affine, tensor in zip(maps, types):
        if len(affine.results) != len(tensor.shape):
            return None
        used = []
        for expr, extent in zip(affine.results, tensor.shape):
            if not isinstance(expr, _sqs_ir.AffineDimExpr):
                return None
            position = _sqs_ir.AffineDimExpr(expr).position
            if (position in used or position >= dims
                    or (extents[position] is not None and extents[position] != extent)):
                return None
            used.append(position)
            extents[position] = extent
        positions.append(used)
    if (sorted(positions[0]) != list(range(dims))
            or sorted(positions[1]) != list(range(dims - 1))
            or any(extent is None for extent in extents)):
        return None
    return positions, extents, order


def _scalarize_squared_sums(ctx, module):
    from torch_mlir import ir as _sqs_ir
    todo = []
    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for child in list(block.operations):
                    spec = _sqs_spec(child)
                    if spec is not None:
                        todo.append((child, spec))
                    else:
                        walk(child.operation)
    walk(module.operation)
    with ctx:
        for old, (positions, extents, add_order) in todo:
            with old.location, _sqs_ir.InsertionPoint(old):
                index = _sqs_ir.IndexType.get()
                scalar = old.regions[0].blocks[0].arguments[0].type
                tensor = old.results[0].type
                def create(name, operands=(), results=(), attributes=None, regions=0):
                    return _sqs_ir.Operation.create(name, operands=list(operands),
                        results=list(results), attributes=attributes or {}, regions=regions)
                def constant(value):
                    return create("arith.constant", results=[index], attributes={
                        "value": _sqs_ir.IntegerAttr.get(index, value)}).results[0]
                zero, one = constant(0), constant(1)
                limits = [constant(extent) for extent in extents]
                source_mul, source_add = list(old.regions[0].blocks[0].operations)[:2]
                def output_loop(depth, current, coordinates):
                    if depth < len(extents) - 1:
                        loop = create("scf.for", [zero, limits[depth], one, current],
                                      [tensor], regions=1)
                        block = _sqs_ir.Block.create_at_start(loop.regions[0], [index, tensor])
                        with _sqs_ir.InsertionPoint(block):
                            updated = output_loop(depth + 1, block.arguments[1],
                                                  [*coordinates, block.arguments[0]])
                            create("scf.yield", [updated])
                        return loop.results[0]
                    out_indices = [coordinates[d] for d in positions[1]]
                    initial = create("tensor.extract", [current, *out_indices], [scalar]).results[0]
                    loop = create("scf.for", [zero, limits[-1], one, initial],
                                  [scalar], regions=1)
                    block = _sqs_ir.Block.create_at_start(loop.regions[0], [index, scalar])
                    with _sqs_ir.InsertionPoint(block):
                        all_indices = [*coordinates, block.arguments[0]]
                        value = create("tensor.extract", [old.operands[0],
                            *[all_indices[d] for d in positions[0]]], [scalar]).results[0]
                        product = create("arith.mulf", [value, value], [scalar],
                            {name: source_mul.attributes[name] for name in source_mul.attributes}).results[0]
                        values = [product, block.arguments[1]]
                        total = create("arith.addf", [values[d] for d in add_order], [scalar],
                            {name: source_add.attributes[name] for name in source_add.attributes}).results[0]
                        create("scf.yield", [total])
                    return create("tensor.insert", [loop.results[0], current, *out_indices],
                                  [tensor]).results[0]
                result = output_loop(0, old.operands[1], [])
                for name in old.attributes:
                    if name.startswith("prov."):
                        result.owner.attributes[name] = old.attributes[name]
                previous = old.attributes["prov.transforms"] if "prov.transforms" in old.attributes else None
                prefix = _sqs_ir.StringAttr(previous).value + "," if previous is not None else ""
                result.owner.attributes["prov.transforms"] = _sqs_ir.StringAttr.get(
                    prefix + "scalar_squared_sum_accumulator")
            old.results[0].replace_all_uses_with(result)
            old.operation.erase()
    module.operation.verify()
    return len(todo)


def _broadcast_math_spec(op):
    from torch_mlir import ir as _bh_ir
    if any("strictfp" in name or "strictfp" in str(op.attributes[name])
           for name in op.attributes):
        return None
    if (op.operation.name != "linalg.generic" or len(op.results) != 1
            or len(op.regions) != 1 or len(op.regions[0].blocks) != 1
            or len(op.operands) < 2):
        return None
    types = [value.type for value in op.operands]
    if any(not isinstance(t, _bh_ir.RankedTensorType)
           or any(extent < 0 for extent in t.shape) for t in types):
        return None
    maps = [_bh_ir.AffineMapAttr(a).value for a in op.attributes["indexing_maps"]]
    dims = maps[-1].n_dims
    if (dims < 1 or len(maps) != len(types)
            or any(m.n_dims != dims or m.n_symbols for m in maps)
            or [str(x) for x in op.attributes["iterator_types"]]
               != ["#linalg.iterator_type<parallel>"] * dims):
        return None
    # Every domain dimension is fixed by the complete output permutation.
    out_dims = []
    extents = [None] * dims
    for expr, extent in zip(maps[-1].results, types[-1].shape):
        if not isinstance(expr, _bh_ir.AffineDimExpr):
            return None
        dimension = _bh_ir.AffineDimExpr(expr).position
        if dimension in out_dims:
            return None
        out_dims.append(dimension)
        extents[dimension] = extent
    if (sorted(out_dims) != list(range(dims))
            or any(extent is None or extent <= 0 for extent in extents)):
        # A producer must never evaluate an operation absent from an empty consumer.
        return None
    dependencies = []
    for affine, tensor in zip(maps, types):
        if len(affine.results) != len(tensor.shape):
            return None
        used = set()
        for expr, extent in zip(affine.results, tensor.shape):
            if isinstance(expr, _bh_ir.AffineDimExpr):
                dimension = _bh_ir.AffineDimExpr(expr).position
                if dimension in used or extents[dimension] != extent:
                    return None
                used.add(dimension)
            elif isinstance(expr, _bh_ir.AffineConstantExpr):
                if _bh_ir.AffineConstantExpr(expr).value != 0 or extent != 1:
                    return None
            else:
                return None
        dependencies.append(frozenset(used))
    block = op.regions[0].blocks[0]
    body = list(block.operations)
    if (len(block.arguments) != len(types) or not body
            or body[-1].operation.name != "linalg.yield"
            or len(body[-1].operands) != 1):
        return None
    # These registered source operations are scalar and have no observable effects.
    # Reject an entire body containing a call/index/memory/unknown operation: moving
    # a pure chain across it could change an observed floating environment.
    allowed = {"arith.constant", "arith.addf", "arith.subf", "arith.mulf", "arith.divf",
               "arith.negf", "arith.maximumf", "arith.minimumf", "arith.cmpf", "arith.cmpi",
               "arith.select", "arith.andi", "arith.ori", "arith.xori", "arith.addi", "arith.subi",
               "arith.muli", "arith.shli", "arith.shrsi", "arith.shrui", "arith.bitcast",
               "arith.sitofp", "arith.uitofp", "arith.fptosi", "arith.fptoui", "arith.extsi",
               "arith.extui", "arith.trunci", "arith.extf", "arith.truncf", "math.rsqrt"}
    scalar_deps = dict(zip(block.arguments, dependencies))
    defining = {}
    candidates = []
    for scalar in body[:-1]:
        if any("strictfp" in name or "strictfp" in str(scalar.attributes[name])
               for name in scalar.attributes):
            return None
        if (scalar.operation.name not in allowed or scalar.regions
                or len(scalar.results) != 1
                or isinstance(scalar.results[0].type, (_bh_ir.VectorType, _bh_ir.ShapedType))):
            return None
        if ("fastmath" in scalar.attributes
                and str(scalar.attributes["fastmath"]) != "#arith.fastmath<none>"):
            return None
        deps = frozenset().union(*(scalar_deps.get(value, frozenset())
                                  for value in scalar.operands))
        scalar_deps[scalar.results[0]] = deps
        defining[scalar.results[0]] = scalar
        if (scalar.operation.name == "math.rsqrt"
                and isinstance(scalar.results[0].type,
                               (_bh_ir.F16Type, _bh_ir.F32Type, _bh_ir.F64Type))
                and any(d not in deps and extents[d] > 1 for d in range(dims))):
            candidates.append(scalar)
    if not candidates:
        return None
    target = candidates[0]
    needed = set()
    def need(value):
        if value in needed:
            return
        needed.add(value)
        scalar = defining.get(value)
        if scalar is not None:
            for operand in scalar.operands:
                need(operand)
    need(target.results[0])
    # A dependency on an output-init block argument cannot be read by a fresh producer.
    if block.arguments[-1] in needed:
        return None
    inputs = [i for i, value in enumerate(block.arguments[:-1]) if value in needed]
    chain = [scalar for scalar in body[:-1] if scalar.results[0] in needed]
    keep = sorted(scalar_deps[target.results[0]])
    return maps, extents, body, target, inputs, chain, keep


def _hoist_broadcast_source_math(ctx, module):
    from torch_mlir import ir as _bh_ir
    todo = []
    def walk(operation):
        if any("strictfp" in name or "strictfp" in str(operation.attributes[name])
               for name in operation.attributes):
            return
        for region in operation.regions:
            for block in region.blocks:
                for inner in list(block.operations):
                    spec = _broadcast_math_spec(inner)
                    if spec is not None:
                        todo.append((inner, spec))
                    else:
                        walk(inner.operation)
    walk(module.operation)
    with ctx:
        for old, (maps, extents, body, target, inputs, chain, keep) in todo:
            old_args = list(old.regions[0].blocks[0].arguments)
            with old.location:
                reduced_type = _bh_ir.RankedTensorType.get(
                    [extents[d] for d in keep], target.results[0].type)
            def create(name, operands=(), results=(), attributes=None, regions=0):
                return _bh_ir.Operation.create(name, operands=list(operands), results=list(results),
                                              attributes=attributes or {}, regions=regions)
            def attrs(operation):
                return {name: operation.attributes[name] for name in operation.attributes}
            def clone(scalar, values):
                with scalar.location:
                    copy = create(scalar.operation.name,
                                  [values.get(v, v) for v in scalar.operands],
                                  [v.type for v in scalar.results], attrs(scalar))
                values[scalar.results[0]] = copy.results[0]
            def map_on_reduced(affine):
                results = []
                for expr in affine.results:
                    if isinstance(expr, _bh_ir.AffineDimExpr):
                        position = _bh_ir.AffineDimExpr(expr).position
                        results.append(_bh_ir.AffineDimExpr.get(keep.index(position)))
                    else:
                        results.append(_bh_ir.AffineConstantExpr.get(0))
                return _bh_ir.AffineMapAttr.get(_bh_ir.AffineMap.get(len(keep), 0, results))
            with old.location, _bh_ir.InsertionPoint(old):
                empty = create("tensor.empty", results=[reduced_type])
                producer_attrs = {
                    "indexing_maps": _bh_ir.ArrayAttr.get(
                        [map_on_reduced(maps[i]) for i in inputs]
                        + [_bh_ir.AffineMapAttr.get(_bh_ir.AffineMap.get_identity(len(keep)))]),
                    "iterator_types": _bh_ir.ArrayAttr.get(
                        [_bh_ir.Attribute.parse("#linalg.iterator_type<parallel>")] * len(keep)),
                    "operandSegmentSizes": _bh_ir.DenseI32ArrayAttr.get([len(inputs), 1]),
                }
                producer = create("linalg.generic", [*[old.operands[i] for i in inputs], empty.results[0]],
                                  [reduced_type], producer_attrs, 1)
                producer_block = _bh_ir.Block.create_at_start(
                    producer.regions[0], [*[old_args[i].type for i in inputs], target.results[0].type])
                with _bh_ir.InsertionPoint(producer_block):
                    values = {old_args[i]: producer_block.arguments[j] for j, i in enumerate(inputs)}
                    for scalar in chain:
                        clone(scalar, values)
                    create("linalg.yield", [values[target.results[0]]])
                consumer_attrs = attrs(old)
                consumer_attrs["indexing_maps"] = _bh_ir.ArrayAttr.get(
                    [*[_bh_ir.AffineMapAttr.get(m) for m in maps[:-1]],
                     _bh_ir.AffineMapAttr.get(_bh_ir.AffineMap.get(maps[-1].n_dims, 0,
                                               [_bh_ir.AffineDimExpr.get(d) for d in keep])),
                     _bh_ir.AffineMapAttr.get(maps[-1])])
                consumer_attrs["operandSegmentSizes"] = _bh_ir.DenseI32ArrayAttr.get([len(old.operands), 1])
                consumer = create("linalg.generic", [*old.operands[:-1], producer.results[0], old.operands[-1]],
                                  [old.results[0].type], consumer_attrs, 1)
                consumer_block = _bh_ir.Block.create_at_start(
                    consumer.regions[0], [*[v.type for v in old_args[:-1]],
                                         target.results[0].type, old_args[-1].type])
                with _bh_ir.InsertionPoint(consumer_block):
                    values = {old_args[i]: consumer_block.arguments[i] for i in range(len(old_args) - 1)}
                    values[old_args[-1]] = consumer_block.arguments[-1]
                    values[target.results[0]] = consumer_block.arguments[-2]
                    # Backward liveness preserves every other use of a chain intermediate.
                    live = set(body[-1].operands)
                    retained = []
                    for scalar in reversed(body[:-1]):
                        if scalar is target:
                            continue
                        if scalar.results[0] in live:
                            retained.append(scalar)
                            live.update(scalar.operands)
                    for scalar in reversed(retained):
                        clone(scalar, values)
                    create("linalg.yield", [values.get(v, v) for v in body[-1].operands])
            old.results[0].replace_all_uses_with(consumer.results[0])
            old.operation.erase()
    module.operation.verify()
    return len(todo)


def _merlin_host_ordinary_scope(operation):
    ancestor = operation
    while ancestor is not None:
        for name in ancestor.attributes:
            if ("strictfp" in name or "strictfp" in str(ancestor.attributes[name])
                    or "fenv" in name):
                return False
        ancestor = ancestor.parent
    return True


def _merlin_optimize_host_tensors(ctx, module, choices):
    counts = dict.fromkeys(choices, 0)
    # Explicit environment scopes anywhere in this module are unsupported.
    # The existing individual rewrites also check all typed body operations.
    def scoped(operation):
        if not _merlin_host_ordinary_scope(operation):
            return False
        return all(scoped(inner.operation) for region in operation.regions
                   for block in region.blocks for inner in block.operations)
    if not scoped(module.operation):
        return counts
    if choices['broadcast_math']:
        counts['broadcast_math'] = _hoist_broadcast_source_math(ctx, module)
    if choices['contraction']:
        selected = []
        def collect(operation):
            for region in operation.regions:
                for block in region.blocks:
                    for inner in block.operations:
                        spec = _scalar_contraction_spec(inner)
                        if spec is not None and spec[1][-1] > 1:
                            selected.append(inner.operation)
                        else:
                            collect(inner.operation)
        collect(module.operation)
        counts['contraction'] = _scalarize_tensor_contractions(ctx, module, selection=selected)
    if choices['squared_sum']:
        counts['squared_sum'] = _scalarize_squared_sums(ctx, module)
    module.operation.verify()
    return counts

_AUTO_HOST_CHOICES = {'contraction': True, 'squared_sum': True, 'broadcast_math': True}
_AUTO_HOST_ORIGINAL = _run_stages


def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(), pre_generalize=()):
    import json as _auto_json
    import hashlib as _auto_hashlib
    passes = [p for p in pipeline.split(',') if p]
    marker = '__merlin_automatic_scalar_host__'
    if marker not in passes:
        return _AUTO_HOST_ORIGINAL(ctx, module, pipeline, erase, mid, late, post_openmp, pre_generalize)
    if passes.count(marker) != 1:
        raise ValueError('automatic host stage is duplicated')
    at = passes.index(marker)
    head = passes[:at]
    generalizes = any('linalg-generalize-named-ops' in p for p in head)
    _AUTO_HOST_ORIGINAL(ctx, module, ','.join(head), 0, (), (), (),
                        pre_generalize if generalizes else ())
    before = str(module)
    # Call the existing typed rewrites, including their own shape/use/effect
    # refusals. No pattern matching by source name or captured provenance.
    if not generalizes and (_AUTO_HOST_CHOICES['contraction'] or _AUTO_HOST_CHOICES['squared_sum']):
        _AUTO_HOST_ORIGINAL(ctx, module, 'func.func(linalg-generalize-named-ops)',
                            0, (), (), (), pre_generalize)
    counts = _merlin_optimize_host_tensors(ctx, module, _AUTO_HOST_CHOICES)
    report = {
        'schema': 'merlin.automatic_host_features.v1',
        'choices': _AUTO_HOST_CHOICES, 'rewrites': counts,
        'source_sha256': _auto_hashlib.sha256(before.encode()).hexdigest(),
        'output_sha256': _auto_hashlib.sha256(str(module).encode()).hexdigest(),
        'selection_basis': 'typed_source_exactness_and_reduced_recurrence_materialization',
        'unselected_cost_alternatives': ['packet_widths', 'reduction_unrolling', 'lookup_tables'],
        'hardware_cycles': 'not_measured',
    }
    print('OK automatic_scalar_host ' + _auto_json.dumps(report, sort_keys=True))
    _AUTO_HOST_ORIGINAL(ctx, module, ','.join(passes[at+1:]), erase, mid, late, post_openmp, ())

_OBL_CALL_ROSTER = ()
_OBL_CALL_OUTPUT = None


def _obl_callee(op):
    if op.name != 'func.call' or op.results:
        return None
    try:
        return ir.FlatSymbolRefAttr(op.attributes['callee']).value
    except (ValueError, KeyError):
        return None


def _obl_plural_roots(op, indices, ignored):
    # A fresh allocation is disjoint from live caller-owned buffers. Unknown
    # owners and public output block arguments need a separate original ABI
    # disjointness premise and are deliberately outside this first scope.
    def root(value):
        seen = set()
        while isinstance(value, ir.OpResult):
            if value in seen:
                raise ValueError('cyclic plural result provenance')
            seen.add(value)
            owner = value.owner
            if owner.name in _OBL_VIEWS and owner.operands:
                value = owner.operands[0]
            elif owner.name == 'memref.alloc':
                return owner.operation
            else:
                raise ValueError('unknown plural caller buffer owner')
        return value
    roots = tuple(None if i in ignored else root(value) for i, value in enumerate(op.operands))
    outputs = tuple(roots[i] for i in indices)
    if (any(not isinstance(op.operands[i], ir.OpResult)
            or op.operands[i].owner.name != 'memref.alloc' for i in indices)
            or len(set(outputs)) != len(outputs)
            or any(output == value for i, value in enumerate(roots)
                   if i not in (*indices, *ignored) for output in outputs)):
        raise ValueError('plural caller results require distinct fresh allocation owners')


def _obl_borrowed_calls(module):
    if not _OBL_CALL_ROSTER:
        return set(), set(), []
    functions, supported, violations, plural_calls, checked_wrappers = {}, set(), [], {}, set()
    for op in module.body.operations:
        if op.operation.name != 'func.func':
            continue
        symbol = ir.StringAttr(op.attributes['sym_name']).value
        if symbol in functions:
            return set(), set(), ['borrowed ABI: duplicate current function symbol ' + symbol]
        functions[symbol] = op.operation
    for record in _OBL_CALL_ROSTER:
        try:
            effects = record['effects']
            if (not effects['qualification_identity'] or any(effects.get(k) is not True for k in (
                    'no_retained_aliases', 'no_releases', 'complete_before_return'))):
                raise ValueError('original complete borrowed effects are absent')
            wrapper_name, borrowed = record['wrapper_symbol'], record['borrowed_symbol']
            types = tuple(ir.Type.parse(t, context=module.context) for t in record['argument_types'])
            if any(not _obl_numeric_type(ir.MemRefType(t).element_type) for t in types):
                raise ValueError('borrowed ABI requires ranked numeric memory arguments')
            plural = 'result_arguments' in record
            result_indices = tuple(record['result_arguments']) if plural else (record['result_argument'],)
            if (not result_indices or len(set(result_indices)) != len(result_indices)
                    or any(type(i) is not int or not 0 <= i < len(types) for i in result_indices)):
                raise ValueError('original ordered result positions are invalid')
            declaration = functions.get(borrowed)
            if declaration is None:
                raise ValueError('original borrowed declaration is missing: ' + borrowed)
            typ = ir.FunctionType(ir.TypeAttr(declaration.attributes['function_type']).value)
            if (tuple(typ.inputs) != types or typ.results or any(r.blocks for r in declaration.regions)
                    or 'llvm.emit_c_interface' not in declaration.attributes):
                raise ValueError('current borrowed declaration/ABI differs: ' + borrowed)
            if plural:
                attrs = ir.ArrayAttr(declaration.attributes['arg_attrs'])
                if (len(attrs) != len(types) or any(
                        ir.StringAttr(ir.DictAttr(attr)['bufferization.access']).value
                        != ('write' if i in result_indices else 'read')
                        for i, attr in enumerate(attrs))):
                    raise ValueError('current plural readonly/write argument roles differ')
            wrapper = functions.get(wrapper_name)
            if wrapper is not None:
                typ = ir.FunctionType(ir.TypeAttr(wrapper.attributes['function_type']).value)
                if (tuple(typ.inputs) != (*types, *(types[i] for i in result_indices)) or typ.results
                        or len(wrapper.regions) != 1 or len(wrapper.regions[0].blocks) != 1):
                    raise ValueError('current fresh-writer wrapper ABI differs: ' + wrapper_name)
                block = wrapper.regions[0].blocks[0]
                body = tuple(op.operation for op in block.operations)
                replacements = {index: len(types) + position for position, index in enumerate(result_indices)}
                expected = tuple(block.arguments[replacements.get(i, i)] for i in range(len(types)))
                if (len(body) != 2 or _obl_callee(body[0]) != borrowed
                        or tuple(body[0].operands) != expected
                        or body[1].name != 'func.return' or body[1].operands):
                    raise ValueError('current fresh-writer forwarding body differs: ' + wrapper_name)
                supported.add(wrapper_name)
                if plural:
                    checked_wrappers.add(wrapper)
                    plural_calls[wrapper_name] = (
                        tuple(range(len(types), len(types) + len(result_indices))), result_indices)
            supported.add(borrowed)
            if plural:
                plural_calls[borrowed] = (result_indices, ())
        except (ValueError, KeyError, IndexError) as error:
            violations.append('borrowed ABI: ' + str(error))
    if violations:
        return set(), set(), violations
    # Bind grants to actual operations in checked root functions. A matching
    # flat spelling in a nested symbol table grants no permission.
    uses = set()
    scopes = ('scf.for', 'scf.if', 'linalg.generic', 'linalg.fill')
    for function in functions.values():
        for op in _obl_ops(function):
            name = _obl_callee(op)
            if name not in supported:
                continue
            parent = op.block.owner.operation
            while parent.name in scopes:
                parent = parent.block.owner.operation
            if parent != function:
                continue
            typ = ir.FunctionType(ir.TypeAttr(functions[name].attributes['function_type']).value)
            if tuple(v.type for v in op.operands) == tuple(typ.inputs):
                if name in plural_calls and function not in checked_wrappers:
                    try:
                        _obl_plural_roots(op, *plural_calls[name])
                    except ValueError as error:
                        violations.append('borrowed ABI: ' + str(error))
                        continue
                uses.add(op.operation)
            elif name in plural_calls:
                violations.append('borrowed ABI: current plural call operand types differ')
    return (set(), set(), violations) if violations else (uses, supported, [])


def _obl_known_use(op, calls):
    return op.name in _OBL_VIEWS | _OBL_USES or op.operation in calls


def _obl_write_observation(module, violations):
    if _OBL_CALL_OUTPUT is None:
        return
    import hashlib, json
    text = module.operation.get_asm(print_generic_op_form=True)
    with open(_OBL_CALL_OUTPUT['checkpoint'], 'w') as stream:
        stream.write(text)
    _, accepted, problems = _obl_borrowed_calls(module)
    version = 'v2' if any('result_arguments' in row for row in _OBL_CALL_ROSTER) else 'v1'
    report = dict(schema='merlin.borrowed_buffer_lifetime.' + version,
                  status='checked' if not violations and not problems else 'refused',
                  original=_OBL_CALL_OUTPUT['original'], violations=list(violations),
                  checkpoint_sha256=hashlib.sha256(text.encode()).hexdigest(),
                  accepted_symbols=sorted(accepted),
                  scope='conditional original borrowed effects and current typed forwarding bodies')
    with open(_OBL_CALL_OUTPUT['report'], 'w') as stream:
        json.dump(report, stream, sort_keys=True)
    with open(_OBL_CALL_OUTPUT['report'], 'rb') as stream:
        digest = hashlib.sha256(stream.read()).hexdigest()
    print('OK borrowed_buffer_lifetime', digest, report['checkpoint_sha256'])


_OBL_VIEWS = frozenset((
    'memref.cast', 'memref.subview', 'memref.reinterpret_cast',
    'memref.collapse_shape', 'memref.expand_shape', 'memref.extract_strided_metadata',
))
_OBL_USES = frozenset(('memref.copy', 'memref.load', 'memref.store', 'memref.dim',
                       'linalg.generic', 'linalg.fill'))
_OBL_SCALARS = frozenset((
    'arith.constant', 'arith.addi', 'arith.subi', 'arith.muli', 'arith.divsi',
    'arith.divui', 'arith.remsi', 'arith.remui', 'arith.andi', 'arith.ori',
    'arith.xori', 'arith.shli', 'arith.shrsi', 'arith.shrui', 'arith.cmpi',
    'arith.cmpf', 'arith.select', 'arith.index_cast', 'arith.index_castui',
    'arith.extsi', 'arith.extui', 'arith.trunci', 'arith.sitofp', 'arith.uitofp',
    'arith.fptosi', 'arith.fptoui', 'arith.extf', 'arith.truncf', 'arith.bitcast',
    'arith.addf', 'arith.subf', 'arith.mulf', 'arith.divf', 'arith.negf',
    'arith.maximumf', 'arith.minimumf', 'arith.maxnumf', 'arith.minnumf',
    'arith.maxsi', 'arith.minsi', 'arith.maxui', 'arith.minui',
    'math.exp', 'math.exp2', 'math.log', 'math.log2', 'math.sqrt', 'math.rsqrt',
    'math.absf', 'math.ceil', 'math.floor', 'math.roundeven', 'math.fma',
    'linalg.index', 'linalg.yield',
))


def _obl_constant(value):
    if not isinstance(value, ir.OpResult) or value.owner.name != 'arith.constant':
        return None
    if str(value.type) not in ('index', 'i1'):
        return None
    return ir.IntegerAttr(value.owner.attributes['value']).value


def _obl_ops(operation):
    for region in operation.regions:
        for block in region.blocks:
            for inner in block.operations:
                yield inner.operation
                yield from _obl_ops(inner.operation)


def _obl_numeric_type(typ):
    if str(typ) in ('index', 'f16', 'bf16', 'f32', 'f64'):
        return True
    try:
        ir.IntegerType(typ)
        return True
    except ValueError:
        return False


def _obl_memory_types(op):
    for value in (*op.operands, *op.results):
        if _memref_typed(value):
            try:
                element = ir.MemRefType(value.type).element_type
            except ValueError:
                raise ValueError('unranked or unsupported buffer representation') from None
            if not _obl_numeric_type(element):
                raise ValueError('buffer element may contain an escaping handle')
        elif not _obl_numeric_type(value.type):
            raise ValueError('unsupported nonnumeric memory operand or result')
    if op.name == 'memref.store' and _memref_typed(op.operands[0]):
        raise ValueError('buffer cannot be a stored payload')
    if op.name == 'linalg.fill' and any(_memref_typed(v) for v in op.operands[:-1]):
        raise ValueError('buffer cannot be a fill payload')
    if op.name == 'memref.load' and any(_memref_typed(v) for v in op.results):
        raise ValueError('buffer handle cannot be loaded')


def _obl_alias_uses(value):
    # Each view preserves its own provenance. A subview may be read, but it is
    # not an allocation owner that can be deallocated. Only casts/base extraction
    # of an owner preserve release authority.
    aliases, pending, uses = {value: True}, [value], {}
    while pending:
        current = pending.pop()
        for use in current.uses:
            op = use.owner.operation
            uses[op.operation] = op
            if op.name not in _OBL_VIEWS:
                continue
            if use.operand_number != 0:
                raise ValueError('buffer used as a non-source view operand')
            owner = aliases[current] and op.name in ('memref.cast', 'memref.extract_strided_metadata')
            for result in op.results:
                if not _memref_typed(result):
                    continue
                if result not in aliases:
                    aliases[result] = owner
                    pending.append(result)
    return aliases, tuple(uses.values())


def _obl_release_value(op, aliases):
    return (op.name == 'memref.dealloc' and len(op.operands) == 1
            and aliases.get(op.operands[0], False))


def _obl_scalar_loop(op):
    # This admits the serial container, never its captured effects. Descendants
    # remain in _obl_prove's complete effect walk and previous-owner uses still
    # have to precede their outer guarded release. Numeric iteration state
    # introduces no buffer generation, even for dynamic or zero trip counts.
    if (len(op.regions) != 1 or len(op.regions[0].blocks) != 1
            or len(op.operands) < 3 or len(op.attributes)):
        raise ValueError('unsupported nested scalar loop structure or attributes')
    body = op.regions[0].blocks[0]
    if any(not _obl_numeric_type(v.type)
           for v in (*op.operands, *op.results, *body.arguments)):
        raise ValueError('nested serial loop must carry only numeric scalars')
    operations = tuple(body.operations)
    if (not operations or operations[-1].operation.name != 'scf.yield'
            or tuple(v.type for v in operations[-1].operands)
            != tuple(v.type for v in op.results)):
        raise ValueError('unsupported nested scalar loop yield')
    op.verify()


def _obl_prove(loop, calls):
    loop = loop.operation
    body = loop.regions[0].blocks[0]
    operations = tuple(op.operation for op in body.operations)
    terminal = operations[-1]
    incoming, carried = tuple(loop.operands[3:]), tuple(body.arguments[1:])
    yielded = tuple(terminal.operands)
    changed = tuple(i for i, arg in enumerate(carried)
                    if _memref_typed(arg) and yielded[i] != arg)
    if not changed:
        return None
    _obl_memory_types(loop)
    if any(_memref_typed(v) for v in terminal.operands) and terminal.operation.name != 'scf.yield':
        raise ValueError('unsupported loop terminator')
    lo, hi, step = (_obl_constant(v) for v in loop.operands[:3])
    if lo is None or hi is None or step is None or step <= 0 or hi <= lo:
        raise ValueError('fresh carried ownership requires a statically positive serial trip count')
    fresh = {}
    for i in changed:
        value = yielded[i]
        if (not isinstance(value, ir.OpResult) or value.owner.name != 'memref.alloc'
                or value.owner.operation.block != body or len(value.owner.results) != 1):
            raise ValueError('each replaced carried buffer must yield its own direct body allocation')
        fresh[i] = value.owner.operation
    if len({op.operation for op in fresh.values()}) != len(fresh):
        raise ValueError('replaced carried buffers share a fresh owner')
    flags = {arg for i, arg in enumerate(carried)
             if str(arg.type) == 'i1' and _obl_constant(incoming[i]) == 0
             and _obl_constant(yielded[i]) == -1}
    # MLIR's integer attribute prints boolean true as -1 (signless i1).
    if not flags:
        raise ValueError('ownership must enter false and yield true')
    position = {op.operation: i for i, op in enumerate(operations)}
    all_inside = {op.operation: op for op in _obl_ops(loop)}
    guards = {}
    for op in operations:
        if op.operation.name != 'scf.if':
            continue
        if op.results or len(op.operands) != 1 or op.operands[0] not in flags:
            raise ValueError('only the original ownership flag may guard previous releases')
        then = tuple(op.regions[0].blocks[0].operations)
        if any(inner.operation.name != 'memref.dealloc' for inner in then[:-1]):
            raise ValueError('ownership branch must contain only previous-buffer releases')
        if then[-1].operation.name != 'scf.yield' or then[-1].operands:
            raise ValueError('unsupported ownership branch terminator')
        for region in op.regions[1:]:
            for block in region.blocks:
                if any(inner.operation.name != 'scf.yield' or inner.operands for inner in block.operations):
                    raise ValueError('ownership false branch must not release or use buffers')
        for inner in then[:-1]:
            guards[inner.operation] = op
    releases = set()
    for i in changed:
        aliases, uses = _obl_alias_uses(carried[i])
        frees = [op for op in uses if op.name == 'memref.dealloc']
        if len(frees) != 1 or not _obl_release_value(frees[0], aliases):
            raise ValueError('each previous buffer requires exactly one owner release')
        release = frees[0]
        guard = guards.get(release.operation)
        if guard is None:
            raise ValueError('previous buffer release is not guarded by its ownership flag')
        releases.add(release.operation)
        for use in uses:
            if use.operation not in all_inside:
                raise ValueError('previous generation escapes the loop')
            if use.operation == release.operation:
                continue
            if not _obl_known_use(use, calls):
                raise ValueError('unknown or escaping previous-buffer use: ' + use.name)
            ancestor = use
            while ancestor.block != body:
                ancestor = ancestor.block.owner.operation
            if position[ancestor.operation] >= position[guard.operation]:
                raise ValueError('previous buffer or view used after its guarded release')
        aliases, uses = _obl_alias_uses(yielded[i])
        for use in uses:
            if use.operation not in all_inside:
                raise ValueError('fresh generation escapes the loop')
            if use.operation == terminal.operation:
                if tuple(j for j, v in enumerate(yielded) if v in aliases) != (i,):
                    raise ValueError('fresh owner yielded in more than one position')
            elif not _obl_known_use(use, calls):
                raise ValueError('unknown, releasing or escaping fresh-buffer use: ' + use.name)
    if releases != set(guards):
        raise ValueError('ownership branch releases an unrelated or duplicate buffer')
    # Close every operation/effect, including operations with no tracked operand
    # and scalar regions that could capture a buffer without a parent operand.
    for op in all_inside.values():
        name = op.operation.name
        _obl_memory_types(op)
        if name in _OBL_SCALARS | _OBL_VIEWS | _OBL_USES:
            if name in _OBL_SCALARS and any(_memref_typed(v) for v in (*op.operands, *op.results)):
                raise ValueError('memref escapes through a scalar operation')
            continue
        if op.operation in calls:
            continue
        if name == 'scf.for':
            _obl_scalar_loop(op)
            continue
        if name in ('memref.alloc', 'scf.yield'):
            continue
        if name == 'scf.if' and op.block == body:
            continue
        if name == 'memref.dealloc':
            if op.operation in releases:
                continue
            value = op.operands[0]
            if (op.block == body and isinstance(value, ir.OpResult)
                    and value.owner.name == 'memref.alloc' and value.owner.operation.block == body
                    and value.owner.operation not in {a.operation for a in fresh.values()}):
                continue
        raise ValueError('unknown or unsupported loop effect: ' + name)
    # The final generation must have one local release after all synchronous
    # observations. Returning ownership or conditional release needs another proof.
    outer = tuple(loop.block.operations)
    outer_position = {op.operation: i for i, op in enumerate(outer)}
    for i in changed:
        aliases, uses = _obl_alias_uses(loop.results[i])
        frees = [op for op in uses if op.name == 'memref.dealloc']
        if (len(frees) != 1 or frees[0].block != loop.block
                or not _obl_release_value(frees[0], aliases)):
            raise ValueError('final fresh generation requires exactly one same-block owner release')
        for use in uses:
            if use.block != loop.block:
                raise ValueError('final generation escapes into unsupported control flow')
            if use.operation == frees[0].operation:
                continue
            _obl_memory_types(use)
            if not _obl_known_use(use, calls):
                raise ValueError('unknown or escaping final-buffer use: ' + use.name)
            for inner in _obl_ops(use):
                _obl_memory_types(inner)
                if inner.name not in _OBL_SCALARS or any(
                        _memref_typed(v) for v in (*inner.operands, *inner.results)):
                    raise ValueError('unknown or capturing final-buffer consumer effect: ' + inner.name)
            if outer_position[use.operation] >= outer_position[frees[0].operation]:
                raise ValueError('final buffer or view used after release')
    return fresh


def _owned_buffer_loops(order, calls):
    proved, violations = {}, []
    for loop in order:
        if loop.operation.name != 'scf.for':
            continue
        try:
            fact = _obl_prove(loop, calls)
            if fact:
                proved[loop.operation] = fact
        except ValueError as error:
            # Refuse unsupported replacements, rather than relying on the old
            # allocation-site union to accidentally produce a diagnostic.
            violations.append('loop-carried buffer ownership: ' + str(error))
    return proved, violations

def _memref_typed(v):
    return str(v.type).startswith('memref')


def _op_key(o):
    # OpView and Operation both answer `.operation`, and the bindings hand back equal (hashable)
    # objects for the same op however it was reached -- so this is a stable dict key.
    return o.operation


def _dealloc_placement_violations(module):
    # (violations, n_allocs, n_sunk) for a post-deallocation memref module.
    order, block_of, index_of, owner_of_block = [], {}, {}, {}
    counter = [0]

    def visit(block, owner):
        bid = counter[0]
        counter[0] += 1
        owner_of_block[bid] = owner
        for i, op in enumerate(block.operations):
            k = _op_key(op)
            block_of[k], index_of[k] = bid, i
            order.append(op)
            for region in op.regions:
                for b in region.blocks:
                    visit(b, op)

    for op in module.body.operations:
        for region in op.regions:
            for b in region.blocks:
                visit(b, op)

    calls, _, call_violations = _obl_borrowed_calls(module)
    proved, loop_violations = _owned_buffer_loops(order, calls)
    loop_violations.extend(call_violations)
    fresh_sites = {alloc.operation for fact in proved.values() for alloc in fact.values()}
    alias, roots, root_values, uses = {}, {}, {}, {}
    for op in order:
        name = op.operation.name
        if name == 'memref.alloc' and len(op.results) == 1:
            if op.operation in fresh_sites:
                continue  # all iteration-local uses were independently checked
            rid = len(roots)
            roots[rid] = op
            root_values[rid] = op.results[0]
            alias[op.results[0]] = {rid}
            continue
        hit = set()
        for v in op.operands:
            hit |= alias.get(v, set())
        if not hit and op.operation not in proved:
            continue
        for rid in hit:
            uses.setdefault(rid, []).append(op)
        if op.operation in proved:
            fact = proved[op.operation]
            # Non-replaced slots are positional pass-through aliases. Replaced
            # slots have a distinct final-generation owner after positive trips.
            body = op.regions[0].blocks[0]
            for i, (incoming, result) in enumerate(zip(op.operands[3:], op.results)):
                if not _memref_typed(result):
                    continue
                if i in fact:
                    rid = len(roots)
                    roots[rid], root_values[rid] = op, result
                    alias[result] = {rid}
                else:
                    value_roots = alias.get(incoming, set())
                    alias[result] = set(value_roots)
                    alias[body.arguments[i + 1]] = set(value_roots)
            continue
        for r in op.results:
            if _memref_typed(r):
                alias.setdefault(r, set()).update(hit)
        for region in op.regions:
            for b in region.blocks:
                for a in b.arguments:
                    if _memref_typed(a):
                        alias.setdefault(a, set()).update(hit)

    def ancestor_in_block(op, bid):
        o = op
        while True:
            k = _op_key(o)
            if k not in block_of:
                return None
            if block_of[k] == bid:
                return o
            parent = owner_of_block[block_of[k]]
            if parent is None:
                return None
            o = parent

    violations, sunk = list(loop_violations), 0
    for rid, alloc in roots.items():
        abid = block_of[_op_key(alloc)]
        # Moving an allocation outside a loop must also replan its free.
        # A same-block last-use check misses a retained per-iteration free.
        for use in uses.get(rid, []):
            if use.operation.name != 'memref.dealloc':
                continue
            bid = block_of[_op_key(use)]
            while bid != abid:
                parent = owner_of_block[bid]
                if parent is None:
                    break
                if parent.operation.name in ('scf.for', 'scf.while', 'scf.parallel', 'scf.forall',
                                              'affine.for', 'affine.parallel'):
                    violations.append(str(root_values[rid].type) +
                                      ': freed inside repeating control flow outside its allocation lifetime')
                    break
                bid = block_of[_op_key(parent)]
        here = [u for u in uses.get(rid, [])
                if u.operation.name == 'memref.dealloc' and block_of[_op_key(u)] == abid]
        if not here:
            continue                      # freed in a conditional/another block: not sunk
        sunk += 1
        d = min(here, key=lambda o: index_of[_op_key(o)])
        di = index_of[_op_key(d)]
        ty = str(root_values[rid].type)
        for u in uses.get(rid, []):
            if _op_key(u) == _op_key(d):
                continue
            anc = ancestor_in_block(u, abid)
            if anc is None:
                violations.append(ty + ': use by ' + u.operation.name +
                                  ' is outside the block its free was placed in')
            elif index_of[_op_key(anc)] > di:
                violations.append(ty + ': freed at #' + str(di) + ' but used at #' +
                                  str(index_of[_op_key(anc)]) + ' by ' + u.operation.name)
    return violations, len(roots), sunk

_ORIG_RUN_STAGES = _run_stages
_SINK_MARK = 'optimize-allocation-liveness'


def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(),
                pre_generalize=()):
    passes = [p for p in pipeline.split(',') if p]
    k = next((i for i, p in enumerate(passes) if _SINK_MARK in p), -1)
    if k < 0:
        return _ORIG_RUN_STAGES(ctx, module, pipeline, erase, mid, late, post_openmp,
                                pre_generalize)
    head, tail = passes[:k + 1], passes[k + 1:]
    # `erase`/`mid` open their window after buffer-loop-hoisting, and `late` opens its own before
    # convert-scf-to-openmp; hand each to whichever half contains its anchor, or it would be
    # accepted here and silently never run. A `late` whose anchor is in neither half goes to the
    # tail, where the wrapped runner still applies it at the end rather than dropping it.
    hoist_head = any('buffer-loop-hoisting' in p for p in head)
    late_head = any('convert-scf-to-openmp' in p for p in head)
    post_head = late_head
    _ORIG_RUN_STAGES(ctx, module, ','.join(head),
                     erase if hoist_head else 0, mid if hoist_head else (),
                     late if late_head else (), post_openmp if post_head else (),
                     pre_generalize)
    bad, n_alloc, n_sunk = _dealloc_placement_violations(module)
    _obl_write_observation(module, bad)
    print('OK dealloc_placement', n_alloc, 'allocations', n_sunk, 'sunk', len(bad), 'violations')
    if bad:
        raise RuntimeError('post-deallocation lifetime verification failed '
                           '(' + str(len(bad)) + ' violations):\n  ' + '\n  '.join(bad[:20]))
    if tail:
        _ORIG_RUN_STAGES(ctx, module, ','.join(tail),
                         0 if hoist_head else erase, () if hoist_head else mid,
                         () if late_head else late, () if post_head else post_openmp, ())

def _split_fresh_local_deallocs(ctx, module):
    from collections import defaultdict

    def is_alloc(value):
        owner = value.owner
        return hasattr(owner, 'operation') and owner.operation.name == 'memref.alloc'

    def is_device_output_alias(op, declarations):
        name = str(op.attributes['callee']).lstrip('@')
        decl = declarations.get(name)
        return (
            name.startswith('merlin_dev_') and decl is not None
            and len(op.operands) >= 3 and len(op.results) == 1
            and str(op.results[0].type) == str(op.operands[-1].type)
            and len(decl.arg_attrs) == len(op.operands)
            and all('bufferization.access' in attrs for attrs in decl.arg_attrs)
            and [str(attrs['bufferization.access']) for attrs in decl.arg_attrs]
            == ['"read"'] * (len(op.operands) - 1) + ['"write"']
        )

    def nested_operations(parent):
        for region in parent.regions:
            for block in region.blocks:
                for op in block.operations:
                    yield op
                    yield from nested_operations(op)

    declarations = {
        str(op.attributes['sym_name']).strip('"'): op
        for op in module.body.operations if op.operation.name == 'func.func'
    }
    moved = dropped_aliases = 0
    for func in module.body.operations:
        if func.operation.name != 'func.func' or not func.regions[0].blocks:
            continue
        # An unrecognized callee may retain a memref even when it returns void.
        # Search nested regions too: a call in a branch can outlive a top-level
        # aggregate dealloc just as a direct call can.
        unknown_memref_call = any(
            op.operation.name == 'func.call'
            and any(str(value.type).startswith('memref') for value in (*op.operands, *op.results))
            and not is_device_output_alias(op, declarations)
            for op in nested_operations(func)
        )
        for block in func.regions[0].blocks:
            # Merlin's generated device shim returns its last, writable output
            # descriptor. Earlier operands are read-only, including extra group
            # inputs such as a bias; the shim retains no pointer. Check the exact
            # declaration contract before treating the result as an alias.
            call_alias = {}
            for op in block.operations:
                if op.operation.name != 'func.call' or not any(
                    str(result.type).startswith('memref') for result in op.results
                ):
                    continue
                if is_device_output_alias(op, declarations):
                    call_alias[op.results[0]] = op.operands[-1]
            for old in list(block.operations):
                if old.operation.name != 'bufferization.dealloc' or old.results or len(old.operands) % 2:
                    continue
                n = len(old.operands) // 2
                mems, conds = list(old.operands[:n]), list(old.operands[n:])
                groups = defaultdict(list)
                # An unknown result alias could duplicate a fresh root. Keep the
                # whole aggregate under upstream's ownership semantics in that case.
                if unknown_memref_call:
                    continue
                ambiguous = False
                for i, mem in enumerate(mems):
                    owner = mem.owner
                    if not hasattr(owner, 'operation') or owner.operation.name != 'memref.extract_strided_metadata':
                        ambiguous = True
                        break
                    base = owner.operands[0]
                    if is_alloc(base):
                        groups[base].append((i, 'direct'))
                    elif base in call_alias and is_alloc(call_alias[base]):
                        groups[call_alias[base]].append((i, 'call_alias'))
                    elif not isinstance(base, ir.BlockArgument):
                        ambiguous = True
                        break
                if ambiguous:
                    continue
                chosen = {}
                for root, entries in groups.items():
                    direct = [i for i, kind in entries if kind == 'direct']
                    if len(direct) != 1:
                        continue
                    cond_op = conds[direct[0]].owner
                    if (not hasattr(cond_op, 'operation') or cond_op.operation.name != 'arith.constant'
                            or str(cond_op.attributes['value']) != 'true'):
                        continue
                    chosen[root] = {i for i, _ in entries}
                if not chosen:
                    continue
                removed = set().union(*chosen.values())
                keep = [(mem, cond) for i, (mem, cond) in enumerate(zip(mems, conds)) if i not in removed]
                with ctx, old.location, ir.InsertionPoint(old):
                    for root in chosen:
                        ir.Operation.create('memref.dealloc', operands=[root])
                    if keep:
                        ir.Operation.create(
                            'bufferization.dealloc',
                            operands=[*(mem for mem, _ in keep), *(cond for _, cond in keep)],
                            attributes={'operandSegmentSizes': ir.DenseI32ArrayAttr.get(
                                [len(keep), len(keep), 0])})
                old.operation.erase()
                moved += len(chosen)
                dropped_aliases += len(removed) - len(chosen)
    module.operation.verify()
    print('OK split_fresh_local_deallocs', moved, 'fresh allocs', dropped_aliases, 'device aliases')

_SPLIT_ORIG_RUN_STAGES = _run_stages


def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(),
                pre_generalize=()):
    passes = [p for p in pipeline.split(',') if p]
    if '__merlin_split_fresh_deallocs__' not in passes:
        return _SPLIT_ORIG_RUN_STAGES(ctx, module, pipeline, erase, mid, late,
                                      post_openmp, pre_generalize)
    if passes.count('__merlin_split_fresh_deallocs__') != 1:
        raise RuntimeError('aggregate dealloc split marker must appear exactly once')
    at = passes.index('__merlin_split_fresh_deallocs__')
    head, tail = passes[:at], passes[at + 1:]
    if 'ownership-based-buffer-deallocation' not in head:
        raise RuntimeError('aggregate dealloc split must follow ownership analysis')
    if any('convert-scf-to-openmp' in p for p in head):
        raise RuntimeError('aggregate dealloc split must precede parallel control-flow lowering')
    hoist_head = any('buffer-loop-hoisting' in p for p in head)
    _SPLIT_ORIG_RUN_STAGES(ctx, module, ','.join(head),
                           erase if hoist_head else 0, mid if hoist_head else (), (), (),
                           pre_generalize)
    _split_fresh_local_deallocs(ctx, module)
    _SPLIT_ORIG_RUN_STAGES(ctx, module, ','.join(tail),
                           0 if hoist_head else erase, () if hoist_head else mid, late,
                           post_openmp, ())

def _slice_unused_elementwise_outputs(ctx, module):
    changed = 0
    for func in module.body.operations:
        if func.operation.name != 'func.func' or not func.regions[0].blocks:
            continue
        for block in func.regions[0].blocks:
            for alloc in list(block.operations):
                if alloc.operation.name != 'memref.alloc' or len(alloc.results) != 1:
                    continue
                value = alloc.results[0]
                users = [use.owner for use in value.uses]
                if len(users) != 3:
                    continue
                generics = [op for op in users if op.operation.name == 'linalg.generic']
                subviews = [op for op in users if op.operation.name == 'memref.subview']
                deallocs = [op for op in users if op.operation.name == 'memref.dealloc']
                if len(generics) != 1 or len(subviews) != 1 or len(deallocs) != 1:
                    continue
                generic, subview, dealloc = generics[0], subviews[0], deallocs[0]
                if len(generic.operands) != 2 or generic.operands[1] != value or generic.results:
                    continue
                source = generic.operands[0]
                old_ty, src_ty = ir.MemRefType(value.type), ir.MemRefType(source.type)
                if list(old_ty.shape) != list(src_ty.shape):
                    continue
                rank = len(old_ty.shape)
                maps = generic.attributes['indexing_maps']
                if len(maps) != 2 or any(
                    m.value != ir.AffineMap.get_identity(rank, context=ctx) for m in maps
                ):
                    continue
                if any(
                    str(it) != '#linalg.iterator_type<parallel>'
                    for it in generic.attributes['iterator_types']
                ):
                    continue
                if any(
                    child.operation.name != 'linalg.yield'
                    and not child.operation.name.startswith(('arith.', 'math.'))
                    for region in generic.regions
                    for body in region.blocks
                    for child in body.operations
                ):
                    continue
                if len(subview.operands) != 1 or subview.operands[0] != value or len(subview.results) != 1:
                    continue
                offsets = list(subview.attributes['static_offsets'])
                sizes = list(subview.attributes['static_sizes'])
                steps = list(subview.attributes['static_strides'])
                if (
                    len(offsets) != rank or len(sizes) != rank or len(steps) != rank
                    or len(ir.MemRefType(subview.results[0].type).shape) != rank
                    or any(x < 0 for x in offsets)
                    or any(x <= 0 for x in sizes + steps)
                ):
                    continue
                if sizes == list(old_ty.shape) or any(
                    o + (s - 1) * step >= dim
                    for o, s, step, dim in zip(offsets, sizes, steps, old_ty.shape)
                ):
                    continue
                if any(step != 1 for step in steps):
                    continue
                sub_users = [use.owner for use in subview.results[0].uses]
                if len(sub_users) != 1 or sub_users[0].operation.name != 'memref.collapse_shape':
                    continue
                collapse = sub_users[0]
                collapse_users = [use.owner for use in collapse.results[0].uses]
                if (
                    len(collapse_users) != 1
                    or collapse_users[0].operation.name != 'memref.copy'
                    or collapse_users[0].operands[0] != collapse.results[0]
                ):
                    continue
                strides, base_off = src_ty.get_strides_and_offset()
                if base_off < 0 or any(stride < 0 for stride in strides):
                    continue
                result_strides = [stride * step for stride, step in zip(strides, steps)]
                result_off = base_off + sum(o * stride for o, stride in zip(offsets, strides))
                with ctx, alloc.location, ir.InsertionPoint(generic):
                    slice_type = ir.MemRefType.get(
                        sizes, src_ty.element_type,
                        layout=ir.StridedLayoutAttr.get(result_off, result_strides),
                    )
                    input_slice = ir.Operation.create(
                        'memref.subview', results=[slice_type], operands=[source],
                        attributes={
                            'static_offsets': ir.DenseI64ArrayAttr.get(offsets),
                            'static_sizes': ir.DenseI64ArrayAttr.get(sizes),
                            'static_strides': ir.DenseI64ArrayAttr.get(steps),
                            'operandSegmentSizes': ir.DenseI32ArrayAttr.get([1, 0, 0, 0]),
                        },
                    )
                    small_type = ir.MemRefType.get(sizes, old_ty.element_type)
                    attrs = {'alignment': alloc.attributes['alignment']} if 'alignment' in alloc.attributes else {}
                    small = ir.Operation.create('memref.alloc', results=[small_type], attributes=attrs)
                generic.operands[0] = input_slice.results[0]
                generic.operands[1] = small.results[0]
                with ctx, collapse.location, ir.InsertionPoint(collapse):
                    collapsed_ty = ir.MemRefType.get(
                        list(ir.MemRefType(collapse.results[0].type).shape), old_ty.element_type,
                    )
                    new_collapse = ir.Operation.create(
                        'memref.collapse_shape', results=[collapsed_ty],
                        operands=[small.results[0]],
                        attributes={'reassociation': collapse.attributes['reassociation']},
                    )
                collapse.results[0].replace_all_uses_with(new_collapse.results[0])
                dealloc.operands[0] = small.results[0]
                collapse.operation.erase()
                subview.operation.erase()
                alloc.operation.erase()
                changed += 1
    module.operation.verify()
    print('OK sliced_elementwise_output', changed, flush=True)

_SLICE_ORIG_RUN_STAGES = _run_stages


def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(),
                pre_generalize=()):
    passes = [p for p in pipeline.split(',') if p]
    marker = '__merlin_slice_unused_elementwise_output__'
    if marker not in passes:
        return _SLICE_ORIG_RUN_STAGES(ctx, module, pipeline, erase, mid, late,
                                      post_openmp, pre_generalize)
    if passes.count(marker) != 1:
        raise RuntimeError('output-slice marker must appear exactly once')
    at = passes.index(marker)
    head, tail = passes[:at], passes[at + 1:]
    if not any('optimize-allocation-liveness' in p for p in head):
        raise RuntimeError('output-slice rewrite requires checked deallocation sinking')
    if any('convert-scf-to-openmp' in p for p in head):
        raise RuntimeError('output-slice rewrite must precede parallel control-flow lowering')
    hoist_head = any('buffer-loop-hoisting' in p for p in head)
    _SLICE_ORIG_RUN_STAGES(ctx, module, ','.join(head),
                           erase if hoist_head else 0, mid if hoist_head else (), (), (), pre_generalize)
    _slice_unused_elementwise_outputs(ctx, module)
    bad, n_alloc, n_sunk = _dealloc_placement_violations(module)
    print('OK sliced_dealloc_placement', n_alloc, 'allocations', n_sunk, 'sunk', len(bad), 'violations')
    if bad:
        raise RuntimeError('output-slice rewrite invalidated deallocation placement: '
                           + '; '.join(bad[:20]))
    if tail:
        _SLICE_ORIG_RUN_STAGES(ctx, module, ','.join(tail),
                               0 if hoist_head else erase, () if hoist_head else mid, late,
                               post_openmp, ())


def _residual_vector_ops(module):
    """Return vector-dialect op names still present at the LLVM translation edge."""
    found = []

    def visit(op):
        name = op.operation.name
        if name.startswith('vector.'):
            found.append(name)
        for region in op.regions:
            for block in region.blocks:
                for child in block.operations:
                    visit(child)

    for top in module.body.operations:
        visit(top)
    return sorted(set(found))

src_path, out_path, pipeline = sys.argv[1], sys.argv[2], sys.argv[3]
ctx = ir.Context()
with open(src_path) as f:
    module = ir.Module.parse(f.read(), ctx)

# int_softmax_table (default-off): restructure every integer softmax in the captured IR, exactly,
# before any other rewrite sees it. See llvmlower/int_softmax_table.py.
if _INT_SOFTMAX_TABLE:
    _ist_run_and_report(ctx, module)
# fuse_transpose_b (default-off): fold `matmul(A, transpose(B))` into a transpose-b matmul BEFORE
# the pass manager runs, so the (still-named) linalg.matmul carries the transposed-B indexing map
# and the frozen RVV schedule tiles+vectorizes it while the scalar weight transpose disappears.
if _FUSE_TRANSPOSE_B:
    print("OK fuse_transpose_b", _fuse_transpose_b(module, ctx))
# fold_weight_transpose (default-off): the general form of the fold above -- a loop-invariant weight
# transpose sinks into the indexing_maps of EVERY linalg consumer, not just a linalg.matmul's B
# operand. The quantized datapath emits its contraction as a linalg.generic, so the matmul-only fold
# fires zero times on an int8 model; this one folds all 15 of small_llama's weight transposes.
if _FOLD_WEIGHT_TRANSPOSE:
    _fwt_n, _fwt_report = _fold_weight_transposes(module, ctx)
    for _fwt_kind, _fwt_detail in _fwt_report:
        print("OK fold_weight_transpose", _fwt_kind, _fwt_detail)
    print("OK fold_weight_transpose folded", _fwt_n)
# fold_broadcast_into_generic (default-off): replace a sole-use broadcast intermediate by a
# projected indexing map on an already-all-parallel generic. Reduction consumers are refused.
if _FOLD_BROADCAST:
    _bf_n, _bf_report = _fold_broadcasts(module, ctx)
    for _bf_kind, _bf_detail in _bf_report:
        print("OK fold_broadcast_into_generic", _bf_kind, _bf_detail)
    print("OK fold_broadcast_into_generic folded", _bf_n)
# concat_dps (default-off): a `tensor.concat` operand produced by a destination-passing op is
# retargeted to write STRAIGHT INTO the concatenated buffer, so bufferization has no data movement
# left to emit for it. Must run BEFORE the pass manager: after one-shot-bufferize the destination is
# no longer an operand, only a copy. See llvmlower/concat_dps.py.
if _CONCAT_DPS:
    _cd_n, _cd_report = _concat_dps(module, ctx)
    for _cd_kind, _cd_detail in _cd_report:
        print("OK concat_dps", _cd_kind, _cd_detail)
    print("OK concat_dps rewrote", _cd_n)
_run_stages(ctx, module, pipeline, _ERASE_SELF_COPY, _MID_STAGES, _LATE_STAGES,
            _POST_OPENMP_STAGES, _PRE_GENERALIZE_STAGES)
_vector_residue = _residual_vector_ops(module)
if _vector_residue:
    raise RuntimeError(
        'vector dialect survived the LLVM lowering edge: ' + ', '.join(_vector_residue))

with open(out_path, "w") as f:
    len(sys.argv) > 17 and sys.argv[17] and module.operation.attributes.__setitem__('llvm.data_layout', ir.StringAttr.get(sys.argv[17], ctx)); f.write(str(llvm.translate_module_to_llvmir(module.operation)))
print("OK")
