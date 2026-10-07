"""Keep elementwise fusion from re-evaluating a producer once per element of a broadcast.

Upstream ``linalg-fuse-elementwise-ops`` inlines an all-parallel producer's body into its consumer
whenever the producer has one use. Through a BROADCAST that multiplies the producer's work: a value
with one entry per row (a layer norm's mean and reciprocal standard deviation, a rescale's
``log``/``pow``, an integer square root's Newton steps), read by every element of its row, is
recomputed for every element. On one SmolVLA SigLIP encoder layer this made the GELU cost ~590
retired instructions per element and each layer norm ~350.

The rule this module adds to the fusion is a control function: **refuse to fuse a producer whose
fused body would run more times than the producer has elements** (the consumer's iteration space is
larger than the operand it reads), unless the body computes nothing (it only yields block arguments
and constants, so fusing it is pure index remapping). Everything else fuses exactly as upstream does.

HOW, given that the pass takes no control option and the Python bindings cannot hand it one: the
pass's own control function is ``producer->hasOneUse()``. Immediately before each run of the pass the
runner gives every refused producer a second use, an unregistered ``merlin.fusion_pin`` op (unknown
side effects, so no pattern erases it), and immediately after the pass it erases every pin. Fusion of
OTHER producers into a pinned one (the per-row chain collapsing into one per-row op) is unaffected:
the default control function reads only the producer's uses.

It is applied by wrapping the PassManager in the runner (:data:`RUNNER_PRELUDE`), so every runner
variant and every stage that runs the pass is covered, including feature-spliced fusion stages, without
the pass list changing spelling. ``MERLIN_FUSION_GUARD=0`` disables it, for A/B measurement.
"""

from __future__ import annotations

import os

#: The pass the guard wraps, exactly as every pipeline in this package spells it.
FUSE_PASS = "func.func(linalg-fuse-elementwise-ops)"
#: The line the runner prints after each guarded run: ``OK fusion_guard <pinned> pinned``.
TOKEN = "OK fusion_guard"


def enabled() -> bool:
    """On unless deselected (``--no-pass fusion-guard``) or ``MERLIN_FUSION_GUARD`` is ``0``."""
    from .optional_passes import switched

    return switched("fusion-guard", os.environ.get("MERLIN_FUSION_GUARD", "1") != "0")


# Self-contained: executes in the compiler venv (torch_mlir), not Merlin's Python environment. These
# bindings downcast values, types and affine expressions themselves, so plain isinstance works. It must
# not contain the text "PassManager.parse(": the IR-inspection binder rewrites that spelling.
RUNNER_PRELUDE = r'''
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
'''


def runner_prelude() -> str:
    """The prelude, bound to whether the guard is enabled for this build."""
    return f"\n_FG_ENABLED = {enabled()!r}\n" + RUNNER_PRELUDE


def inject(source: str) -> str:
    """``source`` (a runner script) with the guard installed ahead of its own code.

    Prepended, so the runner's own ``from torch_mlir.passmanager import PassManager`` (and every local
    import of it inside a stage) resolves to the guarded manager."""
    return runner_prelude() + source
