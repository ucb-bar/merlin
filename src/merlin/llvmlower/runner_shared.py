"""Target-independent runner fragments shared with out-of-tree codegen packages."""

from .uniform_fill_copy import MID_STAGE_SRC as _UNIFORM_FILL_COPY_MID_SRC
from .uniform_fill_copy import RUNNER_PRELUDE as _UNIFORM_FILL_COPY_PRELUDE

# The normal runner reserves argv[17] for the target data layout.
_UNIFORM_FILL_COPY_MID_SRC = _UNIFORM_FILL_COPY_MID_SRC.replace(
    'if len(sys.argv) > 17 and sys.argv[17] == "1":',
    'if len(sys.argv) > 18 and sys.argv[18] == "1":',
)

# --- The use-after-free check that guards the sinking stage ---------------------------------
#
# Sinking a dealloc is the one edit here whose failure mode is WRONG NUMBERS, not a broken build: a
# free moved before a use is a use-after-free, and with the static arena bound on top of it the
# freed bytes are immediately handed to another buffer. `optimize-allocation-liveness` decides where
# a free may go from upstream's `BufferViewFlowAnalysis`; this re-checks the RESULT, on the IR the
# stage actually produced, in terms that do not depend on that analysis being right.
#
# The rule: for every `memref.alloc` whose `memref.dealloc` the stage placed in the alloc's OWN block
# (an allocation still freed inside an `scf.if` was not sunk -- upstream placed it, and it is left
# alone), every use of the allocation AND of every value derived from it must come before that free.
#
# "Derived from" is deliberately OVER-approximated: any op result of memref type, and any memref
# block argument of a region, of an op that consumes an aliasing value joins the alias set. An
# over-approximation can only make the check REFUSE a placement it could have allowed; it cannot let
# a real use-after-free through by narrowing what counts as a use. Anything it cannot read -- a use
# whose ancestor chain does not reach the alloc's block -- is reported, never skipped.
DEALLOC_CHECK_PRELUDE = r"""
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

    alias, roots, uses = {}, {}, {}
    for op in order:
        name = op.operation.name
        if name == 'memref.alloc' and len(op.results) == 1:
            rid = len(roots)
            roots[rid] = op
            alias[op.results[0]] = {rid}
            continue
        hit = set()
        for v in op.operands:
            hit |= alias.get(v, set())
        if not hit:
            continue
        for rid in hit:
            uses.setdefault(rid, []).append(op)
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

    violations, sunk = [], 0
    for rid, alloc in roots.items():
        abid = block_of[_op_key(alloc)]
        here = [u for u in uses.get(rid, [])
                if u.operation.name == 'memref.dealloc' and block_of[_op_key(u)] == abid]
        if not here:
            continue                      # freed in a conditional/another block: not sunk
        sunk += 1
        d = min(here, key=lambda o: index_of[_op_key(o)])
        di = index_of[_op_key(d)]
        ty = str(alloc.results[0].type)
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
"""

#: Runner glue: wrap ``_run_stages`` so the placement check runs on the IR the sinking stage
#: produced, in EVERY runner variant, without any of them having to know about it. The wrapper
#: delegates unchanged when the pass list carries no sinking stage.
DEALLOC_CHECK_RUNNER = r"""
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
    print('OK dealloc_placement', n_alloc, 'allocations', n_sunk, 'sunk', len(bad), 'violations')
    if bad:
        raise RuntimeError('dealloc placement moved a free before a use of the buffer '
                           '(' + str(len(bad)) + ' violations):\n  ' + '\n  '.join(bad[:20]))
    if tail:
        _ORIG_RUN_STAGES(ctx, module, ','.join(tail),
                         0 if hoist_head else erase, () if hoist_head else mid,
                         () if late_head else late, () if post_head else post_openmp, ())
"""

#: The line the runner prints once the check has run. `lower_to_llvm_ir` REQUIRES it whenever the
#: pipeline carries the sinking stage: a runner variant that drove the PassManager itself and never
#: reached the wrapper would otherwise sink the frees with nothing checking them, and report success.
DEALLOC_CHECK_TOKEN = "OK dealloc_placement"
