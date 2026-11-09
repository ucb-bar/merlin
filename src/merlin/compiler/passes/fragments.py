"""Shared native runner fragments preserving current lifetime admission."""

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
    _obl_write_observation(module, bad)
    print('OK dealloc_placement', n_alloc, 'allocations', n_sunk, 'sunk', len(bad), 'violations')
    if bad:
        raise RuntimeError('post-deallocation lifetime verification failed '
                           '(' + str(len(bad)) + ' violations):\n  ' + '\n  '.join(bad[:20]))
    if tail:
        _ORIG_RUN_STAGES(ctx, module, ','.join(tail),
                         0 if hoist_head else erase, () if hoist_head else mid,
                         () if late_head else late, () if post_head else post_openmp, ())
"""

