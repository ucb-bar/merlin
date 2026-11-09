"""Native pass-manager stage execution, embedded in the selected compiler process.

The native bindings belong to that process, not Merlin's Python environment.
This source invokes its existing PassManager and preserves the established
bufferization and OpenMP extension windows. It does not select passes, resolve
toolchains, or admit generated output. The caller supplies any selected rewrite
callbacks (including the existing self-copy callback).

Keep this source byte-stable when relocating it: generated runner source is an
input to lowering receipts and frozen reproduction.
"""

STAGED_RUNNER = r'''def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(),
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


'''
