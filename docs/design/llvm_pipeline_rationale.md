---
title: "LLVM lowering pipeline rationale"
kind: design
status: current
owner: core
last_verified: 2026-10-16
related: [runtime]
code_refs: [src/merlin/llvmlower/pipeline.py]
---

# LLVM lowering pipeline rationale

The named functions in `src/merlin/llvmlower/pipeline.py` implement these choices.
The text below retains their detailed rationale alongside the public pipeline code.

## `_dealloc_passes`

Free the buffers ``one-shot-bufferize`` allocates.

Bufferization turns every ``tensor.empty`` into a ``memref.alloc``; it does NOT insert the
matching frees — that is a separate pass, and it was missing from all three pipelines here. The
lowered whole-model ``forward`` therefore called ``malloc`` once per intermediate and ``free``
never, so a model's heap demand was the SUM of every temporary it ever materialized rather than
the peak that is live at once. Measured on whisper_tiny: 678 ``@malloc`` calls, 0 ``@free``.

Small models absorbed it (deepjscc and spectformer fit their cumulative total in the region we
size for them, which is why they pass). whisper does not: its encoder attention materializes
6x1500x1500 tensors, i.e. 54 MB apiece, and the run dies partway through the first attention
block when a 54 MB ``malloc`` returns NULL and the generated code stores through it — observed
on FireSim as ``mcause=7 Store/AMO access fault`` at op 337 with the destination register zero.
No region size fixes that on a 512 MB board; the allocations had to stop leaking.

These are upstream's own deallocation passes — ownership-based deallocation, then the
simplification and lowering passes that turn ``bufferization.dealloc`` into ``memref.dealloc`` —
inserted immediately after bufferization as its documentation prescribes.

Upstream's composite ``buffer-deallocation-pipeline`` is NOT usable here: it interleaves
``canonicalize``, and at this point in the RVV pipeline the module still holds ``vector.mask``
regions that have not been through ``lower-vector-mask``. Canonicalization sinks a constant into
one of those regions, which then fails its own verifier (``'vector.mask' op expects only one
operation to mask``) — whisper_tiny hits it, deepjscc does not. So the three passes are named
individually and the cleanup is left to the ``canonicalize``/``cse`` the pipeline already runs at
the end, after the vector ops have been lowered.

``MERLIN_NO_DEALLOC`` restores the previous behavior for an A/B — it exists because this changes
generated code for every model, and the honest way to defend "numerics unchanged" is to be able
to rebuild both.

``MERLIN_SINK_DEALLOCS`` (or ``sink=True``) appends the sinking stage described in
:func:`_sink_dealloc_passes`, which moves direct frees from the end of the function to just
after the buffer's last use. An aggregate ``bufferization.dealloc`` can still pin many local
buffers until return; ``MERLIN_SPLIT_FRESH_DEALLOCS`` controls a guarded split for that case
and also enables the sinking and placement check. The guarded split and output-slice rewrite
are enabled by default; set either switch to ``0`` for an A/B build.

## `_sink_dealloc_passes`

Move each free from function exit to just after the buffer's last use.

CORRECTING WHAT THIS COMMENT USED TO SAY. It recorded that
``func.func(optimize-allocation-liveness)`` "moved nothing" and concluded that
``ownership-based-buffer-deallocation`` "is already placing the frees tightly enough that there is
no slack to reclaim". The first half was an accurate observation; the second half was the wrong
explanation, and the emitted IR contradicts it. The frees are not tight — they are ALL at the end
of ``@forward``: measured on the deepjscc int8 build, the last ``malloc`` is at line 10,530 of a
12,112-line function and every ``free`` sits after the last compute. The static arena binder
(:mod:`merlin.llvmlower.arena_bind`), which can only reuse bytes between a free and a later
malloc, therefore gets 1.00x reuse on all three int8 models, while planning the SAME buffers under
last-use liveness gives 5.77x / 7.67x / 11.73x. The slack was real; the pass simply could not see
it.

WHY IT MOVED NOTHING. ``bufferization-lower-deallocations`` does not emit a bare
``memref.dealloc``: it emits the dealloc inside an ``scf.if`` guarded by the buffer's ownership
flag, which ``buffer-deallocation-simplification`` has already folded to a constant ``true``::

    scf.if %true {
      memref.dealloc %alloc : memref<64x64xf32>
    }

``OptimizeAllocationLiveness`` looks for a user of the allocation with a Free memory effect and
bails unless that user is in the SAME BLOCK as the alloc — deliberately, so it never hoists a
dealloc out of a conditional (llvm-project ``mlir/lib/Dialect/Bufferization/Transforms/
OptimizeAllocationLiveness.cpp``, ``deallocOp->getBlock() != allocOp->getBlock()``). The dealloc
is in the ``scf.if``'s body block, so EVERY allocation is skipped and the pass is a guaranteed
no-op wherever it is placed after the lowering — and, measured, equally a no-op placed BEFORE it.
Adding it changed nothing not because the frees were tight but because it could not read them.

THE FIX is the ``canonicalize`` this pipeline dropped. Upstream's own composite
``buffer-deallocation-pipeline`` runs ``cse`` + ``canonicalize`` AFTER ``lower-deallocations``
(llvm-project ``mlir/lib/Dialect/Bufferization/Pipelines/BufferizationPipelines.cpp:36-38``); this
pipeline names the three passes individually — for the ``vector.mask`` reason in
:func:`_dealloc_passes` — and lost that cleanup with them. Canonicalization folds the
statically-true ``scf.if`` away, leaving the ``memref.dealloc`` a direct user of the alloc in the
alloc's own block, and the liveness pass then moves those direct frees to their last uses.
Whole-model bufferization may instead form one aggregate ``bufferization.dealloc`` for
hundreds of allocations. The opt-in fresh-local split handles that separate case.

ORDER IS LOAD-BEARING and is asserted by ``merlin/tests/ir/test_dealloc_sinking.py``:
``bufferization-lower-deallocations`` → ``canonicalize`` → ``optimize-allocation-liveness``.
Placed before the lowering, or after it without the canonicalize, the pass silently moves nothing
— which is exactly how this was mis-diagnosed the first time.

The canonicalize is safe HERE for the same reason the composite pipeline was not: it runs where
``__DEALLOC__`` is spliced, which in the RVV pipeline is already past ``lower-vector-mask`` and
``convert-vector-to-scf`` (no ``vector.mask`` region left to sink a constant into), and in the
scalar pipelines the module carries no vector ops at all.

Sinking a free is the one edit in this file that can produce SILENTLY WRONG NUMBERS rather than a
build failure: a dealloc moved before a use is a use-after-free, and with the arena bound on top
of it the freed bytes are handed to another buffer. So the flagged path does not merely trust the
pass — the runner re-checks the placement structurally on the IR the stage produced (every use of
the allocation and of every value derived from it must precede its dealloc) and FAILS THE BUILD
on a violation. See :data:`DEALLOC_CHECK_PRELUDE`.

## `_generalize_before_fuse`

Whether ``linalg-generalize-named-ops`` runs BEFORE ``linalg-fuse-elementwise-ops``.

Default OFF (``MERLIN_GENERALIZE_BEFORE_FUSE=1``), so every unflagged build -- including the
``MERLIN_FUSE_POST`` / ``fuse_elementwise_post_contraction`` arm, whose in-tree numbers were all
measured in the current order -- is byte-identical to the frozen baseline.

WHY THE ORDER MATTERS, and it is not a style question. Upstream's ``FuseElementwiseOps`` pattern
is an ``OpRewritePattern<linalg::GenericOp>`` whose producer must ALSO be a ``linalg.GenericOp``
(``mlir/lib/Dialect/Linalg/Transforms/ElementwiseOpFusion.cpp``). A ``linalg.broadcast`` /
``linalg.transpose`` / ``linalg.copy`` is therefore INVISIBLE to it: not declined, not counted --
never matched. So a fusion stage placed in front of the generalization is blind to exactly the ops
the stage exists to remove.

MEASURED on the three int8 recaptures, at the point in ``build_rvv_pipeline`` where the stage
runs: ``linalg-specialize-generic-ops`` turns 0 ``linalg.broadcast`` into 116 (deepjscc), 64
(small_llama) and 246 (lstmnetvit) -- and every one of them is a named op that the very next
pass, in the current order, cannot see.

Deliberately NOT auto-coupled to the fusion lever the way ``vectorize_non_contraction_generics``
is coupled to its self-copy erase. The self-copy erase CANCELS its lever's payoff, so shipping
them apart makes the lever inert; this one CHANGES what the fusion stage does, and every number
recorded for ``fuse_elementwise_post_contraction`` in ``impr_features`` was measured without it.
Silently re-ordering underneath that feature would invalidate its recorded evidence rather than
complete it, so the two are measured separately and coupled only once the board says which wins.

## `_reorder_generalize_before_fuse`

Move ``linalg-generalize-named-ops`` in front of the ``linalg-fuse-elementwise-ops`` it follows.

One rule for all three pipelines, because the two spellings differ only in what sits between the
pair: the scalar/parallel lists have them adjacent, and ``build_rvv_pipeline`` has
``fuse, canonicalize, cse, generalize``. Moving the GENERALIZE (rather than swapping in place)
keeps the ``canonicalize``/``cse`` attached to the fusion they clean up after -- which is not
incidental: ``impr_features.FUSE_ELEMENTWISE_STAGE`` records that most of the temporary collapse
is the cleanup, not the fusion.

A no-op on a list that has no such pair, or where the generalization already runs first, so it is
safe to apply to any pass list. Never moves a pass across ``transform-interpreter``: a
generalization hoisted in front of the schedule would leave ``ops{["linalg.matmul"]}`` nothing to
match (silent 0-vectorization), which is the failure the current order was written to avoid.

## `parallel_transform_schedule`

Transform schedule that wraps each contraction in an `scf.forall` over ``n_harts``.

Runs BEFORE the package's own schedule (separate entry point, see above), so the inner
tiling/vectorization — and therefore the emitted `vfmacc`/`vwmacc` — is untouched.
``matmul_dim`` / ``batch_matmul_dim`` name the parallel dim to split; K is never
tileable here (it is the reduction dim and splitting it would race).

``chunks`` (from :func:`perop_blocks.distinct_parallel_arms`) selects the PER-OP form, and it is
the form that keeps the claim in the paragraph above true. The class-wide ``num_threads`` split
below leaves each hart a ``ceil(dim / n_harts)`` tile the register block then has to mask, so the
block — and with it the emitted kernel — became a function of the hart count: measured on
lstmnetvit int8 at 8 harts, 5 matmuls fell out of the block table to scalar loops and the linked
ELF issued +63.5% instructions with 63% fewer vector ops before any thread was created. With
``chunks``, merlin has already derived a tile that divides the dim EXACTLY and that the op's own
block still lowers on (``perop_blocks.parallel_chunk_table``), one arm per distinct (class, tile),
matched by the ``merlin.par_*`` tag the same prepare step applies. An op merlin could not split
that way carries no tag, matches no arm, and stays serial — with the 1-hart kernel intact.

``chunks=None`` keeps the class-wide ``num_threads`` schedule byte-identical, for the packages
that carry no per-op block table for a split to be derived against.

## `vec_pre_schedule`

The transform library that vectorizes the tagged non-contraction generics, or None.

WHY THIS EXISTS -- a COVERAGE bug, measured, not inferred. The prepare pass tags every
all-parallel ``linalg.generic`` it can vectorize with a discardable ``merlin.vec_r{rank}``
attribute, and ``vectorize_non_contraction_generics`` matches on it. But
``func.func(linalg-specialize-generic-ops)`` -- which must run before the interpreter, so the
CONTRACTION arms can match named ``linalg.matmul``/``batch_matmul`` -- rewrites most of those
tagged generics into named ops (``linalg.broadcast`` above all) and a rewrite does not carry a
DISCARDABLE attribute onto the op it produces. The tag is gone, and with it the match.

MEASURED, tags present in the module at each stage (int8 recaptures, host lowering). Every tag
that reaches an arm is consumed -- the count after the interpreter is 0 in both placements -- so
"reaches" and "vectorized" are the same number here::

    model         tagged   after canonicalize,cse   after specialize   reaching an arm
                  by prep  (where the arms run now) (where they ran)   before -> after
    deepjscc       93       85                       15                 15  ->  85  (16% -> 91%)
    small_llama   107       84                       33                 33  ->  84  (31% -> 79%)
    lstmnetvit    189      160                       36                 36  -> 160  (19% -> 85%)

So four fifths of the ops the lever was written for never reached an arm, while every tile the
remaining fifth produced paid the full loop and destination-buffer overhead -- which is the
coverage half of why the lever measured 1.28x SLOWER (the realization half, a ``memref.copy`` of
each tile onto itself, was fixed separately and is now implied by the feature).

The residual (93 -> 85, 107 -> 84, 189 -> 160) is the ``canonicalize``/``cse`` this pipeline
already ran before any of this: the generic count falls further than the tag count on all three
models, which is consistent with CSE merging duplicate ops rather than with tags being lost.

NOT A SPEED CLAIM. What is measured here is coverage and the emitted vector census (vector loads
108 -> 268 deepjscc, 134 -> 249 small_llama, 285 -> 560 lstmnetvit, with the malloc count FALLING
on all three). Whether that pays is a board measurement, and 1.28x is the number it has to beat.

THE FIX IS A PLACEMENT, NOT A PRESERVATION. Restoring the attribute across specialization would
need op identity to survive a rewrite that is not 1:1 -- specialize DECOMPOSES generics (deepjscc:
278 generics in, 199 generics + 116 broadcasts + 20 matmuls out), so there is no correspondence to
key a restore on. Running the arms BEFORE specialization needs no correspondence at all: the tags
are still on the ops that carry them. It is also the better IR to vectorize -- the pre-specialize
form is the FUSED one, so an arm tiles one elementwise chain instead of the broadcast + consumer
pair specialize would have split it into.

Realized as a SEPARATE library with its own entry point rather than by re-ordering
``__transform_main``: ``transform-preload-library`` merges several files and the interpreter runs
once per entry point, so the package's own schedule -- and every ``impr_features`` edit that
anchors on ``__transform_main`` -- composes completely unchanged, and the contraction arms still
run exactly once, after specialization, on exactly the IR they run on today.

Returns None when the lever is off (then the pipeline string is byte-identical to the baseline),
when the feature set splices no arms, or under ``MERLIN_VEC_AFTER_SPECIALIZE``.
