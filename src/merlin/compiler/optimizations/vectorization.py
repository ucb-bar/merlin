"""Bounded all-parallel generic vectorization schedules and legality hygiene."""

from __future__ import annotations

from merlin.compiler.lower.native.policy_features import PROMOTE_STACK_NAME, _promote_buffers_to_stack
from merlin.compiler.lower.options.selfcopy import FEATURE as _SELF_COPY_FEATURE
from merlin.compiler.selection.features import ImprFeature

# ---- pointwise ---------------------------------------------------------------------------------

VEC_NONCONTRACTION_NAME = "vectorize_non_contraction_generics"


VEC_NONCONTRACTION_LANES = 8


def _vec_noncontraction_hygiene() -> frozenset[str]:
    """The lowering hygiene the per-rank vectorize arms cannot be measured without.

    Each arm TILES the all-parallel generic on TENSORS and then vectorizes the tile, so every tile
    yields a ``tensor.insert_slice`` back into the loop-carried destination. Bufferization realizes
    that destructive update as a subview of the destination, a ``vector.transfer_write`` into it, a
    SECOND structurally identical subview, and a ``memref.copy`` between the two -- i.e. the tile is
    copied ONTO ITSELF, once per tile, INSIDE the innermost loop::

        %sv   = memref.subview %arg99[0, %i, %j, %k] [1, 1, 1, 8] [1, 1, 1, 1]
        vector.transfer_write %v, %sv[...]        // the result is already in place here
        %sv_0 = memref.subview %arg99[0, %i, %j, %k] [1, 1, 1, 8] [1, 1, 1, 1]
        memref.copy %sv, %sv_0                    // ...and this copies it onto itself

    That is the same defect ``selfcopy`` documents for the tiled CONTRACTION epilogue, and it is why
    this lever measured 1.28x SLOWER at bit-identical output with the vector count up 4.9x: the
    per-tile copy is emitted per 8 ELEMENTS, so the vector win is spent on a memcpy of the bytes the
    vector store just wrote. The earlier attribution missed it because it counted ``@memrefCopy``
    call sites -- these copies are contiguous in the innermost dim, so they lower to ``llvm.memcpy``
    instead and the memrefCopy count does not move at all.

    MEASURED at the post-bufferization split point (int8 recaptures, impr_tuned_wholemodel_vf_int8,
    host lowering), copy CALL SITES emitted inside ``forward``:

        deepjscc      lever off  72 memcpy | lever on  87 (+15, one per vectorized op)
                      + erase    52 memcpy | + erase   52 (+0)
        small_llama   lever off  52 memcpy | lever on  85 (+33, one per vectorized op)
                      + erase    39 memcpy | + erase   39 (+0)

    Erasing a ``memref.copy %x, %x`` is unconditionally value-preserving (identical SSA operand =>
    identical base, offsets and region) and is a no-op on a lowering that has none, so implying it
    can only help or do nothing -- and enabling it is what makes ``lower_to_llvm_ir`` splice the
    post-bufferization ``canonicalize,cse`` that collapses the two subviews into the one SSA value
    the erase keys on.

    CORRECTION (supersedes the closing claim of commit ffd3c40f, "the output is bit-identical on
    every arm"). That statement was WRONG for small_llama int8 and it was wrong because of the
    sample, not the reasoning: the copy census above was taken on deepjscc and small_llama, and only
    the OUTPUT DIGEST of deepjscc was compared. On small_llama the lever's output was cos 0.968247 /
    rel 0.46352 against the baseline's 0.999966 / 0.00836 -- a MISCOMPILE, not a rounding difference,
    and one that produced two different answers (``756d00f36c43``, ``d9d8a01aa32e``) from the SAME
    shared object depending on the host process's memory layout. The cause is unrelated to the
    self-copy erase (it is the sub-byte element hazard :func:`_vec_bytewise_matchers` documents and
    now refuses); what the erase claim got wrong is only that it generalized a bit-identity measured
    on one model to "every arm". With the refusal in place the output IS bit-identical to the
    baseline on all three int8 recaptures, in both arm placements, under three initial memory
    layouts -- which is the statement this note is willing to make.

    It is an ``implies`` rather than a note in the description for the reason ``_tile_epilogue_hygiene``
    gives: a default-off lever whose payoff is cancelled by a separate default-off fix is an inert
    lever, because everyone naming it directly in ``compiler_features`` gets the cancelled version.
    """
    return frozenset({_SELF_COPY_FEATURE})


VEC_BYTEWISE_ATTR = "merlin.vec_ok"


VEC_BYTEWISE_MIN_BITS = 8


VEC_BYTEWISE_MAX_INPUTS = 6


def _vec_bytewise_matcher_prefix(text: str) -> str | None:
    """Symbol prefix for the matchers spliced into ``text``, derived from ITS OWN entry point.

    ``transform-preload-library`` merges every library it is given into one transform module, and the
    non-contraction arms are spliced into TWO of them (the pre-specialization library and the package
    schedule). A fixed symbol name would collide there. The entry-point symbol is what distinguishes
    the two libraries, so deriving the prefix from it makes the names unique without anyone choosing
    one. Returns None when the text has no named sequence to derive from (then nothing is spliced).
    """
    key = "transform.named_sequence @"
    for line in text.splitlines():
        head, sep, tail = line.partition(key)
        if not sep:
            continue
        name = tail.split("(", 1)[0].strip()
        if name:
            return f"__merlin_vec_bytewise_{name.lstrip('_')}_"
    return None


def _vec_bytewise_matchers(
    prefix: str, *, min_bits: int = VEC_BYTEWISE_MIN_BITS, max_inputs: int = VEC_BYTEWISE_MAX_INPUTS
) -> str:
    """Named matcher sequences that accept a ``linalg.generic`` iff EVERY tensor it reads or writes
    has an element at least ``min_bits`` wide.

    WHY THIS EXISTS -- a MISCOMPILE, measured, not inferred. The arms vectorize an op by tiling it and
    writing each tile with a ``vector.transfer_write``. For an element type narrower than a byte the
    two sides of that write disagree about the layout of the SAME buffer: LLVM stores a `vector<8xi1>`
    PACKED (one byte for eight lanes) while the memref the tile belongs to is addressed one byte per
    element, and every scalar consumer reads it that way. MEASURED on small_llama int8 (host lowering,
    old arm placement), the causal mask is a `tensor<8x8xi1>`::

        %3147 = call ptr @malloc(i64 128)          ; 64 elements, ONE BYTE EACH
        ...
        %3163 = getelementptr i1, ptr %3152, i64 %3157   ; %3157 = row * 8
        store <8 x i1> %3162, ptr %3164, align 1   ; ...but this writes ONE byte
        ...
        %3204 = getelementptr inbounds nuw i1, ptr %3152, i64 %3203
        %3205 = load i1, ptr %3204, align 1        ; scalar consumer, one byte per element
        %3208 = select i1 %3205, float 0xFFF0000000000000, float %3207

    8 of the 64 bytes are written and 56 are read back UNINITIALISED, straight into the attention
    mask's `select`. The result is wrong (cos 0.9682 / rel 0.4635 against a baseline 0.99997 / 0.0084)
    and depends on what happened to be in that allocation, so the same binary answers differently
    under a different initial stack/heap layout. Both directions are unsound: a vectorized WRITE of a
    sub-byte destination, and a vectorized READ of a sub-byte input some scalar loop wrote -- which is
    why the inputs are checked as well as the destination.

    Realized as matchers rather than as a tagging-time predicate because the tag is applied by the
    prepare pass and the arms are the last thing that decides; a schedule that carries the arms
    carries the refusal with them, and cannot be armed without it.

    One sequence per ``ins`` arity: ``match.structured.elemental_bitwidth`` takes a SINGLE value, and
    ``%s[all]`` is rejected by its single-value trait, so each input is checked by index and the arity
    is pinned with ``num_inputs``. An op outside the enumerated arities matches nothing and is
    refused. ``num_inits`` is pinned to 1 for the same reason -- a second destination would go
    unchecked otherwise.

    WHAT THIS IS NOT, checked and ruled out. The obvious suspect was PARTIAL COVERAGE -- the extra
    destination buffer the arms allocate per vectorized op, written only where the tile lands and
    read where it does not. It is not that. The tagging predicate admits an op only when its
    innermost extent is a whole multiple of the lane count and the arms tile every other dim by 1, so
    the tiles cover the destination exactly: MEASURED on small_llama int8 at the post-interpreter
    module, every one of the 33 (old placement) / 84 (new placement) tile writes carries
    ``in_bounds = [true...]``, there is not one ``affine.min`` trip-count clamp and not one
    ``vector.mask`` in the whole module. The uninitialised bytes come from the ELEMENT WIDTH, not
    from the tiling.
    """
    out: list[str] = []
    for n in range(max_inputs + 1):
        lines = [
            f"  transform.named_sequence @{prefix}{n}(%op: !transform.any_op "
            "{transform.readonly}) -> !transform.any_op {",
            '    transform.match.operation_name %op ["linalg.generic"] : !transform.any_op',
            "    transform.match.structured %op : !transform.any_op {",
            "    ^bb0(%s: !transform.any_op):",
            f"      %bits = transform.param.constant {min_bits} : i64 -> !transform.param<i64>",
            "      %one = transform.param.constant 1 : i64 -> !transform.param<i64>",
            f"      %arity = transform.param.constant {n} : i64 -> !transform.param<i64>",
            "      %ni = transform.match.structured.num_inputs %s : (!transform.any_op) -> !transform.param<i64>",
            "      transform.match.param.cmpi eq %ni, %arity : !transform.param<i64>",
            "      %no = transform.match.structured.num_inits %s : (!transform.any_op) -> !transform.param<i64>",
            "      transform.match.param.cmpi eq %no, %one : !transform.param<i64>",
            "      %init = transform.match.structured.init %s[0] : (!transform.any_op) -> !transform.any_value",
            "      %bwinit = transform.match.structured.elemental_bitwidth %init : "
            "(!transform.any_value) -> !transform.param<i64>",
            "      transform.match.param.cmpi ge %bwinit, %bits : !transform.param<i64>",
        ]
        for k in range(n):
            lines += [
                f"      %in{k} = transform.match.structured.input %s[{k}] : (!transform.any_op) -> "
                "!transform.any_value",
                f"      %bw{k} = transform.match.structured.elemental_bitwidth %in{k} : "
                "(!transform.any_value) -> !transform.param<i64>",
                f"      transform.match.param.cmpi ge %bw{k}, %bits : !transform.param<i64>",
            ]
        lines += ["    }", "    transform.yield %op : !transform.any_op", "  }"]
        out.append("\n".join(lines))
    return "\n".join(out) + "\n"


def _vec_bytewise_annotate(prefix: str, *, max_inputs: int = VEC_BYTEWISE_MAX_INPUTS) -> str:
    """The arm-body lines that run those matchers and mark what they accept."""
    return "".join(
        f"    %ok{n} = transform.collect_matching @{prefix}{n} in %arg0 : "
        f"(!transform.any_op) -> !transform.any_op\n"
        f'    transform.annotate %ok{n} "{VEC_BYTEWISE_ATTR}" : !transform.any_op\n'
        for n in range(max_inputs + 1)
    )


VEC_NONCONTRACTION_MIN_RANK = 2


VEC_NONCONTRACTION_MAX_RANK = 4


def _vec_rank_arms(lanes: int, prefix: str | None = None, max_rank: int = VEC_NONCONTRACTION_MAX_RANK) -> str:
    """The per-rank tile+vectorize arms at ``lanes`` innermost lanes, for ranks 2..``max_rank``.

    ``prefix`` names the byte-addressable-element matchers (see :func:`_vec_bytewise_matchers`); the
    arms then match ``merlin.vec_r{rank}`` AND the attribute those matchers annotate, so a sub-byte
    op is refused rather than mis-vectorized. Default None keeps the arms self-contained for the
    tests that read them without a module to splice matchers into.

    One arm per rank, GENERATED rather than written out, because the tile-size list and the loop
    arity are both functions of the rank: writing them by hand is what fixed the coverage at rank 4.
    At the default ``max_rank`` the emitted text is byte-identical to the hand-written arms it
    replaces (pinned by ``test_impr_features``).
    """
    gate = f", {VEC_BYTEWISE_ATTR}" if prefix else ""
    mark = _vec_bytewise_annotate(prefix) if prefix else ""
    arms = []
    for rank in range(VEC_NONCONTRACTION_MIN_RANK, max_rank + 1):
        sizes = ", ".join(["1"] * (rank - 1) + [str(lanes)])
        # `tile_using_for` yields the tiled op plus one loop handle per tiled dim.
        loops = ", ".join(["!transform.any_op"] * (rank + 1))
        arms.append(
            f"    %g{rank} = transform.structured.match attributes{{merlin.vec_r{rank}{gate}}} in "
            f"%arg0 : (!transform.any_op) -> !transform.any_op\n"
            f"    %gt{rank}, %gl{rank}:{rank} = transform.structured.tile_using_for %g{rank} "
            f"tile_sizes [{sizes}] : (!transform.any_op) -> ({loops})\n"
            f"    transform.structured.vectorize %gt{rank} vector_sizes [{sizes}] : "
            f"!transform.any_op\n"
        )
    return (
        mark
        + "".join(arms)
        + """\
    %vecf = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %vecf {
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
      transform.apply_patterns.vector.drop_unit_dims_with_shape_cast
      transform.apply_patterns.vector.lower_shape_cast
    } : !transform.any_op
"""
    )


def _splice_vec_rank_arms(
    text: str, lanes: int = VEC_NONCONTRACTION_LANES, max_rank: int = VEC_NONCONTRACTION_MAX_RANK
) -> str:
    """Insert the per-rank vectorize arms just before the schedule's func-level pattern block.

    ADDITIVE (not schedule_replace), so it layers on whatever micro-kernel recipe is in play: the
    contraction arms above it are untouched and only the previously-unmatched generics gain arms.
    A schedule that already carries them, or has no func anchor, is returned unchanged.
    """
    anchor = '    %f = transform.structured.match ops{["func.func"]}'
    if "merlin.vec_r" in text or anchor not in text:
        return text
    prefix = _vec_bytewise_matcher_prefix(text)
    head = _vec_module_header(text)
    if prefix is None or head is None:
        # Nowhere to put the refusal matchers => splice NO arms. Arming without them is the
        # miscompile `_vec_bytewise_matchers` documents, and a lever that is off is recoverable
        # where a lever that is silently wrong is not.
        return text
    text = text.replace(head, head + _vec_bytewise_matchers(prefix), 1)
    return text.replace(anchor, _vec_rank_arms(lanes, prefix, max_rank) + anchor, 1)


def _vec_module_header(text: str) -> str | None:
    """The ``module ... {`` line the matcher sequences are inserted after, or None.

    Structural, not positional: the first line that OPENS a module. A schedule whose module this
    cannot find gets no arms at all rather than arms without their refusal matchers.
    """
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("module") and stripped.endswith("{"):
            return line
    return None


def _vec_noncontraction_point(name: str) -> "tuple[int, int] | None":
    """``(lanes, max_rank)`` a non-contraction-vectorize feature NAME asks for, or None.

    The name is the single source of truth for both numbers, so the TAGGING predicate and the
    SCHEDULE arms cannot drift apart -- and so a point resolves identically in the lowering
    SUBPROCESS, which re-imports this module and never sees a registration the parent made at run
    time. Grammar: the base name, then any of ``_l<lanes>`` and ``_r<max_rank>`` in any order. An
    unparsable tail yields None (the feature is then simply not this family) rather than a default,
    because silently reading a mistyped point as the default is how an experiment measures the
    baseline and reports it as the lever.
    """
    if not name.startswith(VEC_NONCONTRACTION_NAME):
        return None
    tail = name[len(VEC_NONCONTRACTION_NAME) :]
    lanes, rank = VEC_NONCONTRACTION_LANES, VEC_NONCONTRACTION_MAX_RANK
    while tail:
        if not tail.startswith("_") or len(tail) < 3:
            return None
        kind, tail = tail[1], tail[2:]
        digits = ""
        while tail and tail[0].isdigit():
            digits, tail = digits + tail[0], tail[1:]
        if not digits:
            return None
        if kind == "l":
            lanes = int(digits)
        elif kind == "r":
            rank = int(digits)
        else:
            return None
    if lanes <= 0 or rank < VEC_NONCONTRACTION_MIN_RANK:
        return None
    return lanes, rank


def vec_noncontraction_lanes(features) -> int | None:
    """Innermost lane count the enabled non-contraction-vectorize feature asks for, else None.

    The TAGGING predicate (which admits an op only when its innermost extent is a whole multiple of the
    lane count, so nothing is masked) and the SCHEDULE arms must agree on this number; deriving it from
    the feature name is what keeps them from drifting apart.
    """
    for f in features or ():
        point = _vec_noncontraction_point(f)
        if point is not None:
            return point[0]
    return None


def vec_noncontraction_max_rank(features) -> int | None:
    """Highest loop rank the enabled non-contraction-vectorize feature asks for, else None.

    Same contract as :func:`vec_noncontraction_lanes` and for the same reason: the tagger admits an
    op only up to this rank and the schedule only carries arms up to it, so one number has to drive
    both. None means no such feature is enabled, and the tagger then runs not at all.
    """
    for f in features or ():
        point = _vec_noncontraction_point(f)
        if point is not None:
            return point[1]
    return None


def ensure_vec_noncontraction(lanes: int, max_rank: int = VEC_NONCONTRACTION_MAX_RANK) -> str:
    """Register (on demand) the non-contraction vectorize point at ``lanes`` innermost lanes.

    Exposing the width as a registered point per value is what lets a search find one that pays
    instead of a human guessing. It is NOT the axis the first measurement moved: cycles were flat at
    0.78x across 8/16/32, because every width paid the same per-tile self-copy (see
    ``_vec_noncontraction_hygiene``). Each point implies the same hygiene as the bare name, so a
    width the search picks is measured in the same realization as the default one.
    """
    from merlin.compiler.optimizations.registry import known, register

    if lanes == VEC_NONCONTRACTION_LANES and max_rank == VEC_NONCONTRACTION_MAX_RANK:
        return VEC_NONCONTRACTION_NAME
    name = VEC_NONCONTRACTION_NAME
    if lanes != VEC_NONCONTRACTION_LANES:
        name += f"_l{lanes}"
    if max_rank != VEC_NONCONTRACTION_MAX_RANK:
        name += f"_r{max_rank}"
    if name in known():
        return name
    register(
        ImprFeature(
            name=name,
            action_class="PASS",
            description=(
                f"Bounded per-rank vectorize of the non-contraction all-parallel generics at "
                f"{lanes} innermost lanes, loop ranks "
                f"{VEC_NONCONTRACTION_MIN_RANK}..{max_rank} (see {VEC_NONCONTRACTION_NAME}). "
                f"Default-off; both numbers must be chosen by measurement, not assumed."
            ),
            edit_schedule=lambda t, _l=lanes, _r=max_rank: _splice_vec_rank_arms(t, _l, _r),
            implies=_vec_noncontraction_hygiene(),
        )
    )
    return name


# ---- passes ------------------------------------------------------------------------------------
# Existing stack-promotion, activation and reduction strategy construction.


def ensure_promote_stack(nbytes: int) -> str:
    """Register (on demand) a stack-promotion variant whose per-buffer cap is ``nbytes``.

    The cap is a LEVER, not just an overflow guard, and the difference is measured. On small_llama
    int8 (K1, sustained, cos identical on every arm) the SAME feature pays back very differently
    depending only on this number:

        <=  4 KB  4,904,957 ns  1.00x      <= 256 KB  3,649,518 ns  1.34x
        <= 16 KB  4,795,823 ns  1.03x      <=   1 MB  3,579,566 ns  saturated
        <= 64 KB  4,678,494 ns  1.05x

    Monotonic in the cap and flat past 256 KB, which locates the largest intermediate worth promoting
    between 64 KB and 256 KB -- a per-MODEL fact about intermediate sizes, not a constant.

    Why this function exists: the cap was reachable ONLY through ``MERLIN_PROMOTE_STACK_BYTES``, and
    no fork can vary an environment variable. So the beam always built at the 16 KB default, measured
    1.03x, and ranked the lever below levers it should beat -- in the run that prompted this, the
    stack-promotion fork came out SLOWER than its own parent and the search concluded the lever was
    not worth its width. The lever was never weak; the search could not reach the part of it that
    works. Naming the cap in the FEATURE puts it on the channel feature names already travel
    (proposer -> knobs.yaml ``compiler_features`` -> ``normalize`` -> build), so it needs no new
    plumbing and cannot silently disagree with the build the way an ambient env var can.
    """
    from merlin.compiler.optimizations.registry import known, register

    n = int(nbytes)
    if n <= 0:
        raise ValueError(f"{PROMOTE_STACK_NAME}: per-buffer cap must be positive, got {nbytes!r}")
    name = f"{PROMOTE_STACK_NAME}_{n}"
    if name in known():
        return name
    register(
        ImprFeature(
            name=name,
            action_class="PASS",
            description=(
                f"promote small bufferization allocs to the stack, per-buffer cap {n} bytes. "
                f"Same pass as {PROMOTE_STACK_NAME}, with the cap NAMED so the search can vary "
                f"it: the cap is model-dependent (measured 1.03x at 16 KB vs 1.34x at 256 KB on "
                f"the same model and the same feature), and as an env var it was unreachable by "
                f"any fork. Default-off; baseline byte-identical."
            ),
            edit_pipeline=lambda passes, _n=n: _promote_buffers_to_stack(passes, cap=_n),
        )
    )
    return name


_ACT_POLY_SCHEDULE = """\
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t, %l:3 = transform.structured.tile_using_for %mm tile_sizes [4, 8, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.structured.vectorize %t vector_sizes [4, 8, 1] : !transform.any_op
    %bm = transform.structured.match ops{["linalg.batch_matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %bt, %bl:4 = transform.structured.tile_using_for %bm tile_sizes [1, 4, 8, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.structured.vectorize %bt vector_sizes [1, 4, 8, 1] : !transform.any_op
    // Vectorize PRECISELY the activation generics — and ONLY those — the poly rewriter targeted.
    // The rewriter (act_poly.apply_activation_polynomial) replaces math.erf/exp/tanh with an arith
    // mul/add/fma chain ONLY inside a generic the PROVENANCE marks as an elementwise activation
    // (gelu/silu/sigmoid/tanh), and tags each such generic with a `merlin.act_vectorize` unit attr.
    // We match exactly those tagged generics and vectorize them (bare vectorize = rank-agnostic,
    // sizes inferred from the static iteration space) so the poly chain becomes a vector.fma/vfmacc
    // chain. A softmax-exp / normalization / any other generic is NOT tagged (its exp stayed on
    // libm), so it is NOT matched here and lowers exactly as in the baseline.
    //
    // This drops the previous BLANKET design (foreach over EVERY linalg.generic + failures(suppress)
    // + a hard-coded rank-1 tile). That design (a) rewrote+vectorized the softmax exp too, which
    // amplified the minimax error through the row-sum normalization -> openvla whole-model cos 0.541;
    // (b) used failures(suppress), which HID any vectorize miscompile; and (c) vectorized every
    // generic in the model, blowing clang -O2 compile time to 6+ min/config. Matching only the tagged
    // activation generics fixes all three: correct (softmax stays libm), no suppression (a genuine
    // vectorize failure surfaces), and bounded compile time (only the few activation generics
    // vectorize). No `failures(suppress)` — these tagged generics are static-shape elementwise and
    // always statically vectorize; if one ever didn't, we WANT the error, not a silent scalar fallback.
    %eg = transform.structured.match ops{["linalg.generic"]} attributes{"merlin.act_vectorize"} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.foreach %eg : !transform.any_op {
    ^bb_eg(%one_eg: !transform.any_op):
      transform.structured.vectorize %one_eg : !transform.any_op
    }
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {
      transform.apply_patterns.vector.lower_contraction
      transform.apply_patterns.vector.lower_masked_transfers
      transform.apply_patterns.vector.lower_transpose
      transform.apply_patterns.vector.lower_shape_cast
    } : !transform.any_op
    transform.yield
  }
}
"""


_VECTORIZE_REDUCTION_BLOCK = """\
    %redr1 = transform.structured.match ops{["linalg.generic"]} attributes{iterator_types = [#linalg.iterator_type<reduction>]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %redr1 : !transform.any_op
    %redr2 = transform.structured.match ops{["linalg.generic"]} attributes{iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %redr2 : !transform.any_op
    %redr3 = transform.structured.match ops{["linalg.generic"]} attributes{iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %redr3 : !transform.any_op
"""


_REDUCTION_ANCHOR = '    %f = transform.structured.match ops{["func.func"]}'


_VECTORIZE_REDUCTION_STANDALONE = f"""\
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{
{_VECTORIZE_REDUCTION_BLOCK}\
    %f = transform.structured.match ops{{["func.func"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {{
      transform.apply_patterns.vector.lower_masked_transfers
      transform.apply_patterns.vector.lower_shape_cast
    }} : !transform.any_op
    transform.yield
  }}
}}
"""


def vectorize_reduction_schedule(text: str) -> str:
    """Insert the reduction match+vectorize block into the baseline schedule (before its func.func
    lowering-pattern block), keeping the contraction handling intact. Falls back to a reduction-only
    schedule if the anchor is absent (never a silent no-op)."""
    if _REDUCTION_ANCHOR in text:
        return text.replace(_REDUCTION_ANCHOR, _VECTORIZE_REDUCTION_BLOCK + _REDUCTION_ANCHOR, 1)
    return _VECTORIZE_REDUCTION_STANDALONE


def vectorize_reduction_pipeline(passes: list[str]) -> list[str]:
    """Make the emitted reduction a single-instruction horizontal vector reduce (`vfredusum`/`vredsum`):
    switch `lower-vector-multi-reduction` to the inner-reduction strategy (-> `vector.reduction`) and
    enable `reassociate-fp-reductions` on `convert-vector-to-llvm` (-> unordered `vfredusum.vs`). Only
    runs when this feature is enabled; the baseline pass list is untouched otherwise."""
    out: list[str] = []
    for p in passes:
        if p == "func.func(lower-vector-multi-reduction)":
            out.append("func.func(lower-vector-multi-reduction{lowering-strategy=inner-reduction})")
        elif p == "convert-vector-to-llvm":
            out.append("convert-vector-to-llvm{reassociate-fp-reductions}")
        else:
            out.append(p)
    return out
