"""`fuse_elementwise_post_contraction`: the fusion stage the tuning loop could not name.

The stage itself is not new — `pipeline.build_rvv_pipeline` has carried it, with its own paragraph of
measurement, sitting directly above `linalg-generalize-named-ops`. What was missing is a way to ASK
for it: it was gated on the `MERLIN_FUSE_POST` environment variable, and no fork of the beam can vary
an environment variable, so a lever the pipeline fully supported was one the search could never
select. These tests pin the three properties that make it selectable and safe:

* the feature's pass list is IDENTICAL to the one the environment variable produces, so every
  measurement taken through the env var describes the feature (measured: the K1 cross-compile is
  byte-identical either way — model.o sha 13f0506a15343a18, model.ll sha d5868600a852e88f);
* naming the feature while the env var is also set does NOT fuse twice — a second canonicalize/cse
  pair changes the emitted code, so a doubled stage would mean a manual A/B and a feature-driven one
  were not comparing the same build;
* an empty feature set leaves the pipeline byte-identical, so the frozen baseline is untouched.

And the fail-closed one: with no anchor pass to insert against there is no position provably after
the transform interpreter and before bufferization, so the splice RAISES rather than inserting at a
guessed index and reporting the feature as applied.
"""
from __future__ import annotations

import pytest

from merlin.llvmlower import pipeline as P
from merlin.llvmlower.impr_features import (
    FUSE_ELEMENTWISE_ANCHOR,
    FUSE_ELEMENTWISE_NAME,
    FUSE_ELEMENTWISE_STAGE,
    apply_pipeline,
    get,
    known,
    normalize,
)

#: The config the whole-model search currently calls best on `small_llama_int8_consistent`. The
#: feature has to compose with THIS, not with an empty set: a lever that cannot be stacked on the
#: winner is a lever the beam can never propose.
BEST = [
    "prepack_weight_layout",
    "perop_register_block",
    "promote_buffers_to_stack",
    "expand_memref_copy",
    "cse_through_provenance",
]

_SCHED = "/nonexistent/schedule.mlir"      # never read: build_rvv_pipeline only interpolates the path


def test_registered_eagerly() -> None:
    """Importing the registry is enough — no satellite module has to be imported first.

    `wholemodel_proposer._composes` resolves every lever name through `impr_features` and SWALLOWS a
    KeyError as "does not compose", so a lazily-registered name is silently never proposed rather
    than rejected. This feature's hook lives in `impr_features` itself, so there is no import order
    that can leave it unregistered.
    """
    assert FUSE_ELEMENTWISE_NAME in known()
    assert get(FUSE_ELEMENTWISE_NAME).edit_pipeline is not None
    # Additive, not a schedule replacement: it must layer onto whichever micro-kernel recipe is in
    # play rather than clobbering it (`apply_schedule` refuses two replacements).
    assert get(FUSE_ELEMENTWISE_NAME).schedule_replace is False
    assert get(FUSE_ELEMENTWISE_NAME).edit_schedule is None


def test_empty_features_leave_the_pipeline_byte_identical() -> None:
    """The frozen baseline invariant every default-off feature in this registry carries."""
    passes = ["canonicalize", "cse", FUSE_ELEMENTWISE_ANCHOR, "one-shot-bufferize"]
    assert apply_pipeline(passes, frozenset()) == passes
    base = P.build_rvv_pipeline(_SCHED, features=frozenset())
    assert P.build_rvv_pipeline(_SCHED, features=frozenset()) == base


def test_stage_is_spliced_immediately_before_the_anchor() -> None:
    """Position is the correctness property, not the presence of the pass.

    The fusion must run AFTER `transform-interpreter` (fusing earlier folds matmuls into generics, and
    the schedule's `ops{["linalg.matmul"]}` then matches nothing — a silent 0-vectorization) and
    BEFORE bufferization (afterwards there are no producer/consumer tensors left to fuse).
    """
    # deliberately NO bare "canonicalize"/"cse" here: the stage carries its own, and a survival
    # check written over names the stage also contains would pass on a list it never inspected.
    passes = ["transform-interpreter{entry-point=__transform_main}",
              FUSE_ELEMENTWISE_ANCHOR, "one-shot-bufferize"]
    out = apply_pipeline(passes, frozenset({FUSE_ELEMENTWISE_NAME}))
    at = out.index(FUSE_ELEMENTWISE_ANCHOR)
    assert tuple(out[at - len(FUSE_ELEMENTWISE_STAGE):at]) == FUSE_ELEMENTWISE_STAGE
    assert out.index("transform-interpreter{entry-point=__transform_main}") < at
    assert out.index("one-shot-bufferize") > at
    # every original pass survives, in order — the edit inserts, it never reorders or drops
    assert [p for p in out if p in passes] == passes


def test_feature_pipeline_equals_the_env_var_pipeline(monkeypatch) -> None:
    """The env var stays as the manual-A/B escape hatch, and the two must not diverge.

    Every number recorded for this feature was measured through `MERLIN_FUSE_POST`; if the two
    spliced different pass lists, those measurements would describe a build the feature never emits.
    """
    baseline = P.build_rvv_pipeline(_SCHED, features=frozenset())
    by_feature = P.build_rvv_pipeline(_SCHED, features=frozenset({FUSE_ELEMENTWISE_NAME}))
    monkeypatch.setenv("MERLIN_FUSE_POST", "1")
    by_env = P.build_rvv_pipeline(_SCHED, features=frozenset())
    assert by_feature == by_env
    assert by_feature != baseline


def test_feature_and_env_var_together_do_not_fuse_twice(monkeypatch) -> None:
    """A doubled stage is not merely wasteful — the extra canonicalize/cse changes emitted code."""
    monkeypatch.setenv("MERLIN_FUSE_POST", "1")
    by_env = P.build_rvv_pipeline(_SCHED, features=frozenset())
    both = P.build_rvv_pipeline(_SCHED, features=frozenset({FUSE_ELEMENTWISE_NAME}))
    assert both == by_env
    assert both.count(FUSE_ELEMENTWISE_STAGE[0]) == 1


def test_missing_anchor_raises_instead_of_silently_doing_nothing() -> None:
    """Fail closed: "enabled and changed nothing" is the failure mode this repo keeps re-learning."""
    with pytest.raises(ValueError) as exc:
        apply_pipeline(["canonicalize", "cse"], frozenset({FUSE_ELEMENTWISE_NAME}))
    assert FUSE_ELEMENTWISE_ANCHOR in str(exc.value)


def test_composes_with_the_current_best_config() -> None:
    """Including through the proposer's own gate, which is where an unregistered name disappears."""
    from merlin.mining import wholemodel_proposer as W

    assert W._composes([*BEST, FUSE_ELEMENTWISE_NAME])
    assert FUSE_ELEMENTWISE_NAME in [n for n, _ in W.RANKED_LEVERS]
    # `normalize` is what a build calls; it must resolve every name in the stack.
    assert FUSE_ELEMENTWISE_NAME in normalize([*BEST, FUSE_ELEMENTWISE_NAME])


def test_proposer_registers_the_best_configs_own_features() -> None:
    """The winner must be a proposable PARENT, not just a nameable config.

    `_composes` returns False on an unregistered name, so while `cse_through_provenance` — which the
    current best config carries — was registered only by `llvmlower.lower` (a module the proposer
    never imports), every stack built on the winner failed the gate and the beam had nothing to build
    on. Pins the registration rather than the symptom.
    """
    from merlin.mining import wholemodel_proposer as W

    assert W._composes(BEST), "the current best config must itself compose"
    for name in BEST:
        assert name in known(), f"{name} unregistered when only the proposer is imported"


def test_distinct_from_the_refuted_after_generalize_lever() -> None:
    """The two fusion levers sit on OPPOSITE sides of the anchor, and that is the whole difference.

    `fuse_elementwise_after_generalize` inserts a bare fusion pass AFTER
    `linalg-generalize-named-ops` and MEASURED 1.22x slower on this model: past that pass every named
    op is generic, the dequant chain fuses into the producers, and the shape the transform schedule
    had already vectorized the contraction into is perturbed (its scalar bucket fell 3.1 -> 2.8 ms
    while the contraction rose 1.8 -> 2.8 ms). This feature runs BEFORE the anchor and brings the
    canonicalize/cse cleanup the other omits — decoded on the emitted object, vwmacc stays 152 and
    vredmax 104, so the contraction is provably not the thing being disturbed.

    Pinned as a test because the names are one word apart and a future reader who conflates them
    would either re-attempt a refuted lever or discard a live one.
    """
    from merlin.llvmlower.impr_features import FUSE_AFTER_GENERALIZE_NAME

    assert FUSE_AFTER_GENERALIZE_NAME != FUSE_ELEMENTWISE_NAME
    passes = ["transform-interpreter{entry-point=__transform_main}",
              FUSE_ELEMENTWISE_ANCHOR, "one-shot-bufferize"]
    mine = apply_pipeline(passes, frozenset({FUSE_ELEMENTWISE_NAME}))
    theirs = apply_pipeline(passes, frozenset({FUSE_AFTER_GENERALIZE_NAME}))
    fuse = FUSE_ELEMENTWISE_STAGE[0]
    assert mine.index(fuse) < mine.index(FUSE_ELEMENTWISE_ANCHOR), "this one fuses BEFORE the anchor"
    assert theirs.index(fuse) > theirs.index(FUSE_ELEMENTWISE_ANCHOR), "that one fuses AFTER it"
    # and only this one carries the cleanup the pipeline's own comment calls load-bearing
    assert "canonicalize" in mine and "canonicalize" not in theirs
