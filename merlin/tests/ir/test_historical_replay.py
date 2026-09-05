"""The replay's honesty properties, which are the only reason its number means anything.

A detection rate is trivially manufacturable: pick the commits, pick the denominator, rerun until the
number is good. Each test here pins one of the moves that would do that.
"""
from __future__ import annotations

import pytest

from merlin.common.paths import repo_root


def test_the_sample_is_reproducible_from_the_seed():
    """Two draws with one seed are the same draw. Without this the record is not checkable."""
    from merlin.verify.replay import draw

    pool = [f"c{i}" for i in range(200)]
    assert draw(pool, 20, 7) == draw(pool, 20, 7)
    assert draw(pool, 20, 7) != draw(pool, 20, 8), "different seeds must give different samples"


def test_a_larger_sample_extends_the_smaller_one():
    """Shuffle-then-take, not `random.sample`.

    This is what stops a bad result being quietly rerolled: with the same seed, n=40 is a SUPERSET of
    n=20, so a later, larger run cannot silently replace the earlier commits with friendlier ones. A
    reader can check the property directly from the two records.
    """
    from merlin.verify.replay import draw

    pool = [f"c{i}" for i in range(200)]
    assert draw(pool, 40, 11)[:20] == draw(pool, 20, 11)


def test_the_population_only_holds_commits_that_touched_an_observed_path():
    """The denominator's definition. A commit the layers cannot see is not a miss -- it is out of scope.

    Reported with the record rather than left implicit, because "detected 3 of 25" means nothing
    without knowing which 25.
    """
    from merlin.verify.replay import OBSERVED_ROOTS, population

    pool = population(repo_root())
    if not pool:
        pytest.skip("no history in this checkout")
    for sha, _subject, files in pool:
        assert files, f"{sha[:8]} is in the population with no observed file"
        for f in files:
            assert any(f.startswith(r) for r in OBSERVED_ROOTS), (
                f"{sha[:8]} contributes {f}, which no layer can observe")


def test_pytest_exit_codes_separate_a_rejection_from_a_run_that_did_not_happen():
    """The soundness hole that would flatter the rate most.

    A package shadowed at an old parent can fail to IMPORT, because a sibling module has moved since.
    Scoring that as a rejection credits the layer with catching a defect it never looked at. pytest
    exits 1 for a genuine test failure and >= 2 when the run itself did not happen (2 interrupted,
    which is what a collection error reports; 3 internal; 4 usage; 5 nothing collected).
    """
    import inspect

    from merlin.verify import replay

    src = inspect.getsource(replay._run_layers)
    assert '{0: "green", 1: "red"}' in src, (
        "the exit-code mapping changed; a non-1 exit code must not be scored as a rejection")
    assert '"error"' in src, "a run that did not happen must have its own verdict, not 'red'"


def test_the_shadow_replaces_code_but_never_the_data_paths():
    """The flaw that invalidated the first run, and the reason a negative control is not optional.

    `repo_root()` resolves from the package's own file location, so inside a shadow it points at the
    temp directory — where there is no capsule corpus, no lit suite and no llvm-build. Every layer then
    finds nothing to check and exits clean, and the replay records a MISS for a defect no layer ever
    looked at. The first 25-commit run reported 0 detections that way.

    Caught by running the shadow at the parent of the COMMIT readout fix (a defect whose effect is
    known and large) and finding every layer green. With MERLIN_REPO_ROOT pinned to the real checkout,
    the engines layer goes red, which is the answer the instrument is supposed to give.
    """
    import inspect

    from merlin.verify import replay

    src = inspect.getsource(replay._run_layers)
    assert "MERLIN_REPO_ROOT=str(repo)" in src, (
        "the shadow must pin the data root to the real checkout; without it every layer reports a "
        "clean pass because it cannot find its inputs")


def test_the_instrument_contains_the_layers_that_exist():
    """A rate measured with a convenient subset of the layers understates the work it is describing.

    The first run wired three pytest files and left out the static layer (lit/FileCheck over the
    passes) and the numeric oracle over the real corpus — the two checks most likely to see a lowering
    defect. Both are in `LAYERS` now.
    """
    from merlin.verify.replay import LAYERS

    assert "lit-pass-tests" in LAYERS, "the static layer is missing from the instrument"
    assert "numeric-golden" in LAYERS, (
        "the numeric oracle is missing; without it a detection cannot be attributed to the new layer "
        "rather than to the dynamic check that already existed")


def test_an_unreplayable_commit_is_reported_and_never_counted_as_a_miss():
    """The denominator again, from the other side.

    Two things make a commit unreplayable: its parent files are gone (deleted or renamed), or the
    shadowed package does not run. Both are reported. Folding either into `missed` would understate
    detection; dropping either from the record would overstate it by shrinking the denominator.
    """
    import inspect

    from merlin.verify import replay

    src = inspect.getsource(replay.replay)
    assert src.count('"unreplayable"') >= 2, (
        "both unreplayable paths (missing parent files, shadow that will not run) must be recorded")
    rendered = replay.render({
        "population_size": 101, "sample_size": 2, "seed": 1,
        "population_definition": {"ref": "0" * 40},
        "baseline": {"a-layer": "green"}, "detected_of_replayable": "0/1",
        "detected_of_replayable_historical": "0/1", "layers_landed": "abc12345",
        "counts": {"missed": 1, "unreplayable": 1},
        "results": [
            {"sha": "aaaaaaaa", "subject": "fix(x): a", "layers_red": [], "outcome": "missed",
             "predates_layers": True},
            {"sha": "bbbbbbbb", "subject": "fix(y): b", "layers_red": [], "outcome": "unreplayable",
             "predates_layers": True},
        ],
    })
    assert "1 unreplayable" in rendered, "the report must state the unreplayable count"
    assert "never folded into 'missed'" in rendered


def test_a_fix_that_postdates_the_layers_is_flagged_and_excluded_from_the_citable_rate():
    """A fix that shipped WITH its own regression test would be caught by that test, not by the layer.

    Such commits stay in the sample -- removing them after seeing the outcome is the exact move this
    module exists to prevent -- but the rate worth citing is computed without them.
    """
    from merlin.verify.replay import LAYERS_LANDED, _ancestors_of_layers

    hist = _ancestors_of_layers(repo_root())
    if not hist:
        pytest.skip("the commit that introduced the layers is not in this checkout")
    assert LAYERS_LANDED in hist, "a commit is its own ancestor; the boundary is off by one"
    import subprocess
    head = subprocess.run(("git", "rev-parse", "HEAD"), cwd=repo_root(),
                          capture_output=True, text=True, check=True).stdout.strip()
    assert head not in hist, "HEAD postdates the layers; it must not count as historical"


def test_every_declared_layer_can_actually_run():
    """A layer wired to a name that does not exist reports `red` or `error` and is then disqualified.

    That happened twice: `merlin.verify.replay_lit` and `merlin/tests/ir/test_golden_engines_agree.py`
    were both invented in the LAYERS table and never created, and both runs quietly measured three
    layers while the surrounding text described five. This resolves each entry to a real file or
    importable module without running it, so the mistake fails here in milliseconds instead of an hour
    into a sweep.
    """
    import importlib.util

    from merlin.verify.replay import LAYERS

    for name, argv in LAYERS.items():
        argv = list(argv)
        if argv[:2] == ["-m", "pytest"]:
            target = repo_root() / argv[-1]
            assert target.is_file(), f"layer {name!r} runs {argv[-1]}, which does not exist"
        else:
            assert argv[0] == "-m", f"layer {name!r} has an unrecognised invocation: {argv}"
            assert importlib.util.find_spec(argv[1]), (
                f"layer {name!r} runs module {argv[1]!r}, which cannot be imported")
