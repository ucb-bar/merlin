"""No capsule may declare an epilogue the target's readout does not apply at the width it commits.

MEASURED TWICE, and the second time is why this file scans the INTERFACE rather than the YAML.

First: twelve capsules declared ``relu`` while committing ``i32``. On this target the activation exists
only on the narrowing readout; the full-width readout writes the raw accumulator and applies nothing.
The grade-time check (``merlin.verify.epilogue_applicability``) refused them one at a time, hours into
an agent run, having read the very declaration the capsule writer never consulted.

Second, after that was fixed: four more slipped through, because the first audit keyed on the capsule
YAML's top-level ``operation.attributes.epilogue``. A ``resident_reuse`` capsule carries its stages
INSIDE ``attributes.matmuls``, and two hand-authored capsules had a corrected YAML beside an
interface that still said ``i32``. The compiler reads the interface, and so does the grade. So this
scans every ``epilogue = [...]`` / ``output_dtype`` pair in every emitted interface, which is the one
place the question is always askable however the capsule is shaped.

Parsed structurally with ``str.split`` -- no regex, per the repo's derive-don't-overfit rule.
"""

from __future__ import annotations

import pathlib

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen.readout_facet import epilogue_readouts
from merlin.verify.epilogue_applicability import selectors_applying

CORPUS = repo_root() / "merlin/contract/capsules"

#: Targets whose corpora live under this root but whose readouts are their own. A capsule for another
#: target must never be judged against this one's declaration.
OTHER_TARGET_DIRS = ("/radiance/", "/atlas/", "/saturn_opu/")

#: Known stale artifacts of `group_capsules.promote`, which writes outside `generate_corpus`. Neither
#: is in the graded cohort (they are model-layer inputs to the whole-model capstone), and the generator
#: now fails closed on this, so a re-promote fixes or refuses them. This list may only SHRINK.
STALE = frozenset(
    {
        "G_conv2d_c1024x14x14_k1x1s2_n2048_bias_add",
        "G_matmul_m1k2048n1000_bias_add",
    }
)


def _commit_sites(path: pathlib.Path):
    """``(stages, committed_dtype)`` for every commit in one interface that declares an epilogue."""
    for line in path.read_text(encoding="utf-8").splitlines():
        if "epilogue = [" not in line or "output_dtype" not in line:
            continue
        inner = line.split("epilogue = [", 1)[1].split("]", 1)[0]
        stages = [t.strip().strip('"') for t in inner.split(",") if t.strip()]
        if not stages:
            continue
        yield stages, line.split('output_dtype = "', 1)[1].split('"', 1)[0]


def _gemmini_interfaces():
    for f in sorted(CORPUS.rglob("capsule.interface.mlir")):
        if any(d in str(f) for d in OTHER_TARGET_DIRS):
            continue
        yield f


def test_every_declared_epilogue_is_applied_by_the_readout_it_commits_at():
    readouts = epilogue_readouts("gemmini")
    assert readouts, "gemmini declares no readouts; this check would be vacuous"

    violations, scanned = [], 0
    for f in _gemmini_interfaces():
        for stages, odt in _commit_sites(f):
            scanned += 1
            if odt not in selectors_applying(readouts, stages):
                violations.append((f.parent.name, stages, odt))

    # Commit sites carrying a NON-EMPTY epilogue. Measured at 40 for the gemmini corpus; the floor
    # guards against a parse change silently scanning nothing and reporting a clean sweep.
    assert scanned >= 30, f"only {scanned} epilogue-bearing commit sites scanned; the parse changed shape"
    unexpected = [v for v in violations if v[0] not in STALE]
    assert not unexpected, (
        "these capsules declare an epilogue no readout applies at the width they commit — the grade "
        f"will refuse them as protocol violations: {unexpected}"
    )


def test_the_stale_allowance_only_shrinks():
    """Every name in STALE must still be violating; a fixed one has to leave the list."""
    readouts = epilogue_readouts("gemmini")
    still = set()
    for f in _gemmini_interfaces():
        if f.parent.name not in STALE:
            continue
        for stages, odt in _commit_sites(f):
            if odt not in selectors_applying(readouts, stages):
                still.add(f.parent.name)
    gone = STALE - still
    assert not gone, f"these no longer violate and must be removed from STALE: {sorted(gone)}"


@pytest.mark.parametrize("stage", ["relu", "acc_scale", "bias_add", "maxpool"])
def test_the_target_applies_the_stages_its_requirement_demands(stage):
    """The requirement and the readout declaration must not drift apart again.

    The conformance spec's epilogue axis is intersected with this declaration, so a stage it requires
    must be one some readout applies. If this fails, the spec is asking for something unbuildable.
    """
    readouts = epilogue_readouts("gemmini")
    assert selectors_applying(readouts, [stage]), (
        f"the requirement demands a fused {stage!r} but no declared readout applies it"
    )


def test_no_capsule_declares_its_output_dtype_twice_in_disagreement():
    """A capsule may state its output element type twice; the two must agree.

    The interface scan above cannot see this: it reads the emitted MLIR, while `numeric_policy.dtype`
    lives only in the YAML. MEASURED — correcting two hand-authored capsules' attributes and interface
    to i8 left `numeric_policy.dtype: i32` behind, and the runner refused both as RUNNER_CRASH mid-run.
    Refusing is right (the two size the same DRAM slot, so one would silently mis-size it); being
    refused for the first time inside a graded run is not.
    """
    import yaml

    from merlin.targetgen.capsule_dram import declared_output_dtype

    unresolved, checked = [], 0
    for f in sorted(CORPUS.rglob("capsule.yaml")):
        try:
            doc = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001 -- a malformed capsule is another test's business
            continue
        outs = [o.get("name") for o in (doc.get("outputs") or []) if o.get("name")]
        out = (doc.get("operation") or {}).get("attributes", {}).get("out") or (outs[0] if outs else "Y0")
        checked += 1
        try:
            declared_output_dtype(doc, out)
        except Exception as exc:  # noqa: BLE001 -- the refusal IS the finding
            unresolved.append((f.parent.name, f"{type(exc).__name__}: {exc}"[:160]))

    assert checked > 300, f"only {checked} capsules checked; the corpus or the walk changed shape"
    assert not unresolved, f"these capsules cannot resolve their output dtype: {unresolved}"
