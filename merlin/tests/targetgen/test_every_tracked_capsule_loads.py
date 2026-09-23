"""Every capsule in the tracked corpus must LOAD — schema included — or the corpus is unusable.

WHY THIS EXISTS, measured 2026-09-22. `corpus_synth` began emitting `SY_carried_relu_i8` once the
carried-state axis reached the coverage gate's verdict, and the capsule was generated, graded locally,
committed, and shipped. Its `semantic.generalization_axis` is `carried_state`, which
`merlin/contract/schemas/capsule.schema.json` did not list. `capsule_common.load_capsule` validates
against that schema and raises BEFORE any label filter, so one row of 122 killed every discovery path
at once: the launcher's task scope, the readiness check, and the grader. A phase-1 launch could not
reach its first round.

WHAT LET IT SHIP. Every gate that reads the corpus reads it through a *derived* view -- the conformance
gate measures cover cells, `check_phase_split` reads tiers, `check_defect_reach` globs -- and none of
them loads a capsule through the validator. The corpus was checked for what it COVERS and never for
whether it can be READ. So the schema and the synthesizer drifted apart with nothing in between.

This is deliberately the cheapest possible check: no target resolution, no descriptor, no oracle. It
walks the tracked corpus and calls the same loader the harness calls. A new axis, a new field, a
renamed enum member -- anything that makes a capsule unreadable -- fails here in under a second rather
than at launch after a 13.7 GB bundle snapshot.
"""

from __future__ import annotations

import pytest

from merlin.common.paths import merlin_dir
from merlin.targetgen import capsule_common as CC

_CORPUS = merlin_dir() / "contract" / "capsules"


def _capsule_dirs() -> list:
    """Every directory holding a `capsule.yaml`, tracked or generated, in a stable order."""
    return sorted(p.parent for p in _CORPUS.rglob("capsule.yaml"))


def test_the_corpus_is_not_empty():
    """A zero-capsule sweep would make every assertion below vacuous -- the shape of check this repo
    keeps finding, so it is ruled out first."""
    dirs = _capsule_dirs()
    assert len(dirs) > 100, f"only {len(dirs)} capsule(s) found under {_CORPUS}; the walk is wrong"


def test_every_capsule_loads_and_schema_validates():
    """THE REGRESSION. One unreadable capsule makes the whole corpus unusable, so this reports ALL of
    them rather than stopping at the first -- a list of three is a different morning from a list of one.
    """
    failures = []
    for d in _capsule_dirs():
        try:
            CC.load_capsule(str(d))
        except Exception as exc:  # noqa: BLE001 -- any failure to read is the failure under test
            failures.append(f"{d.name}: {type(exc).__name__}: {str(exc)[:200]}")
    assert not failures, (
        f"{len(failures)} capsule(s) in the tracked corpus cannot be loaded. Every discovery path -- "
        f"the launcher's task scope, the readiness check and the grader -- raises on the first of "
        f"these, so the corpus is unusable until they are fixed:\n  " + "\n  ".join(failures[:10])
    )


@pytest.mark.parametrize("axis", ["carried_state"])
def test_an_axis_the_synthesizer_emits_is_in_the_schema(axis: str):
    """The synthesizer and the schema must agree about the generalization axes that exist.

    Pinned by name for the axis that broke: `corpus_synth` writes
    `generalization: {generalization_axis: carried_state}` for the carried-state member, and the schema
    enum has to contain it or that member cannot be read. Parametrised so the next axis added to the
    requirement is one line here rather than a launch-time discovery.
    """
    import json

    schema = json.loads((_CORPUS.parent / "schemas" / "capsule.schema.json").read_text(encoding="utf-8"))

    def _find(node):
        if isinstance(node, dict):
            if "generalization_axis" in node and isinstance(node["generalization_axis"], dict):
                return node["generalization_axis"].get("enum")
            for value in node.values():
                found = _find(value)
                if found:
                    return found
        elif isinstance(node, list):
            for value in node:
                found = _find(value)
                if found:
                    return found
        return None

    enum = _find(schema)
    assert enum, "no generalization_axis enum found in capsule.schema.json"
    assert axis in enum, f"the synthesizer emits generalization_axis {axis!r}, which the schema forbids: {enum}"
