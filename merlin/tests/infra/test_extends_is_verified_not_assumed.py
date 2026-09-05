"""An `extends` that names an uncertified sibling is not a remedy, and the check must be able to say so.

THE REGRESSION THIS PINS. check_cert_affordability treated any non-empty `extends` as a remedy for an
over-budget capsule -- which is the failure the field exists to prevent, reached through the field
itself. Worse, the first fix was placed AFTER the filter that drops every capsule not demanding L3, and
a capsule resting on a sibling is exactly the one that does not demand L3: it was capped to the cheap
tier because it could not afford certification. The check reported zero unverified while 19 capsules on
disk declared the field.
"""
from __future__ import annotations

import importlib.util

from merlin.common.paths import repo_root


def _gate():
    path = repo_root() / "build_tools" / "scripts" / "check_cert_affordability.py"
    spec = importlib.util.spec_from_file_location("_cert_afford_probe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_gate_examines_capsules_that_do_not_demand_l3():
    """A capsule capped at the cheap tier is the ONLY kind that carries an extends, so a check placed
    behind the L3 filter can never fire."""
    mod = _gate()
    rep = mod.audit(budget_s=300.0)
    assert "n_extends_declared" in rep, "the gate must report how many capsules declare an extends"
    assert rep["n_extends_declared"] > 0, (
        "no capsule was examined for an extends; the corpus declares some, so the check is placed "
        "behind a filter that removes exactly what it is meant to inspect"
    )


def test_an_unverified_extends_is_reported_rather_than_counted_as_remedied():
    mod = _gate()
    rep = mod.audit(budget_s=300.0)
    unver = rep.get("unverified_extends")
    assert unver is not None, "the report must carry the unverified set, even when it is empty"
    for row in unver:
        assert row.get("reason"), f"{row['capsule']}: an unverified verdict must carry its reason"
        assert row.get("extends"), "an unverified row must name the sibling it rests on"


def test_the_target_label_matches_the_one_the_prices_use():
    """Two derivations of 'which target is this' drift, and a drift would verify a sibling against
    another target's certification history."""
    from pathlib import Path

    mod = _gate()
    cy = Path("/x/merlin/contract/capsules/atlas/isa/A0/capsule.yaml")
    assert mod._target_label_for(cy) == "atlas"
    root_level = Path("/x/merlin/contract/capsules/isa/A0/capsule.yaml")
    assert mod._target_label_for(root_level) == mod._default_corpus_target()
