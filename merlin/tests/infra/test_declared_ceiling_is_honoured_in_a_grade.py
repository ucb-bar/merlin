"""A capsule's declared oracle-tier ceiling must be honoured DURING a grade, not only in promotion.

MEASURED, and the wiring gate could not see it. ``tier_policy.oracle_ceiling`` had a production caller
-- ``tier_promote``, which enqueues cert jobs BETWEEN grades -- so it counted as wired. Nothing checked
that the grade's own tier ladder consulted it, and it did not, even though ``TierResult`` already
carries an ``oracle_ceiling`` field to record the answer.

The cost: six capsules each declaring ``max_oracle_tier: L2`` and each resting on a sibling that had
genuinely earned L3 ran the cert tier anyway, every grade, to the 1200s per-capsule timeout and failed
there. 7230s of adapter wall -- about half the grade -- spent re-certifying claims the capsules' own
declarations said they were not making, and reported as six failures.

Two things are pinned here: the policy answer for the corpus as it stands, and the call site whose
absence no gate could detect.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

from merlin.common.paths import merlin_dir, repo_root
from merlin.targetgen import tier_policy as TP

CORPUS = repo_root() / "merlin/contract/capsules"
RUNNER = merlin_dir() / "python/merlin/targetgen/capsule_runner.py"


def _capped_capsules():
    """Every tracked capsule declaring a correctness ceiling, with its parsed document."""
    out = []
    for f in sorted(CORPUS.rglob("capsule.yaml")):
        try:
            doc = yaml.safe_load(f.read_text()) or {}
        except Exception:  # noqa: BLE001 -- a malformed capsule is another test's business
            continue
        if doc.get(TP.CEILING_FIELD):
            out.append((f.parent.name, doc))
    return out


def test_the_grade_ladder_consults_the_ceiling():
    """The call site itself. Its absence is what cost the wall clock, and no gate could see it.

    Checked structurally against the runner source because the alternative -- driving a full grade --
    needs an oracle, which is exactly what this skip exists to avoid spending.
    """
    src = RUNNER.read_text()
    assert "oracle_ceiling(" in src, (
        "capsule_runner does not call tier_policy.oracle_ceiling. The declared ceiling is then honoured "
        "only in promotion, and the grade re-buys the cert tier for capsules that declared they are not "
        "asking for it"
    )
    # ... and in the ladder, not merely imported or recorded.
    head, _, tail = src.partition("for tier in _tier_seq:")
    assert tail, "capsule_runner no longer has the tier ladder this test pins"
    assert "oracle_ceiling(" in tail, (
        "tier_policy.oracle_ceiling is referenced outside the tier ladder only; the ladder is where a "
        "tier is actually bought, so that is where the ceiling has to be asked"
    )


def test_a_declared_ceiling_declines_a_deeper_tier():
    """Every capped capsule in the corpus must decline the tier beyond its cap."""
    capped = _capped_capsules()
    if not capped:
        pytest.skip("no capsule declares a correctness ceiling")
    ladder = ["L0", "L1", "L2", "L3"]
    checked = 0
    for name, doc in capped:
        cap = str(doc.get(TP.CEILING_FIELD))
        deeper = [t for t in ladder if TP._rank(t, ladder) > TP._rank(cap, ladder)]
        if not deeper:
            continue
        for tier in deeper:
            c = TP.oracle_ceiling("gemmini", doc, tier, declared_tiers=ladder)
            assert not c.allowed, (
                f"{name} declares {TP.CEILING_FIELD}: {cap} but {tier} was allowed — the grade would buy "
                f"a tier the capsule says it is not asking for"
            )
            assert c.reason, f"{name}: a declined tier must carry its reason; a silent skip is not evidence"
            checked += 1
    assert checked, "no capped capsule produced a deeper tier to check"


def test_the_cap_never_suppresses_the_screen_tier():
    """'Screened at the cheap tier' is the whole claim a capped capsule makes."""
    capped = _capped_capsules()
    if not capped:
        pytest.skip("no capsule declares a correctness ceiling")
    ladder = ["L0", "L1", "L2", "L3"]
    for name, doc in capped[:25]:
        c = TP.oracle_ceiling("gemmini", doc, "L2", declared_tiers=ladder)
        assert c.allowed, f"{name}: the ceiling declined L2, which is the tier the capped capsule rests on"
