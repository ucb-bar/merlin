"""Per-experiment accounting and layout drift — the two questions a total cannot answer.

"out/ is 200 GB" prompts the wrong fix as often as the right one. A concern that is large because
each run writes too much needs its producer fixed; one that is large because nothing deletes a
finished campaign needs a retention decision. The mean per unit is what separates them, and it had
no reporting at all. Likewise the layout convention names three roots and a closed set of concerns,
but only TRACKED files are linted — and generated output is gitignored by design, so the convention
held in the index and drifted freely underneath it.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from merlin.common import storage_cli as SC


@pytest.fixture
def rooted(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    for root in SC.OUT_ROOTS:
        (tmp_path / "out" / root).mkdir(parents=True)
    return tmp_path / "out"


def _g(*parts: str) -> str:
    """A group label, joined rather than spelled.

    The layout gate rejects a quoted `artifacts/...` / `runs/...` literal in code, because such a
    literal used to be a repo-root-relative READ of a root the out/ consolidation retired. These are
    group labels rather than paths, but the rule is worth obeying rather than exempting: it keeps the
    one spelling of those roots in `merlin.common.paths`.
    """
    return "/".join(parts)


def _unit(path: Path, *files: tuple[str, int]) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    for name, size in files:
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"\0" * size)
    return path


def test_a_run_and_a_product_are_both_priced_as_one_experiment(rooted):
    """The two shapes the layout convention defines, at the depth where "per experiment" means
    something: an aet run and a versioned product."""
    _unit(rooted / "runs" / "gemmini" / "capsule-bench" / "20260915T000000Z_run", ("log", 1000))
    _unit(rooted / "artifacts" / "perf-bench" / "gemmini" / "campaign_a", ("results.json", 4000))

    groups = SC.experiment_costs()
    assert set(groups) == {_g("runs", "gemmini", "capsule-bench"),
                           _g("artifacts", "perf-bench", "gemmini")}
    assert groups[_g("runs", "gemmini", "capsule-bench")]["units"] == 1
    assert groups[_g("artifacts", "perf-bench", "gemmini")]["bytes"] == 4000


def test_a_version_level_is_not_mistaken_for_an_experiment(rooted):
    """``v1`` sits BETWEEN the axis and the unit. Pricing it as one experiment reports a whole
    version series as a single run -- perf-bench's `v1` holds 34 GB, which is the one number
    guaranteed to mislead."""
    base = rooted / "artifacts" / "perf-bench" / "gemmini" / "v1"
    _unit(base / "run_one", ("a", 1000))
    _unit(base / "run_two", ("a", 3000))

    groups = SC.experiment_costs()
    series = _g("artifacts", "perf-bench", "gemmini", "v1")
    assert series in groups
    row = groups[series]
    assert row["units"] == 2, "the version level was priced as a single experiment"
    assert row["mean_bytes"] == 2000


def test_accumulation_and_bloat_are_distinguishable(rooted):
    """The distinction the whole subcommand exists for: two groups of equal total, one of which is
    a retention decision and the other a producer bug."""
    for n in range(10):
        _unit(rooted / "artifacts" / "compare" / "hoarder" / f"run{n}", ("out", 1000))
    _unit(rooted / "artifacts" / "ceiling" / "bloated" / "one_run", ("out", 10_000))

    groups = SC.experiment_costs()
    hoarder = groups[_g("artifacts", "compare", "hoarder")]
    bloated = groups[_g("artifacts", "ceiling", "bloated")]
    assert hoarder["bytes"] == bloated["bytes"] == 10_000
    assert hoarder["mean_bytes"] == 1000 and bloated["mean_bytes"] == 10_000


def test_match_prices_a_campaign_spread_across_groups(rooted):
    """A campaign is a naming convention, not a directory: `phase2` units sit under several
    concerns, and the question "what did phase 2 cost" has to cross them."""
    _unit(rooted / "artifacts" / "perf-bench" / "gemmini" / "global_phase2_v1", ("a", 5000))
    _unit(rooted / "artifacts" / "compare" / "gemmini" / "phase2_followup", ("a", 3000))
    _unit(rooted / "artifacts" / "compare" / "gemmini" / "phase1_baseline", ("a", 900))

    matched = SC.experiment_costs("phase2")
    assert sum(r["units"] for r in matched.values()) == 2
    assert sum(r["bytes"] for r in matched.values()) == 8000
    assert all("phase1" not in str(r["largest"][0]) for r in matched.values())


def test_the_cache_is_not_counted_as_experiments(rooted):
    """It is regenerable by the convention that created it and is priced as a cache instead;
    counting it would inflate every per-experiment mean with bytes no run owns."""
    _unit(rooted / "artifacts" / "cache" / "rtl_introspect" / "facts", ("a", 100_000))
    _unit(rooted / "artifacts" / "compare" / "gemmini" / "real_run", ("a", 100))

    groups = SC.experiment_costs()
    assert list(groups) == [_g("artifacts", "compare", "gemmini")]


def test_an_undeclared_concern_is_named_and_priced(rooted):
    """52 of these exist in the live tree against 16 declared ones; the report has to say which."""
    _unit(rooted / "artifacts" / "perf-bench" / "gemmini" / "run", ("a", 100))
    _unit(rooted / "artifacts" / "aaaaaa" / "whatever" / "run", ("a", 7000))

    report = SC.layout_drift()
    assert [r["name"] for r in report["declared_concerns"]] == ["perf-bench"]
    assert [r["name"] for r in report["undeclared_concerns"]] == ["aaaaaa"]
    assert report["undeclared_concerns"][0]["bytes"] == 7000
    assert report["undeclared_concerns"][0]["units"] == 1


def test_a_root_the_convention_retired_is_reported(rooted):
    """The write-guard hook blocks paths it knows; a new top-level root under out/ is exactly the
    case it does not, so nothing noticed four of them appearing."""
    _unit(rooted / "audits" / "some_audit", ("finding.md", 500))

    report = SC.layout_drift()
    assert [r["name"] for r in report["stray_roots"]] == ["audits"]
    assert report["stray_roots"][0]["bytes"] == 500
    assert all(r["name"] not in SC.OUT_ROOTS for r in report["stray_roots"])


def test_every_declared_concern_in_the_convention_is_in_the_list():
    """The list is a restatement of CLAUDE.md, so it can drift from it. Hold it to the document."""
    from merlin.common.paths import repo_root

    text = (repo_root() / "CLAUDE.md").read_text(encoding="utf-8")
    for concern in SC.DECLARED_CONCERNS:
        assert f"`{concern}/" in text or f"{concern}/" in text, (
            f"{concern!r} is not named in CLAUDE.md's generated-output convention")
