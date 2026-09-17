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
    for root in SC.out_roots():
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
    _unit(rooted / "runs" / "a-target" / "capsule-bench" / "20260915T000000Z_run", ("log", 1000))
    _unit(rooted / "artifacts" / "perf-bench" / "a-target" / "campaign_a", ("results.json", 4000))

    groups = SC.experiment_costs()
    assert set(groups) == {_g("runs", "a-target", "capsule-bench"), _g("artifacts", "perf-bench", "a-target")}
    assert groups[_g("runs", "a-target", "capsule-bench")]["units"] == 1
    assert groups[_g("artifacts", "perf-bench", "a-target")]["bytes"] == 4000


def test_a_version_level_is_not_mistaken_for_an_experiment(rooted):
    """``v1`` sits BETWEEN the axis and the unit. Pricing it as one experiment reports a whole
    version series as a single run -- perf-bench's `v1` holds 34 GB, which is the one number
    guaranteed to mislead."""
    base = rooted / "artifacts" / "perf-bench" / "a-target" / "v1"
    _unit(base / "run_one", ("a", 1000))
    _unit(base / "run_two", ("a", 3000))

    groups = SC.experiment_costs()
    series = _g("artifacts", "perf-bench", "a-target", "v1")
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
    _unit(rooted / "artifacts" / "perf-bench" / "a-target" / "global_phase2_v1", ("a", 5000))
    _unit(rooted / "artifacts" / "compare" / "a-target" / "phase2_followup", ("a", 3000))
    _unit(rooted / "artifacts" / "compare" / "a-target" / "phase1_baseline", ("a", 900))

    matched = SC.experiment_costs("phase2")
    assert sum(r["units"] for r in matched.values()) == 2
    assert sum(r["bytes"] for r in matched.values()) == 8000
    assert all("phase1" not in str(r["largest"][0]) for r in matched.values())


def test_the_cache_is_not_counted_as_experiments(rooted):
    """It is regenerable by the convention that created it and is priced as a cache instead;
    counting it would inflate every per-experiment mean with bytes no run owns."""
    _unit(rooted / "artifacts" / "cache" / "rtl_introspect" / "facts", ("a", 100_000))
    _unit(rooted / "artifacts" / "compare" / "a-target" / "real_run", ("a", 100))

    groups = SC.experiment_costs()
    assert list(groups) == [_g("artifacts", "compare", "a-target")]


def test_an_undeclared_concern_is_named_and_priced(rooted):
    """52 of these existed against 16 declared ones; the report has to say which is which."""
    undeclared = "a-name-the-contract-does-not-declare"
    assert undeclared not in SC.declared_concerns()
    _unit(rooted / "artifacts" / "perf-bench" / "a-target" / "run", ("a", 100))
    _unit(rooted / "artifacts" / undeclared / "whatever" / "run", ("a", 7000))

    report = SC.layout_drift()
    assert [r["name"] for r in report["declared_concerns"]] == ["perf-bench"]
    assert [r["name"] for r in report["undeclared_concerns"]] == [undeclared]
    assert report["undeclared_concerns"][0]["bytes"] == 7000
    assert report["undeclared_concerns"][0]["units"] == 1


def test_a_root_the_convention_retired_is_reported(rooted):
    """The write-guard hook blocks paths it knows; a new top-level root under out/ is exactly the
    case it does not, so nothing noticed four of them appearing."""
    stray = "a-root-the-contract-does-not-declare"
    assert stray not in SC.out_roots()
    _unit(rooted / stray / "some_audit", ("finding.md", 500))

    report = SC.layout_drift()
    assert [r["name"] for r in report["stray_roots"]] == [stray]
    assert report["stray_roots"][0]["bytes"] == 500
    assert all(r["name"] not in SC.out_roots() for r in report["stray_roots"])


def test_every_declared_concern_carries_a_purpose():
    """A roster of bare names is how the last one drifted: nobody could tell a standing concern from
    an afternoon's detour, so nobody pruned either. Each entry has to say what it holds."""
    concerns = SC.contract().get("concerns") or {}
    assert concerns, "the storage contract declares no concerns"
    for name, purpose in concerns.items():
        assert isinstance(purpose, str) and len(purpose.split()) >= 3, (
            f"{name!r} is declared without saying what it holds"
        )


def test_the_convention_document_and_the_contract_agree():
    """CLAUDE.md prose and the contract are two statements of one roster, so they can disagree.

    The contract is the authority -- it is what the tool and the gate read -- but a document that
    contradicts it sends a reader to the wrong place, which is how a convention stops being obeyed.
    """
    from merlin.common.paths import repo_root

    text = (repo_root() / "CLAUDE.md").read_text(encoding="utf-8")
    missing = [name for name in SC.declared_concerns() if f"{name}/" not in text]
    assert not missing, f"declared but absent from CLAUDE.md's convention: {missing}"


def test_a_fold_target_is_always_a_declared_concern():
    """A fold that moved a directory somewhere undeclared would launder drift into the roster
    instead of resolving it, and the next audit would report the destination."""
    declared = set(SC.declared_concerns())
    roots = set(SC.out_roots())
    for entry in SC.contract().get("folds") or []:
        parts = str(entry["into"]).split("/")
        assert parts[0] in roots, f"fold destination {entry['into']!r} is not under a declared root"
        assert parts[1] in declared, f"fold destination {entry['into']!r} is not a declared concern"


def test_every_fold_source_is_undeclared():
    """The inverse: folding a DECLARED concern away would contradict the roster that declares it."""
    declared = set(SC.declared_concerns())
    for entry in SC.contract().get("folds") or []:
        parts = str(entry["from"]).split("/")
        if len(parts) > 1 and parts[0] == "artifacts":
            assert parts[1] not in declared, f"{entry['from']!r} is folded away and declared at the same time"
        else:
            assert parts[0] not in SC.out_roots(), f"{entry['from']!r} folds away one of the three declared roots"


def test_only_a_group_written_both_ways_is_reported_as_drift(rooted):
    """A group where NOTHING is timestamped follows a convention of its own -- a codegen package is
    named by package id, and the layout convention says so. Reporting those would bury the real
    signal: a tree written through new_product() AND by something that just made a directory."""
    own = rooted / "artifacts" / "targets" / "rvv"
    _unit(own / "hand_v0", ("schedule.json", 10))
    _unit(own / "impr_tuned_v3", ("schedule.json", 10))
    mixed = rooted / "artifacts" / "perf-bench" / "a-target"
    _unit(mixed / "perf-bench_gemmini_v1_20260910T020821Z_abfe2c7", ("a.json", 10))
    _unit(mixed / "full_graph_attempt1", ("a.json", 10))

    drift = {row["group"]: row for row in SC.layout_drift()["unconventional_units"]}
    assert set(drift) == {_g("artifacts", "perf-bench", "a-target")}
    assert drift[_g("artifacts", "perf-bench", "a-target")]["undated"] == 1


def test_a_product_is_priced_against_its_own_manifest(rooted):
    """The one drift number that needs no threshold: a manifest IS the statement of what the product
    contains, so bulk the build left beside it is undeclared by the producer's own account. Measured
    2026-09-16: one unit held 35.06 GiB of ELFs and weight blobs against 43 declared receipts."""
    unit = rooted / "artifacts" / "perf-bench" / "a-target" / "v1" / "p_a-target_v1_20260910T0208Z_a"
    _unit(unit, ("receipt.json", 10))
    (unit / "harness.elf").write_bytes(b"e" * 5000)
    (unit / "manifest.yaml").write_text("artifacts:\n- receipt.json\n")

    rows = SC.layout_drift()["undeclared_payload"]
    assert [r["unit"] for r in rows] == [unit.name]
    assert rows[0]["bytes"] == 5000
    assert rows[0]["declared"] == 1


def test_a_product_that_declares_its_contents_does_not_appear(rooted):
    """What makes the number trustworthy: the 9.53 GiB delivery bundle reports 0% undeclared."""
    unit = rooted / "artifacts" / "delivery" / "a-board" / "v1" / "d_a-board_v1_20260818T1735Z_a"
    _unit(unit, ("bundle.zip", 9000))
    (unit / "manifest.yaml").write_text("artifacts:\n- bundle.zip\n")

    assert SC.layout_drift()["undeclared_payload"] == []
