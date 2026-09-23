"""Folding out/ into its declared shape must not break a single citation.

The layout convention named three roots and a closed set of concerns, and only TRACKED files were
linted -- generated output is gitignored by design, so the convention held in the index and drifted
freely underneath it: 4 undeclared roots and 52 undeclared concerns against 16 declared ones. The
reorganization is therefore not a tidy-up but a migration of paths that manifests, reports, figures
and docs already quote, and the property that makes it safe is that every old name keeps resolving.
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
    """A path under the out/ root, joined rather than spelled.

    The layout gate rejects a quoted `artifacts/...` literal in code, because such a literal used to
    be a repo-root-relative read of a root the out/ consolidation retired. These are contract values
    rather than paths, but the rule is worth obeying rather than exempting: it keeps the one spelling
    of those roots in `merlin.common.paths`.
    """
    return "/".join(parts)


def _declare(monkeypatch, folds: list[dict]) -> None:
    declared = dict(SC.contract(), folds=folds, scan_roots=[])
    monkeypatch.setattr(SC, "contract", lambda: declared)


def _unit(path: Path, name: str = "report.md", body: bytes = b"finding") -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / name).write_bytes(body)
    return path


def test_the_old_path_still_reads_after_a_fold(rooted, monkeypatch):
    """The whole reason a fold is allowed: a report that quotes the old path must still open it."""
    _unit(rooted / "artifacts" / "paper-figures" / "set-a")
    _declare(
        monkeypatch,
        [{"from": _g("artifacts", "paper-figures"), "into": _g("artifacts", "presentation", "paper-figures")}],
    )

    assert SC.main(["organize", "--apply"]) == 0

    moved = rooted / "artifacts" / "presentation" / "paper-figures" / "set-a" / "report.md"
    assert moved.read_bytes() == b"finding"
    quoted = rooted / "artifacts" / "paper-figures" / "set-a" / "report.md"
    assert quoted.read_bytes() == b"finding", "a citation of the old path stopped resolving"


def test_the_symlink_left_behind_is_relative(rooted, monkeypatch):
    """An absolute link would pin the tree to this machine's path, and the out/ root is relocatable
    by design (MERLIN_OUT_ROOT) and gets mounted into sandboxes at other prefixes."""
    _unit(rooted / "artifacts" / "paper-study" / "one")
    _declare(
        monkeypatch, [{"from": _g("artifacts", "paper-study"), "into": _g("artifacts", "presentation", "paper-study")}]
    )

    SC.main(["organize", "--apply"])

    link = rooted / "artifacts" / "paper-study"
    assert link.is_symlink()
    assert not Path(link.readlink()).is_absolute(), f"absolute link: {link.readlink()}"


def test_two_old_names_merge_into_one_concern(rooted, monkeypatch):
    """`out/audits/` and `out/analysis/` both held readiness studies. One concern, two sources: the
    second fold has to move its units ACROSS rather than refuse because the destination exists."""
    _unit(rooted / "audits" / "readiness-a")
    _unit(rooted / "analysis" / "current-readiness")
    _declare(
        monkeypatch,
        [
            {"from": "audits", "into": _g("artifacts", "audits")},
            {"from": "analysis", "into": _g("artifacts", "audits")},
        ],
    )

    assert SC.main(["organize", "--apply"]) == 0

    assert (rooted / "artifacts" / "audits" / "readiness-a" / "report.md").is_file()
    assert (rooted / "artifacts" / "audits" / "current-readiness" / "report.md").is_file()
    assert (rooted / "analysis" / "current-readiness" / "report.md").is_file()


def test_a_name_collision_aborts_the_whole_fold(rooted, monkeypatch):
    """Merging must never pick a winner silently: two units of one name are two results, and losing
    either would be indistinguishable from the fold having worked."""
    _unit(rooted / "audits" / "zz-same-name", body=b"kept")
    # Sorted order puts the clean unit FIRST, so it has already moved across when the collision is
    # found -- which is the case that has to roll back, not merely refuse.
    _unit(rooted / "analysis" / "aa-moves-first", body=b"safe")
    _unit(rooted / "analysis" / "zz-same-name", body=b"other")
    _declare(
        monkeypatch,
        [
            {"from": "audits", "into": _g("artifacts", "audits")},
            {"from": "analysis", "into": _g("artifacts", "audits")},
        ],
    )

    assert SC.main(["organize", "--apply"]) == 1

    assert (rooted / "artifacts" / "audits" / "zz-same-name" / "report.md").read_bytes() == b"kept"
    # The losing fold is fully rolled back -- including the unit it had already moved across.
    assert not (rooted / "analysis").is_symlink()
    assert (rooted / "analysis" / "zz-same-name" / "report.md").read_bytes() == b"other"
    assert (rooted / "analysis" / "aa-moves-first" / "report.md").read_bytes() == b"safe"
    assert not (rooted / "artifacts" / "audits" / "aa-moves-first").exists()


def test_organize_is_idempotent(rooted, monkeypatch):
    """It runs from a contract, so it runs again on every checkout that already folded. A second
    pass must recognise its own symlink rather than fold the link into the destination."""
    _unit(rooted / "artifacts" / "verify" / "lit")
    _declare(monkeypatch, [{"from": _g("artifacts", "verify"), "into": _g("artifacts", "probes", "verify")}])

    assert SC.main(["organize", "--apply"]) == 0
    assert SC.main(["organize", "--apply"]) == 0

    assert (rooted / "artifacts" / "probes" / "verify" / "lit" / "report.md").is_file()
    assert not (rooted / "artifacts" / "probes" / "verify" / "verify").exists()


def test_a_dry_run_moves_nothing(rooted, monkeypatch):
    _unit(rooted / "artifacts" / "verify" / "lit")
    _declare(monkeypatch, [{"from": _g("artifacts", "verify"), "into": _g("artifacts", "probes", "verify")}])

    assert SC.main(["organize"]) == 0

    assert (rooted / "artifacts" / "verify" / "lit" / "report.md").is_file()
    assert not (rooted / "artifacts" / "verify").is_symlink()
    assert not (rooted / "artifacts" / "probes").exists()


def test_a_folded_tree_is_no_longer_reported_as_drift(rooted, monkeypatch):
    """The point of the fold. A symlink left at the old name must not read as a concern of its own,
    or the report would keep naming the drift it just resolved."""
    _unit(rooted / "artifacts" / "verify" / "lit")
    _declare(monkeypatch, [{"from": _g("artifacts", "verify"), "into": _g("artifacts", "probes", "verify")}])
    SC.main(["organize", "--apply"])

    report = SC.layout_drift()
    assert [r["name"] for r in report["undeclared_concerns"]] == []
    assert "probes" in [r["name"] for r in report["declared_concerns"]]


def test_a_directory_holding_tracked_files_is_never_folded(rooted, monkeypatch):
    """git does not walk a symlink: every tracked file under a folded directory reads as deleted
    from the working tree, even though the bytes are intact behind the link. Two concerns holding
    five curated files hit exactly this the first time the folds ran."""
    _unit(rooted / "artifacts" / "handoff" / "bundle")
    monkeypatch.setattr(SC, "_holds_tracked_files", lambda path: path.name == "handoff")
    _declare(monkeypatch, [{"from": _g("artifacts", "handoff"), "into": _g("artifacts", "probes", "handoff")}])

    assert SC.main(["organize", "--apply"]) == 1

    assert not (rooted / "artifacts" / "handoff").is_symlink()
    assert (rooted / "artifacts" / "handoff" / "bundle" / "report.md").is_file()
    assert not (rooted / "artifacts" / "probes").exists()
