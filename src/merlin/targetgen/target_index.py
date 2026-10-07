"""The generated per-target index: ``out/artifacts/targets/<target>/INDEX.yaml``.

One file answers "what has this target produced, and where did each piece come from": every sealed
(or awaiting-review) phase-0 corpus release, every phase-1 frozen compiler, every phase-2 run's
``best`` and every exported champion with its numbers and lineage. It is a projection of records
that already exist -- release ``private/*.json`` receipts, the harness-owned ``oot/`` histories, the
champions' ``.merlin/*.json`` -- so it is regenerated, never edited, and carries no timestamp of its
own: regenerating an unchanged tree gives identical bytes, which is what ``--check`` relies on.

Only aggregate identities are projected. A release's review note, reviewer, member names and hidden
inventory stay in its owner-only private records.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..common import oot_repo, paths
from ..common.yaml import dump_yaml
from . import package_records

SCHEMA = "merlin_target_index_v1"
INDEX_NAME = "INDEX.yaml"
#: The storage contract's ``product_roots`` entry for sealed phase-0 releases.
RELEASES_HOME = "phase0-releases"
RELEASE_PREFIX = "phase0-"


def index_path(target: str, *, artifacts_root: str | Path | None = None) -> Path:
    from ..common.artifacts import declared_home
    from .champions import CHAMPIONS_HOME

    return declared_home(CHAMPIONS_HOME, artifacts_root=artifacts_root) / package_records.component(target) / INDEX_NAME


def _rel(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(paths.out_dir().resolve()).as_posix()
    except ValueError:
        return str(path)


def _dirs(parent: Path, prefix: str = "") -> list[Path]:
    if not parent.is_dir():
        return []
    return sorted(p for p in parent.iterdir() if p.is_dir() and not p.is_symlink() and p.name.startswith(prefix))


def _json(path: Path) -> dict | None:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) else None


def _releases(target: str, artifacts: Path, problems: list) -> list[dict]:
    rows = []
    from ..common.artifacts import declared_home

    for release in _dirs(declared_home(RELEASES_HOME, artifacts_root=artifacts) / target, RELEASE_PREFIX):
        prepared = _json(release / "private" / "preparation.json")
        if prepared is None:
            problems.append({"path": _rel(release), "problem": "no readable preparation record"})
            continue
        if prepared.get("target") != target:
            problems.append({"path": _rel(release), "problem": f"release target is {prepared.get('target')!r}"})
            continue
        sealed = _json(release / "private" / "seal.json")
        rows.append(
            {
                "release": _rel(release),
                "state": "sealed" if sealed else "awaiting_operator_review",
                "review_digest": (sealed or {}).get("review_digest"),
                "payload_sha256": prepared.get("payload_sha256"),
                "source_run": prepared.get("source_run"),
                "prepared_at": prepared.get("prepared_at"),
                "sealed_at": ((sealed or {}).get("review") or {}).get("at"),
                "retention_pin": bool((sealed or {}).get("retention_pin")),
            }
        )
    return rows


def _repo_rows(phase_root: Path, tag_name: str, problems: list) -> list[tuple[Path, Path, dict, list]]:
    """``(run, repo, tags, history-to-tag)`` for every run whose ``oot/`` carries ``tag_name``."""
    rows = []
    for run in _dirs(phase_root):
        repo = run / "oot"
        if not (repo / ".git").is_dir():
            continue
        try:
            tags = oot_repo.tags(repo)
            if tag_name not in tags:
                continue
            rows.append((run, repo, tags, oot_repo.history(repo, tag_name)))
        except oot_repo.OotRepoError as exc:
            problems.append({"path": _rel(repo), "problem": str(exc)})
    return rows


def _phase1(target: str, problems: list) -> list[dict]:
    out = []
    for run, repo, tags, history in _repo_rows(paths.phase_runs_root(target, 1), oot_repo.FROZEN_TAG, problems):
        frozen = history[-1]
        out.append(
            {
                "run": _rel(run),
                "oot": _rel(repo),
                "frozen_commit": tags[oot_repo.FROZEN_TAG],
                "package_digest": frozen.package_digest,
                "rounds": len(history),
                "frozen_at": frozen.committed_at,
            }
        )
    return out


def _phase2(target: str, problems: list) -> list[dict]:
    out = []
    for run, repo, tags, history in _repo_rows(paths.phase_runs_root(target, 2), oot_repo.BEST_TAG, problems):
        best = history[-1]
        out.append(
            {
                "run": _rel(run),
                "oot": _rel(repo),
                "origin": oot_repo.origin(repo),
                "best_commit": tags[oot_repo.BEST_TAG],
                "package_digest": best.package_digest,
                "measured": sorted(name for name in tags if name.startswith(oot_repo.MEASURED_PREFIX)),
            }
        )
    return out


def _champions(target: str, artifacts_root: str | Path | None, problems: list) -> list[dict]:
    from .champions import champions_root, layout_problems, read_champion

    out = []
    for root in _dirs(champions_root(target, artifacts_root=artifacts_root)):
        if root.name.startswith("."):
            continue  # an interrupted export's staging tree is not a champion
        gaps = layout_problems(root)
        if gaps:
            problems.append({"path": _rel(root), "problem": f"not a standalone champion: {gaps}"})
            continue
        records = read_champion(root)
        provenance, measured = records["provenance"], records["measurements"]
        legacy = (provenance.get("lineage") or {}).get("legacy")
        firesim = measured.get("firesim") or {}
        out.append(
            {
                "package_id": root.name,
                "path": _rel(root),
                "package_digest": provenance.get("package_digest"),
                "firesim": {
                    "cycles": firesim.get("cycles"),
                    "machine": firesim.get("machine"),
                    "header": firesim.get("header"),
                    "control_in_batch": (firesim.get("control") or {}).get("in_batch"),
                },
                "gsim_verdict": (records["certification"].get("gsim") or {}).get("verdict"),
                "isa_prohibition": {
                    "scope": records["isa_prohibition"].get("scope"),
                    "verdict": records["isa_prohibition"].get("verdict"),
                },
                "lineage": {
                    "phase1_run": (provenance.get("phase1") or {}).get("run"),
                    "frozen_commit": (provenance.get("phase1") or {}).get("frozen_commit"),
                    "phase2_run": (provenance.get("phase2") or {}).get("run"),
                    "best_commit": (provenance.get("phase2") or {}).get("best_commit"),
                    "corpus_seal_digest": provenance.get("corpus_seal_digest"),
                    "phase0_evidence_digest": provenance.get("phase0_evidence_digest"),
                    "unsealed_legacy": legacy is not None,
                    "legacy_run_dirs": list((legacy or {}).get("run_dirs") or ()),
                    "reconstructed": bool((provenance.get("phase2") or {}).get("reconstructed")),
                    "composed": provenance.get("composition") is not None,
                },
            }
        )
    return out


def build_index(target: str, *, artifacts_root: str | Path | None = None) -> dict[str, Any]:
    """The index document for ``target``, deterministic for an unchanged tree."""
    package_records.component(target)
    artifacts = Path(artifacts_root) if artifacts_root else paths.artifacts_dir()
    problems: list[dict] = []
    return {
        "schema": SCHEMA,
        "target": target,
        "generated_by": "merlin experiment index",
        "phase0_releases": _releases(target, artifacts, problems),
        "phase1_frozen": _phase1(target, problems),
        "phase2_best": _phase2(target, problems),
        "champions": _champions(target, artifacts_root, problems),
        "problems": problems,
    }


def render(document: dict) -> str:
    return "# Generated by `merlin experiment index`; do not edit.\n" + dump_yaml(document)


def write_index(target: str, *, artifacts_root: str | Path | None = None) -> Path:
    path = index_path(target, artifacts_root=artifacts_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{INDEX_NAME}.tmp")
    temporary.write_text(render(build_index(target, artifacts_root=artifacts_root)), encoding="utf-8")
    temporary.replace(path)
    return path


def refresh_for_run(run_dir: str | Path, phase: int | str) -> Path | None:
    """Regenerate the index of the target whose phase-``phase`` run root holds ``run_dir``.

    Called when a phase-1 run freezes and when a phase-2 run's ``best`` moves, so the index follows
    those events instead of waiting for someone to run ``merlin experiment index``.  A run outside the
    canonical ``out/runs/<target>/phase<N>/`` root is not one the index lists, so nothing is written.
    NEVER RAISES: the event it follows has already happened and is recorded in its own run; a failed
    refresh is printed (and leaves the index stale, which ``index --check`` reports), never silent.
    """
    import sys

    try:
        run = Path(run_dir).resolve()
        target = run.parent.parent.name
        if not target or paths.phase_runs_root(target, phase).resolve() != run.parent:
            return None
        return write_index(target)
    except Exception as exc:  # noqa: BLE001 -- see the docstring: reported, never raised
        print(
            f"[index] INDEX.yaml not refreshed after {run_dir}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return None


def is_current(target: str, *, artifacts_root: str | Path | None = None) -> bool:
    path = index_path(target, artifacts_root=artifacts_root)
    expected = render(build_index(target, artifacts_root=artifacts_root))
    return path.is_file() and path.read_text(encoding="utf-8") == expected


def rows_citing(target: str, run_dir: str | Path, *, artifacts_root: str | Path | None = None) -> dict[str, list]:
    """The index rows that name ``run_dir`` -- as itself, its source run, or a champion's lineage."""
    wanted = {_rel(Path(run_dir)), str(Path(run_dir).resolve())}
    document = build_index(target, artifacts_root=artifacts_root)

    def cites(row: dict) -> bool:
        lineage = (row.get("lineage") or {}).values()
        values = [row.get("run"), row.get("source_run")]
        values += [item for value in lineage for item in (value if isinstance(value, list) else [value])]
        return any(
            isinstance(v, str) and (v in wanted or (Path(v).is_absolute() and _rel(Path(v)) in wanted)) for v in values
        )

    sections = ("phase0_releases", "phase1_frozen", "phase2_best", "champions")
    return {name: [row for row in document[name] if cites(row)] for name in sections}
