"""Host-owned phase-1 seed preparation and persisted run-input verification.

All paths, target facts and identities are supplied by the caller. Importing this module
selects no target, workspace or environment. The native controller retains setup/resume and
final-grading order; these helpers preserve its records, byte hashing and refusal semantics.
Private snapshot records belong only in host-owned run evidence, never candidate metadata.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from collections.abc import Mapping
from pathlib import Path

import yaml

#: Names a grade's snapshot copy never carries (``shutil.ignore_patterns``: any depth, file or directory).
SNAPSHOT_IGNORED = ("build", "__pycache__", ".git")
#: Build-state names :func:`strip_build_state` deletes from a snapshot copy.
BUILD_STATE = ("CMakeCache.txt", "CMakeFiles", "build.ninja", ".ninja_deps", ".ninja_log", "cmake_install.cmake")


def outside_snapshot(relative_parts) -> bool:
    """Whether a submission-relative path is one a grade's snapshot copy omits or strips.

    The grader never sees such bytes, so they cannot be what a graded command read.
    """
    return any(part in SNAPSHOT_IGNORED or part in BUILD_STATE for part in relative_parts)


def strip_build_state(root: Path) -> None:
    """Delete all cmake/ninja build state under `root` so a graded copy builds from scratch in its OWN
    path. Excluding a dir named 'build' is not enough — a stale CMakeCache / ninja state elsewhere pins the
    original source dir and makes cmake error ('source does not match cache'). The abc9 baseline L3-0/20
    'build' bug. Guarantees a clean, relocatable build for every grade."""
    for pat in BUILD_STATE:
        for p in list(Path(root).rglob(pat)):
            try:
                shutil.rmtree(p) if p.is_dir() else p.unlink()
            except Exception:
                pass


def _make_agent_owned_tree_writable(root: Path) -> None:
    """Make a copied seed authorable without changing its preserved source modes.

    Frozen submissions intentionally have no write bits. ``copytree`` preserves those modes, but the
    destination is a fresh agent workspace rather than another frozen artifact. Restore only owner
    read/write (and directory traversal), preserving executable and group/other permission bits.
    """
    for path in (root, *root.rglob("*")):
        mode = stat.S_IMODE(path.stat().st_mode)
        if path.is_dir():
            path.chmod(mode | stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        elif path.is_file():
            path.chmod(mode | stat.S_IRUSR | stat.S_IWUSR)


def validate_seed_submission_source(source: str | Path, seeded: Path) -> Path:
    """Resolve a self-contained seed and reject overlap before any destination cleanup."""
    source_arg = Path(source).expanduser()
    if source_arg.is_symlink():
        raise RuntimeError(f"seed submission source must not be a symlink: {source_arg}")
    try:
        source_dir = source_arg.resolve(strict=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"seed submission source does not exist: {source_arg}") from exc
    if not source_dir.is_dir():
        raise RuntimeError(f"seed submission source is not a directory: {source_dir}")
    if not (source_dir / "manifest.yaml").is_file():
        raise RuntimeError(f"seed submission has no manifest.yaml: {source_dir}")

    seeded_resolved = Path(seeded).resolve(strict=False)
    if (
        source_dir == seeded_resolved
        or source_dir.is_relative_to(seeded_resolved)
        or seeded_resolved.is_relative_to(source_dir)
    ):
        raise RuntimeError(f"seed submission source and destination overlap: {source_dir} -> {seeded_resolved}")

    # copytree's default link handling can dereference a link into data outside the declared seed.  Reject
    # every link instead so the content identity below covers the complete input and is independently
    # reproducible from the preserved directory.
    for root, dirs, files in os.walk(source_dir, followlinks=False):
        for name in (*dirs, *files):
            path = Path(root) / name
            if path.is_symlink():
                raise RuntimeError(f"seed submission contains a symlink: {path}")
    return source_dir


def validate_seal_current_request(*, seal_current: bool, resume: bool, legacy_continuous: bool) -> None:
    """Keep an operator-requested incomplete seal on the certified resume path.

    This is not a success override.  It only lets a resumed run stop authoring at its last completed
    checkpoint and continue through the ordinary official grade/freeze path.  Downstream admission still
    sees a non-converged run and must name every completeness waiver explicitly.
    """
    if not seal_current:
        return
    if not resume:
        raise RuntimeError("--seal-current requires --resume of an existing checkpointed run")
    if legacy_continuous:
        raise RuntimeError("--seal-current requires the certified --schedule path, not --continuous")


def _seed_content(root: Path) -> dict:
    if root.is_symlink() or not root.is_dir() or not (root / "manifest.yaml").is_file():
        raise RuntimeError("seed submission snapshot is absent or indirect")
    content = hashlib.sha256()
    n_files = 0
    n_bytes = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise RuntimeError("seed submission snapshot contains an indirect or nonregular member")
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        data = path.read_bytes()
        content.update(rel.encode("utf-8"))
        content.update(b"\0")
        content.update(str(len(data)).encode("ascii"))
        content.update(b"\0")
        content.update(data)
        content.update(b"\0")
        n_files += 1
        n_bytes += len(data)
    return {"content_sha256": content.hexdigest(), "n_files": n_files, "n_bytes": n_bytes}


def _source_lineage(source: Path) -> dict:
    """Accept only the frozen final output of a previous run in this lineage."""
    from merlin.common import oot_repo as oot

    if source.name != "submission":
        raise RuntimeError("authoring seed must be a prior run's final submission")
    previous = source.parent
    environment = previous / "environment.yaml"
    if not environment.is_file():
        raise RuntimeError("authoring seed has no prior run environment")
    record = yaml.safe_load(environment.read_text(encoding="utf-8")) or {}
    run_id, lineage_id = record.get("run_id"), record.get("lineage_id")
    if not isinstance(run_id, str) or not run_id or not isinstance(lineage_id, str) or not lineage_id:
        raise RuntimeError("authoring seed has no recorded lineage")
    try:
        repo = previous / "oot"
        final = oot.resolve(repo, "HEAD")
        if oot.resolve(repo, oot.FROZEN_TAG) != final:
            raise RuntimeError("authoring seed is not the prior run's final oot commit")
        digest = oot.tree_digest(repo, final)
        if digest != oot.package_digest(source):
            raise RuntimeError("authoring seed bytes differ from the final oot commit tree")
    except oot.OotRepoError as exc:
        raise RuntimeError(f"authoring seed lacks a verified final oot commit: {exc}") from exc
    return {"lineage_id": lineage_id, "source_run_id": run_id,
            "source_commit": final, "source_tree_sha256": digest}


def verify_seed_submission(ws: Path, run_dir: Path, expected: Mapping) -> None:
    """Require the staged candidate and its run-owned seed receipt to match."""
    receipt = Path(run_dir) / "seed_submission.json"
    if receipt.is_symlink() or not receipt.is_file():
        raise RuntimeError("unpaid qualification has no ordinary seed receipt")
    try:
        recorded = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError("unpaid qualification seed receipt is unreadable") from exc
    if not isinstance(expected, Mapping) or recorded != dict(expected) or expected.get("version") != 1:
        raise RuntimeError("unpaid qualification seed receipt differs from the environment")
    if expected.get("authoring_seed") is True:
        source = validate_seed_submission_source(expected.get("source", ""), Path(ws) / "submission")
        if expected.get("lineage") != _source_lineage(source):
            raise RuntimeError("authoring seed lineage changed after selection")
    observed = _seed_content(Path(ws) / "submission")
    if any(expected.get(key) != value for key, value in observed.items()):
        raise RuntimeError("unpaid qualification submission changed after selection")


def verify_selected_seed_source(ws: Path, expected: Mapping) -> None:
    """Keep the operator-selected source stable across the official grade."""
    from merlin.common.tree_hash import hash_tree

    if not isinstance(expected, Mapping) or not isinstance(expected.get("selected_source_tree_sha256"), str):
        raise RuntimeError("unpaid qualification has no selected source tree identity")
    source = validate_seed_submission_source(expected.get("source", ""), Path(ws) / "submission")
    if hash_tree(source)["sha256"] != expected["selected_source_tree_sha256"]:
        raise RuntimeError("unpaid qualification selected source changed after freezing")


def graded_source_hash(root: Path) -> str:
    """Use the formal freeze's source identity, rejecting indirect members."""
    from merlin.common.tree_hash import hash_tree

    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("unpaid qualification grading submission is absent or indirect")
    for path in root.rglob("*"):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise RuntimeError("unpaid qualification grading submission contains an indirect member")
    return hash_tree(root)["sha256"]


def snapshot_seed_for_grade(ws: Path, run_dir: Path, expected: Mapping) -> Path:
    """Copy only a verified staged candidate into a fresh official grading path."""
    verify_seed_submission(ws, run_dir, expected)
    source = Path(ws) / "submission"
    destination = Path(run_dir) / "submission"
    if destination.exists() or destination.is_symlink():
        raise RuntimeError("unpaid qualification grading submission already exists")
    shutil.copytree(source, destination)
    if any(expected.get(key) != value for key, value in _seed_content(destination).items()):
        raise RuntimeError("unpaid qualification copied submission differs from selected bytes")
    verify_seed_submission(ws, run_dir, expected)
    return destination


def seed_submission(ws: Path, source: str | Path, run_dir: Path, *, require_stable_source: bool = False,
                    require_lineage: bool = False) -> dict:
    """Seed a *fresh* workspace from a preserved candidate and record its exact bytes.

    A changed public contract requires a new sealed run rather than a resume.  This copies only the
    candidate submission into that new treatment, refusing links so the seed is self-contained, and
    removes path-bound build products before computing the identity recorded with the run.
    """
    seeded = Path(ws).resolve(strict=True) / "submission"
    source_dir = validate_seed_submission_source(source, seeded)
    lineage = _source_lineage(source_dir) if require_lineage else None
    if require_stable_source:
        from merlin.common.tree_hash import hash_tree

        source_before = hash_tree(source_dir)["sha256"]

    if seeded.is_symlink():
        raise RuntimeError(f"seed submission destination must not be a symlink: {seeded}")
    if seeded.exists():
        shutil.rmtree(seeded)

    ignored_dirs = {"build", "__pycache__", ".git"}

    def _ignore(_directory: str, names: list[str]) -> set[str]:
        return {name for name in names if name in ignored_dirs}

    shutil.copytree(source_dir, seeded, ignore=_ignore)
    _make_agent_owned_tree_writable(seeded)
    strip_build_state(seeded)

    if require_stable_source and hash_tree(source_dir)["sha256"] != source_before:
        raise RuntimeError("unpaid qualification selected submission changed while copying")
    record = {
        "version": 1,
        "source": str(source_dir),
        **_seed_content(seeded),
        **({"selected_source_tree_sha256": source_before} if require_stable_source else {}),
        **({"authoring_seed": True, "lineage": lineage} if require_lineage else {}),
    }
    record_path = Path(run_dir) / "seed_submission.json"
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def stage_operator_errata(run_dir: Path, source: str | Path) -> dict:
    """Archive a pre-launch correction that ``round_brief`` will serve to the agent."""
    source_arg = Path(source).expanduser()
    if source_arg.is_symlink():
        raise RuntimeError(f"operator errata source must not be a symlink: {source_arg}")
    try:
        source_path = source_arg.resolve(strict=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"operator errata source does not exist: {source_arg}") from exc
    if not source_path.is_file():
        raise RuntimeError(f"operator errata source is not a regular file: {source_path}")
    data = source_path.read_bytes()
    destination = Path(run_dir) / "ERRATA.md"
    shutil.copyfile(source_path, destination)
    if destination.read_bytes() != data:
        raise RuntimeError("archived operator errata differs from its source")
    return {
        "source": str(source_path),
        "n_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def verify_operator_errata(expected: Mapping, run_dir: Path) -> None:
    """Fail closed if a pre-launch erratum no longer matches its recorded bytes."""
    if not isinstance(expected, Mapping):
        raise RuntimeError("persisted operator errata record is missing or malformed")
    path = Path(run_dir) / "ERRATA.md"
    try:
        observed_sha, observed_size = _file_digest(path)
    except RuntimeError as exc:
        raise RuntimeError(f"operator errata drifted after setup: {exc}") from exc
    if expected.get("sha256") != observed_sha or expected.get("n_bytes") != observed_size:
        raise RuntimeError(
            "operator errata drifted after setup: "
            f"expected {expected.get('sha256')}/{expected.get('n_bytes')} bytes, "
            f"observed {observed_sha}/{observed_size} bytes"
        )


# Merlin-arm-only docs staged into the workspace alongside the (shared) graded task.
MERLIN_WS_DOCS = ("TASK_ADDENDUM.md", "ALLOWED_MERLIN_TOOLS.md", "MERLIN_PROVENANCE_TEMPLATE.md")
_TREATMENT_BUNDLE_DECLARATIONS = (
    "input_bundle_manifest.yaml",
    "allowed_files.txt",
    "tools.txt",
)


def _file_digest(path: Path) -> tuple[str, int]:
    """Return ``(sha256, bytes)`` for one ordinary file, refusing symlink indirection."""
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"treatment input is not an ordinary file: {path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def bundle_manifest_identity(path: Path, bundle: dict) -> str:
    """Bind exact archived bytes to the declaration actually used for isolation."""
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("archived input bundle manifest is not an ordinary file")
    raw = path.read_bytes()
    if yaml.safe_load(raw) != bundle:
        raise RuntimeError("archived input bundle manifest differs from the loaded bundle")
    return hashlib.sha256(raw).hexdigest()


def _treatment_file_row(name: str, path: Path, *, required: bool) -> dict:
    """A stable row for a treatment file, including intentional absence of optional docs."""
    if path.exists() or path.is_symlink():
        digest, size = _file_digest(path)
        return {"name": name, "path": str(path), "present": True, "n_bytes": size, "sha256": digest}
    if required:
        raise RuntimeError(f"required treatment input is missing: {path}")
    return {"name": name, "path": str(path), "present": False, "n_bytes": 0, "sha256": None}


def treatment_snapshot_record(ws: Path, run_dir: Path, bundle_dir: Path, resolved_tools) -> dict:
    """Seal the exact prompt/docs and tool declarations that define one agent treatment.

    The bundle input payload already has its own immutable snapshot.  These files live outside that
    payload but still determine what the agent is told and which brokers start, so they need an equally
    explicit binding.  Optional declarations are represented by an absence row: creating one after setup
    is drift too (notably ``tools.txt``, whose later appearance changes tool resolution).
    """
    specs: list[tuple[str, Path, bool]] = [
        ("served/TASK.md", ws / "TASK.md", True),
        ("archived/TASK.md", run_dir / "TASK.md", True),
        ("archived_bundle/input_bundle_manifest.yaml", run_dir / "input_bundle_manifest.yaml", True),
    ]
    # New run-owned corpus views add private inputs to an effective declaration.
    # Keep the authored bytes as the source comparison while independently binding
    # the effective declaration above. Historical records retain their exact shape.
    authored = run_dir / "authored_input_bundle_manifest.yaml"
    authored_name = "archived_bundle/input_bundle_manifest.yaml"
    if authored.exists() or authored.is_symlink():
        authored_name = "archived_bundle/authored_input_bundle_manifest.yaml"
        specs.append((authored_name, authored, True))
    specs.extend((f"served/{name}", ws / name, False) for name in MERLIN_WS_DOCS)
    # Fresh corpus guidance is sealed alongside TASK.md. Historical treatments
    # without it retain their original record shape; never regenerate on resume.
    capability_docs = ("PUBLIC_CAPABILITIES.json", "PUBLIC_CAPABILITIES.md", "capability_query.py")
    if any((root / name).exists() or (root / name).is_symlink()
           for root in (ws, run_dir) for name in capability_docs):
        specs.extend((f"{kind}/{name}", root / name, True)
                     for kind, root in (("served", ws), ("archived", run_dir))
                     for name in capability_docs)
    for name in ("construction_actions.py", "CONSTRUCTION_ACTIONS.json", "OPTIMIZATION_BY_OPERATION.json",
                 "COMMAND_ORDERING_SELECTION.json", "REFERENCE_EXPERIMENTS.json", "REFERENCE_MAP.json"):
        if any((root / name).exists() or (root / name).is_symlink() for root in (ws, run_dir)):
            specs.extend((f"{kind}/{name}", root / name, True)
                         for kind, root in (("served", ws), ("archived", run_dir)))
    # Staged task-section skills (and their record) are served instructions too. A run without them
    # keeps its historical record shape.
    from .task_disclosure import staged_paths

    specs.extend((name, path, True) for name, path in staged_paths(ws, run_dir))
    from .skill_staging import staged_paths as staged_author_skills

    specs.extend((name, path, True) for name, path in staged_author_skills(ws, run_dir))
    specs.extend(
        (f"source_bundle/{name}", bundle_dir / name, name == "input_bundle_manifest.yaml")
        for name in _TREATMENT_BUNDLE_DECLARATIONS
    )
    rows = [_treatment_file_row(name, path, required=required) for name, path, required in specs]
    if len({row["name"] for row in rows}) != len(rows):
        raise RuntimeError("treatment input names are not unique")
    by_name = {row["name"]: row for row in rows}
    if by_name[authored_name]["sha256"] != by_name["source_bundle/input_bundle_manifest.yaml"]["sha256"]:
        raise RuntimeError("source input bundle manifest changed while the run was being staged")

    tool_ids = list(resolved_tools)
    if any(not isinstance(tool, str) or not tool for tool in tool_ids):
        raise RuntimeError("resolved tool ids must be non-empty strings")
    if len(set(tool_ids)) != len(tool_ids):
        raise RuntimeError("resolved tool ids must be unique")
    aggregate = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item["name"]):
        aggregate.update(row["name"].encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(b"1" if row["present"] else b"0")
        aggregate.update(b"\0")
        aggregate.update(str(row["n_bytes"]).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update((row["sha256"] or "-").encode("ascii"))
        aggregate.update(b"\n")
    for index, tool_id in enumerate(tool_ids):
        aggregate.update(f"tool\0{index}\0{tool_id}\n".encode())
    return {
        "version": 1,
        "content_sha256": aggregate.hexdigest(),
        "n_files_present": sum(bool(row["present"]) for row in rows),
        "files": rows,
        "resolved_tool_ids": tool_ids,
    }


def _treatment_row_identity(row: Mapping) -> dict:
    """What actually defines the treatment for one file.

    ``path`` records WHERE the file was read, not WHAT was served, and
    ``treatment_snapshot_record``'s aggregate deliberately excludes it.  The same
    workspace legitimately spells differently across runs -- a pinned descriptor
    directory reaches the shared ``_qa_ws`` through a symlink, so a resumed run
    sees ``targets/<pin>/_qa_ws/...`` where setup recorded ``targets/<target>/_qa_ws/...``.
    Comparing the spelling would fail a run whose served bytes are identical, which
    is drift theatre: it blocks the honest resume it was written to protect.
    """
    return {key: value for key, value in row.items() if key != "path"}


def verify_treatment_snapshot(expected: Mapping, ws: Path, run_dir: Path, bundle_dir: Path, resolved_tools) -> dict:
    """Recompute a treatment binding and fail closed on any prompt/tool drift."""
    if not isinstance(expected, Mapping) or expected.get("version") != 1:
        raise RuntimeError("persisted treatment snapshot is missing or malformed")
    try:
        observed = treatment_snapshot_record(ws, run_dir, bundle_dir, resolved_tools)
    except RuntimeError as exc:
        raise RuntimeError(f"experiment treatment drifted after setup: {exc}") from exc
    expected_rows = {
        row.get("name"): _treatment_row_identity(row) for row in expected.get("files", []) if isinstance(row, Mapping)
    }
    observed_rows = {row["name"]: _treatment_row_identity(row) for row in observed["files"]}
    changed = sorted(
        name for name in set(expected_rows) | set(observed_rows) if expected_rows.get(name) != observed_rows.get(name)
    )
    if expected.get("resolved_tool_ids") != observed["resolved_tool_ids"]:
        changed.append("resolved_tool_ids")
    if not changed and expected.get("content_sha256") != observed["content_sha256"]:
        changed.append("content_sha256")
    if not changed and expected.get("n_files_present") != observed["n_files_present"]:
        changed.append("n_files_present")
    if changed:
        raise RuntimeError("experiment treatment drifted after setup: " + ", ".join(changed))
    return observed


def _snapshot_path_for_live_path(snapshot_root: Path, live_path: Path, repo: Path) -> Path:
    """Map a declared live destination to its private bundle-snapshot location."""
    live_path = live_path.absolute()
    repo = repo.absolute()
    try:
        return snapshot_root / "repo" / live_path.relative_to(repo)
    except ValueError:
        return snapshot_root / "external" / Path(*live_path.parts[1:])


def hidden_snapshot_dir(snapshot_root: Path, te, repo: Path) -> Path:
    """Resolve this target's hidden sibling inside the immutable bundle snapshot, never live."""
    corpus = Path(te.capsule_corpus)
    live_corpus = corpus if corpus.is_absolute() else repo / corpus
    return _snapshot_path_for_live_path(snapshot_root, live_corpus.parent / "hidden", repo)


def subtree_snapshot_record(root: Path) -> dict:
    """Canonical file/count/byte binding for an operator-only frozen subtree."""
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"hidden snapshot subtree is missing or unsafe: {root}")
    rows: list[dict] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"hidden snapshot subtree contains a symlink: {path}")
        if path.is_dir():
            continue
        digest, size = _file_digest(path)
        rows.append({"path": path.relative_to(root).as_posix(), "n_bytes": size, "sha256": digest})
    if not rows:
        raise RuntimeError(f"hidden snapshot subtree is empty: {root}")
    aggregate = hashlib.sha256()
    for row in rows:
        aggregate.update(row["path"].encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(row["n_bytes"]).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(row["sha256"].encode("ascii"))
        aggregate.update(b"\n")
    return {
        "version": 1,
        "path": str(root.resolve(strict=True)),
        "content_sha256": aggregate.hexdigest(),
        "n_files": len(rows),
        "n_bytes": sum(row["n_bytes"] for row in rows),
        "n_capsules": sum(Path(row["path"]).name == "capsule.yaml" for row in rows),
    }


def verify_subtree_snapshot(expected: Mapping) -> tuple[Path, dict]:
    """Verify a persisted hidden-subtree record and return its already-frozen path."""
    if not isinstance(expected, Mapping) or expected.get("version") != 1:
        raise RuntimeError("persisted hidden subtree snapshot is missing or malformed")
    raw_path = expected.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise RuntimeError("persisted hidden subtree snapshot path is missing")
    root = Path(raw_path)
    observed = subtree_snapshot_record(root)
    if dict(expected) != observed:
        raise RuntimeError(
            f"hidden capsule snapshot drifted after setup at {root}: "
            f"expected {expected.get('content_sha256')}, observed {observed['content_sha256']}"
        )
    return root, observed


def verify_persisted_run_inputs(
    environment: Mapping,
    *,
    identity: Mapping,
    task_scope: Mapping,
    ws: Path,
    run_dir: Path,
    bundle_dir: Path,
    resolved_tools,
    expected_hidden_dir: Path | None = None,
) -> Path | None:
    """Common fail-closed gate for resume and the post-authoring official grade."""
    if not isinstance(environment, Mapping):
        raise RuntimeError("persisted environment record is missing or malformed")
    for field, observed in identity.items():
        if environment.get(field) != observed:
            raise RuntimeError(f"persisted environment {field} changed ({environment.get(field)!r} != {observed!r})")
    if environment.get("task_scope") != dict(task_scope):
        raise RuntimeError("descriptor-derived task scope drifted")
    verify_treatment_snapshot(environment.get("treatment_snapshot"), ws, run_dir, bundle_dir, resolved_tools)
    if environment.get("operator_errata") is not None:
        verify_operator_errata(environment["operator_errata"], run_dir)
    if expected_hidden_dir is None:
        return None
    hidden_dir, _ = verify_subtree_snapshot(environment.get("hidden_capsule_snapshot"))
    if hidden_dir.resolve(strict=True) != expected_hidden_dir.resolve(strict=True):
        raise RuntimeError("hidden capsule record does not name this target's private bundle-snapshot subtree")
    return hidden_dir
