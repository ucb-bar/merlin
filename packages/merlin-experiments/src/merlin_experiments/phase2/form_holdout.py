"""Commit/reveal of a form-scale performance holdout cohort.

The PK holdout (:mod:`.holdout_corpus`) selects unseen points of one tile-scale law. A cohort at the
extents where a compiler's tiling, residency and movement decide its cost needs a different source:
form-perf members minted from a PRIVATE roster of independent performance-scale workloads -- the same
generators and form classes as the public tuning roster, at other widths. The operator generates
those members in a separate, host-private Phase 0 run before authoring starts.

:func:`commit_form_holdout` selects the members whose representative application is in the private
roster, refuses any whose exact workload a public tuning member already measures, and writes a
public commitment that carries counts and digests only. :func:`reveal_form_holdout` runs after every
candidate is sealed: it re-verifies the committed bytes, copies the members into a fresh corpus and
writes the same v2 reveal manifest the PK holdout writes, so the existing held-out qualification and
paired measurement read it unchanged. Nothing here generates a workload or names a target.
"""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from . import gsim_gate as GATE
from . import gsim_workload as WORKLOAD
from . import holdout_corpus as HC
from . import revealed_corpus as RC

SCHEMA = "merlin.phase2.form_holdout_commitment.v1"
COHORT = "PW_form_scale_generalization"
PERFORMANCE_CATEGORY = "_perf"


def _descriptor(directory: Path) -> Mapping[str, Any]:
    path = HC._assert_plain_file(directory / "capsule.yaml", label=f"{directory.name} descriptor")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping) or document.get("name") != directory.name:
        raise HC.HoldoutError(f"capsule descriptor identity differs from its directory: {directory.name}")
    return document


def _members(root: Path) -> list[Path]:
    category = Path(root) / PERFORMANCE_CATEGORY
    if category.is_symlink() or not category.is_dir():
        raise HC.HoldoutError(f"generated corpus has no {PERFORMANCE_CATEGORY} category: {root}")
    found = []
    for directory in sorted(category.iterdir()):
        if directory.is_symlink():
            raise HC.HoldoutError(f"performance member is linked: {directory}")
        if directory.is_dir():
            found.append(directory)
    return found


def select_members(generated_root: Path, *, applications: Sequence[str], family: str = "PW") -> list[Path]:
    """Form members of ``family`` whose representative group comes from a private roster application."""
    roster = {str(label) for label in applications}
    if not roster:
        raise HC.HoldoutError("the private performance-scale roster is empty")
    selected = []
    for directory in _members(generated_root):
        performance = _descriptor(directory).get("performance") or {}
        form = performance.get("form") or {}
        source = (form.get("representative") or {}).get("application")
        if performance.get("family") == family and source in roster:
            selected.append(directory)
    return selected


def _workloads(directories: Sequence[Path]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for directory in directories:
        identity = GATE.workload_sha256(WORKLOAD.derive_workload(directory / "capsule.yaml"))
        if identity in out:
            raise HC.HoldoutError("two holdout members measure the same workload")
        out[identity] = directory
    return out


def _refuse_heldout_layers(directories: Sequence[Path], heldout_layers: Path | None) -> int:
    """Operator guard: refuse a cohort member whose workload equals a held-out network layer."""
    if heldout_layers is None:
        return 0
    from merlin_experiments.phase0 import heldout_layers as HL

    layers = HL.load(heldout_layers)
    collisions = []
    for directory in directories:
        shape = WORKLOAD.derive_workload(directory / "capsule.yaml").get("shape") or {}
        hit = HL.workload_collision(shape, layers)
        if hit is not None:
            extents = [shape.get("m"), shape.get("k"), shape.get("n")]
            collisions.append({"member": directory.name, "extents": extents, "network": hit[0], "layer": hit[1]})
    HL.refuse(collisions, what="form-holdout")
    return len(directories)


def _source_tree(directories: Sequence[Path]) -> dict[str, Any]:
    rows = []
    for directory in sorted(directories, key=lambda item: item.name):
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise HC.HoldoutError(f"holdout member contains a symlink: {path}")
            if path.is_file():
                rows.append(
                    {
                        "path": f"{directory.name}/{path.relative_to(directory).as_posix()}",
                        "bytes": path.stat().st_size,
                        "sha256": HC._sha256_file(path),
                    }
                )
    return {"files": rows, "sha256": HC._sha256(HC._canonical_json(rows))}


def roster_sha256(applications: Sequence[str]) -> str:
    """Digest of the private roster: what a public record may carry instead of its labels."""
    return hashlib.sha256(HC._canonical_json(sorted({str(label) for label in applications}))).hexdigest()


def commit_form_holdout(
    generated_root: Path,
    public_commitment: Path,
    host_private_dir: Path,
    *,
    target: str,
    applications: Sequence[str],
    tuning_root: Path,
    candidate_ids: Sequence[str],
    family: str = "PW",
    agent_view_root: Path | None = None,
    heldout_layers: Path | None = None,
) -> dict[str, Path]:
    """Freeze the private form-scale cohort before authoring; publish only counts and digests."""
    ids = [str(value) for value in candidate_ids]
    if not ids or len(set(ids)) != len(ids) or any(not value or "/" in value for value in ids):
        raise HC.HoldoutError("candidate ids must be unique non-empty path components")
    public_commitment = HC._fresh_parent(Path(public_commitment), label="public commitment")
    host_private_dir = HC._fresh_parent(Path(host_private_dir), label="host-private holdout directory")
    if agent_view_root is not None:
        view = Path(agent_view_root)
        if HC._inside(host_private_dir, view) or HC._inside(Path(generated_root), view):
            raise HC.HoldoutError("the private holdout or its generated corpus is inside the agent view")
        if not HC._inside(public_commitment, view):
            raise HC.HoldoutError("public commitment is not inside the declared agent view")
    members = select_members(generated_root, applications=applications, family=family)
    if len(members) < HC.MIN_MEMBERS:
        raise HC.HoldoutError(f"form-scale holdout has {len(members)} member(s); need {HC.MIN_MEMBERS}")
    _refuse_heldout_layers(members, heldout_layers)
    held = _workloads(members)
    tuning = _workloads(_members(tuning_root))
    overlap = len(set(held) & set(tuning))
    if overlap:
        # Counts only: naming the member would publish a held-out workload.
        raise HC.HoldoutError(f"{overlap} holdout member(s) repeat a public tuning workload")
    tree = _source_tree(members)
    public = {
        "schema": SCHEMA,
        "target": str(target),
        "family": family,
        "cohort": COHORT,
        "member_count": len(members),
        "members_tree_sha256": tree["sha256"],
        "roster_sha256": roster_sha256(applications),
        "tuning_workloads_checked": len(tuning),
        "disjoint_from_tuning": True,
        "expected_candidate_count": len(ids),
    }
    private = {
        "schema": SCHEMA,
        "public_commitment_sha256": HC._sha256(HC._canonical_json(public)),
        "generated_root": str(Path(generated_root).resolve()),
        "applications": sorted({str(label) for label in applications}),
        "members": [directory.name for directory in members],
        "candidate_ids": ids,
        "members_tree": tree,
    }
    host_private_dir.mkdir(mode=0o700)
    host_private_dir.chmod(0o700)
    state = host_private_dir / "state.json"
    HC._write_exclusive(state, HC._canonical_json(private), 0o600)
    HC._write_exclusive(public_commitment, HC._canonical_json(public), 0o444)
    return {
        "public_commitment": public_commitment.resolve(),
        "host_private_dir": host_private_dir.resolve(),
        "state": state.resolve(),
    }


def reveal_form_holdout(
    public_commitment: Path,
    host_private_dir: Path,
    output_dir: Path,
    *,
    candidate_seals: Mapping[str, Path],
    heldout_layers: Path | None = None,
) -> Path:
    """After every candidate is sealed: verify the committed bytes and write the v2 reveal."""
    public_path = HC._assert_frozen_file(Path(public_commitment), label="public form-holdout commitment")
    public = HC._load_json(public_path, label="public form-holdout commitment")
    private = HC._load_json(Path(host_private_dir) / "state.json", label="host-private form-holdout state")
    if public.get("schema") != SCHEMA or private.get("public_commitment_sha256") != HC._sha256(
        HC._canonical_json(public)
    ):
        raise HC.HoldoutError("public form-holdout commitment changed after it was prepared")
    receipts = HC._verify_candidate_seals(list(private["candidate_ids"]), candidate_seals)
    category = Path(private["generated_root"]) / PERFORMANCE_CATEGORY
    members = [category / name for name in private["members"]]
    if _source_tree(members) != private["members_tree"]:
        raise HC.HoldoutError("committed form-holdout members changed before the reveal")
    _refuse_heldout_layers(members, heldout_layers)
    output_dir = HC._fresh_parent(Path(output_dir), label="form-holdout reveal directory")
    output_dir.mkdir(mode=0o700)
    rows = []
    for source in members:
        target_dir = output_dir / PERFORMANCE_CATEGORY / source.name
        shutil.copytree(source, target_dir, symlinks=False)
        shape = WORKLOAD.derive_workload(target_dir / "capsule.yaml").get("shape") or {}
        rows.append(
            {
                "name": source.name,
                "path": f"{PERFORMANCE_CATEGORY}/{source.name}",
                "family": public["family"],
                "cohort": COHORT,
                "M": shape.get("m"),
                "N": shape.get("n"),
                "K": shape.get("k"),
            }
        )
    manifest_path = output_dir / "holdout_manifest.json"
    manifest = {
        "schema_version": HC.SCHEMA_VERSION,
        "kind": "generated_performance_holdout_reveal",
        "commitment": {"path": str(public_path), "sha256": HC._sha256_file(public_path), "schema": SCHEMA},
        "domain": {"target": public["target"], "family": public["family"], "cohort": COHORT},
        "cohorts": {
            COHORT: {
                "family": public["family"],
                "claim": "DIFFERENTIAL",
                "member_count": len(rows),
                "scope": (
                    "paired candidate/vendor form members minted from a private roster of independent "
                    "performance-scale workloads; same generators and form classes as the tuning roster, "
                    "disjoint workloads"
                ),
            }
        },
        "members": rows,
        "candidate_seals": receipts,
        "corpus": RC.tree_without_manifest(output_dir, manifest_path),
    }
    HC._write_exclusive(manifest_path, HC._canonical_json(manifest), 0o400)
    HC._make_host_readonly(output_dir)
    return manifest_path.resolve()


__all__ = ["COHORT", "SCHEMA", "commit_form_holdout", "reveal_form_holdout", "roster_sha256", "select_members"]
