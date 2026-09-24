"""Read-only conformance coverage for one verified Phase 0 source run."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from merlin.targetgen import conformance

from ..spec import SpecError
from .preparation import _members, source_run


def _public_category_roots(corpus: Path) -> tuple[list[Path], int]:
    """Resolve the category roots understood by the shared capsule scanners.

    Phase 0 owns ``corpus/<category>/<name>/capsule.yaml``; the scanners take
    category roots and look one directory below each. Validate that no source
    member is silently omitted before asking them to classify coverage.
    """
    members = _members(corpus)
    all_paths = {path.relative_to(corpus).as_posix() for path in corpus.rglob("capsule.yaml")}
    expected_paths = {f"{key}/capsule.yaml" for key in members}
    if all_paths != expected_paths:
        raise SpecError("phase-0 corpus has capsule descriptors outside category/member layout")
    provenance = yaml.safe_load((corpus / "MANIFEST.yaml").read_text(encoding="utf-8"))
    if not isinstance(provenance, dict):
        raise SpecError("phase-0 corpus manifest must be a mapping")
    declared = provenance.get("generated")
    if not isinstance(declared, list) or not declared or any(not isinstance(key, str) for key in declared):
        raise SpecError("phase-0 corpus manifest must declare generated public members")
    public = {key: document for key, (_, document) in members.items() if not key.startswith("hidden/")}
    if len(declared) != len(set(declared)) or set(declared) != set(public):
        raise SpecError("phase-0 corpus manifest does not account for every public capsule")
    hidden = len(members) - len(public)
    held_out = provenance.get("held_out") or {}
    if not isinstance(held_out, dict) or held_out.get("n_generated", 0) != hidden:
        raise SpecError("phase-0 corpus manifest does not account for every hidden capsule")
    n_public = sum(document.get("label") == "public" for document in public.values())
    if not n_public:
        raise SpecError("phase-0 corpus has no public-labelled capsules to measure")
    roots = sorted({corpus / key.split("/", 1)[0] for key in public})
    return roots, n_public


def inspect_run(run_dir: Path, spec_path: Path) -> dict:
    """Measure public source-pool coverage without grading or admitting a cohort.

    The completed run and its frozen inputs are verified before reading capsules.
    The conformance spec is an explicit, separately hashed diagnostic input; this
    command neither changes that reference nor turns a source-pool match into a
    numerical or hardware verdict.
    """
    source = run_dir.expanduser().resolve(strict=True)
    plan, _, corpus = source_run(source)
    selected = spec_path.expanduser().resolve(strict=True)
    if not selected.is_file():
        raise SpecError("conformance spec must be a regular file")
    raw = selected.read_bytes()
    try:
        spec = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise SpecError(f"invalid conformance YAML: {selected}") from exc
    if not isinstance(spec, dict) or not isinstance(spec.get("cells"), list):
        raise SpecError("conformance spec must contain a cells list")
    target = plan["target"]
    if spec.get("target") != target:
        raise SpecError(f"conformance spec target {spec.get('target')!r} differs from run target {target!r}")
    edge = (spec.get("boundaries") or {}).get("tile_edge")
    if edge is not None and (type(edge) is not int or edge < 1):
        raise SpecError("conformance spec tile_edge must be a positive integer or absent")
    category_roots, n_public = _public_category_roots(corpus)
    result = conformance.uncovered(spec, category_roots, labels={"public"}, tile_dim=edge)
    if not result["corpus_cells"]:
        raise SpecError("phase-0 public capsules yielded no classifiable coverage cells")
    return {
        "schema_version": 1,
        "target": target,
        "phase0_run": str(source),
        "corpus": str(corpus),
        "spec": {"path": str(selected), "sha256": hashlib.sha256(raw).hexdigest()},
        "scope": "generated public source pool; not admitted, graded, or certified",
        "n_public_capsules_scanned": n_public,
        "coverage": result,
    }
