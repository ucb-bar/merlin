"""Explicit release preparation for an already selected historical corpus.

This path records preserved provenance, never a synthetic Phase 0 execution or
new derivation. Review and native snapshot/answer isolation remain mandatory.
"""

from datetime import UTC, datetime
from pathlib import Path

from ..spec import SpecError
from .preparation import admission, copy_input, ordinary_tree, private_json, scaffold


def verify_preserved(root: Path, prepared: dict) -> None:
    from ..runner import fingerprint

    if prepared.get("generation_lineage") is not None or prepared.get("source_run") is not None:
        raise SpecError("preserved corpus cannot claim Phase 0 execution lineage")
    selected = prepared.get("preserved_sources")
    if not isinstance(selected, list) or not selected:
        raise SpecError("preserved release has no selected source commitments")
    for row in selected:
        source = Path(row["path"])
        ordinary_tree(source)
        if fingerprint(source) != row["sha256"]:
            raise SpecError("preserved corpus source changed before review")
    if not (root / "payload/corpus/MANIFEST.yaml").is_file():
        raise SpecError("preserved corpus has no provenance manifest")


def _select_sources(descriptor: Path, output: Path | None):
    from merlin.common.paths import out_dir
    from merlin.targetgen.target_experiment import load_target_experiment

    descriptor = descriptor.expanduser().resolve(strict=True)
    ordinary_tree(descriptor)
    te = load_target_experiment(descriptor)
    if output is None:
        raise SpecError("preserved corpus requires an explicit fresh --output")
    root = output.expanduser().absolute()
    if root.exists() or root.is_symlink() or not root.resolve().is_relative_to(out_dir().resolve() / "artifacts"):
        raise SpecError("preserved release requires a fresh configured artifact destination")
    if te.graded_release_admission:
        raise SpecError("preserved corpus requires explicit cohort counts, not derived release admission")
    if te.graded_expected_source_capsules is None or te.graded_expected_admitted_capsules is None:
        raise SpecError("preserved corpus requires explicit expected public counts")
    roots = list(dict.fromkeys([*te.graded_roots(), *te.hidden_roots(), *te.perf_roots(), *te.model_layer_roots()]))
    sources = [descriptor, *roots, te.capsule_corpus.parent / "MANIFEST.yaml"]
    for source in sources:
        ordinary_tree(source)
        if root == source or root.is_relative_to(source) or source.is_relative_to(root):
            raise SpecError("preserved release overlaps a selected source")
    return te, root, roots, sources


def _stage_preserved(te, root: Path, roots: list[Path], *, rtl_facts: Path | None) -> dict:
    from ..runner import fingerprint

    contract = te.declared_contract_path() if te.declared_contract else None
    root.mkdir(parents=True, mode=0o700)
    private = root / "private"
    private.mkdir(mode=0o700)
    corpus = root / "payload/corpus"
    corpus.mkdir(parents=True, mode=0o700)
    for source in roots:
        copy_input(source, corpus / source.name, private=source in te.hidden_roots())
    copy_input(te.capsule_corpus.parent / "MANIFEST.yaml", corpus / "MANIFEST.yaml")
    return scaffold(
        te,
        corpus,
        root / "payload/experiment",
        private=private,
        selected_facts=rtl_facts,
        selected_contract=(contract, fingerprint(contract)) if contract else None,
    )


def prepare_preserved(descriptor: Path, output: Path | None, *, rtl_facts: Path | None) -> dict:
    from merlin.common.storage_lifecycle import lease

    from ..runner import fingerprint
    from .release import inspect_release

    te, root, roots, sources = _select_sources(descriptor, output)
    selected = [{"path": str(path), "sha256": fingerprint(path)} for path in sources]
    with lease(root, owner="preserved-corpus-preparation"):
        scaffolding = _stage_preserved(te, root, roots, rtl_facts=rtl_facts)
        private = root / "private"
        checked = admission(
            root / "payload/experiment/target_experiment.yaml", coverage_output=private / "workload-coverage.json"
        )
        prepared = {
            "schema_version": 1,
            "target": te.target,
            "source_kind": "preserved_corpus",
            "source_run": None,
            "generation_lineage": None,
            "preserved_sources": selected,
            "scaffolding": scaffolding,
            "admission": checked,
            "payload_sha256": fingerprint(root / "payload"),
            "prepared_at": datetime.now(UTC).isoformat(),
        }
        verify_preserved(root, prepared)
        private_json(private / "preparation.json", prepared)
    return inspect_release(root)
