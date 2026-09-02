"""A fresh QA run may continue from a preserved sandboxed candidate under a new sealed bundle.

This is required when the public contract itself changes: mutating an existing run's frozen bundle would
mix two treatments, while starting from an empty submission would throw away the compiler work already
earned under the prior contract.
"""
from __future__ import annotations

import importlib.util
import json
import sys

import pytest

from merlin.common.paths import merlin_dir


def _loop():
    harness = merlin_dir() / "experiments" / "capsule_bench" / "harness"
    if str(harness) not in sys.path:
        sys.path.insert(0, str(harness))
    path = harness / "run_baseline_qa_loop.py"
    spec = importlib.util.spec_from_file_location("seed_submission_loop", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seed_copies_candidate_cleanly_and_records_exact_identity(tmp_path):
    loop = _loop()
    source = tmp_path / "preserved" / "submission"
    (source / "mlir_oot").mkdir(parents=True)
    (source / "manifest.yaml").write_text("commands: {}\n", encoding="utf-8")
    (source / "mlir_oot" / "backend.py").write_text("VALUE = 1\n", encoding="utf-8")
    (source / "mlir_oot" / "CMakeCache.txt").write_text("stale\n", encoding="utf-8")
    (source / "mlir_oot" / "__pycache__").mkdir()
    (source / "mlir_oot" / "__pycache__" / "backend.pyc").write_bytes(b"stale")
    ws, run_dir = tmp_path / "new-ws", tmp_path / "new-run"
    (ws / "submission").mkdir(parents=True)
    (ws / "submission" / "empty-marker").write_text("replace me")
    run_dir.mkdir()

    record = loop._seed_submission(ws, source, run_dir)

    seeded = ws / "submission"
    assert (seeded / "manifest.yaml").is_file()
    assert (seeded / "mlir_oot" / "backend.py").read_text() == "VALUE = 1\n"
    assert not (seeded / "empty-marker").exists()
    assert not (seeded / "mlir_oot" / "CMakeCache.txt").exists()
    assert not (seeded / "mlir_oot" / "__pycache__").exists()
    assert (source / "mlir_oot" / "CMakeCache.txt").is_file(), "the preserved source is read-only"
    persisted = json.loads((run_dir / "seed_submission.json").read_text())
    assert persisted == record
    assert record["content_sha256"] and record["n_files"] == 2
    assert record["source"] == str(source.resolve())


def test_seed_refuses_unsafe_or_ungradeable_sources(tmp_path):
    loop = _loop()
    ws, run_dir = tmp_path / "ws", tmp_path / "run"
    ws.mkdir(); run_dir.mkdir()
    missing_manifest = tmp_path / "missing-manifest"
    missing_manifest.mkdir()
    with pytest.raises(RuntimeError, match="manifest.yaml"):
        loop._seed_submission(ws, missing_manifest, run_dir)

    source = tmp_path / "with-link"
    source.mkdir()
    (source / "manifest.yaml").write_text("commands: {}\n")
    (source / "escape").symlink_to(tmp_path / "outside")
    with pytest.raises(RuntimeError, match="symlink"):
        loop._seed_submission(ws, source, run_dir)


def test_seed_refuses_source_target_overlap_without_deleting_candidate(tmp_path):
    loop = _loop()
    ws, run_dir = tmp_path / "ws", tmp_path / "run"
    source = ws / "submission"
    source.mkdir(parents=True)
    (source / "manifest.yaml").write_text("commands: {}\n", encoding="utf-8")
    run_dir.mkdir()

    with pytest.raises(RuntimeError, match="overlap"):
        loop._seed_submission(ws, source, run_dir)

    assert (source / "manifest.yaml").is_file(), "a refused seed must not delete its source"


def test_seed_option_is_fresh_run_only_and_wired_before_launch():
    loop = _loop()
    import inspect
    body = inspect.getsource(loop.main)
    assert '"--seed-submission"' in body
    assert "--seed-submission cannot be combined with --resume" in body
    assert "_seed_submission(ws, _seed_source_preflight, run_dir)" in body
    assert body.index("_validate_seed_submission_source(") < body.index("shutil.rmtree(ws_root)")
    assert '"--operator-errata"' in body
    assert "--operator-errata cannot be combined with --resume" in body
    assert "_stage_operator_errata(run_dir, a.operator_errata)" in body


def test_operator_errata_is_archived_exactly_and_rejects_links(tmp_path):
    loop = _loop()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    source = tmp_path / "ERRATA.source.md"
    source.write_text("# Corrected contract\n\nRe-derive the stride.\n", encoding="utf-8")

    record = loop._stage_operator_errata(run_dir, source)

    archived = run_dir / "ERRATA.md"
    assert archived.read_bytes() == source.read_bytes()
    assert record == {
        "source": str(source.resolve()),
        "n_bytes": len(source.read_bytes()),
        "sha256": loop.hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    link = tmp_path / "linked.md"
    link.symlink_to(source)
    with pytest.raises(RuntimeError, match="symlink"):
        loop._stage_operator_errata(run_dir, link)


def test_operator_errata_provenance_detects_archived_content_drift(tmp_path):
    loop = _loop()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    source = tmp_path / "ERRATA.source.md"
    source.write_text("# Original correction\n", encoding="utf-8")
    record = loop._stage_operator_errata(run_dir, source)

    loop._verify_operator_errata(record, run_dir)
    (run_dir / "ERRATA.md").write_text("# Different correction\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="operator errata drifted"):
        loop._verify_operator_errata(record, run_dir)
