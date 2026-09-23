"""A fresh QA run may continue from a preserved sandboxed candidate under a new sealed bundle.

This is required when the public contract itself changes: mutating an existing run's frozen bundle would
mix two treatments, while starting from an empty submission would throw away the compiler work already
earned under the prior contract.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
import sys

import pytest
from merlin_experiments.phase1 import run_inputs as RI

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

    record = RI.seed_submission(ws, source, run_dir)

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


def test_seed_from_a_frozen_run_is_writable_without_mutating_the_source(tmp_path):
    """A frozen submission is a valid seed, but its read-only modes are not part of the new treatment.

    Keeping those modes makes the new agent unable to add a compiler file, update its manifest, or
    create ``READY_FOR_BARRIER``. The source must remain frozen while only the workspace copy becomes
    authorable.
    """
    source = tmp_path / "preserved" / "submission"
    package = source / "mlir_oot"
    package.mkdir(parents=True)
    manifest = source / "manifest.yaml"
    backend = package / "backend.py"
    manifest.write_text("commands: {}\n", encoding="utf-8")
    backend.write_text("VALUE = 1\n", encoding="utf-8")
    for directory in (package, source):
        directory.chmod(0o555)
    for file in (manifest, backend):
        file.chmod(0o444)
    ws, run_dir = tmp_path / "new-ws", tmp_path / "new-run"
    ws.mkdir()
    run_dir.mkdir()

    RI.seed_submission(ws, source, run_dir)

    seeded = ws / "submission"
    assert seeded.stat().st_mode & stat.S_IWUSR
    assert (seeded / "mlir_oot").stat().st_mode & stat.S_IWUSR
    assert (seeded / "manifest.yaml").stat().st_mode & stat.S_IWUSR
    assert (seeded / "mlir_oot/backend.py").stat().st_mode & stat.S_IWUSR
    assert not (source.stat().st_mode & stat.S_IWUSR)
    assert not (manifest.stat().st_mode & stat.S_IWUSR)


def test_seed_refuses_unsafe_or_ungradeable_sources(tmp_path):
    ws, run_dir = tmp_path / "ws", tmp_path / "run"
    ws.mkdir()
    run_dir.mkdir()
    missing_manifest = tmp_path / "missing-manifest"
    missing_manifest.mkdir()
    with pytest.raises(RuntimeError, match="manifest.yaml"):
        RI.seed_submission(ws, missing_manifest, run_dir)

    source = tmp_path / "with-link"
    source.mkdir()
    (source / "manifest.yaml").write_text("commands: {}\n")
    (source / "escape").symlink_to(tmp_path / "outside")
    with pytest.raises(RuntimeError, match="symlink"):
        RI.seed_submission(ws, source, run_dir)


def test_seed_refuses_source_target_overlap_without_deleting_candidate(tmp_path):
    ws, run_dir = tmp_path / "ws", tmp_path / "run"
    source = ws / "submission"
    source.mkdir(parents=True)
    (source / "manifest.yaml").write_text("commands: {}\n", encoding="utf-8")
    run_dir.mkdir()

    with pytest.raises(RuntimeError, match="overlap"):
        RI.seed_submission(ws, source, run_dir)

    assert (source / "manifest.yaml").is_file(), "a refused seed must not delete its source"


def test_seed_option_is_fresh_run_only_and_wired_before_launch():
    loop = _loop()
    import ast
    import inspect

    from merlin_experiments.phase1 import session

    body = (
        inspect.getsource(loop.main) + inspect.getsource(session.validate_options) + inspect.getsource(session.prepare)
    )
    from merlin_experiments.phase1 import authoring

    body += inspect.getsource(authoring.execute)
    from merlin_experiments.phase1.options import parse_options

    assert parse_options(["--run-id", "probe", "--seed-submission", "candidate"]).seed_submission == "candidate"
    assert "--seed-submission cannot be combined with --resume" in body
    assert "RI.seed_submission(ws, _seed_source_preflight, run_dir)" in body
    calls = [node for node in ast.walk(ast.parse(body)) if isinstance(node, ast.Call)]
    lines = {
        name: [node.lineno for node in calls if ast.unparse(node.func) == name]
        for name in ("RI.validate_seed_submission_source", "RI.seed_submission", "_launch")
    }
    lease_lines = [
        node.lineno
        for node in calls
        if isinstance(node.func, ast.Attribute)
        and node.func.attr == "acquire"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "_storage"
    ]
    assert len(lines["RI.validate_seed_submission_source"]) == len(lines["RI.seed_submission"]) == len(lease_lines) == 1
    assert lines["_launch"], "the ordering assertion must cover a real authoring call"
    assert lines["RI.validate_seed_submission_source"][0] < lease_lines[0] < lines["RI.seed_submission"][0]
    assert lines["RI.seed_submission"][0] < min(lines["_launch"])
    assert not any(
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "rmtree"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "ws_root"
        for node in calls
    ), "an existing workspace may contain the only preserved candidate and must never be silently deleted"
    assert parse_options(["--run-id", "probe", "--operator-errata", "note"]).operator_errata == "note"
    assert "--operator-errata cannot be combined with --resume" in body
    assert "RI.stage_operator_errata(run_dir, a.operator_errata)" in body


@pytest.mark.parametrize("unsafe_source", ["missing_manifest", "symlink"])
def test_seed_validation_refuses_before_replacing_existing_candidate(tmp_path, unsafe_source):
    ws, run_dir, source = tmp_path / "workspace", tmp_path / "run", tmp_path / "source"
    existing = ws / "submission"
    existing.mkdir(parents=True)
    preserved = existing / "only-candidate.txt"
    preserved.write_bytes(b"irreplaceable candidate")
    run_dir.mkdir()
    source.mkdir()
    if unsafe_source == "symlink":
        (source / "manifest.yaml").write_text("commands: {}\n")
        (source / "escape").symlink_to(tmp_path / "outside")

    with pytest.raises(RuntimeError, match="manifest.yaml|symlink"):
        RI.seed_submission(ws, source, run_dir)

    assert preserved.read_bytes() == b"irreplaceable candidate"
    assert list(existing.iterdir()) == [preserved]
    assert not (run_dir / "seed_submission.json").exists()


def test_operator_errata_is_archived_exactly_and_rejects_links(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    source = tmp_path / "ERRATA.source.md"
    source.write_text("# Corrected contract\n\nRe-derive the stride.\n", encoding="utf-8")

    record = RI.stage_operator_errata(run_dir, source)

    archived = run_dir / "ERRATA.md"
    assert archived.read_bytes() == source.read_bytes()
    assert record == {
        "source": str(source.resolve()),
        "n_bytes": len(source.read_bytes()),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    link = tmp_path / "linked.md"
    link.symlink_to(source)
    with pytest.raises(RuntimeError, match="symlink"):
        RI.stage_operator_errata(run_dir, link)


def test_operator_errata_provenance_detects_archived_content_drift(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    source = tmp_path / "ERRATA.source.md"
    source.write_text("# Original correction\n", encoding="utf-8")
    record = RI.stage_operator_errata(run_dir, source)

    RI.verify_operator_errata(record, run_dir)
    (run_dir / "ERRATA.md").write_text("# Different correction\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="operator errata drifted"):
        RI.verify_operator_errata(record, run_dir)
