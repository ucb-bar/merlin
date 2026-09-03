"""A re-freeze re-verifies an already-certified run; it never manufactures the evidence it lacks.

The performance campaign's input gate demands an immutable bundle-input snapshot v2 record that the
functional harness only began emitting later, so an older but fully certified run cannot be consumed.
``refreeze_functional_run`` closes that gap by re-running the grade against the same submission bytes
under the current schema.  The whole value of that tool is what it REFUSES to do, so these tests pin
the refusals: an audit hit cannot be cleared, a non-passing source cannot be re-frozen, a cohort the
frozen corpus no longer holds cannot be reproduced, and the original score files are never carried
forward as if they were the new grade.  No simulator is launched here.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from merlin.common.paths import repo_root


def _load_refreeze():
    scripts = repo_root() / "merlin/experiments/gemmini_perf_bench/scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "_refreeze_functional_run", scripts / "refreeze_functional_run.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RF = _load_refreeze()


def _round(**over):
    row = {"round": 0, "all_pass": True, "answer_access_clean": True, "audit_hits": [],
           "conformance": {"conformant": True, "checks": {"no_regex_ok": True, "asm_used": None}}}
    row.update(over)
    return row


def _summary(**over):
    doc = {"rounds": [_round()], "converged": True,
           "finalize": {"answer_access_clean": True, "audit_hits": [], "regrade_all_pass": True}}
    doc.update(over)
    return doc


def _score(names, **over):
    doc = {
        "functional_pass": 1, "gradeable": True, "n_capsules": len(names), "n_passed": len(names),
        "per_capsule": [{"capsule": n, "status": "pass",
                         "tiers": {"L0": "pass", "L1": "pass", "L2": "pass", "L3": "pass"}}
                        for n in names],
    }
    doc.update(over)
    return doc


# ------------------------------------------------------------------ the schema migration is narrow
def test_migration_reads_the_two_added_keys_off_the_final_round():
    migrated, provenance = RF.migrate_qa_loop_summary(_summary())
    assert migrated["numeric_all_pass"] is True
    assert migrated["workflow_conformant"] is True
    # every migrated value must name where it came from
    assert "rounds[-1].all_pass" in provenance["numeric_all_pass"]
    assert "rounds[-1].conformance.conformant" in provenance["workflow_conformant"]


def test_migration_keeps_every_other_field_of_the_original_summary():
    source = _summary(wall_seconds=1633.9, n_rounds=1)
    migrated, _ = RF.migrate_qa_loop_summary(source)
    for key, value in source.items():
        assert migrated[key] == value


def test_migration_refuses_a_summary_that_disagrees_with_its_own_round():
    with pytest.raises(RF.RefreezeError):
        RF.migrate_qa_loop_summary(_summary(numeric_all_pass=False))


@pytest.mark.parametrize("summary", [
    _summary(rounds=[_round(audit_hits=[{"tool": "Bash", "kind": "path_read"}],
                            answer_access_clean=False)]),
    _summary(converged=False),
    _summary(rounds=[_round(all_pass=False)]),
    _summary(rounds=[_round(conformance={"conformant": False, "checks": {}})]),
    _summary(rounds=[_round(conformance={"conformant": True, "checks": {"isa_tools_used": False}})]),
    _summary(finalize={"answer_access_clean": True, "audit_hits": [], "regrade_all_pass": False}),
    _summary(rounds=[]),
])
def test_migration_refuses_every_unclean_source(summary):
    """A re-freeze must not launder the original authoring session's recorded failures."""
    with pytest.raises(RF.RefreezeError):
        RF.migrate_qa_loop_summary(summary)


# ---------------------------------------------------------- the cohort comes from the score itself
def test_cohort_is_the_capsule_identities_not_a_count():
    assert RF._capsule_names(_score(["B0", "A0"]), label="public") == ["A0", "B0"]


@pytest.mark.parametrize("score", [
    _score(["A0"], per_capsule=[{"capsule": "A0", "status": "fail",
                                 "tiers": {"L2": "pass", "L3": "pass"}}]),
    _score(["A0"], per_capsule=[{"capsule": "A0", "status": "pass", "tiers": {"L2": "pass"}}]),
    _score(["A0"], functional_pass=0),
    _score(["A0"], gradeable=False),
    _score(["A0"], n_capsules=2),
    _score(["A0", "A0"]),
    _score([]),
])
def test_cohort_refuses_a_source_that_was_not_already_certified(score):
    with pytest.raises(RF.RefreezeError):
        RF._capsule_names(score, label="public")


# ------------------------------------------------------- cohorts are reproduced from frozen bytes
def _corpus(root: Path, names, *, label="public"):
    for name in names:
        d = root / name
        d.mkdir(parents=True)
        (d / "capsule.yaml").write_text(yaml.safe_dump(
            {"name": name, "label": label, "required_oracle_tiers": ["L0", "L1", "L2", "L3"]}))
        (d / "golden.yaml").write_text("values: [1]\n")


def _snapshot(tmp_path: Path, public, hidden):
    repo = tmp_path / "repo_src"
    snapshot = tmp_path / "snap"
    corpus = repo / "caps" / "isa"
    _corpus(corpus, public)
    _corpus(repo / "caps" / "hidden", hidden, label="hidden")
    for rel in ("caps/isa", "caps/hidden"):
        dst = snapshot / "repo" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        import shutil as _sh
        _sh.copytree(repo / rel, dst)
    te = SimpleNamespace(capsule_corpus=corpus, corpus_siblings=lambda: [],
                         hidden_corpus=lambda: "caps/hidden/")
    return repo, snapshot, te


def test_public_cohort_is_materialized_from_the_snapshot_and_is_exactly_the_source_set(tmp_path):
    repo, snapshot, te = _snapshot(tmp_path, ["A0", "A1", "A2"], ["H0"])
    dest = tmp_path / "out"
    written = RF.stage_public_cohort(te, snapshot, repo, ["A0", "A2"], dest)
    assert written == ["A0", "A2"]
    assert sorted(p.name for p in dest.iterdir()) == ["A0", "A2"]
    # the materializer is the real one: it carries the capsule payload, not just the descriptor
    assert (dest / "A0" / "golden.yaml").is_file()
    # and it must NOT seal a descriptor admission boundary it is not reproducing
    assert not (dest / ".cohort_admission.json").exists()


def test_public_cohort_refuses_a_capsule_the_frozen_corpus_no_longer_holds(tmp_path):
    repo, snapshot, te = _snapshot(tmp_path, ["A0"], ["H0"])
    with pytest.raises(RF.RefreezeError):
        RF.stage_public_cohort(te, snapshot, repo, ["A0", "A9_deleted"], tmp_path / "out")


def test_hidden_cohort_copies_only_the_named_capsules(tmp_path):
    repo, snapshot, te = _snapshot(tmp_path, ["A0"], ["H0", "H1", "H2"])
    dest = tmp_path / "hid"
    assert RF.stage_hidden_cohort(te, snapshot, repo, ["H0", "H2"], dest) == ["H0", "H2"]
    assert sorted(p.name for p in dest.iterdir()) == ["H0", "H2"]


def test_hidden_cohort_refuses_a_capsule_the_frozen_corpus_no_longer_holds(tmp_path):
    repo, snapshot, te = _snapshot(tmp_path, ["A0"], ["H0"])
    with pytest.raises(RF.RefreezeError):
        RF.stage_hidden_cohort(te, snapshot, repo, ["H0", "H9_deleted"], tmp_path / "hid")


# ------------------------------------------------------------------- the old grade is not reused
def test_carried_provenance_never_includes_the_source_grade(tmp_path):
    """The original scores are provenance to POINT AT, never bytes to present as the new grade."""
    source = tmp_path / "src"
    (source / "grading_public").mkdir(parents=True)
    (source / "grading_hidden").mkdir(parents=True)
    (source / "grading_public" / "score_capsule.json").write_text("{}")
    (source / "grading_hidden" / "score_capsule.json").write_text("{}")
    (source / "rounds").mkdir()
    (source / "rounds" / "round_00.transcript.jsonl").write_text("{}\n")
    (source / "TASK.md").write_text("task\n")
    (source / "freeze.json").write_text("{}")
    (source / "run_manifest.yaml").write_text("run_id: x\n")
    run_dir = tmp_path / "new"
    run_dir.mkdir()
    carried = RF._copy_carried_provenance(source, run_dir)
    assert "TASK.md" in carried and "rounds/" in carried
    assert not (run_dir / "grading_public").exists()
    assert not (run_dir / "grading_hidden").exists()
    # freeze.json and run_manifest.yaml are the GRADER's outputs; a copy would be a stale claim
    assert not (run_dir / "freeze.json").exists()
    assert not (run_dir / "run_manifest.yaml").exists()


def test_the_record_declares_itself_a_refreeze(tmp_path):
    env = RF.build_environment(
        {"run_id": "old", "sandbox": "bwrap", "model": "m"}, new_run_id="new",
        snapshot_record={"version": 2, "content_sha256": "0" * 64, "n_files": 1, "n_bytes": 1},
        host_lane={"package": "p"},
        refreeze={"kind": "functional_refreeze", "of_run_id": "old",
                  "is_independent_result": False})
    assert env["run_id"] == "new"
    assert env["refreeze"]["of_run_id"] == "old"
    assert env["refreeze"]["is_independent_result"] is False
    assert env["bundle_input_snapshot"]["version"] == 2
    assert env["model_host_lane_snapshot"] == {"package": "p"}
    assert env["sandbox"] == "bwrap"       # the original treatment record is carried, not rewritten


def test_snapshot_record_must_be_v2(tmp_path, monkeypatch):
    """A snapshot the machinery reports at another version is a refusal, not something to relabel."""
    import merlin.targetgen.sandbox.bwrap as BW
    monkeypatch.setattr(BW, "materialize_bundle_inputs", lambda *a, **k: {})
    monkeypatch.setattr(BW, "verify_bundle_snapshot", lambda *a, **k: {})
    root = tmp_path / "bundle_inputs"
    root.mkdir()
    monkeypatch.setattr(BW, "bundle_snapshot_root", lambda ws: root)
    monkeypatch.setattr(BW, "snapshot_record", lambda ws: {"version": 1, "path": str(root)})
    with pytest.raises(RF.RefreezeError):
        RF.materialize_snapshot(tmp_path / "ws", {}, tmp_path)


# ------------------------------------------- grading writes bytecode back into the frozen submission
def _submission(tmp_path: Path) -> tuple[Path, str]:
    from merlin.benchharness import hash_tree
    sub = tmp_path / "submission" / "mlir_oot"
    sub.mkdir(parents=True)
    (sub / "tool.py").write_text("x = 1\n")
    (sub.parent / "manifest.yaml").write_text("name: pkg\n")
    return sub.parent, hash_tree(sub.parent)["sha256"]


def test_bytecode_written_by_grading_is_removed_and_the_digest_is_unchanged(tmp_path):
    """`grade_agent_run` imports the package, so CPython leaves __pycache__ in the frozen tree.

    hash_tree skips those names, so the digest never sees them -- which is exactly why the campaign
    gate refuses them. Every graded run would otherwise end up permanently gate-invalid.
    """
    submission, digest = _submission(tmp_path)
    cache = submission / "mlir_oot" / "__pycache__"
    cache.mkdir()
    (cache / "tool.cpython-313.pyc").write_bytes(b"\x00\x01")
    (submission / "mlir_oot" / "stray.pyc").write_bytes(b"\x00")
    removed = RF.purge_interpreter_bytecode(submission, digest)
    assert not cache.exists()
    assert not (submission / "mlir_oot" / "stray.pyc").exists()
    assert any("__pycache__" in r for r in removed) and any(r.endswith("stray.pyc") for r in removed)
    assert (submission / "mlir_oot" / "tool.py").is_file()   # hashed bytes are untouched


def test_purge_is_idempotent(tmp_path):
    submission, digest = _submission(tmp_path)
    assert RF.purge_interpreter_bytecode(submission, digest) == []
    assert RF.purge_interpreter_bytecode(submission, digest) == []


def test_purge_refuses_to_sweep_build_or_git_state(tmp_path):
    """`build/` and `.git/` mean something real is in the tree; deciding that for the submitter hides it."""
    submission, digest = _submission(tmp_path)
    (submission / "mlir_oot" / "build").mkdir()
    (submission / "mlir_oot" / "build" / "a.o").write_bytes(b"\x00")
    with pytest.raises(RF.RefreezeError):
        RF.purge_interpreter_bytecode(submission, digest)
    assert (submission / "mlir_oot" / "build" / "a.o").is_file()


def test_purge_refuses_when_the_graded_digest_would_change(tmp_path):
    submission, digest = _submission(tmp_path)
    (submission / "mlir_oot" / "extra.py").write_text("y = 2\n")   # a real, hashed addition
    with pytest.raises(RF.RefreezeError):
        RF.purge_interpreter_bytecode(submission, digest)
