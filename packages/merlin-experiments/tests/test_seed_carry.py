"""A seed run's verdicts carry into a seeded run on another export; its evidence stages outside submission."""
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from merlin.common.provenance import source_digest
from merlin.common.tree_hash import hash_tree
from merlin.targetgen import tier_cache as TC
from merlin_experiments.phase1 import joint_control as J, joint_seed as S, seed_certificates as C
from merlin_experiments.phase1.options import parse_options
from merlin_experiments.phase1.run_inputs import graded_source_hash

JUDGE = ("src/merlin/targetgen/capsule_runner.py", "src/merlin/targetgen/tier_cache.py")
PINS = {"rtl": "a" * 40}


def _export(root: Path) -> list[Path]:
    files = []
    for name in JUDGE:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# judge {name}\n")
        files.append(path)
    return files


def _earned(tmp_path, old_files, elf, *, run_id="seedrun", tier="L3", engine="gsim"):
    """A verdict exactly as the seed run (absolute instrument naming) stored it."""
    rtl = tier == "L3"
    grading = source_digest([str(path) for path in old_files])
    instrument = C._instrument(engine if rtl else "", tier, grading, C.BASIS_ABSOLUTE)
    identity = TC.execution_identity(target="t", executables=[elf], toolchain_shas=PINS)
    result = {"status": "pass", "cycles": 5558, "derived_from_rtl": rtl, "engine": engine if rtl else "spike",
              "submission": {"package": f"/task/.tmp/{run_id}/operator/selfcheck/submission"}}
    store = tmp_path / "store"
    path = TC.record("case_a", tier, identity, instrument, status="pass", tier_result=result, root=store,
                     run_id="case_a")
    assert path is not None
    return store, identity


def _lookup_in_new_run(monkeypatch, new_root, new_files, identity, dest, *, tier="L3", engine="gsim"):
    """What the seeded run's grader asks: the same program on this export's portable judge."""
    monkeypatch.setenv(TC.PORTABLE_ENV, "1")
    monkeypatch.setattr("merlin.common.paths.checkout_root", lambda: new_root)
    monkeypatch.setattr(TC, "grading_path", lambda target=None: tuple(sorted(new_files)))
    monkeypatch.setattr(TC, "_engine_token", lambda target, t, rtl: engine if rtl else "")
    TC._INSTRUMENT_MEMO.clear()
    instrument = TC.instrument_digest("t", tier, rtl_tier=tier == "L3")
    return TC.lookup("case_a", tier, identity, instrument, root=dest, ledgers=False)


@pytest.fixture
def exports(tmp_path, monkeypatch):
    monkeypatch.delenv(TC.PORTABLE_ENV, raising=False)
    old_root, new_root = tmp_path / "frameworks/old", tmp_path / "frameworks/new"
    elf = tmp_path / "build/program.elf"
    elf.parent.mkdir(parents=True)
    elf.write_bytes(b"\x7fELF same program")
    yield old_root, _export(old_root), new_root, _export(new_root), elf
    TC._INSTRUMENT_MEMO.clear()


def test_identical_executable_reuses_the_seed_certificate_across_exports(tmp_path, monkeypatch, exports):
    old_root, old_files, new_root, new_files, elf = exports
    store, identity = _earned(tmp_path, old_files, elf)
    dest = tmp_path / "dest"
    report = C.rekey(store=store, seed_framework=old_root, seed_run_id="seedrun", target="t", dest=dest,
                     new_root=new_root, files=new_files)
    assert report["judge"]["identical"] and report["carried"] == 1 and report["by_tier"] == {"L3": 1}
    assert report["decision"].startswith("re-keyed")
    # Without the re-key the absolute-path judge of the old export never matches this run.
    assert _lookup_in_new_run(monkeypatch, new_root, new_files, identity, store) is None
    hit = _lookup_in_new_run(monkeypatch, new_root, new_files, identity, dest)
    assert hit is not None and hit["tier_result"]["cycles"] == 5558
    block = TC.carried_block(hit)
    assert block["source"] == "carried from seedrun" and block["earned_by_run"] == "seedrun"
    assert hit["carried_from"]["rekeyed"] is True and hit["carried_from"]["path_basis"] == C.BASIS_ABSOLUTE
    # The seeded grade's ordering index names the carried test case (never a pass by itself).
    from merlin.targetgen import seeded_grade

    assert seeded_grade.proven("seedrun", root=dest) == {"case_a"}
    assert seeded_grade.proven("another-run", root=dest) == frozenset()


def test_changed_executable_or_other_run_is_not_carried(tmp_path, monkeypatch, exports):
    old_root, old_files, new_root, new_files, elf = exports
    store, _ = _earned(tmp_path, old_files, elf)
    _earned(tmp_path, old_files, elf, run_id="another-run", tier="L2")
    dest = tmp_path / "dest"
    report = C.rekey(store=store, seed_framework=old_root, seed_run_id="seedrun", target="t", dest=dest,
                     new_root=new_root, files=new_files)
    assert report["carried"] == 1 and report["other_runs"] == 1
    elf.write_bytes(b"\x7fELF edited program")
    changed = TC.execution_identity(target="t", executables=[elf], toolchain_shas=PINS)
    assert _lookup_in_new_run(monkeypatch, new_root, new_files, changed, dest) is None


def test_changed_grading_code_reruns_and_says_which_files(tmp_path, exports):
    old_root, old_files, new_root, new_files, elf = exports
    store, _ = _earned(tmp_path, old_files, elf)
    new_files[0].write_text("JUDGE = 2  # a different judge\n")
    report = C.rekey(store=store, seed_framework=old_root, seed_run_id="seedrun", target="t",
                     dest=tmp_path / "dest", new_root=new_root, files=new_files)
    assert report["carried"] == 0 and report["rerun"] == 1 and report["decision"].startswith("re-run")
    assert [row["file"] for row in report["judge"]["changed"]] == [JUDGE[0]]
    assert not (tmp_path / "dest").exists() or not list((tmp_path / "dest").glob("*.json"))


def test_comment_only_or_reviewed_judge_changes_still_rekey(tmp_path, monkeypatch, exports):
    old_root, old_files, new_root, new_files, elf = exports
    store, identity = _earned(tmp_path, old_files, elf)
    new_files[0].write_text('"""A reworded docstring."""\n# judge, now with a comment\n')
    old_files[1].write_text("LIMIT = 1\n")
    _, identity = _earned(tmp_path, old_files, elf)  # the seed run's judge, as it was
    new_files[1].write_text("LIMIT = 2\n")
    report = C.rekey(store=store, seed_framework=old_root, seed_run_id="seedrun", target="t",
                     dest=tmp_path / "dest", new_root=new_root, files=new_files)
    assert report["carried"] == 0 and report["judge"]["equivalent_code"] == [JUDGE[0]]
    changed = report["judge"]["changed"]
    assert [row["file"] for row in changed] == [JUDGE[1]]
    reviewed = {JUDGE[1]: {**changed[0], "reason": "limit is not read for a stored pass"}}
    report = C.rekey(store=store, seed_framework=old_root, seed_run_id="seedrun", target="t",
                     dest=tmp_path / "dest", new_root=new_root, files=new_files, reviewed=reviewed)
    assert report["carried"] == 1 and report["decision"].startswith("re-keyed")
    assert report["judge"]["reviewed_equivalent"] == [{"file": JUDGE[1], "reason": reviewed[JUDGE[1]]["reason"]}]
    assert _lookup_in_new_run(monkeypatch, new_root, new_files, identity, tmp_path / "dest") is not None
    new_files[1].write_text("LIMIT = 3\n")  # bytes other than the reviewed ones
    report = C.rekey(store=store, seed_framework=old_root, seed_run_id="seedrun", target="t",
                     dest=tmp_path / "other", new_root=new_root, files=new_files, reviewed=reviewed)
    assert report["carried"] == 0


def test_portable_digest_matches_the_cache_for_its_own_checkout(tmp_path, monkeypatch, exports):
    _, _, new_root, new_files, _ = exports
    monkeypatch.setattr("merlin.common.paths.checkout_root", lambda: new_root)
    assert C.portable_digest(new_files, new_root) == TC._portable_source_digest(new_files)


def test_unreproducible_seed_judge_is_not_carried(tmp_path, exports):
    old_root, old_files, new_root, new_files, elf = exports
    store, _ = _earned(tmp_path, old_files[:1], elf)  # earned under a judge this export cannot name
    report = C.rekey(store=store, seed_framework=old_root, seed_run_id="seedrun", target="t",
                     dest=tmp_path / "dest", new_root=new_root, files=new_files)
    assert report["carried"] == 0 and report["judge_unreproducible"] == 1


def test_functional_results_follow_their_own_selection(tmp_path, exports):
    old_root, old_files, new_root, new_files, elf = exports
    store, _ = _earned(tmp_path, old_files, elf, tier="L2")
    report = C.rekey(store=store, seed_framework=old_root, seed_run_id="seedrun", target="t",
                     dest=tmp_path / "dest", new_root=new_root, files=new_files, results=False)
    assert report["carried"] == 0 and report["not_selected"] == 1


def _origin(tmp_path, source, **extra):
    review = {"schema": "operator.joint_seed_review.v1", "approved": True, "review_id": "r",
              "source": str(source), "sha256": graded_source_hash(source),
              "inherit_clock": False, "inherit_results": False, "variants": [], **extra}
    path = tmp_path / "review.json"
    path.write_text(json.dumps(review))
    origin = {k: v for k, v in review.items() if k not in ("schema", "approved")}
    origin.update(schema=S.SCHEMA, review={"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"version": 1, "duration_s": 21600, "final_model_s": 3600, "drain_s": 3600,
                                  "compiler_origin": origin}))
    return replace(parse_options(["--run-id", "new"]), joint_campaign=str(policy), seed_submission=str(source),
                   fresh_backend=False, sandbox="bwrap", schedule="continuous")


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    for name in (J.ENV, "MERLIN_JOINT_PERFORMANCE_SELECTION", TC.PORTABLE_ENV):
        monkeypatch.delenv(name, raising=False)
    source = tmp_path / "checkpoint"
    source.mkdir()
    (source / "manifest.yaml").write_text("version: 1\n")
    out = tmp_path / "evidence"
    (out / "SEED_NOTES").mkdir(parents=True)
    (out / "SEED_NOTES/REPORT.md").write_text("old harness narrative\n")
    (out / "OPTIMIZATION_RULES.json").write_text('{"rules": []}\n')
    return source, out


def test_seed_evidence_stages_at_the_workspace_root_outside_submission(tmp_path, evidence):
    source, out = evidence
    pin = {"path": str(out), "sha256": hash_tree(out)["sha256"]}
    options = _origin(tmp_path, source, **{S.EVIDENCE: pin})
    workspace, run = tmp_path / "ws", tmp_path / "run"
    (workspace / "submission").mkdir(parents=True)
    run.mkdir()
    result = S.stage(options, workspace, run)
    assert result["seed_evidence"]["staged"] == ["OPTIMIZATION_RULES.json", "SEED_NOTES"]
    assert (workspace / "SEED_NOTES/REPORT.md").read_text() == "old harness narrative\n"
    assert not (workspace / "submission/SEED_NOTES").exists()
    assert os.access(workspace / "OPTIMIZATION_RULES.json", os.W_OK)  # the author keeps maintaining it
    assert not os.access(workspace / "SEED_NOTES/REPORT.md", os.W_OK) or os.geteuid() == 0


def test_seed_evidence_must_match_review_and_digest(tmp_path, evidence):
    source, out = evidence
    pin = {"path": str(out), "sha256": hash_tree(out)["sha256"]}
    options = _origin(tmp_path, source, **{S.EVIDENCE: pin})
    (out / "OPTIMIZATION_RULES.json").write_text('{"rules": ["edited after review"]}\n')
    with pytest.raises(ValueError, match="seed evidence changed"):
        J.validate(options)
