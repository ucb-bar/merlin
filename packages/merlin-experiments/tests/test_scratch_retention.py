"""Grade and self-check scratch keeps its evidence and loses its build trees; low disk stops admission."""
import json
import os
import threading

from merlin_experiments.phase1.feedback import dual_snapshots, scratch_retention


def _capsule(root, name, *, rtl=None):
    """A capsule run dir shaped like the grader's: a result plus a generated/ build tree."""
    capsule = root / "runs" / "suite" / name
    generated = capsule / "generated"
    (generated / "compiler-source" / ("a" * 64) / "compiler").mkdir(parents=True)
    (generated / "compiler-source" / ("a" * 64) / "compiler" / "lower.py").write_text("x = 1\n")
    (generated / "executed" / "spike" / ("b" * 64)).mkdir(parents=True)
    (generated / "executed" / "spike" / ("b" * 64) / "package_kernel.elf").write_bytes(b"\x7fELF")
    (generated / "harness.c").write_bytes(b"0," * (scratch_retention.KEEP_FILE_BYTES + 1))
    (generated / "kernel.o").write_bytes(b"obj")
    (generated / "numeric_report.yaml").write_text("status: pass\n")
    (generated / "invocations" / "1").mkdir(parents=True)
    (generated / "invocations" / "1" / "invocation.json").write_text("{}")
    tiers = {"L2": {"status": "pass", "engine": "spike", "derived_from_rtl": False}}
    if rtl is not None:
        tiers["L3"] = {"status": rtl, "engine": "gsim", "derived_from_rtl": True}
    result = {"capsule": name, "tiers": tiers, "command_buffer_artifact": {"command_buffer": {"W": [0] * 64}}}
    (capsule / "capsule_result.json").write_text(json.dumps(result, indent=2))
    return capsule


def test_slim_keeps_results_and_receipts_and_removes_build_products(tmp_path):
    capsule = _capsule(tmp_path, "loop_only")
    result_bytes = (capsule / "capsule_result.json").read_bytes()
    generated = capsule / "generated"
    before = os.stat(generated).st_mtime_ns

    freed = scratch_retention.slim_tree(tmp_path)

    assert freed > scratch_retention.KEEP_FILE_BYTES
    assert (capsule / "capsule_result.json").read_bytes() == result_bytes
    assert (generated / "numeric_report.yaml").is_file()
    assert (generated / "invocations/1/invocation.json").is_file()
    for gone in ("compiler-source", "executed", "harness.c", "kernel.o"):
        assert not (generated / gone).exists(), gone
    assert os.stat(generated).st_mtime_ns == before


def test_slim_leaves_every_capsule_with_rtl_tier_evidence_whole(tmp_path):
    passed, failed = _capsule(tmp_path, "certified", rtl="pass"), _capsule(tmp_path, "refuted", rtl="fail")
    unrun = _capsule(tmp_path, "unrun", rtl="unavailable")
    scratch_retention.slim_tree(tmp_path)
    for kept in (passed, failed):
        assert (kept / "generated/harness.c").is_file() and (kept / "generated/compiler-source").is_dir()
    assert not (unrun / "generated/harness.c").exists()


def test_large_result_is_judged_from_its_leading_tier_record(tmp_path, monkeypatch):
    capsule = _capsule(tmp_path, "large", rtl="pass")
    monkeypatch.setattr(scratch_retention, "_RESULT_BYTES_READ", 16)
    assert scratch_retention._paid_rtl_evidence(capsule)
    loop = _capsule(tmp_path, "large_loop")
    assert not scratch_retention._paid_rtl_evidence(loop)
    (loop / "capsule_result.json").write_text('{"capsule": "large_loop", "status": ')
    assert scratch_retention._paid_rtl_evidence(loop)  # unreadable means keep


def _owner(run, token, state):
    owners = run / "_qa_work/background_snapshot_owners"
    owners.mkdir(parents=True, exist_ok=True)
    runs = run / "_qa_work" / f"fruns_{token}"
    _capsule(runs, f"cap_{token}")
    (owners / f"{token}.json").write_text(json.dumps(
        {"token": str(token), "state": state, "runs_dir": str(runs)}))
    return runs


def test_prune_keeps_newest_published_and_unfinished_grades(tmp_path):
    run, ws = tmp_path / "run", tmp_path / "ws"
    (ws / "qa").mkdir(parents=True)
    grades = {token: _owner(run, token, "complete") for token in (10, 20, 30, 40, 50, 60)}
    grades[5] = _owner(run, 5, "draining")
    (ws / "qa/.snapshot-publication.json").write_text(json.dumps({"token": "20"}))

    assert scratch_retention.prune_superseded_grades(run, ws, keep_newest=3) > 0

    def slimmed(token):
        return not (grades[token] / "runs/suite" / f"cap_{token}" / "generated/harness.c").exists()

    assert [token for token in sorted(grades) if slimmed(token)] == [10, 30]
    for token in (10, 30):
        marker = json.loads((grades[token] / scratch_retention.MARKER).read_text())
        assert marker["schema"] == scratch_retention.SCHEMA
        assert (grades[token] / "runs/suite" / f"cap_{token}" / "capsule_result.json").is_file()
    assert scratch_retention.prune_superseded_grades(run, ws, keep_newest=3) == 0


def test_prune_ignores_an_owner_that_names_another_directory(tmp_path):
    run = tmp_path / "run"
    elsewhere = tmp_path / "elsewhere"
    _capsule(elsewhere, "cap")
    for token in (1, 2, 3, 4):
        _owner(run, token, "complete")
    owners = run / "_qa_work/background_snapshot_owners"
    (owners / "1.json").write_text(json.dumps({"token": "1", "state": "complete", "runs_dir": str(elsewhere)}))
    scratch_retention.prune_superseded_grades(run, keep_newest=3)
    assert (run / "_qa_work/fruns_1/runs/suite/cap_1/generated/harness.c").is_file()
    assert (elsewhere / "runs/suite/cap/generated/harness.c").is_file()


def test_free_space_floor_comes_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv(scratch_retention.MIN_FREE_ENV, "0")
    assert scratch_retention.free_space_ok(tmp_path)
    monkeypatch.setenv(scratch_retention.MIN_FREE_ENV, str(1 << 62))
    assert not scratch_retention.free_space_ok(tmp_path)
    monkeypatch.setenv(scratch_retention.MIN_FREE_ENV, "not a number")
    assert scratch_retention.min_free_bytes() == scratch_retention.DEFAULT_MIN_FREE_BYTES
    monkeypatch.setattr(scratch_retention, "free_bytes", lambda path: None)
    assert scratch_retention.free_space_ok(tmp_path, 1 << 62)  # an unknown volume never blocks


def test_low_space_prunes_first_and_refuses_only_when_still_low(tmp_path, monkeypatch):
    run = tmp_path / "run"
    for token in (1, 2, 3, 4):
        _owner(run, token, "complete")
    free = {"bytes": 0}
    monkeypatch.setattr(scratch_retention, "free_bytes", lambda path: free["bytes"])
    monkeypatch.setenv(scratch_retention.MIN_FREE_ENV, "100")
    assert not scratch_retention.admit_grade(run)
    assert (run / "_qa_work/fruns_1" / scratch_retention.MARKER).is_file()  # pruned before refusing
    free["bytes"] = 100
    assert scratch_retention.admit_grade(run)


def test_dual_snapshots_admits_no_grade_while_the_volume_is_low(tmp_path, monkeypatch):
    ws, run = tmp_path / "workspace", tmp_path / "run"
    (ws / "submission").mkdir(parents=True)
    (ws / "submission/manifest.yaml").write_text("{}")
    stop, checked, started = threading.Event(), threading.Event(), []

    def low(run_dir, workspace=None):
        checked.set()
        return False

    monkeypatch.setattr(scratch_retention, "admit_grade", low)
    manager = threading.Thread(target=lambda: dual_snapshots.run(
        ws, run, stop, interval_s=0, first_grade=lambda tick: started.append(tick) or {},
        grade=lambda tick: started.append(tick) or {}, source_identity=lambda: "a" * 64, poll_s=.002))
    manager.start()
    assert checked.wait(2)
    stop.set()
    manager.join(3)
    assert not manager.is_alive() and started == []


def test_dual_snapshots_survives_an_unwritable_owner_record(tmp_path, monkeypatch):
    ws, run = tmp_path / "workspace", tmp_path / "run"
    (ws / "submission").mkdir(parents=True)
    (ws / "submission/manifest.yaml").write_text("{}")
    stop, graded = threading.Event(), threading.Event()
    real = type(run).write_text

    claims = []

    def full(self, *args, **kwargs):
        if self.name.endswith(".next.json"):
            # Owner and readiness records always fail; the publication claim fails once.
            claim = self.parent.name == "qa"
            if not claim or True not in claims:
                claims.append(claim)
                raise OSError(28, "No space left on device")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(type(run), "write_text", full)
    monkeypatch.setattr(dual_snapshots, "LOW_SPACE_RETRY_S", 0)

    def grade(tick):
        graded.set()
        stop.set()
        return {}

    dual_snapshots.run(ws, run, stop, interval_s=0, first_grade=grade, grade=grade,
                       source_identity=lambda: "a" * 64, poll_s=.002)
    assert graded.is_set() and True in claims


def test_selfcheck_scope_removes_temporaries_and_slims_raw_runs(tmp_path):
    copy, raw = tmp_path / "selfcheck_iso_x", tmp_path / "selfchecks" / "selfcheck_y"
    (copy / "submission").mkdir(parents=True)
    capsule = _capsule(raw, "cap")
    outside = tmp_path / "outside"
    outside.mkdir()

    @scratch_retention.releases_selfcheck_scratch
    def check():
        scratch_retention.temporary(copy)
        scratch_retention.raw_runs(raw)
        return 7

    scratch_retention.temporary(outside)  # outside any check: no owner, nothing removed
    assert check() == 7
    assert not copy.exists() and outside.is_dir()
    assert (capsule / "capsule_result.json").is_file() and not (capsule / "generated/harness.c").exists()
    assert (raw / scratch_retention.MARKER).is_file()


def test_selfcheck_scope_releases_scratch_when_the_check_raises(tmp_path):
    copy = tmp_path / "selfcheck_caps_x"
    copy.mkdir()

    @scratch_retention.releases_selfcheck_scratch
    def check():
        scratch_retention.temporary(copy)
        raise RuntimeError("grade failed")

    try:
        check()
    except RuntimeError:
        pass
    assert not copy.exists()
