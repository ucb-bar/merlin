"""Read-only status must not manufacture liveness, promotion or retained improvements."""
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

from merlin.common.paths import merlin_dir

spec = importlib.util.spec_from_file_location("phase2_status", merlin_dir() / "experiments/gemmini_perf_bench/scripts/render_phase2_status.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_launch_metadata_alone_never_means_live(tmp_path):
    write(tmp_path / "launch.json", {"objective": "host-model", "source_snapshot": "/frozen"})
    status = S.collect_status(tmp_path, process_reader=lambda *_: [])
    assert "terminal status unknown" in status["state"]
    assert not status["live_processes"]
    assert status["measured_evidence"]["full_model_speedup"] == "UNPROVEN"
    assert "No bound retained structural change" in S.render_markdown(status)


def test_terminal_receipt_and_failed_draft_do_not_replace_retained(tmp_path, monkeypatch):
    folder = tmp_path / "global_iterations"
    old = {"schema": "global_perf_iteration_v1", "iteration": 0, "candidate_sha256": "a" * 64, "analysis": {"seed": True}}
    final = {**old, "iteration": 1, "candidate_sha256": "b" * 64, "analysis": {"retained": True}, "context_receipts": [{"path": "short.json"}]}
    write(folder / "iteration_0000.json", old)
    digest = write(folder / "iteration_0001.json", final)
    write(folder / "iteration_0002.json", {**old, "iteration": 2, "candidate_sha256": "c" * 64})
    seal_path = folder / "round_0000_candidate.json"
    seal_digest = write(seal_path, {"candidate_sha256": "b" * 64, "iteration_record": str(folder / "iteration_0001.json"), "iteration_record_sha256": digest})
    sequence = {"status": "budget_complete", "checkpoints": [{"path": str(seal_path), "candidate_sha256": "b" * 64, "sha256": seal_digest}]}
    write(folder / "agent_sequence.json", sequence)
    from merlin.perf import structural_delta
    called = []
    monkeypatch.setattr(structural_delta, "compare_full_model_structure", lambda a, b: called.append((a, b)) or {"status": "compared"})
    status = S.collect_status(tmp_path, process_reader=lambda *_: [])
    assert status["state"] == "terminal_receipt:budget_complete"
    assert status["iterations"]["latest_candidate_sha256"] == "c" * 64
    assert status["retained_checkpoint"]["candidate_sha256"] == "b" * 64
    assert called == [(old["analysis"], final["analysis"])]
    assert status["measured_evidence"]["retained_controlled_context_receipts"] == [{"path": "short.json"}]
    final["candidate_sha256"] = "d" * 64
    write(folder / "iteration_0001.json", final)
    bad = S.collect_status(tmp_path, process_reader=lambda *_: [])
    assert not bad["retained_iteration_binding_verified"]
    assert bad["structural_seed_to_retained"]["status"] == "UNKNOWN"
    for expected_digest in (None, "f" * 64):
        sequence["checkpoints"][0]["sha256"] = expected_digest
        write(folder / "agent_sequence.json", sequence)
        bad = S.collect_status(tmp_path, process_reader=lambda *_: [])
        assert not bad["retained_iteration_binding_verified"]
        assert any("receipt digest" in warning["reason"] for warning in bad["warnings"])


def test_only_exact_snapshot_output_process_is_published(tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "uptime").write_text("100 0")
    run = tmp_path / "run"
    script = "/frozen/merlin/experiments/gemmini_perf_bench/scripts/launch_global_agent_experiment.py"
    for pid, args in ((1, ["python", script, "--output", str(run), "--private", "SECRET"]),
                      (2, ["python", script, "--output", "/other"]),
                      (3, ["python", "/unrelated", "--output", str(run)])):
        entry = proc / str(pid)
        entry.mkdir()
        (entry / "cmdline").write_bytes("\0".join(args).encode() + b"\0")
        (entry / "stat").write_text(f"{pid} (python with spaces) S " + " ".join(["0"] * 18 + ["10"]))
    rows = S.matching_processes(run, "/frozen", proc=proc)
    assert [row["pid"] for row in rows] == [1]
    assert "SECRET" not in json.dumps(rows)
    assert S.matching_processes(run, None, proc=proc) == []


@pytest.mark.parametrize('mode,expected', [('relative', True), ('absolute', True),
    ('fake_cwd', False), ('missing_cwd', False), ('inaccessible_cwd', False), ('zombie', False)])
def test_process_paths_use_own_cwd_without_publishing_arguments(tmp_path, monkeypatch, mode, expected):
    proc = tmp_path/'proc'
    entry = proc/'123'
    entry.mkdir(parents=True)
    (proc/'uptime').write_text('100 0')
    cwd = tmp_path/'workspace'
    cwd.mkdir()
    snapshot = cwd/'snapshot'
    script = Path('snapshot/merlin/experiments/gemmini_perf_bench/scripts/launch_global_agent_experiment.py')
    run = cwd/'run'
    args = ['python', str(cwd/script) if mode == 'absolute' else str(script),
            '--output', str(run) if mode == 'absolute' else 'run', '--private', 'SECRET']
    (entry/'cmdline').write_bytes(('\0'.join(args)+'\0').encode())
    if mode not in ('missing_cwd', 'absolute'):
        (entry/'cwd').symlink_to(cwd if mode != 'fake_cwd' else tmp_path)
    if mode == 'inaccessible_cwd':
        original_readlink = Path.readlink
        def readlink(path):
            if path == entry/'cwd':
                raise PermissionError('test inaccessible process cwd')
            return original_readlink(path)
        monkeypatch.setattr(Path, 'readlink', readlink)
    state = 'Z' if mode == 'zombie' else 'S'
    (entry/'stat').write_text(f'123 (python) {state} '+ ' '.join(['0']*18+['10']))
    result = S.matching_processes(run, str(snapshot), proc=proc)
    assert bool(result) is expected
    assert 'SECRET' not in json.dumps(result)


def test_incomplete_receipt_is_explicit_and_notes_are_separate(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "launch.json").write_text("{")
    notes = tmp_path / "notes.json"
    write(notes, {"next": "validate held-out application after freeze"})
    before = (run / "launch.json").read_bytes()
    status = S.collect_status(run, milestones=notes, process_reader=lambda *_: [])
    assert status["warnings"]
    assert status["milestones"]["next"].startswith("validate")
    assert (run / "launch.json").read_bytes() == before


def test_explicit_startup_snapshot_matches_process_then_must_match_launch(tmp_path):
    seen = []
    def process_reader(root, snapshot):
        seen.append(snapshot)
        return [{"pid": 42}]
    status = S.collect_status(tmp_path, source_snapshot=Path("/frozen"), process_reader=process_reader)
    assert status["state"] == "live_launcher_observed"
    assert status["source_snapshot_origin"] == "explicit_host_startup_path"
    assert seen == ["/frozen"]
    write(tmp_path / "launch.json", {"source_snapshot": "/different"})
    with pytest.raises(ValueError, match="differs"):
        S.collect_status(tmp_path, source_snapshot=Path("/frozen"), process_reader=process_reader)
    assert seen == ["/frozen"]


@pytest.mark.parametrize("terminal,live,expected_refreshes", [(False, False, 2), (True, False, 1), (True, True, 2)])
def test_watch_is_bounded_and_waits_for_terminal_plus_process_exit(terminal, live, expected_refreshes):
    now, calls = [0], []
    def refresh():
        calls.append(now[0])
        return {"terminal_receipt_present": terminal, "live_processes": [42] if live else []}
    S.refresh_loop(refresh, watch_seconds=60, interval_seconds=30,
                   monotonic=lambda: now[0], sleep=lambda seconds: now.__setitem__(0, now[0] + seconds))
    assert len(calls) == expected_refreshes
    assert now[0] <= 60
    with pytest.raises(ValueError):
        S.refresh_loop(refresh, watch_seconds=3601)


def test_dashboard_atomic_replace_and_one_shot(tmp_path, monkeypatch):
    status = S.collect_status(tmp_path / "run", process_reader=lambda *_: [])
    output = tmp_path / "dashboard"
    replaced = []
    original = S.os.replace
    def replace(source, target):
        assert Path(source).is_file()
        if str(target).endswith(".json"):
            assert json.loads(Path(source).read_text())["schema"] == "phase2_read_only_status_v1"
        replaced.append(Path(target).name)
        original(source, target)
    monkeypatch.setattr(S.os, "replace", replace)
    calls = []
    def refresh():
        calls.append(1)
        S.write_dashboard(output, status)
        return status
    S.refresh_loop(refresh)
    assert calls == [1]
    assert replaced == ["status.json", "STATUS.md"]
    assert sorted(path.name for path in output.iterdir()) == ["STATUS.md", "status.json"]


def test_initial_failure_is_blocked_not_inferred_terminal_and_cause_is_visible(tmp_path):
    worker = tmp_path / "host_analysis_workers/analysis_fixture/result.json"
    cause = "whole-model baseline lowering declined: no region is a mesh-admitted contraction"
    write(worker, {"failure": {"type": "StageGateError", "reason": cause}})
    write(tmp_path / "global_iterations/iteration_0000.json", {
        "schema": "global_perf_iteration_v1", "iteration": 0,
        "analysis": {"failure": {"type": "RuntimeError", "reason": "worker artifacts: " + str(worker.parent)}},
        "readiness": {"status": "blocked"}})
    status = S.collect_status(tmp_path, process_reader=lambda *_: [])
    assert status["state"] == "blocked_initial_analysis; launcher terminal receipt absent"
    assert not status["terminal_receipt_present"]
    assert status["latest_analysis_failure"]["reason"] == cause
    assert cause in S.render_markdown(status)
    write(tmp_path / "terminal_failure.json", {"schema": "global_launch_terminal_failure_v1", "reason": cause})
    status = S.collect_status(tmp_path, process_reader=lambda *_: [])
    assert status["state"] == "terminal_receipt:failed"
    assert status["terminal_receipt_present"]


@pytest.mark.parametrize("stage", ["configure", "sequence", "success"])
def test_launcher_preserves_initial_exception_in_terminal_receipt(tmp_path, stage):
    import sys
    sys.path.insert(0, str(merlin_dir() / "experiments/gemmini_perf_bench/scripts"))
    import launch_global_agent_experiment as launcher
    calls = []
    original = ValueError("initial baseline lowering declined")
    def configure():
        calls.append("configure")
        if stage == "configure":
            raise original
    def sequence():
        calls.append("sequence")
        if stage == "sequence":
            raise original
        return {"status": "budget_complete"}
    if stage == "success":
        assert launcher.run_authoring_with_terminal_receipt(tmp_path, configure=configure, sequence=sequence)["status"] == "budget_complete"
        assert not (tmp_path / "terminal_failure.json").exists()
    else:
        with pytest.raises(ValueError) as error:
            launcher.run_authoring_with_terminal_receipt(tmp_path, configure=configure, sequence=sequence)
        assert error.value is original
        receipt = json.loads((tmp_path / "terminal_failure.json").read_text())
        assert receipt["reason"] == str(original)
        assert receipt["completed_round_receipts"] == 0
        assert receipt["global_speedup_proven"] is False
        assert calls == (["configure"] if stage == "configure" else ["configure", "sequence"])
