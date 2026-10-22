"""Unified console preserves evidence labels and terminal boundaries."""

from merlin_experiments.tracking import terminal_views as view


def test_one_screen_has_all_cores_and_panels_without_fabricated_metrics():
    sample = {"campaign": {"run_id": "run"}, "monotonic": 10,
              "cores": [{"cpu": i, "measured_utilization_pct": 0} for i in range(9, 29)]}
    history = view.History()
    history.add(sample)
    text = view.render(sample, {}, history, width=150, height=36, color=False)
    for cpu in range(9, 29):
        assert f" {cpu:>3} │" in text
    for title in ("AGENT TREE", "PERFORMANCE", "OPTIMIZATION ATTEMPTS", "CURRENT JOBS"):
        assert title in text
    assert "No qualified cycles recorded yet" in text
    assert len(text.splitlines()) == 36
    assert all(view.visible_width(line) < 150 for line in text.splitlines())
    assert "\x1b" not in text


def test_history_cannot_mix_runs_and_record_text_cannot_control_terminal():
    history = view.History(10)
    history.add({"campaign": {"run_id": "old"}, "monotonic": 1, "cores": [{"cpu": 9}]})
    history.add({"campaign": {"run_id": "new"}, "monotonic": 2, "cores": [{"cpu": 10}]})
    assert all(cell is None for cell in history.cells(9, 20))
    sample = {"campaign": {"run_id": "bad\x1b[2J\nrecord"}}
    metadata = {"agents": [{"id": "long-agent-identity-" * 3, "status": "working"}]}
    text = view.render(sample, metadata, history, width=150, height=36, color=False)
    assert "\x1b" not in text
    assert "author long-age" in text
    assert view.number(None) == "—"
    assert view.number(float("nan")) == "—"


def test_color_clipping_does_not_wrap_or_truncate_escape_sequences():
    line = view.paint("abc界def", "38;5;45")
    clipped = view.clip_ansi(line, 5)
    assert view.visible_width(clipped) == 5
    assert clipped.endswith(view.RESET)
    assert "def" not in clipped


def test_watch_task_routes_to_unified_console_and_old_watch_keeps_default(monkeypatch, tmp_path):
    from merlin_experiments import cli, tracking
    from merlin_experiments.tracking import terminal
    calls = []
    monkeypatch.setattr(terminal, "main", lambda argv: calls.append(argv) or 0)
    assert cli.main(["watch", "--task", str(tmp_path), "--operator-private", "--once"]) == 0
    assert calls[0] == ["--task", str(tmp_path), "--window", "120", "--operator-private", "--once"]
    monkeypatch.setattr(tracking, "watch", lambda *a, **kw: calls.append(kw) or 0)
    assert cli.main(["watch", str(tmp_path), "--once"]) == 0
    assert calls[-1]["interval"] == 30.0


def test_unified_health_keeps_snapshot_scope_and_all_twenty_core_rows():
    sample = {'campaign': {'run_id': 'live'}, 'monotonic': 10,
              'cores': [{'cpu': cpu, 'measured_utilization_pct': 1} for cpu in range(9, 29)]}
    history = view.History()
    history.add(sample)
    health = {'numeric': {'completed': 2, 'passed': 1, 'failed': 1},
              'requests': [{'source_sha256': 'a' * 64, 'queue_wall_s': 3,
                            'started_to_completion_wall_s': 4}],
              'alerts': [{'code': 'REPEATED_PRE_EXECUTION_FAILURE', 'severity': 'warning'}]}
    text = view.render(sample, {'progress_health': health}, history, width=150, height=36, color=False)
    assert 'observed snapshot/capsule pairs' in text
    assert 'snapshot aaaaaaaaaaaa' in text
    assert 'queue 3.0s' in text and 'stage unknown' in text
    assert 'Repeated failure before numerical execution' in text
    for cpu in range(9, 29):
        assert f' {cpu:>3} │' in text
    assert len(text.splitlines()) == 36
    assert all(view.visible_width(line) < 150 for line in text.splitlines())


def test_compact_aggregate_evidence_is_visible_without_inventing_case_counts():
    from merlin_experiments.tracking.terminal_health import summary
    lines = summary({'numeric': {'completed': 0, 'passed': 0, 'aggregate_pass_evidence': 3}})
    assert 'aggregate execution passes 3' in lines[0]
    assert 'not additive' in lines[0]
