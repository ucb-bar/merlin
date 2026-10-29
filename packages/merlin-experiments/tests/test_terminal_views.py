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


def test_capsule_cohorts_stay_below_queue_with_all_twenty_core_rows():
    sample = {'campaign': {'run_id': 'live'}, 'monotonic': 10,
              'cores': [{'cpu': cpu, 'measured_utilization_pct': 1} for cpu in range(9, 29)]}
    history = view.History(); history.add(sample)
    counts = {'source_sha256': 'a' * 64, 'source_current': True, 'evaluated_at': '2026-10-22T17:00:00Z',
              'cohorts': [{'cohort': name, 'total': total, 'all': 0, 'L2': 2, 'L3': 1, 'refusal': 0}
                          for name, total in [('functional', 153), ('calibration', 20), ('scalability', 77)]],
              'total': {'total': 250, 'all': 0, 'L2': 6, 'L3': 3, 'refusal': 0},
              'historical_union': {'L2': 40, 'L3': 30}}
    health = {'numeric': {'completed': 2}, 'requests': [{'source_sha256': 'b' * 64}]}
    text = view.render(sample, {'capsule_counts': counts, 'progress_health': health}, history,
                       width=160, height=36, color=False)
    assert text.index('CURRENT JOBS') < text.index('CAPSULES aaaaaaaaaaaa')
    for phrase in ('functional 153', 'calibration 20', 'scalability 77', 'total 250',
                   'all-required 0', 'L2/Spike', 'L3/gSIM', 'history union', 'not current',
                   'perf round not recorded', 'current unverified', 'N/A (not required)',
                   'diag 1; required ?'):
        assert phrase in text
    for cpu in range(9, 29):
        assert f' {cpu:>3} │' in text
    assert len(text.splitlines()) == 36
    assert all(view.visible_width(line) < 160 for line in text.splitlines())


def test_capsule_header_keeps_perf_at_120_columns_and_colors_only_labels():
    metadata = {'capsule_counts': {'source_sha256': 'a' * 64, 'evaluated_at': '2026-10-22T17:01:02.123+00:00', 'completed_perf_evaluations': 3}}
    plain = view._capsule_counts(metadata, False)
    colored = view._capsule_counts(metadata, True)
    assert 'perf evaluations 3' in view.clip_ansi(plain[0], 119)
    assert '17:01:02' in plain[0] and '2026-10-22' not in plain[0]
    assert '\x1b' not in ''.join(plain)
    for row, code in zip(colored[1:4], (view.COLORS['L2'], view.COLORS['calibration'], view.COLORS['agent'])):
        assert f'\x1b[{code}m' in row
