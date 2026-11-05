"""The richer Phase 1 page: activity, timeline, compiler evolution, family rates, cost -- all read-only.

The fixture run carries a real ``oot/`` snapshot history (written through :mod:`merlin.common.oot_repo`),
so the compiler-evolution reader is exercised against git objects, not a mock.  A live view against a
read-only copy of the run must refresh without writing anything.
"""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path

import pytest
from dashboard_fixtures import HOUR, T0, phase1_run
from merlin_experiments.tracking import compiler, records, serve_dashboard, views, write_dashboard


@pytest.fixture()
def out_root(tmp_path, monkeypatch):
    root = tmp_path / "out"
    for sub in ("runs", "artifacts", "build"):
        (root / sub).mkdir(parents=True)
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(root))
    return root


def _snapshot(*roots: Path) -> dict[str, tuple[int, int]]:
    return {
        str(p): (p.stat().st_mtime_ns, p.stat().st_size)
        for root in roots
        for p in root.rglob("*")
        if p.is_file() and ".git" not in p.parts
    }


def test_the_run_page_tells_the_whole_story_from_records(out_root, tmp_path):
    run_dir = phase1_run(out_root, running=True)
    ws = out_root / "build" / "agent-workspaces" / "ws"
    before = _snapshot(run_dir, ws)
    result = write_dashboard(run_dir=run_dir, out=tmp_path / "p1.html", now=T0 + 3 * HOUR)
    page = Path(result["dashboard"]).read_text(encoding="utf-8")
    assert _snapshot(run_dir, ws) == before
    for expected in (
        "Agent activity",
        "Running now",
        "Working on the epilogue scale",
        "undefined reference to tile_mm",
        "Timeline: agent, grades, simulator jobs, freeze",
        "simjob: gsim",
        "simjob: verilator",
        "selfcheck: spike",
        ">now<",
        "Compiler evolution",
        "toy-fuse-epilogue",
        "toy-tile-schedule",
        "tile_matmul",
        "mvin",
        "+2 passes",
        "lower_interface_to_target",
        "tile-size",
        "Pass rate by capsule family",
        "elementwise_map",
        "Tokens and time",
        "40,097",
        "think/generate",
        "data as of",
    ):
        assert expected in page, expected
    assert "http://" not in page and "https://" not in page and "<link" not in page
    assert page.count("<script>") == 1


def test_compiler_evolution_diffs_successive_snapshots(out_root):
    run_dir = phase1_run(out_root)
    p1 = records.run_summary(run_dir, now=T0 + 5 * HOUR)["phase1"]
    evo = compiler.evolution(run_dir, p1["oot_commits"], records.Inventory(), {})
    first, second = (c["analysis"] for c in evo["checkpoints"])
    assert first["passes"] == ["toy-lower"] and first["dialects"] == ["toy"] and first["ops"] == ["tile_matmul"]
    assert second["passes"] == ["toy-fuse-epilogue", "toy-lower", "toy-tile-schedule"]
    delta = evo["checkpoints"][1]["diff"]
    assert delta["ops"]["added"] == ["mvin"] and delta["files_added"] == ["python/passes.py"]
    assert delta["loc_delta"] == second["loc_total"] - first["loc_total"] > 0


def test_missing_records_are_not_recorded(out_root, tmp_path):
    run_dir = out_root / "runs" / "toy" / "phase1" / "bare"
    (run_dir / "qa_history").mkdir(parents=True)
    (run_dir / "oot_commits.jsonl").write_text("", encoding="utf-8")
    page = Path(write_dashboard(run_dir=run_dir, out=tmp_path / "bare.html")["dashboard"]).read_text(encoding="utf-8")
    for expected in ("Agent event stream", "cost_time_toolcalls.yaml", "timing_detailed.json", "Snapshots"):
        assert expected in page, expected
    assert page.count("not recorded") >= 6


def test_live_refreshes_a_read_only_run_without_writing_and_reuses_unchanged_sections(out_root, tmp_path):
    source = phase1_run(out_root, running=True)
    frozen = tmp_path / "frozen-run"
    shutil.copytree(source, frozen, symlinks=True)
    for path in [frozen, *frozen.rglob("*")]:
        if not path.is_symlink():
            mode = path.stat().st_mode
            path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    before = _snapshot(frozen)
    out = tmp_path / "live" / "page.html"
    pages, ctx_seen = [], []
    real_context = views.Context

    class Spy(real_context):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            ctx_seen.append(self)

    views.Context = Spy
    try:
        code = serve_dashboard(
            out=out,
            interval=5,
            port=0,
            iterations=3,
            run_dir=frozen,
            sleep=lambda _s: pages.append(out.read_text(encoding="utf-8")),
            announce=lambda _m: None,
            isolation=False,
        )
    finally:
        views.Context = real_context
        for path in [frozen, *frozen.rglob("*")]:
            if not path.is_symlink():
                path.chmod(path.stat().st_mode | stat.S_IWUSR)
    assert code == 0 and len(pages) == 2
    assert _snapshot(frozen) == before
    assert 'http-equiv="refresh" content="5"' in pages[0] and "refreshed every 5 s (live)" in pages[0]
    assert ctx_seen[-1].reused > 0  # unchanged sections were not rebuilt
    assert sorted(os.listdir(out.parent)) == ["page.html"]


def test_live_refuses_to_write_into_the_run(out_root):
    from merlin_experiments.spec import SpecError

    run_dir = phase1_run(out_root)
    with pytest.raises(SpecError):
        serve_dashboard(
            out=run_dir / "dash.html", iterations=1, port=0, run_dir=run_dir, announce=lambda _m: None, isolation=False
        )


def test_watch_adds_what_the_agent_is_doing_now(out_root):
    import io

    from merlin_experiments.tracking import watch

    run_dir = phase1_run(out_root, running=True)
    buffer = io.StringIO()
    assert watch(run_dir, once=True, stream=buffer, now=T0 + 3 * HOUR) == 0
    text = buffer.getvalue()
    assert "AGENT  last event" in text and "running  selfcheck" in text and "epilogue scale" in text
