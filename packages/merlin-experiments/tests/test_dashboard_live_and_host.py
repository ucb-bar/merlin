"""The dashboard's operator-side views and its ``--live`` mode stay read-only and say what is absent.

Host load samples and monitor notes are read as written (absent is "not recorded"); the live server
serves exactly one page on 127.0.0.1 with a meta refresh, and refuses to write inside a directory it
reads.
"""

from __future__ import annotations

import threading
import urllib.error
import urllib.request

import pytest
from merlin_experiments.spec import SpecError
from merlin_experiments.tracking import charts, live, records, resources

LOAD = (
    "utc\tload1\tcpu_busy_pct\tgsim\tspike\tverilator\tcodex\n"
    "2026-10-09T07:07:00Z\t3.10\t12\t0\t1\t0\t1\n"
    "2026-10-09T07:08:00Z\t9.50\t71\t4\t2\t1\t1\n"
    "not-a-time\t1\t1\t1\t1\t1\t1\n"
    "2026-10-09T07:09:00Z\t11.0\t88\t6\t0\t2\n"
    "2026-10-09T07:10:00Z\t12.0\t93\t8\t0\t2\t1\n"
)

MONITOR = """# monitor
## 2026-10-09T07:10:00Z
STATUS: OK
- 12/33 capsules pass; working on conv epilogue
## 2026-10-09T07:20:00Z
STATUS: WATCH
- same build error three times: "undefined reference to tile_matmul"
## 2026-10-09T07:30:00Z
STATUS: STUCK
- no new events for 18 min
"""


def test_load_samples_are_read_as_written_and_malformed_rows_counted(tmp_path):
    path = tmp_path / "load.tsv"
    path.write_text(LOAD, encoding="utf-8")
    inventory = records.Inventory()
    load = resources.load_samples(path, inventory)
    assert [r["cpu_busy_pct"] for r in load["rows"]] == [12.0, 71.0, 93.0]
    assert load["bad"] == 2 and load["rows"][-1]["gsim"] == 8.0
    page = resources.resource_section(load, now=load["rows"][-1]["at"], requested=True)
    assert "CPU busy" in page and "gsim processes" in page and "<svg" in page and "2 malformed" in page


def test_absent_load_and_monitor_files_are_not_recorded(tmp_path):
    inventory = records.Inventory()
    assert resources.load_samples(tmp_path / "nope.tsv", inventory) is None
    assert resources.monitor_notes(tmp_path / "nope.md", inventory) is None
    assert {r["state"] for r in inventory.rows} == {"absent"}
    assert "not recorded" in resources.resource_section(None, 0.0, requested=True)
    assert "not recorded" in resources.monitor_section(None, 0.0, requested=True)
    assert resources.resource_section(None, 0.0, requested=False) == ""


def test_monitor_notes_take_the_first_status_line_of_each_check(tmp_path):
    path = tmp_path / "MONITOR.md"
    path.write_text(MONITOR, encoding="utf-8")
    notes = resources.monitor_notes(path, records.Inventory())
    assert [n["status"] for n in notes["notes"]] == ["OK", "WATCH", "STUCK"]
    assert notes["counts"] == {"OK": 1, "WATCH": 1, "STUCK": 1}
    page = resources.monitor_section(notes, notes["notes"][-1]["at"], requested=True)
    assert "no new events for 18 min" in page and "critical" in page and "tile_matmul" in page


def test_gantt_marks_running_spans_and_the_now_line():
    lanes = [
        {"lane": "agent turns", "spans": [{"start": 0.0, "end": 600.0, "kind": "turn", "tip": "turn 1"}]},
        {"lane": "sim", "spans": [{"start": 300.0, "end": None, "kind": "sim", "tip": "job 7"}]},
    ]
    svg = charts.gantt(lanes, now=900.0, colours={"turn": "var(--c1)", "sim": "var(--c4)"}, label="g")
    assert 'stroke-dasharray="3 2"' in svg and "RUNNING" in svg and ">now<" in svg
    assert "not recorded" in charts.gantt([], now=0.0, colours={}, label="nothing")


def test_treemap_folds_past_eight_entities_into_other():
    groups = {f"family{i}": {"op": i + 1} for i in range(10)}
    svg = charts.treemap(groups, label="t")
    assert "2 more (grey)" in svg and svg.count("<rect") >= 20


def test_live_page_carries_a_meta_refresh_once():
    page = '<!doctype html><html><head><meta charset="utf-8"><title>x</title></head></html>'
    once = live.with_refresh(page, 60)
    assert once.count('http-equiv="refresh"') == 1 and 'content="60"' in once
    assert live.with_refresh(once, 30) == once


def test_live_refuses_an_output_inside_a_directory_it_reads(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(SpecError):
        live.check_output(run / "dash.html", [run])
    live.check_output(tmp_path / "elsewhere" / "dash.html", [run, None])


def test_live_serves_one_page_on_localhost_and_nothing_else(tmp_path):
    out = tmp_path / "dash" / "page.html"
    renders = []
    served = {}
    started = threading.Event()

    def render() -> str:
        renders.append(1)
        return f'<!doctype html><html><head><meta charset="utf-8"></head><body>render {len(renders)}</body></html>'

    def ready(server) -> None:
        served["port"] = server.server_address[1]
        started.set()

    def sleep(_seconds: float) -> None:
        # Between refreshes, fetch the page as a browser would, and try a path that is not the page.
        base = f"http://{live.HOST}:{served['port']}"
        served.setdefault("bodies", []).append(urllib.request.urlopen(base + "/", timeout=5).read().decode())
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(base + "/../../etc/passwd", timeout=5)
        served["other"] = err.value.code

    code = live.serve(render, out, interval=5, port=0, iterations=2, sleep=sleep, announce=lambda _: None, ready=ready)
    assert code == 0 and started.is_set() and len(renders) == 2
    assert "render 1" in served["bodies"][0] and 'http-equiv="refresh" content="5"' in served["bodies"][0]
    assert served["other"] == 404
    assert sorted(p.name for p in out.parent.iterdir()) == ["page.html"]  # only the page is written


def test_isolation_lowers_priority_and_pins_cpus_in_its_own_process():
    import json
    import os
    import subprocess
    import sys

    cpu = sorted(os.sched_getaffinity(0))[0]
    code = (
        "import json, os\n"
        "from merlin_experiments.tracking import isolate\n"
        f"applied = isolate('{cpu}')\n"
        "print(json.dumps({'nice': os.nice(0), 'cpus': sorted(os.sched_getaffinity(0)), 'io': applied['ionice']}))\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    result = json.loads(done.stdout.strip().splitlines()[-1])
    assert result["nice"] == 19 and result["cpus"] == [cpu]
    assert result["io"] in ("idle",) or result["io"].startswith(("unchanged", "unavailable"))
