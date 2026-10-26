"""The LIBRARY insight panel: what the author's compiler imports, what its tool calls mention, what A/Bs measured."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from merlin_experiments.tracking.console.io import Reader
from merlin_experiments.tracking.console.model import INSIGHTS, SOURCES, Context
from merlin_experiments.tracking.console.sources import agents, insight_common, library

NOW = time.time()
FIXTURE = json.loads(Path(__file__).with_name("library_pass_fresh4.json").read_text(encoding="utf-8"))
SCHED = "merlin.compile.scheduling."


def _write(path: Path, document) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document if isinstance(document, str) else json.dumps(document))
    return path


def _stamp(at: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(at)) + ".000Z"


def _rollout(path: Path, commands: list[str], edits: list[tuple[str, str]]) -> Path:
    rows = [
        {
            "timestamp": _stamp(NOW - 3000 + i),
            "type": "event_msg",
            "payload": {"type": "item_completed", "item": {"type": "CommandExecution", "command": ["bash", "-lc", c]}},
        }
        for i, c in enumerate(commands)
    ]
    rows += [
        {
            "timestamp": _stamp(NOW - 100),
            "type": "event_msg",
            "payload": {
                "type": "item_completed",
                "item": {"type": "FileChange", "changes": {target: {"type": "update", "unified_diff": diff}}},
            },
        }
        for target, diff in edits
    ]
    return _write(path, "\n".join(json.dumps(r) for r in rows) + "\n")


def _experiment(tmp_path: Path) -> dict:
    run = tmp_path / "out" / "runs" / "tgt" / "phase1" / "run"
    ws = tmp_path / "out" / "build" / "ws" / "run" / "workspace"
    (run / "qa_history").mkdir(parents=True)
    capabilities = {
        "capabilities": [
            {
                "name": "segmented_pipeline",
                "symbols": [SCHED + "dense_segmented_pipeline.schedule_dense_segmented_pipeline"],
            },
            {"name": "wide_loads", "symbols": [SCHED + "tile_load_coalescing.propose_tile_load_coalescing"]},
        ]
    }
    public = [
        SCHED + "compact_dense",
        SCHED + "tile_load_coalescing",
        "merlin.compile.capacity",
        SCHED + "rolling_window",
    ]
    _write(ws / "shared-library.json", {"public_modules": public, "scheduling_capabilities": capabilities})
    _write(ws / "OPTIMIZATION_SUGGESTIONS.json", FIXTURE["suggestions"])
    _write(ws / "submission" / "lowering.py", f"from {SCHED[:-1]} import tile_load_coalescing\n")
    _write(ws / "submission" / "dense.py", f"from {SCHED}compact_dense import schedule_compact_dense\n")
    _write(ws / "variants" / "seg" / "lowering.py", f"import {SCHED}dense_segmented_pipeline\n")
    _write(ws / "evidence" / "seg" / "report.json", {"note": f"used {SCHED}dense_segmented_pipeline"})
    # One variant self-check measured on gSIM against the submission's earlier cycles of the same case.
    channel = ws / ".qa_channel"
    _write(channel / "req_7_1.json", {"submission": "variants/seg", "capsules": "c1"})
    _write(
        channel / "resp_7_1.json",
        {
            "sim": "gsim",
            "graded_at_unix_ns": int((NOW - 50) * 1e9),
            "variant": {"name": "seg"},
            "per_capsule": [{"capsule": "c1", "pass": True, "barrier_tier": "L3", "barrier_cycles": 800}],
        },
    )
    _write(channel / "req_8_1.json", {"submission": "submission", "capsules": "c1"})
    _write(channel / "resp_8_1.json", {"per_capsule": []})
    ledger = {"schema": "merlin.phase1.perf_history.v1", "t": NOW - 500, "test_case": "c1", "kind": "measured"}
    _write(ws / "qa" / "perf_history.jsonl", json.dumps({**ledger, "cycles": 1000}) + "\n")
    return {"id": str(run), "run_id": "run", "run_dir": str(run), "workspace": str(ws), "state": "LIVE"}


def _ctx(experiment: dict, *, private: bool = False, state: dict | None = None) -> Context:
    return Context(
        experiment=experiment,
        reader=Reader(),
        state=state if state is not None else {},
        operator_private=private,
        now=NOW,
        live=True,
    )


def test_panel_shows_imports_mentions_underuse_and_ab_effects(tmp_path):
    experiment = _experiment(tmp_path)
    ws = experiment["workspace"]
    commands = ["sed -n 1,80p submission/compiler_library/merlin/compile/scheduling/compact_dense.py"] * 3
    commands += ["python3 -c 'import merlin.compile.capacity'", "ls"] + ["true"] * 50
    live = _rollout(
        tmp_path / "home" / "sessions" / "rollout-a.jsonl",
        commands,
        [(f"{ws}/variants/seg/lowering.py", f"+import {SCHED}dense_segmented_pipeline")],
    )
    snapshot = Path(experiment["run_dir"]) / "rounds" / "snap" / "rollout-a.jsonl"
    _write(snapshot, live.read_text())  # the run's snapshot copy of the same stream is read once
    ctx = _ctx(experiment, private=True)
    agents.index_for(ctx).files = [live, snapshot]
    section = library.read(ctx)
    assert {"title", "summary", "table", "series", "notes"} <= set(section) and section["title"] == "LIBRARY"
    rows = {r["key"]: r for r in section["rows"]}
    assert rows["capability:wide_loads"]["imported_by"] == 1  # ``from package import module``
    assert rows["capability:segmented_pipeline"]["imported_by"] == 0
    compact = rows[f"module:{SCHED}compact_dense"]
    assert compact["imported_by"] == 1 and compact["touches"] == 3 and compact["imported_files"] == ["dense.py"]
    assert rows["mechanism:capacity_tiles"]["touches"] == 1  # its module ``compile.capacity``: a leaf pair
    segmented = rows["capability:segmented_pipeline"]
    assert segmented["touches"] == 1 and segmented["evidence"] == 1  # the variant edit; the evidence file
    (effect,) = segmented["effects"]
    assert effect["variant"] == "variants/seg" and effect["kind"] == "measured" and abs(effect["ratio"] - 0.8) < 1e-9
    assert section["counts"]["variants"] == 1 and section["counts"]["attributed"] == 1
    summary = {line["label"]: line for line in section["summary"]}
    assert summary["library mentions / commands"]["value"].endswith("/ 55")  # one stream, not two copies
    assert summary["passes never touched"]["level"] == "warn"
    assert summary["best measured pass effect"]["value"] == "×0.80 via variants/seg"
    cells = {r["key"]: r["cells"] for r in section["table"]["rows"]}
    assert cells["capability:segmented_pipeline"][-1] == "×0.80 gSIM (1 A/B)"
    # The table leads with the least-touched passes.
    first = section["table"]["rows"][0]
    assert first["level"] == "warn" and first["cells"][3] == "0"
    detail = library.inspect(ctx, "capability:segmented_pipeline")
    lines = [line for part in detail["sections"] for line in part["lines"]]
    assert any("variants/seg: ×0.800 gSIM over 1 case(s) (c1)" in line for line in lines)
    public = library.read(_ctx(experiment))
    assert all(r["imported_files"] == [] for r in public["rows"])
    hidden = library.inspect(_ctx(experiment, state={"rows": {}}), "nope")
    assert hidden["notice"] == "no such pass"


def test_an_empty_workspace_renders_without_inventing_numbers(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    experiment = {"id": str(run), "run_id": "r", "run_dir": str(run), "workspace": str(tmp_path / "ws")}
    section = library.read(_ctx(experiment))
    assert section["rows"] == [] and section["counts"]["passes"] == 0
    assert all(line["value"] == "—" for line in section["summary"])


def test_library_is_a_registered_insight():
    assert "library" in INSIGHTS and ("library", "slow") in SOURCES


def test_needles_cut_paths_and_dotted_names_without_regex():
    rows = library.offered(
        {"scheduling_capabilities": {"capabilities": [{"name": "p", "symbols": [SCHED + "rolling_window.run"]}]}},
        None,
    )
    needles = library.needles_for(rows, ["merlin.xdsl_dialects.schedule"])
    assert needles.touched("cat merlin/compile/scheduling/rolling_window.py") == {"capability:p"}
    assert needles.touched("from merlin.xdsl_dialects.schedule import X") == {"module:merlin.xdsl_dialects.schedule"}
    assert needles.touched("schedule the work") == set()  # a bare common word is not the module
    assert insight_common.unique_streams([str(Path("/a/x.jsonl")), str(Path("/b/x.jsonl"))]) == ["/a/x.jsonl"]
    assert os.sep == "/"
