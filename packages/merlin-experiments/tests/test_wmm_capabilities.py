"""Each machine's capabilities come from its own header and declared limits; a launch warns when the
chosen machine lacks one another registered machine has, and every result carries its machine's report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import wmm_fixtures as FX
import yaml
from merlin_experiments.phase2.whole_model_measured import capabilities as C
from merlin_experiments.phase2.whole_model_measured import jobs as J
from merlin_experiments.phase2.whole_model_measured import runs as RUNS
from merlin_experiments.phase2.whole_model_measured import service as S

from merlin.common.paths import repo_root

FULL = "#define DIM 16\n#define ACC_ROWS 1024\n#define FULL_READOUT\n#define MAX_BLOCK (DIM*4)\n"
LEAN = "#define DIM 16\n#define ACC_ROWS 512\n#define MAX_BLOCK (DIM*4)\n"
LIMIT = {"output_element_bytes": 4, "compare": "exact", "reason": "no full-width readout on this design"}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _registry(tmp_path: Path) -> Path:
    (tmp_path / "headers").mkdir()
    (tmp_path / "headers" / "full.h").write_text(FULL)
    (tmp_path / "lean.h").write_text(LEAN)  # NOT among the declared headers: only a section names it
    board = {"kind": "firesim", "hw_config": "x", "chipyard": "/nowhere", "workload": "w"}
    document = {
        "schema": "merlin.phase2.whole_model_measured.machines.v1",
        "target": "toy",
        "capability_headers": ["headers/full.h", {"env": "MERLIN_TEST_UNSET_CAPABILITY_HEADER", "join": "x.h"}],
        "machines": {
            "big_board": {**board, "program_header_sha256": _sha(FULL)},
            "small_board": {**board, "program_header_sha256": _sha(LEAN), "cannot_express": [LIMIT]},
            "batched_small": {"kind": "batched", "timing": "small_board", "local": "functional"},
            "functional": {"kind": "spike", "command": ["/bin/true"]},
        },
    }
    path = tmp_path / "machines.yaml"
    path.write_text(yaml.safe_dump(document))
    return path


def test_a_header_states_flags_values_and_expressions(tmp_path):
    (tmp_path / "h.h").write_text(FULL)
    doc = C.header_capabilities(tmp_path / "h.h")
    assert doc["flags"] == ["FULL_READOUT"] and doc["values"] == {"ACC_ROWS": 1024, "DIM": 16}
    assert doc["expressions"] == ["MAX_BLOCK"] and doc["status"] == "derived"


def test_the_lean_choice_warns_about_every_capability_the_other_board_has(tmp_path):
    registry = _registry(tmp_path)
    report = C.report(registry, "batched_small", header=tmp_path / "lean.h", environment={})
    assert report["chosen"]["machine"] == "small_board" and set(report["peers"]) == {"big_board"}
    lacks = {row["capability"]: row for row in report["lacks"]}
    assert lacks["FULL_READOUT"]["has"] == ["big_board"] and lacks["FULL_READOUT"]["basis"] == "header flag"
    limit = next(row for row in report["lacks"] if row["basis"] == "cannot_express")
    assert limit["has"] == ["big_board"] and "no full-width readout" in limit["why"]
    assert report["differs"] == [{"value": "ACC_ROWS", "mine": 512, "others": {"big_board": 1024}}]
    assert any("lacks FULL_READOUT, which big_board has" in w for w in report["warnings"])
    assert report["header_sources"]["unavailable"]  # the unset env reference is said, not dropped


def test_the_full_choice_lacks_nothing(tmp_path):
    """MUTATION of the above: the same registry from the other board's side raises no lack."""
    report = C.report(_registry(tmp_path), "big_board", environment={})
    assert report["chosen"]["header"]["status"] == "derived" and report["lacks"] == []
    # Its peer's header is not declared anywhere it can be found: unknown, said, and never a lack.
    assert report["unknown_peers"][0]["machine"] == "small_board"


def test_a_header_that_is_not_the_machines_is_a_conflict_and_a_warning(tmp_path):
    report = C.report(_registry(tmp_path), "small_board", header=tmp_path / "headers" / "full.h", environment={})
    assert report["chosen"]["header"]["status"] == "conflict"
    assert report["warnings"][0].startswith("small_board's capabilities are conflict")


def test_every_job_and_result_carries_its_machines_capabilities(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "spawn", lambda argv, **kw: type("P", (), {"pid": 999_999_999})())
    spec, pin = FX.write_builder(tmp_path)
    compact = C.compact(C.report(_registry(tmp_path), "big_board", environment={}))
    service = S.MeasurementService(
        tmp_path / "store",
        target="toy",
        builder=spec,
        builder_sha256=pin,
        machine=FX.spike_machine(tmp_path),
        machine_capabilities=compact,
    )
    job = service.request(FX.package(tmp_path, "p"))
    assert job["machine_capabilities"]["flags"] == ["FULL_READOUT"]
    assert J.result(job, timing_status="MEASURED")["machine_capabilities"]["machine"] == "big_board"


def test_prepare_records_the_report_and_its_warnings(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    registry = _registry(tmp_path)
    config = {
        "schema": "merlin_whole_model_objective_config_v1",
        "store": str(tmp_path / "store"),
        "builder": {"spec": FX.write_builder(tmp_path)[0], "sha256": FX.write_builder(tmp_path)[1]},
        "screen": {
            "machine": {"registry": str(registry), "name": "batched_small"},
            "build_options": {"header": str(tmp_path / "lean.h")},
        },
    }
    monkeypatch.setattr(RUNS.CFG, "store_roots", lambda document, environment=None: {"screen": tmp_path / "store"})
    prepared = RUNS.prepare(
        target="toy",
        method="m",
        objective_config=config,
        seed=FX.package(tmp_path, "seed"),
        prohibited_roles=[],
        why="test",
        run_factory=lambda **kw: tmp_path / "run",
    )
    record = json.loads((prepared.run_dir / C.RECORD).read_text())
    assert record["sections"]["screen"]["chosen"]["machine"] == "small_board"
    assert any(w.startswith("screen: small_board lacks FULL_READOUT") for w in prepared.machine_warnings)


@pytest.mark.target("gemmini")
def test_the_example_registry_warns_that_the_lean_board_cannot_express_a_full_width_readout():
    registry = repo_root() / "examples" / "gemmini" / "phase2" / "whole-model-machines.yaml"
    report = C.report(registry, "lean_u250_board", environment={})
    limits = [row for row in report["lacks"] if row["basis"] == "cannot_express"]
    assert limits and set(limits[0]["has"]) >= {"full_u250_board", "stock_u250_board"}
    full = C.report(registry, "full_u250_board", environment={})
    assert full["chosen"]["header"]["status"] == "derived" and full["lacks"] == []
