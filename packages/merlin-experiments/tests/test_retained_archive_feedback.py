"""Historical joins and actual process custody; synthetic bytes qualify no target."""

import hashlib
import io
import json
import sys
import tarfile
from copy import deepcopy

import pytest
from merlin_experiments.phase2 import _archive_feedback as H
from merlin_experiments.phase2 import _archive_feedback_context as X
from merlin_experiments.phase2.contracts import StageGateError, sha256_file
from merlin_timing import contracts as C

from merlin.common import invocation_record as I


@pytest.fixture
def joined():
    sha = C.digest
    caller, source, elf, scope, device = (sha(name) for name in ("caller", "source", "elf", "scope", "device"))
    case = {
        "case_id": "original",
        "source": {"source_sha256": source},
        "elf_sha256": elf,
        "input_sha256": sha({"schema": "merlin.embedded_input_context.v1", "harness_sha256": caller}),
        "scope_sha256": scope,
        "device_sha256": device,
    }
    rows = [
        {
            "record_id": sha(["record", i]),
            "label_receipt_sha256": sha(["receipt", i]),
            "scope_binding": {"begin_pc": 100, "end_pc": 200, "occurrence": 0, "evidence_sha256": sha(i)},
            "originally_measured": True,
            "original_numerical_status": "pass",
            "cycles": cycles,
            "native_observation_available": True,
            "caller_sha256": caller,
        }
        for i, cycles in enumerate((100, 130))
    ]
    catalog = {"schema": H.SCHEMA, "archive_sha256": sha("archive"), "cases": [{"case": case, "records": rows}]}
    compiled = {
        "status": "observed",
        "elf_sha256": elf,
        "candidate_diagnostics": {"artifacts": {"source": {"sha256": source}}},
    }
    report = {key: case[key] for key in ("elf_sha256", "scope_sha256", "device_sha256")}
    report.update(
        input_sha256=sha("current capsule closure"), scope_binding={"begin_pc": 100, "end_pc": 200, "occurrence": 0}
    )
    return catalog, compiled, report, caller


def test_same_source_elf_is_insufficient_without_actual_original_caller_bridge(joined):
    catalog, compiled, report, _caller = joined
    result = H.match(catalog, compiled=compiled, report=report, callers=())
    assert result["status"] == "UNKNOWN" and result["measurement"] is None


def test_all_original_repeats_and_distinct_input_identity_schemes_are_preserved(joined):
    catalog, compiled, report, caller = joined
    result = H.match(catalog, compiled=compiled, report=report, callers={caller})
    assert result["status"] == "retained_historical_measurement"
    assert result["worst_repeat_cycles"] == 130 and len(result["retained_repeats"]) == 2
    assert result["measurement"]["cycles"] == 130 and result["measurement"]["measured_now"] is False
    bridge = result["input_identity_bridge"]
    assert bridge["current_frozen_context_sha256"] == report["input_sha256"]
    assert bridge["original_embedded_context_sha256s"] != [report["input_sha256"]]
    assert result["promotion_authority"] == "NONE" and not result["fresh_candidate_measurement"]


@pytest.mark.parametrize("damage", ["source", "elf", "scope", "device", "pc", "caller", "input"])
def test_changed_source_program_caller_or_boundary_cannot_borrow_cycles(joined, damage):
    catalog, compiled, report, caller = joined
    if damage == "source":
        compiled["candidate_diagnostics"]["artifacts"]["source"]["sha256"] = C.digest("changed")
    elif damage == "elf":
        compiled["elf_sha256"] = C.digest("changed")
    elif damage in {"scope", "device"}:
        report[damage + "_sha256"] = C.digest("changed")
    elif damage == "pc":
        report["scope_binding"]["end_pc"] += 4
    elif damage == "caller":
        caller = C.digest("changed")
    else:
        catalog["cases"][0]["case"]["input_sha256"] = C.digest("changed")
    assert H.match(catalog, compiled=compiled, report=report, callers={caller})["measurement"] is None


@pytest.mark.parametrize(
    "damage",
    [
        {"scope_binding": None},
        {"native_observation_available": False},
        {"original_numerical_status": "fail"},
        {"originally_measured": False},
        {"cycles": 0},
        {"cycles": float("nan")},
    ],
)
def test_unavailable_or_failed_repeat_cannot_be_dropped_to_keep_faster_repeat(joined, damage):
    catalog, compiled, report, caller = joined
    catalog["cases"][0]["records"][1].update(damage)
    assert H.match(catalog, compiled=compiled, report=report, callers={caller})["measurement"] is None


def test_conflicting_raw_source_observers_do_not_inherit_capsule_identity(joined):
    _catalog, compiled, _report, _caller = joined
    compiled["artifact_join"] = {"source_sha256": C.digest("different original source")}
    assert H.source_identity(compiled) is None


def test_disagreeing_original_repeat_coordinates_cannot_choose_the_other_repeat(joined):
    catalog, compiled, report, caller = joined
    catalog["cases"][0]["records"][1]["scope_binding"]["end_pc"] += 4
    assert H.match(catalog, compiled=compiled, report=report, callers={caller})["measurement"] is None


def archive(tmp_path, *, name="run/source.mlir", raw=b"module {}", duplicate=False):
    path = tmp_path / "source.tar"
    with tarfile.open(path, "w") as target:
        for _ in range(2 if duplicate else 1):
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            target.addfile(info, io.BytesIO(raw))
    return path, {name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}}


def test_source_reader_reopens_indexed_bytes_without_extraction(tmp_path):
    path, wanted = archive(tmp_path)
    assert H._source_artifacts(path, wanted) == {"run/source.mlir": b"module {}"}
    assert not (tmp_path / "run").exists()
    wanted["run/source.mlir"]["sha256"] = C.digest("invented")
    with pytest.raises(StageGateError, match="source changed"):
        H._source_artifacts(path, wanted)


def test_duplicate_original_source_member_refuses(tmp_path):
    path, wanted = archive(tmp_path, duplicate=True)
    with pytest.raises(StageGateError, match="repeat"):
        H._source_artifacts(path, wanted)


@pytest.fixture
def produced(tmp_path):
    root = tmp_path / "original"
    root.mkdir()
    caller, obj, elf = (root / name for name in ("caller.c", "caller.o", "program.elf"))
    caller.write_text("/* synthetic caller, never a qualified hardware execution */")
    command = "import pathlib,sys;pathlib.Path(sys.argv[2]).write_bytes(pathlib.Path(sys.argv[1]).read_bytes())"
    for source, target, stage in ((caller, obj, "harness_object"), (obj, elf, "elf")):
        result = I.run(
            [sys.executable, "-c", command, str(source), str(target)],
            directory=root,
            stage=stage,
            inputs=(source,),
            outputs=(target,),
            capture_output=True,
            cwd=root,
            env={},
            timeout=10,
        )
        assert result.returncode == 0
    compiled = {"status": "observed", "elf_sha256": sha256_file(elf)}
    public = {"functional_status": "PASS", "compiled": compiled}
    original = {**public, "original_grade": {"tiers": {tier: {"status": "pass"} for tier in ("L0", "L1", "L2")}}}
    path = root / "ordinary_observation.json"
    path.write_text(json.dumps(original))
    value = {**public, "evidence_sha256": sha256_file(path)}
    return root, caller, compiled, value


def test_context_walks_actual_object_and_link_processes_to_consumed_caller(produced):
    root, caller, compiled, value = produced
    receipt = X.reopen(root, value=value, compiled=compiled, report=compiled, expected_callers={sha256_file(caller)})
    assert receipt["caller_sha256s"] == [sha256_file(caller)]
    assert len(receipt["producer_sha256s"]) == 2 and not receipt["qualification_issued"]


def test_changed_actual_caller_or_numerical_receipt_refuses(produced):
    root, caller, compiled, value = produced
    expected = {sha256_file(caller)}
    caller.write_text("changed after original compiler consumed it")
    with pytest.raises((StageGateError, ValueError, OSError)):
        X.reopen(root, value=value, compiled=compiled, report=compiled, expected_callers=expected)
    damaged = deepcopy(value)
    damaged["functional_status"] = "FAIL"
    with pytest.raises(StageGateError, match="receipt differs"):
        X.reopen(root, value=damaged, compiled=compiled, report=compiled, expected_callers=expected)
