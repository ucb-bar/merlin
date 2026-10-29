"""Loop/storage compiler input custody, using synthetic proof completions."""

import json
from pathlib import Path

import pytest

from merlin.verify import compiler_transforms as transforms
from merlin.verify.gate_errors import GateUnavailable


def job_at(tmp_path):
    job = dict(directory=str(tmp_path), params=dict(intermediates=["middle"], factor=2))
    for side, raw in (("before", "before\r\n"), ("after", "after\r\n")):
        path = tmp_path / f"{side}.mlir"
        path.write_bytes(raw.encode())
        job[side] = str(path)
    (tmp_path / "request.json").write_text(json.dumps(job))
    return job


@pytest.mark.parametrize("side", ["before", "after"])
def test_transient_reads_cannot_replace_referenced_buffers(tmp_path, monkeypatch, side):
    job = job_at(tmp_path)
    from merlin.verify import configuration_elision_bridge as bridge

    original_ref, original_read = bridge.file_ref, Path.read_bytes
    armed = False

    def arm(path):
        nonlocal armed
        ref = original_ref(path)
        if str(path) == job[side]:
            armed = True
        return ref

    def read(path):
        if armed and str(path) == job[side]:
            return b"different valid contents"
        return original_read(path)

    monkeypatch.setattr(bridge, "file_ref", arm)
    monkeypatch.setattr(Path, "read_bytes", read)
    with pytest.raises(GateUnavailable, match="independent read"):
        transforms._input_snapshot(job)


@pytest.mark.parametrize("mutation", ["before", "after", "middle", "factor", "request"])
def test_late_input_mutations_block_after_snapshot(tmp_path, mutation):
    job = job_at(tmp_path)
    snapshot = transforms._input_snapshot(job)
    if mutation in ("before", "after"):
        Path(job[mutation]).write_text("changed")
    elif mutation == "middle":
        job["params"]["intermediates"][0] = "changed"
    elif mutation == "factor":
        job["params"]["factor"] = 999
    else:
        (tmp_path / "request.json").write_text("{}")
    with pytest.raises(GateUnavailable, match="after proof"):
        transforms._recheck_inputs(snapshot, job)


def test_exact_raw_newlines_and_intermediates_are_retained(tmp_path):
    job = job_at(tmp_path)
    snapshot = transforms._input_snapshot(job)
    assert snapshot.buffers == ("before\r\n", "middle", "after\r\n")
    transforms._recheck_inputs(snapshot, job)
