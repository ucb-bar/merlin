"""Pinned recorder admission, bounded binary streams and equivalent command decoding."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from merlin.perf import command_recorder as R
from merlin.perf.streamed_executed_trace import iter_recorded_commands, iter_source_commands

OPCODE = 0x0B
WORD = (2 << 25) | (11 << 20) | (10 << 15) | OPCODE
FENCE = 0x0FF0000F
RECORD = struct.Struct("<IIQQ")


@pytest.fixture
def selected(tmp_path, monkeypatch):
    engine, library = tmp_path / "engine", tmp_path / "recorder.so"
    engine.write_bytes(b"engine")
    engine.chmod(0o700)
    library.write_bytes(b"library")
    manifest = {
        "schema": R.SCHEMA, "byte_order": "little", "source_sha256": R._sha(R.SOURCE),
        "engine": {"path": str(engine), "sha256": R._sha(engine)},
        "library": {"path": str(library), "sha256": R._sha(library)},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    pin = {"path": str(path), "sha256": R._sha(path)}
    monkeypatch.setenv(R.SELECTION_ENV, json.dumps(pin))
    return engine, library, path, pin


def test_selection_reopens_manifest_library_and_source(selected, monkeypatch):
    engine, library, manifest, pin = selected
    assert R.selected()["pin"] == pin
    library.write_bytes(b"changed")
    assert R.selected() is None
    library.write_bytes(b"library")
    monkeypatch.setattr(R, "SOURCE", engine)
    assert R.selected() is None
    manifest.write_text("{}")
    assert R.selected() is None


@pytest.mark.parametrize("bad", ["{}", "[]", "null", "false", "1", '"text"', "malformed"])
def test_malformed_selection_falls_back(monkeypatch, bad):
    monkeypatch.setenv(R.SELECTION_ENV, bad)
    assert R.selected() is None


def test_exact_invocation_preserves_receiver_order_and_operands(selected, tmp_path):
    engine, library, *_ = selected
    original = [str(engine), "--extlib=receiver.so", "--extension=receiver", "-p1", "--log-commits", "p.elf"]
    command, env = R.invocation(original, {"KEEP": "yes"}, tmp_path / "out", R.selected())
    assert command == [
        str(engine), "--extlib=receiver.so", "--extension=receiver", f"--extlib={library}",
        f"--extension={R.EXTENSION}", "-p1", "p.elf",
    ]
    assert env["KEEP"] == "yes" and env[R.INNER_ENV] == "receiver"
    assert env[R.OUT_ENV] == str(tmp_path / "out")
    engine.write_bytes(b"changed engine")
    assert R.invocation(original, {}, tmp_path / "out", R.selected()) is None


@pytest.mark.parametrize("extra", [
    ["-l"], ["--log=trace"], ["--log", "trace"], ["-p2"], ["-p", "2"],
    ["--hartids=0,1"], ["--nprocs=2"], ["--extension=other"], ["-d"],
])
def test_unsupported_invocations_retain_commit_log(selected, tmp_path, extra):
    engine, *_ = selected
    argv = [str(engine), "--extension=receiver", "--log-commits", *extra, "p.elf"]
    assert R.invocation(argv, {}, tmp_path / "out", R.selected()) is None


def _stream(tmp_path, records=((WORD, 0, 10, 20), (FENCE, 0, 0, 0)), *, flags=0, retired=12, count=None):
    path = tmp_path / "stream"
    path.write_bytes(
        R._MAGIC + b"".join(RECORD.pack(*r) for r in records)
        + RECORD.pack(0, flags, retired, len(records) if count is None else count)
    )
    return path


@pytest.mark.parametrize("damage", ["flags", "hart", "zero", "count", "retired", "short", "long", "magic", "events"])
def test_incomplete_or_excessive_records_are_refused(tmp_path, damage):
    changes = {
        "flags": {"flags": 1}, "hart": {"records": ((WORD, 1, 10, 20),)},
        "zero": {"records": ((0, 0, 10, 20),)}, "count": {"count": 9},
        "retired": {"retired": 51}, "events": {"records": ((WORD, 0, 10, 20),) * 5},
    }
    path = _stream(tmp_path, **changes.get(damage, {}))
    if damage in ("short", "long", "magic"):
        data = path.read_bytes()
        path.write_bytes(data[:-1] if damage == "short" else data + b"x" if damage == "long" else b"X" + data[1:])
    assert R.read(path, max_retired=100, max_commands=4) is None


def test_recorded_commands_match_exact_register_replay_and_fence_order(tmp_path):
    path = _stream(tmp_path)

    class View:
        def splitlines(self):
            return iter([
                "core 0: 3 0x1000 (0x13) x10 0xa x11 0x14",
                f"core 0: 3 0x1004 (0x{WORD:x}) x10 0xff",
                f"core 0: 3 0x1008 (0x{FENCE:x})",
            ])

    recorded = list(iter_recorded_commands(R.read(path, max_retired=100), isa={"CUSTOM_OPCODE": OPCODE}))
    assert recorded == list(iter_source_commands(View(), isa={"CUSTOM_OPCODE": OPCODE}, raw=True))
    assert recorded == [(2, 10, 20), None]


def test_recorded_hazard_has_the_identical_verdict_as_the_commit_log(tmp_path):
    from merlin.common.paths import merlin_dir
    from merlin.targetgen import completion_order as C

    facts = json.loads((merlin_dir() / "tests/data/completion_order/gemmini_command_ordering.json").read_text())
    model = C.OrderingModel.from_facts(facts, row_masks={"scratchpad": 0x3FFF, "accumulator": 0x3FF})
    events = [None, (0, 0x100001, 0x40), (2, 0xFF << 48, (16 << 48) | (16 << 32)),
              (2, 1 << 48, (1 << 48) | (16 << 32) | 3), None]
    records, lines = [], []
    for at, event in enumerate(events):
        if event is None:
            records.append((FENCE, 0, 0, 0))
            lines.append(f"core 0: 3 0x{at * 12:x} (0x{FENCE:x})")
            continue
        funct, rs1, rs2 = event
        word = (funct << 25) | (11 << 20) | (10 << 15) | OPCODE
        records.append((word, 0, rs1, rs2))
        lines.extend([
            f"core 0: 3 0x{at * 12:x} (0x13) x10 0x{rs1:x} x11 0x{rs2:x}",
            f"core 0: 3 0x{at * 12 + 4:x} (0x{word:x})",
        ])

    class View:
        def splitlines(self):
            return iter(lines)

    recorded = C.check_commands(iter_recorded_commands(R.read(_stream(tmp_path, records=records), max_retired=100),
                                                      isa={"CUSTOM_OPCODE": OPCODE}), model)
    logged = C.check_commands(iter_source_commands(View(), isa={"CUSTOM_OPCODE": OPCODE}, raw=True), model)
    assert recorded == logged
    assert recorded["verdict"] == "hazard" and recorded["hazard_pairs"] == 1


def test_reader_bounds_before_loading_a_large_sparse_file(tmp_path):
    path = tmp_path / "huge"
    with path.open("wb") as handle:
        handle.write(R._MAGIC)
        handle.truncate(1 << 30)
    assert R.read(path, max_retired=100, max_commands=4) is None


def test_recorded_stream_retains_command_limit_and_no_command_refusal():
    with pytest.raises(ValueError, match="issued event limit"):
        list(iter_recorded_commands([(WORD, 1, 2), (FENCE, 0, 0)], isa={"CUSTOM_OPCODE": OPCODE}, max_commands=1))
    with pytest.raises(ValueError, match="no accelerator commands"):
        list(iter_recorded_commands([(FENCE, 0, 0)], isa={"CUSTOM_OPCODE": OPCODE}))


def test_failed_and_changed_observations_fall_back(selected, tmp_path, monkeypatch):
    from types import SimpleNamespace

    engine, library, *_ = selected
    argv = [str(engine), "--extension=receiver", "--log-commits", "p.elf"]

    def failed(*args, **kwargs):
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(R.subprocess, "run", failed)
    assert R.observe(argv, {}, scratch=tmp_path, timeout_s=1, max_retired=100) is None

    def changed(*args, **kwargs):
        path = _stream(tmp_path)
        path.rename(tmp_path / "commands.rec")
        library.write_bytes(b"changed during run")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(R.subprocess, "run", changed)
    assert R.observe(argv, {}, scratch=tmp_path, timeout_s=1, max_retired=100) is None
