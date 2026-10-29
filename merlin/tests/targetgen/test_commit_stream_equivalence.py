"""The functional engine's command stream, read through each observation path, gives one record.

Runs only against real executables and an installed functional engine: set
``MERLIN_TEST_COMMIT_STREAM_TARGET`` to the target and ``MERLIN_TEST_COMMIT_STREAM_ELFS`` to
``os.pathsep``-separated executables (with the target's engine environment selected as a grade selects
it). Each executable is checked as the completion-order gate checks it, with and without the engine's
``-l`` execution lines; the commit lines must be byte-identical and the records equal.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

TARGET_ENV = "MERLIN_TEST_COMMIT_STREAM_TARGET"
ELFS_ENV = "MERLIN_TEST_COMMIT_STREAM_ELFS"
ELFS = [p for p in os.environ.get(ELFS_ENV, "").split(os.pathsep) if p]

pytestmark = pytest.mark.skipif(not ELFS or not os.environ.get(TARGET_ENV), reason=f"set {TARGET_ENV} and {ELFS_ENV}")


def _commit_lines_digest(argv, env):
    import subprocess

    digest, lines = hashlib.sha256(), 0
    with subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env) as process:
        for raw in process.stderr:
            parts = raw.split()
            if len(parts) >= 5 and parts[0] == b"core" and parts[2].isdigit():
                digest.update(raw)
                lines += 1
    assert process.returncode == 0
    return digest.hexdigest(), lines


def _without_wall(record):
    return {k: v for k, v in record.items() if k != "wall_s"}


@pytest.mark.parametrize("elf", ELFS)
def test_execution_lines_change_neither_commit_lines_nor_the_record(elf, tmp_path, monkeypatch):
    from merlin.perf.command_recorder import SELECTION_ENV
    from merlin.perf import executed_trace as ET
    from merlin.targetgen import completion_order_gate as G
    from merlin.targetgen.elf_lanes import PACKAGE_ELF_NAME

    target = os.environ[TARGET_ENV]
    monkeypatch.delenv(SELECTION_ENV, raising=False)
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / PACKAGE_ELF_NAME).write_bytes(Path(elf).read_bytes())
    monkeypatch.setenv(G.WORKER_ENV, "off")
    original = ET.trace_invocation
    argv, env = original(target, generated / PACKAGE_ELF_NAME)
    assert "-l" not in argv and argv.count("--log-commits") == 1

    def with_execution_lines(target, elf):
        argv, env = original(target, elf)
        at = argv.index("--log-commits")
        return [*argv[:at], "-l", *argv[at:]], env

    at = argv.index("--log-commits")
    assert _commit_lines_digest(argv, env) == _commit_lines_digest([*argv[:at], "-l", *argv[at:]], env)

    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out-commit"))
    commit = G._check(target, generated)
    monkeypatch.setattr(ET, "trace_invocation", with_execution_lines)
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out-execution"))
    execution = G._check(target, generated)
    assert commit["cache"] == execution["cache"] == "measured"
    assert "-l" in execution.pop("engine_argv") and "-l" not in commit.pop("engine_argv")
    assert _without_wall(commit) == _without_wall(execution)


@pytest.mark.parametrize("elf", ELFS)
def test_recorder_matches_every_command_operand_fence_and_verdict(elf, tmp_path):
    """Run the selected recorder and original log independently, then hash every ordered event."""
    import json
    import struct
    import subprocess
    import threading
    import time

    from merlin.perf import command_recorder as R
    from merlin.perf.executed_trace import trace_invocation
    from merlin.perf.streamed_executed_trace import DEFAULT_MAX_RETIRED, iter_recorded_commands, iter_source_commands
    from merlin.targetgen import completion_order as C
    from merlin.targetgen.rtl import command_ordering_selection as S
    from merlin.targetgen.rtl.facts import load_facts

    assert R.selected() is not None, f"select an operator-built recorder with {R.SELECTION_ENV}"
    target = os.environ[TARGET_ENV]
    document = load_facts(target, regenerate=False)
    body = document["facts"]
    if selected := os.environ.get(S.ENV):
        ordering, _identity = S.read(json.loads(selected), target=target, document=document)
        body = {**body, "command_ordering": ordering}
    model = C.OrderingModel.for_target(target, body=body)
    isa = C.trace_isa(target)
    argv, env = trace_invocation(target, Path(elf))
    timeout = float(os.environ.get("MERLIN_TEST_COMMIT_STREAM_TIMEOUT_S", "1200"))

    def digest_check(events):
        digest = hashlib.sha256()

        def hashed():
            for event in events:
                digest.update(struct.pack("<iQQ", -1, 0, 0) if event is None else struct.pack("<iQQ", *event))
                yield event

        verdict = C.check_commands(hashed(), model)
        return digest.hexdigest(), verdict

    start = time.perf_counter()
    observed = R.observe(argv, env, scratch=tmp_path, timeout_s=timeout, max_retired=DEFAULT_MAX_RETIRED)
    assert observed is not None, "this executable did not produce an admissible complete recorder stream"
    recorded = digest_check(iter_recorded_commands(observed[0], isa=isa))
    recorder_s = time.perf_counter() - start

    start = time.perf_counter()
    process = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env, text=True)
    timer = threading.Timer(timeout, process.kill)
    timer.start()

    class View:
        def splitlines(self):
            return process.stderr

    try:
        logged = digest_check(iter_source_commands(View(), isa=isa, raw=True))
    except BaseException:
        process.kill()
        raise
    finally:
        timer.cancel()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        process.stderr.close()
    assert process.returncode == 0
    log_s = time.perf_counter() - start
    assert recorded == logged
    print(json.dumps({
        "case": Path(elf).parent.name, "recorder_s": round(recorder_s, 3), "commit_log_s": round(log_s, 3),
        "event_sha256": recorded[0], "verdict": recorded[1]["verdict"],
        "commands": recorded[1]["commands"], "fences": recorded[1]["fences"],
    }), flush=True)


@pytest.mark.parametrize("elf", ELFS)
def test_real_recorder_command_limit_and_counter_access_refuse(elf, tmp_path):
    """Exercise refusal flags in the actual extension, including a read of the retirement counter."""
    import struct

    from merlin.perf import command_recorder as R
    from merlin.perf.executed_trace import trace_invocation
    from merlin.perf.streamed_executed_trace import DEFAULT_MAX_RETIRED
    from merlin.targetgen.elf_lanes import executable_sections, instruction_words

    manifest = R.selected()
    assert manifest is not None
    target = os.environ[TARGET_ENV]
    argv, env = trace_invocation(target, Path(elf))
    assert R.observe(argv, env, scratch=tmp_path, timeout_s=120, max_retired=DEFAULT_MAX_RETIRED, max_commands=1) is None
    trailer = R._RECORD.unpack_from((tmp_path / "commands.rec").read_bytes(), -R._RECORD.size)
    assert trailer[1] & 16, "the real recorder must flag its output limit"

    header = Path(manifest["include_root"]) / "riscv/encoding.h"
    constants = {}
    for line in header.read_text().splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == "#define" and parts[1] in {"MATCH_FENCE", "MASK_FENCE", "MATCH_CSRRS", "CSR_MINSTRET"}:
            constants[parts[1]] = int(parts[2], 0)
    assert len(constants) == 4
    original = bytearray(Path(elf).read_bytes())
    replaced = False
    for _name, offset, size, _address in executable_sections(original):
        for address, word in instruction_words(original[offset:offset + size], offset):
            if word & constants["MASK_FENCE"] == constants["MATCH_FENCE"]:
                # csrrs x0,minstret,x0 replaces one fence; the functional engine still completes.
                struct.pack_into("<I", original, address, constants["MATCH_CSRRS"] | constants["CSR_MINSTRET"] << 20)
                replaced = True
                break
        if replaced:
            break
    assert replaced
    changed = tmp_path / "counter-read.elf"
    changed.write_bytes(original)
    argv, env = trace_invocation(target, changed)
    assert R.observe(argv, env, scratch=tmp_path, timeout_s=120, max_retired=DEFAULT_MAX_RETIRED) is None
    trailer = R._RECORD.unpack_from((tmp_path / "commands.rec").read_bytes(), -R._RECORD.size)
    assert trailer[1] & 1, "the real recorder must flag retirement-counter access"


@pytest.mark.parametrize("elf", ELFS)
def test_real_gate_reuses_identical_bytes_in_a_fresh_build_directory(elf, tmp_path, monkeypatch):
    from merlin.perf import command_recorder as R
    from merlin.targetgen import completion_order_gate as G
    from merlin.targetgen.elf_lanes import PACKAGE_ELF_NAME

    assert R.selected() is not None
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    generated = [tmp_path / name for name in ("grade-one", "grade-two")]
    for directory in generated:
        directory.mkdir()
        (directory / PACKAGE_ELF_NAME).write_bytes(Path(elf).read_bytes())
    first = G._check(os.environ[TARGET_ENV], generated[0])
    second = G._check(os.environ[TARGET_ENV], generated[1])
    assert first["cache"] == "measured" and second["cache"] == "carried"
    assert first["command_stream"]["source"] == R.EXTENSION
    assert {k: v for k, v in first.items() if k != "cache"} == {k: v for k, v in second.items() if k != "cache"}
