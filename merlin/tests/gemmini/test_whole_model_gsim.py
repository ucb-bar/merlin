"""The GSIM whole-model runner: a memory dump is read strictly, and nothing short of a complete one passes.

The dump file, the reader over it, the console parser and the runner end to end are exercised here
without GSIM: a stand-in emulator (a short Python script with a receipt beside it) prints the console
and writes the dump the real harness would, so each refusal can be provoked on purpose -- a truncated
dump, a missing region, a missing group line -- and each mutation (one corrupted byte) must fail the
grade rather than pass it.
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
import selected_driver

from merlin.perf import whole_model_build as W
from merlin.perf import whole_model_gsim as G
from merlin.perf.layer_bench import reference as ref

if selected_driver.selected_is_generic("gemmini"):
    pytest.skip(
        "tests compiler/harness modules a gemmini support package ships itself; the selected "
        "generic data provider (merlin.runtime.backends.chipyard_rocc) does not ship them",
        allow_module_level=True,
    )

pytestmark = pytest.mark.target("gemmini")

_TARGET = "gemmini"


def _dump_bytes(regions, source=2):
    body = b"".join(struct.pack("<QQ", a, len(blob)) + blob for a, blob in regions)
    total = sum(len(blob) for _, blob in regions)
    return (
        b"GSIMDMP1"
        + struct.pack("<QQ", source, len(regions))
        + body
        + b"GSIMEND1"
        + struct.pack("<QQ", len(regions), total)
    )


def _digest(values):
    return ref.fnv1a64_words(np.asarray(values, dtype=np.int64).astype("<i8").tobytes()) & ref.DIGEST_MASK


# ---------------------------------------------------------------------------------- the dump file


def test_a_complete_dump_round_trips(tmp_path):
    path = tmp_path / "d.dump"
    path.write_bytes(_dump_bytes([(0x1000, b"\x01\x02\x03\x04"), (0x2000, b"\xff" * 8)]))
    parsed = G.read_dump(path, [(0x1000, 4), (0x2000, 8)])
    assert parsed["source"] == "hybrid"
    assert parsed["regions"][(0x1000, 4)] == b"\x01\x02\x03\x04"


@pytest.mark.parametrize("cut", [1, 9, 30, 40])
def test_a_truncated_dump_is_refused(tmp_path, cut):
    """A dump cut anywhere -- inside the trailer, a region's bytes, a region header -- is not a dump."""
    whole = _dump_bytes([(0x1000, bytes(16)), (0x2000, bytes(16))])
    path = tmp_path / "d.dump"
    path.write_bytes(whole[: len(whole) - cut])
    with pytest.raises(G.DumpError):
        G.read_dump(path, [(0x1000, 16), (0x2000, 16)])


def test_a_dump_missing_a_region_is_refused(tmp_path):
    path = tmp_path / "d.dump"
    path.write_bytes(_dump_bytes([(0x1000, bytes(16))]))
    with pytest.raises(G.DumpError, match="1 missing"):
        G.read_dump(path, [(0x1000, 16), (0x2000, 16)])


def test_a_trailer_that_disagrees_with_the_body_is_refused(tmp_path):
    whole = bytearray(_dump_bytes([(0x1000, bytes(16))]))
    whole[-8:] = struct.pack("<Q", 17)  # the trailer claims one byte more than the body holds
    path = tmp_path / "d.dump"
    path.write_bytes(bytes(whole))
    with pytest.raises(G.DumpError, match="trailer"):
        G.read_dump(path)


def test_a_read_outside_the_dump_raises_and_is_never_zero():
    read, _ = G._reader({(0x1000, 16): bytes(range(16))})
    assert read(0x1004, 4) == bytes([4, 5, 6, 7])
    with pytest.raises(G.DumpError):
        read(0x1010, 1)
    with pytest.raises(G.DumpError):
        read(0x100C, 8)  # straddles the end of the region


def test_every_buffer_record_is_dumped_wherever_the_map_puts_it():
    """A later map version may add inputs for local grading under any field; each is still dumped,
    and one lying wholly in a read-only ELF segment is left to the ELF."""
    layout = {
        "groups": [
            {
                "group": 1,
                "address": 0x9000,
                "bytes": 4,
                "inputs": [{"address": 0x8000, "bytes": 8}, {"address": 0x100, "bytes": 16}],
                "lhs": {"address": 0x9000, "bytes": 4},
            }
        ]
    }
    assert G.dump_regions(layout) == [(0x9000, 4), (0x8000, 8), (0x100, 16)]
    assert G.dump_regions(layout, constant=[(0x0, 0x1000, 0)]) == [(0x9000, 4), (0x8000, 8)]


# ------------------------------------------------------------------------------------ the console


def _protocol():
    from merlin.runtime.backends import base as backends

    # The console spellings are the selected support provider's driver: none selected, nothing to read.
    selected_driver.require_support(_TARGET)

    return dict(backends.whole_model_driver(_TARGET).program.UART)


def _console(protocol, groups, *, window=1000, argmax=3, want=3):
    lines = [
        protocol["group"].format(group=g, kind=k, cycles=c, sum="UNKNOWN", checksum="UNKNOWN") for g, k, c in groups
    ]
    lines.append(protocol["full_model"].format(cycles=window))
    lines.append(protocol["argmax"].format(got=argmax, want=want, agrees=int(argmax == want)))
    lines.append(protocol["metric"].format(cycles=window - 10))
    return "\n".join(lines) + "\n"


def test_the_console_is_read_with_the_targets_own_spelling():
    protocol = _protocol()
    parsed = G.parse_console(_console(protocol, [(1, "conv2d", 50), (2, "sum", 7)]), protocol)
    assert parsed["groups"] == [
        {"group": 1, "kind": "conv2d", "cycles": 50},
        {"group": 2, "kind": "sum", "cycles": 7},
    ]
    assert (parsed["full_model"], parsed["metric"], parsed["argmax"]["got"]) == (1000, 990, 3)
    assert G.parse_console("GM_GROUP one conv2d x sum=UNKNOWN fnv1a=UNKNOWN\n", protocol)["groups"] == []


# -------------------------------------------------------------------------- the runner, end to end

_FAKE = '''#!{python}
"""A stand-in for the dump-capable emulator: prints a console, writes the dump from a memory image."""
import json, os, struct, sys
scenario = json.load(open(os.environ["FAKE_GSIM_SCENARIO"]))
args = dict(a[1:].split("=", 1) for a in sys.argv[2:] if a.startswith("+") and "=" in a)
sys.stdout.write(scenario["console"])
memory = bytes.fromhex(scenario["memory_hex"])
regions = [(int(a, 0), int(n)) for a, n in (l.split() for l in open(args["dump-regions"]) if l.strip())]
blobs = [memory[a - scenario["base"] : a - scenario["base"] + n] for a, n in regions]
body = b"".join(struct.pack("<QQ", a, n) + b for (a, n), b in zip(regions, blobs))
head = b"GSIMDMP1" + struct.pack("<QQ", 2, len(regions))
tail = b"GSIMEND1" + struct.pack("<QQ", len(regions), sum(n for _, n in regions))
data = head + body + tail
data = data[: len(data) - scenario.get("truncate", 0)]
for _ in range(scenario.get("flood_lines", 0)):
    sys.stderr.write("[gsim-ar] #0 addr=0x80000000 " + "x" * 64 + "\\n")
if scenario.get("assertion"):
    import time
    sys.stderr.write("Assertion failed: " + scenario["assertion"] + "\\n    at DMACommandTracker.scala:88 assert(v)\\n")
    sys.stderr.flush()
    time.sleep(60)
if not scenario.get("no_dump"):
    open(args["dump-out"], "wb").write(data)
sys.stderr.write("[gsim-emu] FINISHED: cycles=5000 wall=0.1s (1 cyc/s) done=1 exit_code=0\\n")
sys.exit(scenario.get("exit", 0))
'''


def _minimal_elf(path: Path) -> Path:
    """An ELF64 little-endian header with no program headers: no read-only segment to serve from."""
    header = bytearray(64)
    header[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<HH", header, 0x36, 56, 0)
    path.write_bytes(bytes(header) + b"program")
    return path


def _place(address, symbol="B"):
    return {"symbol": symbol, "address": address, "bytes": 4, "elements": 4, "element_bytes": 1}


class _Local:
    """A stand-in LocalReference: each group's reference given the inputs the run handed back.

    Group 2 reads group 1's output; its reference is a function of those DUMPED values, so a wrong
    group 1 does not make group 2 wrong -- the property local grading exists for.
    """

    def __init__(self, conv, logits):
        self.conv, self.logits = np.asarray(conv, dtype=np.int64), np.asarray(logits, dtype=np.int64)

    def expected(self, group, inputs):
        if group == 1:
            return self.conv
        return self.logits + 0 * np.asarray(inputs["B_g1"])  # raises KeyError if not handed back


@pytest.fixture
def rig(tmp_path, monkeypatch):
    """A 2-group map (an exact conv and its logits), its oracle, an ELF and a receipted stand-in emulator."""
    base = 0x80001000
    conv = np.array([1, -2, 3, 4], dtype=np.int8)
    logits = np.array([0, 5, 9, 2], dtype=np.int8)
    memory = bytearray(64)
    memory[0:4], memory[16:20] = conv.tobytes(), logits.tobytes()
    elf = _minimal_elf(tmp_path / "program.elf")
    layout = {
        "schema": W.MEMORY_MAP_SCHEMA,
        "elf_sha256": hashlib.sha256(elf.read_bytes()).hexdigest(),
        "groups": [
            {"group": 1, "kind": "conv2d", "compare": "exact", "inputs": [], **_place(base, "B_g1")},
            {
                "group": 2,
                "kind": "matmul",
                "compare": "exact",
                "inputs": [{**_place(base, "B_g1"), "produced_by": 1}],
                **_place(base + 16, "B_g2"),
            },
        ],
    }
    local = _Local(conv, logits)
    oracle = {
        "groups": {"1": {"fnv1a": _digest(conv)}, "2": {"fnv1a": _digest(logits)}},
        "argmax": 2,
        "golden_argmax": 2,
    }
    engine = tmp_path / "engine"
    engine.mkdir()
    emulator = engine / "emulator"
    emulator.write_text(_FAKE.replace("{python}", sys.executable), encoding="utf-8")
    emulator.chmod(0o755)
    (engine / "build_receipt.json").write_text(
        json.dumps({"binary_sha256": hashlib.sha256(emulator.read_bytes()).hexdigest(), "firrtl_sha256": "f" * 64})
    )
    protocol = _protocol()

    def run(*, memory_edit=None, console_groups=None, **scenario):
        image = bytearray(memory)
        for offset, value in (memory_edit or {}).items():
            image[offset] = value
        groups = console_groups if console_groups is not None else [(1, "conv2d", 40), (2, "matmul", 60)]
        spec = {"console": _console(protocol, groups, argmax=2, want=2), "memory_hex": image.hex(), "base": base}
        (tmp_path / "scenario.json").write_text(json.dumps({**spec, **scenario}))
        monkeypatch.setenv("FAKE_GSIM_SCENARIO", str(tmp_path / "scenario.json"))
        return G.run_gsim_whole_model(
            elf, layout, oracle, target=_TARGET, out=tmp_path / "run", emulator=emulator, local=run.local
        )

    run.local, run.layout, run.oracle = local, layout, oracle
    return run


def test_a_clean_run_is_graded_and_quotable(rig):
    verdict = rig()
    assert verdict["status"] == "graded" and verdict["quotable"]
    assert verdict["whole_window_cycles"] == 1000
    assert [g["cycles"] for g in verdict["per_group"]] == [40, 60]
    assert all(g["correct"] for g in verdict["per_group"])
    assert verdict["argmax"]["from_dump"] == 2 and verdict["argmax"]["agrees_with_oracle"]
    assert verdict["cycles_adjudication"]["state"]  # the ladder's word travels with the cycles


def test_one_corrupted_byte_fails_the_grade(rig):
    """MUTATION: a single byte of one group's output, changed in memory, must make THAT group wrong --
    and only that one: its consumer is graded on the value it actually read, so it stays correct."""
    verdict = rig(memory_edit={2: 0x7F})
    assert verdict["status"] == "graded" and not verdict["quotable"]
    assert [g["correct"] for g in verdict["per_group"]] == [False, True]
    assert verdict["grade"]["gate"] == "local"


def test_without_a_local_reference_nothing_exact_is_agreed(rig):
    rig.local = None
    verdict = rig()
    assert verdict["status"] == "graded" and not verdict["quotable"]
    assert len(verdict["grade"]["unverified"]) == 2 and not any(g["correct"] for g in verdict["per_group"])


def test_an_existing_run_is_regraded_from_its_dump_without_simulating(rig, tmp_path):
    """regrade() reads the run's own files; a byte corrupted IN THE DUMP afterwards fails the regrade."""
    first = rig()
    run = tmp_path / "run"
    again = G.regrade(run, rig.layout, rig.oracle, target=_TARGET, local=rig.local)
    assert again["quotable"] and again["whole_window_cycles"] == first["whole_window_cycles"]
    dump = bytearray((run / "memory.dump").read_bytes())
    dump[24 + 16] ^= 0x01  # the first data byte of the first region
    (run / "memory.dump").write_bytes(bytes(dump))
    worse = G.regrade(run, rig.layout, rig.oracle, target=_TARGET, local=rig.local)
    assert not worse["quotable"] and [g["correct"] for g in worse["per_group"]] == [False, True]
    (run / "memory.dump").write_bytes(bytes(dump[:-3]))
    assert G.regrade(run, rig.layout, rig.oracle, target=_TARGET, local=rig.local)["status"] == "refused"


def test_a_changed_logit_fails_the_argmax_too(rig):
    verdict = rig(memory_edit={16: 0x7F})
    assert not verdict["quotable"]
    assert verdict["argmax"]["from_dump"] == 0 and not verdict["argmax"]["agrees_with_oracle"]


def test_a_truncated_dump_is_refused_not_graded(rig):
    verdict = rig(truncate=5)
    assert verdict["status"] == "refused" and not verdict["quotable"]
    assert "dump is not complete" in verdict["refusal"]


def test_no_dump_at_all_is_refused(rig):
    verdict = rig(no_dump=True)
    assert verdict["status"] == "refused" and not verdict["quotable"]


def test_a_missing_group_line_is_refused(rig):
    """70 of 71 group lines means the program did not finish its window; nothing is graded."""
    verdict = rig(console_groups=[(1, "conv2d", 40)])
    assert verdict["status"] == "refused" and "group line" in verdict["refusal"]


def test_a_nonzero_exit_is_refused(rig):
    verdict = rig(exit=5)
    assert verdict["status"] == "refused"


def test_a_hardware_assertion_refuses_the_run_without_waiting_for_its_budget(rig):
    import time

    started = time.monotonic()
    verdict = rig(assertion="cmds(cmd_id).valid")
    assert time.monotonic() - started < 40
    assert verdict["status"] == "refused" and not verdict["quotable"]
    assert verdict["refusal"].startswith("hardware_assertion: cmds(cmd_id).valid at DMACommandTracker.scala:88")
    assert verdict["stderr_capture"]["stopped_on_assertion"]


def test_a_flooded_stderr_is_bounded_and_the_run_is_still_read(rig, tmp_path):
    verdict = rig(flood_lines=120_000)  # ~11 MB of diagnostics
    capture = verdict["stderr_capture"]
    assert capture["elided_bytes"] > 0 and capture["total_bytes"] > 10_000_000
    assert (tmp_path / "run" / "emulator.stderr.txt").stat().st_size < 8 * 1024 * 1024 + 4096
    # The FINISHED line sits after the flood and is still parsed from the kept tail.
    assert verdict["simulated"]["emulator"]["done"] == "1"
    assert verdict["status"] == "graded" and verdict["quotable"]


def test_a_map_for_another_elf_is_refused_before_running(rig, tmp_path):
    other = _minimal_elf(tmp_path / "other.elf")
    other.write_bytes(other.read_bytes() + b"different")
    with pytest.raises(G.GsimWholeModelError, match="recorded for ELF"):
        G.run_gsim_whole_model(other, {"schema": W.MEMORY_MAP_SCHEMA, "elf_sha256": "0" * 64, "groups": []}, {},
                               target=_TARGET, out=tmp_path / "x")  # fmt: skip


def test_the_pipeline_measurer_returns_correctness_apart_from_cycles(rig, tmp_path, monkeypatch):
    """The orchestrator's measurer contract: cycles and correctness are separate fields, the machine is
    what the emulator's receipt says it models (UNKNOWN when no registry entry declares its FIRRTL),
    and the memory image slices to exactly the dumped bytes at the map's addresses."""
    from merlin.targetgen import gsim_emulator

    first = rig()  # writes the scenario the stand-in emulator replays
    layout = {
        "schema": W.MEMORY_MAP_SCHEMA,
        "elf_sha256": first["elf_sha256"],
        "groups": [
            *rig.layout["groups"],
        ],
    }
    oracle = rig.oracle
    monkeypatch.setattr(G, "_local", lambda local, target: rig.local)
    (tmp_path / "memory_map.json").write_text(json.dumps(layout))
    (tmp_path / "oracle.json").write_text(json.dumps(oracle))
    monkeypatch.setattr(gsim_emulator, "engine_home", lambda target, engine: tmp_path / "engine")
    record = {"memory_map": str(tmp_path / "memory_map.json"), "oracle": {"path": str(tmp_path / "oracle.json")}}
    result = G.measure(
        elf=first["elf"],
        elf_sha256=first["elf_sha256"],
        machine="some_machine",
        target=_TARGET,
        build_record=record,
        out=tmp_path / "measured",
    )
    assert result["cycles"] == 1000 and result["engine"] == "gsim"
    assert result["correctness"]["quotable"] is True
    assert result["machine"].startswith("UNKNOWN") and result["machine_requested"] == "some_machine"
    image, base = Path(result["memory_dump"]["path"]).read_bytes(), result["memory_dump"]["base"]
    assert W.grade_memory(lambda a, n: image[a - base : a - base + n], layout, oracle, local=rig.local)["quotable"]
    with pytest.raises(G.GsimWholeModelError, match="asked to measure"):
        G.measure(elf=first["elf"], elf_sha256="0" * 64, machine="m", target=_TARGET, build_record={}, out=tmp_path)


_SMOKE_ELF = (
    "gemmini/mesh_run/runs/gemmini-contract/mesh_layer_16x16x16_i8_i32_input_"
    "e08ded3cb5e9def906f52e3d15ee9e802cfcc81792382235fbc2b35be23f3096/generated/package_kernel.elf"
)


def test_a_signalled_real_run_refuses_to_dump(tmp_path):
    """The HARNESS's signal guard, on the real emulator: fesvr turns SIGTERM into a graceful stop(),
    which is where the dump runs, so without the guard a killed run would dump mid-program memory and
    exit as if finished. It must exit non-zero and leave no dump."""
    import os
    import signal
    import subprocess
    import time

    from merlin.common.paths import runs_dir
    from merlin.targetgen.gsim_emulator import engine_home

    emulator, elf = engine_home(_TARGET, G.DUMP_ENGINE) / "emulator", runs_dir() / _SMOKE_ELF
    if not emulator.is_file() or not elf.is_file():
        pytest.skip("needs the built dump-capable GSIM engine and its smoke ELF")
    regions = tmp_path / "regions.txt"
    regions.write_text("0x80000000 64\n")
    out = tmp_path / "killed.dump"
    proc = subprocess.Popen(
        [str(emulator), str(elf), "+max-cycles=50000000", f"+loadmem={elf}", f"+dump-regions={regions}",
         f"+dump-out={out}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env={**os.environ, "MERLIN_GSIM_LOADMEM": "1"},
    )  # fmt: skip
    time.sleep(2.0)
    proc.send_signal(signal.SIGTERM)
    _, err = proc.communicate(timeout=120)
    assert proc.returncode != 0 and not out.exists()
    assert b"stopped by a signal" in err
