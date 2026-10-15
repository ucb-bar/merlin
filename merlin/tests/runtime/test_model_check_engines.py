"""A gSIM replay image is a board image: it fits the RTL board's DRAM, and both engines are judged per call."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from merlin.common.paths import repo_root
from merlin.model_check import cli
from merlin.model_check import engines as EN
from merlin.model_check import replay as RP

MIB = 1 << 20


def _board(**overrides) -> EN.BoardImage:
    fields = dict(
        name="b",
        catalog="c",
        catalog_sha256="0" * 64,
        dram_base=0x80000000,
        dram_bytes=256 * MIB,
        code_reserve=64 * MIB,
    )
    fields.update(overrides)
    return EN.BoardImage(**fields)


def test_the_rtl_config_selects_its_board_from_the_target_catalog():
    catalog = repo_root() / "examples" / "gemmini" / "platform" / "board-catalog.yaml"
    board = EN.board_for("gemmini", "GemminiRocketConfig", catalog=catalog)
    assert board.dram_bytes == 256 * MIB and board.code_reserve == 64 * MIB
    assert board.weights_base == board.dram_base + board.code_reserve
    assert board.capacity == board.dram_bytes - board.code_reserve
    with pytest.raises(EN.EngineError):
        EN.board_for("gemmini", "NoSuchConfig", catalog=catalog)


def test_board_symbols_declare_the_span_spike_must_model():
    from merlin.runtime.backends import spike_model

    syms = _board().defsyms()
    assert syms[spike_model.DRAM_SPAN_SYMBOL] == 256 * MIB
    assert syms["MERLIN_WEIGHTS_BASE"] == 0x80000000 + 64 * MIB


def _elf(path: Path, segments: list[tuple[int, int]]) -> Path:
    """A minimal little-endian ELF64 with one PT_LOAD per ``(paddr, memsz)``."""
    phoff, phentsize = 64, 56
    header = bytearray(64)
    header[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<Q", header, 0x20, phoff)
    struct.pack_into("<HH", header, 0x36, phentsize, len(segments))
    table = b"".join(struct.pack("<IIQQQQQQ", 1, 0, 0, a, a, 0, n, 8) for a, n in segments)
    path.write_bytes(bytes(header) + table)
    return path


def test_an_image_that_does_not_fit_the_board_is_refused(tmp_path):
    board = _board()
    ok = _elf(tmp_path / "ok.elf", [(board.dram_base, 4096), (board.weights_base, 8 * MIB)])
    assert EN.check_fits(ok, board)["image_end"] == board.weights_base + 8 * MIB
    code_overrun = _elf(tmp_path / "a.elf", [(board.dram_base, 65 * MIB)])
    beyond = _elf(tmp_path / "b.elf", [(board.dram_base, 4096), (board.weights_base, 200 * MIB)])
    below = _elf(tmp_path / "c.elf", [(board.dram_base - 4096, 4096)])
    for elf in (code_overrun, beyond, below):
        with pytest.raises(EN.EngineError):
            EN.check_fits(elf, board)


def _case(symbol: str, nbytes: int, label: str = "") -> RP.ReplayCase:
    return RP.ReplayCase(
        symbol=symbol,
        label=label or symbol,
        roles=("input_0", "out_0"),
        operands=(np.zeros(nbytes, np.int8), np.zeros(4, np.int32)),
        expected=(np.zeros(4, np.int32),),
    )


def test_images_are_per_symbol_and_split_only_when_the_board_is_full():
    board = _board(dram_bytes=64 * MIB + 3 * MIB)  # a 3 MiB operand/result region
    chunks = EN.board_chunks([_case("a", MIB), _case("b", MIB), _case("a", MIB), _case("a", MIB)], board)
    assert [[c.symbol for c in chunk] for chunk in chunks] == [["a", "a"], ["a"], ["b"]]
    with pytest.raises(EN.EngineError):
        EN.board_chunks([_case("a", 4 * MIB)], board)


def test_a_per_call_bound_sizes_the_image_bound():
    cases = [_case("a", 16), _case("a", 16)]
    bound = EN.image_max_cycles(cases, call_max_cycles=1000)
    assert bound == EN.BOOT_CYCLES + 2 * (1000 + EN.DIGEST_CYCLES_PER_BYTE * 64)


def _job(cases):
    plan = [
        {"case": i, "symbol": c.symbol, "label": c.label, "outputs": [{"role": "out_0", "digest": (1, 2)}]}
        for i, c in enumerate(cases)
    ]
    return EN.ImageJob(0, Path("."), list(cases), plan, Path("x.elf"), 1000, {})


def _run(engine, status, console):
    return EN.EngineRun(engine, status, console, 1.0, 0, None)


def test_calls_are_judged_on_gsim_and_compared_with_spike_call_by_call():
    job = _job([_case("a", 8, "a0"), _case("a", 8, "a1"), _case("a", 8, "a2")])
    spike = _run("spike", "completed", "MCCASE 0 0 0x1 0x2\nMCCASE 1 0 0x1 0x2\nMCCASE 2 0 0x1 0x2\nMCDONE\n")
    gsim = _run("gsim", "cycle_cap", "MCCYC 0 0x10\nMCCASE 0 0 0x1 0x2\nMCCYC 1 0x20\nMCCASE 1 0 0x9 0x2\n")
    rows = EN._case_rows(job, gsim, spike, call_max_cycles=0x18)  # noqa: SLF001
    assert [r["exact"] for r in rows] == [True, False, False]
    assert [r["engines_agree"] for r in rows] == [True, False, None]
    assert [r["status"] for r in rows] == ["completed", "completed", "cycle_cap"]
    assert [r["spike"]["exact"] for r in rows] == [True, True, True]
    assert [r.get("over_call_budget") for r in rows] == [False, True, None]
    kernels = [{"symbol": "a", "cases": 3, "exact": 1, "status": "fail"}]
    EN.annotate_kernels(kernels, rows, cycles_kind="rtl")
    assert kernels[0]["status"] == "unfinished" and kernels[0]["unfinished"] == ["a2"]
    assert kernels[0]["cycles"] == {"kind": "rtl", "min": 0x10, "max": 0x20, "calls": 2}
    assert kernels[0]["engines_agree"] == 1 and kernels[0]["over_call_budget"] == 1


def test_gsim_argv_keeps_the_bound_template_and_preloads_the_same_elf():
    engine = EN.GsimEngine.__new__(EN.GsimEngine)
    engine.binary = Path("/sims/emulator")
    engine.template = ("{elf}", "+max-cycles=20000000", "+loadmem={elf}")
    assert engine.argv(Path("/w/r.elf"), max_cycles=5000) == [
        "/sims/emulator",
        "/w/r.elf",
        "+max-cycles=5000",
        "+loadmem=/w/r.elf",
    ]
    with pytest.raises(EN.EngineError):
        engine.argv(Path("/w/r.elf"), max_cycles=0)


def test_a_reaped_engine_reports_its_cpu_time_and_a_late_one_is_killed():
    import subprocess
    import sys
    import time

    quick = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    cpu = EN._reap(quick, deadline=time.monotonic() + 30)  # noqa: SLF001
    assert cpu is not None and cpu >= 0 and quick.returncode == 0
    slow = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    assert EN._reap(slow, deadline=time.monotonic() + 0.3) is None  # noqa: SLF001
    EN._kill_group(slow)  # noqa: SLF001
    assert slow.returncode is not None
    run = EN.EngineRun("gsim", "completed", "", 10.0, 0, 3000, cpu_s=1.5)
    assert run.record()["cycles_per_cpu_second"] == 2000


_FAKE_EMULATOR = """#!/bin/sh
# argv: <elf> +max-cycles=N +loadmem=<elf>; behaviour chosen by the FAKE_MODE file next to this script
mode=$(cat "$(dirname "$0")/FAKE_MODE")
echo "MCCYC 0 0x10"
echo "MCCASE 0 0 0x1 0x2"
case "$mode" in
  ok) echo MCDONE; echo "Cycles: 4242" >&2; exit 0 ;;
  cap) echo "GSIM timeout: no RTL/TSI completion after 500 cycles" >&2; exit 124 ;;
  assert) echo "Assertion failed: bad command" >&2; echo MCDONE; exit 0 ;;
  trap) echo "TRAP mcause=2"; exit 7 ;;
  harness) echo "GSIM harness error: invalid plusarg" >&2; exit 2 ;;
  hang) sleep 30 ;;
esac
"""


@pytest.mark.parametrize(
    ("mode", "status"),
    [("ok", "completed"), ("cap", "cycle_cap"), ("assert", "fault"), ("trap", "fault"), ("harness", "error")],
)
def test_a_gsim_run_says_whether_the_program_or_the_engine_stopped_it(tmp_path, monkeypatch, mode, status):
    monkeypatch.setenv("MERLIN_GSIM_SLOT_ROOT", str(tmp_path / "slots"))
    emulator = tmp_path / "emulator"
    emulator.write_text(_FAKE_EMULATOR)
    emulator.chmod(0o755)
    (tmp_path / "FAKE_MODE").write_text(mode)
    elf = tmp_path / "r.elf"
    elf.write_bytes(b"\x7fELF")
    engine = EN.GsimEngine.__new__(EN.GsimEngine)
    engine.binary, engine.template, engine.environment = emulator, ("{elf}", "+max-cycles=1", "+loadmem={elf}"), {}
    engine.failure_markers, engine._revalidate = ("TRAP ",), lambda: None
    run = engine.run(elf, max_cycles=500, timeout_s=20, workdir=tmp_path)
    assert run.status == status
    assert "MCCASE 0 0 0x1 0x2" in run.console  # finished calls survive any stop
    if mode == "ok":
        assert run.cycles == 4242 and run.cpu_s is not None
    if mode == "cap":
        assert run.cycles == 500


def test_the_wall_bound_kills_the_emulator_and_keeps_its_console(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_GSIM_SLOT_ROOT", str(tmp_path / "slots"))
    emulator = tmp_path / "emulator"
    emulator.write_text(_FAKE_EMULATOR)
    emulator.chmod(0o755)
    (tmp_path / "FAKE_MODE").write_text("hang")
    elf = tmp_path / "r.elf"
    elf.write_bytes(b"\x7fELF")
    engine = EN.GsimEngine.__new__(EN.GsimEngine)
    engine.binary, engine.template, engine.environment = emulator, ("{elf}", "+max-cycles=1"), {}
    engine.failure_markers, engine._revalidate = (), lambda: None
    run = engine.run(elf, max_cycles=500, timeout_s=1.5, workdir=tmp_path)
    assert run.status == "timeout" and "wall bound" in run.detail and run.wall_s < 15
    assert "MCCASE" in run.console


def test_a_busy_slot_pool_is_a_reported_timeout_not_a_crash(tmp_path, monkeypatch):
    from contextlib import contextmanager

    from merlin.targetgen import rtl_engine_policy

    @contextmanager
    def busy(**_kw):
        raise TimeoutError("all 5 GSim slots stayed busy")
        yield  # pragma: no cover

    monkeypatch.setattr(rtl_engine_policy, "gsim_runtime_slot", busy)
    elf = tmp_path / "r.elf"
    elf.write_bytes(b"\x7fELF")
    engine = EN.GsimEngine.__new__(EN.GsimEngine)
    engine.binary, engine.template, engine.environment = Path("/bin/false"), ("{elf}", "+max-cycles=1"), {}
    engine.failure_markers, engine._revalidate = (), lambda: None
    run = engine.run(elf, max_cycles=500, timeout_s=5, workdir=tmp_path)
    assert run.status == "timeout" and "no gSIM slot" in run.detail and "stayed busy" in run.detail


def test_a_spent_budget_is_reported_not_run(tmp_path, monkeypatch):
    import time

    job = _job([_case("a", 8, "a0")])
    monkeypatch.setattr(EN, "build_images", lambda *a, **k: [job])
    monkeypatch.setattr(EN, "run_spike", lambda *a, **k: pytest.fail("spike ran after the deadline"))
    engine = type("E", (), {"board": _board(), "default_max_cycles": 1000})()
    engine.run = lambda *a, **k: pytest.fail("gsim ran after the deadline")
    monkeypatch.setattr(EN, "_sha256", lambda _p: "0" * 64)
    monkeypatch.setattr(Path, "stat", lambda _p: type("S", (), {"st_size": 1})())
    monkeypatch.setattr(Path, "write_text", lambda *_a, **_k: 0)
    rows, records = EN.replay_on_board(
        "t", engine, job.cases, {}, {}, [], None, tmp_path, timeout_s=60, deadline=time.monotonic() - 1
    )
    assert rows[0]["status"] == "timeout" and "budget was spent" in rows[0]["stopped"]
    assert records[0]["gsim"]["status"] == "timeout"


def test_replay_kernel_takes_unique_and_engine_bounds():
    args = cli.build_parser().parse_args(
        [
            "replay-kernel",
            "--submission",
            "S",
            "--model",
            "m",
            "--all",
            "--unique",
            "--split",
            "visible",
            "--engine",
            "gsim",
            "--jobs",
            "2",
            "--call-max-cycles",
            "1000000",
            "--out",
            "o.json",
        ]
    )
    assert args.unique and args.all_groups and args.program is None and args.jobs == 2
    assert args.call_max_cycles == 1_000_000 and args.max_cycles is None


def _bundle(tmp_path: Path):
    """A one-program bundle: symbol ``s0`` called three times (two groups), ``s1`` once, on two inputs."""
    import json

    from merlin.model_check import bundle as B
    from merlin.model_check import catalog as CAT

    root = tmp_path / "bundle"
    program = root / "m" / "programs" / "p"
    (program / "dev").mkdir(parents=True)
    (program / "catalog.json").write_text(
        json.dumps(
            {
                "schema": CAT.CATALOG_SCHEMA,
                "target": "t",
                "stage": "s",
                "ref_prepared": "r",
                "dev_prepared": "d",
                "routing": {},
                "interfaces": {"s0": "", "s1": ""},
            }
        )
    )
    routed = [{"symbol": "s0", "source_region_id": "g0"}, {"symbol": "s0", "source_region_id": "g1"}]
    (program / "dev" / "device_signatures.json").write_text(json.dumps({"routed": routed}))
    blobs = CAT.BlobStore(root / "m" / "blobs")
    for input_id, scale in (("visible/a", 1), ("visible/b", 2)):
        calls = []
        for ordinal, (symbol, n) in enumerate((("s0", 64), ("s0", 8), ("s1", 4), ("s0", 8))):
            weight = blobs.put(np.full(n * scale, ordinal, np.int8))
            calls.append(
                {
                    "symbol": symbol,
                    "ordinal": ordinal,
                    "roles": ["weight_0", "out_0"],
                    "operands": [weight, blobs.put(np.zeros(1, np.int32))],
                    "expected": [blobs.put(np.zeros(1, np.int32))],
                }
            )
        CAT.write_calls(root / "m" / input_id / "p" / "calls.json", calls)
    return B.Bundle(root)


def test_unique_keeps_one_recorded_call_per_symbol_and_never_reshapes_it(tmp_path):
    from merlin.model_check import commands

    bundle = _bundle(tmp_path)
    inputs = bundle.inputs("m", "visible")
    cases = commands._cases(bundle, "m", "p", inputs, group=None, all_calls=True, unique=True)  # noqa: SLF001
    assert [(c.symbol, c.label) for c in cases] == [("s0", "visible/a/p/s0#1"), ("s1", "visible/a/p/s1#2")]
    assert cases[0].operands[0].shape == (8,)  # the recorded operand, as recorded
    only = commands._cases(bundle, "m", "p", inputs, group="g1", all_calls=True, unique=True)  # noqa: SLF001
    assert [c.symbol for c in only] == ["s0"]
    every = commands._cases(bundle, "m", "p", inputs, group=None, all_calls=True)  # noqa: SLF001
    assert len(every) == 8
