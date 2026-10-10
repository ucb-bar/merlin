"""A group program's emulator run is graded from its own dump, and a run that leaves none is refused:
no reading, never a pass.  The console spellings are the target driver's own, so a real target names them."""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pytest
import selected_driver

from merlin.perf import whole_model_group_timing as T
from merlin.perf import whole_model_verdict as V

if selected_driver.selected_is_generic("gemmini"):
    pytest.skip(
        "tests compiler/harness modules a gemmini support package ships itself; the selected "
        "generic data provider (merlin.runtime.backends.chipyard_rocc) does not ship them",
        allow_module_level=True,
    )

pytestmark = pytest.mark.target("gemmini")


def _fake_emulator(tmp_path: Path, console: str, *, dump: bool) -> Path:
    home = tmp_path / "engine"
    home.mkdir()
    binary = home / "emulator"
    body = tmp_path / "console.txt"
    body.write_text(console, encoding="utf-8")
    binary.write_text(
        "#!/bin/bash\n"
        f"cat {body}\n"
        + ("" if not dump else "echo unreachable\n")
        + "echo '[gsim-emu] FINISHED: cycles=10 wall=1.0s (1 cyc/s) done=1 exit_code=0' 1>&2\n",
        encoding="utf-8",
    )
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    (home / "build_receipt.json").write_text(json.dumps({"binary_sha256": T._sha256(binary)}), encoding="utf-8")
    return binary


def test_a_group_run_that_dumps_no_output_is_refused(tmp_path, monkeypatch):
    selected_driver.require_support("gemmini")
    from merlin.perf import whole_model_build as W

    elf = Path(sys.executable).resolve()  # any little-endian ELF64: nothing here executes it
    t = V.PROTOCOL_TEMPLATES
    console = "\n".join(
        [
            t["group"].format(group=5, kind="matmul", cycles=1234, sum="UNKNOWN", checksum="UNKNOWN"),
            t["full_model"].format(cycles=2000),
        ]
    )
    emulator = _fake_emulator(tmp_path, console + "\n", dump=False)
    layout = {
        "schema": W.MEMORY_MAP_SCHEMA,
        "elf": str(elf),
        "elf_sha256": T._sha256(elf),
        "groups": [
            {
                "group": 5,
                "kind": "matmul",
                "symbol": "B_g5",
                "address": 1 << 40,
                "bytes": 8,
                "elements": 8,
                "element_bytes": 1,
                "compare": "exact",
                "inputs": [],
            }
        ],
    }
    (tmp_path / "map.json").write_text(json.dumps(layout))
    (tmp_path / "oracle.json").write_text(json.dumps({"groups": {"5": {"fnv1a": 1, "sum": 1}}, "argmax": 0}))
    monkeypatch.setattr(T, "_local_for", lambda *a, **k: None)
    programs = {5: {"elf": str(elf), "memory_map": str(tmp_path / "map.json"), "oracle": str(tmp_path / "oracle.json")}}
    timed = T.time_group_programs(
        # The console's spellings are the target's own driver's, so a real target names them.
        programs,
        target="gemmini",
        model_capsule=tmp_path,
        out=tmp_path / "runs",
        emulator=emulator,
        max_parallel=1,
        cache=None,
    )
    row = timed[5]
    assert row["status"] == "refused" and "dump" in row["refusal"]
    assert "cycles" not in row and "correct" not in row  # a refused run carries no reading
    # A program that could not be built is refused before anything runs.
    refused = T.time_group_programs(
        {7: {"refusal": "the rebuilt object differs"}}, target="t", model_capsule=tmp_path, out=tmp_path / "r2"
    )
    assert refused[7]["status"] == "refused" and "differs" in refused[7]["refusal"]
