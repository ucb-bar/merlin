"""`merlin-compile --matrix-unit ...`: argument contract and the shape of the result, without native work."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from merlin import compile_cli
from merlin.runtime.backends import matrix_image
from merlin.runtime.backends import zephyr_model as zm

_MATRIX = ["--matrix-unit", "unit_a", "--matrix-config", "ConfigA", "--matrix-support-target", "chip_a"]


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--workload", "w", "--dtype", "int8", "--matrix-unit", "unit_a"], "given together"),
        (["--workload", "w", "--dtype", "fp32", *_MATRIX, "--run", "spike", "--board", "b"], "--dtype int8"),
        (["--workload", "w", "--dtype", "int8", *_MATRIX, "--run", "k1"], "--run spike|zephyr --board"),
    ],
)
def test_matrix_flags_are_refused_outside_their_route(argv, message, capsys):
    with pytest.raises(SystemExit):
        compile_cli.main(argv)
    assert message in capsys.readouterr().err


def test_the_twin_is_what_runs_and_both_binaries_are_reported(tmp_path, monkeypatch):
    seen = {}
    ship_dir = tmp_path / "board_h2"
    pair = matrix_image.MatrixPair(
        ship={"elf": tmp_path / "ship.elf"},
        twin={"elf": tmp_path / "twin.elf", "ram_bytes": 1 << 20},
        debug=None,
        lowering_digest="abc123",
        unit_instruction_counts={"ACC": 10},
        debug_instruction_counts={},
        report={"routed_contractions": 3, "opu_mac_share": 0.75, "not_routed_by_reason": {"x": 1}},
        ship_dir=ship_dir,
    )

    def fake_pair(bundle, work, **kw):
        seen["pair"] = kw
        return pair

    def fake_spike(elf, **kw):
        seen["spike"] = (elf, kw)
        return {"prefix": np.ones(4, dtype=np.float32), "metrics": {"cycles": 99}}

    monkeypatch.setattr(matrix_image, "build_matrix_pair", fake_pair)
    monkeypatch.setattr(zm, "run_on_spike", fake_spike)
    monkeypatch.setattr(zm, "image_cpus", lambda brd, harts: harts)
    monkeypatch.setattr(zm, "_gate", lambda got, refs: {"ok": True, "fp32_cos": 1.0})
    monkeypatch.setattr(
        "merlin.runtime.boards.board",
        lambda name: SimpleNamespace(name=name, vlen=512, dram_base=0x80000000),
    )
    pkg = SimpleNamespace(schedule_text="schedule", cflags=["-march=rv64gcv"])
    out = compile_cli._compile_matrix(
        {}, Path("bundle"), tmp_path, pkg, ["perop"], {"fp32": np.ones(4)},
        board="chip_board", harts=2, iters=1, warmup=0, timeout=60, matrix=("unit_a", "ConfigA", "chip_a"),
    )
    assert seen["spike"][0] == tmp_path / "twin.elf"  # the stand-in runs, never the shipped image
    assert seen["pair"]["routing"].identity() == {"support_target": "chip_a", "unit": "unit_a", "config": "ConfigA"}
    assert "perop" in seen["pair"]["features"] and seen["pair"]["vlen"] == 512
    assert out["binary"] == str(tmp_path / "ship.elf") and out["twin_binary"] == str(tmp_path / "twin.elf")
    assert out["status"] == "verified" and out["verify"]["of"] == "stand-in twin"
    assert out["matrix"]["opu_mac_share"] == 0.75
    assert out["matrix"]["routing_report"] == str(ship_dir / matrix_image.ROUTING_REPORT)
