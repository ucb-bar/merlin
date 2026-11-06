"""`merlin-compile --matrix-unit ...`: argument contract and the shape of the result, without native work."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from merlin import compile_cli
from merlin.compile import matrix_leg
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
    monkeypatch.setattr(matrix_leg, "_run_shipped_on_device_simulator", lambda *a, **k: {"status": "not_declared"})
    monkeypatch.setattr(zm, "image_cpus", lambda brd, harts: harts)
    monkeypatch.setattr(zm, "_gate", lambda got, refs: {"ok": True, "fp32_cos": 1.0})
    monkeypatch.setattr(
        "merlin.runtime.boards.board",
        lambda name: SimpleNamespace(name=name, vlen=512, dram_base=0x80000000),
    )
    pkg = SimpleNamespace(schedule_text="schedule", cflags=["-march=rv64gcv"])
    out = compile_cli._compile_matrix(
        {},
        Path("bundle"),
        tmp_path,
        pkg,
        ["perop"],
        {"fp32": np.ones(4)},
        board="chip_board",
        harts=2,
        iters=1,
        warmup=0,
        timeout=60,
        matrix=("unit_a", "ConfigA", "chip_a"),
    )
    assert seen["spike"][0] == tmp_path / "twin.elf"  # the stand-in runs, never the shipped image
    assert seen["pair"]["routing"].identity() == {"support_target": "chip_a", "unit": "unit_a", "config": "ConfigA"}
    assert "perop" in seen["pair"]["features"] and seen["pair"]["vlen"] == 512
    assert out["binary"] == str(tmp_path / "ship.elf") and out["twin_binary"] == str(tmp_path / "twin.elf")
    assert out["status"] == "verified" and out["verify"]["of"] == "stand-in twin"
    assert out["matrix"]["opu_mac_share"] == 0.75
    assert out["matrix"]["routing_report"] == str(ship_dir / matrix_image.ROUTING_REPORT)

    # A torchAO integer source already owns its quantization. The matrix pair must
    # receive the preserve-source choice instead of re-quantizing float attention.
    compile_cli._compile_matrix(
        {},
        Path("bundle"),
        tmp_path,
        pkg,
        ["perop"],
        {"w8a8": np.ones(4)},
        board="chip_board",
        harts=2,
        iters=1,
        warmup=0,
        timeout=60,
        matrix=("unit_a", "ConfigA", "chip_a"),
        int8_compute=False,
    )
    assert seen["pair"]["int8_compute"] is False


def test_a_multi_hart_block_that_does_not_lower_is_re_derived_once(tmp_path, monkeypatch):
    """As on the RVV route: the register block is narrowed to the per-hart tile only when the build
    actually rejects it (`vector.mask`), and only for a multi-hart image."""
    calls = []
    pair = matrix_image.MatrixPair(
        ship={"elf": tmp_path / "s.elf"},
        twin={"elf": tmp_path / "t.elf", "ram_bytes": 1},
        debug=None,
        lowering_digest="d",
        unit_instruction_counts={"ACC": 1},
        debug_instruction_counts={},
        report={},
        ship_dir=tmp_path,
    )

    def fake_pair(bundle, work, **kw):
        calls.append(sorted(kw["features"]))
        if len(calls) == 1:
            raise RuntimeError("'vector.mask' op expects only one operation to mask")
        return pair

    monkeypatch.setattr(matrix_image, "build_matrix_pair", fake_pair)
    monkeypatch.setattr(compile_cli, "_workload_features", lambda *a, **k: ["narrow"])
    monkeypatch.setattr(matrix_leg, "_run_shipped_on_device_simulator", lambda *a, **k: {"status": "not_declared"})
    monkeypatch.setattr(zm, "run_on_spike", lambda elf, **kw: {"prefix": np.ones(1), "metrics": {}})
    monkeypatch.setattr(zm, "image_cpus", lambda brd, harts: harts)
    monkeypatch.setattr("merlin.runtime.boards.board", lambda n: SimpleNamespace(vlen=512, dram_base=0))
    pkg = SimpleNamespace(schedule_text="s", cflags=[])
    out = compile_cli._compile_matrix(
        {},
        Path("b"),
        tmp_path,
        pkg,
        ["wide"],
        {},
        board="b",
        harts=2,
        iters=1,
        warmup=0,
        timeout=1,
        matrix=("u", "c", "t"),
    )
    assert len(calls) == 2 and "narrow" in calls[1] and "wide" not in calls[1]
    assert out["harts_split_block_retry"] == {"tried": ["wide"], "used": ["narrow"]}


def test_build_gate_audits_matrix_images_without_running_either_simulator(tmp_path, monkeypatch):
    pair = matrix_image.MatrixPair(
        ship={"elf": tmp_path / "ship.elf"},
        twin={"elf": tmp_path / "twin.elf"},
        debug=None,
        lowering_digest="same-source-lowering",
        unit_instruction_counts={"ACC": 7},
        debug_instruction_counts={},
        report={"routed_contractions": 1},
        ship_dir=tmp_path,
    )
    monkeypatch.setattr(matrix_image, "build_matrix_pair", lambda *a, **k: pair)
    monkeypatch.setattr(zm, "run_on_spike", lambda *a, **k: pytest.fail("compile-only gate ran Spike"))
    monkeypatch.setattr(
        matrix_leg,
        "_run_shipped_on_device_simulator",
        lambda *a, **k: pytest.fail("compile-only gate ran unit simulator"),
    )
    monkeypatch.setattr(zm, "image_cpus", lambda brd, harts: harts)
    monkeypatch.setattr("merlin.runtime.boards.board", lambda n: SimpleNamespace(vlen=512, dram_base=0))
    result = compile_cli._compile_matrix(
        {}, Path("bundle"), tmp_path, SimpleNamespace(schedule_text="", cflags=[]), [], {}, board="b", harts=1,
        iters=1, warmup=0, timeout=1, matrix=("u", "c", "t"), matrix_gate="build", int8_compute=False,
    )  # fmt: skip
    assert result["status"] == "compiled"
    assert result["binary"] == str(pair.ship["elf"])
    assert result["twin_binary"] == str(pair.twin["elf"])
    assert result["matrix"]["twin_executed"] is False
    assert result["matrix"]["unit_instruction_counts"] == {"ACC": 7}
    assert result["verification_scope"].startswith("none;")


_NOTHING = {
    "routed_contractions": 0,
    "not_routed_contractions": 3,
    "not_routed_by_reason": {"declined by the selector": 3},
    "opu_mac_share": 0.0,
    "not_routed": [{"why": "declined by the selector: min(M=8, N=2048) < tile edge 64"}],
}


def test_a_model_with_nothing_to_route_says_why_and_builds_no_twin(tmp_path, monkeypatch):
    """A decode-length token count leaves every contraction smaller than one tile. That used to surface as
    "the shipped image contains NONE of the unit's instructions" -- true, and silent about why -- after a
    stand-in twin had been built for an image that could never ship."""
    built = []

    def fake_build(bundle, work, **kw):
        built.append(kw["matrix_scalar_tile"])
        Path(work).mkdir(parents=True, exist_ok=True)
        return {"elf": Path(work) / "zephyr.elf"}

    provider = SimpleNamespace(
        load_contract=lambda unit: None, derive_encodings=lambda c: SimpleNamespace(encodings={})
    )
    routing = SimpleNamespace(unit="unit_a", config="ConfigA", support_target="t", provider=lambda: provider)
    monkeypatch.setattr(
        "merlin.runtime.boards.board",
        lambda name: SimpleNamespace(check_matrix_route=lambda *route: None, check_matrix_image=lambda h: None),
    )
    monkeypatch.setattr("merlin.targetgen.unit_pins.unit_pins", lambda target, unit=None: [])
    monkeypatch.setattr(zm, "build_app", fake_build)
    monkeypatch.setattr(matrix_image, "_audit", lambda elf, enc: {})
    monkeypatch.setattr(matrix_image, "routing_report", lambda build_dir, routing: dict(_NOTHING))
    with pytest.raises(matrix_image.NothingRouted, match=r"min\(M=8, N=2048\) < tile edge 64") as raised:
        matrix_image.build_matrix_pair(
            Path("bundle"), tmp_path, board="b", harts=1, cpus=2, routing=routing, features=frozenset(),
            rvv_schedule="s", cflags=[],
        )  # fmt: skip
    assert built == [False]  # the shipped image only: no twin for an image that cannot ship
    assert raised.value.report_path.is_file()


def test_merlin_compile_reports_a_model_with_nothing_to_route(tmp_path, monkeypatch):
    def nothing(bundle, work, **kw):
        raise matrix_image.NothingRouted("unit_a", dict(_NOTHING), tmp_path / "routing_report.json")

    monkeypatch.setattr(matrix_image, "build_matrix_pair", nothing)
    monkeypatch.setattr(zm, "run_on_spike", lambda *a, **k: pytest.fail("simulated an image that was not built"))
    monkeypatch.setattr(zm, "image_cpus", lambda brd, harts: harts)
    monkeypatch.setattr("merlin.runtime.boards.board", lambda n: SimpleNamespace(vlen=512, dram_base=0))
    out = compile_cli._compile_matrix(
        {}, Path("b"), tmp_path, SimpleNamespace(schedule_text="s", cflags=[]), [], {}, board="b", harts=1,
        iters=1, warmup=0, timeout=1, matrix=("unit_a", "c", "t"),
    )  # fmt: skip
    assert out["status"] == "not_routed" and "tile edge 64" in out["reason"]
    assert out["matrix"]["routed_contractions"] == 0
    assert out["matrix"]["not_routed_by_reason"] == {"declined by the selector": 3}
