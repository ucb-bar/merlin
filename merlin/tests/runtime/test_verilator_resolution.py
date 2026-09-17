"""The Zephyr Verilator path finds its simulator from the BOARD's registry entry, not from constants.

Which SoC config elaborates a board, and which target's variable points at a prebuilt binary, used to be
literals in the shared backend -- so a second vector SoC could not reach this path without a core edit.
They are now fields of the board entry, and these tests pin both the derivation and that the default
board still resolves the variable and config existing setups rely on.
"""

from __future__ import annotations

import pytest

from merlin.common.paths import target_env_name
from merlin.runtime import boards
from merlin.runtime.backends import zephyr_model as zm


@pytest.fixture()
def acme_board(monkeypatch):
    b = boards.Board(name="acme_board", dram_bytes=1 << 28, harts=2, target="acme", rtl_sim_config="AcmeSoCConfig")
    monkeypatch.setitem(boards.BOARDS, b.name, b)
    monkeypatch.delenv(target_env_name("acme", "VERILATOR"), raising=False)
    return b


def test_the_override_variable_and_config_derive_from_the_board(acme_board, tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_CHIPYARD", str(tmp_path))
    assert zm.verilator_sim(board=acme_board.name) is None, "nothing is built yet"
    sim = tmp_path / "sims" / "verilator" / "simulator-chipyard.harness-AcmeSoCConfig"
    sim.parent.mkdir(parents=True)
    sim.write_text("")
    assert zm.verilator_sim(board=acme_board.name) == sim
    explicit = tmp_path / "prebuilt_sim"
    explicit.write_text("")
    monkeypatch.setenv(target_env_name("acme", "VERILATOR"), str(explicit))
    assert zm.verilator_sim(board=acme_board.name) == explicit, "the target's explicit binary wins"


def test_a_board_that_declares_no_simulator_fails_closed(monkeypatch, tmp_path):
    bare = boards.Board(name="bare_board", dram_bytes=1 << 28, harts=1)
    monkeypatch.setitem(boards.BOARDS, bare.name, bare)
    monkeypatch.setenv("MERLIN_CHIPYARD", str(tmp_path))
    assert zm.verilator_sim(board=bare.name) is None
    with pytest.raises(zm.ZephyrModelError, match="rtl_sim_config"):
        zm.run_on_verilator(tmp_path / "x.elf", board=bare.name)


def test_the_default_board_keeps_the_variable_and_config_it_always_had():
    desc = boards.board(zm.VERILATOR_BOARD)
    cfg, env_name = zm._verilator_facts(zm.VERILATOR_BOARD, None)
    assert cfg and cfg == desc.rtl_sim_config
    assert env_name == target_env_name(desc.target, "VERILATOR")
    # The names existing setups export and build; derived now, but unchanged.
    assert env_name == "MERLIN_SATURN_VERILATOR"
    assert cfg == "MultiSaturnV256D128ShuttleConfig"
