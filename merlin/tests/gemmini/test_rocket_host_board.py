"""The selected Gemmini host is one scalar Rocket, not a Kodiak vector SoC."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from merlin.common.paths import repo_root
from merlin.compile.host_lane import require_host_isa_dts
from merlin.compile_cli import compile_rvv
from merlin.runtime import boards
from merlin.runtime.elf_audit import ElfAuditError, upload_bytes
from merlin.targetgen.target_experiment import load_target_experiment

CATALOG = "examples/gemmini/target/board-catalog.yaml"
ROCKET_ISA = "rv64imafdcbzicsr_zifencei_zihpm_zfh_zba_zbb_zbs_xrocket"


def test_selected_board_facts_and_unknown_upload_fail_closed(monkeypatch):
    entries = boards.load_boards(repo_root() / CATALOG)
    experiment = load_target_experiment(repo_root() / "examples/gemmini/target/descriptor.yaml")
    assert experiment.host_board == "gemmini_rocket_verilator"
    assert set(entries) == {
        "gemmini_rocket_verilator",
        "firesim_gemmini_rocket_u250_30mhz",
        "gemmini_rocket_spike_functional",
    }
    assert [entries[name].dram_bytes for name in sorted(entries)] == [16 << 30, 4 << 30, 256 << 20]
    # The functional model's declared span shares the RTL board's host CPU roster (same DTS).
    assert (
        entries["gemmini_rocket_spike_functional"].host_dts_sha256
        == entries["gemmini_rocket_verilator"].host_dts_sha256
    )
    for selected in entries.values():
        assert selected.target == "gemmini"
        assert selected.harts == 1
        assert selected.n_vector_harts == 0
        assert selected.flow == boards.FLOW_BAREMETAL
        assert selected.loader is None and selected.loader_baud is None
        with pytest.raises(boards.BoardRegistryError, match="no qualified upload loader"):
            _ = selected.loader_bytes_per_s
        with pytest.raises(ElfAuditError, match="no qualified upload loader"):
            upload_bytes(selected, [], {})

    monkeypatch.setattr(boards, "board", lambda name: entries[name])
    result = compile_rvv(
        "not_a_capture",
        "int8",
        run="verilator",
        verify=False,
        package=None,
        auto_capture=False,
        timeout=5,
        board=experiment.host_board,
        selected_host_dts_required=True,
        expected_board_target="gemmini",
    )
    assert result["status"] == "not_run"
    assert "no qualified compile_cli bare-metal execution path" in result["reason"]


@pytest.mark.parametrize(
    "name,relative_dts",
    [
        (
            "gemmini_rocket_verilator",
            "sims/verilator/generated-src/"
            "chipyard.harness.TestHarness.GemminiRocketConfig/"
            "chipyard.harness.TestHarness.GemminiRocketConfig.dts",
        ),
        (
            "firesim_gemmini_rocket_u250_30mhz",
            "sims/firesim-staging/generated-src/"
            "firechip.chip.FireSim.FireSimGemminiRocketConfig/"
            "firechip.chip.FireSim.FireSimGemminiRocketConfig.dts",
        ),
    ],
)
def test_selected_elaborated_dts_is_the_pinned_nonvector_rocket(name, relative_dts):
    chipyard = os.environ.get("MERLIN_CHIPYARD_GEMMINI_ROCKET")
    if not chipyard:
        pytest.skip("selected MERLIN_CHIPYARD_GEMMINI_ROCKET checkout not available")
    dts = Path(chipyard) / relative_dts
    if not dts.is_file():
        pytest.skip(f"selected elaborated DTS not available: {dts}")
    selected = boards.load_boards(repo_root() / CATALOG)[name]
    assert require_host_isa_dts(
        ["-march=rv64gc_zba_zbb_zbs_zfh"],
        dts,
        expected_sha256=selected.host_dts_sha256,
    ) == [ROCKET_ISA]
    with pytest.raises(ValueError, match=r"missing \['v'\]"):
        require_host_isa_dts(["-march=rv64gcv"], dts, expected_sha256=selected.host_dts_sha256)
