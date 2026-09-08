"""The Muon GSIM tier grades a declared result page, not console silence."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from merlin.runtime.backends.base import get_backend
from merlin.targetgen.isa_model import IsaModel


BACKEND = get_backend("muon")
H = BACKEND.muon_harness
MO = BACKEND.muon_oracles
MU = BACKEND.muon
RP = BACKEND.muon_result_page

MODEL = IsaModel(target="synthetic", runtime_abi={
    "special_csrs": {"mhartid": 0xF14},
    "apertures": {"console_mmio": 0xFF080000},
})


def _arg(name: str, count: int, dtype: str = "f32") -> object:
    return H.TensorArg(name, 1, count, [0.0] * count, dtype)


def test_result_page_harness_uses_linker_visible_buffers_and_ready_handshake() -> None:
    harness = H.build_external_kernel_main(
        [_arg("X", 2)], [_arg("Y", 2)], kernel_symbol="kernel", model=MODEL,
        result_page=True,
    )

    assert "merlin_result_0[2]" in harness.source
    assert 'section(".data.merlin_result")' in harness.source
    assert "merlin_result_status" in harness.source
    assert "while(merlin_result_status[2]" in harness.source
    assert "volatile uint32_t _out_Y[2]" not in harness.source
    assert harness.results == [{"name": "Y", "symbol": "merlin_result_0",
                                "elements": 2, "dtype": "f32"}]


def test_carrier_is_generated_from_declared_layout_and_policy() -> None:
    manifest = {
        "status": {"soc_address": 0x110004000},
        "outputs": [
            {"name": "Y", "soc_address": 0x110004040, "elements": 2, "dtype": "f32"},
            {"name": "Z", "soc_address": 0x110004080, "elements": 1, "dtype": "f32"},
        ],
    }
    source = RP.render_carrier(
        manifest,
        {"Y": [1.0, -2.0], "Z": [0.5]},
        {"compare": "tolerance_float", "atol": 0.03125, "rtol": 0.015625},
    )

    assert "0x110004000ULL" in source
    assert "0x110004040ULL" in source and "0x110004080ULL" in source
    assert "merlin_numeric_pass" in source and "merlin_numeric_fail" in source
    assert "ordered_f32" in source
    assert "STATUS[2] = MERLIN_RESULT_ACK" in source


def test_outcome_requires_the_final_pc_to_reach_a_retained_symbol() -> None:
    symbols = {"merlin_numeric_pass": 0x80000086, "merlin_numeric_fail": 0x800000C6}
    assert RP.outcome_from_console(
        "[gsim-probe final] rocket_pc=0x80000086\n[gsim-emu] FINISHED: cycles=120000\n",
        symbols,
    ) == "pass"
    assert RP.outcome_from_console(
        "[gsim-probe final] rocket_pc=0x800000c6\n",
        symbols,
    ) == "fail"
    assert RP.outcome_from_console("[gsim-emu] FINISHED: cycles=120000\n", symbols) is None


@pytest.mark.parametrize(("final_pc", "status"), [
    (0x80000086, "pass"),
    (0x800000C6, "fail"),
])
def test_gsim_adapter_prefers_numeric_pc_witness_over_cycle_cap(
        final_pc, status, monkeypatch, tmp_path) -> None:
    cb = {
        "target": "synthetic",
        "_oracle_expected_outputs": {"Y": [1.0]},
        "_oracle_numeric_policy": {"compare": "tolerance_float", "atol": 0.01},
    }
    monkeypatch.setenv("MERLIN_MUON_GSIM_MAXCYCLES", "120000")
    monkeypatch.setattr(MO, "gsim_status", lambda target: (True, "stub"))
    from merlin.targetgen import gsim_emulator as GE
    monkeypatch.setattr(GE, "emulator_path", lambda *a, **k: tmp_path / "emu")
    monkeypatch.setattr(MU, "is_mlir_artifact", lambda src: True)
    monkeypatch.setattr(MU, "compile_mlir_forkfree", lambda *a, **k: tmp_path / "kernel.elf")
    monkeypatch.setattr(H, "args_from_cb", lambda cb: ([], [_arg("Y", 1)]))
    monkeypatch.setattr(RP, "manifest_from_elf", lambda *a, **k: {
        "status": {"soc_address": 0x110004000},
        "outputs": [{"name": "Y", "soc_address": 0x110004040,
                     "elements": 1, "dtype": "f32"}],
    })
    monkeypatch.setattr(RP, "render_carrier", lambda *a, **k: "int main(void){return 0;}\n")
    monkeypatch.setattr(MU, "fuse_soc_elf", lambda *a, **k: tmp_path / "kernel.soc.elf")
    monkeypatch.setattr(RP, "symbol_addresses", lambda *a, **k: {
        "merlin_numeric_pass": 0x80000086, "merlin_numeric_fail": 0x800000C6})
    monkeypatch.setattr(MO, "flops_from_cb", lambda cb: 0)
    monkeypatch.setattr(
        __import__("subprocess"), "run",
        lambda *a, **k: SimpleNamespace(returncode=0),
    )
    console = (f"[gsim-probe final] rocket_pc=0x{final_pc:x}\n"
               "[gsim-emu] FINISHED: cycles=120000\n")
    monkeypatch.setattr(MU, "_read_console", lambda log: (console, len(console), False))

    result = MO.gsim_muon_adapter("synthetic")(cb, "llvm.func @kernel()", tmp_path, 60)

    assert result["numeric_verdict"]["status"] == status
    assert result["numeric_verdict"]["elements_checked"] == 1
    assert not result.get("completion_only")
