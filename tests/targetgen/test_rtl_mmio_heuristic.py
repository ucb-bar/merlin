"""F6 regression: pure-Verilog MMIO accelerators must classify as
``mmio_accelerator``, not as ``rtl_or_systemc_model``.

Phase 1 surfaced that NVDLA — which is functionally an MMIO accelerator
(host writes config registers, reads status) — was being missed by the
classifier because the chisel/RTL scanners only looked for
Rocket-Chip-style TLRegisterRouter / RegField patterns. Pure Verilog
accelerators expose their control plane through one of three idioms:

  (a) NVDLA CSB regfile (``reg_rd_data`` / ``reg_offset`` / ...)
  (b) Vivado AXI-Lite slv_regN array
  (c) Generic AXI-Lite handshake (awaddr + awvalid + wdata + wvalid + bvalid)

This test asserts each idiom triggers ``rtl_mmio_register`` and that the
classifier promotes the target to ``mmio_accelerator``.
"""

from __future__ import annotations

from pathlib import Path

from targetgen.intake import build_source_inventory, classify_inventory
from targetgen.intake.rtl_scanner import _looks_like_mmio_register


def test_csb_regfile_quad_triggers_mmio() -> None:
    text = """
        module foo (
            input  [31:0] reg_wr_data,
            input  [11:0] reg_offset,
            input         reg_wr_en,
            output [31:0] reg_rd_data
        );
        endmodule
    """
    is_mmio, evidence = _looks_like_mmio_register(text)
    assert is_mmio
    assert evidence is not None
    assert "CSB" in evidence


def test_partial_csb_quad_does_not_trigger() -> None:
    """Two-of-four matches is too weak — must require at least three."""
    text = "input [31:0] reg_wr_data; input [11:0] reg_offset;"
    is_mmio, _ = _looks_like_mmio_register(text)
    assert not is_mmio


def test_slv_reg_array_triggers_mmio() -> None:
    text = "reg [31:0] slv_reg0; reg [31:0] slv_reg1; reg [31:0] slv_reg2;"
    is_mmio, evidence = _looks_like_mmio_register(text)
    assert is_mmio
    assert "slv_reg" in (evidence or "").lower()


def test_axi_lite_handshake_triggers_mmio() -> None:
    text = "S_AXI_AWADDR S_AXI_AWVALID S_AXI_WDATA S_AXI_WVALID S_AXI_BVALID"
    is_mmio, evidence = _looks_like_mmio_register(text.lower())
    assert is_mmio
    assert "handshake" in (evidence or "").lower()


def test_random_verilog_does_not_trigger_mmio() -> None:
    text = "module fifo(clk, rst, data_in, data_out, valid, ready); endmodule"
    is_mmio, _ = _looks_like_mmio_register(text)
    assert not is_mmio


def test_synthetic_nvdla_style_fixture_classifies_as_mmio(tmp_path: Path) -> None:
    """End-to-end: a tiny Verilog dir mimicking an NVDLA regfile must
    classify as mmio_accelerator."""
    src = tmp_path / "fake_nvdla"
    src.mkdir()
    (src / "regfile.v").write_text(
        """
module fake_acc_regfile (
    input  [31:0] reg_wr_data,
    input  [11:0] reg_offset,
    input         reg_wr_en,
    output [31:0] reg_rd_data
);
endmodule
"""
    )
    inv = build_source_inventory(target="fake_nvdla", sources=[src])
    assert "rtl_mmio_register" in inv.detected_source_kinds
    cls = classify_inventory(inv)
    assert "mmio_accelerator" in cls.source_styles, f"expected mmio_accelerator in {cls.source_styles}"
    assert (
        "rtl_or_systemc_model" not in cls.source_styles
    ), "mmio_accelerator must take precedence over rtl_or_systemc_model"
