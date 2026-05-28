from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from targetgen.intake import (  # noqa: E402
    AVAILABLE_SCANNERS,
    build_source_inventory,
    chipyard_scanner,
    chisel_scanner,
    cmake_scanner,
    docs_scanner,
    hal_scanner,
    mlir_scanner,
    rtl_scanner,
)
from targetgen.model import SourceFinding, SourceInventory  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


def _kinds(findings: list[SourceFinding]) -> set[str]:
    return {f.kind for f in findings}


def test_available_scanners_listed() -> None:
    names = [name for name, _, _ in AVAILABLE_SCANNERS]
    assert names == [
        "mlir",
        "cmake",
        "llvm",
        "hal",
        "chipyard",
        "chisel",
        "rtl",
        "systemc",
        "docs",
    ]


def test_mlir_scanner_detects_dialect_and_passes() -> None:
    fixture = FIXTURES / "external_mlir_cuda_tile"
    findings = mlir_scanner.scan(fixture)
    kinds = _kinds(findings)
    assert "mlir_dialect" in kinds
    assert "mlir_op_definition" in kinds
    assert "mlir_pass" in kinds
    dialect_finding = next(f for f in findings if f.kind == "mlir_dialect")
    assert dialect_finding.symbol == "cuda_tile"


def test_cmake_scanner_detects_external_hal_driver_and_dialect() -> None:
    radiance = FIXTURES / "radiance_gluon_gpu"
    cuda = FIXTURES / "external_mlir_cuda_tile"
    radiance_kinds = _kinds(cmake_scanner.scan(radiance))
    cuda_kinds = _kinds(cmake_scanner.scan(cuda))
    assert "iree_external_hal_driver_registration" in radiance_kinds
    assert "mlir_dialect_cmake" in cuda_kinds
    assert "cmake_project" in radiance_kinds
    assert "cmake_project" in cuda_kinds


def test_hal_scanner_detects_driver_and_registration() -> None:
    findings = hal_scanner.scan(FIXTURES / "radiance_gluon_gpu")
    kinds = _kinds(findings)
    assert "hal_driver_source" in kinds
    assert "hal_device_source" in kinds
    assert "hal_registration" in kinds
    assert "hal_command_buffer" in kinds
    assert "hal_executable_format" in kinds


def test_chipyard_and_chisel_scanners_detect_rocc() -> None:
    fixture = FIXTURES / "chipyard_gemmini_rocc"
    cy_kinds = _kinds(chipyard_scanner.scan(fixture))
    chisel_kinds = _kinds(chisel_scanner.scan(fixture))
    assert "chipyard_project" in cy_kinds
    assert "firesim_collateral" in cy_kinds
    assert "chisel_attachment_rocc" in chisel_kinds


def test_chisel_and_rtl_scanners_detect_mmio_axi() -> None:
    fixture = FIXTURES / "fft_generator_mmio"
    chisel_kinds = _kinds(chisel_scanner.scan(fixture))
    rtl_findings = rtl_scanner.scan(fixture)
    rtl_kinds = _kinds(rtl_findings)
    assert "chisel_attachment_mmio" in chisel_kinds
    assert "verilog_module" in rtl_kinds
    assert "systemverilog_interface" in rtl_kinds
    assert "rtl_axi_port" in rtl_kinds


def test_docs_scanner_detects_isa_and_runtime_sections() -> None:
    findings = docs_scanner.scan(FIXTURES / "radiance_gluon_gpu")
    kinds = _kinds(findings)
    assert "isa_documentation" in kinds
    assert "memory_model_documentation" in kinds
    assert "synchronization_documentation" in kinds
    assert "runtime_documentation" in kinds
    assert "driver_documentation" in kinds
    assert "simulator_documentation" in kinds


def test_build_source_inventory_aggregates_findings() -> None:
    inv = build_source_inventory(
        target="fake_radiance",
        sources=[FIXTURES / "radiance_gluon_gpu"],
    )
    assert isinstance(inv, SourceInventory)
    assert inv.target == "fake_radiance"
    assert len(inv.repositories) == 1
    assert inv.repositories[0].name == "radiance_gluon_gpu"
    assert "hal_driver_source" in inv.detected_source_kinds
    assert "isa_documentation" in inv.detected_source_kinds
    assert inv.missing_information == []


def test_build_source_inventory_records_missing_path(tmp_path: Path) -> None:
    bogus = tmp_path / "does_not_exist"
    inv = build_source_inventory(target="t", sources=[bogus])
    assert inv.findings == []
    assert any("does not exist" in m for m in inv.missing_information)


def test_build_source_inventory_scanner_filter_unknown_raises() -> None:
    with pytest.raises(ValueError):
        build_source_inventory(
            target="t",
            sources=[FIXTURES / "external_mlir_cuda_tile"],
            scanners=["nonexistent_scanner"],
        )


def test_ingest_cli_writes_artifacts(tmp_path: Path) -> None:
    import targetgen_cmd

    fixture = FIXTURES / "external_mlir_cuda_tile"
    out_dir = tmp_path / "out"
    parser = __import__("argparse").ArgumentParser()
    targetgen_cmd.setup_parser(parser)
    args = parser.parse_args(
        [
            "ingest",
            "--target-name",
            "fake_cuda_tile",
            "--source",
            str(fixture),
            "--out-dir",
            str(out_dir),
        ]
    )
    rc = targetgen_cmd.main(args)
    assert rc == 0
    target_dir = out_dir / "fake_cuda_tile"
    inventory = json.loads((target_dir / "source_inventory.json").read_text())
    assert inventory["target"] == "fake_cuda_tile"
    kinds = inventory["detected_source_kinds"]
    assert "mlir_dialect" in kinds
    assert "mlir_dialect_cmake" in kinds
    evidence = json.loads((target_dir / "evidence_graph.json").read_text())
    assert "mlir_dialect" in evidence["findings_by_kind"]
