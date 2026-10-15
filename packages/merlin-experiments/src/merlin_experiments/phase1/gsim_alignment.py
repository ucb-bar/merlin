"""Audit the gSIM TestHarness selected for a FireSim-targeted Phase 1 run.

The chip source revisions must match the selected full FireSim bitstream. The
harness configuration remains distinct from the FireSim wrapper configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import yaml

from merlin.compile.host_lane import dts_string_values
from merlin.targetgen import gsim_emulator
from merlin.targetgen.rtl import elaboration

# A new Chipyard TestHarness config must retain the gSIM clock/serial adapters
# while matching the FireSim chip's execution-visible settings (including DRAM).
GSIM_CONFIG = "GemminiGsimFireSimAlignedConfig"
DRAM_BASE = 0x80000000
DRAM_BYTES = 16 * (1 << 30)
CONFIG_SOURCE = "generators/firechip/chip/src/main/scala/GemminiGsimFireSimAlignedConfig.scala"
CONFIG_SOURCE_SHA256 = "5392120d1ed547b3a70c3786dd5a3d3d697f7817f9ce55e1ffc25b6c2ff6b2ec"
BUILD_SUBMODULES = ("generators/gemmini", "generators/saturn", "sims/firesim")


def _sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"selected gSIM member is absent or indirect: {path}")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _dts_identity(path: Path) -> dict:
    """Read the enabled top-level DRAM span and CPU ISA roster from Chipyard DTS."""
    source = path.read_text(encoding="utf-8")
    cells = []
    for key in ("#address-cells", "#size-cells"):
        match = re.search(rf"{re.escape(key)}\s*=\s*<\s*([^>]+)\s*>\s*;", source)
        if match is None:
            raise ValueError(f"DTS has no root {key}: {path}")
        cells.append(int(match.group(1), 0))
    if any(count not in (1, 2) for count in cells):
        raise ValueError(f"DTS has unsupported root address/size cell counts: {path}")
    spans = []
    for match in re.finditer(r"(?:^|\n)\s*(?:\w+:\s*)?memory@[0-9a-fA-F]+\s*\{([^{}]*)\};", source, re.S):
        body = match.group(1)
        if re.search(r'\bstatus\s*=\s*"disabled"', body):
            continue
        if not re.search(r'\bdevice_type\s*=\s*"memory"', body):
            continue
        reg = re.search(r"\breg\s*=\s*<([^>]+)>\s*;", body)
        if reg is None:
            raise ValueError(f"enabled memory node has no single register span: {path}")
        words = [int(word, 0) for word in reg.group(1).split()]
        if len(words) != sum(cells) or any(word < 0 or word > 0xFFFFFFFF for word in words):
            raise ValueError(f"enabled memory node has an unsupported register span: {path}")
        address_cells, size_cells = cells
        base = 0
        for word in words[:address_cells]:
            base = (base << 32) | word
        size = 0
        for word in words[address_cells:address_cells + size_cells]:
            size = (size << 32) | word
        spans.append((base, size))
    isa = dts_string_values(source, "riscv,isa")
    if len(spans) != 1 or len(isa) != 1:
        raise ValueError(f"DTS must have one enabled DRAM span and one CPU ISA: {path}")
    return {"dram_base": spans[0][0], "dram_bytes": spans[0][1], "host_isa": isa[0], "harts": len(isa)}


def _source_variant(root: Path, base_revision: str, source: dict) -> bool:
    """Accept only the recorded FireSim tree and its build submodules plus the gSIM config."""
    revision = source.get("revision")
    if source.get("config_file") != CONFIG_SOURCE or source.get("config_sha256") != CONFIG_SOURCE_SHA256:
        return False
    if not isinstance(revision, str) or revision == base_revision:
        return False
    ancestor = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", base_revision, revision],
        capture_output=True, check=False,
    )
    if ancestor.returncode != 0:
        return False
    changed = subprocess.run(
        ["git", "-C", str(root), "diff", "--name-only", base_revision, revision],
        capture_output=True, text=True, check=True,
    )
    if changed.stdout.splitlines() != [CONFIG_SOURCE]:
        return False
    selected_submodules = source.get("submodules")
    if not isinstance(selected_submodules, dict):
        return False
    for name in BUILD_SUBMODULES:
        gitlink = subprocess.run(
            ["git", "-C", str(root), "ls-tree", base_revision, "--", name],
            capture_output=True, text=True, check=True,
        )
        metadata, separator, selected_name = gitlink.stdout.strip().partition("\t")
        fields = metadata.split()
        if (
            not separator
            or selected_name != name
            or len(fields) != 3
            or fields[:2] != ["160000", "commit"]
            or selected_submodules.get(name) != fields[2]
        ):
            return False
    return True


def audit(
    contract: Path, facts_path: Path, chipyard_root: Path, hardware_pins: Path,
    board_catalog: Path, gsim_dts: Path,
) -> dict:
    """Require one exact source -> FIRRTL -> facts -> emulator chain."""
    selected = yaml.safe_load(contract.read_text())
    facts = json.loads(facts_path.read_text())
    registry = yaml.safe_load(hardware_pins.read_text())
    pins = registry["pins"]
    bitstream = registry["artifacts"]["firesim_gemmini_rocket_30mhz"]
    board = yaml.safe_load(board_catalog.read_text())["boards"]["firesim_gemmini_rocket_u250_30mhz"]
    firesim_dts = chipyard_root / (
        "sims/firesim-staging/generated-src/firechip.chip.FireSim.FireSimGemminiRocketConfig/"
        "firechip.chip.FireSim.FireSimGemminiRocketConfig.dts"
    )
    firesim_host = _dts_identity(firesim_dts)
    gsim_host = _dts_identity(gsim_dts)
    if bitstream.get("built_from") != ["gemmini_rocket_chipyard", "gemmini_rocket_rtl"]:
        raise ValueError("selected FireSim bitstream names another source lineage")
    block = selected["runner"]["chipyard_rocc"]
    engine = block["engines"]["gsim"]
    config = block["config"]
    inputs = facts["inputs"]
    consistency = facts["source_consistency"]
    source = facts["facts"]["source"]
    production = consistency.get("production") or {}
    elaboration_pin = production.get("elaboration") or {}
    if not isinstance(elaboration_pin.get("path"), str):
        raise ValueError("selected facts have no reproduced source-to-FIRRTL elaboration receipt")
    firrtl = Path(inputs["fir_path"])
    receipt_path = Path(elaboration_pin["path"])
    reproduced = elaboration.verify(receipt_path, firrtl=firrtl, config=config)
    source_identity = reproduced["source"]
    binary = Path(engine["binary"]["path"])
    build_receipt = Path(engine["receipt"]["path"])
    binary_sha = _sha256(binary)
    status, reason, lineage = gsim_emulator._validate_receipt(
        "gemmini", binary, binary_sha, build_receipt
    )
    observed = {
        "phase1_contract_sha256": _sha256(contract),
        "rtl_facts_sha256": _sha256(facts_path),
        "elaboration_receipt_sha256": _sha256(receipt_path),
        "firrtl_sha256": _sha256(firrtl),
        "gsim_binary_sha256": binary_sha,
        "gsim_receipt_sha256": _sha256(build_receipt),
        "gsim_receipt_status": status,
        "gsim_receipt_reason": reason,
        "source": source_identity,
        "firesim_dts_sha256": _sha256(firesim_dts),
        "gsim_dts_sha256": _sha256(gsim_dts),
        "firesim_host": firesim_host,
        "gsim_host": gsim_host,
    }
    checks = {
        "harness_config_selected": (
            config == GSIM_CONFIG
            and selected.get("runtime", {}).get("rtl_sim_config") == GSIM_CONFIG
            and source.get("config") == GSIM_CONFIG
            and consistency.get("config") == GSIM_CONFIG
        ),
        "facts_source_verified": consistency.get("status") == "verified",
        "reproduced_elaboration_pinned": (
            observed["elaboration_receipt_sha256"] == elaboration_pin.get("sha256")
            and (consistency.get("elaboration") or {}).get("status") == "reproduced_exact_firrtl"
            and (consistency.get("elaboration") or {}).get("receipt_sha256") == observed["elaboration_receipt_sha256"]
        ),
        "matches_firesim_recorded_source_pins": (
            Path(source_identity["root"]).resolve() == chipyard_root.resolve()
            and _source_variant(chipyard_root, pins["gemmini_rocket_chipyard"]["commit"], source_identity)
            and source_identity["submodules"].get("generators/gemmini")
            == pins["gemmini_rocket_rtl"]["commit"]
        ),
        "facts_match_reproduced_firrtl": (
            inputs.get("target") == "gemmini"
            and inputs.get("fir_sha256") == observed["firrtl_sha256"]
            and source.get("fir_sha256") == observed["firrtl_sha256"]
        ),
        "host_map_and_isa_match_firesim": (
            firesim_host == gsim_host
            and firesim_host["dram_base"] == DRAM_BASE
            and firesim_host["dram_bytes"] == DRAM_BYTES
            and board.get("dram_base") == DRAM_BASE
            and board.get("dram_bytes") == "16 GiB"
            and board.get("harts") == 1
            and board.get("vector_harts") == 0
            and observed["firesim_dts_sha256"] == board.get("host_dts_sha256")
        ),
        "strict_gsim_receipt_matches_facts": (
            status == "bound"
            and lineage is not None
            and lineage.get("schema_version") == gsim_emulator.STRICT_RECEIPT_SCHEMA
            and lineage.get("firrtl_sha256") == observed["firrtl_sha256"]
            and lineage.get("binary_sha256") == binary_sha
            and observed["gsim_receipt_sha256"] == engine["receipt"].get("sha256")
            and binary_sha == engine["binary"].get("sha256")
        ),
    }
    return {
        "schema": "merlin.phase1.gsim_alignment.v1",
        "checks": checks,
        "observed": observed,
        "gsim_source_aligned": all(checks.values()),
        "chip_equivalence_proven": False,
        "timing_transfer_proven": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--rtl-facts", required=True, type=Path)
    parser.add_argument("--chipyard-root", required=True, type=Path)
    parser.add_argument("--hardware-pins", required=True, type=Path)
    parser.add_argument("--board-catalog", required=True, type=Path)
    parser.add_argument("--gsim-dts", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = audit(
            args.contract, args.rtl_facts, args.chipyard_root, args.hardware_pins,
            args.board_catalog, args.gsim_dts,
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"gSIM target alignment: {exc}\n")
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0 if report["gsim_source_aligned"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
