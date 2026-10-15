"""Audit the gSIM TestHarness selected for a FireSim-targeted Phase 1 run.

The chip source revisions must match the selected full FireSim bitstream. The
harness configuration remains distinct from the FireSim wrapper configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from merlin.targetgen import gsim_emulator
from merlin.targetgen.rtl import elaboration

GSIM_CONFIG = "GemminiGsimSerialClkConfig"


def _sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"selected gSIM member is absent or indirect: {path}")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audit(contract: Path, facts_path: Path, chipyard_root: Path, hardware_pins: Path) -> dict:
    """Require one exact source -> FIRRTL -> facts -> emulator chain."""
    selected = yaml.safe_load(contract.read_text())
    facts = json.loads(facts_path.read_text())
    registry = yaml.safe_load(hardware_pins.read_text())
    pins = registry["pins"]
    bitstream = registry["artifacts"]["firesim_gemmini_rocket_30mhz"]
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
            and source_identity["revision"] == pins["gemmini_rocket_chipyard"]["commit"]
            and source_identity["submodules"].get("generators/gemmini")
            == pins["gemmini_rocket_rtl"]["commit"]
        ),
        "facts_match_reproduced_firrtl": (
            inputs.get("target") == "gemmini"
            and inputs.get("fir_sha256") == observed["firrtl_sha256"]
            and source.get("fir_sha256") == observed["firrtl_sha256"]
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
    args = parser.parse_args(argv)
    try:
        report = audit(args.contract, args.rtl_facts, args.chipyard_root, args.hardware_pins)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.exit(2, f"gSIM target alignment: {exc}\n")
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0 if report["gsim_source_aligned"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
