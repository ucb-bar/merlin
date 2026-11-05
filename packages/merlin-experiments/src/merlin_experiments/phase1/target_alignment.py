"""Audit a selected gSIM/FireSim source correspondence using target-owned data.

The chip source revisions must match the selected full FireSim bitstream. The
harness configuration remains distinct from the FireSim wrapper configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import yaml

from merlin.compile.host_lane import _isa_parts
from merlin.model_check.binding_sources import dts_memory, read_dts
from merlin.runtime.boards import load_boards
from merlin.targetgen import gsim_emulator
from merlin.targetgen.rtl import elaboration


def _selection(path: Path) -> dict:
    selected = yaml.safe_load(path.read_bytes())
    keys = {"schema", "target", "config", "config_source", "build_submodules", "hardware"}
    if (
        not isinstance(selected, dict)
        or set(selected) != keys
        or selected["schema"] != "merlin.phase1.target_alignment.v1"
    ):
        raise ValueError("unknown target alignment selection")
    hardware = selected["hardware"]
    if (
        not isinstance(hardware, dict)
        or set(hardware)
        != {"bitstream", "board", "chipyard_pin", "accelerator_pin", "accelerator_submodule", "firesim_dts"}
        or any(not isinstance(value, str) or not value for value in hardware.values())
    ):
        raise ValueError("incomplete alignment hardware selection")
    source = selected["config_source"]
    if not isinstance(source, dict) or set(source) != {"path", "sha256"}:
        raise ValueError("alignment requires an exact configuration source pin")
    digest = source["sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("invalid configuration source SHA256")
    modules = selected["build_submodules"]
    if (
        not isinstance(modules, list)
        or not modules
        or len(modules) != len(set(modules))
        or any(not isinstance(value, str) or not value for value in modules)
        or hardware["accelerator_submodule"] not in modules
    ):
        raise ValueError("invalid alignment build submodule roster")
    for value in (source["path"], hardware["firesim_dts"], *modules):
        if not isinstance(value, str) or Path(value).is_absolute() or ".." in Path(value).parts:
            raise ValueError("alignment source members must be relative and contained")
    if any(not isinstance(selected[key], str) or not selected[key] for key in ("target", "config")):
        raise ValueError("alignment target/config is missing")
    return selected


def _sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"selected gSIM member is absent or indirect: {path}")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _dts_identity(path: Path) -> dict:
    """Read literal enabled memory and CPU nodes through the shared DTS parser."""
    root = read_dts(path.read_text(encoding="utf-8"))
    isas = []
    for parent, node in root.walk():
        if node.strings("device_type") == ["memory"]:
            if parent is not root:
                raise ValueError("alignment requires root-addressed memory, without bus translation")
            for key in ("#address-cells", "#size-cells"):
                if root.cells(key) not in ([1], [2]):
                    raise ValueError("unsupported root memory cell count")
            if any(type(word) is not int or not 0 <= word <= 0xFFFFFFFF for word in node.cells("reg")):
                raise ValueError("memory registers require unsigned literal cells")
        if node.strings("device_type") == ["cpu"]:
            if parent is not None and parent.strings("status") not in ([], ["ok"], ["okay"]):
                raise ValueError("unsupported disabled CPU container")
            if node.strings("status") in ([], ["ok"], ["okay"]):
                values = node.strings("riscv,isa")
                if len(values) != 1:
                    raise ValueError("each enabled CPU requires one ISA")
                isas.extend(values)
    spans = [span for span in dts_memory(root) if span["enabled"]]
    if len(spans) != 1 or spans[0]["bytes"] <= 0 or not isas:
        raise ValueError("DTS requires one enabled memory span and a nonempty CPU roster")
    return {
        "dram_base": spans[0]["base"],
        "dram_bytes": spans[0]["bytes"],
        "host_isas": isas,
        "harts": len(isas),
        "vector_harts": sum("v" in _isa_parts(isa)[1] for isa in isas),
    }


def _source_variant(root: Path, base_revision: str, source: dict, selection: dict) -> bool:
    """Accept only the recorded FireSim tree and its build submodules plus the gSIM config."""
    revision = source.get("revision")
    if (
        source.get("config_file") != selection["config_source"]["path"]
        or source.get("config_sha256") != selection["config_source"]["sha256"]
    ):
        return False
    if not isinstance(revision, str) or revision == base_revision:
        return False
    ancestor = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", base_revision, revision],
        capture_output=True,
        check=False,
    )
    if ancestor.returncode != 0:
        return False
    changed = subprocess.run(
        ["git", "-C", str(root), "diff", "--name-only", base_revision, revision],
        capture_output=True,
        text=True,
        check=True,
    )
    if changed.stdout.splitlines() != [selection["config_source"]["path"]]:
        return False
    selected_submodules = source.get("submodules")
    if not isinstance(selected_submodules, dict):
        return False
    for name in selection["build_submodules"]:
        gitlink = subprocess.run(
            ["git", "-C", str(root), "ls-tree", base_revision, "--", name],
            capture_output=True,
            text=True,
            check=True,
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
    contract: Path,
    facts_path: Path,
    chipyard_root: Path,
    hardware_pins: Path,
    board_catalog: Path,
    gsim_dts: Path,
    *,
    selection: Path,
) -> dict:
    """Require one exact source -> FIRRTL -> facts -> emulator chain."""
    policy = _selection(selection)
    hardware = policy["hardware"]
    selected = yaml.safe_load(contract.read_text())
    facts = json.loads(facts_path.read_text())
    registry = yaml.safe_load(hardware_pins.read_text())
    pins = registry["pins"]
    bitstream = registry["artifacts"][hardware["bitstream"]]
    board = load_boards(board_catalog)[hardware["board"]]
    firesim_dts = chipyard_root / hardware["firesim_dts"]
    firesim_host = _dts_identity(firesim_dts)
    gsim_host = _dts_identity(gsim_dts)
    if bitstream.get("built_from") != [hardware["chipyard_pin"], hardware["accelerator_pin"]]:
        raise ValueError("selected bitstream names another source lineage")
    block = selected["runner"]["chipyard_rocc"]
    engine = block["engines"]["gsim"]
    config = block["config"]
    inputs = facts["inputs"]
    consistency = facts["source_consistency"]
    source = facts["facts"]["source"]
    production = consistency.get("production") or {}
    execution_consistency = consistency
    from merlin.targetgen.rtl.source_selection import load_selection
    from merlin.targetgen.rtl.unit_correspondence import ExecutionBindingError, execution_binding

    binding = None
    try:
        binding = execution_binding(facts, config=config)
    except ExecutionBindingError:
        pass  # retain a failed alignment report, never infer correspondence
    if binding is not None and binding["basis"] == "simulation_correspondence":
        pin = binding["selection"]
        if _sha256(Path(pin["path"])) != pin["sha256"]:
            raise ValueError("simulation source selection differs from the correspondence")
        simulation = load_selection(pin["path"], target=inputs["target"])
        if simulation["config"] != config or any(
            simulation["sources"][role] != binding[role] for role in ("firrtl", "hierarchy", "core_hw")
        ):
            raise ValueError("simulation bundle differs from the correspondence")
        production = simulation.get("production") or {}
        from merlin.targetgen.rtl.source_selection import production_consistency

        execution_consistency = production_consistency(simulation)
    elaboration_pin = production.get("elaboration") or {}
    if not isinstance(elaboration_pin.get("path"), str):
        raise ValueError("selected facts have no reproduced source-to-FIRRTL elaboration receipt")
    firrtl = Path(binding["firrtl"]["path"] if binding is not None else inputs["fir_path"])
    receipt_path = Path(elaboration_pin["path"])
    reproduced = elaboration.verify(receipt_path, firrtl=firrtl, config=config)
    source_identity = reproduced["source"]
    binary = Path(engine["binary"]["path"])
    build_receipt = Path(engine["receipt"]["path"])
    binary_sha = _sha256(binary)
    status, reason, lineage = gsim_emulator._validate_receipt(
        policy["target"],
        binary,
        binary_sha,
        build_receipt,
    )
    observed = {
        "alignment_selection_sha256": _sha256(selection),
        "phase1_contract_sha256": _sha256(contract),
        "rtl_facts_sha256": _sha256(facts_path),
        "elaboration_receipt_sha256": _sha256(receipt_path),
        "firrtl_sha256": _sha256(firrtl),
        "facts_firrtl_sha256": _sha256(Path(inputs["fir_path"])),
        "facts_config": source.get("config"),
        "execution_config": config,
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
            config == policy["config"]
            and selected.get("runtime", {}).get("rtl_sim_config") == policy["config"]
            and binding is not None
        ),
        "facts_source_verified": consistency.get("status") == "verified",
        "reproduced_elaboration_pinned": (
            observed["elaboration_receipt_sha256"] == elaboration_pin.get("sha256")
            and (execution_consistency.get("elaboration") or {}).get("status") == "reproduced_exact_firrtl"
            and (execution_consistency.get("elaboration") or {}).get("receipt_sha256")
            == observed["elaboration_receipt_sha256"]
        ),
        "matches_firesim_recorded_source_pins": (
            Path(source_identity["root"]).resolve() == chipyard_root.resolve()
            and _source_variant(chipyard_root, pins[hardware["chipyard_pin"]]["commit"], source_identity, policy)
            and source_identity["submodules"].get(hardware["accelerator_submodule"])
            == pins[hardware["accelerator_pin"]]["commit"]
        ),
        "facts_match_reproduced_firrtl": (
            inputs.get("target") == policy["target"]
            and inputs.get("fir_sha256") == observed["facts_firrtl_sha256"]
            and source.get("fir_sha256") == observed["facts_firrtl_sha256"]
            and binding is not None
            and binding["firrtl"]["sha256"] == observed["firrtl_sha256"]
        ),
        "host_map_and_isa_match_firesim": (
            firesim_host == gsim_host
            and firesim_host["dram_base"] == board.dram_base
            and firesim_host["dram_bytes"] == board.dram_bytes
            and firesim_host["harts"] == board.harts
            and firesim_host["vector_harts"] == board.n_vector_harts
            and observed["firesim_dts_sha256"] == board.host_dts_sha256
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
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--rtl-facts", required=True, type=Path)
    parser.add_argument("--chipyard-root", required=True, type=Path)
    parser.add_argument("--hardware-pins", required=True, type=Path)
    parser.add_argument("--board-catalog", required=True, type=Path)
    parser.add_argument("--gsim-dts", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = audit(
            args.contract,
            args.rtl_facts,
            args.chipyard_root,
            args.hardware_pins,
            args.board_catalog,
            args.gsim_dts,
            selection=args.selection,
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"gSIM target alignment: {exc}\n")
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0 if report["gsim_source_aligned"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
