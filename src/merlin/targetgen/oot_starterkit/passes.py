"""Answer-free inspection of a compiler's shared-pass interface declarations.

This tool never runs a compiler or grants effect, numerical or execution
qualification. Actual changed-IR consumption and execution need independent
checks through the ordinary compiler and final program path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from merlin.targetgen.shared_pass_contract import shared_pass_requirement, validate_shared_pass_contract


def _read(path):
    path = Path(path)
    if path.stat().st_size > 2 << 20:
        raise ValueError("shared pass interface input exceeds 2 MiB")
    raw = path.read_bytes()
    if len(raw) > 2 << 20:
        raise ValueError("shared pass interface input exceeds 2 MiB")
    return yaml.safe_load(raw), {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}


def inspect_manifest(manifest_path, *, requirement_path=None):
    """Inspect exact supplied declarations, without starting code generation."""
    manifest, manifest_pin = _read(manifest_path)
    if requirement_path is None:
        requirement, requirement_pin = shared_pass_requirement(), None
    else:
        requirement, requirement_pin = _read(requirement_path)
    contract = validate_shared_pass_contract(manifest, requirement=requirement)
    return {
        "schema": "merlin.shared_pass_interface_inspection.v1",
        "status": "declared_interfaces",
        "scope": "structural declaration only; no compiler execution or qualification",
        "inputs": {"manifest": manifest_pin, "requirement": requirement_pin},
        "requirement": requirement,
        "contract": contract,
        "changed_ir_consumption": "NOT_RUN",
        "numerical_agreement": "NOT_RUN",
        "performance": "NOT_MEASURED",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    requirement = commands.add_parser("requirement", help="print the versioned structural requirements")
    requirement.add_argument("--output", type=Path)
    requirement.add_argument("--contract-version", type=int, choices=(1, 2), default=1)
    requirement.add_argument("--require-supported", action="append", default=[], metavar="FAMILY")
    inspect = commands.add_parser("inspect", help="check a manifest without running its compiler")
    inspect.add_argument("--manifest", type=Path, required=True)
    inspect.add_argument("--requirement", type=Path)
    inspect.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        document = (
            shared_pass_requirement(version=args.contract_version, required_supported=args.require_supported)
            if args.command == "requirement"
            else inspect_manifest(args.manifest, requirement_path=args.requirement)
        )
    except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
        parser.exit(2, f"shared pass interfaces: {exc}\n")
    payload = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        # Keep selected source/requirement files and earlier inspections intact.
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
