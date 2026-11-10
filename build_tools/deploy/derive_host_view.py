"""Export only a descriptor's currently selected host view, without captures.

This is a source-selection diagnostic, not corpus sealing, numerical
qualification, or permission to replace an existing frozen host view. Ordinary
Phase 0 preparation uses the same HostLane.resolve_capabilities API. Select a
new output directory for each prospective declaration successor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from merlin.targetgen.target_experiment import HostLaneMatrix


def derive(*, descriptor: Path, source_root: Path, output: Path) -> dict:
    producer = Path(__file__).resolve()
    producer_bytes = producer.read_bytes()
    raw = descriptor.read_bytes()
    matrix = HostLaneMatrix.from_mapping(yaml.safe_load(raw).get("host_lane"), descriptor=descriptor)
    if matrix is None:
        raise ValueError("descriptor has no selected host lane")
    if output.exists():
        raise ValueError("host-view destination must be new; frozen views are never replaced")
    views, sources = {}, []
    for name, lane in sorted(matrix.profiles.items()):
        path, document, identity = lane.resolve_capabilities(root=source_root, descriptor=descriptor)
        if path is None or document is None:
            raise ValueError("selected host lane has no capability declaration")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != identity["capability_spec_sha256"]:
            raise ValueError("host capability bytes changed during derivation")
        views[name] = {**identity, "capability_spec": document}
        sources.append({"path": str(path), "sha256": digest})
    # Reopen the same ordinary source and immutable-package selection immediately
    # before publishing; a changed YAML or payload must not acquire old custody.
    for name, lane in sorted(matrix.profiles.items()):
        _, document, identity = lane.resolve_capabilities(root=source_root, descriptor=descriptor)
        if {**identity, "capability_spec": document} != views[name]:
            raise ValueError("host selection changed before publication")
    if descriptor.read_bytes() != raw:
        raise ValueError("descriptor changed during host-view derivation")
    if producer.read_bytes() != producer_bytes:
        raise ValueError("host-view producer changed during derivation")
    output.mkdir(parents=True)
    artifact = output / "software/host-capabilities.json"
    artifact.parent.mkdir()
    artifact.write_text(json.dumps(views, indent=2, sort_keys=True) + "\n")
    receipt = {
        "schema": "merlin.host_view_derivation.v1",
        "scope": "prospective descriptor host-view selection only; no capture or execution qualification",
        "producer": {"path": str(producer), "sha256": hashlib.sha256(producer_bytes).hexdigest()},
        "descriptor": {"path": str(descriptor), "sha256": hashlib.sha256(raw).hexdigest()},
        "default_profile": matrix.default,
        "sources": sources,
        "profiles": {
            name: {key: value for key, value in view.items() if key != "capability_spec"}
            for name, view in views.items()
        },
        "artifact": {"path": str(artifact), "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()},
    }
    (output / "host-view-derivation.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    derive(
        descriptor=args.descriptor.resolve(strict=True),
        source_root=args.source_root.resolve(strict=True),
        output=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
