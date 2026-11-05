#!/usr/bin/env python3
"""Copy a small, value-free slice of a sealed Phase 0 generation into a regeneration-canary fixture.

The canary (``merlin/tests/<target>/test_phase0_regeneration_canary.py``) replays one sealed model
capsule's device groups through today's grouping, group writer and software admission, and fails
when today's code would refuse a group the sealed corpus contains. It must not read live evidence
at test time, so this script copies exactly what that replay reads -- and nothing numerical beyond
the model's own declared quantization scales:

* the model capsule's interface program and weight manifest (names, shapes, digests; no weights);
* the sealed corpus's qualifying group capsules: their written interface program, the capsule
  record without its golden, and the admission status the sealed corpus recorded;
* the evidence views admission reads (software spec, capability view, host capabilities, readout
  facets, normalized RTL facts), and the raw facts file and capability contract the sealed run
  selected, so the grouping oracle sees the same target the sealed run did;
* the recipe's ``datapath`` block the corpus binding is derived from.

Goldens, inputs, weights and private (hidden) members are never copied. Every copied source is
recorded with its digest, so the fixture says exactly which sealed bytes it is a slice of.

usage: extract_regeneration_canary.py --run <phase0-run> --model <capsule> --recipe <recipe.yaml>
       --rtl-facts <facts.json> --capability-contract <target_contract.json> --output <dir>
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

import yaml

SCHEMA = "merlin.phase0_regeneration_canary.v1"
_GOLDEN_KEYS = {"software_screen", "numeric_falsifiability", "golden", "golden_sha256"}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _gzip(path: Path, raw: bytes) -> dict:
    path.write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))
    return {"file": path.name, "sha256": _sha(raw), "bytes": len(raw)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True, help="the sealed Phase 0 generation run directory")
    parser.add_argument("--model", required=True, help="one model capsule of that run, e.g. SY_source_<name>")
    parser.add_argument("--recipe", type=Path, required=True, help="the recipe the run declared (datapath block)")
    parser.add_argument("--rtl-facts", type=Path, required=True, help="the raw facts file the run selected")
    parser.add_argument("--capability-contract", type=Path, required=True, help="the contract the run selected")
    parser.add_argument("--derivation", type=Path, help="the reviewed derivation the run consumed (provenance)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    from merlin_experiments.phase0.evidence import load_exported_evidence

    phase0 = args.run.resolve() / "phase0"
    evidence = load_exported_evidence(phase0)
    model_dir = phase0 / "capsules" / "model" / args.model
    model = yaml.safe_load((model_dir / "capsule.yaml").read_bytes())
    qualification = model.get("model_qualification") or {}
    if qualification.get("status") != "qualified":
        raise SystemExit(f"{args.model} is not a qualified model capsule of the sealed run")
    args.output.mkdir(parents=True, exist_ok=True)
    existing = args.output / "canary.json"
    previous = json.loads(existing.read_text()) if existing.is_file() else {}
    interface_raw = (model_dir / str(model.get("interface_mlir") or "capsule.interface.mlir")).read_bytes()
    manifests = sorted(model_dir.glob("*.manifest.json"))
    groups = []
    for row in qualification.get("group_capsules") or []:
        directory = phase0 / "capsules" / str(row["capsule"])
        capsule = yaml.safe_load((directory / "capsule.yaml").read_bytes())
        program = (directory / str(capsule.get("interface_mlir") or "capsule.interface.mlir")).read_bytes()
        screen = capsule.get("software_screen") or {}
        groups.append(
            {
                "capsule": row["capsule"],
                "device_groups": row.get("groups"),
                "sealed_status": screen.get("status"),
                "sealed_reason": screen.get("reason"),
                "sealed_program_sha256": screen.get("program_sha256"),
                "capsule_record": {k: v for k, v in capsule.items() if k not in _GOLDEN_KEYS},
                "program": program.decode(),
                "group_identity_sha256": capsule.get("group_identity_sha256"),
            }
        )
    recipe = yaml.safe_load(args.recipe.read_bytes()) or {}
    document = {
        "schema": SCHEMA,
        "target": evidence.target,
        "model": args.model,
        "provenance": {
            "sealed_run": str(args.run.resolve()),
            "derivation": str(args.derivation.resolve()) if args.derivation else None,
            "model_capsule_sha256": _sha((model_dir / "capsule.yaml").read_bytes()),
            "recipe_sha256": _sha(args.recipe.read_bytes()),
            "evidence_manifest_sha256": _sha((phase0 / "evidence-manifest.json").read_bytes()),
            "scope": "value-free public slice: programs, declared scales, evidence views; no goldens or weights",
        },
        "datapath": recipe.get("datapath") or {},
        "interface": _gzip(args.output / "model.interface.mlir.gz", interface_raw),
        "weights_manifest": json.loads(manifests[0].read_bytes()) if manifests else None,
        "views": _gzip(
            args.output / "views.json.gz",
            json.dumps(
                {
                    "software_spec": evidence.software_spec,
                    "contract": evidence.contract,
                    "host_capabilities": evidence.host_capabilities,
                    "readout_facets": [f.to_dict() if hasattr(f, "to_dict") else f for f in evidence.readout_facets],
                    "isa_taxonomy": evidence.isa_taxonomy,
                },
                sort_keys=True,
                default=str,
            ).encode(),
        ),
        "loaded_facts": _gzip(args.output / "loaded-facts.json.gz", json.dumps(evidence.loaded_facts).encode()),
        "rtl_facts": _gzip(args.output / "facts.json.gz", args.rtl_facts.read_bytes()),
        "capability_contract": _gzip(args.output / "target_contract.json.gz", args.capability_contract.read_bytes()),
        "groups": groups,
        # Reviewer-edited: a refusal accepted on review, kept across re-extraction of the same slice.
        "accepted_drift": previous.get("accepted_drift") or [],
    }
    (args.output / "canary.json").write_text(json.dumps(document, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({"output": str(args.output), "groups": len(groups)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
