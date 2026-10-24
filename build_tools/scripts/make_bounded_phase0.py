#!/usr/bin/env python3
"""Define a strict retained-call Phase 0 experiment from a bounded suite.

Every lane follows general native admission; no operator roster or exception list
participates in selection. Host refusals remain explicit in the derived manifest.
Use the printed CLI commands after generation; Phase 0 packaging is not an
execution/numerical verdict on Spike or RTL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def write_yaml(path, document):
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    path.chmod(0o600)


def generate(args):
    suite = json.loads(args.suite.read_bytes())
    cases = suite.get("selected_cases", suite.get("cases"))
    if not cases:
        raise ValueError("suite must contain cases or selected_cases")
    ids = [case.get("case_id", case["overload"]) for case in cases]
    identities = set(ids)
    if len(identities) != len(ids):
        raise ValueError("duplicate case identities")
    selected = cases if args.limit is None else cases[: args.limit]
    if len(selected) < 2:
        raise ValueError("strict experiment requires disjoint public and hidden cases")
    # Commit semantic truth before asking the compiler for any placement.
    from merlin.targetgen.functional_truth import expectation

    commitments = {c.get("case_id", c["overload"]): expectation(c) for c in selected}
    native = [c for c in selected if commitments[c.get("case_id", c["overload"])]["lane"] == "device"]
    guards = [c for c in selected if commitments[c.get("case_id", c["overload"])]["lane"] != "device"]
    if len(native) < 2:
        raise ValueError("strict truth experiment needs disjoint public and hidden native cases")
    public, hidden = native[1:], native[0]
    output = args.output
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    (output / "expectations.json").write_text(json.dumps(commitments, sort_keys=True, indent=2) + "\n")
    selected_suite = output / "cases.json"
    selected_suite.write_text(
        json.dumps({"cases": selected, "denominator_sha256": suite.get("denominator_sha256")}, sort_keys=True) + "\n"
    )
    selected_suite.chmod(0o600)
    descriptor = yaml.safe_load(args.descriptor.read_bytes())
    target = descriptor["target"]
    (descriptor.get("workload_spec") or {}).pop("applications", None)
    write_yaml(output / "descriptor.yaml", descriptor)
    flow = {
        "policy": "spec_int8",
        "software_spec": str(args.software_spec),
        "host_capabilities": str(args.host_capabilities),
        "execution_provider": str(args.execution_provider),
        "rtl_facts": str(args.facts),
        "device_package": str(args.device_package),
    }

    def row(select, lane):
        return {
            "cases": str(selected_suite),
            "captures": str(args.captures),
            "select": sorted(c.get("case_id", c["overload"]) for c in select),
            "lane": lane,
        }

    recipe = {
        "capsule_policy": "derived_only",
        "capsules": [],
        "performance_oracles": {"L2": "spike", "L3": "verilator"},
        "datapath": {
            "semantic_defaults": {"must_accelerate": True},
            "compare": "exact_int",
            "requant_output_dtype": "i8",
            "requant_shift": 3,
            "required_oracle_tiers": ["L0", "L1", "L2", "L3"],
        },
        "core_aten": {
            "workers": args.workers,
            "allow_empty_host_guards": True,
            "overlay": str(args.overlay),
            "numeric_flow": flow,
            "cohorts": {
                "public": [row(public, "truth")],
                "hidden": [row([hidden], "truth")],
                "host_guard": [row(guards, "truth")] if guards else [],
            },
        },
    }
    write_yaml(output / "recipe.yaml", recipe)
    derive = output / "derive"
    for name, recipe_path in (
        ("experiment.yaml", output / "recipe.yaml"),
        ("experiment-run.yaml", derive / "recipe.yaml"),
    ):
        write_yaml(
            output / name,
            {
                "schema_version": 1,
                "id": f"{target}-bounded-core-aten-phase0",
                "target": target,
                "description": "Phase 0 (functional, all native ops): strict independent truth lanes.",
                "inputs": {"target": str(output / "descriptor.yaml")},
                "phases": {
                    0: {
                        "adapter": "capsule_derivation",
                        "config": {
                            "recipe": str(recipe_path),
                            "descriptor": str(output / "descriptor.yaml"),
                            "capability_contract": str(args.capability_contract),
                            "evidence_mode": "diagnostic",
                            "performance_template": str(args.performance_template),
                        },
                    }
                },
            },
        )
    receipt = {
        "suite_sha256": hashlib.sha256(args.suite.read_bytes()).hexdigest(),
        "selected_count": len(selected),
        "lane_policy": "functional_truth",
        "hidden_count": 1,
        "device_count": len(native),
        "no_work_count": sum(c["lane"] == "no-work" for c in commitments.values()),
        "host_guard_count": len(guards),
        "placement": "user-reviewed semantic truth; compiler placement must match in both directions",
        "scope": "definition only; no execution claim",
    }
    (output / "definition.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (output / "definition.json").chmod(0o600)
    print(json.dumps(receipt, indent=2))
    import shlex

    def quote(path):
        return shlex.quote(str(path))

    print(
        f"python -m merlin.cli experiment corpus derive {quote(output / 'experiment.yaml')} "
        f"--rtl-facts {quote(args.facts)} --output {quote(derive)}"
    )
    print(
        f"python -m merlin.cli experiment run {quote(output / 'experiment-run.yaml')} --phase 0 "
        f"--phase0-rtl-facts {quote(args.facts)} --phase0-conformance-spec {quote(derive / 'requirements.yaml')} "
        f"--phase0-synth-profile {quote(derive / 'synth.yaml')} --phase0-evidence-mode diagnostic "
        f"--run-dir {quote(output / 'run')}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "suite",
        "captures",
        "output",
        "descriptor",
        "facts",
        "device-package",
        "overlay",
        "software-spec",
        "host-capabilities",
        "execution-provider",
        "capability-contract",
        "performance-template",
    ):
        parser.add_argument("--" + name, type=lambda p: Path(p).absolute(), required=True)
    parser.add_argument("--limit", type=int, help="optional deterministic prefix of the suite")
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=8)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 2:
        parser.error("--limit must be at least 2")
    generate(args)


if __name__ == "__main__":
    main()
