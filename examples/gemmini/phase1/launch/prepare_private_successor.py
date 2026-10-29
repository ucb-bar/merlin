"""Select new private host evidence while retaining the original public corpus seal."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import yaml


def pin(path):
    path = Path(path).absolute()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def prepare(*, plan_path, config, prep, output):
    """Ordinary selection only: no captures, references, compiler or target execution."""
    from merlin_experiments.phase0.evidence import export_evidence, load_exported_evidence, select_evidence
    from merlin_experiments.phase1 import launch_config as L
    from merlin_experiments.phase1.feedback import private_full_models as F

    from merlin.targetgen.target_experiment import load_target_experiment

    producer_inputs = {str(Path(path).absolute()): pin(path) for path in (plan_path, config, __file__)}
    plan = json.loads(Path(plan_path).read_bytes())
    if plan.get("schema") != "merlin.operator.private_successor_plan.v1":
        raise ValueError("unsupported private successor plan")

    def selected(name):
        row = plan[name]
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise ValueError("successor input needs an exact path and SHA256")
        path = Path(row["path"])
        if not path.is_absolute() or not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("successor input must be an absolute regular non-indirect file")
        if pin(path) != row:
            raise ValueError(f"successor input changed: {name}")
        return path

    names = (
        "original_private_spec",
        "descriptor",
        "host_capabilities",
        "hardware_spec",
        "conformance_spec",
        "application_inventory",
        "rtl_audit",
        "public_corpus_seal",
    )
    inputs = {name: selected(name) for name in names}
    original_bytes = inputs["original_private_spec"].read_bytes()
    original = yaml.safe_load(original_bytes)
    first = original["models"][0]
    for row in original["models"]:
        for field in ("software_spec", "capability_contract", "rtl_facts"):
            if row[field] != first[field] or row[field + "_sha256"] != first[field + "_sha256"]:
                raise ValueError("successor models do not share the original selected contracts")
            if pin(row[field])["sha256"] != row[field + "_sha256"]:
                raise ValueError("original selected model input changed")
    source = Path(__file__).with_name("launch_phase1.py")
    module_spec = importlib.util.spec_from_file_location("successor_operator_environment", source)
    operator = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(operator)
    launch = L.build(L.load(Path(config)), task_root=Path(plan["task_root"]), prep=Path(prep))
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("private successor requires a fresh output directory")
    output.mkdir(parents=True, mode=0o700)
    facts = output / "rtl" / "facts.json"
    facts.parent.mkdir(mode=0o700)
    facts.write_bytes(Path(first["rtl_facts"]).read_bytes())
    facts.with_name("validation.json").write_bytes(inputs["rtl_audit"].read_bytes())
    with operator.configured_environment(launch):
        descriptor = load_target_experiment(inputs["descriptor"], source_root=launch.path("${REPO}"))
        selection = select_evidence(
            original["target"],
            descriptor=descriptor,
            capability_contract_path=first["capability_contract"],
            facts_path=facts,
            hardware_spec=inputs["hardware_spec"],
            software_spec=first["software_spec"],
            conformance_spec=inputs["conformance_spec"],
            inventory_path=inputs["application_inventory"],
            rtl_source_relocations=plan["rtl_source_relocations"],
        )
        if selection.status != "verified" or selection.diagnostics:
            (output / "selection-refusal.json").write_text(json.dumps(selection.diagnostics, indent=2) + "\n")
            raise ValueError("successor ordinary evidence is diagnostic; original inputs remain unchanged")
        if (
            len(selection.host_capabilities) != 1
            or next(iter(selection.host_capabilities.values()))["capability_spec_sha256"]
            != plan["host_capabilities"]["sha256"]
        ):
            raise ValueError("ordinary descriptor did not select the exact successor host contract")
        export_evidence(selection, output / "evidence")
        load_exported_evidence(output / "evidence")
    successor = copy.deepcopy(original)
    if "numerical_qualification" in plan:
        successor["numerical_qualification"] = copy.deepcopy(plan["numerical_qualification"])
    for row in successor["models"]:
        row.update(
            host_capabilities=str(inputs["host_capabilities"]),
            host_capabilities_sha256=plan["host_capabilities"]["sha256"],
            recipe_derivation_root=str(output / "evidence"),
            recipe_derivation_manifest_sha256=pin(output / "evidence" / "evidence-manifest.json")["sha256"],
        )
    spec = output / "private-models.yaml"
    spec.write_text(yaml.safe_dump(successor, sort_keys=False))
    spec.chmod(0o600)
    if "numerical_qualification" in successor:
        from merlin_experiments.phase1.feedback.private_numerical_selection import read

        read(spec)
    proofs = {}
    for row in successor["models"]:
        if pin(row["capture_selection"])["sha256"] != row["capture_selection_sha256"]:
            raise ValueError("original capture selection changed")
        capture = json.loads(Path(row["capture_selection"]).read_bytes())
        proofs[row["id"]] = F._recipe_derivation(
            spec,
            row,
            capture["plan"],
            target=original["target"],
            software_sha256=row["software_spec_sha256"],
            capability_sha256=row["capability_contract_sha256"],
            facts_sha256=row["rtl_facts_sha256"],
            host_capabilities_sha256=row["host_capabilities_sha256"],
        )[0]
        if pin(row["capture_selection"])["sha256"] != row["capture_selection_sha256"]:
            raise ValueError("original capture selection changed during recipe derivation")
    for name in names:
        selected(name)
    if inputs["original_private_spec"].read_bytes() != original_bytes:
        raise ValueError("original private specification changed during successor preparation")
    if any(pin(path) != expected for path, expected in producer_inputs.items()):
        raise ValueError("successor plan, config or producer changed during selection")
    receipt = {
        "schema": "merlin.operator.private_successor_selection.v1",
        "plan": producer_inputs[str(Path(plan_path).absolute())],
        "producer_inputs": producer_inputs,
        "private_spec": pin(spec),
        "evidence_manifest": pin(output / "evidence" / "evidence-manifest.json"),
        "selected_inputs": {name: plan[name] for name in names},
        "recipe_bindings": proofs,
        "scope": "verified private input evidence only; public R26 seal and capsules unchanged; no new corpus seal",
        "captures_references_recipes_facts": "original selections retained without recapture or regeneration",
        "compiler_or_target_execution": False,
        "model_qualification": False,
    }
    (output / "selection.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "config", "prep", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args(argv)
    result = prepare(plan_path=args.plan, config=args.config, prep=args.prep, output=args.output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
