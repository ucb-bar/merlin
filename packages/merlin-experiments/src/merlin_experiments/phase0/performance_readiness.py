"""Pure selection checks for explicitly bound Phase 0 performance inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from merlin.targetgen import corpus_spec as CS
from merlin.targetgen.rtl.facts import observed_facts
from merlin.targetgen.target_experiment import load_target_experiment
from merlin.targetgen.target_registry import observed_contract

from .evidence import evidence_artifacts
from .freeze import selection
from .profiles import load_profile
from .sweeps import expand_sweeps


def check_performance_oracles(command: dict, *, target: str, profile_target: str, diagnostic: bool) -> None:
    inputs = command["inputs"]
    selected = selection(command, target)
    if hashlib.sha256(selected.views_json).hexdigest() != command["phase0_evidence"]["views_sha256"]:
        raise ValueError("selected Phase 0 evidence changed before performance preflight")
    profile = load_profile(
        profile_target, descriptor=inputs["descriptor"], include_holdouts=False, diagnostic=diagnostic,
        **{key: inputs[key] for key in (
            "recipe", "performance_template", "conformance_spec", "synth_profile", "smt_profile", "software_spec",
        ) if key in inputs},
    )
    requirement_raw = Path(inputs["conformance_spec"]).read_bytes() if inputs.get("conformance_spec") else None
    requirement = yaml.safe_load(requirement_raw) if requirement_raw is not None else None
    if requirement is not None and not isinstance(requirement, dict):
        raise ValueError("selected conformance requirement must be a mapping")
    basis_raw = evidence_artifacts(selected).get("coverage/performance-basis.json") if requirement is not None else None
    with (
        observed_contract(target, selected.contract, source_path=Path(inputs["capability_contract"])),
        observed_facts(target, selected.refreshed_facts,
                       Path(inputs["rtl_facts"]) if inputs.get("rtl_facts") else None),
    ):
        binding = CS.derive_binding(
            load_target_experiment(inputs["descriptor"]), profile.get("datapath", {}),
            contract=selected.contract, facts=selected.loaded_facts, taxonomy=selected.isa_taxonomy,
        )
        expand_sweeps(
            profile, binding, evidence=selected, trait_facts=selected.performance_facts,
            skipped=[], blocked_unimplemented=[], errors=[], selected_requirement=requirement,
            requirement_sha256=hashlib.sha256(requirement_raw).hexdigest() if requirement_raw is not None else None,
            selected_performance_basis=json.loads(basis_raw) if basis_raw is not None else None,
            performance_basis_sha256=hashlib.sha256(basis_raw).hexdigest() if basis_raw is not None else None,
        )
