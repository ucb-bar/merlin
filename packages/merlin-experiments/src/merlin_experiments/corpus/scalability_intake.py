"""Prepare explicitly selected source contracts for the separate Spike suite.

This creates a new functional policy epoch, preserving the original source and
numeric contract. It never imports previous numerical/timing verdicts or turns
an original performance claim into qualification of the new suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from merlin.common.digest import sha256_file
from merlin.targetgen.capsule_common import load_capsule

from ..phase0.program_admission import accelerator_placement_refusal, screen_written
from ..runner import fingerprint
from . import public_scalability as S
from .preparation import copy_input, ordinary_tree


def prepare(sources: list[Path], destination: Path, *, target: str, evidence, evidence_root: Path) -> Path:
    """Recompute selected placement, retain mathematical bytes, and pin all inputs."""
    if not sources or len(sources) > 512 or destination.exists():
        raise ValueError("scalability intake needs a bounded explicit roster and fresh output")
    destination.mkdir(parents=True)
    members, lineage, names = [], [], set()
    for source in sources:
        source = source.absolute()
        ordinary_tree(source)
        original = load_capsule(source)
        name = original["name"]
        if name in names or original.get("label") not in {"public", "dev"} or original.get("kind") == "model":
            raise ValueError("scalability source must be a unique public capsule, not a full model")
        names.add(name)
        source_digest = fingerprint(source)
        cap = {key: value for key, value in original.items() if key != "__dir__"}
        for key in ("performance", "comparison_group"):
            cap.pop(key, None)
        cap["required_oracle_tiers"] = list(S.POLICY["tiers"])
        # ``extends`` is the old scheduler's predecessor, not source inheritance.
        # The standalone source remains complete; retain the old relationship in
        # lineage while this separate suite owns its own exact L0/L1/L2 policy.
        if cap.get("max_oracle_tier") not in (None, "L2", "L3", "L4", "L5"):
            raise ValueError("scalability source does not permit functional execution")
        cap.pop("max_oracle_tier", None)
        cap.pop("extends", None)
        member = "sources/" + name
        selected = destination / member
        copy_input(source, selected, expected_sha256=source_digest, private=True)
        screen = screen_written(cap, selected, target=target, evidence=evidence)
        refusal = accelerator_placement_refusal(screen)
        if refusal is not None:
            raise ValueError(f"{name}: {refusal}")
        cap["software_screen"] = screen
        (selected / "capsule.yaml").chmod(0o600)
        (selected / "capsule.yaml").write_text(yaml.safe_dump(cap, sort_keys=False))
        load_capsule(selected)
        ir = cap.get("interface_mlir") or cap.get("linalg_mlir")
        source_group = sha256_file(selected / ir)
        members.append({"member": member, "sha256": fingerprint(selected),
                        "source_group": source_group, "split": "public"})
        lineage.append({"name": name, "source": str(source), "source_sha256": source_digest,
                        "source_group": source_group, "original_required_tiers": original.get("required_oracle_tiers"),
                        "original_max_oracle_tier": original.get("max_oracle_tier"),
                        "original_scheduler_predecessor": original.get("extends"),
                        "selected_required_tiers": S.POLICY["tiers"],
                        "numeric_policy_unchanged": cap.get("numeric_policy") == original.get("numeric_policy"),
                        "source_bytes_unchanged": sha256_file(source / ir) == source_group,
                        "original_performance_policy_retained_privately": original.get("performance"),
                        "placement": screen})
    provenance = destination / "provenance.json"
    provenance.write_text(json.dumps({"schema": "merlin.corpus.scalability_intake.v1", "target": target,
        "role_transition": "new_epoch_public_development", "historical_holdout_claim": False,
        "scope": "functional source contracts; neither executable success nor physical timing qualification",
        "evidence_manifest": {"path": str(evidence_root / "evidence-manifest.json"),
                              "sha256": sha256_file(evidence_root / "evidence-manifest.json")},
        "members": lineage}, indent=2) + "\n")
    selection = destination / "suite.json"
    selection.write_text(json.dumps({"schema": S.SCHEMA, "target": target, "origin": "operator_requested",
        "policy": S.POLICY, "members": members,
        "provenance": {"path": str(provenance), "sha256": sha256_file(provenance)}}, indent=2) + "\n")
    S.read(selection, target=target)
    return selection
