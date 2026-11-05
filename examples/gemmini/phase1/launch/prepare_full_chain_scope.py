"""Prepare private full-session bindings without executing or admitting a model.

This emits a blocked operator proposal, never a native qualification receipt or
an activatable numerical selection. Array contents and compiler packages are not
opened. Existing reference digest declarations remain declarations until the
ordinary final verifier reopens their payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml
from merlin_experiments.phase1.feedback.private_numerical_selection import MODE, POLICY


def pin(path):
    path = Path(path).absolute()
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f"missing or indirect metadata input: {path}")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def prepare(private_spec):
    spec_pin = pin(private_spec)
    spec = yaml.safe_load(Path(private_spec).read_text())
    bundle = Path(spec["numerics_bundle"]["path"])
    refs = {
        k: pin(bundle / name)
        for k, name in (
            ("models", "configuration/models.yaml"),
            ("reference_manifest", "MANIFEST.json"),
            ("reference_sums", "SHA256SUMS"),
        )
    }
    config = yaml.safe_load(Path(refs["models"]["path"]).read_text())
    names = {m["id"] for m in spec["models"]}
    manifest = json.loads(Path(refs["reference_manifest"]["path"]).read_text())
    if names != set(config["models"]) or names != set(manifest["models"]):
        raise ValueError("private spec, numerical configuration and manifest rosters differ")
    # Retain exact pre-existing digest declarations, without reading private arrays.
    payload_pins = []
    for line in Path(refs["reference_sums"]["path"]).read_text().splitlines():
        digest, member = line.split(maxsplit=1)
        member = member.lstrip("*")
        if "/visible/v00/" in member and Path(member).suffix == ".npz":
            payload_pins.append({"member": member, "sha256": digest})
    models, blockers, limitations = [], [], []
    for row in spec["models"]:
        name = row["id"]
        session_path = Path(row["capture"]) / "session_contract.yaml"
        session_pin = pin(session_path)
        session = yaml.safe_load(session_path.read_text())
        numerical = config["models"][name]
        declared_steps = sum(x["steps"] for x in session["stage_schedule"])
        selected_steps = sum(x.get("steps", 1) for x in numerical["programs"])
        if declared_steps != selected_steps:
            blockers.append(
                {
                    "model": name,
                    "code": "session_reference_scope_mismatch",
                    "declared_steps": declared_steps,
                    "selected_reference_steps": selected_steps,
                    "required_artifact": (
                        "Session-aligned numerical configuration and independently produced "
                        "reference manifest/sums; do not edit step counts alone."
                    ),
                }
            )
        if session.get("paper_ready") is False:
            limitations.append(
                {
                    "model": name,
                    "code": "declared_not_paper_ready",
                    "claim_scope": (
                        "Preserve declared synthetic or provisional input provenance. "
                        "Numerical execution can proceed within that scope; it does not "
                        "establish real-input model quality or paper readiness."
                    ),
                }
            )
        models.append(
            {
                "id": name,
                "deployment_dtype": row["deployment_dtype"],
                "capture_tree_sha256": row["capture_tree_sha256"],
                "session_contract": session_pin,
                "session": {
                    k: session[k]
                    for k in (
                        "version",
                        "kind",
                        "steps",
                        "stage_schedule",
                        "programs",
                        "bindings",
                        "states",
                        "streams",
                        "parameters",
                        "paper_ready",
                    )
                    if k in session
                },
                "selected_input_bindings": row["selected_input_bindings"],
                "input_provenance": row.get("input_provenance"),
                "capture_selection": {"path": row["capture_selection"], "sha256": row["capture_selection_sha256"]},
                "capture_execution_attestation": {
                    "path": row["capture_execution_attestation"],
                    "sha256": row["capture_execution_attestation_sha256"],
                },
                "numerical_chain": numerical["chain"],
                "numerical_programs": numerical["programs"],
                "required_output_scope": (
                    "Every declared output of every executed program at every session step, "
                    "including recurrent state."
                ),
            }
        )
    blockers.extend(
        [
            {
                "code": "all_output_producer_not_selected",
                "required_artifact": (
                    "Pinned merlin.operator.all_output_plan.v1 with each model's admitted "
                    "full_declared_output_reference.v1 reference.json, SHA256SUMS and every-step "
                    "output payloads matching these exact session contracts."
                ),
            },
            {
                "code": "native_external_launch_not_selected",
                "required_artifact": (
                    "Current source/runtime-bound declarative launch_config, infrastructure "
                    "readiness receipt, private numerical_qualification "
                    "config/static_root/deadline; --validation-scope full alone is insufficient."
                ),
            },
        ]
    )
    return {
        "schema": "merlin.operator.full_chain_scope_proposal.v1",
        "status": "blocked_not_selected",
        "execution_performed": False,
        "author_visible": False,
        "private_spec": spec_pin,
        "reference_metadata": refs,
        "reference_payload_declarations": payload_pins,
        "payload_bytes_verified": False,
        "models": models,
        "criterion_preview": {
            "mode": MODE,
            **refs,
            "comparison": "quantized-reference",
            "output_scope": "all_declared",
            "float_policy": POLICY,
            "kernel_exact": True,
        },
        "blockers": blockers,
        "limitations": limitations,
        "deferred_until_compiler_freeze": [
            "Capsule qualification/handoff pin selecting the authored compiler package from the empty-backend run.",
            (
                "Native static-index.json and original per-program build receipts bound to that "
                "exact candidate hash, source, inputs and runtime."
            ),
            (
                "Full-chain reports and all-output audit receipts, then full-chain-"
                "qualification.json from the existing supervisor."
            ),
        ],
        "candidate_boundary": (
            "Only the frozen authored compiler supplies candidate lowering/support and ELF "
            "artifacts. Retained/private reference compilers are not candidate substitutes or "
            "author grants."
        ),
        "activation": (
            "Resolve scope/reference blockers first; use external_validation.py after capsule "
            "qualification under a separate finite budget. Never feed this proposal to native "
            "qualification as a receipt."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.private_spec)
    args.out.mkdir(parents=True, mode=0o700, exist_ok=False)
    path = args.out / "full-chain-scope.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    path.chmod(0o600)
    print(
        json.dumps(
            {
                "status": result["status"],
                "models": len(result["models"]),
                "blockers": len(result["blockers"]),
                "proposal": str(path),
            }
        )
    )


if __name__ == "__main__":
    main()
