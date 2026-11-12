"""Validate diagnostic requests, select declared inputs and render bounded evidence replies."""

from __future__ import annotations

import os
from pathlib import Path

SCHEMA = "merlin.candidate_verification.v1"


_MAX_REPLY = 48 << 10


_NOT_ESTABLISHED = (
    "dynamic_issue_order",
    "hardware_execution",
    "performance",
    "semantic_equivalence",
    "behaviour_on_unlisted_shapes",
)


def public_capsules_root(descriptor):
    """The selected descriptor's public capsule set (declared shapes); None when it cannot be derived."""
    if descriptor is None:
        return None
    try:
        from merlin.development.targetgen.target_experiment import load_target_experiment
        from merlin.experiments.corpus.admission import public_capsules_for

        return public_capsules_for(load_target_experiment(descriptor))
    except Exception:  # noqa: BLE001 -- declared shapes then stay unknown, never guessed
        return None


def validate(value):
    """Shape-check one verify request before any bytes are read."""
    allowed = {
        "operation",
        "path",
        "candidate",
        "mechanism",
        "sweep",
        "verdicts",
        "format",
        "offset",
        "limit",
        "symbol",
        "entry",
        "member",
        "baseline",
    }
    if not isinstance(value, dict) or set(value) - allowed or value.get("operation") != "verify":
        raise ValueError("invalid_analysis_request")
    if value.get("entry") or value.get("symbol") or value.get("member"):
        raise ValueError("verify_requires_owned_path")
    other = value.get("candidate")
    if not isinstance(other, dict) or set(other) - {"path", "format", "symbol", "offset"}:
        raise ValueError("invalid_analysis_request")
    if not isinstance(other.get("symbol", ""), str) or len(other.get("symbol", "")) > 512:
        raise ValueError("invalid_analysis_request")
    for key in ("sweep", "verdicts"):
        items = value.get(key, [])
        if not isinstance(items, list) or len(items) > 32 or any(not isinstance(i, str) for i in items):
            raise ValueError("invalid_analysis_request")
    if not isinstance(value.get("mechanism", ""), str) or len(value.get("mechanism", "")) > 128:
        raise ValueError("invalid_analysis_request")
    if "baseline" in value and (not isinstance(value["baseline"], str) or not value["baseline"]):
        raise ValueError("invalid_analysis_request")


def _contract_document(contract, descriptor):
    """The selected target contract, resolved as the operand decoder resolves it (``{}`` when none)."""
    import yaml

    if descriptor is not None:
        from merlin.development.targetgen.target_experiment import load_target_experiment, selected_experiment_contract

        contract = selected_experiment_contract(load_target_experiment(descriptor), environment=os.environ)
    return (yaml.safe_load(Path(contract).read_bytes()) or {}) if contract is not None else {}


def _raised(flags):
    return sorted(k for k, v in flags.items() if isinstance(v, dict) and v.get("status") is True)


def _summary(report):
    """A bounded reply: verdicts and compact deltas; complete records stay in the detail."""

    def pair(row):
        evidence = row["intervention_applied"].get("evidence") or {}
        effect = {
            side: (evidence.get(side) or {}).get("status") for side in ("before", "after") if "mechanism" in evidence
        }
        return dict(
            capsule=row["capsule"],
            numerically_correct=row["numerically_correct"]["status"],
            **(dict(mechanism_effect_status=effect) if effect else {}),
            intervention_applied={k: row["intervention_applied"].get(k) for k in ("status", "reason")},
            traffic_reduced=row["traffic_reduced"],
            storage_valid={k: row["storage_valid"][k] for k in ("before", "after")},
            loads_per_compute=row["traffic_delta"]["loads_per_compute"],
            dynamic_delta={role: d["dynamic_delta"] for role, d in row["traffic_delta"]["per_role"].items()},
            fell_back_to_seed_or_before=row["fell_back_to_seed_or_before"]["status"],
            regressions={k: v["status"] for k, v in row["regressions"].items() if isinstance(v, dict)},
            **(
                {
                    "regressions_vs_baseline": {
                        k: v["status"] for k, v in row["against_baseline"]["regressions"].items() if isinstance(v, dict)
                    }
                }
                if "against_baseline" in row
                else {}
            ),
        )

    out = dict(
        kind=report["kind"],
        mechanism=report["mechanism"],
        pairs=[pair(r) for r in report["pairs"][:8]],
        n_pairs=len(report["pairs"]),
        not_established=report["not_established"],
    )
    if "sweep" in report:
        region = report["sweep"]["applicability_region"]
        out["sweep"] = dict(
            counts=region["counts"],
            extrapolation=region["extrapolation"],
            identical_emitted_code={
                k: region["identical_emitted_code"][k] for k in ("counts", "observed_values_by_axis")
            },
            rows=[
                dict(
                    capsule=r["capsule"],
                    correct=r["numerically_correct"]["status"],
                    applied=r["intervention_applied"]["status"],
                    traffic=r["traffic_reduced"]["status"],
                    why=r["intervention_applied"].get("reason"),
                    fell_back_to_seed_or_before=r["fell_back_to_seed_or_before"]["status"],
                    **(
                        {"regressed_vs_baseline": _raised(r["against_baseline"]["regressions"])}
                        if "against_baseline" in r
                        else {}
                    ),
                    blockers=sorted((r.get("blockers") or {}).keys()),
                )
                for r in report["sweep"]["rows"][:48]
            ],
        )
    return out


def bank_decoding(workspace, bundle_manifest, repo, pin):
    """``(contract, facts)`` of the selected CCA bank's own pinned decoder, or ``None`` for older banks.

    Applicability decodes the candidate with the bank's decoder; verify uses the same one so both
    endpoints describe an author's build under one decoder identity."""
    from merlin.development.backend.feedback import reference_bank

    if pin is None or bundle_manifest is None or repo is None:
        return None
    root, bank = reference_bank.open_bank(Path(workspace), bundle_manifest, repo, pin)
    decoder = reference_bank._bank_decoder(root, bank)
    return None if decoder is None else (decoder["contract"], decoder["facts"])


GAPS = {
    "verify_requires_owned_path": "verify reads only your own submission/, selfcheck_out/, artifacts/ or "
    "variants/<name>/ paths: a selfcheck_out/<capsule> case, a selfcheck_out/requests/<id> directory, "
    "or a lowered.llvm.mlir/command_buffer.json file.",
    "verify_input_missing": "The named owned path does not exist.",
    "verify_no_selfcheck_case_artifacts": "No lowered.llvm.mlir or command_buffer.json case directories found.",
    "verify_before_after_share_no_capsule": "Pair one case with one case, or two request directories that "
    "share capsule names.",
    "verify_mechanism_not_a_derived_effect": "Use a derived-effect name listed in accepted_mechanisms.",
    "verify_emitted_ir_unparseable": "The emitted lowered.llvm.mlir does not parse; rebuild it with selfcheck.",
    "verify_case_bound_exceeded": "At most 128 cases per directory.",
    "verify_requires_cca_bank_arm": "verify is served by the CCA bank arm's analysis service only.",
}


# Preserve established callable pickle identities through the explicit public exports.
for _export in (
    public_capsules_root,
    validate,
    _contract_document,
    _raised,
    _summary,
    bank_decoding,
):
    _export.__module__ = "merlin.development.backend.feedback.candidate_verification"
