"""Check evidence coverage: are contract claims backed by source evidence?"""

from __future__ import annotations

from pathlib import Path


def run(run_dir: Path, manifest: dict) -> dict:
    metrics: dict = {
        "validator": "evidence",
        "evidence_coverage": None,
        "unsupported_claim_rate": None,
        "contract_files_found": [],
        "errors": [],
    }

    contracts_dir = run_dir / "contracts"
    expected = ["target_contract.yaml", "dialect_plan.yaml",
                "lowering_plan.yaml", "runtime_adapter_plan.yaml"]

    found = [f for f in expected if (contracts_dir / f).exists()]
    metrics["contract_files_found"] = found

    if not found:
        metrics["errors"].append(
            "No contract files found under contracts/; "
            "evidence coverage cannot be computed for an empty run"
        )
        return metrics

    import yaml
    total_claims = 0
    evidenced_claims = 0
    unsupported_claims = 0

    for fname in found:
        try:
            with open(contracts_dir / fname) as f:
                doc = yaml.safe_load(f) or {}
            claims = doc.get("claims", [])
            total_claims += len(claims)
            evidenced_claims += sum(1 for c in claims if c.get("evidence"))
            unsupported_claims += sum(1 for c in claims if c.get("unsupported"))
        except Exception as e:
            metrics["errors"].append(f"{fname}: {e}")

    if total_claims > 0:
        metrics["evidence_coverage"] = round(evidenced_claims / total_claims, 3)
        metrics["unsupported_claim_rate"] = round(unsupported_claims / total_claims, 3)

    return metrics
