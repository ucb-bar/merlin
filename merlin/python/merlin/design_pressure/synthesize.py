"""Contract synthesis: pressure vector -> legal software-visible contracts.

Consumes a Region Pressure Vector and the mined policies (``output/kernels/policy_rules.yaml``)
and decides which interface contracts are legal. The mined policies are the legality
verifiers — evaluated via ``merlin.kernels.policy.evaluate_when`` — so thresholds are never
re-invented here.

Design decision (structural legality vs policy endorsement): ``resident_packed_tensor`` is
*structurally legal* when its operand pressure holds (reuse>=2, immutable, fits resident
store). The mined ``packed_rhs_policy`` additionally requires ``K>=256``; that condition is
evaluated separately as *policy endorsement* (reported, not gating). This keeps the canonical
``repeated_rhs_matmul`` benchmark (K=128) recommending residency while still letting the VLA
experiment sweep K to show the legal->endorsed transition at K=256.

``accumulator_commit`` is independent of residency: it is legal whenever its mined policy fires
(a contraction with a live accumulator across a fused epilogue), regardless of weight reuse.
"""
from __future__ import annotations

from pathlib import Path

from merlin.common import paths
from merlin.common.yaml import load_yaml
from merlin.kernels import policy

FEATURE_RESIDENT = "resident_packed_tensor"
FEATURE_ACCUMULATOR = "accumulator_commit"

# The I0–I3 interface ladder for the cost experiment. ``features`` lists the mined
# abstractions a contract exposes; ``requires`` lists the legality predicates it needs.
CONTRACTS = [
    {
        "id": "I0", "name": "opaque_call",
        "interface_ops": ["opaque_call"],
        "interface_types": ["transient_tensor"],
        "features": [], "requires": [],
    },
    {
        "id": "I1", "name": "explicit_scratchpad_dma",
        "interface_ops": ["dma_load", "compute", "dma_store"],
        "interface_types": ["scratchpad_view", "transient_tensor"],
        "features": [], "requires": [],
    },
    {
        "id": "I2", "name": "resident_packed_tensor",
        "interface_ops": ["resident_pack", "resident_matmul", "evict"],
        "interface_types": ["resident_packed_tensor"],
        "features": [FEATURE_RESIDENT], "requires": ["resident"],
    },
    {
        "id": "I3", "name": "resident_packed_tensor+accumulator_commit",
        "interface_ops": ["resident_pack", "resident_matmul", "commit_epilogue", "evict"],
        "interface_types": ["resident_packed_tensor", "accumulator"],
        "features": [FEATURE_RESIDENT, FEATURE_ACCUMULATOR],
        "requires": ["resident", "accumulator"],
    },
]


def load_policies(path: str | Path | None = None) -> list[dict]:
    """Load the mined policy rules (defaults to ``output/kernels/policy_rules.yaml``)."""
    p = Path(path) if path else paths.artifacts_dir() / "kernel-index" / "policy_rules.yaml"
    data = load_yaml(p)
    return list(data or [])


def _policy(policies: list[dict], name: str) -> dict | None:
    return next((r for r in policies if r.get("policy") == name), None)


def _resident_structural(rpv: dict, policies: list[dict],
                         resident_store_bytes: int | None) -> tuple[bool, list[str]]:
    """Structural legality of resident_packed_tensor: operand pressure + capacity.

    Evaluates ``packed_rhs_policy.when`` against facts with ``K`` removed (K>=256 is treated
    as endorsement, not structural legality).
    """
    rule = _policy(policies, "packed_rhs_policy")
    blocked: list[str] = []
    if rule is None:
        return False, ["no packed_rhs_policy"]
    facts = dict(rpv["facts"])
    facts.pop("K", None)
    if not policy.evaluate_when(rule["when"], facts):
        blocked.append("reuse/immutability not met")
        return False, blocked
    # Capacity: distinct resident weights must fit in resident storage, when one is given.
    if resident_store_bytes is not None:
        need = int(rpv["metrics"].get("distinct_weights", 1)) * int(
            rpv["metrics"].get("pack_bytes", 0))
        if need > resident_store_bytes:
            blocked.append(
                f"capacity: need {need}B > resident_store {resident_store_bytes}B")
            return False, blocked
    return True, blocked


def _resident_endorsed(rpv: dict, policies: list[dict]) -> bool:
    """Full mined-policy endorsement of resident_packed_tensor (includes K>=256)."""
    rule = _policy(policies, "packed_rhs_policy")
    if rule is None:
        return False
    return policy.evaluate_when(rule["when"], rpv["facts"])


def _accumulator_legal(rpv: dict, policies: list[dict],
                       accumulator_entries: int | None) -> tuple[bool, list[str]]:
    rule = _policy(policies, "accumulator_commit_policy")
    blocked: list[str] = []
    if rule is None:
        return False, ["no accumulator_commit_policy"]
    if not policy.evaluate_when(rule["when"], rpv["facts"]):
        blocked.append("no live accumulator across epilogue")
        return False, blocked
    if accumulator_entries is not None:
        mn = (rpv["metrics"].get("M") or 0) * (rpv["metrics"].get("N") or 0)
        if mn > accumulator_entries:
            blocked.append(f"accumulator fit: M*N {mn} > entries {accumulator_entries}")
            return False, blocked
    return True, blocked


def legal_contracts(rpv: dict, policies: list[dict],
                    resident_store_bytes: int | None = None,
                    accumulator_entries: int | None = None) -> list[dict]:
    """Return the I0–I3 ladder annotated with legality, endorsement and justification."""
    res_legal, res_block = _resident_structural(rpv, policies, resident_store_bytes)
    res_endorsed = _resident_endorsed(rpv, policies)
    acc_legal, acc_block = _accumulator_legal(rpv, policies, accumulator_entries)

    out: list[dict] = []
    for c in CONTRACTS:
        legal = True
        blocked: list[str] = []
        justifying: list[str] = []
        if "resident" in c["requires"]:
            legal = legal and res_legal
            blocked += res_block
            if res_legal:
                justifying.append("packed_rhs_policy")
        if "accumulator" in c["requires"]:
            legal = legal and acc_legal
            blocked += acc_block
            if acc_legal:
                justifying.append("accumulator_commit_policy")
        entry = {
            "id": c["id"],
            "name": c["name"],
            "legal": legal,
            "interface_ops": list(c["interface_ops"]),
            "interface_types": list(c["interface_types"]),
            "features": list(c["features"]),
            "justified_by": {"policies": justifying},
            "blocked_by": sorted(set(blocked)),
        }
        if "resident" in c["requires"]:
            entry["policy_endorsed"] = res_endorsed
        out.append(entry)
    return out


def recommended_features(rpv: dict, policies: list[dict],
                         resident_store_bytes: int | None = None) -> list[str]:
    """The mined abstraction features that structurally fire for this region.

    Independent (residency and accumulator-commit do not require each other). Returned in a
    canonical, deterministic order so it can be compared to a benchmark's golden.
    """
    feats: list[str] = []
    res_legal, _ = _resident_structural(rpv, policies, resident_store_bytes)
    if res_legal:
        feats.append(FEATURE_RESIDENT)
    acc_legal, _ = _accumulator_legal(rpv, policies, None)
    if acc_legal:
        feats.append(FEATURE_ACCUMULATOR)
    return feats
