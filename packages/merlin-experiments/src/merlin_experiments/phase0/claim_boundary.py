"""Keep held-out claim models out of public corpus candidate payloads."""

from __future__ import annotations

from pathlib import Path


def _is_model(value: object, model: str) -> bool:
    text = str(value or "")
    return text == model or text.startswith(model + "_")


def assert_no_claim_capsules(entries: list[dict], claim_models: list[str]) -> None:
    """Reject a claim-model source even if a capsule is renamed or relabelled.

    The selected public profile feeds Phase 1. Claim models are evaluated by
    the owner after that compiler is frozen, never turned into public examples.
    """
    for entry in entries:
        model = entry.get("model")
        loader_parts = Path(str(entry.get("loader") or "")).parts
        generated_name = str(entry.get("name") or "").removeprefix("SY_model_")
        for claim in claim_models:
            if (
                _is_model(model, claim)
                or any(_is_model(part, claim) for part in loader_parts)
                or (str(entry.get("name") or "").startswith("SY_model_") and _is_model(generated_name, claim))
            ):
                raise ValueError(
                    f"public Phase 0 profile contains held-out claim model {claim!r}; "
                    "evaluate it owner-side only after the Phase-1 compiler is frozen"
                )
