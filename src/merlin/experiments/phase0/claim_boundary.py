"""Keep held-out models out of public corpus candidate payloads.

Two classes of model are held out of derivation: the descriptor's claim models (evaluated owner-side
after the Phase-1 compiler is frozen) and the reviewed evaluation-only models (whole models used to
check generality, never to shape the corpus). Both are refused on the same terms.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def _is_model(value: object, model: str) -> bool:
    text = str(value or "")
    return text == model or text.startswith(model + "_")


def is_held_out(value: object, held_out: Iterable[str]) -> str | None:
    """The held-out model ``value`` names (exactly, or as a ``<model>_...`` bundle), or ``None``."""
    matches = [model for model in held_out if _is_model(value, str(model))]
    return max(matches, key=len) if matches else None


def held_out_models(te) -> list[str]:
    """Every model barred from derivation for this experiment: the descriptor's claim roster, its
    declared evaluation-only models, and the reviewed registry's claim and evaluation-only models."""
    from merlin.targetgen import claim_models as CM

    spec = getattr(te, "workload_spec", None) or {}
    declared = [*(spec.get("models") or ()), *(spec.get("evaluation_only_models") or ())]
    return list(dict.fromkeys(str(model) for model in (*declared, *CM.held_out_models())))


# What an objective variant may change about its claim model's capture: the scheme and the capture
# capabilities. Anything else (a loader, a dtype, a recipe) would make it a different workload.
_VARIANT_KEYS = frozenset({"quant_scheme", "quant_activation_contractions", "quant_integer_nonlinear"})


def claim_objective_variants(te) -> list[dict]:
    """Owner-side model entries for the descriptor's declared objective variants of its claim models.

    A variant is a claim network captured under a capture capability -- a different numerical model
    the owner evaluates after the Phase-1 freeze, on the claim model's own terms. Each entry names that
    claim model, so :func:`assert_no_claim_capsules` refuses it in any public payload: a variant is an
    objective, never a derivation source.
    """
    spec = getattr(te, "workload_spec", None) or {}
    declared = spec.get("claim_objective_variants") or {}
    if not isinstance(declared, dict):
        raise ValueError("workload_spec.claim_objective_variants must map a claim model to its variants")
    claims = [str(model) for model in spec.get("models") or ()]
    entries: list[dict] = []
    for model, variants in declared.items():
        if str(model) not in claims:
            raise ValueError(
                f"objective variants declared for {model!r}, which is not a claim model in workload_spec.models "
                f"{claims}; a variant of a non-claim model would be a derivation source"
            )
        if not isinstance(variants, dict) or not variants:
            raise ValueError(f"claim model {model!r} declares no objective variants")
        for name, decl in variants.items():
            decl = dict(decl or {})
            unknown = sorted(set(decl) - _VARIANT_KEYS)
            if unknown:
                raise ValueError(
                    f"objective variant {model}/{name} sets {unknown}; only {sorted(_VARIANT_KEYS)} may vary"
                )
            if decl.get("quant_integer_nonlinear") and not decl.get("quant_activation_contractions"):
                raise ValueError(
                    f"objective variant {model}/{name} asks for integer nonlinears without activation contractions"
                )
            entries.append(
                {
                    "name": f"SY_model_{model}_{name}",
                    "kind": "model",
                    "cat": "model",
                    "op": "model",
                    "model": str(model),
                    "objective_variant": str(name),
                    **decl,
                }
            )
    return entries


def assert_no_claim_capsules(entries: list[dict], claim_models: list[str]) -> None:
    """Reject a held-out model source even if a capsule is renamed or relabelled.

    The selected public profile feeds Phase 1. Held-out models are evaluated by the owner after that
    compiler is frozen, never turned into public examples -- neither as a whole-model capsule nor as
    a form derived from one (a model-form block names its source model).
    """
    for entry in entries:
        model = entry.get("model")
        form_model = (entry.get("model_form") or {}).get("model") if isinstance(entry.get("model_form"), dict) else None
        perf_model = [
            row.get("application")
            for row in (((entry.get("performance") or {}).get("form") or {}).get("members") or [])
            if isinstance(row, dict)
        ]
        loader_parts = Path(str(entry.get("loader") or "")).parts
        name = str(entry.get("name") or "")
        generated_name = name.removeprefix("SY_model_")
        derived_name = name.removeprefix("MF_")
        for claim in claim_models:
            if (
                _is_model(model, claim)
                or _is_model(form_model, claim)
                or any(_is_model(value, claim) for value in perf_model)
                or any(_is_model(part, claim) for part in loader_parts)
                or (name.startswith("SY_model_") and _is_model(generated_name, claim))
                or (name.startswith("MF_") and _is_model(derived_name, claim))
            ):
                raise ValueError(
                    f"public Phase 0 profile contains held-out claim model {claim!r}; "
                    "evaluate it owner-side only after the Phase-1 compiler is frozen"
                )
