"""Select submitted model compilation from declared lanes and fresh source eligibility."""


def owned_model_required(capsule, *, target):
    """Select a compiler route, granting no execution or numerical evidence."""
    if bool((capsule.get("semantic") or {}).get("must_accelerate")):
        return True
    if "on_mesh" not in ((capsule.get("lanes") or {}).get("require") or []):
        return False
    from .native_model_execution import independent_frozen_source_eligibility

    try:
        certificate = independent_frozen_source_eligibility(capsule, target=target)
    except Exception:
        return False
    eligible = certificate.get("n_eligible")
    return type(eligible) is int and eligible > 0
