"""Select submitted model compilation from frozen placement and fresh original source eligibility."""


def owned_model_required(capsule, *, target):
    """Select a compiler route, granting no execution or numerical evidence.

    A whole-program capsule whose frozen Phase 0 placement gives the author accelerator work always
    takes the owned-host route: Merlin compiles its host part, and the route fails closed if the
    placement is unbound or disagrees with the build. It never falls back to the legacy graph.
    """
    from .frozen_placement import requires_merlin_host

    if requires_merlin_host(capsule):
        return True
    if bool((capsule.get("semantic") or {}).get("must_accelerate")):
        return True
    if capsule.get("kind") != "model" and "on_mesh" not in ((capsule.get("lanes") or {}).get("require") or []):
        return False
    from .native_model_execution import independent_frozen_source_eligibility

    try:
        certificate = independent_frozen_source_eligibility(capsule, target=target)
    except Exception:
        return False
    eligible = certificate.get("n_eligible")
    return type(eligible) is int and eligible > 0
