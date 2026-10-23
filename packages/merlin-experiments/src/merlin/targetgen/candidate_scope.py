"""Scope the optional source-product compiler transport before package execution."""

from collections.abc import Callable

from merlin.targetgen.core_aten_capsules import is_full_call
from merlin.targetgen.package_runtime import CertFailure
from merlin.targetgen.tier_policy import required_tiers


def with_candidate_transport(adapters: dict[str, Callable], candidate_transport) -> dict[str, Callable]:
    """Bind an explicit compiler transport to every selected source-bundle oracle."""
    if candidate_transport is None:
        return adapters
    bound = {}
    selected = False
    for tier, adapter in adapters.items():
        if not getattr(adapter, "compiles_source_bundle", False):
            bound[tier] = adapter
            continue
        factory = getattr(adapter, "_merlin_with_candidate_transport", None)
        if not callable(factory):
            raise NotImplementedError(f"{tier} oracle cannot consume explicit candidate product transport")
        bound[tier] = factory(candidate_transport)
        selected = True
    if not selected:
        raise NotImplementedError("no oracle can consume explicit candidate product transport")
    return bound


def require_candidate_product_scope(pkg, adapters, *, capsule=None, package_dir=None, contract=None) -> None:
    """Refuse executable paths outside the selected source-product transport."""
    if not any(getattr(a, "_merlin_candidate_transport", None) is not None for a in (adapters or {}).values()):
        return
    if pkg is None:
        from merlin.targetgen.package_runtime import load_package

        pkg = load_package(package_dir, contract=contract)
    if "build" in pkg.manifest:
        raise CertFailure(
            "candidate_compilation",
            "protocol_violation",
            "source-product compilation does not support submitted package build commands",
        )
    if capsule is not None and not is_full_call(capsule):
        raise CertFailure(
            "candidate_compilation",
            "protocol_violation",
            "source-product compilation requires exported full-call capsules",
        )
    if capsule is not None and any(
        getattr((adapters or {}).get(tier), "_merlin_candidate_transport", None) is None
        for tier in required_tiers(capsule) or []
    ):
        raise CertFailure(
            "candidate_compilation",
            "protocol_violation",
            "every required full-call tier must use selected source-product compilation",
        )


def adapter_options(readback_policy, candidate_transport) -> dict:
    """Keep absent choices absent, preserving legacy adapter construction seams."""
    options = {}
    if readback_policy is not None:
        options["readback_policy"] = readback_policy
    if candidate_transport is not None:
        options["candidate_transport"] = candidate_transport
    return options


def checkpoint_adapters(factory, target, sim_via, readback_policy, candidate_transport):
    """Construct the full contract-derived checkpoint ladder with explicit selections."""
    return factory(target, sim_via, **adapter_options(readback_policy, candidate_transport))


def functional_selfcheck(legacy, factory, sim, target, sim_via, **kwargs):
    """Select the supported functional source-product boundary for self-check."""
    candidate_transport = kwargs.pop("candidate_transport", None)
    if candidate_transport is None:
        return legacy(sim, target, sim_via, **kwargs)
    if sim != "spike":
        raise ValueError("candidate catalog compilation is selected for the functional self-check tier")
    return factory(target, sim_via, declared_tiers={"L2"}, candidate_transport=candidate_transport, **kwargs), sim


def with_readback_policy(adapters: dict[str, Callable], readback_policy) -> dict[str, Callable]:
    """Retarget only declared native simulator adapters; never silently ignore policy."""
    from merlin.targetgen.contract.readback_policy import selected

    policy = selected(readback_policy)
    if policy is None:
        return adapters
    if not adapters:
        raise NotImplementedError("no oracle can consume explicit full-value readback policy")
    bound = {}
    for tier, adapter in adapters.items():
        factory = getattr(adapter, "_merlin_with_readback_policy", None)
        if not callable(factory):
            raise NotImplementedError(f"{tier} oracle cannot consume explicit full-value readback policy")
        bound[tier] = factory(policy)
    return bound
