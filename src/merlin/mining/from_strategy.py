"""Compatibility calls for provider-owned vector schedules and optional forks.

RVV schedule rendering, blocking legality and microkernel realization live in
the selected OOT provider. Importing this module registers no target. The
historical RVV defaults select that named installed provider; an explicit support
selection is authoritative, including when unavailable. Scalar loading does not
need this module or the vector provider.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .registry import RvvPackage

_LEGACY_SUPPORT = "rvv"


def _codegen(codegen_provider=None, *, support_target: str | None = None):
    if support_target is not None and (not isinstance(support_target, str) or not support_target.strip()):
        raise ValueError("support_target must explicitly name a provider")
    if codegen_provider is not None and support_target is not None:
        raise ValueError("select a codegen_provider or support_target, not both")
    if codegen_provider is None:
        from merlin.common.paths import env

        if support_target is not None or env("MERLIN_TARGET_PATH"):
            from merlin.targetgen.plugins import load_declared

            codegen_provider = load_declared(support_target or _LEGACY_SUPPORT, "rvv_codegen")
        else:
            from merlin.compiler.backend.service import resolve

            try:
                _, service = resolve(_LEGACY_SUPPORT, group="merlin.backends", methods=("codegen_provider",))
            except ValueError as exc:
                raise RuntimeError(
                    "vector schedule policy requires its OOT package; select codegen_provider/support_target "
                    "or install the named vector backend"
                ) from exc
            codegen_provider = service.codegen_provider()
    if getattr(codegen_provider, "API_VERSION", None) != 1:
        raise ValueError("selected vector codegen provider must implement API_VERSION=1")
    return codegen_provider


def render_schedule(knobs: dict[str, Any], *, codegen_provider=None, support_target=None) -> str:
    """Render through the selected OOT implementation, retaining historical text."""
    provider = _codegen(codegen_provider, support_target=support_target)
    return _checked_call(provider, provider.render_schedule, knobs)


def _rvv_blocking_lowers(MR, NR, M, N, *, codegen_provider=None, support_target=None):
    provider = _codegen(codegen_provider, support_target=support_target)
    return _checked_call(
        provider, provider.blocking_cost_api()._rvv_blocking_lowers, MR, NR, M, N, operation="blocking"
    )


def _rvv_best_block(MR_cap, NR_cap, extents, *, codegen_provider=None, support_target=None):
    provider = _codegen(codegen_provider, support_target=support_target)
    return _checked_call(
        provider, provider.blocking_cost_api()._rvv_best_block, MR_cap, NR_cap, extents, operation="blocking"
    )


def microkernel_features(
    mk: dict[str, Any],
    target: str = _LEGACY_SUPPORT,
    shapes: Any = (),
    *,
    codegen_provider=None,
    support_target=None,
) -> list[str]:
    """Use an explicit provider or a deliberately registered target resolver.

    Custom registered targets retain their generic dispatch. Only the exact
    historical RVV identity has a named legacy OOT selection; arbitrary package
    or SoC names never imply that provider.
    """
    from merlin.kernels.microkernel import MicrokernelSpec, registered_targets, resolve_for_shapes

    if codegen_provider is None and support_target is None:
        if target in registered_targets() or target != _LEGACY_SUPPORT:
            return list(resolve_for_shapes(target, MicrokernelSpec.from_knobs(mk), shapes))
    provider = _codegen(codegen_provider, support_target=support_target)
    return list(_checked_call(provider, provider.microkernel_features, mk, shapes=shapes))


def _register_rvv_microkernel_resolver(*, codegen_provider=None, support_target=None) -> None:
    """Explicit legacy registry opt-in; importing this module has no side effects."""
    _codegen(codegen_provider, support_target=support_target).register_microkernel_resolver()


def policy_source_record(codegen_provider, *, operation=None) -> dict:
    """Snapshot actual selected provider policy bytes for generated fork attribution.

    This record covers provider-owned sources and the adapter, not a complete
    native toolchain or transitive shared-library closure and not qualification.
    """
    provider = _codegen(codegen_provider)
    source_files = getattr(provider, "policy_source_files", None)
    if not callable(source_files):
        raise ValueError("selected vector provider cannot identify its policy source files")
    selected = tuple(source_files(operation=operation)) if operation is not None else tuple(source_files())
    if not selected or any(not isinstance(path, (str, Path)) for path in selected):
        raise ValueError("selected vector provider has no actual policy source closure")
    from merlin.kernels import microkernel

    paths = tuple(dict.fromkeys((Path(__file__), Path(microkernel.__file__), *(Path(path) for path in selected))))
    return {
        "schema": "merlin.vector-policy-sources.v1",
        "members": [
            {"path": str(path.resolve(strict=True)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in paths
        ],
        "complete_dependency_identity": False,
        "scope": "observed_source_snapshot; import_drift_checked; loaded_python_code_not_attested",
    }


def _checked_call(provider, action, *args, operation=None, **kwargs):
    before = policy_source_record(provider, operation=operation)
    result = action(*args, **kwargs)
    if policy_source_record(provider, operation=operation) != before:
        raise ValueError("selected vector policy source changed during its operation")
    return result


def mint_fork(
    parent: RvvPackage | str | Path,
    overrides: dict[str, Any],
    *,
    version: int,
    depth: int,
    timestamp: str,
    source_evidence: list[str],
    lever: str,
    target: str = _LEGACY_SUPPORT,
    out_root: str | Path = "out/artifacts/targets",
    generated_by_agent: bool = False,
    codegen_provider=None,
    support_target=None,
) -> Path:
    """Delegate fork creation to the optional mining workflow."""
    try:
        from .fork import mint_fork as create
    except ModuleNotFoundError as exc:
        if exc.name != f"{__package__}.fork":
            raise
        raise ModuleNotFoundError(
            "Merlin installation lacks fork creation; reinstall 'merlin[mining]'.",
            name=exc.name,
        ) from exc
    return create(
        parent,
        overrides,
        version=version,
        depth=depth,
        timestamp=timestamp,
        source_evidence=source_evidence,
        lever=lever,
        target=target,
        out_root=out_root,
        generated_by_agent=generated_by_agent,
        codegen_provider=codegen_provider,
        support_target=support_target,
    )
