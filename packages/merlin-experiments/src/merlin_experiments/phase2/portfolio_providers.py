"""Portfolio provider assembly from explicit selected-backend capabilities."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from merlin.perf.controlled_context_provider import ControlledSourcePrefixProvider
from merlin.perf.host_physical_transition_qualifier import (
    ChangedRegionQualifierDispatch,
    HostPhysicalTransitionQualifier,
)
from merlin.perf.host_region_qualifier import HostChangedRegionQualifier
from merlin.perf.isolated_probe_provider import IsolatedPrimitiveProbeProvider
from merlin.perf.lane_migration_qualifier import LaneMigrationContractionQualifier
from merlin.perf.paired_context_provider import PairedControlledContextProvider
from merlin.perf.source_program_pair_provider import SourceProgramPairProvider


@dataclass(frozen=True)
class PortfolioProviders:
    provider: IsolatedPrimitiveProbeProvider | None
    semantic_provider: ChangedRegionQualifierDispatch | None
    context_provider: ControlledSourcePrefixProvider | None
    paired_context_provider: PairedControlledContextProvider | None
    source_pair_provider: SourceProgramPairProvider | None


def _capability(backend: ModuleType, name: str, methods: tuple[str, ...]) -> ModuleType | None:
    missing = object()
    capability = getattr(backend, name, missing)
    if capability is missing:
        return None
    if not isinstance(capability, ModuleType) or any(
        not callable(getattr(capability, method, None)) for method in methods
    ):
        raise ValueError(f"selected backend has malformed {name} capability")
    return capability


def assemble(
    *,
    target: str,
    backend: ModuleType,
    output: Path,
    semantic_only: bool,
    probe_interface: Path | None,
    probe_runtime_receipt: Path | None,
    profile_counters: bool,
) -> PortfolioProviders:
    """Preserve scientific provider order without inferring target-owned module names.

    Missing optional declarations leave providers unavailable. Malformed declarations
    and failures loading a declared capability are errors, never fallback discovery.
    """
    provider = semantic_provider = context_provider = paired_context_provider = source_pair_provider = None
    if not semantic_only and all(
        callable(getattr(backend, name, None))
        for name in ("short_program_environment", "prepare_short_program_build", "prepare_short_program_execution")
    ):
        source_pair_provider = SourceProgramPairProvider(
            target=target, adapter=backend, output=output / "source_pair_runtime"
        )
    abi = _capability(backend, "host_witness_abi", ("derive_native_witness_abi",))
    if abi is not None:
        native_abi = abi.derive_native_witness_abi(target=target)
        lane_migration = None
        if source_pair_provider is not None:
            lane_migration = LaneMigrationContractionQualifier(
                target=target,
                runtime_provider=source_pair_provider,
                abi_provenance=native_abi["abi_provenance"],
                output=output / "lane_migration_witnesses",
            )
        semantic_provider = ChangedRegionQualifierDispatch(
            physical=HostPhysicalTransitionQualifier(**native_abi, output=output / "physical_transition_witnesses"),
            legacy=HostChangedRegionQualifier(**native_abi, output=output / "semantic_witnesses"),
            lane_migration=lane_migration,
        )
    adapter = _capability(
        backend,
        "primitive_probe",
        ("prepare_primitive_probe", "execute_prepared_primitive", "isolated_primitive_signature", "runtime_elf_digest"),
    )
    if adapter is not None:
        if not isinstance(getattr(adapter, "__file__", None), str) or not adapter.__file__:
            raise ValueError("selected backend primitive_probe lacks module source provenance")
        if "include_operand_movement" in inspect.signature(adapter.prepare_primitive_probe).parameters:
            context_provider = ControlledSourcePrefixProvider(
                target=target, adapter=adapter, output=output / "controlled_contexts"
            )
            paired_context_provider = PairedControlledContextProvider(
                target=target, adapter=adapter, output=output / "paired_contexts"
            )
    if probe_interface:
        if adapter is None:
            raise ValueError("explicit isolated probe requires selected backend primitive_probe capability")
        provider = IsolatedPrimitiveProbeProvider(
            target=target,
            short_interface=probe_interface,
            adapter=adapter,
            runtime_receipt=probe_runtime_receipt,
            output=output / "isolated_probes",
            profile_counters=profile_counters,
        )
    return PortfolioProviders(
        provider, semantic_provider, context_provider, paired_context_provider, source_pair_provider
    )
