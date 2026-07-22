"""Generic, DERIVATION-DRIVEN target backend — the target-agnostic replacement for per-target plugins.

The hard rule (see the derive-dont-overfit memory / repo convention): everything target-specific is
DERIVED from mlc's RTL discovery, never hand-written per target. Given a ``target``, this module reads
mlc's discovery — the legal opcode set (the ISA, from the decoder's ``comb.icmp`` fan-out), the memory
map (operand/accumulator banks), and the mesh DIM — and derives the compiler-modification surface: the
structural levers the discovered hardware IMPLIES, routed to the target's own (OOT) codegen seams.

A new accelerator plugs in by being registered with mlc (its RTL → firtool/arcilator → discovery); no
new Python here. Anything genuinely not derivable is the rare EXCEPTION and belongs in a declarative
per-target artifact (YAML/MLIR) or a tool parameter, never in this module or the agnostic core.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TargetProfile:
    """What the generic backend needs, entirely DERIVED from mlc RTL discovery (no hand facts)."""
    target: str
    legal_opcodes: tuple[int, ...] | None   # the ISA the decoder actually matches
    memory_map: dict | None                 # operand/accumulator bank handles + row bytes
    dim: int | None                         # systolic mesh DIM (None if the target has no mesh)

    @property
    def has_mesh(self) -> bool:
        return self.dim is not None

    @property
    def has_accumulator(self) -> bool:
        return bool(self.memory_map and self.memory_map.get("accum_mem"))


def target_profile(target: str) -> TargetProfile:
    """Derive the target's profile from mlc discovery. Fields are None when mlc / the artifact is
    unavailable — the caller degrades honestly (never fabricates a fact)."""
    from .rtl import mlc_bridge
    ops = mlc_bridge.discover_legal_opcodes(target) if mlc_bridge.mlc_available()[0] else {}
    return TargetProfile(
        target=target,
        legal_opcodes=tuple(ops.get("legal_opcodes") or ()) or None,
        memory_map=mlc_bridge.discovered_memory_map(target),
        dim=mlc_bridge.discovered_dim(target),
    )


def derived_levers(profile: TargetProfile) -> list[str]:
    """The structural compiler levers the DISCOVERED hardware implies — derived, never hand-listed.

    A systolic mesh implies a dataflow choice (WS/OS); an accumulator memory implies an
    accumulator-residency choice. Targets without that structure simply don't expose those levers. This
    is how the CCA/route surface stays target-agnostic: the levers come from what the RTL has."""
    levers: list[str] = []
    if profile.has_mesh:
        levers.append("spatial.dataflow")
    if profile.has_accumulator:
        levers.append("spatial.accumulator_resident")
    return levers
