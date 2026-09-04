"""Compute-unit-KIND registry — the routing axis that keeps core generation target-name-free.

A target's compute-unit ``kind`` (the canonical taxonomy ``merlin.targetgen.compute_units.KINDS``)
selects the GENERATION defaults: which codegen endpoint, whether an op->``.insn`` encoding derivation
applies, which trace-gate plugin (if any), the RTL grading tiers, and the perf metrics. Core modules
consult this registry by ``kind`` so they NEVER branch on a target *name* — the overfit smell test.

Everything here is a DATA default. The per-target capability manifest (the human-reviewed cache derived
from RTL facts + the designer's docs) may override any field; this registry only supplies the family
baseline so a brand-new accelerator of a known kind brings up with zero per-target code.

⚠️ ``kind`` IS A SET, PER TARGET — and most of this module reads only one of them. A target declares
several compute units (a SIMT cluster containing a matrix PE; an NPU with a matrix array AND a vector
engine), and ``contract_endpoint_kind`` below collapses that set to the PRIMARY kind. That is the right
question for "which codegen endpoint does the outermost unit want", and the WRONG question for anything
describing what the silicon can do: measured on the atlas corpus, 62% of its engine-driving expert
kernels touch a vector engine its contract does not even declare. Before adding a caller that keys on
the primary kind, check whether the honest answer is the set — see :mod:`merlin.kernels.engines`, which
maps each kind to the CCA facet that describes it and derives the coarse
``runtime.backends.base.TargetClass`` from the whole set. That module also holds the KINDS -> TargetClass
correspondence this docstring used to assert in prose (five kinds cannot "align with" three tokens by
inspection); it is a checked map now, not a comment.
"""
from __future__ import annotations

from dataclasses import dataclass

from merlin.targetgen.compute_units import KINDS

# Codegen endpoints, in elegance order (see memory: no-forked-toolchain-bringup). The default is the
# fork-free path — the target's own dialect lowered to ``llvm.inline_asm``/``.insn`` on STOCK LLVM.
# ``command_buffer`` is the ISA-less endpoint: the accelerator has no command ISA at all (it is driven
# by a command buffer over one-hot control ports, e.g. a spatial tensor tile), so the 4th artifact IS
# the schema-valid command buffer the target's runtime consumes — no ``.insn`` assembly.
ENDPOINT_KINDS: tuple[str, ...] = ("inline_asm_insn", "upstream_target", "external_backend",
                                   "command_buffer")

# The single NEUTRAL example target the onboarding synthesizers calibrate against (see design
# target_agnostic_core T11: keep exactly one toy_npu example). It is NOT hardware overfit — it seeds the
# FAMILY-DEFAULT plans (contract / runtime adapter / zephyr / llvm) that a brand-new target inherits
# before its own contract + RTL facts refine them. Synthesizers key on THIS constant + the contract's
# family/features/endpoint, never on a specific hardware name (gemmini/saturn/…).
DEFAULT_EXAMPLE_TARGET = "toy_npu"


@dataclass(frozen=True)
class FamilyProfile:
    """Generation defaults for one compute-unit kind (all overridable by a capability manifest)."""
    kind: str
    endpoint_kind_default: str          # ENDPOINT_KINDS[*] — how the 4th artifact is produced
    encoding_required: bool             # does the op->.insn encoding derivation + trace gate apply?
    trace_gate: str | None              # the trace-gate plugin name (e.g. "rocc_insn") or None
    default_rtl_tiers: tuple[str, ...]  # RTL grading tiers for this kind
    perf_fields: tuple[str, ...]        # perf metrics the runner extracts (empty -> cycles only)
    fact_extractor: str                 # RTL fact-extraction family this kind routes to (see mlc_bridge
                                        # .fact_bundle_for): "circt_static" (the decoder/HW-dialect static
                                        # bundle + the generic no-arc fallback), "simt_config" (the SIMT
                                        # config+FIRRTL introspect), or "opu" (the spatial tensor-tile
                                        # state-manifest introspect). Never a target *name* — a routing key.
    #: HOW this kind's COMPUTE ELEMENT is located in an elaborated design — the routing axis the
    #: datapath reader (:mod:`merlin.targetgen.rtl.datapaths`) uses instead of a per-target branch:
    #:
    #:   ``array_element``    the replicated cell of a discovered compute array (``arrays[*].element``);
    #:   ``lane_replication`` the module replicated once per declared lane;
    #:   ``none``             this kind has no replicated compute element, so a cell-geometry read
    #:                        cannot reach its datapath and says so rather than guessing.
    #:
    #: The default is deliberately the honest one: a kind added without stating this reports "not
    #: locatable" rather than being handed the wrong module and publishing its widths as a datapath.
    compute_element: str = "none"


# Fork-free ``.insn`` on stock LLVM is the default wherever the accelerator exposes a command ISA
# (systolic MXUs; a SIMT core's op-level ISA is also pure .insn+CSR — see no-forked-toolchain-bringup).
# vector/scalar lower through UPSTREAM LLVM targets (RVV/base RISC-V), so no per-target encoding.
_PROFILES: dict[str, FamilyProfile] = {
    "systolic": FamilyProfile("systolic", "inline_asm_insn", True, "rocc_insn", ("L3", "L4", "L5"), (),
                              "circt_static", compute_element="array_element"),
    "simt":     FamilyProfile("simt", "inline_asm_insn", False, None, ("L3",),
                              ("flops", "gflops", "pct_fp_peak"), "simt_config",
                              compute_element="lane_replication"),
    "vector":   FamilyProfile("vector", "upstream_target", False, None, (), (), "circt_static",
                              compute_element="lane_replication"),
    "scalar":   FamilyProfile("scalar", "upstream_target", False, None, (), (), "circt_static"),
    # A scalar pipe has no replicated compute element: its datapath is the register file's width, which
    # a cell-geometry read cannot reach -- so `compute_element` stays the default "none" and the reader
    # reports that instead of naming whichever module happens to be widest.
    # Spatial tensor tile (Saturn OuterProductUnit family): a grid of accumulator cells driven by a
    # COMMAND BUFFER over one-hot op ports (macc/mvin/shift) — NOT a RoCC command ISA. So there is no
    # op->``.insn`` encoding to derive (encoding_required=False) and no rocc_insn trace gate; the 4th
    # artifact IS the schema-valid command buffer (command_buffer endpoint). Its facts come from the OPU
    # state-manifest geometry (cluster x cell tile, MRF depth, int8/fp8 datapaths) via the "opu" fact
    # extractor. Perf is tensor-tile MAC throughput. (See memory: opu-endpoint-is-command-buffer-not-rocc.)
    "spatial":  FamilyProfile("spatial", "command_buffer", False, None, ("L3", "L4", "L5"),
                              ("macs", "mac_per_cycle", "pct_mac_peak"), "opu",
                              compute_element="array_element"),
}


def family_profile(kind: str) -> FamilyProfile:
    """The generation defaults for a compute-unit ``kind`` (raises for an unknown kind — fail-closed)."""
    try:
        return _PROFILES[kind]
    except KeyError:
        raise KeyError(f"unknown compute-unit kind {kind!r}; known: {sorted(_PROFILES)}") from None


def known_kinds() -> tuple[str, ...]:
    return tuple(sorted(_PROFILES))


def contract_endpoint_kind(contract: dict) -> str | None:
    """The codegen endpoint the contract's PRIMARY compute-unit family selects, or ``None`` when the
    contract declares no ``compute_units`` (the neutral reference examples / skeleton contracts). Keyed
    on the compute-unit ``kind`` via :func:`family_profile` — never on a target name. Onboarding
    synthesizers use it to route the fork/adapter posture by family instead of ``if name==``."""
    from .compute_units import compute_units
    from .target_experiment import _primary_kind
    units = compute_units(contract)
    if not units:
        return None
    return family_profile(_primary_kind(units)).endpoint_kind_default
