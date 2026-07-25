"""Compute-unit-KIND registry — the routing axis that keeps core generation target-name-free.

A target's compute-unit ``kind`` (``systolic`` | ``simt`` | ``vector`` | ``scalar`` — the canonical
taxonomy ``merlin.targetgen.compute_units.KINDS``, aligned with ``runtime.backends.base.TargetClass``)
selects the GENERATION defaults: which codegen endpoint, whether an op->``.insn`` encoding derivation
applies, which trace-gate plugin (if any), the RTL grading tiers, and the perf metrics. Core modules
consult this registry by ``kind`` so they NEVER branch on a target *name* — the overfit smell test.

Everything here is a DATA default. The per-target capability manifest (the human-reviewed cache derived
from RTL facts + the designer's docs) may override any field; this registry only supplies the family
baseline so a brand-new accelerator of a known kind brings up with zero per-target code.
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


# Fork-free ``.insn`` on stock LLVM is the default wherever the accelerator exposes a command ISA
# (systolic MXUs; a SIMT core's op-level ISA is also pure .insn+CSR — see no-forked-toolchain-bringup).
# vector/scalar lower through UPSTREAM LLVM targets (RVV/base RISC-V), so no per-target encoding.
_PROFILES: dict[str, FamilyProfile] = {
    "systolic": FamilyProfile("systolic", "inline_asm_insn", True, "rocc_insn", ("L3", "L4", "L5"), (),
                              "circt_static"),
    "simt":     FamilyProfile("simt", "inline_asm_insn", False, None, ("L3",),
                              ("flops", "gflops", "pct_fp_peak"), "simt_config"),
    "vector":   FamilyProfile("vector", "upstream_target", False, None, (), (), "circt_static"),
    "scalar":   FamilyProfile("scalar", "upstream_target", False, None, (), (), "circt_static"),
    # Spatial tensor tile (Saturn OuterProductUnit family): a grid of accumulator cells driven by a
    # COMMAND BUFFER over one-hot op ports (macc/mvin/shift) — NOT a RoCC command ISA. So there is no
    # op->``.insn`` encoding to derive (encoding_required=False) and no rocc_insn trace gate; the 4th
    # artifact IS the schema-valid command buffer (command_buffer endpoint). Its facts come from the OPU
    # state-manifest geometry (cluster x cell tile, MRF depth, int8/fp8 datapaths) via the "opu" fact
    # extractor. Perf is tensor-tile MAC throughput. (See memory: opu-endpoint-is-command-buffer-not-rocc.)
    "spatial":  FamilyProfile("spatial", "command_buffer", False, None, ("L3", "L4", "L5"),
                              ("macs", "mac_per_cycle", "pct_mac_peak"), "opu"),
}


def family_profile(kind: str) -> FamilyProfile:
    """The generation defaults for a compute-unit ``kind`` (raises for an unknown kind — fail-closed)."""
    try:
        return _PROFILES[kind]
    except KeyError:
        raise KeyError(f"unknown compute-unit kind {kind!r}; known: {sorted(_PROFILES)}") from None


def known_kinds() -> tuple[str, ...]:
    return tuple(sorted(_PROFILES))
