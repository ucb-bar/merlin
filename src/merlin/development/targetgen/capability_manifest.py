"""Derive a target's compiler-facing capability manifest from its contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class CapabilityManifest:
    """The per-target capability model that drives GENERATION — a human-reviewed cache derived from RTL
    facts + the designer's docs (the committed ``target_contract.yaml``), NOT hand-invented for merlin.

    It resolves the target's PRIMARY compute-unit ``kind`` (the unit not embedded in another) and, via
    the family registry, the generation defaults (codegen endpoint, RTL tiers, perf fields, whether an
    op->``.insn`` encoding derivation + trace gate apply). Any default may be overridden by an optional
    ``runner``/``endpoint_kind`` block in the contract. Core generators consult this by ``kind`` so they
    never branch on a target name."""

    target: str
    kind: str  # primary compute-unit kind (systolic|simt|vector|scalar)
    endpoint_kind: str  # inline_asm_insn (default) | upstream_target | external_backend | command_buffer
    suite: str
    dtype: str  # run-identity dtype token (e.g. i8xi8_i32, f32)
    fourth_output_name: str | None  # None -> the runner derives it from endpoint_kind
    tier_sim: dict  # tier -> sim name (empty -> family/arc default)
    rtl_tiers: tuple[str, ...]
    perf_fields: tuple[str, ...]
    trace_gate: str | None  # trace-gate plugin name (e.g. "rocc_insn") or None
    force_match_policy: dict | None  # optional oracle output-equality override (float target -> {compare,atol})
    encoding_required: bool
    encoding: dict  # the ABI encoding surface RTL can't ground (readout_bits/semantic_class/...)
    contract: dict  # the full target_contract.yaml (for consumers that need more)


def _primary_kind(units) -> str:
    """The kind of the target's primary compute unit = the one NOT contained by any other."""
    contained = {c for u in units for c in u.contains}
    primary = [u for u in units if u.name not in contained]
    return (primary[0] if primary else units[0]).kind


def _derived_dtype_token(units) -> str:
    """A run-identity dtype token DERIVED from the primary compute unit's first accumulate rule
    (``<in>x<weight>_<acc>``). Replaces the former ``i8xi8_i32`` fail-open default so a target
    that omits ``runner.dtype`` (e.g. an mx target) is labeled by its OWN datapath, never mislabeled as
    int8. Falls back to ``"unknown"`` (fail-closed, surfaced in the run label) if no rule."""
    for u in units:
        if u.accumulate:
            a = u.accumulate[0]
            if a.inp and a.acc:
                return f"{a.inp}x{a.weight or a.inp}_{a.acc}"
    return "unknown"


def load_capability_manifest(target: str, *, contract_path: str | Path | None = None) -> CapabilityManifest:
    """Load a target's capability manifest from its committed ``target_contract.yaml`` + fill the family
    defaults. Raises if the target has no contract or no compute_units (fail-closed: no fabricated kind).

    ``contract_path`` reads that file instead of asking the registry. It exists for the case where the
    registry resolves NOTHING and a descriptor names the contract explicitly — the alternative there is
    not "use the resolved one", it is "render no prompt at all", which is what used to happen. It is not
    a general override: when the registry does resolve a contract, callers pass nothing and any
    disagreement with the declaration is reported by :func:`declared_vs_resolved_contract`."""
    from merlin.development.targetgen import compute_units, families, target_registry  # lazy: avoid import-order cycles

    if contract_path is not None:
        contract = yaml.safe_load(Path(contract_path).read_text(encoding="utf-8"))
    else:
        contract = target_registry.load_contract(target)
    units = compute_units.compute_units(contract)
    if not units:
        raise ValueError(f"{target}: target_contract has no compute_units — cannot derive a kind")
    kind = _primary_kind(units)
    prof = families.family_profile(kind)
    runner = contract.get("runner") or {}
    endpoint = contract.get("endpoint_kind") or prof.endpoint_kind_default
    if endpoint not in families.ENDPOINT_KINDS:
        raise ValueError(f"{target}: endpoint_kind {endpoint!r} not in {families.ENDPOINT_KINDS}")
    encoding = dict(contract.get("encoding") or {})
    # An address width does not imply any accelerator's flag layout. This loader
    # returns declared data; target-specific derivation belongs to OOT support.
    return CapabilityManifest(
        target=target,
        kind=kind,
        endpoint_kind=endpoint,
        suite=runner.get("suite") or f"{target}-capsule-bench",
        dtype=runner.get("dtype") or _derived_dtype_token(units),
        fourth_output_name=runner.get("fourth_output_name"),
        tier_sim=dict(runner.get("tier_sim") or {}),
        rtl_tiers=tuple(runner.get("rtl_tiers") or prof.default_rtl_tiers),
        perf_fields=tuple(runner.get("perf_fields") or prof.perf_fields),
        # The RoCC-.insn trace gate applies ONLY to an inline_asm_insn (RoCC) endpoint — it decodes a
        # host `.insn` stream from lowered.llvm.mlir. A self-hosted-ISA (external_backend, emits kernel.S)
        # or ISA-less (command_buffer) target has no such stream, so it defaults to no trace gate (unless
        # the contract explicitly declares one). Keys on the endpoint, never a target name.
        trace_gate=runner.get("trace_gate", prof.trace_gate if endpoint == "inline_asm_insn" else None),
        # Optional oracle output-equality override (a float target declares {compare: float, atol: ...}
        # so its oracle comparison is tolerant regardless of the per-capsule numeric_policy). None ->
        # the capsule's own numeric_policy governs (integer capsules -> exact).
        force_match_policy=runner.get("force_match_policy"),
        encoding_required=prof.encoding_required,
        encoding=encoding,
        contract=contract,
    )
