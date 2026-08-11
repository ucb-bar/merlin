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

    @property
    def discovered_nothing(self) -> bool:
        """True when NO field was grounded, i.e. discovery did not run rather than found bare hardware.

        Every field is None exactly when mlc or the artifact was unavailable, and that is
        indistinguishable, downstream, from a target whose RTL genuinely has no mesh and no accumulator.
        :func:`lever_derivation_gaps` exists to keep the two apart.
        """
        return (self.legal_opcodes is None and self.memory_map is None and self.dim is None)


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


def lever_derivation_gaps(profile: TargetProfile) -> tuple[str, ...]:
    """Why :func:`derived_levers` came back short, when the reason is "nothing was discovered".

    An empty lever list has two very different causes and the list itself cannot tell them apart: the RTL
    was read and has no mesh or accumulator memory, or nothing was read at all. MEASURED on the spatial
    tensor tile: every field comes back None and ``derived_levers`` returns ``[]``, which reads as "this
    accelerator exposes no structural levers" when what happened is that the discovery path produced
    nothing. A caller acting on that would report a target with no levers rather than a missing
    capability, so the silence is surfaced here instead of being inferred from an empty list.

    (This is also why the matrix-unit CCA lifter derives residency from the emitted instruction stream
    rather than from this profile: the stream is present whether or not RTL discovery is.)
    """
    if profile.discovered_nothing:
        return (f"no RTL fact was grounded for {profile.target!r} (legal_opcodes, memory_map and dim are "
                "all absent), so an empty lever list means UNKNOWN, not 'this hardware has no levers'",)
    gaps: list[str] = []
    if profile.dim is None:
        gaps.append("mesh dimension not discovered, so no dataflow lever is claimed")
    if not profile.has_accumulator:
        gaps.append("no accumulator memory discovered, so no residency lever is claimed")
    return tuple(gaps)


# Each derived lever's action class + the SpatialFacet target value it drives towards. This is
# agnostic-per-FACET metadata (a property of the shared SpatialFacet schema), NOT per-target: any
# systolic accelerator's dataflow/residency levers mean the same thing.
_LEVER_META = {
    "spatial.dataflow": ("HEURISTIC", "os",
                         "select the systolic dataflow (WS/OS) — a discovered mesh implies this choice"),
    "spatial.accumulator_resident": ("PASS", True,
                                     "keep the output accumulator-resident across the reduction — a "
                                     "discovered accumulator memory implies this choice"),
}


def apply_codegen_opts(opts: dict, features, profile: TargetProfile) -> dict:
    """Enable each derived lever named in ``features`` (each must be a lever the discovered HW implies).
    Empty ``features`` returns ``opts`` unchanged — the baseline. Target-agnostic (keyed by lever name)."""
    if not features:
        return dict(opts)
    levers = set(derived_levers(profile))
    out = dict(opts)
    for f in sorted(features):
        if f not in levers:
            raise KeyError(f"{f!r} is not a derived lever for {profile.target!r} (have {sorted(levers)})")
        out[f] = _LEVER_META[f][1]
    return out


def _resolver(spec, profile: TargetProfile) -> dict:
    """Realize the target-agnostic MicrokernelSpec as codegen knobs clamped to the DISCOVERED DIM."""
    dim = profile.dim or 1
    feats = frozenset({"spatial.accumulator_resident"}) if (getattr(spec, "k_block", False)
                                                            and profile.has_accumulator) else frozenset()
    return {"tile_rows": min(int(getattr(spec, "MR", dim) or dim), dim),
            "tile_cols": min(int(getattr(spec, "NR", dim) or dim), dim),
            "k_tile": getattr(spec, "KC", dim),
            "opts": apply_codegen_opts({}, feats, profile)}


def register(target: str, oot_package: str | None = None) -> None:
    """DERIVE + register the target's compiler-modification surface into the agnostic core — no per-target
    Python. Routes come from ``derived_levers(profile)``; the codegen seam is OOT-package-relative (the
    agent edits ITS generated middle-end); the micro-kernel resolver clamps to the discovered DIM.
    forkable_now is False until the target's OOT codegen threads the opts. Idempotent."""
    from ..kernels import action_catalog as ac, microkernel
    prof = target_profile(target)
    ac.register_seam(
        "rtl_codegen",
        "<oot_package>/lowering/  (the generated OOT backend's command/tile-program emitter — thread the "
        "derived CodegenOpts through it)",
        "the target's OOT codegen emitter (a discovery-derived lever is applied here)", True,
        backend=target)
    for lever in derived_levers(prof):
        action_class, target_value, change = _LEVER_META[lever]
        ac.register_route(target, ac._Route(
            axis=lever,
            when=lambda d, _l=lever: bool(d.expert) and d.expert != d.ours,
            action_class=action_class,
            target_seam=f"rtl_codegen:{lever}",
            change=change,
            forkable_now=False,
            expected_effect="expected (not yet measured): the discovered lever routed to the OOT codegen",
            intended_facet={lever: target_value}))
    microkernel.register_resolver(target, lambda spec, _p=prof: _resolver(spec, _p))


def lift_cca_from_trace(trace: dict, profile: TargetProfile, *, op: str = "matmul", source: str = "trace"):
    """Lift a DECODED accelerator command trace into a CCA via the agnostic ``cca.lift_spatial`` — using
    only generic accelerator-command concepts (a compute op present; an accumulate flag; an accumulator
    readout width) plus the discovered DIM. The raw-stream DECODING is the target's concern (mlc's
    behavioral opcode→effect map, or a target decoder); this consumes the decoded histogram."""
    from ..kernels import cca
    insns = trace.get("instructions", [])
    hist = trace.get("summary", {}).get("class_histogram", {})
    has_compute = any(v for k, v in hist.items() if "COMPUTE" in k.upper())
    acc_resident = (any("ACCUM" in k.upper() and v for k, v in hist.items())
                    or any(i.get("accumulate") for i in insns))
    i32 = any(i.get("readout") == "i32" for i in insns)
    counts = {"acc_resident": acc_resident if has_compute else None, "widening": has_compute,
              "acc_dtype": ("i32" if i32 else "i8") if has_compute else None}
    return cca.lift_spatial(counts, op=op, source=source, pe_rows=profile.dim, pe_cols=profile.dim,
                            backend=profile.target)


# Self-register as the derivation-driven route deriver: a seam-menu call for any non-RVV backend
# (cca_contract / action_catalog) then lazily DERIVES + registers that target's RTL levers via
# action_catalog.ensure_backend -> register(target). Import-time + guarded so a core-only environment
# (no targetgen import) is unaffected. register() is idempotent and no-ops without RTL access.
try:
    from ..kernels import action_catalog as _ac
    _ac.register_deriver(register)
except Exception:  # noqa: BLE001 — kernels package layout differs / partial env
    pass
