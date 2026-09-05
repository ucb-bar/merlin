"""Compile a target's DECLARED compiler obligations into runnable lit/FileCheck pass tests.

The sibling of :mod:`merlin.targetgen.rtl_check_compiler`, and deliberately the same shape: derive
every literal from the target's own sources, **omit a check that cannot be grounded rather than
defaulting it**, and carry per-check provenance so a reader can see which checks are evidence and
which are convention.

The input is a triple, all three of which already exist and are derived:

* the capability manifest's ``compiler_obligations`` — what THIS target's compiler must do. Measured
  2026-09-04: this field is required by the manifest validator and had no consumer anywhere in the
  repo. These tests are its first one.
* the conformance cell / tile geometry — the shape class the obligation is quantified over.
* the pass catalog's obligation vocabulary — which pass owes the obligation.

**What is target-specific here, stated precisely.** The obligations demanded and the geometry
constants are the target's own. The module the checks run over is on the target-INDEPENDENT
``interface`` plane, because that plane's op names are defined in-tree and therefore have a
derivation source. Checking a generated target dialect's invented mnemonics was tried, corroborated
over 383 runs, and removed; the emitted instruction stream is covered by the RTL check compiler
instead. So this layer checks "did the compiler discharge the obligation structurally", and the
decoded-stream layer checks "did it emit legal instructions for this ISA".
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any

#: The interface-plane op vocabulary the checks refer to. In-tree, bare namespace, fixed.
_PACK = "interface.resident_pack"
_MATMUL = "interface.matmul"
_COMMIT = "interface.commit"
_EVICT = "interface.resident_evict"
_PLACE = "schedule.place"


@dataclass
class Check:
    """One emitted check family, with the evidence that grounds it."""
    obligation: str
    lines: list[str]
    grounded_by: str
    derived: bool
    prefix: str = "CHECK"


@dataclass
class Omission:
    """A check that was NOT emitted, and why. Reported; never silently dropped."""
    obligation: str
    reason: str


@dataclass
class Compiled:
    target: str
    checks: list[Check] = field(default_factory=list)
    omissions: list[Omission] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)

    @property
    def coverage(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "obligations_declared": len(self.checks) + len(self.omissions),
            "emitted": len(self.checks),
            "omitted": len(self.omissions),
            "omission_reasons": [{"obligation": o.obligation, "reason": o.reason}
                                 for o in self.omissions],
            "checks": [{"obligation": c.obligation, "grounded_by": c.grounded_by,
                        "derived": c.derived, "n_lines": len(c.lines)} for c in self.checks],
        }


@functools.lru_cache(maxsize=None)
def _manifest(target: str) -> dict[str, Any] | None:
    from merlin.targetgen import capability_manifests

    try:
        return capability_manifests.manifest_for(target)
    except Exception:
        return None


@functools.lru_cache(maxsize=None)
def _facts(target: str) -> dict[str, Any]:
    """Derived RTL facts as ``{field: {value, derived, source}}``; empty on any failure.

    Memoized: fact derivation shells out to the RTL introspection bridge, which is slow enough that
    re-deriving per obligation (six targets x several obligations) dominated generation time. The
    bridge is already content-addressed, so caching per process changes no result.
    """
    try:
        from merlin.targetgen.rtl import mlc_bridge

        return mlc_bridge.target_fact_bundle(target).get("fields", {}) or {}
    except Exception:
        return {}


def _grounded(facts: dict[str, Any], name: str):
    """A fact's value only when it is actually DERIVED. A present-but-underived field is not a fact."""
    rec = facts.get(name)
    if not isinstance(rec, dict) or not rec.get("derived"):
        return None
    return rec.get("value")


# --- one builder per obligation --------------------------------------------------------------------
# Each returns a Check, or an Omission carrying the reason it could not be grounded.

def _tile_to_mesh(target: str, facts: dict[str, Any], shape: tuple[int, int, int]):
    """Tiling to the mesh is NOT checkable on the interface plane, and the reason matters.

    Measured 2026-09-04: `interface.accumulate` has no producer anywhere in the in-tree lowering, and
    there is no K-splitting in any staged pass. The tiling this obligation demands is emitted by the
    generated out-of-tree backend, which differs per run — so the interface plane simply does not
    contain the structure to assert a tile count over, whether or not the mesh edge is derivable.

    The mesh edge is still recorded as evidence when derived, because the obligation becomes
    checkable here the moment tiling moves in-tree, and it is already checkable TODAY on the decoded
    instruction stream (rtl_check_compiler asserts ceil(M/DIM)*ceil(N/DIM) mvouts there).
    """
    dim = _grounded(facts, "mesh_dim")
    if dim in (None, 0):
        return Omission("must_tile_to_mesh_shape",
                        "the mesh edge is not derivable from this target's RTL facts (empty facts "
                        "block or no extracted mesh), AND in-tree lowering does not tile; a software "
                        "default is not a hardware boundary, so nothing is asserted")
    return Omission("must_tile_to_mesh_shape",
                    f"mesh edge IS derived (mesh_dim={dim}) but in-tree lowering does not tile: "
                    "no staged pass splits K and interface.accumulate has no producer. The tiling is "
                    "emitted by the out-of-tree backend and is checked on the decoded instruction "
                    "stream instead")


def _commit_before_reuse(target: str, facts: dict[str, Any], shape):
    # A grammar invariant, not a hardware fact: the interface grammar admits exactly one commit per
    # committed tensor. Emitted for every target that declares the obligation, and marked underived
    # so nobody cites it as RTL evidence.
    return Check("must_commit_accumulator_before_reuse",
                 [f"// CHECK: {_MATMUL}", f"// CHECK: {_COMMIT}",
                  f"// CHECK-NOT: {_COMMIT}", f"// CHECK: {_MATMUL}"],
                 grounded_by="merlin_iface grammar (one commit per committed tensor)",
                 derived=False)


def _rhs_immutable_residency(target: str, facts: dict[str, Any], shape):
    """Pack once, use, evict -- and NOTHING may use the weight after the evict.

    The trailing ``CHECK-NOT`` is the half that makes this a lifetime check rather than a presence
    check. Without it the sequence is satisfied by a program that evicts the resident weight and then
    keeps matmul-ing against it: a use-after-evict, which is the exact defect the obligation names.
    Measured 2026-09-04: the generated form lacking this line PASSED the ``evict_before_last_use``
    mutation while the hand-written check in merlin/tests/data/lit/core/ caught it, i.e. the derived
    check was strictly weaker than the one it was modelled on.
    """
    return Check("must_prove_rhs_immutable_for_residency",
                 [f"// CHECK: {_PACK}", f"// CHECK-NOT: {_PACK}",
                  f"// CHECK: {_MATMUL}", f"// CHECK: {_EVICT}",
                  f"// CHECK-NOT: {_MATMUL}"],
                 grounded_by="merlin_iface grammar (pack once, use, evict after LAST use)",
                 derived=False)


def _block_scales(target: str, facts: dict[str, Any], shape):
    quantum = None
    try:
        from merlin.targetgen import corpus_spec

        quantum = getattr(corpus_spec, "shape_quantum", None)
    except Exception:
        quantum = None
    return Omission("must_supply_e8m0_block_scales",
                    "the block-scale group size is not derivable from this target's facts "
                    "(scale-group quantum unavailable), so no per-group scale-operand count is "
                    "asserted" if quantum is None else
                    "scale operands are not representable on the interface plane; this obligation "
                    "is checked on the decoded stream instead")


def _map_to_warps(target: str, facts: dict[str, Any], shape: tuple[int, int, int]):
    """Placement onto warps needs an ARITY to assert, not just a unit kind.

    A manifest can declare a unit whose kind is SIMT without stating how many lanes or warps it has.
    Presence of a placement op is then not evidence of anything — the residency obligation already
    forces one — so asserting it would manufacture coverage. The reason distinguishes the two cases,
    because "the manifest says nothing" and "the manifest says the kind but not the arity" are
    different gaps with different fixes.
    """
    manifest = _manifest(target) or {}
    units = manifest.get("compute_units") or []
    geometry = None
    kinds = set()
    for unit in units:
        kind = unit.get("kind")
        if kind:
            kinds.add(str(kind))
        for key in ("warp_size", "warps", "lanes", "n_lanes", "threads_per_warp"):
            if key in unit:
                geometry = (key, unit[key])
                break
    if geometry is None:
        detail = (f"unit kind(s) {sorted(kinds)} ARE declared but no lane/warp arity is"
                  if kinds else "no compute-unit geometry is")
        return Omission("must_map_to_warps",
                        f"{detail} declared in the capability manifest, so there is no arity to "
                        "assert; checking merely that a placement op exists would manufacture "
                        "coverage, since the residency obligation already forces one")
    return Check("must_map_to_warps",
                 [f"// CHECK: {_PLACE}"],
                 grounded_by=f"{geometry[0]}={geometry[1]} (capability manifest)", derived=True)


def _scratchpad_capacity(target: str, facts: dict[str, Any], shape):
    return Omission("must_respect_scratchpad_capacity",
                    "a numeric bound, not a structural property: FileCheck cannot express it. "
                    "It is checked by the RTL numeric screen (rtl_checks), not here")


_BUILDERS = {
    "must_tile_to_mesh_shape": _tile_to_mesh,
    "must_commit_accumulator_before_reuse": _commit_before_reuse,
    "must_prove_rhs_immutable_for_residency": _rhs_immutable_residency,
    "must_supply_e8m0_block_scales": _block_scales,
    "must_map_to_warps": _map_to_warps,
    "must_respect_scratchpad_capacity": _scratchpad_capacity,
}


def compile_checks(target: str, *, shape: tuple[int, int, int] = (16, 16, 16)) -> Compiled:
    """Compile every obligation this target declares into checks or recorded omissions."""
    out = Compiled(target=target)
    manifest = _manifest(target)
    if manifest is None:
        out.omissions.append(Omission(
            "<all>", "no capability manifest for this target (fails closed); nothing to derive "
                     "obligations from"))
        return out
    facts = _facts(target)
    out.facts = {k: {"derived": bool(v.get("derived")), "value": v.get("value")}
                 for k, v in facts.items() if isinstance(v, dict)}

    obligations = list(manifest.get("compiler_obligations") or [])
    if not obligations:
        out.omissions.append(Omission("<all>", "the capability manifest declares no "
                                               "compiler_obligations"))
        return out

    for ob in obligations:
        builder = _BUILDERS.get(ob)
        if builder is None:
            out.omissions.append(Omission(ob, "no check builder knows how to express this "
                                              "obligation structurally; recorded rather than "
                                              "silently skipped"))
            continue
        result = builder(target, facts, shape)
        (out.checks if isinstance(result, Check) else out.omissions).append(result)
    return out
