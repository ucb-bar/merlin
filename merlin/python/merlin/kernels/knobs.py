"""Motif -> RVV-knob mapping + the gap-router: turn a structural divergence (S4 compare) into
concrete fork proposals, each tagged with the lever class and the mined evidence that justifies it.

Honest forkable/deferred split (proven by the manual forks):
  * FORKABLE NOW (lever "knob") — edits expressible in the transform schedule today: tile/vector
    width (toward higher LMUL), contraction lowering strategy, lowering-pattern set.
  * DEFERRED (lever "lowering_pattern"/"llvm_requirement") — needs a schedule/pipeline FEATURE that
    does not exist yet, recorded as a work-item rather than auto-applied. The headline example: the
    fused-vfmacc gap needs fast-math `contract` injection at MLIR emission (outerproduct is a no-op
    because `transform.structured.vectorize` lowers the matmul straight to mul+add — no
    vector.contract is ever formed). The router surfaces it; it is not yet a one-knob fork.

`propose_forks(divergences, knobs)` returns a list of `ForkProposal` the beam expands.

SINGLE-ROUTER NOTE: the CCA<->lever router `kernels.action_catalog` (the one the bijection contract
enforces) is the source of truth for what the compiler exposes. `mining.fork_from_action.
propose_forks_from_cca` derives the beam proposer from it (consuming typed CCA `Divergence`s) and is the
successor to this motif-string router. This module is retained for the existing motif-string beam path
until the beam is cut over to CCA divergences (WS-D). NB: the `fma_form` "work-item" note below is
historical — fused vfmacc is now a certified `impr_features:fused_vfmacc_contraction` PASS
(see action_catalog), so that gap is CLOSED; the note is kept only to document the original routing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ForkProposal:
    overrides: dict[str, Any]          # knob overrides applied to the parent (empty for deferred)
    lever: str                          # knob | lowering_pattern | llvm_requirement | feature | work_item
    targets: str                        # which divergence/decision this addresses
    evidence: list[str]                 # mined policy / kernel ids justifying it
    forkable: bool                      # True => beam can mint+certify; False => recorded work-item
    note: str = ""
    # the typed CompilerAction this proposal came from (CCA-native proposer only; None for the legacy
    # motif router). Carries intended_facet so the beam can AUDIT the minted fork (did the emitted asm
    # achieve the promise?) via search_step.audit_fork. Left as Any to avoid importing action_catalog.
    action: Any = None


def _wider_n_overrides(knobs: dict, factor: int) -> dict:
    """Scale the N tile/vector dim of every contraction op (toward higher LMUL grouping)."""
    new = []
    for m in knobs.get("op_match", []):
        tile, vec = list(m["tile"]), list(m["vector"])
        # N is the second-to-last dim (…, M, N, K) for both matmul and batch_matmul here.
        if len(tile) >= 3:
            tile[-2] *= factor
            vec[-2] *= factor
        new.append({"op": m["op"], "tile": tile, "vector": vec})
    return {"op_match": new}


# decision key -> list of (lever, forkable, override-builder | None, evidence-policy, note)
_ROUTES: dict[str, list[dict]] = {
    "lmul_class": [
        {"lever": "knob", "forkable": True, "policy": "lmul_grouping_policy",
         "build": lambda k: _wider_n_overrides(k, 2),
         "note": "widen N tile/vector x2 to push vector grouping toward higher LMUL"},
        {"lever": "knob", "forkable": True, "policy": "lmul_grouping_policy",
         "build": lambda k: _wider_n_overrides(k, 4),
         "note": "widen N tile/vector x4"},
    ],
    "fma_form": [
        {"lever": "knob", "forkable": True, "policy": "fma_broadcast_policy",
         "build": lambda k: {"contraction_strategy": "outerproduct"},
         "note": "try outerproduct contraction lowering (NOTE: proven no-op; kept so the beam "
                 "records it as explored/pruned)"},
        {"lever": "llvm_requirement", "forkable": False, "policy": "fma_broadcast_policy",
         "build": None,
         "note": "RECOVER FUSED vfmacc: inject fast-math `contract` at MLIR emission so clang fuses "
                 "fmul+fadd -> fmuladd -> vfmacc. Not a schedule knob today (needs a lowering "
                 "feature: set fastmath on arith ops / a contract pass). Work-item."},
    ],
    "vl_strategy": [
        {"lever": "llvm_requirement", "forkable": False, "policy": "vl_tail_policy",
         "build": None,
         "note": "expert uses vsetvl-loop (VL-polymorphic); we emit vsetivli (fixed immediate). "
                 "Needs a scalable-vector / VL-loop lowering path. Work-item."},
    ],
    "int_widening": [
        {"lever": "knob", "forkable": True, "policy": "int8_widening_policy",
         "build": lambda k: {"dtype_strategy": "int8_w8a8"},
         "note": "route i8 matmul through the vwmacc integer datapath (passes_quant_int)"},
    ],
}


def propose_forks(divergences: list[str], knobs: dict[str, Any]) -> list[ForkProposal]:
    """From S4 divergence strings (e.g. "lmul_class: expert='m4' vs ours='m2'") + the parent knobs,
    enumerate candidate forks. Forkable proposals carry knob overrides; deferred ones are recorded
    work-items (lever-2/3) the beam reports but cannot auto-apply yet."""
    keys = [d.split(":")[0].strip() for d in divergences]
    out: list[ForkProposal] = []
    for key in keys:
        for route in _ROUTES.get(key, []):
            overrides = route["build"](knobs) if route["build"] else {}
            out.append(ForkProposal(
                overrides=overrides, lever=route["lever"], targets=key,
                evidence=[route["policy"]], forkable=route["forkable"], note=route["note"]))
    return out


# =====================================================================================================
# THE TYPED KNOB LAYER — the ONE mutation surface the optimization phase is allowed to touch
# =====================================================================================================
#
# WHY THIS EXISTS. Everything above this line is the motif-string fork router: it proposes edits, and
# nothing checks that a proposed edit is a *nameable* change. That is the gap. When the optimization
# phase can reach a transform, a heuristic or a pass flag by any route it likes, a score that goes up
# cannot be attributed to anything, and the run teaches nothing — the same failure shape as a bench
# whose headline number turned out to belong to a different tier than the one being quoted.
#
# So the phase gets exactly FOUR surfaces, and a change that does not land on one of them is REFUSED
# by name. The surfaces are not new vocabulary: each is the set of decisions one existing CCA facet
# family already describes, which is what lets an attributed cost be asked for BY SURFACE ("35% is
# dispatch, so do not spend the round tuning a tile for 3%").
#
#   TILE      -> compute.register_block, memory.*, spatial/simt residency   (tile shape, loop order,
#                                                                            staging depth, residency)
#   LAYER     -> layout.*, memory.onchip_resident                           (fusion, packing, operand
#                                                                            residency across a layer)
#   PLACEMENT -> dispatch.*, communication.*                                (A->A, A->H, H->A, dispatch
#                                                                            grouping, pipelining)
#   GLOBAL    -> compute.accumulator_dtype, layout propagation, coverage    (quantization, encoding,
#                                                                            layout, partitioning)
#
# DERIVED, NOT DECLARED. Every domain below comes from the target's own sources — its RTL facts via
# ``targetgen.address_space``, its operand store via ``targetgen.memory_regime``, its declared compute
# units via ``targetgen.compute_units``, and the roles its own ISA licenses via ``kernels.endpoints``.
# Nothing here knows a tile size, a bank count, an array edge or a depth. A quantity that cannot be
# derived is :data:`UNKNOWN` and SAYS SO; it is never given a plausible default, because a plausible
# default on a capacity bound is precisely how a backend addressed every weight tile as simultaneously
# resident and aborted three layers away in a range check with nothing recorded.
#
# THREE STATES, NEVER TWO. ``allowed`` / ``refused`` / ``undeterminable`` stay distinct all the way
# through :func:`check_mutation`. Collapsing the last two is the recurring bug class in this tree: a
# check that could not run reported success and burned a 101-minute round.

from collections.abc import Mapping, Sequence  # noqa: E402 -- the typed layer's imports, kept beside it
from itertools import permutations             # noqa: E402

#: The four surfaces, weakest scope first. A change is on-surface or it is refused; there is no fifth.
TILE = "tile"
LAYER = "layer"
PLACEMENT = "placement"
GLOBAL = "global"
SURFACES: tuple[str, ...] = (TILE, LAYER, PLACEMENT, GLOBAL)

#: Which CCA facets each surface is the knob-side of. Stated as data so a caller holding an attributed
#: cost on a FACET (which is what a lifted CCA carries) can ask which SURFACE to spend effort on, and
#: so a future facet cannot be added to the CCA and silently belong to no surface.
SURFACE_FACETS: dict[str, tuple[str, ...]] = {
    TILE: ("compute", "memory", "vector", "spatial", "simt"),
    LAYER: ("layout", "memory"),
    PLACEMENT: ("dispatch", "communication", "envelope"),
    GLOBAL: ("compute", "layout", "coverage"),
}

#: Domain kinds. ``UNKNOWN`` is a first-class kind, not an empty set: an empty set means "no value is
#: legal" (a refusal), while UNKNOWN means "we cannot say", and those two want opposite responses.
SET = "set"
RANGE = "range"
UNKNOWN_DOMAIN = "unknown"

#: Verdicts. See :class:`Verdict` for why refusal dominates undeterminable in the aggregate.
ALLOWED = "allowed"
REFUSED = "refused"
UNDETERMINABLE = "undeterminable"

#: VALUE VOCABULARIES THAT ARE NOT TARGET FACTS. These are the legal values of a CCA *field*, i.e. a
#: property of our own schema, not of any device — so restating them here bakes in nothing about a
#: target. They are restated at all only because ``cca.py`` declares them in PROSE comments
#: (``# k_major | m_major | n_major``) with nothing machine-readable behind them; see the module
#: report. If those vocabularies ever become data on the facet, these three constants delete.
_OPERAND_MAJOR_VALUES: tuple[str, ...] = ("k_major", "m_major", "n_major")   # cca.LayoutFacet.operand_major
_RESIDENT_VALUES: tuple[str, ...] = ("a", "b", "both", "none")              # cca.MemoryFacet.onchip_resident
_BOOL_VALUES: tuple[bool, ...] = (False, True)

#: The issuing side of an endpoint, when the target's contract does not declare it as a compute unit.
#: Not a target fact and not an assumed one: ``kernels.endpoints`` resolves an Endpoint for every
#: target, and an endpoint is by construction driven by something (the premise ``cca.DispatchFacet``
#: is built on). Naming it is what makes A->H and H->A expressible at all; when the contract DOES
#: declare a scalar unit, that unit's own name is used instead and this token never appears.
_ISSUER_SITE = "host"


@dataclass(frozen=True)
class Domain:
    """The values a knob may take — an explicit set, a bounded range, or an honest UNKNOWN.

    ``source`` names WHERE the domain came from (which derivation, over which target artifact), so a
    reviewer can check a bound rather than trust it. ``why_unknown`` is populated exactly when
    ``kind == UNKNOWN_DOMAIN`` and says which quantity was missing — a domain that is unknown without
    a reason is the silent-default bug this layer exists to prevent.
    """

    kind: str
    values: tuple = ()
    lo: int | None = None
    hi: int | None = None
    source: str = ""
    why_unknown: str = ""

    def admits(self, value: Any) -> bool | None:
        """``True`` allowed, ``False`` refused, ``None`` UNDETERMINABLE. Three states, never two."""
        if self.kind == UNKNOWN_DOMAIN:
            return None
        if self.kind == SET:
            # Membership by equality, not by truthiness: ``False in (0, 1)`` is True in Python, and a
            # boolean knob silently admitting an integer is how a "did it apply?" flag becomes a count.
            return any(value is v or (type(value) is type(v) and value == v) for v in self.values)
        if self.lo is None or self.hi is None:
            return None
        if not isinstance(value, int) or isinstance(value, bool):
            return False
        return self.lo <= value <= self.hi

    def describe(self) -> str:
        if self.kind == UNKNOWN_DOMAIN:
            return f"UNKNOWN ({self.why_unknown})"
        if self.kind == SET:
            return "{" + ", ".join(repr(v) for v in self.values) + "}"
        return f"[{self.lo}..{self.hi}]"

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"kind": self.kind, "source": self.source}
        if self.kind == SET:
            out["values"] = list(self.values)
        elif self.kind == RANGE:
            out["lo"], out["hi"] = self.lo, self.hi
        else:
            out["why_unknown"] = self.why_unknown
        return out


def value_set(values: Sequence, *, source: str) -> Domain:
    """An explicit domain. Order is preserved (it is the enumeration order a search walks)."""
    return Domain(kind=SET, values=tuple(values), source=source)


def bounded_range(lo: int, hi: int, *, source: str) -> Domain:
    """An inclusive integer range. ``lo == hi`` is legal and MEANINGFUL: it is a knob the target pins,
    i.e. a lever that exists in the vocabulary and has no room on this device — which is a different
    report from "we could not derive it"."""
    return Domain(kind=RANGE, lo=int(lo), hi=int(hi), source=source)


def unknown_domain(why: str, *, source: str = "") -> Domain:
    return Domain(kind=UNKNOWN_DOMAIN, why_unknown=why, source=source)


@dataclass(frozen=True)
class Knob:
    """One named, typed, enumerable knob on exactly one surface.

    ``default`` is the value the compiler uses when the phase makes no choice. It is ``None`` when the
    domain is UNKNOWN (nothing to default to) AND, separately, when the domain is known but no default
    is derivable — those two are distinguished by ``why_no_default``, because "we do not know the
    range" and "we know the range and cannot say which point the compiler starts at" are different
    admissions and a caller must not read either as zero.
    """

    name: str
    surface: str
    domain: Domain
    controls: str
    default: Any = None
    why_no_default: str = ""

    def __post_init__(self) -> None:
        if self.surface not in SURFACES:
            raise ValueError(f"knob {self.name!r}: surface {self.surface!r} not in {list(SURFACES)}")
        if self.default is None and not self.why_no_default and self.domain.kind != UNKNOWN_DOMAIN:
            raise ValueError(f"knob {self.name!r}: no default and no reason given — a missing default "
                             f"must be an explicit admission, never an omission")

    @property
    def determinable(self) -> bool:
        return self.domain.kind != UNKNOWN_DOMAIN

    def admits(self, value: Any) -> bool | None:
        return self.domain.admits(value)

    def to_dict(self) -> dict:
        out = {"name": self.name, "surface": self.surface, "controls": self.controls,
               "domain": self.domain.to_dict(), "default": self.default}
        if self.why_no_default:
            out["why_no_default"] = self.why_no_default
        return out


@dataclass(frozen=True)
class KnobInventory:
    """Every knob one target exposes, plus what could not be derived for it.

    ``unknowns`` is the reported half of requirement 1 and is never empty-by-omission: a knob whose
    domain is UNKNOWN appears here AND stays in ``knobs`` (so it is still enumerable and still refuses
    to be silently defaulted), rather than being dropped — a dropped knob reads as "this target has no
    such lever", which is the opposite of "we could not bound it".
    """

    target: str
    knobs: tuple[Knob, ...] = ()
    notes: dict[str, str] = field(default_factory=dict)

    def get(self, name: str) -> Knob | None:
        return next((k for k in self.knobs if k.name == name), None)

    def names(self) -> tuple[str, ...]:
        return tuple(k.name for k in self.knobs)

    def by_surface(self, surface: str) -> tuple[Knob, ...]:
        if surface not in SURFACES:
            raise ValueError(f"surface {surface!r} not in {list(SURFACES)}")
        return tuple(k for k in self.knobs if k.surface == surface)

    @property
    def unknowns(self) -> tuple[Knob, ...]:
        return tuple(k for k in self.knobs if not k.determinable)

    def surface_of(self, name: str) -> str | None:
        k = self.get(name)
        return k.surface if k else None

    def to_dict(self) -> dict:
        return {"target": self.target,
                "knobs": [k.to_dict() for k in self.knobs],
                "unknown_knobs": [k.name for k in self.unknowns],
                "notes": dict(self.notes)}


# ---- derivation -------------------------------------------------------------------------------

def _address_space(target: str):
    """``(AddressSpace | None, reason)``. Never raises: an unreadable artifact is a reported UNKNOWN,
    because this layer is asked for an inventory on targets whose facts are empty and must still
    describe the knobs it COULD derive."""
    try:
        from merlin.targetgen import address_space as _as
        return _as.derive_address_space(target), ""
    except Exception as e:                       # noqa: BLE001 -- no artifact / no toolchain / bad shape
        return None, f"{type(e).__name__}: {str(e)[:160]}"


def _units(target: str):
    """``(units, reason)`` — the target's declared compute units, or an empty tuple with the reason.

    Deliberately catches broadly. ``compute_units`` RAISES on a contract that names a quant format the
    registry does not know (a real, currently-live condition on at least one declared target), and a
    knob layer that propagated it would return no inventory at all for that target instead of an
    inventory whose engine-derived knobs are honestly UNKNOWN.
    """
    try:
        from merlin.targetgen import target_registry as _tr
        from merlin.targetgen.compute_units import compute_units
        return tuple(compute_units(_tr.load_contract(target) or {})), ""
    except Exception as e:                       # noqa: BLE001 -- undeclared / malformed contract
        return (), f"{type(e).__name__}: {str(e)[:160]}"


def _endpoint_roles(target: str) -> tuple[frozenset[str], str]:
    """The union of instruction ROLES the target's own endpoints license, or empty with a reason.

    Roles are the derived answer to "can this device do X at all": ``loop_descriptor`` present means a
    whole loop nest can be handed to the endpoint's sequencer, ``dma`` present means bulk movement can
    be issued to overlap compute. Reading either off a role census is the only target-agnostic way to
    ask — the alternative is matching an opcode NAME out of the decode table, which is the string-match
    prohibition and would silently answer "no" for any device that spells it differently.
    """
    try:
        from merlin.kernels import endpoints as _ep
        seen: set[str] = set()
        for e in _ep.endpoints_for(target):
            seen.update(getattr(e, "roles", {}) or {})
        if not seen:
            return frozenset(), "the target's endpoints license no roles we could read"
        return frozenset(seen), ""
    except Exception as e:                       # noqa: BLE001 -- no endpoint spec for this target
        return frozenset(), f"{type(e).__name__}: {str(e)[:160]}"


def _accumulator_store(space):
    """The WIDEST-row store, which is the accumulator when the target has a separate one.

    Chosen by row width rather than by name, mirroring ``memory_regime.operand_store`` (which takes the
    narrowest): a separate accumulator space exists precisely BECAUSE its row is wider — it holds the
    accumulate type. Selecting by name would bake one target's spelling into shared code.
    """
    stores = [s for s in (getattr(space, "stores", ()) or ()) if getattr(s, "row_bytes", None)]
    if len(stores) < 2:
        return None
    return max(stores, key=lambda s: int(s.row_bytes))


def _tile_knobs(target: str, space, reason: str, store, capacity, roles: frozenset[str],
                role_reason: str, working_set) -> list[Knob]:
    """TILE: tile shape, loop order, staging/prefetch depth, residency."""
    out: list[Knob] = []
    rows = getattr(space, "array_rows", None) if space is not None else None
    cols = getattr(space, "array_cols", None) if space is not None else None
    geom_src = f"targetgen.address_space.derive_address_space({target!r}).array"

    # --- tile shape. The array edge is what the compiler tiles TO, so it is the tile extent's BOUND,
    # not the tile extent itself: a tile narrower than the edge underfills the array and one wider than
    # it does not exist as a single issue. Both extents are separately derived because a non-square
    # array is ordinary and assuming squareness would be an assumed constant wearing a derivation's
    # clothes.
    for axis, extent in (("rows", rows), ("cols", cols)):
        if isinstance(extent, int) and extent > 0:
            dom = bounded_range(1, extent, source=f"{geom_src}.{axis}")
            out.append(Knob(f"tile.extent_{axis}", TILE, dom,
                            controls=f"tile extent along the array's {axis} edge",
                            default=extent))
        else:
            out.append(Knob(f"tile.extent_{axis}", TILE,
                            unknown_domain(reason or f"these facts declare no array {axis} extent",
                                           source=geom_src),
                            controls=f"tile extent along the array's {axis} edge"))

    # --- reduction extent. Bounded by the OPERAND STORE, not by the array: the reduction axis is
    # streamed through the store, and the measured abort was exactly a schedule that addressed more
    # rows than the store has (16384 requested against 16384 present).
    if capacity:
        out.append(Knob("tile.reduction_rows", TILE,
                        bounded_range(1, int(capacity),
                                      source=f"targetgen.memory_regime.operand_store({target!r})"
                                             f".total_rows"),
                        controls="rows of the operand store one tile's reduction slice may occupy",
                        default=None,
                        why_no_default="the point the compiler starts at is a schedule fact, not a "
                                       "target fact; defaulting to the capacity is the shape that "
                                       "aborted in a range check, and defaulting to 1 would claim a "
                                       "residency choice nobody made"))
    else:
        out.append(Knob("tile.reduction_rows", TILE,
                        unknown_domain(f"{target!r} declares no operand-store capacity we can derive",
                                       source="targetgen.memory_regime.operand_store"),
                        controls="rows of the operand store one tile's reduction slice may occupy"))

    # --- loop order. The axis SET is derived, not listed: the array contributes one axis per declared
    # extent, and a reduction axis exists iff the endpoint licenses an ``accumulate`` role (a device
    # with no accumulate has no reduction to order). Enumerating permutations of a derived axis set
    # bakes in no device fact; hardcoding "the six orders of m,n,k" would.
    axes = [a for a, e in (("rows", rows), ("cols", cols)) if isinstance(e, int) and e > 0]
    if "accumulate" in roles:
        axes.append("reduction")
    if len(axes) >= 2:
        orders = tuple(tuple(p) for p in permutations(axes))
        out.append(Knob("tile.loop_order", TILE,
                        value_set(orders, source=f"{geom_src} extents + endpoint roles "
                                                 f"(reduction axis iff an 'accumulate' role exists)"),
                        controls="permutation of the tile loop nest's axes",
                        default=orders[0]))
    else:
        why = (reason or role_reason
               or "fewer than two axes are derivable (no array extents, and no accumulate role)")
        out.append(Knob("tile.loop_order", TILE, unknown_domain(why, source=geom_src),
                        controls="permutation of the tile loop nest's axes"))

    # --- staging / prefetch depth. Two independent bounds, and the tighter wins:
    #   * the store's BANK count — depth N alternates N banks, and a device with one bank cannot stage;
    #   * whether the working set fits N times, via ``memory_regime.classify``. Without a working set
    #     the regime is genuinely unknown, so only the bank bound applies and we say so.
    banks = getattr(store, "banks", None) if store is not None else None
    if isinstance(banks, int) and banks > 0:
        hi, src = banks, f"targetgen.memory_regime.operand_store({target!r}).banks"
        if working_set is not None and capacity:
            regime, fit_hi = _staging_bound_from_regime(working_set, capacity)
            if fit_hi is not None:
                hi = min(hi, fit_hi)
                src += f" ∧ memory_regime.classify -> {regime}"
        out.append(Knob("tile.stage_depth", TILE, bounded_range(1, hi, source=src),
                        controls="how many tiles are staged/prefetched ahead of the one computing",
                        default=1))
    else:
        out.append(Knob("tile.stage_depth", TILE,
                        unknown_domain(reason or "no operand store with a derivable bank count, so "
                                                 "nothing bounds how many tiles can be in flight",
                                       source="targetgen.memory_regime.operand_store"),
                        controls="how many tiles are staged/prefetched ahead of the one computing"))

    # --- residency. A schema vocabulary, not a device fact (see _RESIDENT_VALUES).
    out.append(Knob("tile.operand_resident", TILE,
                    value_set(_RESIDENT_VALUES, source="cca.MemoryFacet.onchip_resident vocabulary"),
                    controls="which operand stays in the on-chip store across the reduction",
                    default="none"))
    return out


def _staging_bound_from_regime(working_set, capacity) -> tuple[str, int | None]:
    """``(regime, max stage depth)`` from ``memory_regime.classify``.

    ``working_set`` is ``(rows_live, rows_total)`` or a single ``rows_live``. The mapping is the
    regime's own documented meaning, not a heuristic: ``fits_double`` is defined as "fits TWICE, so
    movement for the next tile can overlap compute on the current one" — the only regime in which
    staging is possible at all — and every other fitting regime permits exactly one in flight.
    ``unknown`` returns ``None`` so the caller keeps the bank bound rather than tightening on a regime
    nobody measured.
    """
    from merlin.targetgen import memory_regime as _mr
    if isinstance(working_set, (tuple, list)) and len(working_set) == 2:
        live, total = working_set
    else:
        live = total = working_set
    if not live:
        # A zero live set is not a measurement of a program, it is the absence of one -- and
        # ``classify`` reads it as fitting twice over. Tightening (or loosening) a hardware bound on
        # that would report a staging verdict for a program nobody sized.
        return _mr.UNKNOWN, None
    regime = _mr.classify(live, total, capacity)
    if regime == _mr.UNKNOWN:
        return regime, None
    if regime == _mr.FITS_DOUBLE:
        return regime, max(1, int(capacity) // int(live))
    return regime, 1


def _layer_knobs(target: str, capacity: int | None, working_set) -> list[Knob]:
    """LAYER: fusion, operand residency across a layer, packing."""
    out: list[Knob] = []
    if capacity:
        cap_src = f"targetgen.memory_regime.operand_store({target!r}).total_rows"
        out.append(Knob("layer.operand_residency_rows", LAYER,
                        bounded_range(1, int(capacity), source=cap_src),
                        controls="rows a layer's operands may hold across the whole layer",
                        default=None,
                        why_no_default="how much a layer keeps resident is a schedule choice; the "
                                       "target bounds it and does not pick it"))
        if working_set is not None:
            live = working_set[0] if isinstance(working_set, (tuple, list)) else working_set
            if live:
                out.append(Knob("layer.fusion_depth", LAYER,
                                bounded_range(1, max(1, int(capacity) // int(live)),
                                              source=f"{cap_src} / the supplied live working set"),
                                controls="consecutive ops fused without a round trip through memory",
                                default=1))
                out.append(_prepack_knob())
                out.append(_operand_major_knob())
                return out
    else:
        out.append(Knob("layer.operand_residency_rows", LAYER,
                        unknown_domain(f"{target!r} declares no operand-store capacity we can derive",
                                       source="targetgen.memory_regime.operand_store"),
                        controls="rows a layer's operands may hold across the whole layer"))
    # Fusion depth without a working set: genuinely undeterminable, and said so rather than bounded by
    # something convenient. How many ops can be fused is a property of the op GRAPH's intermediates;
    # no facts artifact carries it, and a bound invented from capacity alone would license a fusion the
    # intermediates do not fit.
    out.append(Knob("layer.fusion_depth", LAYER,
                    unknown_domain("fusion depth is a property of the op graph's intermediates, not of "
                                   "the target; pass working_set= (the layer's live rows) to bound it",
                                   source="requires a program, not a facts artifact"),
                    controls="consecutive ops fused without a round trip through memory"))
    out.append(_prepack_knob())
    out.append(_operand_major_knob())
    return out


def _prepack_knob() -> Knob:
    return Knob("layer.prepack", LAYER,
                value_set(_BOOL_VALUES, source="cca.LayoutFacet.prepack_required vocabulary"),
                controls="operand panel packed offline rather than gathered per tile",
                default=False)


def _operand_major_knob() -> Knob:
    return Knob("layer.operand_major", LAYER,
                value_set(_OPERAND_MAJOR_VALUES, source="cca.LayoutFacet.operand_major vocabulary"),
                controls="which axis the operand panel is laid out along",
                default=None,
                why_no_default="the layout the compiler starts from is the model's, not the target's; "
                               "naming one here would assert a packing nobody derived")


def _placement_knobs(target: str, units, unit_reason: str, roles: frozenset[str],
                     role_reason: str) -> list[Knob]:
    """INTER-OP / PLACEMENT: A->A, A->H, H->A, dispatch grouping, pipelining."""
    out: list[Knob] = []
    unit_src = f"targetgen.compute_units.compute_units(contract({target!r}))"
    engines = [u for u in units if getattr(u, "kind", None) != "scalar"]
    scalar = next((u for u in units if getattr(u, "kind", None) == "scalar"), None)
    if units:
        issuer = getattr(scalar, "name", None) or _ISSUER_SITE
        sites = tuple([issuer] + [u.name for u in engines])
        out.append(Knob("placement.site", PLACEMENT,
                        value_set(sites, source=f"{unit_src} + the issuing side of its endpoint"),
                        controls="which engine (or the issuer) an op is placed on",
                        default=issuer))
        # Every ordered pair, INCLUDING the self-pairs: an A->A hand-off (one engine feeding the next
        # op on the same engine without a round trip) is a real placement choice and dropping the
        # diagonal would make it unexpressible.
        pairs = tuple((a, b) for a in sites for b in sites)
        out.append(Knob("placement.transfer", PLACEMENT,
                        value_set(pairs, source=f"ordered pairs of the derived sites ({unit_src})"),
                        controls="the boundary a value crosses between two ops (A->A, A->H, H->A)",
                        default=(issuer, issuer)))
        out.append(Knob("global.partition_count", GLOBAL,
                        bounded_range(1, max(1, len(engines)), source=unit_src),
                        controls="how many engines the model is partitioned across",
                        default=1))
    else:
        why = unit_reason or f"{target!r} declares no compute units, so no placement site is derivable"
        out.append(Knob("placement.site", PLACEMENT, unknown_domain(why, source=unit_src),
                        controls="which engine (or the issuer) an op is placed on"))
        out.append(Knob("placement.transfer", PLACEMENT, unknown_domain(why, source=unit_src),
                        controls="the boundary a value crosses between two ops (A->A, A->H, H->A)"))
        out.append(Knob("global.partition_count", GLOBAL, unknown_domain(why, source=unit_src),
                        controls="how many engines the model is partitioned across"))

    # --- loop offload + dispatch grouping. Both keyed on ROLES the target's own ISA licenses, so a
    # device without a hardware-loop sequencer gets a one-value domain (the lever exists in the
    # vocabulary and has no room here) rather than a lever that proposes an instruction it cannot issue.
    role_src = f"kernels.endpoints.endpoints_for({target!r}) role census"
    if roles:
        can_offload = "loop_descriptor" in roles
        out.append(Knob("placement.loop_offload", PLACEMENT,
                        value_set(_BOOL_VALUES if can_offload else (False,),
                                  source=f"{role_src}: 'loop_descriptor' "
                                         f"{'licensed' if can_offload else 'not licensed'}"),
                        controls="hand a whole loop nest to the endpoint's own sequencer",
                        default=False))
        if can_offload:
            # The group extent is then set by the offloaded NEST, and no facts artifact bounds it.
            # Reporting a bound we cannot derive is exactly the failure this layer refuses.
            out.append(Knob("placement.dispatch_group", PLACEMENT,
                            unknown_domain("this target licenses a loop_descriptor role, so the group "
                                           "extent is whatever nest is offloaded; nothing in its facts "
                                           "bounds it", source=role_src),
                            controls="tiles handed over per issued command group"))
        else:
            out.append(Knob("placement.dispatch_group", PLACEMENT,
                            value_set((1,), source=f"{role_src}: no loop_descriptor role, so one tile "
                                                   f"per issued command"),
                            controls="tiles handed over per issued command group",
                            default=1))
        can_dma = "dma" in roles
        out.append(Knob("placement.pipeline_movement", PLACEMENT,
                        value_set(_BOOL_VALUES if can_dma else (False,),
                                  source=f"{role_src}: 'dma' "
                                         f"{'licensed' if can_dma else 'not licensed'}"),
                        controls="issue bulk movement so it overlaps the compute it feeds",
                        default=False))
    else:
        for name, controls in (("placement.loop_offload",
                                "hand a whole loop nest to the endpoint's own sequencer"),
                               ("placement.dispatch_group",
                                "tiles handed over per issued command group"),
                               ("placement.pipeline_movement",
                                "issue bulk movement so it overlaps the compute it feeds")):
            out.append(Knob(name, PLACEMENT, unknown_domain(role_reason or "no endpoint role census",
                                                           source=role_src), controls=controls))
    return out


def _global_knobs(target: str, units, unit_reason: str, space, reason: str) -> list[Knob]:
    """GLOBAL: quantization, encoding, layout propagation, partitioning.

    (``global.partition_count`` is minted in :func:`_placement_knobs`, where the site derivation it
    depends on already lives — one derivation, not two that must agree.)
    """
    out: list[Knob] = []
    unit_src = f"targetgen.compute_units.compute_units(contract({target!r}))"

    # --- element format. The UNION over declared units, because a hybrid target's two engines accept
    # different formats and taking one unit's list would silently forbid the other engine's work.
    dtypes: list[str] = []
    for u in units:
        for d in getattr(u, "dtypes", ()) or ():
            if d not in dtypes:
                dtypes.append(d)
    if dtypes:
        out.append(Knob("global.element_format", GLOBAL,
                        value_set(tuple(dtypes), source=f"{unit_src} declared dtypes (union)"),
                        controls="element encoding the model is quantized to",
                        # A single declared format is not a choice — it is the only thing the silicon
                        # accepts — so it is also the default. With several, which one the compiler
                        # starts at is a schedule fact and we decline to invent it.
                        default=dtypes[0] if len(dtypes) == 1 else None,
                        why_no_default=("" if len(dtypes) == 1 else
                                        "this target accepts several formats; which one the compiler "
                                        "starts at is a schedule fact, not a target fact")))
    else:
        out.append(Knob("global.element_format", GLOBAL,
                        unknown_domain(unit_reason or f"{target!r} declares no unit dtypes",
                                       source=unit_src),
                        controls="element encoding the model is quantized to"))

    # --- accumulate format. Two derivations, contract first: a unit's own accumulate rules say what it
    # accumulates in. Falling back to the WIDEST-row store's element dtype covers a target that
    # declares its accumulator in RTL facts and not in its contract, and it is the same
    # widest-row = accumulator rule ``memory_regime`` uses in reverse.
    accs: list[str] = []
    for u in units:
        for rule in getattr(u, "accumulate", ()) or ():
            a = getattr(rule, "acc", None)
            if a and a not in accs:
                accs.append(a)
    acc_src = f"{unit_src} accumulate rules"
    if not accs and space is not None:
        acc_store = _accumulator_store(space)
        elem = getattr(acc_store, "element_dtype", None) if acc_store is not None else None
        if elem:
            accs = [elem]
            acc_src = (f"targetgen.address_space.derive_address_space({target!r}): the widest-row "
                       f"store's element dtype")
    if accs:
        out.append(Knob("global.accumulate_format", GLOBAL, value_set(tuple(accs), source=acc_src),
                        controls="width the reduction accumulates in",
                        default=accs[0] if len(accs) == 1 else None,
                        why_no_default=("" if len(accs) == 1 else
                                        "several accumulate widths are declared; the starting one is "
                                        "a schedule fact")))
    else:
        out.append(Knob("global.accumulate_format", GLOBAL,
                        unknown_domain(unit_reason or reason
                                       or "no accumulate rule and no separate accumulator store",
                                       source=acc_src),
                        controls="width the reduction accumulates in"))

    # --- scale kind. The encoding half that is NOT the element width, and the half a per-channel or
    # block-scaled format lives or dies on.
    scales: list[str] = []
    for u in units:
        s = getattr(u, "scaling", None)
        if s and s not in scales:
            scales.append(s)
    if scales:
        out.append(Knob("global.scale_kind", GLOBAL,
                        value_set(tuple(scales), source=f"{unit_src} declared scaling"),
                        controls="how the quantization scale is carried (per-channel, block, none)",
                        default=scales[0] if len(scales) == 1 else None,
                        why_no_default=("" if len(scales) == 1 else
                                        "units declare different scale kinds; the starting one is a "
                                        "schedule fact")))
    else:
        out.append(Knob("global.scale_kind", GLOBAL,
                        unknown_domain(unit_reason or f"{target!r} declares no scaling on any unit",
                                       source=unit_src),
                        controls="how the quantization scale is carried"))

    out.append(Knob("global.layout_propagation", GLOBAL,
                    value_set(_BOOL_VALUES, source="cca.LayoutFacet.transpose_materialized vocabulary"),
                    controls="propagate a layout choice through the graph instead of materializing a "
                             "transpose at each consumer",
                    default=False))
    return out


def derive_knobs(target: str, *, working_set=None) -> KnobInventory:
    """The typed knob inventory for ``target``, derived from that target's own sources. Never raises.

    ``working_set`` is optional and, when given, is ``rows_live`` or ``(rows_live, rows_total)`` in
    OPERAND-STORE ROWS (``address_space.Store.working_set_rows``). It is the only way two knobs
    (``tile.stage_depth``'s regime bound and ``layer.fusion_depth``) become determinable, because both
    are questions about a PROGRAM against a capacity and neither can be answered by a facts artifact
    alone. Omitted, they report UNKNOWN with that as the reason rather than a convenient bound.
    """
    from merlin.targetgen import memory_regime as _mr

    space, reason = _address_space(target)
    units, unit_reason = _units(target)
    roles, role_reason = _endpoint_roles(target)
    try:
        store, capacity = _mr.operand_store(target)
    except Exception as e:                       # noqa: BLE001 -- unresolvable target
        store, capacity, reason = None, None, reason or f"{type(e).__name__}: {str(e)[:160]}"

    knobs: list[Knob] = []
    knobs += _tile_knobs(target, space, reason, store, capacity, roles, role_reason, working_set)
    knobs += _layer_knobs(target, capacity, working_set)
    knobs += _placement_knobs(target, units, unit_reason, roles, role_reason)
    knobs += _global_knobs(target, units, unit_reason, space, reason)

    notes: dict[str, str] = {}
    # The stores_status distinction is carried through verbatim rather than folded into "no capacity":
    # a target whose extractor RAN and found no on-chip store is a fact about the device, while one
    # whose artifact could not be read is a fact about our extraction, and a report that conflates them
    # cannot tell a scalar CPU from a broken toolchain.
    if space is not None:
        notes["stores_status"] = getattr(space, "stores_status", "")
        for u in getattr(space, "unknowns", ()) or ():
            notes[f"address_space.{u.quantity}"] = u.reason
    if reason:
        notes["address_space"] = reason
    if unit_reason:
        notes["compute_units"] = unit_reason
    if role_reason:
        notes["endpoint_roles"] = role_reason
    if working_set is None:
        notes["working_set"] = ("not supplied: the two program-scoped bounds (staging regime, fusion "
                                "depth) are reported UNKNOWN rather than guessed")
    return KnobInventory(target=target, knobs=tuple(knobs), notes=notes)


# ---- enforcement: is a proposed change ON a declared surface? ----------------------------------

@dataclass(frozen=True)
class Finding:
    """One key of a proposed change and what became of it. ``outside`` names WHAT was outside — the
    requirement is that a refusal be attributable, so "refused" alone is not an answer."""

    key: str
    value: Any
    state: str
    outside: str = ""
    surface: str | None = None

    def to_dict(self) -> dict:
        out = {"key": self.key, "value": repr(self.value), "state": self.state}
        if self.outside:
            out["outside"] = self.outside
        if self.surface:
            out["surface"] = self.surface
        return out


@dataclass(frozen=True)
class Verdict:
    """The three-state answer for a whole proposed change.

    AGGREGATION RULE, said out loud because it is the part that is easy to get backwards: a single
    REFUSED key refuses the whole change even when other keys are undeterminable. A refused key is a
    DEFINITE violation — the change touches something no surface exposes, or a value the target
    provably cannot take — and a definite violation is not softened by an unrelated unknown. Only when
    nothing is definitely wrong and something could not be decided is the verdict UNDETERMINABLE, and
    that is never reported as success: a caller that treats it as ALLOWED reproduces the check that
    could not run and said it passed.
    """

    state: str
    findings: tuple[Finding, ...] = ()
    target: str = ""

    @property
    def allowed(self) -> bool:
        return self.state == ALLOWED

    def refusals(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.state == REFUSED)

    def undeterminable(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.state == UNDETERMINABLE)

    def reason(self) -> str:
        """One line per non-allowed key, naming the key and what was outside. Empty when ALLOWED."""
        parts = [f"{f.key}={f.value!r}: {f.state} — {f.outside}"
                 for f in self.findings if f.state != ALLOWED]
        return "; ".join(parts)

    def surfaces_touched(self) -> tuple[str, ...]:
        return tuple(sorted({f.surface for f in self.findings if f.surface}))

    def to_dict(self) -> dict:
        return {"state": self.state, "target": self.target, "reason": self.reason(),
                "findings": [f.to_dict() for f in self.findings]}


def check_mutation(proposed: Mapping[str, Any], inventory: KnobInventory) -> Verdict:
    """Does ``proposed`` lie on a declared CCA surface? ALLOWED / REFUSED / UNDETERMINABLE.

    ``proposed`` is ``{knob name: value}`` — the shape a fork's overrides already take. Three ways a
    key lands:

    * **not in the inventory** -> REFUSED, naming the key and stating that no declared surface exposes
      it. This is the falsifier: it is what stops the optimization phase reaching a transform, a
      heuristic or a pass flag by a route nobody can attribute a score change to.
    * **in the inventory, domain UNKNOWN** -> UNDETERMINABLE, carrying the reason the domain could not
      be derived. NOT refused (the knob is real and on a surface) and NOT allowed (nothing checked it).
    * **in the inventory, value outside the derived domain** -> REFUSED, naming the value AND the
      domain, so the refusal is actionable rather than a bare no.

    An EMPTY proposal is ALLOWED with no findings: proposing nothing changes nothing, and calling that
    a refusal would make "the phase made no change this round" indistinguishable from a violation.
    """
    findings: list[Finding] = []
    for key, value in dict(proposed).items():
        knob = inventory.get(key)
        if knob is None:
            findings.append(Finding(
                key=key, value=value, state=REFUSED,
                outside=(f"no declared CCA surface exposes {key!r}; the declared surfaces are "
                         f"{list(SURFACES)} and their knobs are {list(inventory.names())}")))
            continue
        admits = knob.admits(value)
        if admits is None:
            findings.append(Finding(
                key=key, value=value, state=UNDETERMINABLE, surface=knob.surface,
                outside=(f"{key!r} is on surface {knob.surface!r} but its domain could not be derived "
                         f"for {inventory.target!r}: {knob.domain.why_unknown}")))
        elif admits:
            findings.append(Finding(key=key, value=value, state=ALLOWED, surface=knob.surface))
        else:
            findings.append(Finding(
                key=key, value=value, state=REFUSED, surface=knob.surface,
                outside=(f"{value!r} is outside the derived domain {knob.domain.describe()} of "
                         f"{key!r} (source: {knob.domain.source})")))
    if any(f.state == REFUSED for f in findings):
        state = REFUSED
    elif any(f.state == UNDETERMINABLE for f in findings):
        state = UNDETERMINABLE
    else:
        state = ALLOWED
    return Verdict(state=state, findings=tuple(findings), target=inventory.target)


# ---- attribution: which knob is a measured change actually due to? -----------------------------

@dataclass
class Attribution:
    """One knob change, and the effect measured for it — or the honest absence of one.

    ``status`` is ``"unmeasured"`` until :meth:`AttributionLedger.observe` supplies a before/after from
    a real compile+run. It is NOT initialised to a zero delta: a zero delta means "this knob was moved
    and changed nothing", which is a finding, while an unmeasured entry means nothing has been run yet,
    and a ledger that renders the second as the first is the check that reports success because it
    could not execute.
    """

    knob: str
    surface: str
    before: Any
    after: Any
    status: str = "unmeasured"
    metric: str = ""
    metric_before: float | None = None
    metric_after: float | None = None
    note: str = ""

    @property
    def delta(self) -> float | None:
        if self.metric_before is None or self.metric_after is None:
            return None
        return self.metric_after - self.metric_before

    def to_dict(self) -> dict:
        return {"knob": self.knob, "surface": self.surface,
                "before": repr(self.before), "after": repr(self.after),
                "status": self.status, "metric": self.metric,
                "metric_before": self.metric_before, "metric_after": self.metric_after,
                "delta": self.delta, "note": self.note}


class AttributionLedger:
    """Records WHICH knob each mutation was, so a later measurement has something to attribute to.

    HONESTY, stated where it cannot be missed: this class does not measure anything. Attributing a
    score change to a knob requires compiling and running both sides, which this layer does not do —
    so every entry is born ``unmeasured`` and :meth:`by_surface` reports the unmeasured count BESIDE
    the measured totals rather than summing over what it has. Until a run calls :meth:`observe`, the
    honest answer to "how much of the win is dispatch?" is "nothing has been measured", and that is
    what this returns.

    A mutation is only recorded if it is ALLOWED: recording a refused change would put a knob name on
    a mutation that never lay on a surface, which is the attribution failure this whole layer exists
    to prevent.
    """

    def __init__(self, inventory: KnobInventory) -> None:
        self.inventory = inventory
        self.entries: list[Attribution] = []
        self.rejected: list[Verdict] = []

    def record(self, proposed: Mapping[str, Any], *, before: Mapping[str, Any] | None = None,
               note: str = "") -> Verdict:
        """Check ``proposed`` and, if ALLOWED, open one unmeasured entry per knob it moves."""
        verdict = check_mutation(proposed, self.inventory)
        if not verdict.allowed:
            self.rejected.append(verdict)
            return verdict
        prior = dict(before or {})
        for key, value in dict(proposed).items():
            knob = self.inventory.get(key)
            self.entries.append(Attribution(
                knob=key, surface=knob.surface if knob else "",
                before=prior.get(key, knob.default if knob else None),
                after=value, note=note))
        return verdict

    def observe(self, knob: str, *, metric: str, metric_before: float, metric_after: float) -> bool:
        """Attach a REAL measurement to the most recent unmeasured entry for ``knob``.

        Returns False when there is no such entry — a measurement with nothing to attribute it to is
        dropped and reported, never invented an entry for.
        """
        for entry in reversed(self.entries):
            if entry.knob == knob and entry.status == "unmeasured":
                entry.status = "measured"
                entry.metric = metric
                entry.metric_before = float(metric_before)
                entry.metric_after = float(metric_after)
                return True
        return False

    def by_surface(self) -> dict:
        """Attributed effect per surface — the answer to "where should the next round spend effort?".

        Returns measured totals AND the unmeasured count per surface. A surface with measured=0 and
        unmeasured=7 is not a surface that cost nothing; it is seven changes nobody has run yet, and
        the caller must be able to tell those apart before deciding a tile is worth 3%.
        """
        out: dict[str, dict] = {s: {"measured_delta": 0.0, "n_measured": 0, "n_unmeasured": 0,
                                    "knobs": []} for s in SURFACES}
        for e in self.entries:
            bucket = out.setdefault(e.surface or "unattributed",
                                    {"measured_delta": 0.0, "n_measured": 0, "n_unmeasured": 0,
                                     "knobs": []})
            if e.knob not in bucket["knobs"]:
                bucket["knobs"].append(e.knob)
            if e.status == "measured" and e.delta is not None:
                bucket["measured_delta"] += e.delta
                bucket["n_measured"] += 1
            else:
                bucket["n_unmeasured"] += 1
        total_measured = sum(b["n_measured"] for b in out.values())
        return {"by_surface": out,
                "n_entries": len(self.entries),
                "n_measured": total_measured,
                "n_rejected": len(self.rejected),
                "status": ("measured" if total_measured else "unmeasured"),
                "why": ("" if total_measured else
                        "no entry has been observed: attributing a score change to a knob requires "
                        "compiling and running both sides, which this layer records but does not do")}

    def to_dict(self) -> dict:
        return {"target": self.inventory.target,
                "entries": [e.to_dict() for e in self.entries],
                "rejected": [v.to_dict() for v in self.rejected],
                "summary": self.by_surface()}
