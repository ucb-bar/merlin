"""The OPTIMIZATION SURFACE: every place a target can be changed, as machine-readable data.

WHY. An agent asked to improve a compiler otherwise has to grep it to find out where an optimization
belongs, and a search that cannot enumerate its own action space cannot report what it did NOT try.
Both are how a campaign ends up reporting "we explored the space" for a space nobody wrote down.

DERIVED, NOT AUTHORED. Every entry is assembled from registries that already exist and are already
checked -- ``kernels.regions`` (which seams exist, in which phase, and whether each is forkable),
``kernels.action_catalog`` (which axes actually route for this backend, at which mechanism class,
with what legality and rebuild cost), and ``kernels.cca_contract`` (which axes are levers at all). A
hand-written surface would be a fourth thing to keep in agreement with the other three, and the
recurring failure in this tree is exactly that: a list maintained by hand that silently stops
matching what it describes.

THE FIELD THAT MAKES GATE 2 MECHANICAL is ``inspect_emitted``. An entry says not only how to apply a
change and how to validate it, but how to tell from the EMITTED code whether the change actually
happened. Without it, "did the fork do what it promised" is a judgement call, and the loop credits
actions that compiled and changed nothing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SurfaceEntry:
    """One editable place, and everything needed to use it without reading the compiler."""

    #: stable id: "<region>/<seam>". Stable across runs so a ledger can reference it.
    seam_id: str
    region: str
    phase: str
    #: FLAG | KNOB | HEURISTIC | PASS | CODEGEN | RUNTIME, or the EditPoint kinds REGISTRY/DATA/DIALECT.
    mechanism: str
    #: kernel | dispatch | program — which question this seam changes the answer to.
    scope: str = "kernel"
    #: CCA axes this seam can move. Empty means the seam exists but governs no declared axis, which
    #: is a real state and a reportable one, not a reason to omit the entry.
    cca_axes: tuple[str, ...] = ()
    #: False = a declared GAP: the place is known, the registrable hook is not there yet.
    forkable_now: bool = True
    #: file to edit, and the register() entry point when the seam is a registry.
    file: str = ""
    registry: str | None = None
    how_to_materialize: str = ""
    #: legality the caller must satisfy, and composition constraints.
    preconditions: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    #: what must be rebuilt for a change here to take effect (see action_catalog.REBUILD_SCOPES).
    rebuild_scope: str = "schedule"
    #: the domain to search, for a tunable seam. None = not numeric.
    parameter_domain: Any | None = None
    #: HOW TO TELL FROM THE EMITTED CODE that the change took effect. This is what makes the
    #: intended-vs-achieved audit mechanical rather than a judgement call.
    inspect_emitted: str = ""
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OptimizationSurface:
    """A target's whole surface, plus what it could not describe."""

    target: str
    entries: tuple[SurfaceEntry, ...] = ()
    #: seams that exist but govern no CCA axis — reported, never dropped.
    ungoverned: tuple[str, ...] = ()
    #: axes that route for this backend but reach no seam here — the mirror gap.
    unreachable_axes: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    def forkable(self) -> tuple[SurfaceEntry, ...]:
        return tuple(e for e in self.entries if e.forkable_now)

    def gaps(self) -> tuple[SurfaceEntry, ...]:
        """Declared gaps: a known place with no registrable hook. The work-item list."""
        return tuple(e for e in self.entries if not e.forkable_now)

    def by_axis(self, axis: str) -> tuple[SurfaceEntry, ...]:
        return tuple(e for e in self.entries if axis in e.cca_axes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "n_entries": len(self.entries),
            "n_forkable": len(self.forkable()),
            "n_gaps": len(self.gaps()),
            "entries": [e.to_dict() for e in self.entries],
            "ungoverned_seams": list(self.ungoverned),
            "unreachable_axes": list(self.unreachable_axes),
            "notes": list(self.notes),
        }


#: Which CCA facet belongs to which scope. Derived from the facet's own meaning rather than declared
#: per seam: `coverage` is a whole-model claim, `dispatch`/`communication`/`layout` describe a launch
#: and its traffic, and everything else describes one region's inner loop.
_FACET_SCOPE = {"coverage": "program", "dispatch": "dispatch",
                "communication": "dispatch", "layout": "dispatch"}


#: Fallback scope by compiler PHASE, for a seam that governs no CCA axis. Without it those seams
#: defaulted to "kernel", so a dispatch-phase seam with no axes read as an inner-loop seam -- the
#: default quietly asserting the narrowest answer where the truth was "not derivable from axes".
_PHASE_SCOPE = {
    "frontend": "program", "global": "program",
    "dispatch": "dispatch", "runtime": "dispatch",
    "kernel-codegen": "kernel", "memory": "kernel", "emission": "kernel",
}


def _scope_of(axes, phase: str = "") -> str:
    """The broadest scope these axes imply; falling back to the PHASE when there are none.

    A seam that moves a program-level axis is a program-level seam whatever else it also touches, so
    this takes the maximum rather than the first match.
    """
    order = {"kernel": 0, "dispatch": 1, "program": 2}
    scopes = [_FACET_SCOPE.get(str(a).split(".", 1)[0], "kernel") for a in axes]
    if not scopes:
        # No axis to derive from. The phase is the next-best DERIVED answer; a cross-cutting or
        # target-gen seam has no natural scope and stays "kernel" rather than being invented.
        return _PHASE_SCOPE.get(phase, "kernel")
    return max(scopes, key=lambda s: order[s])


def build(target: str) -> OptimizationSurface:
    """Assemble ``target``'s optimization surface from the live registries."""
    from merlin.kernels import action_catalog as ac
    from merlin.kernels import regions as rg

    # Routes give the mechanism class, legality and rebuild cost per AXIS for this backend.
    try:
        ac.ensure_backend(target)
    except Exception:  # noqa: BLE001 — an underivable backend still has regions to report
        pass
    routes_by_axis: dict[str, list] = {}
    for r in ac._ROUTES.get(target, []):
        routes_by_axis.setdefault(r.axis, []).append(r)

    entries: list[SurfaceEntry] = []
    ungoverned: list[str] = []
    for key, region in rg.REGIONS.items():
        axes = tuple(region.cca_axes)
        for ep in region.edit_points:
            # The cheapest route touching any of this region's axes supplies the action metadata;
            # a seam with no route keeps the EditPoint's own kind, which is the honest answer.
            cands = [r for a in axes for r in routes_by_axis.get(a, [])]
            best = min(cands, key=lambda r: ac._CLASS_ORDER.get(r.action_class, 99)) if cands else None
            entry = SurfaceEntry(
                seam_id=f"{key}/{ep.seam}",
                region=key,
                phase=region.phase,
                mechanism=(best.action_class if best else ep.kind),
                scope=_scope_of(axes, region.phase),
                cca_axes=axes,
                forkable_now=ep.forkable_now,
                file=ep.file,
                registry=ep.registry,
                how_to_materialize=ep.how_to_add,
                preconditions=(tuple(best.preconditions) if best else ()),
                conflicts=(tuple(best.conflicts) if best else ()),
                rebuild_scope=(best.rebuild_scope if best else "schedule"),
                parameter_domain=(best.parameter_domain if best else None),
                inspect_emitted=(
                    "re-lift the emitted CCA and compare "
                    f"{sorted(best.intended_facet)} (kernels.action_catalog.achieved_residual)"
                    if best and best.intended_facet else
                    "no machine-checkable promise: this seam's effect on the emitted code is not "
                    "expressible as a CCA axis, so an audit here is prose"),
            )
            entries.append(entry)
            if not axes:
                ungoverned.append(entry.seam_id)

    reachable = {a for e in entries for a in e.cca_axes}
    unreachable = tuple(sorted(set(routes_by_axis) - reachable))
    notes = []
    if unreachable:
        notes.append(
            f"{len(unreachable)} axis/axes route for {target} but no region declares them, so an "
            f"action on them names a seam nobody can point at")
    return OptimizationSurface(target=target, entries=tuple(entries),
                               ungoverned=tuple(sorted(ungoverned)),
                               unreachable_axes=unreachable, notes=tuple(notes))
