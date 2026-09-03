"""Joint occupancy from per-cycle traces: what was busy together, and what only looked like it.

An occupancy vector answers the one question a *partitioned* activity source cannot: did two units
run in the same cycle? A partition charges every cycle to exactly one owner, so it reports zero
overlap whether or not the hardware overlaps (:func:`merlin.perf.headroom.composition_operator`
refuses such a source for exactly that reason). A per-cycle trace can answer it -- but only after
three ways of fabricating an answer are removed, each of which was observed on real traces:

**A signal counted beside its own components.** A trace may carry a unit's busy signal *and* the
sub-signals it is composed of. Counting them together reports the unit overlapping with itself. On
one measured design this alone produced 204 cycles of "overlap" in a kernel with none.
:func:`subsumed_columns` removes it by containment over the observed cycles, which needs no
knowledge of what the signals are called.

**A unit with no busy port, read as permanently idle.** Not every unit is exposed as a top-level
port; some are only visible through an internal state register. Left out, such a unit contributes no
busy cycles, so it inflates the idle figure and its overlap is unobservable *by construction* -- the
same failure as the partition, one level down. On one measured corpus the busiest unit on the vector
kernels had no port, and including it moved a kernel's idle fraction from 89.9% to 39.2%.
:func:`calibrate_state_idle` recovers those units without assuming what value means idle.

**Two instruments' views of one unit, merged as two units.** Combining traces from two engines
double-counts whatever both can see, and since the two sample at different points in the cycle the
duplicates land in *adjacent* cycles and read as overlap. :func:`merge_engines` derives the sampling
offset instead of assuming it, and admits a column from the second engine only if it carries a cycle
the first could not see.

**An engine nested inside another, folded away as if it were a sub-signal.** This is the one thing
containment CANNOT decide, because both cases look identical in the data: a unit's load and store
halves nest inside it (fold -- one unit), and so does an accelerator embedded in the cluster that
drives it (do not fold -- two engines, and their concurrency is the measurement). So the unit a
column belongs to is DECLARED, via ``unit_of``, and columns in different declared units are never
folded into each other. On a heterogeneous device -- a SIMT cluster containing a systolic array,
say -- deriving it instead deletes the inner engine and reports zero overlap between the two.

**A contract's compute units are not the engine set.** A capability contract declares what a compiler
routes against — arithmetic units. The engines an overlap term is a property of include the command
controllers that move the data, and on the interlocked target here the contract named one unit while
the design carries three decoupled controllers, so the overlap term came out unidentifiable from the
declaration alone. :func:`derived_engines` recovers them from the target's OWN RTL (a module owning a
detected control FSM *and* exposing a completion channel), :func:`engine_set` unions that with the
declaration, and the one fact neither can derive — whether a declared unit and a derived controller
are the same engine seen twice — is DECLARED per unit and cross-checked, never guessed.

Nothing here names a target, a unit, an opcode or a bit-width: every rule is a property of the
measurement, and the two facts that are NOT properties of the measurement -- what kind a unit is,
and which unit a column belongs to -- are declared by the producer rather than guessed. A column
whose meaning cannot be established stays out of the joint counts and is reported as unmeasured --
never defaulted to idle, which is the reading that flatters the result.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

__all__ = [
    "COMPLETION_FIELD", "DERIVED_KIND", "ENGINE_RULE", "Occupancy", "align_offset",
    "calibrate_state_idle", "declared_engines", "derived_engines", "engine_set", "joint_counts",
    "merge_engines", "subsumed_columns", "unit_bindings",
]


class Occupancy(dict):
    """``{column: [busy per cycle]}`` -- a joint occupancy vector over one program."""

    @property
    def cycles(self) -> int:
        return len(next(iter(self.values()))) if self else 0


def subsumed_columns(hot: Mapping[str, Sequence[bool]],
                     prefer=lambda a, b: False,
                     unit_of: Mapping[str, str] | None = None) -> dict[str, str]:
    """Columns that are a sub-signal or a duplicate of another, derived from the trace itself.

    ``unit_of`` maps a column to the DECLARED unit it belongs to, and two columns declared to
    different units are never folded into each other however their busy cycles nest. That
    distinction cannot be derived, because containment in the data looks identical either way: a
    unit's own load and store halves nest inside it (fold -- they are one unit), and so does an
    accelerator embedded in the cluster that drives it (do NOT fold -- they are two engines that can
    run at once). Deriving it would silently delete the inner engine and report zero overlap between
    the two, which on a heterogeneous device is the single quantity worth measuring.

    ``a`` is subsumed by ``b`` when every cycle ``a`` is high ``b`` is high too and ``a`` is high at
    least once. Containment must hold on *every* cycle, so a column that merely correlates is not
    subsumed. Where two columns are cycle-identical exactly one survives, chosen by ``prefer(a, b)``
    -- return True to keep ``a``.

    Returns ``{subsumed: subsumer}``; a column absent from the result is independent.
    """
    out: dict[str, str] = {}
    cols = list(hot)
    units = unit_of or {}
    for a in cols:
        n_a = sum(hot[a])
        if n_a == 0:
            continue
        for b in cols:
            if a == b or b in out:              # never fold a column into an already-folded one
                continue
            ua, ub = units.get(a), units.get(b)
            if ua is not None and ub is not None and ua != ub:
                continue                        # separately declared engines; nesting is structure
            n_b = sum(hot[b])
            if n_b < n_a:
                continue
            if n_b == n_a and (list(hot[a]) != list(hot[b]) or prefer(a, b)):
                continue                        # not a duplicate, or `a` is the keeper
            if all((not hot[a][i]) or hot[b][i] for i in range(len(hot[a]))):
                out[a] = b
                break
    return out


def declared_engines(contract: Mapping) -> dict[str, dict]:
    """The target's own declaration of its engines: ``{name: {kind, contains}}``.

    Read from the contract's compute-unit declaration rather than restated here, because that
    declaration is the one the RTL audit already corrected -- on at least one target it originally
    described a cluster while the silicon also held a systolic array embedded inside it, and
    ``contains`` is how that composition is expressed. An occupancy vector that re-derived the unit
    set would re-make exactly that mistake.

    A device with one engine yields one entry; the rules elsewhere in this module do not care how
    many there are, only that columns belonging to different ones are never folded together.
    """
    from merlin.targetgen.compute_units import compute_units

    return {u.name: {"kind": u.kind, "contains": tuple(u.contains)}
            for u in compute_units(dict(contract))}


#: The port field whose presence makes a module's work separately ATTRIBUTABLE. A completion channel
#: carries the id of the command that finished, so a consumer can say WHICH work a unit ended -- the
#: difference between an engine that can be scheduled against and a sub-unit that is merely active.
#: It is a parameter, not a law: a target whose engines signal completion under another field name is
#: served by passing a different one, which is why nothing downstream reads this constant.
COMPLETION_FIELD = "completed"

#: What a derived engine's ``kind`` says. NOT an archetype from the compute-unit vocabulary: the
#: derivation establishes that a module sequences and completes its own work, and deliberately does
#: NOT establish whether that work is arithmetic or data movement -- deciding that from the module's
#: spelling is precisely the assumption this derivation exists to avoid. The role stays UNKNOWN and
#: says so, rather than being defaulted to the kind the reader expects.
DERIVED_KIND = "control"

#: The bar a module must clear to count as an engine, stated once so a record can quote it.
ENGINE_RULE = ("a module of the target's own elaboration that BOTH owns a control-state register the "
               "FSM extraction detected (it sequences its own work) AND exposes a completion channel "
               "in its elaborated port list (its work can be attributed to a command). Either half "
               "alone is not an engine: an FSM with no completion port is a sub-sequencer nested "
               "inside another engine's datapath, and a completion port with no FSM is a wrapper or "
               "a command tracker")


def derived_engines(target: str | None, *, fsm_registers=None, ports: Mapping | None = None,
                    completion_field: str = COMPLETION_FIELD) -> tuple[dict[str, dict], dict]:
    """Engines the target's OWN RTL evidences, for a contract that names only its compute units.

    A capability contract declares COMPUTE units, because that is what a compiler routes against. The
    engines whose *concurrency* an analytical model has to price are a different set and a larger one:
    on the interlocked target here the contract declares a single arithmetic unit while the design
    carries three decoupled command controllers, so an overlap term -- the only interesting
    performance term the device has -- came out unidentifiable from the declaration alone. Restating
    the three in code would fix that record and break the next target; they are derivable, from two
    extractions this repo already runs, and :data:`ENGINE_RULE` is what the two together establish.

    Both halves are needed and neither is a name match. Measured on that target: 14 modules own a
    detected control FSM and 6 expose a completion channel; the intersection is exactly the three
    controllers, while the FSM-only modules are loop sequencers and DMA sub-units living inside them.

    ``fsm_registers`` / ``ports`` may be supplied (an :class:`~merlin.targetgen.rtl.fsm.FsmRegister`
    sequence and a :func:`~merlin.targetgen.rtl.ports.port_facts` record) or left ``None`` to be read
    for ``target``. An EMPTY ``fsm_registers`` is not the same as ``None``: it means the extraction
    ran and found nothing, and both are reported as UNKNOWN rather than as "this design has no
    further engines" -- the reading that flatters the result, since a machine nobody analysed then
    looks like a machine with nothing in it.

    Returns ``({name: {kind, contains, basis}}, basis)`` where the second is the derivation's own
    account: its status, the rule, and every candidate it REFUSED with the reason it refused it.
    """
    basis: dict = {"rule": ENGINE_RULE, "target": target, "completion_field": completion_field,
                   "status": "unknown", "engines": [], "refused": {}}
    if not target:
        basis["why"] = ("no target was named, so no RTL fact bundle could be opened. Whether this "
                        "device has engines its contract does not name is UNKNOWN, not none")
        return {}, basis

    if fsm_registers is None:
        try:
            from merlin.targetgen.rtl.fsm import fsm_inventory
            fsm_registers = list(fsm_inventory(target))
        except (OSError, ImportError) as exc:
            fsm_registers = None
            basis["fsm_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
    if ports is None:
        try:
            from merlin.targetgen.rtl.ports import port_facts
            ports = port_facts(target, fields=(completion_field,))
        except Exception as exc:                                                   # noqa: BLE001
            ports = {"status": "unavailable", "why": f"{type(exc).__name__}: {str(exc)[:160]}"}

    by_module: dict[str, list[str]] = {}
    for reg in fsm_registers or ():
        module = str(getattr(reg, "module", "") or "")
        if module:
            by_module.setdefault(module, []).append(str(getattr(reg, "qualified", module)))
    basis["fsm_modules"] = sorted(by_module)
    basis["n_fsm_registers"] = sum(len(v) for v in by_module.values())

    port_status = str((ports or {}).get("status") or "unavailable")
    basis["ports_status"] = port_status
    basis["ports_dialect"] = (ports or {}).get("dialect")
    if port_status != "derived":
        basis["why"] = (f"this target's elaboration could not be read "
                        f"({(ports or {}).get('why', 'no port facts')}), so no module's ports could "
                        f"be checked for a {completion_field!r} channel. Engines beyond the "
                        f"contract's declaration are UNKNOWN here, which is NOT the same as the "
                        f"design having none")
        return {}, basis
    if fsm_registers is None:
        basis["why"] = ("the control-FSM extraction could not be read, so no module could be shown "
                        "to sequence its own work. UNKNOWN, not none: this is a statement about the "
                        "extraction and not about the design")
        return {}, basis

    field = ((ports.get("fields") or {}).get(completion_field)) or {}
    completing = {str(m) for m in (field.get("modules") or ())}
    handshaken = {str(m) for m in (field.get("decoupled") or ())}
    basis["n_modules_read"] = ports.get("n_modules")
    basis["modules_completing"] = sorted(completing)
    basis["modules_completing_decoupled"] = sorted(handshaken)

    if not by_module:
        basis["why"] = ("no control-state register was found for this target, so no module could be "
                        "shown to sequence its own work. An absent FSM extraction is a statement "
                        "about the extraction and NOT about the design: this target's engines "
                        "beyond its declaration stay UNKNOWN")
        return {}, basis

    engines: dict[str, dict] = {}
    for module, regs in sorted(by_module.items()):
        if module in completing:
            engines[module] = {
                "kind": DERIVED_KIND, "contains": (), "rtl_module": module,
                "basis": (f"DERIVED from this target's own RTL, not declared: {module} owns "
                          f"control-state register(s) {sorted(regs)} that the FSM extraction "
                          f"detected, and its port list in the elaboration exposes a "
                          f"{completion_field!r} channel"
                          + (" as a ready/valid handshake, so a completion carries the id of the "
                             "command it ends" if module in handshaken else
                             " without a ready/valid handshake, so it signals completion but does "
                             "not tag which command finished")
                          + ". Its ROLE -- arithmetic or data movement -- is deliberately NOT "
                            "established: deciding it would mean reading the module's name, which "
                            "is the assumption this derivation exists to avoid")}
            continue
        basis["refused"][module] = (
            f"owns control-state register(s) {sorted(regs)} but exposes no {completion_field!r} "
            f"channel in the elaboration, so nothing can attribute its work to a command. On this "
            f"evidence it is a sub-sequencer inside another engine rather than an engine of its "
            f"own -- UNKNOWN either way, and NOT counted as one")
    basis["status"] = "derived"
    basis["engines"] = sorted(engines)
    basis["why"] = (f"{len(engines)} engine(s) derived: of {len(by_module)} module(s) owning a "
                    f"detected control FSM, {len(engines)} also expose a {completion_field!r} "
                    f"channel in {ports.get('n_modules')} module(s) of the elaboration "
                    f"({ports.get('dialect')} dialect)")
    if not engines:
        basis["why"] += (". No module clears both halves of the rule, so whether this design has "
                         "engines its contract does not name stays UNKNOWN -- the extractions ran "
                         "and disagree with neither reading")
    return engines, basis


def engine_set(contract: Mapping, *, target: str | None = None, fsm_registers=None,
               ports: Mapping | None = None) -> tuple[dict[str, dict], dict]:
    """The engine set an overlap term is defined over: what the contract DECLARES, widened by what
    the RTL EVIDENCES.

    The union is the point. A contract declares the compute units a compiler routes against; the RTL
    carries the movement/control engines whose concurrency with those units is the measurement. Taking
    either alone loses something real -- the declaration loses the controllers, and the derivation
    loses the compute unit's archetype, its dtypes and its composition, which no port list carries.

    ONE ALIASING QUESTION, ASKED RATHER THAN GUESSED. A declared compute unit and a derived controller
    can be the same engine seen from two sides (the datapath, and the FSM that sequences it), and
    unioning them blind counts one engine twice -- inventing a pair whose "overlap" is a unit
    overlapping itself. Which module realises a declared unit cannot be derived (it is a name match,
    and the containment in the data looks identical either way), so it is DECLARED, per unit, as
    ``rtl_module`` in the contract -- and cross-checked here against the elaboration rather than
    trusted: a module the elaboration does not contain is reported as a mismatch, and a unit that
    declares nothing is reported as an unresolved alias rather than silently merged or silently
    doubled.

    Returns ``(engines, basis)`` in the shape :func:`declared_engines` returns, plus the derivation's
    own account.
    """
    engines = {name: dict(spec) for name, spec in declared_engines(contract).items()}
    target = target or str((contract or {}).get("name") or "") or None
    derived, basis = derived_engines(target, fsm_registers=fsm_registers, ports=ports)

    claimed: dict[str, str] = {}
    for raw in ((contract or {}).get("compute_units") or ()):
        unit, module = str(raw.get("name") or ""), str(raw.get("rtl_module") or "")
        if unit in engines and module:
            claimed[module] = unit
    basis["declared_aliases"] = dict(sorted(claimed.items()))

    known_modules = set(basis.get("fsm_modules") or ()) | set(basis.get("modules_completing") or ())
    unresolved: list[str] = []
    for unit, spec in engines.items():
        module = next((m for m, u in claimed.items() if u == unit), "")
        if module and module in derived:
            spec["rtl_module"] = module
            spec["basis"] = (f"DECLARED by the contract, and its ``rtl_module`` {module!r} "
                             f"CROSS-CHECKED against this target's own elaboration: that module "
                             f"clears the derived-engine rule, so this unit and it are one engine "
                             f"and are counted once")
        elif module and module in known_modules:
            spec["rtl_module"] = module
            spec["basis"] = (f"DECLARED by the contract; its ``rtl_module`` {module!r} IS in this "
                             f"target's elaboration but does not clear the derived-engine rule "
                             f"({basis['refused'].get(module, 'it owns no detected control FSM')}), "
                             f"so the alias holds and no derived engine is folded into it")
        elif module and basis.get("status") == "derived":
            spec["rtl_module"] = module
            spec["basis"] = (f"DECLARED by the contract, and its ``rtl_module`` {module!r} FAILS the "
                             f"cross-check: this target's own elaboration was read and contains no "
                             f"such module. The declaration is recorded and NOT corroborated")
        elif module:
            spec["rtl_module"] = module
            spec["basis"] = (f"DECLARED by the contract; its ``rtl_module`` {module!r} could not be "
                             f"cross-checked because the RTL derivation is UNKNOWN here "
                             f"({basis.get('why', '')})")
        elif derived:
            unresolved.append(unit)
            spec["basis"] = (f"DECLARED by the contract, which does not say which module of the "
                             f"elaboration realises it. Whether one of the derived engine(s) "
                             f"{sorted(derived)} IS this unit is UNKNOWN, so they are kept apart -- "
                             f"which counts one engine twice if any of them is this one. Declaring "
                             f"``rtl_module`` on this unit resolves it")
        else:
            spec["basis"] = ("DECLARED by the contract; no engine beyond the declaration was derived "
                             f"({basis.get('why', '')})")
    basis["unresolved_aliases"] = sorted(unresolved)

    for name, spec in derived.items():
        if name in claimed:
            continue
        engines[claimed.get(name, name)] = dict(spec)
    basis["n_declared_units"] = len(declared_engines(contract))
    basis["n_after_union"] = len(engines)
    return engines, basis


def unit_bindings(columns: Sequence[str], binding: Mapping[str, str],
                  engines: Mapping[str, dict]) -> tuple[dict[str, str], list[str]]:
    """``(unit_of, unbound)`` for ``columns``, validated against the declared engines.

    ``binding`` is the producer's column -> engine map: which engine each traced signal belongs to.
    It is declared, never inferred from a signal's spelling. A binding naming an engine the contract
    does not declare is an error worth raising, because it means the trace and the contract disagree
    about what the device is; a column with no binding is merely unbound, and is returned so the
    caller can report it rather than quietly folding it somewhere.
    """
    unknown = sorted({e for e in binding.values() if e not in engines})
    if unknown:
        raise ValueError(f"trace binds column(s) to undeclared engine(s) {unknown}; "
                         f"the contract declares {sorted(engines)}")
    unit_of = {c: binding[c] for c in columns if c in binding}
    return unit_of, [c for c in columns if c not in binding]


def calibrate_state_idle(traces: Sequence[Mapping[str, Sequence[str]]],
                         state_columns: Sequence[str],
                         port_columns: Sequence[str]) -> dict:
    """Which value of a state column means *idle*, derived across a whole corpus.

    ``0 == idle`` is exactly the kind of baked encoding constant that must not be guessed, and it
    does not have to be: a unit exposing BOTH a busy port and its internal state pins the encoding,
    because the idle value is the one high on precisely the cycles the port is low. That is a
    cycle-exact identity, and it either holds or it does not.

    The calibration is taken over the WHOLE corpus rather than per trace, because the encoding is a
    property of the design: a program that never exercises the paired unit leaves its state constant
    and must not be allowed to withdraw a calibration the rest of the corpus established. Per-trace
    calibration was observed to drop the busiest unit from the vector on exactly the programs where
    it mattered.

    Returns ``{"idle_value", "paired_with", "paired_columns", "checked_traces", "detail"}``. When
    ``idle_value`` is ``None`` nothing pinned the encoding: every unpaired state column then stays
    OUT of the occupancy vector and is reported unmeasured, rather than assumed idle.
    """
    pairings: dict[str, set[tuple[str, str]]] = {}
    checked = 0
    for tr in traces:
        rows_n = len(next(iter(tr.values()))) if tr else 0
        if not rows_n:
            continue
        checked += 1
        hot = {c: [tr[c][i] not in ("0", "") for i in range(rows_n)]
               for c in port_columns if c in tr}
        for s in state_columns:
            if s not in tr:
                continue
            vals = set(tr[s])
            if len(vals) < 2:
                continue                        # constant here; another trace may still settle it
            for port, ph in hot.items():
                if not any(ph):
                    continue
                for cand in sorted(vals):
                    if all((tr[s][i] != cand) == ph[i] for i in range(rows_n)):
                        pairings.setdefault(s, set()).add((cand, port))
    idle_values = {iv for prs in pairings.values() for iv, _ in prs}
    if len(idle_values) != 1:
        return {"idle_value": None, "paired_with": None, "paired_columns": [],
                "checked_traces": checked,
                "detail": ("no state column pairs cycle-exactly with a busy port" if not idle_values
                           else f"paired columns disagree on the idle value ({sorted(idle_values)});"
                                " refusing to pick one")}
    return {"idle_value": idle_values.pop(),
            "paired_with": sorted({p for prs in pairings.values() for _, p in prs}),
            "paired_columns": sorted(pairings), "checked_traces": checked,
            "detail": ("cycle-exact for the state columns that have a busy port; applying it to a "
                       "column with no port is an INFERENCE from a shared encoding convention, not "
                       "a measurement")}


def align_offset(a: Mapping[str, Sequence[bool]], b: Mapping[str, Sequence[bool]],
                 candidates: Sequence[int] = (0, 1, -1, 2, -2)) -> tuple[int, int]:
    """The sampling offset between two instruments, DERIVED as the shift aligning the most columns.

    Two engines sample the same signal at different points in the cycle, so a fixed offset separates
    their traces. Assuming zero makes every shared unit look like two units busy in adjacent cycles,
    which reports overlap on a machine that has none. Returns ``(shift, columns_aligned)``; a
    ``columns_aligned`` of 0 means nothing pinned the offset and the caller should not merge.
    """
    n = min((len(v) for v in list(a.values()) + list(b.values())), default=0)
    best, hits_best = 0, -1
    for shift in candidates:
        lo, hi = max(0, -shift), min(n, n - shift)
        if lo >= hi:
            continue
        hits = 0
        for cb in b.values():
            if not any(cb):
                continue
            for ca in a.values():
                if all(ca[i] == cb[min(max(i + shift, 0), n - 1)] for i in range(lo, hi)):
                    hits += 1
                    break
        if hits > hits_best:
            best, hits_best = shift, hits
    return best, max(hits_best, 0)


def merge_engines(primary: Mapping[str, Sequence[bool]],
                  secondary: Mapping[str, Sequence[bool]]) -> tuple[Occupancy, dict]:
    """Merge two instruments' occupancy vectors, keeping only what each independently contributes.

    A column from ``secondary`` is admitted only if it carries a cycle ``primary`` could not see. A
    signal whose busy cycles are wholly contained in what ``primary`` already reports is a second
    *view* of the same activity, not a second unit -- and admitting it manufactures overlap against
    the very columns that contain it. An aggregate bus-valid signal beside the per-channel ports of
    the same bus does precisely this, and was measured reporting 6.8% overlap on a corpus where no
    two distinct units are ever busy together.

    Returns ``(merged, provenance)`` with the derived shift, what was folded, and what was added.
    """
    shift, aligned = align_offset(primary, secondary)
    n = min((len(v) for v in list(primary.values()) + list(secondary.values())), default=0)
    lo, hi = max(0, -shift), min(n, n - shift)
    covered = [any(primary[c][i] for c in primary) for i in range(n)]

    merged = Occupancy({c: list(primary[c][:n]) for c in primary})
    folded: dict[str, str] = {}
    added: list[str] = []
    for c, col_raw in secondary.items():
        col = [col_raw[min(max(i + shift, 0), n - 1)] for i in range(n)]
        if not any(col):
            continue
        same = next((x for x in primary
                     if all(primary[x][i] == col[i] for i in range(lo, hi))), None)
        if same is not None:
            folded[c] = same
            continue
        if all((not col[i]) or covered[i] for i in range(n)):
            folded[c] = "<covered by the other instrument>"
            continue
        merged[c] = col
        added.append(c)
    return merged, {"shift": shift, "columns_aligned_by_shift": aligned,
                    "folded": folded, "added": added}


def joint_counts(hot: Mapping[str, Sequence[bool]],
                 kinds: Mapping[str, str] | None = None,
                 unit_of: Mapping[str, str] | None = None) -> dict:
    """Idle, overlap and per-column busy over an occupancy vector, after subsumption.

    ``overlap_across_kinds`` counts only columns whose kind the producer DECLARED, so it is a lower
    bound whenever any column is undeclared -- reported, never silently absorbed into one side.
    """
    subsumed = subsumed_columns(hot, unit_of=unit_of)
    cols = [c for c in hot if c not in subsumed]
    n = len(next(iter(hot.values()))) if hot else 0
    kinds = kinds or {}
    undeclared = sorted(c for c in cols if c not in kinds)

    idle = ovl = ovl_kind = 0
    for i in range(n):
        live = [c for c in cols if hot[c][i]]
        if not live:
            idle += 1
        if len(live) >= 2:
            ovl += 1
        if len({kinds[c] for c in live if c in kinds}) >= 2:
            ovl_kind += 1
    # Overlap needs two things that can overlap. A vector with fewer than two live columns reports
    # zero by construction -- arithmetically right and evidentially empty -- and that zero is
    # indistinguishable from a machine that genuinely serialises unless the distinction is carried.
    # A column that is constant across the run is not live: nothing was observed of it either way.
    live = [c for c in cols if any(hot[c]) and not all(hot[c])] or [c for c in cols if any(hot[c])]
    return {"sampled_cycles": n, "joint_columns": cols, "subsumed_columns": subsumed,
            "overlap_observable": len(live) >= 2,
            "live_columns": live,
            "busy": {c: sum(hot[c]) for c in hot}, "idle_cycles": idle, "overlap_any": ovl,
            "overlap_across_kinds": ovl_kind,
            "overlap_across_kinds_is_lower_bound": bool(undeclared),
            "undeclared_columns": undeclared,
            "unbound_columns": sorted(c for c in cols if c not in (unit_of or {}))}
