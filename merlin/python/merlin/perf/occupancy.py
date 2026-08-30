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

Nothing here names a target, a unit, an opcode or a bit-width: every rule is a property of the
measurement, and the two facts that are NOT properties of the measurement -- what kind a unit is,
and which unit a column belongs to -- are declared by the producer rather than guessed. A column
whose meaning cannot be established stays out of the joint counts and is reported as unmeasured --
never defaulted to idle, which is the reading that flatters the result.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

__all__ = [
    "Occupancy", "align_offset", "calibrate_state_idle", "joint_counts", "merge_engines",
    "subsumed_columns",
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
    return {"sampled_cycles": n, "joint_columns": cols, "subsumed_columns": subsumed,
            "busy": {c: sum(hot[c]) for c in hot}, "idle_cycles": idle, "overlap_any": ovl,
            "overlap_across_kinds": ovl_kind,
            "overlap_across_kinds_is_lower_bound": bool(undeclared),
            "undeclared_columns": undeclared,
            "unbound_columns": sorted(c for c in cols if c not in (unit_of or {}))}
