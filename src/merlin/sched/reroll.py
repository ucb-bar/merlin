"""Reroll periodic runs of encoded accelerator commands into counted loops.

WHY THIS EXISTS. A scheduler that materialises a tile nest as one command per tile hands codegen a
flat list thousands of entries long, and emitting one host instruction sequence per entry makes the
kernel's own code the bottleneck: it overruns the host's instruction cache, and every line is then a
compulsory miss with no reuse and no prefetcher. The cost is not the commands -- the device sees the
same ones either way -- it is the *encoding* of the commands. An expert writes the same work as a small
loop, and this pass recovers that spelling from the flat list.

Measured on the first target to hit it, an accelerator whose host has a 16 KiB L1 instruction cache: a
matmul emitted flat was 755 KB - 1.30 MB of ``.text``, 46x to 79x over, against ~1.3 KB for the expert
kernel. Those numbers are one instance, not a constant -- nothing in this module is fitted to them.

WHAT THIS IS, AND WHAT IT IS NOT. This is an ENCODING decision, not a scheduling one. The scheduler
chooses which commands to issue, in which order, against which buffers; this pass does not change that
sequence, its order, or its operands by even one bit. It only chooses whether a stretch is spelled as
N copies or as a counted loop -- exactly what a backend loop-reroller does. The emitted command stream
is identical either way, which is what makes the transform safe to apply unconditionally and what keeps
it out of the schedule's decision space.

WHY PERIODIC AND NOT MERELY CONSECUTIVE. The first version of this pass merged runs of the SAME
command and recovered 11% of a real stream, because an inner nest typically ALTERNATES: a scheduler
emits a small fixed group per tile -- stage an operand, then consume it -- so no two neighbours share
a funct and a same-command reroller finds nothing. The repeating unit is the GROUP, not the command. A
run here is therefore ``trips`` iterations of a body of ``period`` commands, where each SLOT of the
body is separately affine across iterations, which turns ``A B A B A B ...`` into one loop of two
statements rather than into nothing at all.

HOW IT STAYS TARGET-AGNOSTIC. Nothing here knows a field layout, an opcode, or a bit position. A
command is reduced to the triple its encoder already produces -- ``(funct, rs1, rs2)`` -- and the pass
asks one question of the numbers themselves: do they advance in an exact arithmetic progression? The
ENCODER is the oracle. If a field packs non-linearly, or a carry crosses a field boundary, the
progression simply is not exact and the run is refused. There is no encoding assumption to get wrong,
because there is no encoding knowledge here at all.

FAIL CLOSED. A run is accepted only when every member is reproduced EXACTLY by the affine form, over
the whole run rather than a sampled prefix. Anything else is left unrolled. A wrong reroll would be
silent -- the kernel stays well-formed and only the digest changes -- so the check is exhaustive and
the default is to decline.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

#: Shortest body worth a loop, in iterations. Below this the loop's own bookkeeping (index, compare,
#: branch, the operand arithmetic) costs more instructions than the copies it replaces, so rerolling
#: would grow `.text` rather than shrink it. A property of the host's instruction budget, not of any
#: target or shape.
MIN_TRIPS = 4

#: How far apart two commands may be and still be tried as one repeating body. This is NOT a tuned
#: bound: it is derived from the stream itself. The search starts at the smallest period that repeats
#: and then climbs, and the only cap is arithmetic -- a body longer than ``available // MIN_TRIPS``
#: cannot repeat often enough to be worth a loop, whatever the target. Nothing here encodes a tile
#: size, a block shape, or any other fact about one accelerator; an earlier version carried a literal
#: fitted to one target's inner nest, which read as a tuning knob and would have silently under-rolled
#: any target whose groups were shaped differently.


@dataclass(frozen=True)
class Encoded:
    """One command reduced to what the hardware actually receives.

    ``rs1_origin`` distinguishes the two ways rs1 reaches the device. ``None`` means rs1 is an
    immediate the encoder computed; a buffer name means rs1 is a DRAM address, and ``rs1`` is then the
    byte offset into that buffer rather than the address itself -- the base is a runtime value the
    emitter materialises once, so two commands are comparable only when they name the same buffer.
    """

    funct: int
    rs1: int
    rs2: int
    rs1_origin: str | None = None
    #: The scheduler task this command belongs to. Every emitted op carries it back as
    #: ``merlin.global_task``, so a run may not span two tasks: one loop cannot be attributed to two
    #: owners, and silently widening attribution is how a per-task cost model starts lying.
    task: int | None = None


@dataclass(frozen=True)
class Slot:
    """One statement of a rerolled loop body: fixed shape, operands affine in the trip count."""

    funct: int
    rs1_base: int
    rs1_step: int
    rs2_base: int
    rs2_step: int
    rs1_origin: str | None


@dataclass(frozen=True)
class Run:
    """``trips`` iterations of ``slots``, reproducing ``instrs[start:start+count]`` exactly."""

    start: int
    trips: int
    slots: tuple[Slot, ...]
    task: int | None = None

    @property
    def period(self) -> int:
        return len(self.slots)

    @property
    def count(self) -> int:
        return self.trips * self.period

    @property
    def stop(self) -> int:
        return self.start + self.count


def _affine_step(values: list[int]) -> int | None:
    """The common difference of ``values``, or ``None`` if it is not an exact progression.

    Checked over every element, not a prefix: a run that is affine for its first few members and then
    diverges -- which is exactly what a carry out of one packed field looks like -- must be refused,
    and a sampled check would admit it.
    """
    if len(values) < 2:
        return 0
    step = values[1] - values[0]
    for i in range(2, len(values)):
        if values[i] - values[i - 1] != step:
            return None
    return step


def _extend(encoded: list[Encoded | None], start: int, period: int) -> tuple[int, tuple[Slot, ...] | None, int | None]:
    # noqa: D401
    """Longest run at ``start`` with this ``period``, grown forward one group at a time.

    Two groups fix the body: the first supplies each slot's funct, origin and base, the second its
    two steps. Every later group is then a PREDICTION to check, and the run ends at the first group
    that misses -- which is what keeps this linear. Deriving the steps once and testing forward is
    the same work a strength-reduction pass does, and it is why the pass can run over a stream of
    tens of thousands of commands without quadratic blowup.
    """
    n = len(encoded)
    if start + 2 * period > n:
        return 0, None, None
    head = encoded[start]
    if head is None:
        return 0, None, None
    task = head.task
    for t in range(2 * period):
        e = encoded[start + t]
        if e is None or e.task != task:
            return 0, None, None
    meta: list[tuple[int, int, int, int, int, str | None]] = []
    for t in range(period):
        a, b = encoded[start + t], encoded[start + period + t]
        if b.funct != a.funct or b.rs1_origin != a.rs1_origin:  # type: ignore[union-attr]
            return 0, None, None
        meta.append(
            (
                a.funct,
                a.rs1,
                b.rs1 - a.rs1,  # type: ignore[union-attr]
                a.rs2,
                b.rs2 - a.rs2,
                a.rs1_origin,
            )
        )  # type: ignore[union-attr]
    trips = 2
    while start + (trips + 1) * period <= n:
        for t in range(period):
            e = encoded[start + trips * period + t]
            funct, r1b, r1s, r2b, r2s, origin = meta[t]
            if (
                e is None
                or e.task != task
                or e.funct != funct
                or e.rs1_origin != origin
                or e.rs1 != r1b + trips * r1s
                or e.rs2 != r2b + trips * r2s
            ):
                break
        else:
            trips += 1
            continue
        break
    slots = tuple(
        Slot(funct=f, rs1_base=r1b, rs1_step=r1s, rs2_base=r2b, rs2_step=r2s, rs1_origin=origin)
        for f, r1b, r1s, r2b, r2s, origin in meta
    )
    return trips, slots, task


def _barrier(encoded: list[Encoded | None], start: int) -> int:
    """First index at or after ``start`` that no run beginning at ``start`` may reach.

    A run stops at a command this pass cannot encode, and at a change of owning task; both are
    properties of the stream, so the horizon is read off the stream rather than assumed.
    """
    head = encoded[start]
    task = None if head is None else head.task
    stop = start
    n = len(encoded)
    while stop < n:
        e = encoded[stop]
        if e is None or e.task != task:
            break
        stop += 1
    return stop


def _multipliers(trip_hints: Sequence[int], ceiling: int) -> list[int]:
    """Whole-loop multipliers implied by the caller's declared trip counts.

    A body one level up holds exactly one full iteration of the level below, so the periods worth
    trying are the innermost period times a PRODUCT of trip counts. Those counts are a derived fact
    about the machine, not a guess: a scheduler sizes its blocking from the target's own capacities,
    so the caller knows them before a single command is emitted and can simply say so.

    This is what makes the search exact rather than approximate. Inferring the multiplier from how far
    a run happened to extend fails whenever a nest peels an iteration -- a first tile that loads what
    the rest reuse, a last tile that is ragged -- because the observed repeat count is then one short
    of the real trip count and every level above it is missed.
    """
    seen: set[int] = {1}
    for hint in trip_hints:
        if hint < 2:
            continue
        for base in sorted(seen):
            product = base * hint
            if product <= ceiling:
                seen.add(product)
    return sorted(seen)


def _candidate_periods(
    encoded: list[Encoded | None], start: int, limit: int, min_trips: int, trip_hints: Sequence[int] = ()
) -> list[int]:
    """Periods worth trying at ``start``, smallest first, climbing the nest one level at a time.

    The insight that makes this both general and cheap: once a body of ``p`` commands is known to
    repeat ``t`` times, the only larger body that can also repeat is one holding WHOLE copies of it,
    and the next such body is exactly ``p * t`` -- one full iteration of the level below. So the
    search walks the loop nest upward, one candidate per level, instead of trying every length.

    That is why there is no maximum period here. The candidates are derived from what the stream does;
    a target whose inner group is 2 commands and one whose inner group is 200 are both found, and
    neither needs a constant changed.
    """
    available = limit - start
    ceiling = available // min_trips
    if ceiling < 1:
        return []
    head = encoded[start]
    if head is None:
        return []
    # The innermost body cannot be shorter than the first recurrence of the head command's shape.
    first = 0
    for p in range(1, ceiling + 1):
        e = encoded[start + p]
        if e is not None and e.funct == head.funct and e.rs1_origin == head.rs1_origin:
            first = p
            break
    if first == 0:
        return []
    periods: set[int] = set()
    # Declared structure first: the innermost body times every product of the caller's trip counts.
    for multiplier in _multipliers(trip_hints, ceiling):
        candidate = first * multiplier
        if candidate <= ceiling:
            periods.add(candidate)
    # Then the same climb driven by what the stream actually does, so a caller that declares nothing
    # -- another target, a host-lane region, a hand-written stretch -- still gets rerolled.
    period = first
    while period <= ceiling:
        periods.add(period)
        trips, slots, _ = _extend(encoded, start, period)
        if slots is None or trips < 2:
            break
        nxt = period * trips
        if nxt <= period:  # pragma: no cover - defensive
            break
        period = nxt
    return sorted(periods)


def find_runs(
    encoded: list[Encoded | None], *, min_trips: int = MIN_TRIPS, trip_hints: Sequence[int] = ()
) -> list[Run]:
    """Maximal periodic runs over ``encoded``, left to right, never overlapping.

    ``None`` marks a command this pass cannot reduce to a ``(funct, rs1, rs2)`` triple -- a fence, a
    host-lane program, anything whose emission is not one ROCC instruction. Those are hard barriers:
    a run never spans one, because rerolling across it would reorder it relative to its neighbours.

    Among the candidates at one position the winner is the one covering the most COMMANDS, so a long
    body beats a short one that happens to start at the same place; ties go to the shorter body,
    which keeps the emitted loop simpler for identical coverage.
    """
    runs: list[Run] = []
    n = len(encoded)
    i = 0
    while i < n:
        if encoded[i] is None:
            i += 1
            continue
        best: Run | None = None
        limit = _barrier(encoded, i)
        for period in _candidate_periods(encoded, i, limit, min_trips, trip_hints):
            trips, slots, task = _extend(encoded, i, period)
            if slots is None or trips < min_trips:
                continue
            cand = Run(start=i, trips=trips, slots=slots, task=task)
            if best is None or cand.count > best.count:
                best = cand
        if best is not None:
            runs.append(best)
            i = best.stop
        else:
            i += 1
    return runs


def reproduces(run: Run | Nest, encoded: list[Encoded | None]) -> bool:
    """Whether ``run``'s affine form reproduces its members exactly.

    The detector already refuses anything non-affine, so this is a redundant check -- deliberately. A
    mis-rerolled kernel stays well-formed and fails only as a changed digest, which is the hardest
    class of bug to attribute, so the property is asserted at the point of use rather than trusted
    from the point of construction.

    A :class:`Nest` is checked the same way and just as exhaustively: it is expanded to the commands
    its two affine indices generate and every one of them is compared, so the outer level is never
    trusted merely because the inner one was.
    """
    if isinstance(run, Nest):
        return _nest_reproduces(run, encoded)
    if run.trips < 1 or run.period < 1 or run.stop > len(encoded):
        return False
    for k in range(run.trips):
        for t, slot in enumerate(run.slots):
            e = encoded[run.start + k * run.period + t]
            if e is None:
                return False
            if e.funct != slot.funct or e.rs1_origin != slot.rs1_origin or e.task != run.task:
                return False
            if e.rs1 != slot.rs1_base + k * slot.rs1_step:
                return False
            if e.rs2 != slot.rs2_base + k * slot.rs2_step:
                return False
    return True


def _nest_reproduces(nest: Nest, encoded: list[Encoded | None]) -> bool:
    """Expand ``nest`` in full and compare it to the stream, command for command."""
    if nest.trips < 1 or not nest.body or nest.stop > len(encoded):
        return False
    index = nest.start
    for outer in range(nest.trips):
        for unit, steps in zip(nest.body, nest.steps):
            if unit.trips < 1 or unit.period != len(steps):
                return False
            for k in range(unit.trips):
                for slot, (d1, d2) in zip(unit.slots, steps):
                    e = encoded[index]
                    if e is None or e.task != nest.task or unit.task != nest.task:
                        return False
                    if e.funct != slot.funct or e.rs1_origin != slot.rs1_origin:
                        return False
                    if e.rs1 != slot.rs1_base + outer * d1 + k * slot.rs1_step:
                        return False
                    if e.rs2 != slot.rs2_base + outer * d2 + k * slot.rs2_step:
                        return False
                    index += 1
    return index == nest.stop


def savings(runs: list[Run | Nest]) -> int:
    """Commands removed from the emitted stream -- the instruction-count win, before loop overhead.

    Reported rather than inferred: the whole reason this pass exists is a `.text` figure, so the pass
    states what it did in the same units.
    """
    return sum(r.count - (r.statements if isinstance(r, Nest) else r.period) for r in runs)


# --------------------------------------------------------------------------------------------
# LEVEL TWO: a loop whose body is a sequence of loops.
#
# WHY A SECOND LEVEL AT ALL. `find_runs` emits ONE counted loop per periodic stretch, so the code it
# leaves behind is as long as the BODY of the innermost repeating group -- and a real nest has more
# than one level. Measured on a convolution whose device work is one [M,N,K] tile walk per OUTPUT
# ROW: each row's walk rerolled to about 133 statements, but the row loop itself was not recovered
# and those 133 statements were emitted once per row, so `.text` scaled with the output height (27 KB
# at 14 rows, 56 KB at 28) instead of staying flat. Nothing about that is specific to a convolution:
# any schedule whose outer loop body is itself a nest has the same shape, and a single-level reroller
# recovers only its innermost level.
#
# WHY IT IS A SEPARATE PASS RATHER THAN A DEEPER SEARCH. The level-one search asks whether COMMANDS
# repeat; this one asks whether LOOPS repeat, which is a different question over a different alphabet
# and would not be reachable by widening the first search's periods (a period covering a whole outer
# iteration is found by that search only as a flat body of the same length, which is exactly the code
# size this exists to remove). Running it afterwards also keeps `find_runs` -- and every property
# already asserted about it -- untouched: a stream with no second level comes back as the same runs.
#
# HOW IT STAYS TARGET-AGNOSTIC AND FAIL-CLOSED. Two units are candidates to be the same statement of
# one body only when they are structurally IDENTICAL (same trip count, same body length, same functs,
# same operand origins, same inner steps) and their bases advance in an exact arithmetic progression
# across outer iterations -- the same question, and the same refusal, one level up. `reproduces`
# expands a nest in full and compares it command for command, so an accepted nest has been checked
# against the stream it claims to reproduce rather than trusted from its construction.


@dataclass(frozen=True)
class Nest:
    """``trips`` iterations of ``body``, a sequence of counted loops rather than of commands.

    ``body`` is a list of :class:`Run`; a stretch the level-one search could not roll appears here as
    a ``trips == 1`` run, so one representation covers "loop" and "straight-line group" alike and the
    emitter needs no second case for the gaps between loops.

    ``steps`` carries, for each body member and each of ITS slots, how that slot's two operands
    advance per OUTER iteration. The slot's own (inner) step is unchanged and still lives on the slot,
    so the command issued at outer index ``o`` and inner index ``k`` is
    ``rs1_base + o * outer_step + k * rs1_step`` -- affine in both, which is what makes the expansion
    checkable in closed form.
    """

    start: int
    trips: int
    body: tuple[Run, ...]
    steps: tuple[tuple[tuple[int, int], ...], ...]
    task: int | None = None

    @property
    def period(self) -> int:
        """Commands in one outer iteration."""
        return sum(run.count for run in self.body)

    @property
    def count(self) -> int:
        return self.trips * self.period

    @property
    def stop(self) -> int:
        return self.start + self.count

    @property
    def statements(self) -> int:
        """Emitted command statements -- the code-size figure this level exists to reduce."""
        return sum(run.period for run in self.body)


def _shape(run: Run) -> tuple:
    """Everything about ``run`` that two iterations of one outer loop must share.

    The bases are deliberately absent: they are what is allowed to move, and the progression check is
    what decides whether they move affinely. Everything else must match exactly.
    """
    return (run.trips, run.task, tuple((s.funct, s.rs1_origin, s.rs1_step, s.rs2_step) for s in run.slots))


def _outer_steps(first: Run, second: Run) -> tuple[tuple[int, int], ...]:
    """Per-slot ``(rs1, rs2)`` advance from one outer iteration to the next."""
    return tuple((b.rs1_base - a.rs1_base, b.rs2_base - a.rs2_base) for a, b in zip(first.slots, second.slots))


def _iteration_matches(
    units: list[Run], base: int, period: int, k: int, steps: list[tuple[tuple[int, int], ...]]
) -> bool:
    """Whether outer iteration ``k`` is exactly what the derived steps PREDICT it to be."""
    for t in range(period):
        want, got = units[base + t], units[base + k * period + t]
        if _shape(want) != _shape(got):
            return False
        for slot_index, (d1, d2) in enumerate(steps[t]):
            a, b = want.slots[slot_index], got.slots[slot_index]
            if b.rs1_base != a.rs1_base + k * d1 or b.rs2_base != a.rs2_base + k * d2:
                return False
    return True


def _extend_units(
    units: list[Run], start: int, period: int, limit: int
) -> tuple[int, list[tuple[tuple[int, int], ...]] | None]:
    """Longest outer loop at ``start`` with this ``period``, grown one iteration at a time.

    Two iterations fix the per-slot outer steps; every later one is a prediction to check, which is
    what keeps this linear in the run it finds rather than quadratic.
    """
    if start + 2 * period > limit:
        return 0, None
    steps = []
    for t in range(period):
        first, second = units[start + t], units[start + period + t]
        if _shape(first) != _shape(second):
            return 0, None
        steps.append(_outer_steps(first, second))
    trips = 2
    while start + (trips + 1) * period <= limit and _iteration_matches(units, start, period, trips, steps):
        trips += 1
    return trips, steps


def find_nests(units: list[Run], *, min_trips: int = MIN_TRIPS) -> list[Nest | Run]:
    """Fold ``units`` -- a contiguous, barrier-free list of runs -- into outer loops where they repeat.

    ``units`` must tile one region of the stream end to end (see :func:`as_units`), because an outer
    loop's body has to be everything issued in one iteration, not only the parts that happened to
    roll. What comes back tiles the same region: a :class:`Nest` where one was found and the original
    :class:`Run` everywhere else.

    A candidate outer period is any offset at which the head unit's SHAPE recurs. That is not a
    heuristic but a necessary condition -- after a whole outer iteration the same statement must come
    round again -- and it is paired with two exact prefilters (the commands per iteration must match,
    and the second statement must too) so the full check runs only on candidates that can still win.
    """
    out: list[Nest | Run] = []
    n = len(units)
    prefix = [0]
    for run in units:
        prefix.append(prefix[-1] + run.count)
    by_shape: dict[tuple, list[int]] = {}
    for index, run in enumerate(units):
        by_shape.setdefault(_shape(run), []).append(index)
    i = 0
    while i < n:
        ceiling = (n - i) // min_trips
        best: Nest | None = None
        for j in by_shape.get(_shape(units[i]), ()):
            period = j - i
            if period < 1 or period > ceiling:
                continue
            # Necessary conditions, both O(1): one outer iteration issues the same number of commands
            # as the next, and the body's second statement comes round with it.
            if prefix[i + period] - prefix[i] != prefix[i + 2 * period] - prefix[i + period]:
                continue
            if period > 1 and _shape(units[i + 1]) != _shape(units[i + period + 1]):
                continue
            trips, steps = _extend_units(units, i, period, n)
            if steps is None or trips < min_trips:
                continue
            cand = Nest(
                start=units[i].start,
                trips=trips,
                body=tuple(units[i : i + period]),
                steps=tuple(steps),
                task=units[i].task,
            )
            # Prefer the nest that removes the most STATEMENTS: an outer loop is worth having exactly
            # to the extent that it stops repeating the statements inside it.
            if best is None or cand.count - cand.statements > best.count - best.statements:
                best = cand
        if best is not None:
            out.append(best)
            i += len(best.body) * best.trips
        else:
            out.append(units[i])
            i += 1
    return out


def as_units(encoded: list[Encoded | None], runs: list[Run]) -> list[list[Run]]:
    """The stream as contiguous barrier-free REGIONS, each tiled end to end by runs.

    Everything a run did not cover becomes a ``trips == 1`` run of its own, so a region is a list of
    units with no gaps -- the shape :func:`find_nests` needs, because an outer loop body must account
    for every command issued in one iteration.
    """
    at = {run.start: run for run in runs}
    regions: list[list[Run]] = []
    current: list[Run] = []
    i, n = 0, len(encoded)
    while i < n:
        item = encoded[i]
        if item is None:
            if current:
                regions.append(current)
                current = []
            i += 1
            continue
        run = at.get(i)
        if run is not None:
            if current and current[-1].task != run.task:
                regions.append(current)
                current = []
            current.append(run)
            i = run.stop
            continue
        if current and current[-1].task != item.task:
            regions.append(current)
            current = []
        current.append(
            Run(
                start=i,
                trips=1,
                task=item.task,
                slots=(
                    Slot(
                        funct=item.funct,
                        rs1_base=item.rs1,
                        rs1_step=0,
                        rs2_base=item.rs2,
                        rs2_step=0,
                        rs1_origin=item.rs1_origin,
                    ),
                ),
            )
        )
        i += 1
    if current:
        regions.append(current)
    return regions


def plan(
    encoded: list[Encoded | None], *, min_trips: int = MIN_TRIPS, trip_hints: Sequence[int] = ()
) -> list[Nest | Run]:
    """The whole pass: roll periodic commands, then roll periodic loops, then CHECK the result.

    Only units that survive :func:`reproduces` are returned, and a ``trips == 1`` run is dropped back
    to nothing so a caller that emits the stream command by command outside the returned units gets
    exactly the commands it would have emitted anyway.
    """
    planned: list[Nest | Run] = []
    runs = [run for run in find_runs(encoded, min_trips=min_trips, trip_hints=trip_hints) if reproduces(run, encoded)]
    for region in as_units(encoded, runs):
        for unit in find_nests(region, min_trips=min_trips):
            if isinstance(unit, Run) and unit.trips < min_trips:
                continue
            if reproduces(unit, encoded):
                planned.append(unit)
    return planned
