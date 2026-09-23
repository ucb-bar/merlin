"""Rerolling a command stream must reproduce it exactly, or decline.

The pass under test replaces N copies of a command with a counted loop. It is applied to every
emitted kernel, and a mistake in it is SILENT: the kernel stays well-formed, every gate still passes,
and the only symptom is a changed result digest -- which is the hardest class of defect to attribute,
because it looks like a scheduling change rather than an encoding one. So the property asserted here
is not "the loop is shorter" but "expanding the loop yields the original stream, command for command".

The other half is the refusal. A stream the affine form cannot express must come back UNROLLED rather
than approximated, so every negative case below asserts that the non-affine input produced no run
covering it -- never that it produced a nearly-right one.
"""

from __future__ import annotations

from merlin.sched.reroll import MIN_TRIPS, Encoded, Nest, Run, find_runs, plan, reproduces, savings


def expand(runs: list[Run], encoded: list[Encoded | None]) -> list[Encoded | None]:
    """The stream the emitter will actually issue, with every run written back out.

    This is the inverse of the pass: a run becomes the commands its affine form generates, and
    everything outside a run is copied through. If this does not equal the input, the kernel changed.
    """
    covered = {}
    for run in runs:
        for k in range(run.trips):
            for t, slot in enumerate(run.slots):
                covered[run.start + k * run.period + t] = Encoded(
                    funct=slot.funct,
                    rs1=slot.rs1_base + k * slot.rs1_step,
                    rs2=slot.rs2_base + k * slot.rs2_step,
                    rs1_origin=slot.rs1_origin,
                    task=run.task,
                )
    return [covered.get(i, e) for i, e in enumerate(encoded)]


def cmd(funct: int, rs1: int, rs2: int, origin: str | None = None, task: int | None = None):
    return Encoded(funct=funct, rs1=rs1, rs2=rs2, rs1_origin=origin, task=task)


def test_alternating_group_is_one_loop_not_none() -> None:
    """The scheduler alternates preload and compute, so no two NEIGHBOURS share a funct.

    A reroller that only merges identical adjacent commands finds nothing here -- the first version of
    this pass recovered 11% of a real stream for exactly this reason. The repeating unit is the group.
    """
    stream = []
    for k in range(64):
        stream.append(cmd(6, 100 + 2 * k, 500 + 3 * k))  # preload
        stream.append(cmd(4, 900 + 5 * k, 700 + 7 * k))  # compute
    runs = find_runs(stream)
    assert len(runs) == 1
    assert runs[0].period == 2
    assert runs[0].trips == 64
    assert runs[0].count == 128
    assert expand(runs, stream) == stream


def test_expansion_is_exact_for_every_accepted_run() -> None:
    """The load-bearing property: whatever is accepted must round-trip."""
    stream = []
    for k in range(40):
        stream.append(cmd(2, 8 * k, 0, origin="A"))
        stream.append(cmd(2, 16 * k, 1, origin="B"))
        stream.append(cmd(3, 4 * k, 2))
    runs = find_runs(stream)
    assert runs, "a fully affine stream must produce at least one run"
    assert all(reproduces(r, stream) for r in runs)
    assert expand(runs, stream) == stream


def test_a_non_affine_slot_is_refused_not_approximated() -> None:
    """One quadratic slot must sink the run rather than be fitted with a nearby line."""
    stream = []
    for k in range(16):
        stream.append(cmd(6, 100 + 2 * k, 0))
        stream.append(cmd(4, k * k, 0))  # not affine
    runs = find_runs(stream)
    assert expand(runs, stream) == stream
    # No accepted run may cover the quadratic slot at any of its positions.
    quadratic = {1 + 2 * k for k in range(16)}
    for run in runs:
        assert not (set(range(run.start, run.stop)) & quadratic)


def test_a_carry_out_of_a_packed_field_breaks_the_progression() -> None:
    """The failure this pass must not paper over.

    A packed operand advances linearly until a field overflows into its neighbour, and then jumps.
    Sampling the first few members would admit the whole run and silently corrupt the tail.
    """
    stream = [cmd(4, 10 * k, 0) for k in range(6)]
    stream += [cmd(4, 99999 + 10 * k, 0) for k in range(6)]  # discontinuity at index 6
    runs = find_runs(stream)
    assert expand(runs, stream) == stream
    for run in runs:
        assert run.start >= 6 or run.stop <= 6, "no run may span the discontinuity"


def test_a_barrier_is_never_spanned() -> None:
    """``None`` is a command this pass cannot encode; rerolling across it would reorder it."""
    stream = [cmd(6, 2 * k, 0) for k in range(6)] + [None] + [cmd(6, 100 + 2 * k, 0) for k in range(6)]
    runs = find_runs(stream)
    assert expand(runs, stream) == stream
    for run in runs:
        assert run.start > 6 or run.stop <= 6


def test_a_task_boundary_is_never_spanned() -> None:
    """One loop cannot be attributed to two owners, so a run stops at the task change."""
    stream = [cmd(6, 2 * k, 0, task=1) for k in range(6)] + [cmd(6, 100 + 2 * k, 0, task=2) for k in range(6)]
    runs = find_runs(stream)
    assert expand(runs, stream) == stream
    for run in runs:
        tasks = {stream[i].task for i in range(run.start, run.stop)}  # type: ignore[union-attr]
        assert len(tasks) == 1


def test_a_run_shorter_than_the_threshold_is_left_alone() -> None:
    """Below the threshold the loop's own bookkeeping costs more than the copies it replaces."""
    stream = [cmd(4, 3 * k, 0) for k in range(MIN_TRIPS - 1)]
    assert find_runs(stream) == []


def test_a_differing_rs1_origin_does_not_merge() -> None:
    """An immediate and a DRAM offset are not the same quantity, whatever their numbers look like."""
    stream = [cmd(2, 8 * k, 0, origin="A") for k in range(6)] + [cmd(2, 48 + 8 * k, 0, origin="B") for k in range(6)]
    runs = find_runs(stream)
    assert expand(runs, stream) == stream
    for run in runs:
        origins = {stream[i].rs1_origin for i in range(run.start, run.stop)}  # type: ignore[union-attr]
        assert len(origins) == 1


def test_savings_are_reported_in_commands() -> None:
    """The pass exists because of an instruction count, so it reports one."""
    stream = []
    for k in range(32):
        stream.append(cmd(6, 2 * k, 0))
        stream.append(cmd(4, 5 * k, 0))
    runs = find_runs(stream)
    # 64 commands become one body of 2, so 62 are no longer emitted.
    assert savings(runs) == 62


def test_a_mutated_run_is_caught_by_the_reproduction_check() -> None:
    """``reproduces`` is the redundant guard, so prove it actually guards something."""
    stream = [cmd(4, 10 * k, 5 * k) for k in range(10)]
    runs = find_runs(stream)
    assert runs and all(reproduces(r, stream) for r in runs)
    good = runs[0]
    wrong = Run(
        start=good.start,
        trips=good.trips,
        slots=(
            good.slots[0].__class__(
                funct=good.slots[0].funct,
                rs1_base=good.slots[0].rs1_base + 1,
                rs1_step=good.slots[0].rs1_step,
                rs2_base=good.slots[0].rs2_base,
                rs2_step=good.slots[0].rs2_step,
                rs1_origin=good.slots[0].rs1_origin,
            ),
        ),
        task=good.task,
    )
    assert not reproduces(wrong, stream)


def test_scales_linearly_enough_for_a_real_stream() -> None:
    """A real contraction reaches codegen as tens of thousands of commands.

    An earlier version searched candidate lengths downward and was quadratic; it did not finish on a
    75k-command kernel. The bound here is loose on purpose -- it is guarding against a complexity
    regression, not asserting a runtime.
    """
    import time

    stream = []
    for k in range(37_000):
        stream.append(cmd(6, 100 + 2 * k, 500 + 3 * k))
        stream.append(cmd(4, 900 + 5 * k, 700 + 7 * k))
    started = time.monotonic()
    runs = find_runs(stream)
    elapsed = time.monotonic() - started
    assert elapsed < 30.0, f"rerolling 74k commands took {elapsed:.1f}s"
    assert sum(r.count for r in runs) == len(stream)


# --- level two: a loop whose body is a sequence of loops ---------------------------------------
#
# The same property, one level up, and for the same reason: the outer loop is accepted only if
# expanding it yields the original stream. A wrong outer step is exactly as silent as a wrong inner
# one -- the kernel stays well formed and only the digest moves -- so every case here asserts the
# expansion, not the shape.


def expand_unit(unit, out: dict) -> None:
    """Write back the commands ``unit`` will issue, at their stream positions."""
    if isinstance(unit, Nest):
        index = unit.start
        for outer in range(unit.trips):
            for run, steps in zip(unit.body, unit.steps):
                for k in range(run.trips):
                    for slot, (d1, d2) in zip(run.slots, steps):
                        out[index] = cmd(
                            slot.funct,
                            slot.rs1_base + outer * d1 + k * slot.rs1_step,
                            slot.rs2_base + outer * d2 + k * slot.rs2_step,
                            slot.rs1_origin,
                            unit.task,
                        )
                        index += 1
        return
    for k in range(unit.trips):
        for t, slot in enumerate(unit.slots):
            out[unit.start + k * unit.period + t] = cmd(
                slot.funct,
                slot.rs1_base + k * slot.rs1_step,
                slot.rs2_base + k * slot.rs2_step,
                slot.rs1_origin,
                unit.task,
            )


def expand_plan(units, encoded):
    covered: dict = {}
    for unit in units:
        expand_unit(unit, covered)
    return [covered.get(i, e) for i, e in enumerate(encoded)]


#: One row of :func:`two_level_stream`, in commands.
ROW_LEN = 1 + 6 + 1 + 5 + 2 * 7


def two_level_stream(rows: int) -> list[Encoded | None]:
    """``rows`` iterations of the shape a tiled schedule actually has.

    Each row stages one operand (a configuration write, then a run of loads), stages the other, and
    then walks the pairs -- THREE runs of different lengths with two configuration writes between
    them, which is what a real inner nest looks like. It is also what a one-level reroller cannot
    fully recover: the candidate periods it derives start at the first recurrence of the head
    command's shape (here the second configuration write), and the row is not a whole multiple of
    that, so it rolls the three runs and then emits all of them once per row.
    """
    out: list[Encoded | None] = []
    for r in range(rows):
        out.append(cmd(0, 300 + 2 * r, 5))
        for i in range(6):
            out.append(cmd(2, 1000 + 64 * r + 4 * i, 16, "A"))
        out.append(cmd(0, 700 + 3 * r, 9))
        for i in range(5):
            out.append(cmd(2, 8000 + 32 * r + 4 * i, 16, "B"))
        for i in range(7):
            out.append(cmd(6, 5000 + 8 * r + 2 * i, 32))
            out.append(cmd(4, 6000 + 8 * r + 2 * i, 48))
    return out


def test_an_outer_loop_over_inner_loops_is_recovered() -> None:
    stream = two_level_stream(rows=9)
    units = plan(stream)
    nests = [u for u in units if isinstance(u, Nest)]
    assert nests, "the row loop above the inner pair loop was not recovered"
    assert expand_plan(units, stream) == stream
    # The point of the level: the statements emitted stop scaling with the number of rows.
    statements = sum(u.statements if isinstance(u, Nest) else u.period for u in units)
    assert statements < sum(r.period for r in find_runs(stream) if reproduces(r, stream)), (
        "nesting did not reduce the emitted statement count"
    )


def test_a_nest_expansion_is_exact_for_every_row() -> None:
    """Checked over the WHOLE nest, not a sampled row: an outer step that is right for the second
    iteration and wrong for the tenth is precisely the failure a spot check admits."""
    stream = two_level_stream(rows=11)
    units = plan(stream)
    for unit in units:
        assert reproduces(unit, stream)
    assert expand_plan(units, stream) == stream


def test_a_row_that_breaks_the_progression_ends_the_nest() -> None:
    """One row whose configuration write is not on the progression must leave the rows after it
    outside that nest rather than be folded in with a fitted step."""
    stream = two_level_stream(rows=10)
    row_len = ROW_LEN
    broken = stream[6 * row_len]
    stream[6 * row_len] = cmd(0, broken.rs1 + 13, broken.rs2)
    units = plan(stream)
    assert expand_plan(units, stream) == stream
    for unit in units:
        if isinstance(unit, Nest):
            assert not (unit.start < 6 * row_len < unit.stop - unit.period), (
                "a nest was grown across the row that broke the progression"
            )


def test_a_nest_never_spans_a_barrier() -> None:
    """A fence between rows is a hard barrier at this level too: rolling rows across one would move
    it relative to the commands it orders."""
    stream = two_level_stream(rows=8)
    row_len = ROW_LEN
    with_fences: list[Encoded | None] = []
    for r in range(8):
        with_fences.append(None)
        with_fences.extend(stream[r * row_len : (r + 1) * row_len])
    units = plan(with_fences)
    assert expand_plan(units, with_fences) == with_fences
    for unit in units:
        assert all(with_fences[i] is not None for i in range(unit.start, unit.stop))


def test_a_nest_never_spans_two_tasks() -> None:
    stream = two_level_stream(rows=10)
    row_len = ROW_LEN
    owned = [Encoded(e.funct, e.rs1, e.rs2, e.rs1_origin, 0 if i < 5 * row_len else 1) for i, e in enumerate(stream)]
    units = plan(owned)
    assert expand_plan(units, owned) == owned
    for unit in units:
        owners = {owned[i].task for i in range(unit.start, unit.stop)}
        assert len(owners) == 1, "one loop cannot be attributed to two owners"


def test_a_stream_with_no_second_level_is_unchanged_by_it() -> None:
    """A stream whose only periodicity is at the command level must come back as the runs the
    one-level pass already found, so adding the level cannot change an existing kernel's emission."""
    stream = [cmd(2, 100 + 4 * i, 16, "A") for i in range(40)]
    runs = [r for r in find_runs(stream) if reproduces(r, stream)]
    units = plan(stream)
    assert [u for u in units if isinstance(u, Nest)] == []
    assert [(u.start, u.trips, u.slots) for u in units] == [(r.start, r.trips, r.slots) for r in runs]


def test_savings_count_statements_a_nest_removes() -> None:
    stream = two_level_stream(rows=9)
    units = plan(stream)
    nest = next(u for u in units if isinstance(u, Nest))
    assert savings([nest]) == nest.count - nest.statements
