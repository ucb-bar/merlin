"""Natural-loop discovery, bounded trip proofs and source-affine annotations.

Observations retain their exact supported domain and explicit unknowns.
No accelerator semantics, execution multiplicity or hardware law is inferred.
"""


from dataclasses import replace

from .riscv_decode import (
    _REGIONS as _REGIONS,
)
from .riscv_decode import (
    BRANCHES as BRANCHES,
)
from .riscv_decode import (
    CALLS as CALLS,
)
from .riscv_decode import (
    ET_REL as ET_REL,
)
from .riscv_decode import (
    CPUInstruction as CPUInstruction,
)
from .riscv_decode import (
    _Cfg as _Cfg,
)
from .riscv_decode import (
    _ends_segment as _ends_segment,
)
from .riscv_decode import (
    decode_cpu as decode_cpu,
)
from .riscv_decode import (
    linked_input_status as linked_input_status,
)
from .riscv_decode import (
    region_index as region_index,
)
from .riscv_decode import (
    section_of as section_of,
)
from .riscv_decode import (
    signed as signed,
)
from .riscv_forms import (
    _form_add as _form_add,
)
from .riscv_forms import (
    _frame_key as _frame_key,
)
from .riscv_forms import (
    _run_iteration as _run_iteration,
)
from .riscv_state import (
    _CONDITIONS as _CONDITIONS,
)
from .riscv_state import (
    _FRAME as _FRAME,
)
from .riscv_state import (
    _coarse_effect as _coarse_effect,
)
from .riscv_state import (
    _defined_registers as _defined_registers,
)
from .riscv_stream import (
    _MAX_PERIOD as _MAX_PERIOD,
)
from .riscv_stream import (
    _advance as _advance,
)
from .riscv_stream import (
    _identity as _identity,
)
from .riscv_stream import (
    _run_step as _run_step,
)
from .riscv_stream import (
    _slot as _slot,
)
from .riscv_stream import (
    _StreamRecorder as _StreamRecorder,
)
from .riscv_stream import (
    expand_command_stream as expand_command_stream,
)
from .riscv_values import (
    _VALUE_SET_LIMIT as _VALUE_SET_LIMIT,
)
from .riscv_values import (
    MASK as MASK,
)
from .riscv_values import (
    UNKNOWN as UNKNOWN,
)
from .riscv_values import (
    ZERO as ZERO,
)
from .riscv_values import (
    Bits as Bits,
)
from .riscv_values import (
    _affine_record as _affine_record,
)
from .riscv_values import (
    _cut_window as _cut_window,
)
from .riscv_values import (
    _sext as _sext,
)
from .riscv_values import (
    _Table as _Table,
)
from .riscv_values import (
    _window_arithmetic as _window_arithmetic,
)
from .riscv_values import (
    _window_record as _window_record,
)
from .riscv_values import (
    arithmetic as arithmetic,
)


def _reenters_without(cfg, start, body, header):
    """True when ``start`` lies on a cycle inside ``body`` that avoids ``header``."""
    seen, work = set(), [t for t in cfg.succ.get(start, ()) if t in body and t != header]
    while work:
        node = work.pop()
        if node == start:
            return True
        if node in seen:
            continue
        seen.add(node)
        work.extend(t for t in cfg.succ.get(node, ()) if t in body and t != header)
    return False

def _register_name(r):
    return f"x{r}"


def _natural_loops(cfg, observations, outgoing):
    """Natural loops of the reachable CFG with proved induction and trip counts."""
    latches = {}
    for b, targets in cfg.succ.items():
        for h in targets:
            if h in cfg.idom and b in cfg.idom and cfg.dominates(h, b):
                latches.setdefault(h, []).append(b)
    loops = []
    for header, ends in latches.items():
        body, work = {header}, list(ends)
        while work:
            node = work.pop()
            if node not in body:
                body.add(node)
                work.extend(p for p in cfg.pred.get(node, ()) if p in cfg.idom)
        loops.append(dict(header=header, latches=sorted(set(ends)), body=body))
    loops.sort(key=lambda loop: (len(loop["body"]), loop["header"]))
    for loop in loops:
        parents = [o for o in loops if o is not loop and loop["header"] in o["body"] and loop["body"] <= o["body"]]
        loop["parent"] = min(parents, key=lambda o: len(o["body"])) if parents else None
    for loop in loops:
        depth, parent = 1, loop["parent"]
        while parent is not None:
            depth, parent = depth + 1, parent["parent"]
        loop["depth"] = depth
    for loop in sorted(loops, key=lambda o: -o["depth"]):  # inner exits feed outer loops
        _loop_facts(cfg, loop, loops, observations, outgoing)
    loops.sort(key=lambda loop: loop["header"])
    return loops

def _variable_name(var):
    return _register_name(var[1]) if var[0] == "x" else f"frame[{var[1][0]}]:{var[1][1]}"


def _symbol(var):
    return f"header:x{var[1]}" if var[0] == "x" else f"header:frame:{var[1][0]}:{var[1][1]}"


def _loop_facts(cfg, loop, loops, observations, outgoing):
    header, body = loop["header"], loop["body"]
    pcs = [pc for b in sorted(body) for pc in cfg.blocks[b]]
    defs = {}
    for pc in pcs:
        for r in _defined_registers(cfg.decoded[pc]):
            defs.setdefault(r, []).append(pc)
    loop["defs"] = defs
    preheaders = [p for p in cfg.pred.get(header, ()) if p not in body]
    narrow, at_exit = set(), {}
    exiting = sorted(b for b in body if any(t not in body for t in cfg.succ.get(b, ())))
    exit_pc = cfg.blocks[exiting[0]][-1] if len(exiting) == 1 else None

    def visit(pc, regs, slots):
        if pc == exit_pc:
            at_exit["state"] = (dict(regs), slots.copy())

    seed = {r: (0, _symbol(("x", r)), ()) for r in range(1, 32)}
    out = _run_iteration(cfg, loop, loops, seed, lambda key: (0, _symbol(("s", key)), ()), narrow, visit)
    latch_states = [out.get(b) for b in loop["latches"]]

    def transfer(var):
        """('step', s) when every latch leaves var = header value + s; ('const', c)."""
        kinds = set()
        for state in latch_states:
            if state is None:
                return None
            regs, slots = state
            form = regs.get(var[1]) if var[0] == "x" else slots.read(var[1], lambda key: (0, _symbol(("s", key)), ()))
            if form is None or form[2]:
                return None
            if form[1] == _symbol(var):
                kinds.add(("step", form[0]))
            elif form[1] is None:
                kinds.add(("const", form[0] & MASK))
            else:
                return None
        return kinds.pop() if len(kinds) == 1 else None

    def initial(var):
        joined = None
        for p in preheaders:
            registers, stack = outgoing[p]
            value = registers[var[1]] if var[0] == "x" else stack.get((_FRAME, var[1][0], var[1][1]), UNKNOWN)
            joined = value if joined is None else joined.join(value)
        return None if joined is None else joined.exact

    induction, records = {}, []
    variables = [("x", r) for r in sorted(defs) if r]
    for state in latch_states:
        if state is not None:
            variables += [("s", key) for key in state[1].known if ("s", key) not in variables]
    for var in variables:
        kind = transfer(var)
        if kind is None or kind[0] != "step" or kind[1] == 0:
            continue
        init = initial(var)
        item = dict(reg=_variable_name(var), init=init, step=signed(kind[1] & MASK, 64), status="proved")
        if init is None:
            item.update(status="unknown", reason="initial_value_not_constant")
        else:
            induction[var] = dict(init=init, step=item["step"], narrow=_symbol(var) in narrow)
        records.append(item)
    loop["induction"] = induction
    unsupported = None
    if any(cfg.decoded[pc].kind in ("return", "indirect", "opaque") for pc in pcs):
        unsupported = "loop_contains_return_indirect_or_opaque_transfer"
    loop["trip"] = _trip_count(cfg, loop, exiting, at_exit.get("state"), transfer, initial, narrow, unsupported)
    exit_value = loop["trip"].pop("_exit_value", None)
    if exit_value is not None:
        loop["exit_value"] = (exiting[0], *exit_value)
    parent = loop["parent"]
    loop["record"] = dict(
        header_pc=header,
        latch_pcs=[cfg.blocks[b][-1] for b in loop["latches"]],
        blocks=sorted(body),
        induction=records,
        trip_count=loop["trip"],
        nest_depth=loop["depth"],
        parent_header_pc=parent["header"] if parent else None,
    )


def _trip_count(cfg, loop, exiting, exit_state, transfer, initial, narrow, unsupported):
    def unknown(reason, **extra):
        return dict(status="unknown", reason=reason, **extra)

    if unsupported:
        return unknown(unsupported)
    header, body = loop["header"], loop["body"]
    for b in body:
        d = cfg.decoded[cfg.blocks[b][-1]]
        if d.kind in BRANCHES | {"jump"} and d.target not in cfg.decoded:
            return unknown("exit_leaves_selected_span")
    if len(exiting) != 1:
        return unknown("loop_has_multiple_or_no_exits", exits=len(exiting))
    exit_block = exiting[0]
    branch_pc = cfg.blocks[exit_block][-1]
    d = cfg.decoded[branch_pc]
    if d.kind not in BRANCHES:
        return unknown("exit_is_not_a_conditional_branch")
    if not all(cfg.dominates(exit_block, latch) for latch in loop["latches"]):
        return unknown("exit_not_on_every_iteration")
    if _reenters_without(cfg, exit_block, body, header):
        return unknown("exit_on_cycle_without_header")
    if exit_state is None:
        return unknown("exit_state_unavailable")
    target_in, fall_in = d.target in body, branch_pc + d.size in body
    if target_in == fall_in:
        return unknown("exit_branch_direction_unresolved")
    state = cfg.observations[branch_pc]
    regs = exit_state[0]

    def operand(r):
        if r == 0:
            return (0, None, ())
        if state[r].exact is not None:
            return (state[r].exact, None, ())
        return regs.get(r)

    forms = [operand(d.rs1), operand(d.rs2)]
    candidates = []
    for side, form in enumerate(forms):
        if form is None or form[2] or form[1] is None or not form[1].startswith("header:"):
            continue
        parts = form[1].split(":")
        var = ("x", int(parts[1][1:])) if parts[1] != "frame" else ("s", (int(parts[2]), int(parts[3])))
        kind = transfer(var)
        if kind is not None and not (kind[0] == "step" and kind[1] == 0):
            candidates.append((side, var, kind, form[0]))
    if len(candidates) != 1:
        return unknown("exit_compare_has_no_single_induction_variable")
    side, var, kind, offset = candidates[0]
    bound = forms[1 - side]
    if bound is None or bound[2]:
        return unknown("exit_bound_varies_in_loop")
    if bound[1] is not None:
        return unknown("exit_bound_invariant_but_not_constant", bound_symbol=bound[1])
    init = initial(var)
    if init is None:
        return unknown("induction_initial_value_not_constant", induction=_variable_name(var))
    unsigned = d.kind in ("bltu", "bgeu")
    lo, hi = (0, MASK) if unsigned else (-(1 << 63), (1 << 63) - 1)
    if _symbol(var) in narrow:
        lo, hi = max(lo, -(1 << 31)), min(hi, (1 << 31) - 1)

    def domain(value):
        return value & MASK if unsigned else signed(value & MASK, 64)

    limit = domain(bound[0])
    test = _CONDITIONS[d.kind]

    def exits_at(value):
        holds = test(value, limit) if side == 0 else test(limit, value)
        return holds != target_in

    if kind[0] == "const":
        # The variable is init on the first iteration and a constant after.
        values = [domain(init + offset), domain(kind[1] + offset)]
        if any(not lo <= v <= hi for v in values):
            return unknown("induction_out_of_comparison_range")
        if exits_at(values[0]):
            k = 0
        elif exits_at(values[1]):
            k = 1
        else:
            return unknown("loop_does_not_exit_through_compared_variable")
        return _proved_trip(k, var, branch_pc, bound[0], (d.rs1 if side == 0 else d.rs2, values[k]))
    step = signed(kind[1] & MASK, 64)
    start = (init & MASK if unsigned else signed(init & MASK, 64)) + signed(offset & MASK, 64)
    if not lo <= start <= hi:
        return unknown("induction_out_of_comparison_range")
    k_max = (hi - start) // step if step > 0 else (start - lo) // -step

    def exits(k):
        return exits_at(start + k * step)

    if exits(0):
        k = 0
    elif d.kind in ("beq", "bne"):
        if k_max >= 1 and exits(1):
            k = 1
        elif (limit - start) % step == 0 and 0 < (limit - start) // step <= k_max:
            k = (limit - start) // step
            if not exits(k):
                return unknown("exit_value_not_reached")
        else:
            return unknown("exit_value_not_reached_without_wraparound")
    else:
        if not exits(k_max):
            return unknown("induction_wraps_before_exit")
        low, high = 0, k_max
        while high - low > 1:
            middle = (low + high) // 2
            if exits(middle):
                high = middle
            else:
                low = middle
        k = high
    return _proved_trip(k, var, branch_pc, bound[0], (d.rs1 if side == 0 else d.rs2, start + k * step))


def _proved_trip(k, var, branch_pc, bound, exit_value):
    register, value = exit_value
    return dict(
        _exit_value=(register, value & MASK),
        status="proved",
        value=k + 1,
        backedges_taken=k,
        counts="header_executions_per_loop_entry",
        induction=_variable_name(var),
        exit_pc=branch_pc,
        bound=bound & MASK,
    )


def _annotate_affine(cfg, loops, observations, outgoing):
    """Per-site ``base + sum(step * k)`` forms for registers inside proved loops.

    Runs after the fixed point and never changes it: a form is attached only
    where the fixed point lost the value, every term's loop has a proved trip
    count, and every intra-iteration path to the site yields the same form.
    """
    entries = {}  # id(loop) -> (regs, slots) at entry, from the proved parent
    for loop in sorted(loops, key=lambda o: o["depth"]):
        if loop["trip"]["status"] != "proved":
            continue
        header, body = loop["header"], loop["body"]
        count = loop["trip"]["value"]
        preheaders = [p for p in cfg.pred.get(header, ()) if p not in body]
        outer = entries.get(id(loop))
        written = {key for b in body for pc in cfg.blocks[b] if (key := _store_key(cfg, pc)) is not None}
        clears = any(_store_key(cfg, pc) is None and _writes_memory(cfg, pc) for b in body for pc in cfg.blocks[b])

        def outside(var, outer=outer):
            if outer is not None:
                form = outer[0].get(var[1]) if var[0] == "x" else outer[1].read(var[1], outer[2])
                if form is not None:
                    return form
            joined = None
            for p in preheaders:
                registers, stack = outgoing[p]
                value = registers[var[1]] if var[0] == "x" else stack.get((_FRAME, var[1][0], var[1][1]), UNKNOWN)
                joined = value if joined is None else joined.join(value)
            if joined is None:
                return None
            if joined.exact is not None:
                return (joined.exact, None, ())
            return (joined.offset, joined.base, ()) if joined.base else None

        def seeded(var):
            iv = loop["induction"].get(var)
            if iv is not None:
                start = outside(var)
                if start is None or isinstance(start, _Table) or start[1] is not None:
                    start = (iv["init"], None, ())
                return _form_add(start, (0, None, ((header, iv["step"], count),)))
            if var[0] == "x":
                return None if var[1] in loop["defs"] else outside(var)
            lo, hi = var[1][0], var[1][0] + var[1][1] // 8
            if clears or any(lo < k[0] + k[1] // 8 and k[0] < hi for k in written):
                return None
            return outside(var)

        def visit(pc, regs, slots):
            state = observations[pc]
            annotated = list(state)
            for r, form in regs.items():
                if form is not None and (isinstance(form, _Table) or form[2]) and state[r].exact is None:
                    annotated[r] = replace(state[r], affine=form)
            if annotated != list(state):
                observations[pc] = tuple(annotated)

        seed = {r: seeded(("x", r)) for r in range(1, 32)}
        child_in = {}

        def seed_slot(key, seeded=seeded):
            return seeded(("s", key))

        _run_iteration(cfg, loop, loops, seed, seed_slot, None, visit, child_seed=child_in)
        for child in loops:
            if child["parent"] is loop and id(child) in child_in:
                entries[id(child)] = (*child_in[id(child)], seed_slot)


def _store_key(cfg, pc):
    d = cfg.decoded[pc]
    if not _writes_memory(cfg, pc) or d.kind in CALLS or d.kind == "opaque":
        return None
    return _frame_key(cfg, pc, d)


def _writes_memory(cfg, pc):
    d = cfg.decoded[pc]
    return (
        d.kind in CALLS
        or d.kind in ("store", "opaque")
        or (d.kind in ("fp_memory", "unknown", "fp") and _coarse_effect(cfg.words[pc], d)[0] != "none")
    )
