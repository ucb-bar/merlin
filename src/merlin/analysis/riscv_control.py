"""Bounded control execution and issue-time command observations from actual RV64 words.

Only complete admitted scalar control receives an execution count. Unknown data,
effects, calls and exhausted budgets retain the original explicit refusal.
"""

from __future__ import annotations

from collections import Counter

from .riscv_decode import BRANCHES, CALLS, CPUInstruction, decode_cpu, signed
from .riscv_state import _call_target, _coarse_effect, _initial_state, _step, _summary_applies
from .riscv_stream import _StreamRecorder, expand_command_stream
from .riscv_values import Bits, _window_record


def finite_control_counts(
    rows,
    *,
    custom_opcode,
    max_steps=2_000_000,
    max_commands=1_000_000,
    entry_state=None,
    call_summaries=None,
    record_commands=False,
    max_recorded=200_000,
    assume_returning_calls=False,
    symbolic_returns=False,
):
    """Count a complete original body only when every scalar branch is known.

    Symbolic argument pointers preserve identity and offsets. Tensor loads and
    accelerator results stay unknown. Calls, data-dependent control, unsupported
    effects and exhausted budgets refuse the *entire* observation. A call into
    a summarized callee that provably terminates and issues no selected-opcode
    word is stepped over with its summary instead of refusing.

    ``record_commands`` adds, for a proved body only, ``command_stream``: every
    selected-opcode issue in execution order as ``{seq, pc, funct, rs1, rs2}``
    (an ``int`` when the register is exact at issue, else its compact record:
    pointer base and offset, or known bits) and every ``fence`` as
    ``{seq, pc, kind: "fence"}``. Each entry carries ``host_instructions_since_prev``:
    scalar instructions executed since the previous selected-opcode issue.

    The record is kept as periodic affine runs (``command_stream_runs``; see
    :func:`expand_command_stream`), at most ``max_recorded`` runs, so long kernels
    stay bounded. ``command_stream`` expands at most ``max_recorded`` entries;
    ``command_stream_truncated`` is true when it is shorter than the issue stream and
    ``command_stream_runs_complete`` says whether the runs cover all of it. An
    unproved body gains ``command_stream_status`` with the refusal reason; an
    exhausted step budget is ``step_budget_exhausted`` with ``steps_reached``.

    ``assume_returning_calls`` also steps over a call whose callee summary shows it
    issues no selected-opcode word although its termination is not proved: the
    callee is assumed to return, its caller-saved registers and memory become
    unknown, and the proved result lists it under ``assumed_returning_calls``.

    ``symbolic_returns`` gives each stepped-over call whose return register stays unknown a
    fresh symbol per dynamic call instance (``call:<target>@<pc>#<n>``, listed under
    ``symbolic_returns``): a symbolic window over ``a0`` that later arithmetic keeps, so two
    values derived from one return stay comparable. Distinct instances never share a symbol,
    even at one call site, so no two unknown returns are assumed equal or related.
    """
    recorder = _StreamRecorder(max_recorded) if record_commands else None
    result = _finite_control_counts(
        rows,
        custom_opcode,
        max_steps,
        max_commands,
        entry_state,
        call_summaries,
        recorder,
        assume_returning_calls,
        {} if symbolic_returns else None,
    )
    if result["status"] == "proved":
        result["dynamic_command_count"] = dict(
            status="proved", value=result["commands"], basis="bounded_scalar_execution_of_proved_control"
        )
    else:
        result["dynamic_command_count"] = dict(status="unknown", reason=result["reason"])
    if recorder is not None and result["status"] == "proved":
        result["command_stream"] = list(expand_command_stream(recorder.runs, limit=max_recorded))
        result["command_stream_runs"] = recorder.runs
        result["command_stream_entries"] = recorder.entries
        result["command_stream_truncated"] = len(result["command_stream"]) < recorder.entries
        result["command_stream_runs_complete"] = not recorder.overflow
    elif recorder is not None:
        budget = result["reason"] == "finite_instruction_budget_exhausted"
        result["command_stream_status"] = dict(
            status="unknown",
            reason="step_budget_exhausted" if budget else result["reason"],
            **(dict(steps_reached=max_steps) if budget else {}),
        )
    return result


def _issue_operand(bits):
    """An issue-time register: its exact integer, else its bounded symbolic record."""
    if bits.exact is not None:
        return bits.exact
    record = dict(kind="unknown", raw=None)
    if bits.base:
        record.update(pointer_base=bits.base, byte_offset=bits.offset)
    elif bits.mask:
        record["known_bits"] = dict(mask=bits.mask, value=bits.value, width=64)
    if bits.window is not None:
        record.update(_window_record(bits.window))
    return record


def _finite_control_counts(
    rows,
    custom_opcode,
    max_steps,
    max_commands,
    entry_state,
    call_summaries,
    recorder=None,
    assume_calls=False,
    symbols=None,
):
    if not rows:
        return dict(status="unknown", reason="empty_machine_region")
    if any(row.relocations for row in rows):
        return dict(status="unknown", reason="unresolved_object_relocations_require_linked_bytes")
    by_pc = {row.addr: row for row in rows}
    if len(by_pc) != len(rows):
        return dict(status="unknown", reason="repeated_pc_requires_execution_view")
    registers, stack, fp = _initial_state(entry_state)
    pc, executions, command_counts = rows[0].addr, Counter(), Counter()
    commands = fences = host = 0
    summarized, assumed = Counter(), Counter()
    for step in range(max_steps):
        row = by_pc.get(pc)
        if row is None:
            summary = (call_summaries or {}).get(pc) if assume_calls else None
            if summary is not None and custom_opcode not in summary.get("major_opcodes", (custom_opcode,)):
                # A tail transfer to a callee that issues no selected-opcode word ends the body.
                assumed[pc] += 1
                d = CPUInstruction("return")
                row = None
            else:
                return dict(status="unknown", reason="control_leaves_selected_span", pc=pc)
        else:
            d = decode_cpu(row)
        custom = row is not None and len(row.hexcode) == 8 and int(row.hexcode, 16) & 127 == custom_opcode
        stepped = False
        if d.kind in CALLS:
            target = _call_target(d, registers, pc)
            summary = (call_summaries or {}).get(target)
            if (
                _summary_applies(summary, registers, stack)
                and summary.get("terminates") == "proved"
                and custom_opcode not in summary.get("major_opcodes", (custom_opcode,))
            ):
                stepped = True
                summarized[target] += 1
            elif (
                assume_calls
                and summary is not None
                and custom_opcode not in summary.get("major_opcodes", (custom_opcode,))
            ):
                stepped = True
                assumed[target] += 1
        if (
            (d.kind in CALLS and not stepped)
            or d.kind in ("indirect", "opaque")
            # A base-ISA word that only writes an integer register (a pure CSR read,
            # an FP compare or convert) leaves rd unknown; a branch on it still refuses.
            or (d.kind == "unknown" and not custom and _coarse_effect(row.hexcode, d)[0] != "none")
        ):
            return dict(status="unknown", reason="opaque_control_requires_capture", pc=pc)
        executions[pc] += 1
        if custom:
            commands += 1
            command_counts[pc] += 1
            if commands > max_commands:
                return dict(status="unknown", reason="finite_command_budget_exhausted")
            if recorder is not None:
                from merlin.analysis.rocc_commands import fields_of

                word = fields_of(int(row.hexcode, 16))
                recorder.add(
                    pc,
                    host,
                    funct=word["funct"],
                    rs1=_issue_operand(registers[word["rs1"]]),
                    rs2=_issue_operand(registers[word["rs2"]]),
                )
            host = 0
        else:
            if d.kind == "fence" and recorder is not None:
                recorder.add(pc, host, kind="fence")
            host += 1
        if d.kind == "fence":
            fences += 1
        if d.kind == "return":
            return dict(
                status="proved",
                count_basis="proved_finite_scalar_control",
                scope="selected_function_body",
                instruction_executions=dict(executions),
                command_executions=dict(command_counts),
                commands=commands,
                fences=fences,
                steps=step + 1,
                hardware_execution="unknown",
                **(dict(summarized_calls=dict(summarized)) if summarized else {}),
                **(dict(assumed_returning_calls=dict(assumed)) if assumed else {}),
                **(dict(symbolic_returns=dict(symbols)) if symbols else {}),
            )
        following = pc + d.size
        if recorder is not None and d.kind in (*BRANCHES, "jump") and d.target == pc:
            # A zero-displacement transfer is how an unlinked object's unresolved branch reads.
            return dict(status="unknown", reason="unlinked_object_use_linked_elf", pc=pc)
        if d.kind in BRANCHES:
            a, b = registers[d.rs1], registers[d.rs2]
            av, bv = a.exact, b.exact
            if av is None or bv is None:
                # Equality of offsets is invariant under pointer wraparound.
                # Ordered comparisons need a proven address range instead.
                if d.kind in ("beq", "bne") and a.base is not None and a.base == b.base:
                    av, bv = a.offset, b.offset
                else:
                    return dict(status="unknown", reason="data_dependent_or_unbound_control", pc=pc)
            if d.kind in ("blt", "bge"):
                av, bv = signed(av, 64), signed(bv, 64)
            condition = {
                "beq": av == bv,
                "bne": av != bv,
                "blt": av < bv,
                "bge": av >= bv,
                "bltu": av < bv,
                "bgeu": av >= bv,
            }[d.kind]
            following = d.target if condition else following
        elif d.kind == "jump":
            following = d.target
        _step(d, registers, stack, pc, row.hexcode, fp, call_summaries)
        if stepped and symbols is not None and registers[10].exact is None and registers[10].base is None:
            symbol = f"call:{target:#x}@{pc:#x}#{len(symbols)}"
            symbols[symbol] = dict(call_pc=pc, target=target)
            registers[10] = Bits(window=(symbol, 0, 0, 64))
        pc = following
    return dict(status="unknown", reason="finite_instruction_budget_exhausted")
