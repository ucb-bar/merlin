"""Top-down stream facts from a schedule kernel equal the bottom-up facts of its emitted binary.

A neutral synthetic target: every class, field, function code and register layout is test data. The
same kernel is (1) replayed through its target lowering by :mod:`merlin.kernels.schedule_effects` and
(2) emitted as straight-line RISC-V by an independent test emitter, then lifted by the machine stepper.
Both streams go through the same decoder and the same fact derivation.
"""

from __future__ import annotations

from merlin.compile.analysis.disassembly import RawInsn
from merlin.compile.analysis.riscv_machine import finite_control_counts
from merlin.kernels.decoded_effects import _STREAM_FACTS, decode_stream, stream_facts
from merlin.kernels.schedule_effects import (
    attribute,
    effects_record,
    replay_kernel,
    schedule_command_stream,
    schedule_effects,
    validate_stream_document,
)
from merlin.sched.ir import Kernel, Ptr, Stage, TensorArg, Var, add, call, concretize, instances, loop, mul
from merlin.sched.primitives import Cursor, divide_loop, multi_buffer

OPCODE = 0x5B
FUNCT = {"ld_a": 1, "ld_b": 2, "stage": 3, "mac": 4, "out": 5}
CLASS = {1: "LD_A", 2: "LD_B", 3: "STAGE", 4: "MAC", 5: "OUT"}
ROLES = {"LD_A": ["operand_load"], "LD_B": ["operand_load"], "STAGE": ["weight_load"], "MAC": ["accumulate"],
         "OUT": ["readout"]}
#: (field, register, bit offset, width); width ``None`` is the whole register as a memory operand.
LAYOUT = {
    "LD_A": [("src", "rs1", 0, None), ("at", "rs2", 0, 16), ("r", "rs2", 16, 8), ("c", "rs2", 24, 8)],
    "STAGE": [("w", "rs1", 0, 16), ("w_rows", "rs1", 16, 8)],
    "MAC": [("a", "rs1", 0, 16), ("a_rows", "rs1", 16, 8)],
}
LAYOUT["LD_B"] = LAYOUT["OUT"] = LAYOUT["LD_A"]


def _protocol_fields(klass):
    bundle = {"LD_A": "In", "LD_B": "In", "OUT": "Out", "STAGE": "S", "MAC": "M"}[klass]
    rtl = {"at": "laddr", "w": "laddr", "a": "laddr", "r": "nr", "w_rows": "nr", "a_rows": "nr", "c": "nc"}
    return [{"as": "operand", "name": n, "operand": slot} if width is None else
            {"bundle": bundle, "field": rtl[n], "name": n, "operand": slot} for n, slot, _, width in LAYOUT[klass]]


STRUCTURE = {
    "instruction_roles": ROLES,
    "operand_protocol": {"address_field": {"bundle": "Out", "field": "laddr"},
                         "roles": {"classes": {k: {"fields": _protocol_fields(k)} for k in ROLES}}},
    "load_state_selection": {"transfer_bound": {"rows": {"bundle": "In", "name": "nr"},
                                                "cols": {"bundle": "In", "name": "nc"}}},
    "local_geometry": {"uniform_row_elements": 8},
}


def _extract(record, offset, width):
    if type(record) is int:
        return (record >> offset) & ((1 << width) - 1)
    known = (record or {}).get("known_bits")
    need = ((1 << width) - 1) << offset
    if known and known["mask"] & need == need:
        return (known["value"] >> offset) & ((1 << width) - 1)
    return None


class _Observer:
    """The selected decoder of the synthetic target: fields straight from the declared register layout."""

    @staticmethod
    def isa_constants(target):
        return {"CUSTOM_OPCODE": OPCODE}

    @staticmethod
    def decode_instruction(funct, rs1, rs2, isa):
        klass = CLASS[funct]
        slots = {"rs1": rs1, "rs2": rs2}
        fields = {}
        for name, slot, offset, width in LAYOUT[klass]:
            record = slots[slot]
            if width is None:
                fields[name] = dict(record)
            else:
                fields[name] = _extract(record.get("raw") if record.get("kind") == "const" else record, offset, width)
        return klass, fields


DECODER = {"semantics": _Observer(), "target": "synthetic", "structure": {"instruction_roles": ROLES}}


# --------------------------------------------------------------------------- the schedule
def tiled_matmul(*, double_buffered=True):
    """C[8,8] = A[8,16] B[16,8] in 4x4 tiles; B tiles staged then read by the staging port."""
    i, j, k = Var("i"), Var("j"), Var("k")
    body = loop("i", 2, loop("j", 2,
        loop("k", 4,
             call("ld_b", src=Ptr("B", add(mul(k, 32), mul(j, 4))), at=0, rows=4, cols=4,
                  stages=(Stage("spad", 0, 4, writes=True),)),
             call("ld_a", src=Ptr("A", add(mul(i, 64), mul(k, 4))), at=64, rows=4, cols=4,
                  stages=(Stage("spad", 64, 4, writes=True),)),
             call("stage", w=0, rows=4, stages=(Stage("spad", 0, 4),)),
             call("mac", a=64, rows=4, stages=(Stage("spad", 64, 4),))),
        call("out", dst=Ptr("C", add(mul(i, 32), mul(j, 4))), at=128, rows=4, cols=4)))
    kernel = Kernel("mm", (TensorArg("A", (8, 16), "i8", "read"), TensorArg("B", (16, 8), "i8", "read"),
                           TensorArg("C", (8, 8), "i8", "write")), (body,))
    if not double_buffered:
        return kernel
    kernel = divide_loop(kernel, Cursor(("i", "j", "k")), 2)
    where = ("i", "j", "ko", "ki")
    kernel = multi_buffer(kernel, Cursor(where, 0), memory="spad", depth=2, rotate="at", over="ki", advance=16)
    return multi_buffer(kernel, Cursor(where, 2), memory="spad", depth=2, rotate="w", over="ki", advance=16)


def lowering(rows_source=None):
    """The synthetic target's lowering: pack each call's operands into its command registers."""

    def lower_call(ctx, instr, ops):
        fb = ctx.fb
        rows = ops["rows"] if rows_source is None else rows_source(ctx)
        if instr in ("ld_a", "ld_b", "out"):
            word = fb.or_i(fb.or_i(ops["at"], fb.shl_i(fb.and_i(rows, 0xFF), 16)), fb.shl_i(ops["cols"], 24))
            ctx.emit(FUNCT[instr], ops["src" if "src" in ops else "dst"], word)
        else:
            ctx.emit(FUNCT[instr], fb.or_i(ops["w" if instr == "stage" else "a"], fb.shl_i(ops["rows"], 16)), 0)

    return lower_call


def top_down(kernel, lower_call=None, *, spell=None):
    lower_call = lower_call or lowering()

    def entry(ctx):
        return replay_kernel(kernel, lambda instr, ops: lower_call(ctx, instr, ops), ctx, spell=spell)

    document = schedule_command_stream(entry, arguments=[t.name for t in kernel.args],
                                       specs={t.name: dict(dtype=t.dtype, shape=list(t.shape)) for t in kernel.args})
    return document, schedule_effects(document, decoder=DECODER, structure=STRUCTURE, evidence_id="ours")


# --------------------------------------------------------------------------- an independent emitter
def _addi(rd, rs, imm):
    return ((imm & 4095) << 20) | (rs << 15) | (rd << 7) | 0x13


def _lui(rd, imm20):
    return ((imm20 & 0xFFFFF) << 12) | (rd << 7) | 0x37


def _li(rd, value):
    hi, lo = (value + 0x800) >> 12, value & 0xFFF
    lo = lo - 4096 if lo >= 2048 else lo
    return [_lui(rd, hi), _addi(rd, rd, lo)] if hi else [_addi(rd, 0, lo)]


def _custom(funct, rs1, rs2):
    return (funct << 25) | (rs2 << 20) | (rs1 << 15) | (3 << 12) | OPCODE


def emit_binary(kernel, *, mutate=None):
    """Straight-line machine code for every instance: pointers from a0.., words materialized with li."""
    words, names = [], [t.name for t in kernel.args]
    for c, env in instances(kernel):
        ops = concretize(c, env)
        if mutate:
            ops = mutate(c.instr, ops)
        regs = []
        if c.instr in ("ld_a", "ld_b", "out"):
            ptr = ops["src" if "src" in ops else "dst"]
            regs = [("ptr", 10 + names.index(ptr.tensor), ptr.offset),
                    ("int", ops["at"] | ops["rows"] << 16 | ops["cols"] << 24)]
        else:
            regs = [("int", ops["w" if c.instr == "stage" else "a"] | ops["rows"] << 16), ("int", 0)]
        for rd, reg in zip((5, 6), regs, strict=True):
            words += [_addi(rd, reg[1], reg[2])] if reg[0] == "ptr" else _li(rd, reg[1])
        words.append(_custom(FUNCT[c.instr], 5, 6))
    words.append(0x00008067)
    return [RawInsn(4 * n, "ignored", [], f"{w:08x}") for n, w in enumerate(words)]


def bottom_up(rows, evidence_id="emitted"):
    counted = finite_control_counts(rows, custom_opcode=OPCODE, record_commands=True)
    assert counted["status"] == "proved"
    return stream_facts(decode_stream(counted["command_stream"], decoder=DECODER), STRUCTURE, evidence_id=evidence_id)


def _record(facts, evidence_id):
    return effects_record(dict(facts=facts), evidence_id=evidence_id)


# --------------------------------------------------------------------------- tests
def test_top_down_facts_equal_the_lift_of_the_emitted_binary():
    kernel = tiled_matmul()
    document, effects = top_down(kernel)
    assert document["status"] == "derived" and document["entries"] == 2 * 2 * 4 * 4 + 4
    lifted = bottom_up(emit_binary(kernel), evidence_id="ours")
    for name in _STREAM_FACTS:
        assert effects["facts"][name] == lifted[name], name
    buffering = effects["facts"]["operand_buffering"]["value"]["operand_load"]
    assert buffering["a1"]["disjoint_windows"] == 2 and buffering["a1"]["alternating"] is True
    assert buffering["a0"]["disjoint_windows"] == 1
    # Host instruction counts are not a schedule fact: top-down they stay unknown, never invented.
    host = [n for n, f in effects["facts"].items() if n not in _STREAM_FACTS and f.get("status") == "unknown"]
    assert all("host_instructions" in (effects["facts"][n].get("reason") or "") or
               effects["facts"][n].get("evidence", {}) for n in host)
    assert effects["annotations"] is not None and len(effects["annotations"]["multi_buffered_stagings"]) == 2
    assert validate_stream_document(document) is document


def test_emission_realization_flags_an_injected_change():
    kernel = tiled_matmul()
    _, effects = top_down(kernel)
    schedule = _record(effects["facts"], "ours")
    faithful = _record(bottom_up(emit_binary(kernel)), "emitted")
    report = attribute(schedule=schedule, emitted=faithful, expert=faithful)
    assert report["emission_realized_schedule"]["value"] is True and report["emission_changes"] == []

    def undo_rotation(instr, ops):
        return dict(ops, at=0) if instr == "ld_b" else ops

    changed = _record(bottom_up(emit_binary(kernel, mutate=undo_rotation)), "emitted")
    report = attribute(schedule=schedule, emitted=changed, expert=faithful)
    assert report["emission_realized_schedule"]["value"] is False
    assert any(c["effect"] == "operand_buffering" for c in report["emission_changes"])
    rows = [r for r in report["attribution"] if r["effect"] == "operand_buffering"]
    assert rows and {r["attribution"] for r in rows} == {"introduced_by_emission"}


def test_a_difference_already_in_the_schedule_is_a_lowering_choice():
    expert = _record(bottom_up(emit_binary(tiled_matmul())), "expert")
    single = tiled_matmul(double_buffered=False)
    _, effects = top_down(single)
    report = attribute(schedule=_record(effects["facts"], "ours"),
                       emitted=_record(bottom_up(emit_binary(single)), "emitted"), expert=expert)
    assert report["emission_realized_schedule"]["value"] is True
    rows = [r for r in report["attribution"] if r["effect"] == "operand_buffering"]
    assert rows and {r["attribution"] for r in rows} == {"lowering_design_choice"}


def test_a_field_the_schedule_does_not_determine_stays_unknown():
    lower_call = lowering(rows_source=lambda ctx: ctx.fb.load_i64(None, None, "i64"))
    document, effects = top_down(tiled_matmul(), lower_call)
    loads = [e for e in document["stream"] if e.get("funct") in (1, 2)]
    assert loads and all("known_bits" in e["rs2"] for e in loads)  # the address and columns stay known
    buffering = effects["facts"]["operand_buffering"]["value"]["operand_load"]["a1"]
    assert buffering["disjoint_windows"] is None and buffering["status"] == "unknown"
    shapes = effects["facts"]["role_transfer_width"]["value"]["data_movement"]["operand_load"]["shapes"]
    assert {s["rows"] for s in shapes} == {None} and {s["cols"] for s in shapes} == {4}


def test_unspelled_asynchrony_refuses_and_every_fact_is_unknown():
    kernel = tiled_matmul(double_buffered=False)
    first = kernel.body[0].body[0].body[0].body[0]
    tokened = Kernel(kernel.name, kernel.args, _replace_first(kernel.body, first, produces="t"))
    document, effects = top_down(tokened)
    assert document["status"] == "unknown" and "on_produce" in document["reason"]
    assert all(effects["facts"][n]["status"] == "unknown" for n in _STREAM_FACTS)

    class Spell:
        @staticmethod
        def on_produce(c, ctx):
            ctx.fence()

    document, effects = top_down(tokened, spell=Spell())
    fences = sum(e.get("kind") == "fence" for e in document["stream"])
    assert document["status"] == "derived" and fences == 16  # one per dynamic instance of the producer


def _replace_first(body, target, **change):
    from dataclasses import replace

    from merlin.sched.ir import Loop

    out = []
    for s in body:
        if s is target:
            out.append(replace(s, **change))
        elif isinstance(s, Loop):
            out.append(replace(s, body=_replace_first(s.body, target, **change)))
        else:
            out.append(s)
    return tuple(out)


def test_the_llvm_index_bridge_drives_the_replay_and_folds_constant_selections():
    """A lowering that emits through the ordinary index bridge needs nothing replay-specific."""
    from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter, emit_index_program
    from merlin.kernels.schedule_effects import ReplayContext
    from merlin.sched.ir import select

    k = Var("k")
    kernel = Kernel("tail", (TensorArg("A", (4, 16), "i8", "read"),), (loop("k", 4, call(
        "mac", a=add(mul(k, 4), 0), rows=select("k", 3, 2, 4))),))
    ctx = ReplayContext(["A"])
    lower_call = lowering()
    emit_index_program(kernel, FnBuilderIndexAdapter(ctx.fb), lambda instr, ops: lower_call(ctx, instr, ops))
    assert [(e["rs1"] & 0xFFFF, e["rs1"] >> 16) for e in ctx.stream] == [(0, 4), (4, 4), (8, 4), (12, 2)]


def test_order_facts_lead_and_the_shape_sweep_reads_bounds_from_the_facts():
    from merlin.kernels.schedule_effects import ORDER_FACTS, shape_sweep

    expert = _record(bottom_up(emit_binary(tiled_matmul())), "expert")
    single = tiled_matmul(double_buffered=False)
    _, effects = top_down(single)
    report = attribute(schedule=_record(effects["facts"], "ours"),
                       emitted=_record(bottom_up(emit_binary(single)), "emitted"), expert=expert)
    effects_seen = [r["effect"] for r in report["comparisons"]["schedule_vs_expert"]]
    first_count = next(i for i, e in enumerate(effects_seen) if e not in ORDER_FACTS)
    assert all(e not in ORDER_FACTS for e in effects_seen[first_count:])
    assert effects["ordered_stream"]["timing"].startswith("not_claimed")

    def instantiate(double):
        kernel = tiled_matmul(double_buffered=double)
        lower_call = lowering()
        return (lambda ctx: replay_kernel(kernel, lambda i, ops: lower_call(ctx, i, ops), ctx),
                [t.name for t in kernel.args], {t.name: dict(dtype=t.dtype) for t in kernel.args})

    classes = {name: {"entries": {"status": "derived", "value": depth}} for name, depth in (("move", 2), ("run", 8))}
    facts = {"facts": {"timing_structure": {"command_classes": {"classes": classes}}}}
    routing = {"LD_A": "move", "LD_B": "move", "OUT": "move", "STAGE": "run", "MAC": "run"}
    sweep = shape_sweep(instantiate, [False, True], decoder=DECODER, structure=STRUCTURE, facts=facts,
                        routing=routing)
    assert [row["status"] for row in sweep["shapes"]] == ["derived", "derived"]
    assert list(sweep["shapes"][0]["facts"])[0] == ORDER_FACTS[0]
    bounds = {r["role"]: r for r in sweep["shapes"][0]["resource_bounds"]}
    assert bounds["operand_load"]["bound"] == 2 and bounds["operand_load"]["exceeds"] is True
    unrouted = shape_sweep(instantiate, [True], decoder=DECODER, structure=STRUCTURE, facts=facts)
    assert {r["status"] for r in unrouted["shapes"][0]["resource_bounds"]} == {"unknown"}
    absent = shape_sweep(instantiate, [True], decoder=DECODER, structure=STRUCTURE, facts={})
    assert absent["command_class_depths"]["status"] == "unknown"


def test_author_tool_serializes_a_stream_the_consumer_validates(tmp_path, monkeypatch):
    import json
    import sys
    import types

    import pytest

    from merlin.kernels.schedule_effects import main

    kernel = tiled_matmul()
    module = types.ModuleType("synthetic_author_lowering")
    lower_call = lowering()
    module.entry = lambda ctx: replay_kernel(kernel, lambda i, ops: lower_call(ctx, i, ops), ctx)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    out = tmp_path / "stream.json"
    assert main([f"{module.__name__}:entry", "--arguments", "A,B,C", "--out", str(out)]) == 0
    document = validate_stream_document(json.loads(out.read_text()))
    direct, _ = top_down(kernel)
    assert document["stream"] == direct["stream"]
    document["stream"][0]["funct"] = 9
    with pytest.raises(ValueError, match="schedule_stream_digest_mismatch"):
        validate_stream_document(document)
