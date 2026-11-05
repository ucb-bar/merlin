"""Hardware-legality checks of the GENERIC RoCC rule engine, on the Gemmini contract + RTL facts.

``RC`` is :mod:`merlin.targetgen.rtl_checks_generic` evaluating the Gemmini contract's declared
``rtl_checks`` protocol (see ``rocc_generic_test_support``); ``P`` is that protocol.

Three gates, none of which consults any compiler's output:

(a) a MINIMAL LEGAL PROGRAM, written here from the public Gemmini ISA command semantics and encoded
    through the RTL-extracted register-bundle layouts, decodes cleanly and passes every legality rule;
(b) MUTATIONS of that program that each break one legality property (an RTL-illegal funct, a missing
    fence, use before configuration, an out-of-bounds local address, a missing store) are flagged by
    exactly the rule that owns the property;
(c) the decode agrees with the RTL facts for every funct.

Everything else is predicate-based on synthetic traces: the invariant, never "capsule X has N MVOUTs".
"""

from __future__ import annotations

import copy
import os
import struct
from pathlib import Path

import pytest
from rocc_generic_test_support import checks as RC
from rocc_generic_test_support import protocol as P
from rocc_generic_test_support import selected as SELECTED

from merlin.targetgen import rtl_check_compiler as CC
from merlin.targetgen import rtl_check_runner as RUN
from merlin.targetgen.rocc import decode
from merlin.targetgen.rocc import semantics as S
from merlin.targetgen.rtl.facts import load_facts

TARGET = "gemmini"
FACTS = load_facts(TARGET)
#: Extra fact bundles to cross-check decoding against, ``os.pathsep``-separated (operator-selected).
_FACT_SETS = [p for p in os.environ.get("MERLIN_TEST_EXTRA_RTL_FACTS", "").split(os.pathsep) if p and Path(p).is_file()]


@pytest.fixture(autouse=True)
def _generic_decoder(monkeypatch):
    """Decode through the generic semantics (no support backend needs to be selected)."""
    monkeypatch.setattr(decode, "_semantics", lambda target: S)


# ======================================================================= a minimal legal program
def _f32(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", v))[0]


def _pack(bundle: str, **values: int) -> int:
    """Encode named fields with the RTL-extracted bundle layout (no offset is written in this test)."""
    layouts = S.bundle_layouts(FACTS)
    word = 0
    for name, value in values.items():
        offset, width = S.field_slot(layouts.get(bundle), name)
        assert 0 <= value < (1 << width), (bundle, name, value)
        word |= value << offset
    return word


def _subtype(code: int) -> int:
    return _pack("ConfigMvoutRs1", cmd_type=code)  # the CONFIG selector field the RTL decodes


def _program() -> list[tuple]:
    """A 16x16 i8 x i8 -> i32 product C = A @ W, from the public ISA command semantics.

    Entries are ``("fence",)`` or ``(class, rs1, rs2[, funct])``; an operand is an int or
    ``("arg", i, byte_offset)``. CONFIG_EX picks the dataflow; CONFIG_LD / CONFIG_ST set the DRAM row
    stride (and scale) the load / store paths use; MVIN moves rows x cols from DRAM to a local address;
    PRELOAD latches the B tile and names the C address; COMPUTE_PRELOADED multiplies an A tile against it
    (an all-ones B/D operand loads nothing new); MVOUT reads the accumulator back to DRAM.
    """
    isa = S.isa_constants(TARGET)
    acc, full, one = isa["ACC_I8"], isa["FULL_C_BIT"], _f32(1.0)
    tile = {"num_rows": 16, "num_cols": 16}
    sub = {name: code for code, name in isa["CONFIG_SUBTYPE"].items()}
    return [
        ("fence",),
        ("CONFIG_EX", _subtype(sub["CONFIG_EX"]) | _pack("ConfigExRs1", dataflow=1, acc_scale=one), 0),
        ("CONFIG_LD", _subtype(sub["CONFIG_LD"]) | _pack("ConfigMvinRs1", scale=one), 16),
        ("CONFIG_ST", _subtype(sub["CONFIG_ST"]), _pack("ConfigMvoutRs2", stride=64, acc_scale=one)),
        ("MVIN", ("arg", 0, 0), _pack("MvinRs2", local_addr=0, **tile)),
        ("MVIN", ("arg", 1, 0), _pack("MvinRs2", local_addr=16, **tile)),
        ("PRELOAD", _pack("PreloadRs", local_addr=16, **tile), _pack("PreloadRs", local_addr=acc, **tile)),
        ("COMPUTE_PRELOADED", _pack("ComputeRs", local_addr=0, **tile), isa["RETAIN_SENTINEL"]),
        ("MVOUT", ("arg", 2, 0), _pack("MvoutRs2", local_addr=acc | full, **tile)),
        ("fence",),
    ]


CAPSULE = {
    "name": "minimal_legal_matmul",
    "operation": {"op": "matmul", "attributes": {"lhs": "A", "weight": "W", "out": "C", "output_dtype": "i32"}},
    "inputs": [
        {"name": "A", "role": "input", "shape": [16, 16], "dtype": "i8"},
        {"name": "W", "role": "weight", "shape": [16, 16], "dtype": "i8"},
    ],
}


def _mlir(program: list[tuple]) -> str:
    """The program as the ``lowered.llvm.mlir`` a backend emits: ``.insn r`` inline asm over SSA operands."""
    isa = S.isa_constants(TARGET)
    body, asm = [], []
    counter = [3]

    def ssa(text: str) -> str:
        name = f"%{counter[0]}"
        counter[0] += 1
        body.append(f"    {name} = {text}")
        return name

    def operand(v) -> str:
        if isinstance(v, tuple):
            base = ssa(f"llvm.ptrtoint %{v[1]} : !llvm.ptr to i64")
            return ssa(f"llvm.add {base}, {ssa(f'llvm.mlir.constant({v[2]}) : i64')} : i64") if v[2] else base
        return ssa(f"llvm.mlir.constant({v}) : i64")

    for ins in program:
        if ins[0] == "fence":
            asm.append('    llvm.inline_asm has_side_effects "fence", "" : () -> ()')
            continue
        cls, rs1, rs2 = ins[:3]
        funct = ins[3] if len(ins) > 3 else S.instruction_funct(cls, rs1 if isinstance(rs1, int) else 0, isa)
        a, b = operand(rs1), operand(rs2)
        asm.append(
            f'    llvm.inline_asm has_side_effects ".insn r {isa["CUSTOM_OPCODE"]:#x}, {isa["FUNCT3"]:#x}, '
            f'{funct:#x}, x0, $0, $1", "r,r" {a}, {b} : (i64, i64) -> ()'
        )
    head = "builtin.module {\n  llvm.func @gemmini_kernel(%0: !llvm.ptr, %1: !llvm.ptr, %2: !llvm.ptr) {\n"
    return head + "\n".join(body + asm) + "\n    llvm.return\n  }\n}\n"


def _screen(program: list[tuple]):
    trace = decode.decode_text(_mlir(program), source="minimal", target=TARGET)
    return trace, RC.screen(trace, CAPSULE, target=TARGET)


def test_a_minimal_legal_program_decodes_cleanly_and_passes_every_legality_rule():
    trace, rep = _screen(_program())
    assert [i["class"] for i in trace["instructions"]] == [
        "FENCE",
        "CONFIG_EX",
        "CONFIG_LD",
        "CONFIG_ST",
        "MVIN",
        "MVIN",
        "PRELOAD",
        "COMPUTE_PRELOADED",
        "MVOUT",
        "FENCE",
    ]
    assert trace["instructions"][4]["decoded"]["dram"]["kind"] == "argbase"
    assert trace["instructions"][4]["decoded"]["rows"] == 16
    assert trace["instructions"][8]["decoded"]["readout"] == "i32"
    assert rep.verdict == "ok", [c.to_dict() for c in rep.checks if c.status != "pass"]
    assert {c.status for c in rep.checks} == {"pass"}
    assert [s["id"] for s in P.rules] == [c.id for c in rep.checks]


def _without(program, cls):
    k = next(k for k, ins in enumerate(program) if ins[0] == cls)
    return program[:k] + program[k + 1 :]


def _mvin_at(program, row):
    k = next(k for k, ins in enumerate(program) if ins[0] == "MVIN")
    return (
        program[:k]
        + [("MVIN", program[k][1], _pack("MvinRs2", local_addr=row, num_rows=16, num_cols=16))]
        + program[k + 1 :]
    )


SPAD_ROWS = RC.load_default_facts(TARGET)["scratchpad_rows"]

MUTATIONS = {
    "illegal_funct": (lambda p: p[:-1] + [("FLUSH", 0, 0, 99), p[-1]], {"T0.decode_funct_legal", "T0.decode_clean"}),
    "missing_leading_fence": (lambda p: p[1:], {"T0.fence_bracket"}),
    "missing_trailing_fence": (lambda p: p[:-1], {"T0.fence_bracket"}),
    "store_before_config": (lambda p: _without(p, "CONFIG_ST"), {"T0.config_before_use"}),
    "compute_before_config": (lambda p: _without(p, "CONFIG_EX"), {"T0.config_before_use"}),
    "load_before_config": (lambda p: _without(p, "CONFIG_LD"), {"T0.config_before_use"}),
    "compute_without_preload": (lambda p: _without(p, "PRELOAD"), {"T0.preload_before_compute"}),
    "spad_out_of_bounds": (lambda p: _mvin_at(p, SPAD_ROWS - 8), {"T0.local_address_bounds"}),
    "missing_store": (lambda p: _without(p, "MVOUT"), {"T0.output_store_coverage"}),
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_each_illegal_mutation_is_flagged_by_the_rule_that_owns_it(name):
    mutate, owners = MUTATIONS[name]
    _trace, rep = _screen(mutate(_program()))
    assert {c.id for c in rep.checks if c.status == "fail"} == owners, [c.to_dict() for c in rep.checks]
    assert rep.verdict == ("reject" if any(c.severity == "error" for c in rep.checks if c.id in owners) else "warn")


# ============================================================================ decode vs RTL facts
@pytest.mark.parametrize("facts_path", _FACT_SETS or [None])
def test_decode_agrees_with_the_rtl_facts_for_every_funct(monkeypatch, facts_path):
    if facts_path is not None:
        monkeypatch.setenv("MERLIN_RTL_FACTS", facts_path)
    table = next(i for i in load_facts(TARGET)["facts"]["interfaces"] if i["name"] == "funct_decode_table")
    legal = set(table["legal_funct"])
    isa = S.isa_constants(TARGET)
    declared = isa["FUNCT_CLASS"]
    assert set(declared) <= legal, f"the encoding declares functs the RTL decoder rejects: {set(declared) - legal}"
    config = next(k for k, v in declared.items() if v == isa["CONFIG_CLASS"])
    zero = {"raw": 0, "kind": "const", "arg_index": None, "offset": None}
    for funct in range(1 << S.ROCC_FUNCT7_WIDTH):
        cls, _dec = S.decode_instruction(funct, zero, zero, isa)
        if funct == config:
            assert cls in isa["CONFIG_SUBTYPE"].values()
        else:
            assert cls == declared.get(funct, "UNKNOWN"), funct
        if funct in declared and funct != config:
            assert S.instruction_funct(cls, 0, isa) == funct
    # The legality rule reads exactly the decoder's legal set of the selected facts.
    assert set(RC.load_default_facts(TARGET)["legal_funct"]) == legal


# =================================================================== local-address bounds (units)
def _trace(classes_functs):
    return {
        "source": "synthetic",
        "abi": {"custom_opcode": "0x7b", "funct3": "0x3"},
        "instructions": [
            {"index": i, "class": c, "funct": f, "decoded": {}} for i, (c, f) in enumerate(classes_functs)
        ],
    }


def test_accumulator_space_is_decoded_before_the_bound_is_applied():
    facts = RC.load_default_facts(TARGET)
    t = _trace([("MVIN", 2), ("MVIN", 2)])
    t["instructions"][0]["decoded"] = {"spad_addr": facts["accumulator_select_bit"], "rows": 16}
    t["instructions"][1]["decoded"] = {"spad_addr": facts["accumulator_select_bit"] | 16, "rows": 16}
    check = RC._check_local_address_bounds(t, facts, P)
    assert check.status == "pass", check.to_dict()
    assert check.evidence["accumulator_max_row"] == 16
    assert check.evidence["accumulator_row_mask"] == facts["accumulator_rows"] - 1


def test_accumulator_overflow_and_noncanonical_payload_are_rejected():
    facts = RC.load_default_facts(TARGET)
    select = facts["accumulator_select_bit"]
    t = _trace([("MVIN", 2), ("PRELOAD", 6)])
    t["instructions"][0]["decoded"] = {"spad_addr": select | (facts["accumulator_rows"] - 8), "rows": 16}
    t["instructions"][1]["decoded"] = {"c_addr": select | (facts["accumulator_row_mask"] + 1)}
    check = RC._check_local_address_bounds(t, facts, P)
    assert check.status == "fail", check.to_dict()
    assert check.evidence["accumulator_max_row_exclusive"] == facts["accumulator_rows"] + 8
    assert check.evidence["noncanonical_accumulator_instruction_indices"] == [1]


def test_the_retain_sentinel_is_not_an_address():
    facts = RC.load_default_facts(TARGET)
    t = _trace([("COMPUTE_PRELOADED", 4)])
    t["instructions"][0]["decoded"] = {"a_spad": 0, "bd": facts["local_address_sentinel"]}
    assert RC._check_local_address_bounds(t, facts, P).status == "pass"


def test_unknown_selector_never_reads_a_tagged_address_as_scratchpad():
    facts = {
        "scratchpad_rows": 64,
        "scratchpad_row_mask": 0x3F,
        "accumulator_rows": 8,
        "accumulator_row_mask": 0x7,
        "local_address_data_mask": 0x3F,
        "accumulator_select_bit": None,
    }
    t = _trace([("MVIN", 2)])
    t["instructions"][0]["decoded"] = {"spad_addr": 0x100, "rows": 1}
    check = RC._check_local_address_bounds(t, facts, P)
    assert check.status == "skipped", check.to_dict()
    assert check.evidence["unresolved_address_space_instruction_indices"] == [0]


def test_selector_comes_from_facts_not_a_bit31_literal():
    facts = {"scratchpad_rows": 64, "accumulator_rows": 8, "accumulator_select_bit": 0x100, "accumulator_row_mask": 0x7}
    t = _trace([("MVIN", 2)])
    t["instructions"][0]["decoded"] = {"spad_addr": 0x1C0 | 7, "rows": 1}
    check = RC._check_local_address_bounds(t, facts, P)
    assert check.status == "pass"
    assert check.evidence["accumulator_max_row"] == 7


@pytest.mark.parametrize(
    ("rows", "status", "key"),
    [
        (0, "fail", "invalid_row_count_instruction_indices"),
        (-1, "fail", "invalid_row_count_instruction_indices"),
        (None, "skipped", "unresolved_row_count_instruction_indices"),
    ],
)
def test_transfer_row_counts_must_be_positive_and_decoded(rows, status, key):
    facts = RC.load_default_facts(TARGET)
    t = _trace([("MVIN2", 1)])
    t["instructions"][0]["decoded"] = {"spad_addr": 0} if rows is None else {"spad_addr": 0, "rows": rows}
    check = RC._check_local_address_bounds(t, facts, P)
    assert check.status == status, check.to_dict()
    assert check.evidence[key] == [0]


# ====================================================================== store coverage (units)
def _store_trace(pocols=None):
    config = {"out_stride_bytes": 64}
    if pocols is not None:
        config.update(pool_stride=2, porows=2, pocols=pocols)
    t = _trace([("CONFIG_ST", 0), ("MVOUT", 3)])
    t["instructions"][0]["decoded"] = config
    t["instructions"][1]["decoded"] = {"dram": {"kind": "argbase", "arg_index": 2, "offset": 0}, "rows": 16, "cols": 16}
    return t


def test_store_footprint_follows_the_encoded_pooling_fields():
    pooled = copy.deepcopy(CAPSULE)
    pooled["operation"]["attributes"].update(
        epilogue=["maxpool"], pool_in_dims=[4, 4], pool_size=[2, 2], pool_stride=[2, 2]
    )
    supported = {"max_pool_supported": True}
    assert RC._check_output_store_coverage(_store_trace(pocols=2), pooled, supported, None, P).status == "pass"
    assert RC._check_output_store_coverage(_store_trace(pocols=1), pooled, supported, None, P).status == "fail"
    unknown = {"max_pool_supported": None}
    assert RC._check_output_store_coverage(_store_trace(pocols=2), pooled, unknown, None, P).status == "skipped"
    # An elaboration without the pooling feature ignores those fields: the encoded rows are stored.
    disabled = {"max_pool_supported": False}
    assert RC._check_output_store_coverage(_store_trace(pocols=2), CAPSULE, disabled, None, P).status == "pass"


def test_a_store_past_the_declared_extent_is_flagged():
    t = _store_trace()
    t["instructions"][1]["decoded"]["rows"] = 32
    check = RC._check_output_store_coverage(t, CAPSULE, {}, None, P)
    assert check.status == "fail" and "past the extent" in check.message


# ========================================================================= FileCheck assertions
def test_filecheck_asserts_only_facts_grounded_legality():
    compiled = CC.compile_trace_checks(FACTS, CAPSULE, target=TARGET, checks=RC)
    assert "ILLEGAL_FUNCT_COUNT 0{{$}}" in compiled and "UNKNOWN_COUNT 0{{$}}" in compiled
    for schedule_dependent in ("MVOUT_COUNT", "MVIN_COUNT", "COMPUTE_PRESENT", "MVIN_PRESENT"):
        assert schedule_dependent not in compiled


@pytest.mark.skipif(RUN.find_filecheck() is None, reason="FileCheck binary not found")
def test_filecheck_passes_the_legal_program_and_catches_illegal_ones():
    fc = RUN.find_filecheck()
    cc = CC.compile_checks(FACTS, CAPSULE, TARGET, checks=RC)

    def ok(program):
        trace = decode.decode_text(_mlir(program), target=TARGET)
        return RUN.run_filecheck(fc, cc["trace"], RUN.render_trace(trace, FACTS, target=TARGET, checks=RC), "TRACE")[0]

    program = _program()
    assert ok(program)
    assert not ok(MUTATIONS["illegal_funct"][0](program))
    assert ok(program[:-1] + [program[-2], program[-1]])  # more stores than one schedule emits is not illegal


# ============================================================================ selected capability
def test_generic_semantics_expose_the_rule_engine_as_rtl_checks():
    for name in ("isa_constants", "decode_instruction", "instruction_funct"):
        assert callable(getattr(S, name))
    assert S.rtl_checks is RC
    for name in ("load_default_facts", "project_facts", "screen", "compile_trace_checks", "render_trace"):
        assert callable(getattr(S.rtl_checks, name))


@pytest.mark.skipif(SELECTED is None, reason="no explicit Gemmini support selection")
def test_selected_gemmini_support_resolves_the_generic_rule_engine():
    assert SELECTED is RC
