"""Trace requirements judge what a loop-emitted schedule EXECUTES, not its static sites.

A library dense schedule emits one readout inside a counted loop nest: the static stream holds one
MVOUT whose local address is ``base | f(induction)``. Counted as one site with an unknown address, a
correct 1024-tile kernel read as "MVOUT count 1", "no MVOUT has i8 readout" and "unresolved operand
fields", and those false vetoes withheld every natively passing scalability case from final closure.
"""

import pytest

from merlin.targetgen import trace_check as TC
from merlin.targetgen.rocc.decode import _parse_module
from merlin.targetgen.rocc.static_operands import counted_loop_facts, integer_range, known_integer_bits

# A counted loop of TRIPS iterations whose body issues one command with rs2 = FLAG | (iv << SHIFT).
TRIPS, SHIFT, FLAG = 64, 4, 1 << 31
LOOP = f"""
module {{
  llvm.func @kernel(%base: i64) {{
    %zero = llvm.mlir.constant(0 : i64) : i64
    llvm.br ^head(%zero : i64)
  ^head(%iv: i64):
    %trips = llvm.mlir.constant({TRIPS} : i64) : i64
    %more = llvm.icmp "slt" %iv, %trips : i64
    llvm.cond_br %more, ^body, ^exit
  ^body:
    %shift = llvm.mlir.constant({SHIFT} : i64) : i64
    %row = llvm.shl %iv, %shift : i64
    %flag = llvm.mlir.constant({FLAG} : i64) : i64
    %addr = llvm.or %row, %flag : i64
    %one = llvm.mlir.constant(1 : i64) : i64
    %next = llvm.add %iv, %one : i64
    llvm.br ^head(%next : i64)
  ^exit:
    llvm.return
  }}
}}
"""


def _loop():
    module = _parse_module(LOOP)
    assert module is not None
    module.verify()
    function = module.body.block.first_op
    addr = next(op.results[0] for op in function.walk() if op.name == "llvm.or")
    base = function.body.blocks.first.args[0]
    return module, function, addr, base


def test_a_counted_induction_bounds_the_bits_it_feeds_and_its_block_executions():
    module, function, addr, base = _loop()
    facts = counted_loop_facts(module)
    low, high = integer_range(addr, facts["ranges"])
    assert low == FLAG and (TRIPS << SHIFT) | FLAG <= high < 1 << 32
    mask, bits, width = known_integer_bits(addr, {}, facts["ranges"])
    # Without ranges the shifted induction is unknown; with them every bit above it is known zero,
    # so a flag such as bit 30 or 29 of the address is decided, and the set flag bit is known one.
    assert known_integer_bits(addr, {})[0] & (1 << 29) == 0
    assert mask & (1 << 29) and mask & (1 << 30) and bits & FLAG and not bits & ((1 << 29) | (1 << 30))
    for iv in range(TRIPS + 1):  # sound for every value the induction takes, its exit value included
        assert ((iv << SHIFT) | FLAG) & mask == bits
    body = next(block for block in function.body.blocks if any(op.name == "llvm.or" for op in block.ops))
    assert facts["executions"][body] == TRIPS
    assert integer_range(base, facts["ranges"]) is None  # an ABI argument stays unbounded
    assert facts["observation"]["instruction_multiplicity"] == "executions_per_kernel_call"


def _row(cls, executions=1, **decoded):
    return {"class": cls, "funct": 1, "executions": executions, "decoded": decoded}


def _load(executions=1):
    source = {"kind": "argbase", "raw": None, "arg_index": 0, "offset": 0}
    return _row("MVIN", executions, dram=source, spad_addr=0, rows=16, cols=16)


def _check(rows, modes, monkeypatch, cb=None):
    trace = {"instructions": [{"class": "FENCE", "funct": None, "executions": 1, "decoded": {}}, *rows,
                              {"class": "FENCE", "funct": None, "executions": 1, "decoded": {}}]}
    monkeypatch.setattr(TC, "_bound_trace_view", lambda *_: (trace, {}, [], True))
    return TC.check(trace, {"modes": modes}, cb=cb, target="fixture")


def _cb(m, n, commits=1):
    tensors = {"A0": {"shape": [m, 16]}, "W": {"shape": [16, n]}, **{f"Y{i}": {"shape": [m, n]} for i in range(commits)}}
    commands = [{"opcode": "RES_PACK", "operands": {"src": "W", "dst": "W_res"}}]
    for i in range(commits):
        commands += [{"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "A0", "rhs": "W_res", "dst": f"acc{i}"}},
                     {"opcode": "COMMIT", "operands": {"src": f"acc{i}", "dst": f"Y{i}"}}]
    return {"tensors": tensors, "commands": commands}


def test_a_looped_schedule_that_is_correct_passes(monkeypatch):
    looped = [_load(), _row("PRELOAD", 1024), _row("COMPUTE_PRELOADED", 1024),
              _row("MVOUT", 1024, readout="i8", rows=16, cols=16)]
    result = _check(looped, {"i8": True, "resident_reuse": True}, monkeypatch, cb=_cb(512, 512))
    assert result["status"] == "pass" and result["trace_requirement_status"] == "pass", result
    assert TC.requirements_complete(result)
    assert result["tile_schedule_advisory"]["findings"] == []  # 1024 executed readouts = Mt*Nt
    assert TC.executed_count(looped, "MVOUT") == 1024


def test_a_genuinely_wrong_readout_mode_still_fails(monkeypatch):
    rows = [_row("PRELOAD", 64), _row("COMPUTE_PRELOADED", 64), _row("MVOUT", 64, readout="i32", rows=16, cols=16)]
    result = _check(rows, {"i8": True}, monkeypatch)
    assert result["status"] == "fail" and result["trace_requirement_status"] == "failed"
    assert "mode i8 declared but no MVOUT has i8 readout" in result["violations"]


def test_an_unresolved_readout_is_unavailable_evidence_not_a_failure(monkeypatch):
    rows = [_row("PRELOAD", 64), _row("COMPUTE_PRELOADED", 64), _row("MVOUT", 64, rows=16, cols=16)]
    result = _check(rows, {"i8": True}, monkeypatch)
    assert result["violations"] == [] and result["status"] == "pass"
    assert result["trace_requirement_status"] == "unavailable_fields"
    assert result["unavailable_requirements"] == ["i8"] and not TC.requirements_complete(result)


@pytest.mark.parametrize("executions,expected", [(4, "pass"), (None, "unavailable_fields"), (1, "failed")])
def test_resident_reuse_counts_executed_output_commits(monkeypatch, executions, expected):
    rows = [_load(), _row("PRELOAD", executions), _row("COMPUTE_PRELOADED", executions),
            _row("MVOUT", executions, readout="i8", rows=16, cols=16)]
    result = _check(rows, {"resident_reuse": True}, monkeypatch, cb=_cb(64, 16, commits=2))
    assert result["trace_requirement_status"] == expected
    assert (result["status"] == "fail") == (expected == "failed")


def test_one_declared_output_commit_cannot_show_reuse_and_is_not_a_violation(monkeypatch):
    rows = [_load(), _row("PRELOAD"), _row("COMPUTE_PRELOADED"), _row("MVOUT", readout="i8", rows=16, cols=16)]
    result = _check(rows, {"resident_reuse": True}, monkeypatch, cb=_cb(16, 16))
    assert result["status"] == "pass" and result["trace_requirement_status"] == "pass"
    assert any("one output commit" in note for note in result["satisfied_otherwise"])
    # The same single readout for an output larger than it covers is still no reuse at all.
    larger = _check(rows, {"resident_reuse": True}, monkeypatch, cb=_cb(32, 16))
    assert "mode resident_reuse declared but <2 output commits (no reuse visible)" in larger["violations"]


def test_historical_traces_without_executions_count_static_sites():
    rows = [{"class": "MVOUT"}, {"class": "MVOUT"}, {"class": "MVIN"}]
    assert TC.executed_count(rows, "MVOUT") == 2
    assert TC.executed_count([{"class": "MVOUT", "executions": None}], "MVOUT") is None
