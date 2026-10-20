"""Count-only loop summaries preserve exact control, including padded paths."""

import numpy as np
import pytest
from xdsl.dialects.builtin import IntegerAttr, UnitAttr

from merlin.perf import constant_control_cfg as CC
from merlin.perf.constant_control_regions import control_regions
from merlin.targetgen.rocc.decode import _parse_module
from merlin.verify.host_compute_audit import _functions
from merlin.verify.input_copy_batch import execute
from merlin.verify.input_copy_integer import integer


def function(rows=32, columns=16, step=1):
    return _functions(
        _parse_module(f"""module {{
      llvm.func @padding(%input: !llvm.ptr) {{
        %zero = llvm.mlir.constant(0 : i64) : i64
        %one = llvm.mlir.constant(1 : i64) : i64
        %step = llvm.mlir.constant({step} : i64) : i64
        %rows = llvm.mlir.constant({rows} : i64) : i64
        %columns = llvm.mlir.constant({columns} : i64) : i64
        llvm.br ^outer(%zero : i64)
      ^outer(%i: i64):
        %more = llvm.icmp "ult" %i, %rows : i64
        llvm.cond_br %more, ^tile, ^done
      ^tile:
        llvm.br ^inner(%zero : i64)
      ^inner(%j: i64):
        %again = llvm.icmp "ult" %j, %columns : i64
        llvm.cond_br %again, ^choose, ^latch
      ^choose:
        %x = llvm.add %i, %j : i64
        %inside = llvm.icmp "ult" %x, %rows : i64
        llvm.cond_br %inside, ^valid, ^pad
      ^valid:
        %v = llvm.inline_asm has_side_effects asm_dialect = att "fence", "~{{memory}}" : () -> !llvm.void
        llvm.br ^join
      ^pad:
        %p = llvm.inline_asm has_side_effects asm_dialect = att "fence", "~{{memory}}" : () -> !llvm.void
        llvm.br ^join
      ^join:
        %next = llvm.add %j, %step : i64
        llvm.br ^inner(%next : i64)
      ^latch:
        %following = llvm.add %i, %one : i64
        llvm.br ^outer(%following : i64)
      ^done:
        llvm.return
      }}
    }}""")
    )["padding"]


@pytest.mark.parametrize("rows,columns,step", [(32, 16, 1), (7, 9, 2), (1, 1, 1), (0, 9, 1), (4, 0, 1)])
def test_branched_pretested_counts_equal_independent_scalar_trace(rows, columns, step):
    f = function(rows, columns, step)
    assert CC.constant_control_counts(f) == CC.constant_control_counts(f, summarize_loops=False)


def test_padding_summary_keeps_final_false_header_and_each_arm_count():
    f = function()
    assert list(CC.constant_control_counts(f, max_blocks=256).values()) == [
        1,
        33,
        32,
        544,
        512,
        392,
        120,
        512,
        32,
        1,
    ]
    with pytest.raises(ValueError, match="block budget"):
        CC.constant_control_counts(f, max_blocks=256, summarize_loops=False)


@pytest.mark.parametrize("origin", ["llvm.load %input : !llvm.ptr -> i64", 'llvm.inline_asm "", "=r" : () -> i64'])
def test_loaded_or_opaque_branch_operand_never_becomes_constant(origin):
    f = function()
    block = list(f.body.blocks)[4]
    old = block.first_op
    # Parse a real result-producing instruction, rather than assigning a value
    # to the interpreter's state or trusting a test-only opcode.
    module = _parse_module(f"module {{ llvm.func @g(%input: !llvm.ptr) {{ %x = {origin}\n llvm.return }} }}")
    replacement = _functions(module)["g"].body.blocks.first.first_op
    replacement.detach()
    if origin.startswith("llvm.load"):
        replacement.operands = (f.body.blocks.first.args[0],)
    block.insert_op_before(replacement, old)
    old.results[0].replace_all_uses_with(replacement.results[0])
    block.erase_op(old)
    with pytest.raises(ValueError, match="tensor data|unknown|opaque"):
        CC.constant_control_counts(f)


def test_escaping_loop_value_disables_summary():
    from xdsl.dialects import llvm

    f = function()
    blocks = list(f.body.blocks)
    # The loop-header argument dominates its exit, but may not escape a
    # count-only region because it is deliberately cleared after the summary.
    blocks[8].insert_op_before(llvm.AddOp(blocks[3].args[0], blocks[0].first_op.results[0]), blocks[8].first_op)
    assert control_regions(f) == {}
    with pytest.raises(ValueError, match="block budget"):
        CC.constant_control_counts(f, max_blocks=256)


def test_control_region_cannot_be_used_as_memory_transport_permission():
    region = next(iter(control_regions(function()).values()))
    with pytest.raises(ValueError, match="cannot grant a byte-copy"):
        execute(region, {}, (0,), None, {}, set())


def test_early_exit_and_flagged_induction_keep_their_original_refusals():
    f = function()
    blocks = list(f.body.blocks)
    blocks[6].last_op.successors = (blocks[-1],)
    assert control_regions(f) == {}
    # An unknown no-wrap/exact assertion on the induction is not a loop bound.
    f = function()
    update = list(f.body.blocks)[7].first_op
    update.properties["isExact"] = UnitAttr()
    with pytest.raises(ValueError, match="poison flags"):
        CC.constant_control_counts(f)


def test_wrapping_induction_does_not_gain_a_finite_summary():
    f = _functions(_parse_module("module {" + str(function(1, 255, 2)).replace("i64", "i8") + "}"))["padding"]
    assert control_regions(f) == {}
    with pytest.raises(ValueError, match="block budget"):
        CC.constant_control_counts(f, max_blocks=1000)


def test_unsigned_induction_entering_signed_half_retains_scalar_width_semantics():
    text = str(function(1, 200)).replace("i64", "i8").replace('icmp "ult" %x', 'icmp "slt" %x')
    f = _functions(_parse_module("module {" + text + "}"))["padding"]
    assert control_regions(f) == {}
    counts = CC.constant_control_counts(f)
    assert counts == CC.constant_control_counts(f, summarize_loops=False)
    assert list(counts.values())[5] == 73  # zero plus the 72 negative i8 values.


def test_flagged_control_arithmetic_and_wrapping_induction_still_refuse():
    f = _functions(_parse_module("module {" + str(function(127, 127)).replace("i64", "i8") + "}"))["padding"]
    for op in f.walk():
        if op.name == "llvm.add" and op.results[0].name_hint == "x":
            op.properties["overflowFlags"] = IntegerAttr(1, 32)
    # The branch expression eventually violates signed no-wrap at i8 width.
    with pytest.raises(ValueError, match="poison|no-wrap"):
        CC.constant_control_counts(f)


@pytest.mark.parametrize("width", [8, 16, 32, 64])
@pytest.mark.parametrize("name", ["udiv", "urem", "sdiv", "srem"])
def test_vector_division_matches_scalar_integer_width_and_signedness(width, name):
    f = _functions(
        _parse_module(f"""module {{ llvm.func @f(%a: i{width}, %b: i{width}) {{
      %v = llvm.{name} %a, %b : i{width}
      llvm.return
    }} }}""")
    )["f"]
    op = f.body.blocks.first.first_op
    minimum = -(1 << (width - 1))
    a = np.array([minimum, -7, -1, 0, 7, (1 << (width - 1)) - 1], dtype=np.int64)
    b = np.array([1, 3, 2, -3, -3, 2], dtype=np.int64)
    expected = [CC.integer_operation(op, [int(x), int(y)]) for x, y in zip(a, b, strict=True)]
    assert integer(op, [a, b], np.ones(a.shape, bool)).tolist() == expected


@pytest.mark.parametrize("name", ["udiv", "urem", "sdiv", "srem"])
def test_vector_division_zero_overflow_and_unproved_flags_refuse(name):
    f = _functions(
        _parse_module(f"""module {{ llvm.func @f(%a: i64, %b: i64) {{
      %v = llvm.{name} %a, %b : i64
      llvm.return
    }} }}""")
    )["f"]
    op = f.body.blocks.first.first_op
    with pytest.raises(ValueError, match="division by zero"):
        integer(op, [np.array([1]), np.array([0])], np.array([True]))
    assert integer(op, [np.array([1, 3]), np.array([0, 2])], np.array([False, True]))[1] == 1
    if name.startswith("s"):
        with pytest.raises(ValueError, match="division overflow"):
            integer(op, [np.array([-(1 << 63)]), np.array([-1])], np.array([True]))
    op.properties["isExact"] = UnitAttr()
    with pytest.raises(ValueError, match="poison flags"):
        integer(op, [np.array([4]), np.array([2])], np.array([True]))


def test_divisor_zero_is_checked_at_the_declared_integer_width():
    f = _functions(
        _parse_module("""module { llvm.func @f(%a: i8, %b: i8) {
      %v = llvm.udiv %a, %b : i8
      llvm.return
    } }""")
    )["f"]
    with pytest.raises(ValueError, match="division by zero"):
        integer(f.body.blocks.first.first_op, [np.array([1]), np.array([256])], np.array([True]))


def nested_function(extents):
    """Opaque-command padding diamond inside independently bounded loop domains."""
    text = [
        "module { llvm.func @nested() {",
        "%zero = llvm.mlir.constant(0 : i64) : i64",
        "%one = llvm.mlir.constant(1 : i64) : i64",
    ]
    text += [f"%bound{i} = llvm.mlir.constant({bound} : i64) : i64" for i, bound in enumerate(extents)]
    text += ["llvm.br ^h0(%zero : i64)"]
    last = len(extents) - 1
    for i in range(len(extents)):
        text += [
            f"^h{i}(%i{i}: i64):",
            f'%more{i} = llvm.icmp "ult" %i{i}, %bound{i} : i64',
            f"llvm.cond_br %more{i}, ^body{i}, ^exit{i}",
            f"^body{i}:",
        ]
        if i != last:
            text += [f"llvm.br ^h{i + 1}(%zero : i64)"]
    text += [
        f"%x = llvm.add %i0, %i{last} : i64",
        '%inside = llvm.icmp "ult" %x, %bound0 : i64',
        "llvm.cond_br %inside, ^valid, ^pad",
        "^valid:",
        'llvm.inline_asm has_side_effects asm_dialect = att "fence", "~{memory}" : () -> ()',
        "llvm.br ^join",
        "^pad:",
        'llvm.inline_asm has_side_effects asm_dialect = att "fence", "~{memory}" : () -> ()',
        "llvm.br ^join",
        "^join:",
        f"%next{last} = llvm.add %i{last}, %one : i64",
        f"llvm.br ^h{last}(%next{last} : i64)",
    ]
    for i in reversed(range(len(extents))):
        text += [f"^exit{i}:"]
        if i:
            text += [f"%next{i - 1} = llvm.add %i{i - 1}, %one : i64", f"llvm.br ^h{i - 1}(%next{i - 1} : i64)"]
        else:
            text += ["llvm.return"]
    return _functions(_parse_module("\n".join([*text, "} }"])))["nested"]


@pytest.mark.parametrize("extents", [(3, 4, 2), (2, 3, 4, 2), (0, 3, 2), (3, 0, 2), (3, 2, 0)])
def test_nested_pretested_counts_equal_independent_scalar_with_final_false_masks(extents):
    f = nested_function(extents)
    region = control_regions(f)[list(f.body.blocks)[1]]
    assert len(region[1]) == len(extents)
    assert [loop.trips for loop in region[1]] == [extent + 1 for extent in extents]
    assert CC.constant_control_counts(f) == CC.constant_control_counts(f, summarize_loops=False)
    with pytest.raises(ValueError, match="cannot grant a byte-copy"):
        execute(region, {}, (0,), None, {}, set())


def test_nested_selection_bounds_all_external_keys_and_reuses_only_matching_control(monkeypatch):
    from merlin.verify import input_copy_batch as IB
    from merlin.verify.input_copy_loops import region_cost

    f = nested_function((5, 7, 3, 2))
    headers = list(f.body.blocks)[1:8:2]
    regions = control_regions(f)
    budget = region_cost(regions[headers[1]]) * 3
    monkeypatch.setattr(CC, "MAX_OPERATIONS", budget)
    regions = control_regions(f)
    assert headers[0] not in regions and headers[1] not in regions
    assert headers[2] in regions
    original = IB.execute
    calls = []

    def observed(*args, **kwargs):
        calls.append(args[0][0].header)
        return original(*args, **kwargs)

    monkeypatch.setattr(IB, "execute", observed)
    result = CC.constant_control_counts(f, max_operations=budget)
    assert result == CC.constant_control_counts(f, summarize_loops=False)
    assert calls == [headers[2]] * 5  # Five outer IVs, reused across seven middle IVs.
    assert list(result.values())[-5] > 0  # Both padding arms remain observed.


def test_unproved_nested_backedge_cannot_be_hidden_inside_an_outer_domain():
    f = nested_function((3, 4, 2))
    blocks = list(f.body.blocks)
    outer, middle = blocks[1], blocks[3]
    # The scalar trace can evaluate this varying bound, but it is not an
    # independently constant rectangular loop, so enclosing summaries refuse.
    middle.first_op.operands = (middle.args[0], outer.args[0])
    regions = control_regions(f)
    assert outer not in regions and middle not in regions
    assert CC.constant_control_counts(f) == CC.constant_control_counts(f, summarize_loops=False)


def test_unbound_external_control_never_gets_a_cache_domain():
    f = _functions(_parse_module("module {" + str(function()).replace("!llvm.ptr", "i64") + "}"))["padding"]
    outer, inner = list(f.body.blocks)[1], list(f.body.blocks)[3]
    # Give the padding branch an external unknown argument instead of the
    # independently bounded enclosing IV. Even a finite region has no keys.
    outer.args[0].replace_uses_with_if(
        f.body.blocks.first.args[0], lambda use: use.operation.parent_block() is list(f.body.blocks)[4]
    )
    assert inner not in control_regions(f)
    with pytest.raises(ValueError, match="unknown|opaque|tensor"):
        CC.constant_control_counts(f)


def test_nested_counting_does_not_exempt_host_value_arithmetic():
    from merlin.verify.host_compute_audit import GroupSite, audit_group

    f = nested_function((3, 4, 2))
    valid = next(block for block in f.body.blocks if block.first_op.name == "llvm.inline_asm")
    arithmetic = _functions(
        _parse_module("""module { llvm.func @value() {
      %one = llvm.mlir.constant(1.0 : f32) : f32
      %sum = llvm.fadd %one, %one : f32
      llvm.return
    } }""")
    )["value"]
    for op in list(arithmetic.body.blocks.first.ops)[:-1]:
        op.detach()
        valid.insert_op_before(op, valid.last_op)
    row = audit_group(GroupSite(0, "accelerator", "nested", 1, 4), f)
    assert row["verdict"] == "host_compute_in_accelerator_group"
    assert row["host_value_arithmetic"] == CC.constant_control_counts(f)[valid]


def test_key_limit_and_lane_and_live_operation_limits_remain_finite():
    f = nested_function((5000, 1, 1))
    assert list(f.body.blocks)[3] not in control_regions(f)  # More than 4096 distinct outer keys.
    f = nested_function((1, 513, 513))
    region = control_regions(f)[list(f.body.blocks)[1]]
    with pytest.raises(ValueError, match="lane budget"):
        execute(region, {}, (0,), None, {}, set(), control_only=True)
    with pytest.raises(ValueError, match="operation budget"):
        CC.constant_control_counts(nested_function((2, 3, 2)), max_operations=1)
