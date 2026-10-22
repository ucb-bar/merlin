import pytest
from xdsl.dialects.builtin import IntegerAttr, UnitAttr

from merlin.perf.constant_control_cfg import constant_control_counts
from merlin.targetgen.rocc.decode import _parse_module
from merlin.verify.host_compute_audit import _functions


def function(body, args=""):
    return _functions(_parse_module(f"builtin.module {{ llvm.func @f({args}) {{ {body} }} }}"))["f"]


def test_constant_control_counts_variable_step_and_width_signedness():
    f = function("""
      %zero = llvm.mlir.constant(0) : i8
      %one = llvm.mlir.constant(1) : i8
      %three = llvm.mlir.constant(3) : i8
      %five = llvm.mlir.constant(5) : i8
      llvm.br ^loop(%zero : i8)
    ^loop(%i: i8):
      %small = llvm.icmp "ult" %i, %three : i8
      %step = llvm.select %small, %three, %one : i1, i8
      %next = llvm.add %i, %step : i8
      %again = llvm.icmp "slt" %next, %five : i8
      llvm.cond_br %again, ^loop(%next : i8), ^done
    ^done:
      llvm.return
    """)
    assert list(constant_control_counts(f).values()) == [1, 3, 1]
    f = function("""
      %max = llvm.mlir.constant(127) : i8
      %one = llvm.mlir.constant(1) : i8
      %next = llvm.add %max, %one : i8
      %negative = llvm.icmp "slt" %next, %one : i8
      llvm.cond_br %negative, ^yes, ^no
    ^yes:
      llvm.return
    ^no:
      llvm.return
    """)
    assert list(constant_control_counts(f).values()) == [1, 1, 0]


def test_constant_control_refuses_data_dependent_control_poison_and_budget():
    data = function(
        """
      %value = llvm.load %input : !llvm.ptr -> i1
      llvm.cond_br %value, ^yes, ^no
    ^yes:
      llvm.return
    ^no:
      llvm.return
    """,
        "%input: !llvm.ptr",
    )
    with pytest.raises(ValueError, match="tensor data"):
        constant_control_counts(data)
    simple = function("""
      %a = llvm.mlir.constant(127) : i8
      %b = llvm.add %a, %a : i8
      llvm.return
    """)
    with pytest.raises(ValueError, match="operation budget"):
        constant_control_counts(simple, max_operations=1)
    add = next(op for op in simple.walk() if op.name == "llvm.add")
    add.properties["overflowFlags"] = IntegerAttr(1, 32)
    with pytest.raises(ValueError, match="poison flags"):
        constant_control_counts(simple)


def _tiled(rows, tile, *, inner_blocks=1, inner_extra="", step="1", width="i64"):
    """Outer tiles over ``rows``; an inner straight-line loop runs ``min(tile, rows - i)`` trips."""
    chain = []
    for index in range(inner_blocks):
        target = "^inner" if index == inner_blocks - 1 else f"^body{index + 1}"
        args = "(%next : " + width + ")" if index == inner_blocks - 1 else ""
        update = f"%next = llvm.add %j, %step : {width}\n" if index == inner_blocks - 1 else ""
        chain.append(f"""
    ^body{index}:
      %v{index} = llvm.add %i, %j : {width}
      {inner_extra}
      {update}llvm.br {target}{args}""")
    return f"""
      %zero = llvm.mlir.constant(0) : {width}
      %rows = llvm.mlir.constant({rows}) : {width}
      %tile = llvm.mlir.constant({tile}) : {width}
      %step = llvm.mlir.constant({step}) : {width}
      llvm.br ^outer(%zero : {width})
    ^outer(%i: {width}):
      %more = llvm.icmp "slt" %i, %rows : {width}
      llvm.cond_br %more, ^tile, ^done
    ^tile:
      %left = llvm.sub %rows, %i : {width}
      %short = llvm.icmp "ult" %left, %tile : {width}
      %count = llvm.select %short, %left, %tile : i1, {width}
      llvm.br ^inner(%zero : {width})
    ^inner(%j: {width}):
      %again = llvm.icmp "ult" %j, %count : {width}
      llvm.cond_br %again, ^body0, ^latch
    {"".join(chain)}
    ^latch:
      %following = llvm.add %i, %tile : {width}
      llvm.br ^outer(%following : {width})
    ^done:
      llvm.return
    """


@pytest.mark.parametrize(
    ("rows", "tile", "inner_blocks", "step"),
    [(37, 16, 1, "1"), (32, 16, 2, "1"), (5, 16, 1, "3"), (0, 16, 1, "1"), (40, 8, 3, "2")],
)
def test_summarized_remainder_loops_match_the_exact_stepped_trace(rows, tile, inner_blocks, step):
    f = function(_tiled(rows, tile, inner_blocks=inner_blocks, step=step))
    summarized = constant_control_counts(f)
    stepped = constant_control_counts(f, summarize_loops=False)
    assert summarized == stepped
    blocks = list(f.body.blocks)
    expected_body = sum(-(-min(tile, rows - i) // int(step)) for i in range(0, rows, tile))
    assert all(summarized[blocks[4 + k]] == expected_body for k in range(inner_blocks))


def test_summarized_loops_keep_the_block_budget_for_remaining_control():
    # 64 tiles of a 100000-trip inner loop: stepping needs about 13M block visits.
    f = function(_tiled(64 * 100000, 100000))
    counts = constant_control_counts(f, max_blocks=1000)
    assert counts[list(f.body.blocks)[4]] == 64 * 100000
    with pytest.raises(ValueError, match="block budget"):
        constant_control_counts(f, max_blocks=1000, summarize_loops=False)


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        # A body branch, a carried accumulator and an unsigned wrap are stepped exactly.
        ('%odd = llvm.icmp "eq" %v0, %zero : i64', None),
        (None, "carried"),
        (None, "wrap"),
    ],
)
def test_unsummarizable_loops_are_still_stepped_exactly(change, reason):
    if reason == "carried":
        f = function("""
          %zero = llvm.mlir.constant(0) : i64
          %one = llvm.mlir.constant(1) : i64
          %five = llvm.mlir.constant(5) : i64
          llvm.br ^loop(%zero, %zero : i64, i64)
        ^loop(%i: i64, %sum: i64):
          %again = llvm.icmp "ult" %i, %five : i64
          llvm.cond_br %again, ^body, ^done
        ^body:
          %total = llvm.add %sum, %i : i64
          %next = llvm.add %i, %one : i64
          llvm.br ^loop(%next, %total : i64, i64)
        ^done:
          llvm.return
        """)
    elif reason == "wrap":
        # i8 unsigned bound 255 with step 2 from 0: the update after 254 wraps to 0 and keeps looping.
        f = function("""
          %zero = llvm.mlir.constant(0) : i8
          %two = llvm.mlir.constant(2) : i8
          %bound = llvm.mlir.constant(-1) : i8
          llvm.br ^loop(%zero : i8)
        ^loop(%i: i8):
          %again = llvm.icmp "ult" %i, %bound : i8
          llvm.cond_br %again, ^body, ^done
        ^body:
          %next = llvm.add %i, %two : i8
          llvm.br ^loop(%next : i8)
        ^done:
          llvm.return
        """)
        with pytest.raises(ValueError, match="block budget"):
            constant_control_counts(f, max_blocks=10_000)
        return
    else:
        body = _tiled(37, 16, inner_extra=change)
        body = body.replace(
            "llvm.br ^inner(%next : i64)", "llvm.cond_br %odd, ^inner(%next : i64), ^inner(%next : i64)"
        )
        f = function(body)
    assert constant_control_counts(f) == constant_control_counts(f, summarize_loops=False)


def test_summarized_induction_update_keeps_the_poison_refusal():
    # The last update from 126 to 128 violates nsw at i8 width. A loop summary
    # must not hide the poison, even though its body contains no branches.
    f = function(_tiled(127, 127, step="2", width="i8"))
    update = next(op for op in f.walk() if op.name == "llvm.add" and op.results[0].name_hint == "next")
    update.properties["overflowFlags"] = IntegerAttr(1, 32)
    with pytest.raises(ValueError, match="poison flags"):
        constant_control_counts(f)


@pytest.mark.parametrize(
    ("name", "a", "b", "flags", "clean"),
    [
        ("add", 120, 7, 3, True),
        ("add", 127, 1, 1, False),
        ("add", 127, 1, 2, True),
        ("add", -1, 1, 1, True),
        ("add", -1, 1, 2, False),
        ("sub", 3, 2, 3, True),
        ("sub", 0, 1, 1, True),
        ("sub", 0, 1, 2, False),
        ("sub", -128, 1, 1, False),
        ("mul", 12, 10, 3, True),
        ("mul", 16, 8, 1, False),
        ("mul", 16, 16, 2, False),
        ("mul", -3, 2, 1, True),
        ("shl", 7, 4, 3, True),
        ("shl", 8, 4, 1, False),
        ("shl", 16, 4, 2, False),
        ("shl", -2, 2, 1, True),
        ("shl", 1, 8, 3, False),
        ("add", 1, 1, 4, False),
        ("and", 1, 1, 1, False),
    ],
)
def test_concrete_no_wrap_flags_are_proven_before_width_reduction(name, a, b, flags, clean):
    f = function(f"""
      %a = llvm.mlir.constant({a}) : i8
      %b = llvm.mlir.constant({b}) : i8
      %result = llvm.{name} %a, %b : i8
      llvm.return
    """)
    op = next(op for op in f.walk() if op.name == f"llvm.{name}")
    op.properties["overflowFlags"] = IntegerAttr(flags, 32)
    if clean:
        assert list(constant_control_counts(f).values()) == [1]
    else:
        with pytest.raises(ValueError, match="poison"):
            constant_control_counts(f)


@pytest.mark.parametrize(("name", "key"), [("udiv", "isExact")])
def test_unmodeled_poison_assertions_still_refuse(name, key):
    f = function(f"""
      %a = llvm.mlir.constant(4) : i8
      %b = llvm.mlir.constant(2) : i8
      %result = llvm.{name} %a, %b : i8
      llvm.return
    """)
    op = next(op for op in f.walk() if op.name == f"llvm.{name}")
    op.properties[key] = UnitAttr()
    with pytest.raises(ValueError, match="unproved poison flags"):
        constant_control_counts(f)


@pytest.mark.parametrize("rhs,clean", [(2, True), (4, False)])
def test_disjoint_or_assertion_is_checked_against_original_constant_bits(rhs, clean):
    f = function(f"""
      %a = llvm.mlir.constant(4) : i8
      %b = llvm.mlir.constant({rhs}) : i8
      %result = llvm.or %a, %b : i8
      llvm.return
    """)
    op = next(op for op in f.walk() if op.name == "llvm.or")
    op.properties["isDisjoint"] = UnitAttr()
    if clean:
        assert list(constant_control_counts(f).values()) == [1]
    else:
        with pytest.raises(ValueError, match="disjoint-or poison"):
            constant_control_counts(f)


def test_flagged_posttested_exit_on_equality_is_counted_exactly():
    f = function("""
      %zero = llvm.mlir.constant(0) : i8
      %one = llvm.mlir.constant(1) : i8
      %bound = llvm.mlir.constant(7) : i8
      llvm.br ^loop(%zero : i8)
    ^loop(%i: i8):
      %next = llvm.add %i, %one overflow<nsw, nuw> : i8
      %done = llvm.icmp "eq" %next, %bound : i8
      llvm.cond_br %done, ^exit, ^loop(%next : i8)
    ^exit:
      llvm.return
    """)
    assert list(constant_control_counts(f).values()) == [1, 7, 1]


def test_inner_loop_bound_from_tensor_data_stays_unknown():
    body = _tiled(37, 16).replace(
        "%count = llvm.select %short, %left, %tile : i1, i64",
        "%count = llvm.load %input : !llvm.ptr -> i64",
    )
    f = function(body, "%input: !llvm.ptr")
    with pytest.raises(ValueError, match="tensor data"):
        constant_control_counts(f)


def _repeated_branch_loop():
    return """
      %zero = llvm.mlir.constant(0) : i64
      %one = llvm.mlir.constant(1) : i64
      %outerbound = llvm.mlir.constant(5) : i64
      %bound = llvm.mlir.constant(256) : i64
      %split = llvm.mlir.constant(127) : i64
      llvm.br ^outer(%zero : i64)
    ^outer(%iteration: i64):
      "llvm.inline_asm"() <{asm_string = "", constraints = "", has_side_effects}> : () -> ()
      %parity = llvm.and %iteration, %one : i64
      %threshold = llvm.mul %parity, %split : i64
      llvm.br ^inner(%zero : i64)
    ^inner(%j: i64):
      %choose = llvm.icmp "ult" %j, %threshold : i64
      llvm.cond_br %choose, ^left, ^right
    ^left:
      llvm.br ^latch
    ^right:
      llvm.br ^latch
    ^latch:
      %next = llvm.add %j, %one : i64
      %last = llvm.icmp "eq" %next, %bound : i64
      llvm.cond_br %last, ^outer_latch, ^inner(%next : i64)
    ^outer_latch:
      %nextouter = llvm.add %iteration, %one : i64
      %done = llvm.icmp "eq" %nextouter, %outerbound : i64
      llvm.cond_br %done, ^exit, ^outer(%nextouter : i64)
    ^exit:
      llvm.return
    """


def test_repeated_batch_counts_bind_all_external_control_values_within_one_invocation():
    f = function(_repeated_branch_loop())
    expected = constant_control_counts(f, summarize_loops=False)
    assert constant_control_counts(f, max_operations=4000) == expected
    assert list(expected.values())[3:5] == [254, 1026]
    # Reopening changed source must compute a fresh witness.
    changed = function(_repeated_branch_loop().replace('constant(127)', 'constant(53)'))
    actual = constant_control_counts(changed, max_operations=4000)
    assert actual == constant_control_counts(changed, summarize_loops=False)
    assert list(actual.values())[3:5] == [106, 1174]


def test_batched_control_observation_keeps_load_results_unknown_even_inside_a_control_cone():
    body = _repeated_branch_loop().replace(
        '%choose = llvm.icmp "ult" %j, %threshold : i64',
        '%datum = llvm.load %input : !llvm.ptr -> i64\n %choose = llvm.icmp "ult" %j, %datum : i64',
    )
    with pytest.raises(ValueError, match="tensor data"):
        constant_control_counts(function(body, "%input: !llvm.ptr"))


@pytest.mark.parametrize("kind,expected", [("umin", 1), ("umax", -1), ("smin", -1), ("smax", 1)])
def test_integer_minimum_maximum_obeys_source_signedness(kind, expected):
    from merlin.perf.constant_control_cfg import integer_operation

    f = function(f"""
      %a = llvm.mlir.constant(-1) : i8
      %b = llvm.mlir.constant(1) : i8
      %chosen = "llvm.intr.{kind}"(%a, %b) : (i8, i8) -> i8
      llvm.return
    """)
    operation = next(op for op in f.walk() if op.name == "builtin.unregistered")
    assert integer_operation(operation, [-1, 1]) == expected
    assert integer_operation(operation, [None, 1]) is None


@pytest.mark.parametrize("value,flags,clean", [
    (127, 3, True), (128, 1, False), (128, 2, True), (256, 2, False),
    (-128, 1, True), (-129, 1, False), (-1, 2, False),
])
def test_truncation_flags_preserve_the_original_signed_or_unsigned_value(value, flags, clean):
    from merlin.perf.constant_control_cfg import integer_operation

    f = function("""%a = llvm.mlir.constant(1) : i32
      %result = llvm.trunc %a : i32 to i8
      llvm.return""")
    operation = next(op for op in f.walk() if op.name == "llvm.trunc")
    operation.properties["overflowFlags"] = IntegerAttr(flags, 32)
    if clean:
        result = value % 256
        assert integer_operation(operation, [value]) == result - (256 if result >= 128 else 0)
    else:
        with pytest.raises(ValueError, match="poison flags"):
            integer_operation(operation, [value])


def test_nonnegative_extension_assertion_is_proved_before_conversion():
    from merlin.perf.constant_control_cfg import integer_operation

    f = function("""%a = llvm.mlir.constant(1) : i8
      %result = llvm.zext %a : i8 to i32
      llvm.return""")
    operation = next(op for op in f.walk() if op.name == "llvm.zext")
    operation.properties["nonNeg"] = UnitAttr()
    assert integer_operation(operation, [127]) == 127
    with pytest.raises(ValueError, match="nonnegative poison"):
        integer_operation(operation, [-1])


def _closed_host_nest(outer=1000, inner=1000, *, inside='', escape=False):
    return function(f'''
      %zero = llvm.mlir.constant(0 : i64) : i64
      %one = llvm.mlir.constant(1 : i64) : i64
      %outer = llvm.mlir.constant({outer} : i64) : i64
      %inner = llvm.mlir.constant({inner} : i64) : i64
      llvm.br ^outer(%zero : i64)
    ^outer(%i: i64):
      %more = llvm.icmp "slt" %i, %outer : i64
      llvm.cond_br %more, ^enter, ^done
    ^enter:
      llvm.br ^inner(%zero : i64)
    ^inner(%j: i64):
      %again = llvm.icmp "slt" %j, %inner : i64
      llvm.cond_br %again, ^body, ^latch
    ^body:
      {inside}
      %next = llvm.add %j, %one : i64
      llvm.br ^inner(%next : i64)
    ^latch:
      %following = llvm.add %i, %one : i64
      llvm.br ^outer(%following : i64)
    ^done:
      {'%escaped = llvm.add %i, %one : i64' if escape else ''}
      llvm.inline_asm has_side_effects "fence", "~{{memory}}" : () -> ()
      llvm.return
    ''')


def test_watched_operand_visit_summarizes_only_closed_host_rectangles():
    f = _closed_host_nest()
    watched = frozenset(op for op in f.walk() if op.name == 'llvm.inline_asm')
    observations = []
    counts = constant_control_counts(f, max_blocks=100, max_operations=200,
        observe=lambda op, args: observations.append((op, args)), observed_operations=watched)
    assert observations == [(next(iter(watched)), ())]
    assert counts[list(f.body.blocks)[4]] == 1_000_000
    # The caller must explicitly declare its watched set before any observation
    # can be skipped; the legacy all-operation callback still steps every op.
    with pytest.raises(ValueError, match='budget'):
        constant_control_counts(f, max_blocks=100, max_operations=200, observe=lambda *_: None)


def test_observed_commands_inside_nested_loops_are_never_summarized():
    f = _closed_host_nest(4, 3, inside='llvm.inline_asm has_side_effects "fence", "~{memory}" : () -> ()')
    watched = frozenset(op for op in f.walk() if op.name == 'llvm.inline_asm')
    rows = []
    constant_control_counts(f, observe=lambda op, args: rows.append(op), observed_operations=watched)
    assert len(rows) == 13  # Twelve loop visits plus the final fence.
    with pytest.raises(ValueError, match='budget'):
        constant_control_counts(f, max_blocks=10, observe=lambda *_: None, observed_operations=watched)


def test_observation_summaries_refuse_escaping_ssa_and_unknown_loop_bounds():
    for f in (_closed_host_nest(escape=True),):
        watched = frozenset(op for op in f.walk() if op.name == 'llvm.inline_asm')
        with pytest.raises(ValueError, match='budget'):
            constant_control_counts(f, max_blocks=100, max_operations=200,
                                    observe=lambda *_: None, observed_operations=watched)
    f = function('''
      %zero = llvm.mlir.constant(0 : i64) : i64
      %one = llvm.mlir.constant(1 : i64) : i64
      %bound = llvm.load %input : !llvm.ptr -> i64
      llvm.br ^loop(%zero : i64)
    ^loop(%i: i64):
      %more = llvm.icmp "slt" %i, %bound : i64
      llvm.cond_br %more, ^body, ^done
    ^body:
      %next = llvm.add %i, %one : i64
      llvm.br ^loop(%next : i64)
    ^done:
      llvm.inline_asm has_side_effects "fence", "~{memory}" : () -> ()
      llvm.return
    ''', '%input: !llvm.ptr')
    watched = frozenset(op for op in f.walk() if op.name == 'llvm.inline_asm')
    with pytest.raises(ValueError, match='tensor data'):
        constant_control_counts(f, observe=lambda *_: None, observed_operations=watched)


def test_host_summary_does_not_hide_poison_before_observed_command():
    f = _closed_host_nest(2, 2, inside='%bad = llvm.shl %one, %inner : i64')
    shift = next(op for op in f.walk() if op.name == 'llvm.shl')
    shift.properties['overflowFlags'] = IntegerAttr(1, 32)
    # The flagged region must be visited even though the observed command lies
    # after it. Force poison using the selected source width, not a mode label.
    bound = next(op for op in f.walk() if op.name == 'llvm.mlir.constant' and op.value.value.data == 2)
    shift.operands = (shift.operands[0], bound.results[0])
    bound.properties['value'] = IntegerAttr(64, 64)
    watched = frozenset(op for op in f.walk() if op.name == 'llvm.inline_asm')
    with pytest.raises(ValueError, match='poison'):
        constant_control_counts(f, observe=lambda *_: None, observed_operations=watched)
