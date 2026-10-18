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


@pytest.mark.parametrize(("name", "key"), [("udiv", "isExact"), ("or", "isDisjoint")])
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
