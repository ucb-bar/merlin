"""Public depth selection preserves actual ancestors and has bounded allocation."""

import pytest
from xdsl.dialects import scf

from merlin.transforms.certificates import parse_ir
from merlin.transforms.normalization import constant_integer
from merlin.verify import compiler_transforms as transforms
from merlin.verify import pass_validation

SOURCE = """builtin.module {
func.func @forward(%seed:i32, %stop:index) -> i32 {
  %zero = arith.constant 0 : index
  %five = arith.constant 5 : index
  %one = arith.constant 1 : index
  %word = arith.constant 1 : i32
  %result = scf.for %outer = %zero to %stop step %one iter_args(%carry=%seed) -> (i32) {
    %inside = scf.for %inner = %zero to %five step %one iter_args(%value=%carry) -> (i32) {
      %next = arith.addi %value, %word : i32
      scf.yield %next : i32
    }
    scf.yield %inside : i32
  }
  func.return %result : i32
}
}"""


@pytest.fixture(autouse=True)
def clean_selection(monkeypatch):
    for key in (
        transforms.ENV_UNROLL,
        transforms.ENV_TILE,
        transforms.ENV_UNROLL_DEPTH,
        transforms.ENV_TILE_DEPTH,
        transforms.ENV_ALLOCATOR,
        transforms.ENV_MACC,
        pass_validation.ENV_PASSES,
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize("kind", ["unroll", "tile"])
def test_depth_two_keeps_dynamic_outer_loop_and_changes_its_inner_body(tmp_path, kind):
    options = (
        transforms.Options(unroll_factor=2, unroll_depth=2)
        if kind == "unroll"
        else transforms.Options(tile_size=3, tile_depth=2)
    )
    actual, report = transforms.apply(SOURCE, tmp_path, options=options)
    loops = [op for op in parse_ir(actual).walk() if type(op) is scf.ForOp]
    assert len(loops) == 3
    outer = loops[0]
    assert constant_integer(outer.lb) == 0
    assert constant_integer(outer.step) == 1
    assert outer.ub is outer.parent_block().args[1]
    assert report["changed"] == [transforms.UNROLL if kind == "unroll" else transforms.TILE]
    assert report[kind + "_depth"] == 2
    assert report["selected_checks"] == []
    if kind == "unroll":
        assert all(loop.parent_op() is outer for loop in loops[1:])
        assert [constant_integer(loop.step) for loop in loops[1:]] == [2, 1]
    else:
        assert loops[1].parent_op() is outer and loops[2].parent_op() is loops[1]
        assert [constant_integer(loop.step) for loop in loops[1:]] == [3, 1]


@pytest.mark.parametrize("kind", ["unroll", "tile"])
def test_absent_huge_depth_is_identity_without_depth_sized_allocation(tmp_path, kind):
    options = (
        transforms.Options(unroll_factor=2, unroll_depth=10**100)
        if kind == "unroll"
        else transforms.Options(tile_size=3, tile_depth=10**100)
    )
    actual, report = transforms.apply(SOURCE, tmp_path, options=options)
    assert parse_ir(actual).is_structurally_equivalent(parse_ir(SOURCE))
    assert report["changed"] == []
    assert report["unchanged"] == [transforms.UNROLL if kind == "unroll" else transforms.TILE]


@pytest.mark.parametrize(
    "options",
    [
        {"unroll_factor": 2, "unroll_depth": True},
        {"unroll_factor": 2, "unroll_depth": 0},
        {"tile_size": 3, "tile_depth": -1},
        {"tile_size": 3, "tile_depth": 1.5},
        {"unroll_depth": 2},
        {"tile_depth": 2},
        {"buffer_allocator": transforms.ALLOCATOR, "tile_size": 3, "tile_depth": 2},
    ],
)
def test_invalid_or_unselected_depth_refuses_before_any_producer(options):
    with pytest.raises(ValueError, match="depth"):
        transforms.Options(**options)


def test_cli_depth_selection_reaches_typed_options(monkeypatch):
    from merlin.compile.command import apply_pass_validation, parser

    args = parser(("int8",), ("none",), "none", ("k1",)).parse_args(
        [
            "--workload",
            "w",
            "--loop-unroll-factor",
            "2",
            "--loop-unroll-depth",
            "2",
            "--loop-tile-size",
            "3",
            "--loop-tile-depth",
            "3",
        ]
    )
    apply_pass_validation(args)
    assert transforms.from_environment() == transforms.Options(
        unroll_factor=2,
        unroll_depth=2,
        tile_size=3,
        tile_depth=3,
    )


@pytest.mark.parametrize("option", ["--loop-unroll-depth", "--loop-tile-depth"])
@pytest.mark.parametrize("value", ["0", "-1", "2.5"])
def test_cli_rejects_invalid_depths(option, value):
    from merlin.compile.command import parser

    with pytest.raises(SystemExit):
        parser(("int8",), ("none",), "none", ("k1",)).parse_args(["--workload", "w", option, value])
