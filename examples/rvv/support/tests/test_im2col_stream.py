"""Closed admission, explicit geometry, and actual window-panel IR."""

from pathlib import Path

import pytest
from im2col_fixture import _module
from merlin.common import mlir_query as mq
from test_backend import BACKEND

IP = BACKEND.im2col_stream_api()


def target(vlen=128, **kw):
    args = {
        "vlen_bits": vlen,
        "elen_bits": 64,
        "operand_bits": 8,
        "accumulator_bits": 32,
        "operand_lmul": 1,
        "workspace_bytes": 65536,
    }
    return IP.PanelTarget(**(args | kw))


@pytest.mark.parametrize("vlen", [128, 256, 512, 1024])
def test_geometry_uses_vlen_for_registers_and_independent_workspace_for_panel(vlen):
    g = IP.derive(target(vlen), k=144, output_width=1027)
    assert g.nr == vlen // 8
    assert g.panel_width % g.nr == 0
    assert g.workspace_bytes <= 65536
    assert g.source.accumulator_lmul == 4
    assert g.mr * 4 + 2 * 4 + 1 + g.source.reserved_registers <= 32
    assert g.to_dict()["dlen_bits"] is None
    assert g.to_dict()["physical_lane_utilization"] is None
    tight = IP.derive(target(vlen, workspace_bytes=400), k=144, output_width=1027)
    assert tight.panel_width == 1  # budget wins over register capacity


def test_explicit_cache_budget_limits_workspace_without_inventing_cache_capacity():
    g = IP.derive(target(512, cache_budget_bytes=4096), k=144, output_width=1024)
    assert g.workspace_bytes <= 4096
    assert g.panel_width < 512 // 8


@pytest.mark.parametrize(
    "kw",
    [
        {"vlen_bits": 192},
        {"elen_bits": 16},
        {"operand_lmul": 4},
        {"workspace_bytes": 0},
        {"register_budget": 33},
        {"operand_bits": 24, "accumulator_bits": 48},
    ],
)
def test_invalid_target_facts_refuse(kw):
    with pytest.raises(ValueError):
        target(**kw)


def test_budget_cannot_hold_one_column_refuses():
    with pytest.raises(ValueError, match="workspace"):
        IP.derive(target(workspace_bytes=1), k=144, output_width=64)


def test_default_off_never_parses_or_rewrites(tmp_path):
    path = tmp_path / "opaque.mlir"
    path.write_bytes(b"opaque not even MLIR")
    returned, report = IP.rewrite_prepared_file(path, target(), tmp_path)
    assert returned == path and report.streamed == 0
    assert list(tmp_path.iterdir()) == [path]
    module = mq.parse(_module())
    before = str(module)
    assert IP.rewrite_module(module, target()).streamed == 0
    assert str(module) == before


@pytest.mark.parametrize("vlen", [128, 256, 512, 1024])
@pytest.mark.parametrize(
    "params",
    [
        {"n": 2, "ow": 131, "sh": 2, "sw": 2, "dh": 2, "dw": 2},
        {"c": 1, "kh": 1, "kw": 1, "oh": 1, "ow": 7, "f": 3},
    ],
)
def test_local_gather_and_consumer_are_inside_loop_workspace_outside(vlen, params):
    module = mq.parse(_module(**params))
    report = IP.rewrite_module(module, target(vlen), enabled=True)
    module.verify()
    assert report.streamed == 1
    loops = list(mq.walk(module, "scf.for"))
    gathers = [op for op in mq.walk(module, "linalg.generic") if len(op.inputs) == 1]
    assert len(gathers) == len(loops) in (1, 2)
    for loop, gather in zip(loops, gathers):
        assert gather.parent is loop.body.block
        # Fully overwritten workspace is an iterarg, not an allocation in the loop.
        assert gather.outputs[0] is loop.body.block.args[2]
        assert all(op.name != "tensor.empty" for op in loop.body.block.ops)
        assert len(gather.results[0].type.get_shape()) == 4
    assert not any(
        len(op.results[0].type.get_shape()) == 6
        for op in mq.walk(module, "tensor.empty")
    )
    text = IP.lowering_schedule(report)
    assert "lower_contraction" in text
    pipeline = IP.lowering_pipeline(Path("/tmp/stream-schedule.mlir"))
    assert "linalg-specialize-generic-ops" not in pipeline
    assert "linalg-specialize-generic-ops" in BACKEND.build_pipeline("/tmp/base.mlir")


@pytest.mark.parametrize(
    "change,reason",
    [
        (lambda s: s.replace("arith.extsi", "arith.extui", 1), "requires_pure_signed"),
        (
            lambda s: s.replace("%c0 = arith.constant 0", "%c0 = arith.constant 1"),
            "requires_pure_signed",
        ),
    ],
)
def test_unsupported_numeric_contract_is_unchanged(change, reason):
    module = mq.parse(change(_module()))
    before = str(module)
    report = IP.rewrite_module(module, target(), enabled=True)
    assert report.streamed == 0 and str(module) == before
    assert any(reason in k for k in report.refusals)


@pytest.mark.parametrize("operation", ["arith.muli", "arith.addi"])
def test_overflow_poison_contract_is_not_replaced_by_modular_mac(operation):
    from xdsl.dialects.arith import IntegerOverflowAttr, IntegerOverflowFlag

    module = mq.parse(_module())
    op = next(mq.walk(module, operation))
    op.properties["overflowFlags"] = IntegerOverflowAttr([IntegerOverflowFlag.NSW])
    before = str(module)
    report = IP.rewrite_module(module, target(), enabled=True)
    assert report.streamed == 0 and str(module) == before
    assert "requires_pure_signed_i8_i32_zero_mac" in report.refusals


def test_out_of_bounds_and_shared_gather_refuse_without_mutation():
    module = mq.parse(_module())
    gather = next(op for op in mq.walk(module, "linalg.generic") if len(op.inputs) == 1)
    # A second use preserves the original gather and must decline fusion.
    collapse = next(mq.walk(module, "tensor.collapse_shape"))
    duplicate = collapse.clone()
    gather.parent.insert_op_after(duplicate, collapse)
    before = str(module)
    report = IP.rewrite_module(module, target(), enabled=True)
    assert report.streamed == 0 and str(module) == before
    assert report.refusals["refused_shared_value"] == 1


def test_noninteger_target_contract_is_explicitly_unsupported():
    module = mq.parse(_module())
    before = str(module)
    report = IP.rewrite_module(module, target(operand_bits=32), enabled=True)
    assert report.refusals == {"unsupported_numeric_contract": 1}
    assert str(module) == before


def test_out_of_bounds_preserves_original():
    module = mq.parse(_module().replace("x6x34xi8", "x5x34xi8"))
    before = str(module)
    report = IP.rewrite_module(module, target(), enabled=True)
    assert report.streamed == 0 and str(module) == before
    assert any("window exceeds input" in reason for reason in report.refusals)
