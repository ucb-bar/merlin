"""Final physical pointers/readouts are checked independently of virtual views."""

from dataclasses import replace

import pytest
from test_shared_input_schedule import source

from merlin.compile.scheduling.projection_transfer_check import (
    PhysicalProjectionTransfer as P,
)
from merlin.compile.scheduling.projection_transfer_check import (
    VirtualProjectionTransfer as V,
)
from merlin.compile.scheduling.projection_transfer_check import (
    verify_projection_transfers,
)
from merlin.compile.scheduling.projection_views import plan_shared_projection_views
from merlin.compile.scheduling.shared_input_groups import find_shared_input_groups
from merlin.perf.finite_pointer_trace import ArgumentAddress as A


def fixture():
    module = source(3, 7, (4, 6, 2))
    (group,) = find_shared_input_groups(module)
    views = plan_shared_projection_views(
        module, group, weight_element_bytes=1, output_element_bytes=(1, 4, 1), tile_columns=2
    )
    expected = [V("lhs", 0, 0, 3, 7, 0)]
    observed = [P("lhs", A(0), 3, 7, 7, 1, 0)]
    readouts = (("scaled_i8", 0x3F000000, 1), ("full_i32",), ("scaled_i8", 0x3E800000, 0))
    # Deliberately unequal physical row pitches and mixed result formats.
    for member, (first, width, unit) in enumerate(zip(views.boundaries, group.widths, (1, 4, 1))):
        for column in range(0, width, 2):
            expected += [
                V("weight", 0, first + column, 7, 2, 16 + member * 16),
                V("output", 0, first + column, 3, 2, 32 + column),
            ]
            observed += [
                P("weight", A(1 + member, column), 7, 2, width, 1, 16 + member * 16),
                P("output", A(4 + member, column * unit), 3, 2, width * unit, unit, 32 + column, readouts[member]),
            ]
    kwargs = dict(
        lhs_argument=0,
        rhs_arguments=(1, 2, 3),
        output_arguments=(4, 5, 6),
        lhs_element_bytes=1,
        output_readouts=readouts,
        source_module=module,
    )
    return views, tuple(expected), tuple(observed), kwargs


def test_complete_per_member_pointer_pitch_format_readout_and_live_output_cover():
    views, expected, observed, kwargs = fixture()
    report = verify_projection_transfers(views, expected, observed, **kwargs)
    assert report["status"] == "pass" and report["output_elements"] == 36
    assert report["transfers"] == dict(lhs=1, weight=6, output=6)
    assert report["source_contraction_macs"] == 252
    assert report["logical_tensor_bytes"] == dict(lhs=21, weight=84, output=90)
    assert report["requested_transfer_bytes"] == dict(lhs=21, weight=84, output=90)
    assert report["shared_input_request_ratio"] == dict(numerator=21, denominator=21)
    assert report["elapsed_cycles"] is report["utilization"] is None
    assert not report["native_execution"]


def test_repeated_input_transfers_increase_requests_without_changing_logical_work():
    views, expected, observed, kwargs = fixture()
    report = verify_projection_transfers(views, (expected[0], *expected), (observed[0], *observed), **kwargs)
    assert report["requested_transfer_bytes"] == dict(lhs=42, weight=84, output=90)
    assert report["logical_tensor_bytes"] == dict(lhs=21, weight=84, output=90)
    assert report["source_contraction_macs"] == 252
    assert report["shared_input_request_ratio"] == dict(numerator=42, denominator=21)


@pytest.mark.parametrize(
    "change",
    [
        lambda r: replace(r, pointer=A(True, r.pointer.offset)),
        lambda r: replace(r, pointer=A(1, float(r.pointer.offset))),
        lambda r: replace(r, pointer=A(2, r.pointer.offset)),  # Wrong original weight.
        lambda r: replace(r, row_stride_bytes=12),  # VirtualN instead of Ni.
        lambda r: replace(r, pointer=A(r.pointer.argument, r.pointer.offset + 1)),
        lambda r: replace(r, local_row=r.local_row + 1),
        lambda r: replace(r, columns=4),
    ],
)
def test_wrong_actual_rhs_mapping_or_local_operand_refuses(change):
    views, expected, observed, kwargs = fixture()
    actual = list(observed)
    actual[1] = change(actual[1])
    with pytest.raises(ValueError):
        verify_projection_transfers(views, expected, tuple(actual), **kwargs)


@pytest.mark.parametrize(
    "change",
    [
        lambda r: replace(r, readout=("full_i32",)),
        lambda r: replace(r, row_stride_bytes=48),
        lambda r: replace(r, element_bytes=4),
    ],
)
def test_wrong_actual_output_scalar_or_format_refuses(change):
    views, expected, observed, kwargs = fixture()
    actual = list(observed)
    actual[2] = change(actual[2])
    with pytest.raises(ValueError):
        verify_projection_transfers(views, expected, tuple(actual), **kwargs)


def test_cross_member_wide_transfer_extra_transfer_and_uncovered_output_refuse():
    views, expected, observed, kwargs = fixture()
    with pytest.raises(ValueError, match="crosses"):
        verify_projection_transfers(
            views, (V("weight", 0, 2, 7, 4, 16),), (P("weight", A(1, 2), 7, 4, 4, 1, 16),), **kwargs
        )
    with pytest.raises(ValueError):
        verify_projection_transfers(views, expected, observed + observed[-1:], **kwargs)
    with pytest.raises(ValueError, match="uncovered"):
        verify_projection_transfers(views, expected[:-1], observed[:-1], **kwargs)
    with pytest.raises(ValueError, match="overlap"):
        verify_projection_transfers(views, expected + expected[-1:], observed + observed[-1:], **kwargs)


def test_changed_original_lhs_and_aliased_function_arguments_refuse():
    views, expected, observed, kwargs = fixture()
    kwargs["source_module"] = source(3, 7, (4, 6, 2), different_lhs=True)
    with pytest.raises(ValueError, match="source"):
        verify_projection_transfers(views, expected, observed, **kwargs)
    views, expected, observed, kwargs = fixture()
    kwargs["output_arguments"] = (4, 4, 6)
    with pytest.raises(ValueError, match="distinct"):
        verify_projection_transfers(views, expected, observed, **kwargs)


def test_expected_sequence_comes_from_real_counted_kernel_not_a_declared_shape():
    from merlin.compile.scheduling.projection_transfer_check import source_projection_transfers
    from merlin.sched.ir import Call, Const, Kernel, Loop, TensorArg, Var

    views, expected, observed, kwargs = fixture()

    def call(name, column, local):
        return Call(
            name,
            tuple(
                (key, Const(value) if type(value) is int else value)
                for key, value in dict(
                    row=0, column=column, rows=7 if name == "weight" else 3, columns=2, local_row=local
                ).items()
            ),
        )

    body = [
        Call(
            "lhs",
            tuple((key, Const(value)) for key, value in dict(row=0, column=0, rows=3, columns=7, local_row=0).items()),
        )
    ]
    for member, (first, width) in enumerate(zip(views.boundaries, views.source.widths)):
        n = Var("n" + str(member))
        body.append(
            Loop(
                n.name,
                width // 2,
                (call("weight", first + n * 2, 16 + member * 16), call("output", first + n * 2, 32 + n * 2)),
            )
        )
    kernel = Kernel(
        "actual",
        (
            TensorArg("lhs", (3, 7), "i8", "read"),
            TensorArg("rhs", (7, 12), "i8", "read"),
            TensorArg("output", (3, 12), "i32", "write"),
        ),
        tuple(body),
    )

    def project(op, values):
        return V(op.instr, **values)

    actual = source_projection_transfers(kernel, project)
    assert actual == expected
    assert verify_projection_transfers(views, actual, observed, **kwargs)["status"] == "pass"
    with pytest.raises(ValueError, match="budget"):
        source_projection_transfers(kernel, project, max_calls=2)
