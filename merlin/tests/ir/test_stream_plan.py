"""The stream plan prices what lies between compute groups: staging, prepack, fences, overlap."""

from __future__ import annotations

from types import SimpleNamespace

from merlin.xdsl_dialects.lowering import stream_plan as SPL
from merlin.xdsl_dialects.lowering.dispatch_program import Buffer, DispatchProgram, Node


def _program(nodes: list[Node], results: list[str], args: dict[str, int]) -> DispatchProgram:
    buffers = {
        name: Buffer(id=name, shape=[4, 4], dtype="i8", kind="arg", arg_index=index) for name, index in args.items()
    }
    for node in nodes:
        for out in node.outputs:
            buffers[out] = Buffer(id=out, shape=[4, 4], dtype="i8", kind="intermediate")
    return DispatchProgram(entry="forward", args=sorted(args.values()), buffers=buffers, nodes=nodes, results=results)


def _table(**placement) -> list:
    return [SimpleNamespace(symbol=symbol, placement=where) for symbol, where in placement.items()]


def test_two_device_groups_in_a_row_need_no_fence_and_no_staging_between_them() -> None:
    nodes = [
        Node("dispatch", "d0", ["x", "w0"], ["a"]),
        Node("dispatch", "d1", ["a", "w1"], ["b"]),
        Node("dispatch", "h0", ["b"], ["y"]),
    ]
    plan = SPL.plan(
        _program(nodes, ["y"], {"x": 0, "w0": 1, "w1": 2}),
        _table(d0="unit0", d1="unit0", h0="host"),
        weight_args={1, 2},
    )
    assert plan["fences"] == {"device_nodes": 2, "true_dependences": 1, "avoidable": 1}
    assert plan["lifetimes"]["transient"]["buffers"] == 1  # `a` never leaves the device
    assert plan["staging"]["buffers"] == 1  # `b` crosses to the host
    assert plan["lifetimes"]["constant"]["buffers"] == 2 and plan["lifetimes"]["external"]["buffers"] == 2


def test_a_host_region_between_them_costs_a_fence_and_two_staging_buffers() -> None:
    nodes = [
        Node("dispatch", "d0", ["x", "w0"], ["a"]),
        Node("dispatch", "h0", ["a"], ["q"]),
        Node("dispatch", "d1", ["q", "w1"], ["y"]),
    ]
    plan = SPL.plan(
        _program(nodes, ["y"], {"x": 0, "w0": 1, "w1": 2}),
        _table(d0="unit0", h0="host", d1="unit0"),
        weight_args={1, 2},
    )
    assert plan["fences"]["true_dependences"] == 2 and plan["fences"]["avoidable"] == 0
    assert plan["staging"]["buffers"] == 2 and plan["staging"]["per_device_node"] == 1.0


def test_work_on_constants_is_prepack_and_not_per_inference_host_work() -> None:
    nodes = [
        Node("dispatch", "h_fold", ["w0"], ["w_folded"]),  # weights only: offline
        Node("dispatch", "d0", ["x", "w_folded"], ["y"]),
    ]
    known = SPL.plan(_program(nodes, ["y"], {"x": 0, "w0": 1}), _table(h_fold="host", d0="unit0"), weight_args={1})
    assert known["prepack"] == {"dispatches": 1, "elements": 16, "weights_known": True}
    assert known["overlap"]["host_elements"] == 0
    # Without the manifest nothing is provably constant: prepack is understated, never invented.
    unknown = SPL.plan(_program(nodes, ["y"], {"x": 0, "w0": 1}), _table(h_fold="host", d0="unit0"))
    assert unknown["prepack"]["dispatches"] == 0 and unknown["overlap"]["host_elements"] == 16


def test_host_work_independent_of_the_in_flight_device_group_is_overlap() -> None:
    nodes = [
        Node("dispatch", "d0", ["x", "w0"], ["a"]),
        Node("dispatch", "h_side", ["z"], ["s"]),  # reads nothing from d0
        Node("dispatch", "h_join", ["a", "s"], ["y"]),
    ]  # needs d0: not overlap
    plan = SPL.plan(
        _program(nodes, ["y"], {"x": 0, "w0": 1, "z": 2}),
        _table(d0="unit0", h_side="host", h_join="host"),
        weight_args={1},
    )
    assert plan["overlap"] == {"host_elements": 32, "independent_of_in_flight_device": 16, "share": 0.5}


def test_the_manifest_says_which_arguments_are_stored_tensors() -> None:
    manifest = {
        "0": {"weight": "conv.weight", "kind": "param"},
        "1": {"kind": "buffer"},
        "2": {"kind": "input", "name": "image"},
        "meta": {"kind": "param"},
    }
    assert SPL.weight_args_of(manifest) == {0, 1}
    assert SPL.weight_args_of(None) == set()
