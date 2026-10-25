"""Group source-bound current SSA before ordinary per-interface extraction."""

from copy import deepcopy

import pytest
from xdsl.dialects import func

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.connected_spatial import group_connected_spatial, spatial_pair_interface
from merlin.llvmlower.device_offload import DeviceRewrite, Routed
from merlin.targetgen.contract.interface_emit import emit_interface_mlir, parse_interface_mlir
from merlin.xdsl_dialects._common import text

SOURCE = """module {
 func.func @forward(%x:tensor<2x5x5x3xi8>,%w:tensor<27x17xi8>,%b:tensor<17xi32>) ->tensor<18x17xi8> {
  %c=tensor.empty():tensor<50x17xi8>
  %p=func.call @producer(%x,%w,%b,%c):
    (tensor<2x5x5x3xi8>,tensor<27x17xi8>,tensor<17xi32>,tensor<50x17xi8>)->tensor<50x17xi8>
  %y=tensor.empty():tensor<18x17xi8>
  %q=func.call @pool(%p,%y):(tensor<50x17xi8>,tensor<18x17xi8>)->tensor<18x17xi8>
  return %q:tensor<18x17xi8>
 }
 func.func private @producer(tensor<2x5x5x3xi8>,tensor<27x17xi8>,tensor<17xi32>,tensor<50x17xi8>)->tensor<50x17xi8>
 func.func private @pool(tensor<50x17xi8>,tensor<18x17xi8>)->tensor<18x17xi8>
}"""


def fixture():
    pe = dict(op="conv2d", operand_dtype="int8", epilogue=["bias_add", "acc_scale", "relu"], acc_scale=0.5)
    qe = dict(op="maxpool2d", operand_dtype="int8", epilogue=["acc_scale", "maxpool"], acc_scale=0.9900199770927429)
    shapes = {"producer": [(2, 5, 5, 3), (27, 17), (17,), (50, 17)], "pool": [(50, 17), (18, 17)]}
    rewrite = DeviceRewrite(
        "ordinary_selected_target",
        routed=tuple(Routed(s, (), (), ("i8", "i8", "i8")) for s in shapes),
        entries={"producer": pe, "pool": qe},
        programs={s: {"schema": "group_program_v1", "entry": deepcopy(e)} for s, e in (("producer", pe), ("pool", qe))},
        arg_access={"producer": ("read", "read", "read", "write"), "pool": ("read", "write")},
        call_buffers={
            s: [{"shape": list(v), "dtype": "i32" if len(v) == 1 else "i8"} for v in dims] for s, dims in shapes.items()
        },
    )
    return parse_mlir_text(SOURCE), rewrite


def test_current_ssa_group_preserves_public_signature_and_exact_statements():
    module, rewrite = fixture()
    before = text(module)
    selected = group_connected_spatial(module, rewrite)
    assert text(module) == before
    assert selected.module.body.block.first_op.function_type == module.body.block.first_op.function_type
    calls = [op for op in selected.module.walk() if isinstance(op, func.CallOp)]
    assert len(calls) == 1 and len(calls[0].arguments) == 4
    assert len(selected.rewrite.routed) == 1
    assert set(selected.rewrite.signatures) == {calls[0].callee.root_reference.data}
    assert selected.rewrite.signatures[calls[0].callee.root_reference.data] == ()
    entry = next(iter(selected.rewrite.entries.values()))
    assert entry["components"] == [rewrite.entries["producer"], rewrite.entries["pool"]]
    assert selected.receipt["groups"][0]["external_intermediate_observers"] == []
    assert selected.receipt["target_execution"] is False
    selected.module.verify()


@pytest.mark.parametrize(
    "mutation", ["observer", "intervening_effect", "borrowed_output", "program", "format", "shared_callee"]
)
def test_nonclosed_or_unbound_source_is_left_as_two_requests(mutation):
    source = SOURCE
    if mutation == "observer":
        source = source.replace(
            "  %y=tensor.empty()",
            "  %unused=func.call @observe(%p):(tensor<50x17xi8>)->tensor<50x17xi8>\n  %y=tensor.empty()",
        )
        source = source.replace(
            " func.func private @pool",
            " func.func private @observe(tensor<50x17xi8>)->tensor<50x17xi8>\n func.func private @pool",
        )
    elif mutation == "intervening_effect":
        source = source.replace("  %y=tensor.empty()", "  func.call @effect():()->()\n  %y=tensor.empty()")
        source = source.replace(" func.func private @pool", " func.func private @effect()\n func.func private @pool")
    elif mutation == "borrowed_output":
        source = source.replace("%b:tensor<17xi32>)", "%b:tensor<17xi32>,%c:tensor<50x17xi8>)", 1).replace(
            "  %c=tensor.empty():tensor<50x17xi8>\n", ""
        )
    elif mutation == "shared_callee":
        source = source.replace(
            "  return %q",
            "  %other=func.call @pool(%p,%y):(tensor<50x17xi8>,tensor<18x17xi8>)->tensor<18x17xi8>\n  return %q",
        )
    module, rewrite = fixture()
    module = parse_mlir_text(source)
    if mutation == "program":
        rewrite.programs["producer"]["entry"]["acc_scale"] = 0.25
    elif mutation == "format":
        rewrite.call_buffers["producer"][0]["dtype"] = "ui8"
    before = text(module)
    selected = group_connected_spatial(module, rewrite)
    assert selected.receipt["groups"] == []
    assert text(selected.module) == before == text(module)
    assert set(selected.rewrite.entries) == {"producer", "pool"}


def test_composed_request_keeps_both_roundings_and_one_actual_edge(monkeypatch):
    tensors = {
        "IFM": dict(shape=[2, 5, 5, 3], dtype="i8", role="input"),
        "W": dict(shape=[27, 17], dtype="i8", role="weight"),
        "B": dict(shape=[17], dtype="i32", role="bias"),
        "Y": dict(shape=[50, 17], dtype="i8", role="output"),
    }
    first = dict(
        abi_version="0.1",
        target="selected",
        tensors=tensors,
        commands=[
            dict(opcode="RES_PACK", operands=dict(src="W", dst="wr"), attributes=dict(layout="packed_conv_rhs")),
            dict(
                opcode="CONV2D",
                operands=dict(ifm="IFM", weight="wr", dst="Y"),
                attributes=dict(
                    kernel=[3, 3, 3, 17],
                    stride=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    layout="nhwc",
                    bias="B",
                    epilogue=["bias_add", "acc_scale", "relu"],
                    acc_scale=0.5,
                    output_dtype="i8",
                ),
            ),
            dict(opcode="EVICT", operands=dict(handle="wr")),
        ],
    )
    second = dict(
        abi_version="0.1",
        target="selected",
        tensors={"X": {**tensors["Y"], "role": "input"}, "Y": dict(shape=[18, 17], dtype="i8", role="output")},
        commands=[
            dict(
                opcode="MAXPOOL2D",
                operands=dict(src="X", dst="Y"),
                attributes=dict(
                    pool_in_dims=[5, 5],
                    pool_size=[3, 3],
                    pool_stride=[2, 2],
                    pool_padding=[1, 1, 1, 1],
                    pool_pad_value=-128,
                    epilogue=["acc_scale", "maxpool"],
                    acc_scale=0.9900199770927429,
                    output_dtype="i8",
                ),
            )
        ],
    )
    import merlin.llvmlower.whole_program as wp

    monkeypatch.setattr(
        wp, "group_interface", lambda e, *a, **kw: emit_interface_mlir(first if e["op"] == "conv2d" else second)
    )
    _, rewrite = fixture()
    entry = dict(schema="connected_spatial_entry_v1", components=list(rewrite.entries.values()))
    cb = parse_interface_mlir(spatial_pair_interface(entry, object(), source_reference="explicit_public_source"))
    conv, pool = cb["commands"][1], cb["commands"][-1]
    assert conv["attributes"]["acc_scale"] == 0.5
    assert pool["attributes"]["acc_scale"] == 0.9900199770927429
    assert pool["operands"]["src"] == conv["operands"]["dst"]
    assert conv["attributes"]["bias"] == "producer_B"
    assert "pool_X" not in cb["tensors"]
    assert cb["tensors"]["pool_Y"]["shape"] == [18, 17]
