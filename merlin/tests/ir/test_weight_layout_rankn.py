"""Checked rank-N offline layouts: typed source joins to exact raw stored bytes.

These tests qualify only the shared layout mechanism. Immutable invocation
ownership is an explicit original caller premise; no hardware support is inferred.
"""

import itertools
import json
import struct
from dataclasses import replace
from math import prod

import pytest
from xdsl.dialects import builtin, func, tensor
from xdsl.dialects.linalg.ops import TransposeOp

from merlin.capture import rewrite as rw
from merlin.capture.safetensors import load_header
from merlin.common import mlir_query as mq
from merlin.xdsl_dialects._common import text
from merlin.xdsl_dialects.lowering.weight_layout import weight_layout_report


def tensor_type(shape, dtype="i16"):
    return "tensor<" + "x".join(map(str, shape)) + "x" + dtype + ">"


def source(shape=(2, 3, 5), permutation=(1, 2, 0), dtype="i16", *, chain=False, live_init=False):
    shape = tuple(shape)
    result = tuple(shape[i] for i in permutation)
    before, after = tensor_type(shape, dtype), tensor_type(result, dtype)
    body = f"""%init=tensor.empty():{after}
    %moved=linalg.transpose ins(%weight:{before}) outs(%init:{after}) permutation={list(permutation)}
"""
    value = "%moved"
    if chain:
        flat, expanded = tensor_type((prod(shape),), dtype), tensor_type((2, prod(shape) // 2), dtype)
        final = tensor_type((prod(shape) // 2, 2), dtype)
        body += f"""%flat=tensor.collapse_shape %moved {[list(range(len(shape)))]}:{after} into {flat}
    %matrix=tensor.expand_shape %flat [[0,1]] output_shape [2,{prod(shape) // 2}]:{flat} into {expanded}
    %last_init=tensor.empty():{final}
    %last=linalg.transpose ins(%matrix:{expanded}) outs(%last_init:{final}) permutation=[1,0]
"""
        value, after = "%last", final
    returns = f"({after},{tensor_type(result, dtype)})" if live_init else after
    operands = f"{value},%init" if live_init else value
    types = f"{after},{tensor_type(result, dtype)}" if live_init else after
    return f"""builtin.module attributes {{test.module="preserved"}} {{
  func.func @forward_extra(%weight:{before})->{before} {{return %weight:{before}}}
  func.func @forward(%weight:{before})->{returns} attributes {{test.function="preserved"}} {{
    {body}
    return {operands}:{types}
  }}
}}"""


def bundle(path, mlir, *, stored_dtype="I16", manifest_dtype="int16", kind="param", raw=None):
    path.mkdir()
    module = mq.parse(mlir)
    report = weight_layout_report(module)
    shape = report.relayouts[0].shape
    width = {"I8": 1, "I16": 2, "F16": 2, "BF16": 2, "F32": 4, "F64": 8}[stored_dtype]
    raw = raw if raw is not None else b"".join(i.to_bytes(width, "little") for i in range(prod(shape)))
    header = {
        "w": dict(dtype=stored_dtype, shape=shape, data_offsets=[0, len(raw)]),
        "untouched": dict(dtype="I8", shape=[3], data_offsets=[len(raw), len(raw) + 3]),
        "__metadata__": {"test": "unchanged"},
    }
    (path / "model.mlir").write_text(mlir)
    write_blob(path, header, raw + b"\x80\x00\xff")
    manifest = {"0": dict(kind=kind, weight="w", shape=shape, dtype=manifest_dtype)}
    (path / "weights.safetensors.manifest.json").write_text(json.dumps(manifest))
    return path, raw, report.hoistable[0]


def write_blob(path, header, payload):
    data = json.dumps(header).encode()
    data += b" " * (-len(data) % 8)
    (path / "weights.safetensors").write_bytes(struct.pack("<Q", len(data)) + data + payload)


def read_blob(path):
    header, offset = load_header(path / "weights.safetensors")
    payload = (path / "weights.safetensors").read_bytes()[offset:]
    return header, {
        name: payload[s:e] for name, spec in header.items() if name != "__metadata__" for s, e in [spec["data_offsets"]]
    }


def independent_bytes(raw, shape, steps):
    """Coordinate enumeration and byte slicing, independent of NumPy layout operators."""
    width = len(raw) // prod(shape)
    records = [raw[i : i + width] for i in range(0, len(raw), width)]
    shape = tuple(shape)
    for step in steps:
        if step.operation == "linalg.transpose":
            moved = []
            for output in itertools.product(*(range(d) for d in step.result_shape)):
                index = [0] * len(shape)
                for axis, original_axis in enumerate(step.permutation):
                    index[original_axis] = output[axis]
                flat = 0
                for coordinate, dimension in zip(index, shape, strict=True):
                    flat = flat * dimension + coordinate
                moved.append(records[flat])
            records = moved
        shape = step.result_shape
    return b"".join(records)


@pytest.mark.parametrize(
    "shape,permutation,chain",
    [
        ((7,), (0,), False),
        ((3, 5), (1, 0), False),
        ((2, 3, 5), (1, 2, 0), False),
        ((2, 3, 2, 5), (0, 2, 3, 1), True),
        ((2, 1, 3, 2, 5), (4, 2, 0, 3, 1), True),
    ],
)
@pytest.mark.parametrize("generic", [False, True])
def test_rankn_and_full_chain_match_independent_element_coordinates(tmp_path, shape, permutation, chain, generic):
    mlir = source(shape, permutation, chain=chain)
    module = mq.parse(mlir)
    module.verify()
    if generic:
        mlir = text(module, generic=True)
    src, raw, relayout = bundle(tmp_path / "source", mlir)
    original = {p.name: p.read_bytes() for p in src.iterdir()}
    dst = tmp_path / "result"
    record = rw.hoist_weight_transposes(src, dst)
    checked = rw.verify_hoisted_bundle(src, dst)
    assert all(record.effect[key] == value for key, value in checked.items())
    header, values = read_blob(dst)
    assert values["w"] == independent_bytes(raw, shape, relayout.steps)
    assert values["untouched"] == b"\x80\x00\xff" and header["__metadata__"] == {"test": "unchanged"}
    assert header["w"]["shape"] == relayout.result_shape
    assert record.effect["transposes_removed"] == (2 if chain else 1)
    assert record.effect["bytes_moved_per_inference_before"] == len(raw) * (2 if chain else 1)
    assert record.effect["ops_removed"] == (6 if chain else 2)
    selected = mq.parse((dst / "model.mlir").read_text())
    selected.verify()
    assert not any(
        isinstance(op, TransposeOp | tensor.CollapseShapeOp | tensor.ExpandShapeOp) for op in selected.walk()
    )
    functions = {op.sym_name.data: op for op in selected.body.block.ops if isinstance(op, func.FuncOp)}
    assert functions["forward"].args[0].type.get_shape() == tuple(relayout.result_shape)
    assert text(functions["forward_extra"], generic=True) == text(module.body.block.first_op, generic=True)
    assert selected.attributes == module.attributes
    assert functions["forward"].attributes == module.body.block.last_op.attributes
    assert original == {p.name: p.read_bytes() for p in src.iterdir()}


@pytest.mark.parametrize(
    "stored,dtype,manifest,width,words",
    [
        ("BF16", "bf16", "bfloat16", 2, [0x8000, 0, 0x7F81, 0x7FC5, 0xFF80, 0x3F80]),
        ("F16", "f16", "float16", 2, [0x8000, 0, 0x7C01, 0x7E35, 0xFC00, 0x3C00]),
        ("F32", "f32", "float32", 4, [0x80000000, 0, 0x7F800001, 0x7FC01235, 0xFF800000, 0x3F800000]),
        (
            "F64",
            "f64",
            "float64",
            8,
            [0x8000000000000000, 0, 0x7FF0000000000001, 0x7FF8000000001235, 0xFFF0000000000000, 0x3FF0000000000000],
        ),
    ],
)
def test_floating_layouts_preserve_signaling_nan_payloads_and_signed_zero(
    tmp_path, stored, dtype, manifest, width, words
):
    raw = b"".join(word.to_bytes(width, "little") for word in words)
    src, raw, layout = bundle(
        tmp_path / "source", source((2, 3), (1, 0), dtype), stored_dtype=stored, manifest_dtype=manifest, raw=raw
    )
    rw.hoist_weight_transposes(src, tmp_path / "result")
    rw.verify_hoisted_bundle(src, tmp_path / "result")
    _, values = read_blob(tmp_path / "result")
    assert values["w"] == independent_bytes(raw, (2, 3), layout.steps)


def test_live_init_and_all_result_readers_survive_typed_rewrite():
    module = mq.parse(source(live_init=True))
    fn = module.body.block.last_op
    returned = fn.body.block.last_op
    # A second read of the transpose result is safe: both now read the new argument.
    returned.operands = (returned.operands[0], returned.operands[0], returned.operands[1])
    fn.update_function_type()
    module.verify()
    changed, removed = rw.rewrite_transposed_args(text(module, generic=True), {0})
    selected = mq.parse(changed)
    selected.verify()
    fn = selected.body.block.last_op
    assert removed == 1
    assert isinstance(fn.body.block.first_op, tensor.EmptyOp)
    assert fn.body.block.last_op.operands[:2] == (fn.args[0], fn.args[0])
    assert fn.body.block.last_op.operands[2] is fn.body.block.first_op.results[0]


def test_exact_function_name_and_same_shape_permutation_freshness():
    module = mq.parse(source((2, 2, 2), (1, 2, 0)))
    layout = weight_layout_report(module).hoistable[0]
    assert layout.permutation == (1, 2, 0)
    for name in ("forw", "missing", "forward_extra_suffix", ""):
        with pytest.raises(ValueError, match="function|name"):
            weight_layout_report(module, name)
    changed = source((2, 2, 2), (2, 0, 1))
    with pytest.raises(rw.RewriteRefused, match="stale.*permutation"):
        rw.rewrite_transposed_args(changed, {0}, relayouts=(layout,))
    with pytest.raises(rw.RewriteRefused, match="stale.*permutation"):
        rw.rewrite_transposed_args(text(module, generic=True), {0}, relayouts=(replace(layout, permutation=(0, 1, 2)),))


@pytest.mark.parametrize(
    "field,value,expected",
    [
        ("shape", [3, 2, 5], "shapes disagree"),
        ("dtype", "F16", "dtype disagree"),
        ("data_offsets", [0, 61], "extent"),
        ("data_offsets", [-1, 59], "bounds"),
        ("data_offsets", [0, 6000], "bounds"),
        ("dtype", "UNKNOWN", "dtype disagree"),
    ],
)
def test_header_and_source_disagreement_refuses_before_destination(tmp_path, field, value, expected):
    src, _, _ = bundle(tmp_path / "source", source())
    header, values = read_blob(src)
    header["w"][field] = value
    write_blob(src, header, values["w"] + values["untouched"])
    with pytest.raises(rw.RewriteRefused, match=expected):
        rw.hoist_weight_transposes(src, tmp_path / "result")
    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize(
    "change,expected",
    [
        ({"shape": [3, 2, 5]}, "shapes disagree"),
        ({"dtype": "float16"}, "dtype disagree"),
        ({"mutable": True}, "immutable"),
        ({"kind": "buffer"}, "immutable"),
        ({"kind": "input"}, "immutable"),
        ({"stub": True}, "stub"),
    ],
)
def test_manifest_geometry_and_ownership_are_checked(tmp_path, change, expected):
    src, _, _ = bundle(tmp_path / "source", source())
    manifest = json.loads((src / "weights.safetensors.manifest.json").read_text())
    manifest["0"].update(change)
    (src / "weights.safetensors.manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(rw.RewriteRefused, match=expected):
        rw.hoist_weight_transposes(src, tmp_path / "result")


@pytest.mark.parametrize("role", ["states", "streams"])
def test_session_updates_refuse_even_with_explicit_immutable_premise(tmp_path, role):
    src, _, _ = bundle(tmp_path / "source", source(), kind="buffer")
    (src / "session_contract.yaml").write_text(f"version: 1\n{role}:\n- input_arg: 0\n  output_index: 0\n")
    with pytest.raises(rw.RewriteRefused, match="session-carried"):
        rw.hoist_weight_transposes(src, tmp_path / "result", immutable_arguments=(0,))


def test_explicit_original_immutable_buffer_premise_is_supported(tmp_path):
    src, _, layout = bundle(tmp_path / "source", source(), kind="buffer")
    rw.hoist_weight_transposes(src, tmp_path / "result", immutable_arguments=(0,))
    assert read_blob(tmp_path / "result")[0]["w"]["shape"] == layout.result_shape


def test_original_storage_aliases_refuse_even_when_both_arguments_are_hoistable(tmp_path):
    mlir = source((2, 2), (1, 0))
    module = mq.parse(mlir)
    fn = module.body.block.last_op
    # Duplicate a complete transpose chain on a distinct argument with the same stored name.
    fn.body.block.insert_arg(fn.args[0].type, 1)
    first = next(op for op in fn.walk() if isinstance(op, TransposeOp))
    init = tensor.EmptyOp([], first.outputs[0].type)
    other = TransposeOp(fn.args[1], init.tensor, first.permutation)
    fn.body.block.insert_ops_before([init, other], fn.body.block.last_op)
    fn.body.block.last_op.operands = (first.results[0], other.results[0])
    fn.update_function_type()
    module.verify()
    src, _, _ = bundle(tmp_path / "source", text(module, generic=True))
    manifest_path = src / "weights.safetensors.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["1"] = manifest["0"].copy()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(rw.RewriteRefused, match="also read|alias"):
        rw.hoist_weight_transposes(src, tmp_path / "result")


@pytest.mark.parametrize("mutation", ["outs", "nested", "dynamic", "encoded", "body", "attribute", "permutation"])
def test_source_semantic_mutations_are_barriers(mutation):
    module = mq.parse(source((2, 2), (1, 0)))
    fn = module.body.block.last_op
    transpose = next(op for op in fn.walk() if isinstance(op, TransposeOp))
    if mutation == "outs":
        # The only use of the parameter is as the output init, not input operand zero.
        transpose.operands = (transpose.outputs[0], fn.args[0])
    elif mutation == "body":
        transpose.body.block.last_op.operands = (transpose.body.block.args[1],)
    elif mutation == "attribute":
        transpose.attributes["unqualified.execution"] = builtin.StringAttr("changed")
    elif mutation == "permutation":
        transpose.properties["permutation"] = builtin.DenseArrayBase.from_list(builtin.i64, [0, 0])
    elif mutation in {"dynamic", "encoded"}:
        from xdsl.rewriter import Rewriter

        kind = (
            builtin.TensorType(builtin.i16, [-1, 2])
            if mutation == "dynamic"
            else builtin.TensorType(builtin.i16, [2, 2], builtin.StringAttr("physical-layout-unknown"))
        )
        for value in (fn.args[0], transpose.outputs[0], transpose.results[0]):
            Rewriter.replace_value_with_new_type(value, kind)
        fn.update_function_type()
    else:
        from xdsl.dialects import arith, scf
        from xdsl.ir import Block, Region

        empty = transpose.outputs[0].owner
        body = Block([empty.clone(), transpose.clone(), scf.YieldOp()])
        # Remap the cloned transpose's init to its actual region-owned destination.
        list(body.ops)[1].operands = (fn.args[0], body.first_op.results[0])
        fn.body.block.last_op.operands = (empty.results[0],)
        fn.update_function_type()
        fn.body.block.erase_op(transpose)
        condition = arith.ConstantOp.from_int_and_width(1, 1)
        nested = scf.IfOp(condition.result, [], Region(body))
        fn.body.block.insert_ops_before([condition, nested], fn.body.block.last_op)
    try:
        module.verify()
    except Exception:
        with pytest.raises(ValueError, match="verified source"):
            weight_layout_report(module)
    else:
        report = weight_layout_report(module)
        assert not report.hoistable and report.blocked
        with pytest.raises(rw.RewriteRefused, match="sole-input-use"):
            rw.rewrite_transposed_args(text(module, generic=True), {0})


def test_full_chain_stops_before_multiple_readers_and_unknown_body(tmp_path):
    module = mq.parse(source((2, 3, 2, 5), (0, 2, 3, 1), chain=True))
    fn = module.body.block.last_op
    first = next(op for op in fn.walk() if isinstance(op, TransposeOp))
    fn.body.block.last_op.operands = (*fn.body.block.last_op.operands, first.results[0])
    fn.update_function_type()
    module.verify()
    layout = weight_layout_report(module).hoistable[0]
    assert len(layout.steps) == 1 and layout.transposes == 1
    selected, dropped = rw.rewrite_transposed_args(text(module, generic=True), {0}, relayouts=(layout,))
    parsed = mq.parse(selected)
    parsed.verify()
    assert dropped == 2 and sum(isinstance(op, TransposeOp) for op in parsed.walk()) == 1
    assert any(isinstance(op, tensor.CollapseShapeOp) for op in parsed.walk())
    # An unrecognized field on a downstream operation also stops, never crosses, it.
    module = mq.parse(source((2, 3, 2, 5), (0, 2, 3, 1), chain=True))
    collapse = next(op for op in module.walk() if isinstance(op, tensor.CollapseShapeOp))
    collapse.attributes["unqualified.execution"] = builtin.UnitAttr()
    layout = weight_layout_report(module).hoistable[0]
    assert len(layout.steps) == 1


def test_same_shape_second_permutation_and_reshape_geometry_are_checked():
    module = mq.parse(source((2, 2), (1, 0), chain=True))
    layout = weight_layout_report(module).hoistable[0]
    last = [op for op in module.walk() if isinstance(op, TransposeOp)][-1]
    last.properties["permutation"] = builtin.DenseArrayBase.from_list(builtin.i64, [0, 1])
    module.verify()
    with pytest.raises(rw.RewriteRefused, match="stale.*permutation"):
        rw.rewrite_transposed_args(text(module, generic=True), {0}, relayouts=(layout,))
    module = mq.parse(source((2, 2), (1, 0), chain=True))
    collapse = next(op for op in module.walk() if isinstance(op, tensor.CollapseShapeOp))
    # Keep the bad collapse's consumer type-valid, isolating its missing product check.
    from xdsl.rewriter import Rewriter

    Rewriter.replace_value_with_new_type(collapse.result, builtin.TensorType(builtin.i16, [5]))
    fn = module.body.block.last_op
    fn.body.block.last_op.operands = (collapse.result,)
    for op in reversed(list(fn.body.block.ops)[list(fn.body.block.ops).index(collapse) + 1 : -1]):
        fn.body.block.erase_op(op)
    fn.update_function_type()
    module.verify()
    report = weight_layout_report(module)
    assert len(report.hoistable[0].steps) == 1


def test_overlapping_stored_byte_records_and_truncation_refuse(tmp_path):
    src, _, _ = bundle(tmp_path / "source", source())
    header, values = read_blob(src)
    header["untouched"]["data_offsets"] = [0, 3]
    write_blob(src, header, values["w"] + values["untouched"])
    with pytest.raises(rw.RewriteRefused, match="shares bytes"):
        rw.hoist_weight_transposes(src, tmp_path / "result")
    header["untouched"]["data_offsets"] = [len(values["w"]), len(values["w"]) + 3]
    write_blob(src, header, values["w"][:-1])
    with pytest.raises(rw.RewriteRefused, match="bounds"):
        rw.hoist_weight_transposes(src, tmp_path / "result")


def test_selected_function_symbol_is_exact_and_internal_callers_refuse():
    original = source((2, 3), (1, 0))
    selected, removed = rw.rewrite_transposed_args(original, {0}, "forward")
    assert removed == 2
    assert weight_layout_report(mq.parse(selected)).relayouts == []
    # A different exact function has no matching source layout; no first-function fallback.
    with pytest.raises(rw.RewriteRefused, match="sole-input-use"):
        rw.rewrite_transposed_args(original, {0}, "forward_extra")
    module = mq.parse(original)
    fn = module.body.block.last_op
    caller = func.FuncOp("caller", ([fn.args[0].type], list(fn.function_type.outputs)))
    call = func.CallOp("forward", [caller.args[0]], list(fn.function_type.outputs))
    caller.body.block.add_ops([call, func.ReturnOp(*call.results)])
    module.body.block.add_op(caller)
    module.verify()
    with pytest.raises(rw.RewriteRefused, match="internally referenced"):
        rw.rewrite_transposed_args(text(module, generic=True), {0})


@pytest.mark.parametrize(
    "mutation", ["data", "untouched", "source-permutation", "model", "manifest", "header", "trailing"]
)
def test_cache_output_hashes_cannot_endorse_wrong_source_relation(tmp_path, mutation):
    src, _, _ = bundle(tmp_path / "source", source((2, 2, 2), (1, 2, 0)))
    dst = tmp_path / "result"
    rw.hoist_weight_transposes(src, dst)
    rw.verify_hoisted_bundle(src, dst)
    header, values = read_blob(dst)
    if mutation in {"data", "untouched"}:
        name = "w" if mutation == "data" else "untouched"
        values[name] = bytes([values[name][0] ^ 1]) + values[name][1:]
        write_blob(dst, header, values["w"] + values["untouched"])
    elif mutation == "source-permutation":
        module = mq.parse((src / "model.mlir").read_text())
        transpose = next(op for op in module.walk() if isinstance(op, TransposeOp))
        transpose.properties["permutation"] = builtin.DenseArrayBase.from_list(builtin.i64, [2, 0, 1])
        module.verify()
        (src / "model.mlir").write_text(text(module, generic=True))
    elif mutation == "model":
        module = mq.parse((dst / "model.mlir").read_text())
        fn = module.body.block.last_op
        init = tensor.EmptyOp([], fn.args[0].type)
        fn.body.block.insert_op_before(init, fn.body.block.last_op)
        fn.body.block.last_op.operands = (init.tensor,)
        module.verify()
        (dst / "model.mlir").write_text(text(module, generic=True))
    elif mutation == "manifest":
        path = dst / "weights.safetensors.manifest.json"
        manifest = json.loads(path.read_text())
        manifest["0"]["kind"] = "input"
        path.write_text(json.dumps(manifest))
    elif mutation == "header":
        header["w"]["shape"] = [2, 4]
        write_blob(dst, header, values["w"] + values["untouched"])
    else:
        path = dst / "weights.safetensors"
        path.write_bytes(path.read_bytes() + b"\x00")
    # Even a forged producer receipt with current digests supplies no semantic proof.
    from merlin.common.digest import sha256_file

    (dst / rw.REWRITES_FILE).write_text(
        json.dumps(
            {"rewrites": [], "current": {p.name: sha256_file(p) for p in dst.iterdir() if p.name != rw.REWRITES_FILE}}
        )
    )
    with pytest.raises(rw.RewriteRefused, match="bytes|IR|manifest|header|extent"):
        rw.verify_hoisted_bundle(src, dst)


def test_generic_module_provenance_is_retargeted_without_dropping_other_regions(tmp_path):
    module = mq.parse(source(chain=True))
    module.attributes["prov.weights_file"] = builtin.StringAttr("original/staging.safetensors")
    before = text(module, generic=True)
    changed, retargeted = rw.retarget_weights_file(before, tmp_path / "published.safetensors")
    expected = module.clone()
    expected.attributes["prov.weights_file"] = builtin.StringAttr(str(tmp_path / "published.safetensors"))
    assert retargeted
    assert text(mq.parse(changed), generic=True) == text(expected, generic=True)


def test_explicit_selection_leaves_session_state_and_its_transpose_unchanged(tmp_path):
    module = mq.parse(source((2, 2), (1, 0)))
    fn = module.body.block.last_op
    fn.body.block.insert_arg(fn.args[0].type, 1)
    first = next(op for op in fn.walk() if isinstance(op, TransposeOp))
    init = tensor.EmptyOp([], first.outputs[0].type)
    state = TransposeOp(fn.args[1], init.tensor, first.permutation)
    fn.body.block.insert_ops_before([init, state], fn.body.block.last_op)
    fn.body.block.last_op.operands = (first.results[0], state.results[0])
    fn.update_function_type()
    module.verify()
    src, _, _ = bundle(tmp_path / "source", text(module, generic=True))
    header, values = read_blob(src)
    header["state"] = dict(dtype="I16", shape=[2, 2], data_offsets=[11, 19])
    write_blob(src, header, values["w"] + values["untouched"] + values["w"])
    path = src / "weights.safetensors.manifest.json"
    manifest = json.loads(path.read_text())
    manifest["1"] = dict(kind="buffer", weight="state", shape=[2, 2], dtype="int16")
    path.write_text(json.dumps(manifest))
    (src / "session_contract.yaml").write_text("version: 1\nstates:\n- input_arg: 1\n  output_index: 1\n")
    dst = tmp_path / "result"
    record = rw.hoist_weight_transposes(src, dst, immutable_arguments=(0,))
    assert record.effect["weights_pre_transposed"] == 1 and record.effect["outside_immutable_selection"] == 1
    rw.verify_hoisted_bundle(src, dst, immutable_arguments=(0,))
    assert read_blob(dst)[1]["state"] == values["w"]
    selected = mq.parse((dst / "model.mlir").read_text())
    assert sum(isinstance(op, TransposeOp) for op in selected.walk()) == 1
