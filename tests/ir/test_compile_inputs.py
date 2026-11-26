import hashlib
import json
import os
import subprocess
import sys
from struct import pack, unpack
from types import SimpleNamespace

import pytest
from corefixtures.compile_input_fixtures import SOURCE, _write_capture, _write_weighted_capture
from corefixtures.row_table import run_rows

from merlin.common.paths import module_source_path, repo_root
from merlin.compiler.frontend.inputs import load_compile_inputs, stage_compile_inputs
from merlin.compiler.lower.c.float_arithmetic import emit_float_arithmetic_c
from merlin.compiler.semantic.linalg.linalg_bridge import translate_linalg_text
from merlin.compiler.semantic.model import IndexMap, KernelRequest
from merlin.compiler.semantic.reference import TensorValue, evaluate_graph


def test_c_parameter_binding_does_not_freeze_mutable_state(tmp_path):
    from merlin.compiler.lower.c.parameters import freeze_c_arguments
    from merlin.compiler.semantic.target_binding import NativeCompilationError
    from merlin.development.targetgen.execution.native.native_application import _source

    capture, staged = (tmp_path / "capture", tmp_path / "compiler")
    _write_weighted_capture(capture)
    manifest = capture / "weights.safetensors.manifest.json"
    rows = json.loads(manifest.read_text())
    rows["0"]["kind"] = "buffer"
    raw = json.dumps(rows).encode()
    manifest.write_bytes(raw)
    receipt_path = capture / "capture_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["artifacts"][manifest.name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    receipt_path.write_text(json.dumps(receipt))
    stage_compile_inputs(capture, staged)
    with pytest.raises(ValueError, match="state interface"):
        freeze_c_arguments(staged, name="model", source_sha256=load_compile_inputs(staged).program_sha256)
    args = SimpleNamespace(
        compile_inputs=staged, mode="diagnostic", linalg_entry=None, application_parameters="external"
    )
    binding = SimpleNamespace(
        application_adapter=lambda *args, **kwargs: pytest.fail(
            "application hook must not bypass mutable-state admission"
        )
    )
    with pytest.raises(NativeCompilationError, match="explicit state interface"):
        _source(args, binding, "fixture")
    _check_raw_captured_state(tmp_path)


def _check_raw_captured_state(tmp_path):
    import ctypes
    import shutil
    import struct
    from dataclasses import replace

    from merlin.compiler.lower.c.parameters import StateDeclaration
    from merlin.compiler.semantic.model import SemanticNode, TensorType
    from merlin.compiler.semantic.target_binding import NativeCompilationError
    from merlin.development.targetgen.execution.native.native_application import _source

    capture, stage = tmp_path / "raw_capture", tmp_path / "raw_stage"
    source = """builtin.module attributes {prov.weights_file = "origin/weights.safetensors"} {
      func.func @forward(%codes: tensor<4xi8>, %exponent: tensor<1xi32>, %input: tensor<2xf32>) -> tensor<2xf32> {
        %result = "fixture.read_buffers"(%codes, %exponent, %input) :
          (tensor<4xi8>, tensor<1xi32>, tensor<2xf32>) -> tensor<2xf32>
        return %result : tensor<2xf32>
      }
    }"""
    _write_capture(capture, source=source)
    code_bytes, exponent_bytes = bytes((0, 127, 128, 255)), struct.pack("<i", -7)
    header = json.dumps(
        {
            "codes": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]},
            "exponent": {"dtype": "I32", "shape": [1], "data_offsets": [4, 8]},
        }
    ).encode()
    members = {
        "weights.safetensors": struct.pack("<Q", len(header)) + header + code_bytes + exponent_bytes,
        "weights.safetensors.manifest.json": json.dumps(
            {
                "0": {"kind": "buffer", "weight": "codes", "dtype": "uint8", "shape": [4]},
                "1": {"kind": "buffer", "weight": "exponent", "dtype": "int32", "shape": [1]},
                "2": {"kind": "input", "name": "input"},
            }
        ).encode(),
        "input_order.json": b'{"input": 0}',
    }
    receipt = json.loads((capture / "capture_receipt.json").read_text())
    for member, raw in members.items():
        (capture / member).write_bytes(raw)
        receipt["artifacts"][member] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    (capture / "capture_receipt.json").write_text(json.dumps(receipt))
    stage_compile_inputs(capture, stage)
    staged = load_compile_inputs(stage)
    declaration = {
        "schema": "merlin.application_state.v1",
        "source_program_sha256": staged.program_sha256,
        "ports": [
            dict(argument_index=index, name=name, mode="read_only", output_index=None, initialization="captured")
            for index, name in enumerate(("codes", "exponent"))
        ],
    }
    state_path = tmp_path / "raw_state.json"
    state_path.write_text(json.dumps(declaration))
    args = SimpleNamespace(
        compile_inputs=stage, mode="diagnostic", linalg_entry=None, application_parameters="external"
    )

    def translator(op, operands, results):
        return (SemanticNode(results[0], "fixture_read_buffers", operands, TensorType((2,), "f32", "raw-bit-pattern")),)

    binding = SimpleNamespace(
        application_source_translators=lambda policy: {"fixture.read_buffers": translator},
        application_adapter=lambda policy, **kwargs: object(),
    )
    with pytest.raises(NativeCompilationError, match="explicit state interface"):
        _source(args, binding, "fixture")
    args.application_state = state_path
    _, graph, _, state, _, parameters = _source(args, binding, "fixture")
    assert parameters is None and tuple(item.node_id for item in state.ports) == ("arg0", "arg1")
    assert tuple(item.initial for item in state.ports) == (code_bytes, exponent_bytes)
    assert tuple((item.type.dtype, item.type.numerical_policy) for item in state.ports) == (
        ("i8", "signed-i8"),
        ("i32", "i32-wrap-k-ascending"),
    )
    assert not graph.constants and all(graph.node(item.node_id).effect == "input" for item in state.ports)
    state.check(graph)
    # Raw carriers preserve bytes under either explicit representation policy;
    # boolean or exponent promises are not inferred from their storage widths.
    raw_graph = replace(
        graph,
        nodes=tuple(
            replace(node, type=replace(node.type, numerical_policy="raw-bit-pattern"))
            if node.id in {"arg0", "arg1"}
            else node
            for node in graph.nodes
        ),
    )
    parsed = StateDeclaration.from_record(declaration, staged)
    rebound = parsed.bind(raw_graph, {0: code_bytes, 1: exponent_bytes})
    assert tuple(item.initial for item in rebound.ports) == (code_bytes, exponent_bytes)
    unsupported = replace(
        graph,
        nodes=tuple(
            replace(node, type=replace(node.type, numerical_policy="boolean")) if node.id == "arg0" else node
            for node in graph.nodes
        ),
    )
    with pytest.raises(ValueError, match="supported tensor input"):
        parsed.bind(unsupported, {0: code_bytes, 1: exponent_bytes})
    with pytest.raises(ValueError, match="initialization bytes"):
        parsed.bind(graph, {0: code_bytes, 1: exponent_bytes[:-1]})

    metadata, files = state.emit_initializers("raw_buffers")
    directory = tmp_path / "raw_initializer"
    directory.mkdir()
    for filename, text in files.items():
        (directory / filename).write_text(text)
    compiler = shutil.which("cc")
    assert compiler, "captured-state initialization validation requires a local C compiler"
    library = directory / "initializer.so"
    subprocess.run(
        [
            "nice",
            "-n",
            "10",
            compiler,
            "-std=c11",
            "-O2",
            "-fPIC",
            "-shared",
            str(directory / "raw_buffers_state.c"),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
    )
    initialize = getattr(ctypes.CDLL(str(library)), metadata["entry"])
    initialize.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t)]
    initialize.restype = ctypes.c_int
    codes, exponent = (ctypes.c_ubyte * 4)(*([0x55] * 4)), (ctypes.c_ubyte * 4)(*([0x66] * 4))
    destinations = (ctypes.c_void_p * 2)(ctypes.addressof(codes), ctypes.addressof(exponent))
    capacities = (ctypes.c_size_t * 2)(4, 3)
    assert initialize(destinations, capacities) == -1
    assert bytes(codes) == b"\x55" * 4 and bytes(exponent) == b"\x66" * 4
    capacities[1] = 4
    assert initialize(destinations, capacities) == 0
    assert bytes(codes) == code_bytes and bytes(exponent) == exponent_bytes
    assert (capture / "weights.safetensors").read_bytes() == members["weights.safetensors"]
    assert (stage / "weights.safetensors").read_bytes() == members["weights.safetensors"]


@pytest.mark.parametrize("name", ["model.mlir"])
def test_changed_declared_input_refuses_without_publishing(tmp_path, name):
    capture, staged = (tmp_path / "capture", tmp_path / "compiler")
    _write_capture(capture)
    (capture / name).write_bytes((capture / name).read_bytes() + b"mutation")
    with pytest.raises(ValueError, match="differs from receipt"):
        stage_compile_inputs(capture, staged)
    assert not staged.exists()
    _check_named_capture_constant(tmp_path / "named")


def _check_named_capture_constant(tmp_path):
    import io
    import struct

    import numpy as np
    import torch

    from merlin.compiler.frontend.inputs import load_lifted_argument_bytes

    table = torch.cos(torch.tensor([[0.5, 1.0], [2.0, 3.0]])).to(torch.bfloat16)  # nonzero, sign-mixed
    reference = table.view(torch.int16).numpy().astype("<i2").tobytes()
    storage = table.to(torch.float32).numpy()
    weight = "module.table"
    tmp_path.mkdir()

    def build(directory, name="c_module_table", row=None, others=(), stored=None, payload=reference, first=None):
        _write_weighted_capture(directory, shared_weight=bool(others))
        more = "".join(f", %state{i}: tensor<2x2xbf16>" for i in range(1, len(others)))
        source = (directory / "model.mlir").read_text()
        source = source.replace("%shared: tensor<2x2xf32>", "%shared: tensor<2x2xbf16>" + more)
        rows = {"0": first or {"kind": "constant", "name": name, "weight": weight, "dtype": "bfloat16", "shape": [2, 2],
                               **(row or {})}}
        rows.update({str(i): other for i, other in enumerate(others, 1)})
        rows[str(len(rows))] = {"kind": "input", "name": "input"}
        header = json.dumps({weight: stored or {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8]}}).encode()
        extra = io.BytesIO()
        np.savez(extra, **{name: storage})
        members = {
            "model.mlir": source.replace("%weight: tensor<2x2xf32>", "%table: tensor<2x2xbf16>").encode(),
            "weights.safetensors": struct.pack("<Q", len(header)) + header + payload,
            "weights.safetensors.manifest.json": json.dumps(rows).encode(),
            "extra.npz": extra.getvalue(),
        }
        for member, raw in members.items():
            (directory / member).write_bytes(raw)
        trace = json.loads((directory / "frontend-trace.json").read_text())
        trace["mlir"]["sha256"] = hashlib.sha256(members["model.mlir"]).hexdigest()
        (directory / "frontend-trace.json").write_text(json.dumps(trace))
        meta = json.loads((directory / "meta.json").read_text())
        meta["frontend_trace"]["sha256"] = hashlib.sha256((directory / "frontend-trace.json").read_bytes()).hexdigest()
        (directory / "meta.json").write_text(json.dumps(meta))
        receipt = json.loads((directory / "capture_receipt.json").read_text())
        receipt["lifted_constants"] = {name: {"shape": [2, 2], "dtype": "float32",
                                             "sha256": hashlib.sha256(storage.tobytes()).hexdigest()}}
        receipt["materialized_abi"]["lifted_constants"] = [name]
        for member in {*receipt["artifacts"], *members}:
            raw = (directory / member).read_bytes()
            receipt["artifacts"][member] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        (directory / "capture_receipt.json").write_text(json.dumps(receipt))
        return directory

    staged = tmp_path / "staged"
    stage_compile_inputs(build(tmp_path / "capture"), staged)
    loaded = load_compile_inputs(staged)
    assert loaded.argument_bindings[0] == {"kind": "constant", "weight": weight, "shape": [2, 2], "dtype": "bf16",
                                           "file": "weights.safetensors"}
    assert load_lifted_argument_bytes(staged, loaded, kinds=("constant",), max_bytes=8) == {0: reference}
    # Each changed or false claim refuses without publishing.
    buffer = {"kind": "buffer", "weight": weight, "dtype": "bfloat16", "shape": [2, 2]}
    # A constant without a lifted name never reaches the byte check, alone or beside a buffer.
    plain, unnamed = {"kind": "constant", "name": "c_module_table"}, {**buffer, "kind": "constant",
                                                                       "file": "weights.safetensors"}
    for index, (case, message) in enumerate([
        ({"payload": reference[:-1] + b"\x01"}, "differs from its declared source tensor"),
        ({"row": {"weight": "module.absent"}}, "differs from its declared source tensor"),
        ({"row": {"dtype": "float32"}}, "differs from its declared source tensor"),
        ({"stored": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [8, 0]}}, "differs from its declared source"),
        ({"stored": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, "8"]}}, "differs from its declared source"),
        ({"stored": {"dtype": "F16", "shape": [2, 2], "data_offsets": [0, 8]}}, "safetensors shape or dtype differs"),
        ({"row": {"weight": [weight]}}, "unknown manifest fields"),
        ({"row": {"shape": [4, 1]}}, "shape differs from parsed function"),
        ({"others": [buffer]}, "shares its declared source tensor"),
        ({"first": plain, "others": [unnamed]}, "not bound through its receipt-checked copy"),
        ({"first": plain, "others": [unnamed, buffer]}, "not bound through its receipt-checked copy"),
        ({"name": "x"}, "invalid constant declaration"),
        ({"name": "c_t\u00e4ble"}, "invalid constant declaration"),
    ]):
        with pytest.raises(ValueError, match=message):
            stage_compile_inputs(build(tmp_path / f"case{index}", **case), tmp_path / f"refused{index}")
        assert not (tmp_path / f"refused{index}").exists()


def test_typed_stage_reader_preserves_identity_and_refuses_forged_abi(tmp_path):
    capture, staged = (tmp_path / "capture", tmp_path / "compiler")
    _write_capture(capture)
    manifest = stage_compile_inputs(capture, staged)
    loaded = load_compile_inputs(staged)
    assert loaded.entry == "forward"
    assert loaded.original_source_sha256 == hashlib.sha256(SOURCE.encode()).hexdigest()
    assert loaded.program_sha256 == hashlib.sha256(loaded.program.encode()).hexdigest()
    assert loaded.precision_policy is None
    signature = json.loads((staged / "signature.json").read_bytes())
    signature["inputs"][0]["shape"] = [7]
    raw = json.dumps(signature).encode()
    (staged / "signature.json").write_bytes(raw)
    manifest["members"]["signature.json"] = hashlib.sha256(raw).hexdigest()
    (staged / "compile-inputs.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="inputs differ from parsed function"):
        load_compile_inputs(staged)


def _registered_translation(tmp_path, patch):
    from merlin.compiler.semantic.linalg.linalg_bridge import LinalgBridgeError, translate_linalg_text
    from merlin.compiler.semantic.model import SemanticNode, TensorType

    source = """builtin.module {
      func.func @forward(%value: tensor<3xi8>) -> (tensor<3xi8>, tensor<i32>) {
        %codes, %exponent = "fixture.encode"(%value) : (tensor<3xi8>) -> (tensor<3xi8>, tensor<i32>)
        return %codes, %exponent : tensor<3xi8>, tensor<i32>
      }
    }"""

    def translate(op, operands, results):
        return (
            SemanticNode(results[0], "encoded_codes", operands, TensorType((3,), "i8", "signed-i8")),
            SemanticNode(results[1], "encoded_exponent", operands, TensorType((), "i32", "i32-wrap-k-ascending")),
        )

    translated = translate_linalg_text(
        source, entry="forward", target_identity="fixture", source_operation_translators={"fixture.encode": translate}
    )
    assert translated.request.outputs == ("op0r0", "op0r1")
    assert tuple(node.op for node in translated.request.nodes) == ("input", "encoded_codes", "encoded_exponent")
    assert tuple(node.inputs for node in translated.request.nodes[1:]) == (("arg0",), ("arg0",))
    assert translated.request.node("op0r1").type == TensorType((), "i32", "i32-wrap-k-ascending")
    assert translated.source_operations == ("fixture.encode", "func.return")

    # The same registration contract refuses a callback changing a result type.

    source = """builtin.module {
      func.func @forward(%value: tensor<3xi8>) -> tensor<3xi8> {
        %result = "fixture.identity"(%value) : (tensor<3xi8>) -> tensor<3xi8>
        return %result : tensor<3xi8>
      }
    }"""

    def translate(op, operands, results):
        return (SemanticNode(results[0], "identity", operands, TensorType((3,), "i32", "i32-wrap-k-ascending")),)

    with pytest.raises(LinalgBridgeError, match="result types"):
        translate_linalg_text(
            source,
            entry="forward",
            target_identity="fixture",
            source_operation_translators={"fixture.identity": translate},
        )


def test_input_staging_imports_in_a_fresh_process():
    _check_import_order("inputs", "signature")
    _check_import_order("signature", "inputs")


def _check_import_order(first, second):
    subprocess.run(
        [
            sys.executable,
            "-c",
            f"import merlin.compiler.frontend.{first}; import merlin.compiler.frontend.{second}",
        ],
        env={**os.environ, "PYTHONPATH": str(module_source_path("merlin").parent.parent)},
        check=True,
        capture_output=True,
        text=True,
    )


def _bf16_literals(directory, patch):
    from merlin.compiler.semantic.linalg.linalg_bridge import LinalgBridgeError, translate_linalg_text

    source = """builtin.module {
      func.func @forward() -> (tensor<2xbf16>, tensor<3xbf16>) {
        %one = arith.constant 1.5 : bf16
        %negative_zero = arith.constant -0.0 : bf16
        %empty = tensor.empty() : tensor<2xbf16>
        %filled = linalg.fill ins(%one : bf16) outs(%empty : tensor<2xbf16>) -> tensor<2xbf16>
        %splat = tensor.splat %negative_zero : tensor<3xbf16>
        return %filled, %splat : tensor<2xbf16>, tensor<3xbf16>
      }
    }"""
    translated = translate_linalg_text(source, entry="forward", target_identity="fixture")
    assert tuple(translated.constants[key].elements for key in translated.request.outputs) == (
        (0x3FC0, 0x3FC0),
        (0x8000, 0x8000, 0x8000),
    )
    assert tuple(binding.data_hex for binding in translated.request.constants) == ("c03fc03f", "008000800080")
    with pytest.raises(LinalgBridgeError, match="finite typed literal"):
        translate_linalg_text(source.replace("1.5 : bf16", "0x7F80 : bf16"), entry="forward", target_identity="fixture")


def _gather_sigmoid(directory, patch):
    from merlin.compiler.semantic.linalg.linalg_bridge import LinalgBridgeError, translate_linalg_text

    source = """builtin.module {
      func.func @forward(%table: tensor<2x4xbf16>) -> tensor<2x2xbf16> {
        %indices = arith.constant dense<[1, 2]> : tensor<2xi64>
        %empty = tensor.empty() : tensor<2x2xbf16>
        %gather = linalg.generic {
          indexing_maps = [affine_map<(d0,d1)->(d1)>, affine_map<(d0,d1)->(d0,d1)>],
          iterator_types = ["parallel", "parallel"]
        } ins(%indices : tensor<2xi64>) outs(%empty : tensor<2x2xbf16>) {
          ^bb0(%i: i64, %unused: bf16):
            %row = linalg.index 0 : index
            %col = arith.index_cast %i : i64 to index
            %v = tensor.extract %table[%row, %col] : tensor<2x4xbf16>
            linalg.yield %v : bf16
        } -> tensor<2x2xbf16>
        %out = linalg.generic {
          indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>],
          iterator_types = ["parallel", "parallel"]
        } ins(%gather : tensor<2x2xbf16>) outs(%empty : tensor<2x2xbf16>) attrs = {
          merlin.numeric_semantics = "torch-cpu-bf16-sigmoid-v1",
          merlin.input_domain = "non-nan-bf16"
        } {
          ^bb0(%a: bf16, %unused: bf16):
            %wide = arith.extf %a : bf16 to f32
            %one = arith.constant 1.0 : f32
            %neg = arith.negf %wide : f32
            %exp = math.exp %neg : f32
            %sum = arith.addf %one, %exp : f32
            %div = arith.divf %one, %sum : f32
            %r = arith.truncf %div : f32 to bf16
            linalg.yield %r : bf16
        } -> tensor<2x2xbf16>
        return %out : tensor<2x2xbf16>
      }
    }"""
    graph = translate_linalg_text(source, entry="forward", target_identity="fixture").request
    gather = next(node for node in graph.nodes if node.op == "gather")
    assert gather.inputs == ("arg0",)
    assert gather.index_maps[0].apply((0, 0)) == (0, 1)
    assert gather.index_maps[0].apply((1, 1)) == (1, 2)
    result = graph.node(graph.outputs[0])
    assert result.op == "bf16_sigmoid" and result.inputs == (gather.id,)
    assert dict(result.attrs) == {"input_domain": "non-nan-bf16", "numeric_semantics": "torch-cpu-bf16-sigmoid-v1"}
    with pytest.raises(LinalgBridgeError, match="tensor bounds"):
        translate_linalg_text(
            source.replace("dense<[1, 2]>", "dense<[3, 4]>"), entry="forward", target_identity="fixture"
        )
    with pytest.raises(LinalgBridgeError):
        translate_linalg_text(
            source.replace("math.exp %neg", "math.exp %wide"), entry="forward", target_identity="fixture"
        )


# ---- Source translation: one MLIR text per row, translated through the public bridge ----

def _scaled(returned="%result", types="tensor<2x3x4xf32>"):
    return f'''module {{
func.func @work(%x: tensor<2x3x4xf32>) -> ({types}) {{
  %c = arith.constant 1.250000e-01 : f32
  %scale = tensor.splat %c : tensor<2x3x4xf32>
  %empty = tensor.empty() : tensor<2x3x4xf32>
  %result = linalg.generic {{
    indexing_maps = [affine_map<(i,j,k)->(i,j,k)>, affine_map<(i,j,k)->(i,j,k)>, affine_map<(i,j,k)->(i,j,k)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  }} ins(%x, %scale : tensor<2x3x4xf32>, tensor<2x3x4xf32>) outs(%empty : tensor<2x3x4xf32>) {{
    ^bb0(%a: f32, %b: f32, %old: f32):
      %value = arith.mulf %a, %b : f32
      linalg.yield %value : f32
  }} -> tensor<2x3x4xf32>
  func.return {returned} : {types}
}}
}}'''


def _bits(value):
    return unpack("<I", pack("<f", value))[0]


def _splat_elementwise(directory, patch):
    translated = translate_linalg_text(_scaled(), entry="work", target_identity="synthetic")
    request = translated.request
    (binding,) = request.constants
    literal = request.node(binding.node_id)
    assert (binding.encoding, binding.data_hex) == ("f32-le-raw", "0000003e")
    assert literal.type.shape == (1, 1, 1)
    product = request.node(request.outputs[0])
    assert product.op == "f32_mul" and product.inputs[1] == literal.id
    assert product.index_maps[1] == IndexMap(3, ((0, 0, 0),) * 3, (0, 0, 0))
    assert KernelRequest.from_record(request.record()).digest() == request.digest()
    # Nonzero inputs; scaling by a power of two is exact in binary32 here.
    values = [(-1) ** i * (i + 1) * 1.5 for i in range(24)]
    (result,) = evaluate_graph(request, {"arg0": TensorValue(request.node("arg0").type, tuple(map(_bits, values)))},
                               constants=translated.constants)
    assert result.elements == tuple(_bits(value * 0.125) for value in values)
    record, _ = emit_float_arithmetic_c(request, "scale")
    assert record["inputs"] == ["arg0", literal.id]
    # The same splat stays dense once it is returned.
    translated = translate_linalg_text(_scaled("%result, %scale", "tensor<2x3x4xf32>, tensor<2x3x4xf32>"),
                                       entry="work", target_identity="synthetic")
    (binding,) = translated.request.constants
    assert binding.data_hex == "0000003e" * 24
    assert translated.request.node(binding.node_id).type.shape == (2, 3, 4)


def _splat_padded(directory, patch):
    # Asymmetric zero padding: the insertion starts at offset zero, so its placement equals the identity.
    source = '''module {
func.func @work(%x: tensor<1x2x3x3xbf16>) -> tensor<1x2x4x4xbf16> {
  %c = arith.constant 0.000000e+00 : bf16
  %zero = tensor.splat %c : tensor<1x2x4x4xbf16>
  %padded = "tensor.insert_slice"(%x, %zero) <{static_offsets = array<i64: 0, 0, 0, 0>,
    static_sizes = array<i64: 1, 2, 3, 3>, static_strides = array<i64: 1, 1, 1, 1>,
    operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (tensor<1x2x3x3xbf16>, tensor<1x2x4x4xbf16>) -> tensor<1x2x4x4xbf16>
  func.return %padded : tensor<1x2x4x4xbf16>
}
}'''
    translated = translate_linalg_text(source, entry="work", target_identity="synthetic")
    request = translated.request
    padded = request.node(request.outputs[0])
    destination = request.node(padded.inputs[1])
    assert padded.op == "insert_slice" and destination.type.shape == (1, 2, 4, 4)
    assert padded.index_maps[1] == padded.index_maps[2]
    values = tuple(0x3f80 + 3 * i for i in range(18))  # nonzero BF16 bits
    (result,) = evaluate_graph(request, {"arg0": TensorValue(request.node("arg0").type, values)},
                               constants=translated.constants)
    expected = [0] * 32
    for i in range(18):
        channel, row, column = i // 9, i // 3 % 3, i % 3
        expected[channel * 16 + row * 4 + column] = values[i]
    assert result.elements == tuple(expected)


def _closed_decompositions(directory, patch):
    from merlin.compiler.semantic.linalg.linalg_bridge import LinalgBridgeError

    source = (repo_root() / "tests/data/host_compositions/norm_gelu_bf16.mlir").read_text()

    def ops(text):
        return [n.op for n in translate_linalg_text(text, entry="forward", target_identity="fixture").request.nodes
                if n.effect == "pure"]

    graph = translate_linalg_text(source, entry="forward", target_identity="fixture").request
    norm, tanh_gelu, erf_gelu = (graph.node(key) for key in ("op41r0_s0", "op47r0_s0", "op53r0_s0"))
    assert [n.op for n in graph.nodes if n.effect == "pure"] == ["bf16_layer_norm", "bf16_gelu", "bf16_gelu"]
    assert graph.outputs == (tanh_gelu.id, erf_gelu.id) and not graph.constants
    assert norm.inputs == ("arg2", "arg0", "arg1") and dict(norm.attrs) == {
        "affine_operands": "weight,bias", "epsilon_f32_bits": 0x358637BD, "input_domain": "finite-bf16",
        "normalized_rank": 1, "numeric_semantics": "torch-2.11-cpu-fma-bf16-layer-norm-v1"}
    assert tanh_gelu.inputs == (norm.id,) and dict(tanh_gelu.attrs)["approximate"] == "tanh"
    assert erf_gelu.inputs == ("arg2",) and dict(erf_gelu.attrs)["approximate"] == "none"
    # A changed literal or operation, or an escaping intermediate, keeps the original nodes or refuses.
    escape = ("func.return %121, %143 : tensor<2x3x8xbf16>, tensor<2x3x8xbf16>",
              "-> (tensor<2x3x8xbf16>, tensor<2x3x8xbf16>) {")
    mutations = (
        ("4.471500e-02", "4.471600e-02", None),  # tanh coefficient: no admitted f32 body
        ("0.707106769", "0.707106709", None),  # erf scale: no admitted f32 body
        ("%74 = math.rsqrt %72", "%74 = math.exp %72", (0, 2)),
        ("%24 = arith.constant 8.000000e+00", "%24 = arith.constant 7.000000e+00", (0, 2)),  # mean divisor
        ("1.000000e-06 : f32", "-1.000000e-06 : f32", (0, 2)),  # negative epsilon
        ("%46 = arith.constant 0.000000e+00", "%46 = arith.constant 1.000000e+00", (0, 2)),  # nonzero seed
        (escape[0], "func.return %35, %121, %143 : tensor<2x3x8xf32>, tensor<2x3x8xbf16>, tensor<2x3x8xbf16>", (0, 2)),
        (escape[0], "func.return %104, %121, %143 : tensor<2x3x8xf32>, tensor<2x3x8xbf16>, tensor<2x3x8xbf16>", (1, 1)),
    )
    for old, new, expected in mutations:
        assert old in source, old
        text = source.replace(old, new, 1)
        if new.startswith("func.return %"):
            text = text.replace(escape[1], "-> (tensor<2x3x8xf32>, tensor<2x3x8xbf16>, tensor<2x3x8xbf16>) {", 1)
        if expected is None:
            with pytest.raises(LinalgBridgeError):
                ops(text)
            continue
        translated = ops(text)
        assert (translated.count("bf16_layer_norm"), translated.count("bf16_gelu")) == expected, old


def test_source_translation_keeps_typed_literal_and_composition_semantics(tmp_path, monkeypatch):
    run_rows(tmp_path, monkeypatch, {
        "splat_elementwise": _splat_elementwise,
        "splat_padded": _splat_padded,
        "bf16_literals": _bf16_literals,
        "gather_sigmoid": _gather_sigmoid,
        "registered_translation": _registered_translation,
        "closed_decompositions": _closed_decompositions,
    })
