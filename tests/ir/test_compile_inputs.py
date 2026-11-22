import hashlib
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
from corefixtures.compile_input_fixtures import SOURCE, _write_capture, _write_weighted_capture

from merlin.common.paths import module_source_path
from merlin.compiler.frontend.inputs import load_compile_inputs, stage_compile_inputs


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
        application_adapter_for_prepared_source=lambda *args, **kwargs: pytest.fail(
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


def test_registered_source_translation_preserves_ordered_typed_results(tmp_path):
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

    _check_staged_application_source_hooks(tmp_path)


def _check_staged_application_source_hooks(tmp_path):
    from dataclasses import replace

    from merlin.compiler.lower.c.parameters import parameter_source_identity
    from merlin.compiler.semantic.linalg.linalg_bridge import LinalgBridgeError
    from merlin.compiler.semantic.model import SemanticNode, TensorType
    from merlin.compiler.semantic.source_preparation import verify_source_preparation
    from merlin.development.targetgen.execution.native.native_application import _source

    capture, stage = tmp_path / "custom_capture", tmp_path / "custom_stage"
    _write_weighted_capture(capture)
    original = (capture / "model.mlir").read_text()
    custom = original.replace(
        "func.return %input : tensor<2x2xf32>",
        '%unused = "fixture.combine"(%weight, %input) : '
        "(tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>\n"
        '        %result = "fixture.combine"(%weight, %input) : '
        "(tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>\n"
        "        func.return %result : tensor<2x2xf32>",
    )
    assert custom != original
    (capture / "model.mlir").write_text(custom)
    trace = json.loads((capture / "frontend-trace.json").read_text())
    trace["mlir"]["sha256"] = hashlib.sha256(custom.encode()).hexdigest()
    (capture / "frontend-trace.json").write_text(json.dumps(trace))
    metadata = json.loads((capture / "meta.json").read_text())
    metadata["frontend_trace"]["sha256"] = hashlib.sha256((capture / "frontend-trace.json").read_bytes()).hexdigest()
    (capture / "meta.json").write_text(json.dumps(metadata))
    receipt = json.loads((capture / "capture_receipt.json").read_text())
    for member in ("model.mlir", "frontend-trace.json", "meta.json"):
        raw = (capture / member).read_bytes()
        receipt["artifacts"][member] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    (capture / "capture_receipt.json").write_text(json.dumps(receipt))
    stage_compile_inputs(capture, stage)
    args = SimpleNamespace(
        compile_inputs=stage, mode="diagnostic", linalg_entry=None, application_parameters="external"
    )
    adapter = object()
    received = []

    def translate(op, operands, results):
        return (SemanticNode(results[0], "fixture_combined", operands, TensorType((2, 2), "f32", "raw-bit-pattern")),)

    def prepared_factory(policy, *, original_source_sha256, source, staged, compile_inputs):
        assert policy is None and original_source_sha256 == hashlib.sha256(custom.encode()).hexdigest()
        assert compile_inputs == stage
        assert source.source_identity == parameter_source_identity(staged.program_sha256, staged)
        assert tuple(n.op for n in source.nodes) == ("input", "input", "fixture_combined")
        assert source.node(source.outputs[0]).inputs == ("arg0", "arg1")
        received.append(source)
        return adapter

    binding = SimpleNamespace(
        application_source_translators=lambda policy: {"fixture.combine": translate},
        application_adapter_for_prepared_source=prepared_factory,
    )
    staged, source, selected, state, folding, parameters = _source(args, binding, "fixture")
    assert selected is adapter and received == [source] and state is None
    assert parameters.node_ids == ("arg0",)
    parameters.check(source)
    unprepared = verify_source_preparation(source, folding)
    assert tuple(node.op for node in unprepared.nodes).count("fixture_combined") == 2

    # A registration supplies semantics only within the original typed boundary.
    def changed_type(op, operands, results):
        return (replace(translate(op, operands, results)[0], type=TensorType((2, 2), "i32", "i32-wrap-k-ascending")),)

    with pytest.raises(LinalgBridgeError, match="result types"):
        _source(
            args,
            SimpleNamespace(
                application_source_translators=lambda policy: {"fixture.combine": changed_type},
                application_adapter_for_prepared_source=prepared_factory,
            ),
            "fixture",
        )
    assert received == [source]

    # Legacy bindings retain their factory signature; missing hooks cannot make
    # an unrecognized source operation silently enter ordinary application intake.
    def legacy_factory(policy, *, original_source_sha256):
        assert policy is None and original_source_sha256 == staged.original_source_sha256
        return adapter

    with pytest.raises(LinalgBridgeError):
        _source(args, SimpleNamespace(application_adapter=legacy_factory), "fixture")
    _, legacy_source, legacy_adapter, _, _, legacy_parameters = _source(
        args,
        SimpleNamespace(
            application_adapter=legacy_factory, application_source_translators=binding.application_source_translators
        ),
        "fixture",
    )
    assert legacy_source == source and legacy_adapter is adapter and legacy_parameters == parameters


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


def test_bf16_literals_preserve_fill_splat_bits_and_refuse_nonfinite():
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


def test_constant_window_gather_and_declared_sigmoid_preserve_source_contract():
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
