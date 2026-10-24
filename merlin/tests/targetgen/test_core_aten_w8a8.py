"""Opt-in W8A8 numeric contracts, reference independence and honest grading."""

import numpy as np
import pytest
from xdsl.parser import Parser

from merlin.frontends.linalg_mlir import make_context
from merlin.llvmlower.contraction_quantization import SPEC, quantize_module, selected_contractions

QUANT = dict(quant_min=SPEC.quant_min, quant_max=SPEC.quant_max, eps=SPEC.eps)


def module(body=None, init="0.0"):
    contraction = (
        body
        or """%r = linalg.matmul ins(%a, %b : tensor<2x3xf32>, tensor<3x2xf32>)
        outs(%z : tensor<2x2xf32>) -> tensor<2x2xf32>"""
    )
    return Parser(
        make_context(),
        f"""builtin.module {{
      func.func @forward(%a: tensor<2x3xf32>, %b: tensor<3x2xf32>) -> tensor<2x2xf32> {{
        %c = arith.constant {init} : f32
        %e = tensor.empty() : tensor<2x2xf32>
        %z = linalg.fill ins(%c : f32) outs(%e : tensor<2x2xf32>) -> tensor<2x2xf32>
        {contraction}
        func.return %r : tensor<2x2xf32>
      }}
    }}""",
    ).parse_module()


def select(m):
    """The contractions selected for the i8 x i8 -> i32 capability at rank 2."""
    return list(selected_contractions(m, [("i8", "i8", "i32")], (2,)))


def resized(**shapes):
    """``module()`` with its ``2x3x``/``3x2x``/``2x2x`` operand shapes replaced (keys a, b, r)."""
    text = str(module())
    for old, key in (("2x3x", "a"), ("3x2x", "b"), ("2x2x", "r")):
        text = text.replace(old, shapes.get(key, old))
    return Parser(make_context(), text).parse_module()


def test_opt_in_preserves_float_abi_and_uses_existing_lowering():
    m = module()
    original_type = next(op for op in m.walk() if op.name == "func.func").function_type
    chosen = select(m)
    assert len(chosen) == 1
    assert quantize_module(m, chosen) == 1
    assert next(op for op in m.walk() if op.name == "func.func").function_type == original_type
    names = [op.name for op in m.walk()]
    assert names.count("math.roundeven") == 2
    assert names.count("arith.extsi") == 2
    assert names.count("arith.sitofp") >= 1
    m.verify()


@pytest.mark.parametrize(
    "triples,ranks,init",
    [
        ([], (2,), "0.0"),
        ([("f32", "f32", "f32")], (2,), "0.0"),
        ([("i8", "i8", "i32")], (3,), "0.0"),
        ([("i8", "i8", "i32")], (2,), "1.0"),
    ],
)
def test_refuse_unsupported_capability_or_live_initializer(triples, ranks, init):
    m = module(init=init)
    before = str(m)
    assert list(selected_contractions(m, triples, ranks)) == []
    assert str(m) == before


def test_reference_is_integer_pytorch_and_dequantizes_in_order():
    torch = pytest.importorskip("torch")
    from merlin.targetgen.core_aten_w8a8_reference import evaluate

    m = module()
    chosen = select(m)
    a = np.array([[0, 0.5, -1], [1, 0.25, 0]], np.float32)
    b = np.array([[1, -1], [0.5, 0.25], [0, -0.5]], np.float32)
    outputs, scales = evaluate(m, [a, b], chosen, **QUANT)
    s = torch.tensor(1 / 127, dtype=torch.float32)
    qa, qb = [torch.round(torch.from_numpy(x) * (1 / s)).to(torch.int32) for x in (a, b)]
    expected = ((qa @ qb).float().double() * (s.double() * s.double())).float()
    assert torch.equal(outputs[0], expected)
    assert scales[0]["lhs"] == scales[0]["rhs"] == s.item()
    assert scales[0]["accumulator"] == (qa @ qb).tolist()


def test_exact_reference_grade_does_not_use_fp32_error():
    from merlin.targetgen.core_aten_batch_grade import grade_core_aten_batch

    document = dict(kind="tensor", dtype="float32", shape=[1], values=[1.0])
    record = dict(
        overload="aten.mm.default",
        status="bundled",
        output_indices=[0],
        output_abi=[dict(dtype="f32", shape=[1])],
        numeric_policy="int8_w8a8",
        scales=[dict(lhs=0.1, rhs=0.2)],
        case=dict(expected=document, comparison_parameters=dict(rtol=0, atol=0)),
        fp32_case=dict(expected={**document, "values": [2.0]}),
    )
    from merlin.targetgen.core_aten_numeric import policy_record

    record.update(
        w8a8_policy=policy_record(),
        w8a8_accumulators=[dict(index=1, shape=[1])],
        w8a8_reference_bytes=[np.array([1.0], np.float32).tobytes().hex()],
    )
    record["scales"][0]["accumulator"] = [50]
    raw_acc = np.array([50], np.int32).tobytes()
    batch = dict(cases=[record], provenance={"test": True})
    verdict = grade_core_aten_batch(batch, [np.array([1.0], np.float32).tobytes(), raw_acc])["cases"]["aten.mm.default"]
    assert verdict["status"] == "pass"
    assert verdict["fp32_error"][0]["max_abs"] == 1
    assert verdict["numeric_policy"] == "int8_w8a8"
    raw = np.array([np.nextafter(np.float32(1), np.float32(2))], np.float32).tobytes()
    assert grade_core_aten_batch(batch, [raw, raw_acc])["cases"]["aten.mm.default"]["status"] == "mismatch"


@pytest.mark.parametrize("element", ["f16", "f64"])
def test_float_storage_policy_restores_original_dtype(element):
    m = Parser(make_context(), str(module()).replace("f32", element)).parse_module()
    original_type = next(op for op in m.walk() if op.name == "func.func").function_type
    selected = select(m)
    assert quantize_module(m, selected) == 1
    assert next(op for op in m.walk() if op.name == "func.func").function_type == original_type
    m.verify()


def test_non_mac_reduction_and_overflow_are_refused():
    # Keep matmul maps but replace the product with a difference: geometry is insufficient.
    m = module()
    from xdsl.dialects import arith
    from xdsl.rewriter import Rewriter

    mul = next(op for op in m.walk() if isinstance(op, arith.MulfOp))
    Rewriter.replace_op(mul, arith.SubfOp(*mul.operands))
    assert not select(m)
    huge = resized(a="2x200000x", b="200000x2x")
    assert not select(huge)


def test_zero_tensors_use_existing_epsilon_policy():
    torch = pytest.importorskip("torch")
    from merlin.targetgen.core_aten_w8a8_reference import evaluate

    m = module()
    selected = select(m)
    output, scales = evaluate(m, [np.zeros((2, 3), np.float32), np.zeros((3, 2), np.float32)], selected, **QUANT)
    assert torch.equal(output[0], torch.zeros((2, 2)))
    assert scales[0]["lhs"] == np.float32(SPEC.eps)
    assert scales[0]["rhs"] == np.float32(SPEC.eps)


def test_half_operands_with_float_accumulation_are_quantized():
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import FunctionType, TensorType, f16, f32

    m = module()
    op = next(op for op in m.walk() if op.name == "linalg.matmul")
    # Match the frontend's mixed-precision body: extend inputs, then f32 MAC.
    for src in op.operands[:2]:
        src._type = TensorType(f16, src.type.get_shape())
    block = op.body.block
    mul = next(inner for inner in block.ops if isinstance(inner, arith.MulfOp))
    for argument in block.args[:2]:
        argument._type = f16
    casts = [arith.ExtFOp(arg, f32) for arg in block.args[:2]]
    for cast in casts:
        block.insert_op_before(cast, mul)
    mul.operands = tuple(cast.result for cast in casts)
    function = next(inner for inner in m.walk() if inner.name == "func.func")
    function.function_type = FunctionType.from_lists(
        [arg.type for arg in function.body.block.args], [op.results[0].type]
    )
    chosen = select(m)
    assert chosen == [op]
    assert quantize_module(m, chosen) == 1


def numeric_spec():
    from merlin.common.paths import repo_root
    from merlin.targetgen.software_spec import load_software_spec

    return load_software_spec(repo_root() / "examples/gemmini/target/software-spec.yaml")


def test_declared_range_changes_partial_sum_bound_and_refuses_unsplit_k():
    from merlin.targetgen.core_aten_numeric import validate_primitive_dispatch
    from merlin.targetgen.core_aten_w8a8_reference import validate_dispatch
    from merlin.targetgen.operation_numerics import integer_split_k_limit

    spec = numeric_spec()
    bound = integer_split_k_limit(spec, tile_dim=1, full_k=64, operand_range=(SPEC.quant_min, SPEC.quant_max))
    assert bound["max_true_k"] == 32
    assert integer_split_k_limit(spec, tile_dim=1, full_k=64)["max_true_k"] == 31
    with pytest.raises(ValueError, match="primitive K"):
        validate_primitive_dispatch(spec, full_k=64, primitive_ks=[64])
    unsafe = dict(full_k=64, primitive_ks=[64], numerical_semantics=spec["numerical_semantics"])
    with pytest.raises(ValueError, match="primitive K"):
        validate_dispatch(unsafe, full_k=64, quant_min=SPEC.quant_min, quant_max=SPEC.quant_max)
    safe = validate_primitive_dispatch(spec, full_k=64, primitive_ks=[16])
    assert validate_dispatch(safe, full_k=64, quant_min=SPEC.quant_min, quant_max=SPEC.quant_max) == 32
    # An unsplit same-sign witness wraps even though its final i32 result fits.
    bits = bound["mac_result_bits"]
    acc = 64 * SPEC.quant_max**2
    wrapped = ((acc + (1 << (bits - 1))) % (1 << bits)) - (1 << (bits - 1))
    assert acc != wrapped
    pytest.importorskip("torch")
    from merlin.targetgen.core_aten_w8a8_reference import evaluate

    m = resized(a="2x64x", b="64x2x")
    chosen = select(m)
    arrays = [np.ones((2, 64), np.float32), np.ones((64, 2), np.float32)]
    with pytest.raises(ValueError, match="primitive K"):
        evaluate(m, arrays, chosen, **QUANT, dispatch=[unsafe])
    _, sites = evaluate(m, arrays, chosen, **QUANT, dispatch=[safe])
    assert sites[0]["accumulator"][0][0] == acc
    assert sites[0]["k_split"]["split_required"]
    assert sites[0]["accumulator_metrics"]["internal_headroom"] > 0


def test_gate_distinguishes_accumulator_errors_signed_zero_and_nan_payloads():
    from merlin.targetgen.core_aten_numeric import exact_w8a8_gate

    plus = np.array([0.0], np.float32).tobytes()
    acc = np.array([0], np.int32).tobytes()
    record = dict(
        scales=[dict(accumulator=[0])],
        output_indices=[0],
        w8a8_accumulators=[dict(index=1, shape=[1])],
        w8a8_reference_bytes=[plus.hex()],
    )
    assert exact_w8a8_gate(record, [plus, acc])["passed"]
    assert not exact_w8a8_gate(record, [np.array([-0.0], np.float32).tobytes(), acc])["passed"]
    assert not exact_w8a8_gate(record, [plus, np.array([1], np.int32).tobytes()])["passed"]
    nan = np.array([np.nan], np.float32).tobytes()
    record["w8a8_reference_bytes"] = [nan.hex()]
    assert exact_w8a8_gate(record, [nan, acc])["passed"]
    different = np.array([0x7FC00001], np.uint32).tobytes()
    assert not exact_w8a8_gate(record, [different, acc])["passed"]


def test_fp32_error_complete_metrics_and_zero_relative_infinity():
    from merlin.targetgen.core_aten_numeric import fp32_error

    record = dict(
        output_indices=[0],
        output_abi=[dict(dtype="f32")],
        fp32_case=dict(expected=dict(kind="tensor", values=[0.0, 2.0])),
        scales=[dict(lhs=0.25, rhs=0.5)],
    )
    metrics = fp32_error(record, [np.array([1.0, 1.0], np.float32).tobytes()])[0]
    assert metrics["max_rel"] == float("inf")
    assert metrics["relative_inf_at_zero_golden"] == 1
    assert metrics["rmse"] == 1.0
    assert metrics["sqnr_db"] == pytest.approx(10 * np.log10(2))
    assert metrics["normalized_l2"] == pytest.approx(np.sqrt(0.5))
    assert metrics["error_in_output_lsbs"][0]["max_abs"] == 8


def native_model(m, root):
    import subprocess

    from merlin.llvmlower import toolchain
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.codegen import mlir_runtime_c
    from merlin.llvmlower.lower import lower_model_file

    root.mkdir(parents=True, exist_ok=True)
    source = root / "model.mlir"
    source.write_text(str(m))
    lowered = lower_model_file(source, root / "lower", targets=(), textual=True)
    obj, shared, clang = root / "model.o", root / "model.so", str(toolchain.clang())
    subprocess.run([clang, "-O2", "-fPIC", "-c", str(lowered.ll_path), "-o", str(obj)], check=True, capture_output=True)
    link = [clang, "-shared", "-Wl,-Bsymbolic", "-fPIC", str(obj), str(mlir_runtime_c()), "-lm", "-o", str(shared)]
    subprocess.run(link, check=True, capture_output=True)
    return HostModel.load(str(shared))


def test_emitted_dequant_overflow_and_nonfinite_runtime(tmp_path):
    pytest.importorskip("torch")
    from merlin.targetgen.core_aten_w8a8_reference import evaluate

    m = module()
    selected = select(m)
    original = m.clone()
    original_selected = select(original)
    quantize_module(m, selected)
    native = native_model(m, tmp_path / "dequant")
    for a_value, b_value in [(1e38, 1e-4), (0.0, 0.0), (np.nan, 1.0), (np.inf, 1.0), (-np.inf, 1.0)]:
        a = np.full((2, 3), a_value, np.float32)
        b = np.full((3, 2), b_value, np.float32)
        actual = np.zeros((2, 2), np.float32)
        native([(v.ctypes.data, v.shape) for v in (a, b, actual)])
        outputs, _ = evaluate(original, [a, b], original_selected, **QUANT)
        assert actual.tobytes() == outputs[0].numpy().tobytes()
        if np.isfinite(a_value):
            assert np.isfinite(actual).all()
    a = np.ones((2, 3), np.float32)
    b = np.ones((3, 2), np.float32)
    a[0, 1] = np.nan
    actual = np.zeros((2, 2), np.float32)
    native([(v.ctypes.data, v.shape) for v in (a, b, actual)])
    assert np.isnan(actual).all()
    with pytest.raises(ValueError, match="not representable"):
        evaluate(
            original, [np.full((2, 3), 1e38, np.float32), np.full((3, 2), 1e38, np.float32)], original_selected, **QUANT
        )
    # Run the emitted assertion in a child so an expected abort cannot kill pytest.
    import subprocess

    from merlin.llvmlower import toolchain

    code = """import numpy as np
from merlin.llvmlower.abi import HostModel
import sys
native=HostModel.load(sys.argv[1])
a=np.full((2,3),1e38,np.float32);b=np.full((3,2),1e38,np.float32);out=np.zeros((2,2),np.float32)
native([(v.ctypes.data,v.shape) for v in (a,b,out)])
"""
    process = subprocess.run(
        [str(toolchain.compiler_python()), "-c", code, str(tmp_path / "dequant/model.so")],
        capture_output=True,
        text=True,
    )
    assert process.returncode != 0
    import signal

    assert process.returncode == -signal.SIGABRT
    assert "not representable" in (tmp_path / "dequant/lower/model.ll").read_text()


def test_exhaustive_emitted_quantizer_sweep_by_class(tmp_path):
    torch = pytest.importorskip("torch")
    from xdsl.dialects.builtin import FunctionType

    from merlin.targetgen.core_aten_w8a8_reference import quantize_codes, quantize_operand

    # Exhaust all finite binary16 values promoted to f32, plus every reduced
    # int8 tie and its f32 neighbours at several independently fixed scales.
    half = np.arange(1 << 16, dtype=np.uint16).view(np.float16)
    least, tiny = np.nextafter(np.float32(0), np.float32(1)), np.finfo(np.float32).tiny
    classes = {
        "finite_binary16": half[np.isfinite(half)].astype(np.float32),
        "subnormal": np.array([least, tiny / 2, -least, -tiny / 2], np.float32),
        "signed_zero": np.array([0.0, -0.0], np.float32),
    }
    fixed_scales = {}
    for scale in [np.float32(SPEC.eps), np.float32(0.125), np.float32(1.0)]:
        ties = (np.arange(SPEC.quant_min - 1, SPEC.quant_max + 1, dtype=np.float32) + np.float32(0.5)) * scale
        fixed_scales[f"ties_{scale}"] = scale
        classes[f"ties_{scale}"] = np.concatenate([np.nextafter(ties, -np.inf), ties, np.nextafter(ties, np.inf)])
    classes["eps_clamp"] = np.linspace(-SPEC.eps, SPEC.eps, 4097, dtype=np.float32)
    classes["all_zero"] = np.zeros(8, np.float32)
    mismatches = {}
    for name, values in classes.items():
        x = values.reshape(1, -1).copy()
        k = x.size
        m = resized(a=f"1x{k}x", b=f"{k}x1x", r="1x1x")
        chosen = select(m)
        quantize_module(m, chosen)
        contraction = next(op for op in m.walk() if "merlin.numeric_policy" in op.attributes)
        codes = contraction.operands[0]
        # Observe the exact quantizer emitted by quantize_module, including
        # sanitization and dynamic scale computation, through native LLVM.
        qop = codes.owner
        reciprocal = qop.operands[1].owner
        scale = reciprocal.operands[0]
        if name in fixed_scales:
            from xdsl.dialects import arith, tensor
            from xdsl.dialects.builtin import FloatAttr, TensorType, f32

            scalar = arith.ConstantOp(FloatAttr(float(fixed_scales[name]), f32))
            tensor_scale = tensor.SplatOp(scalar.result, [], TensorType(f32, []))
            function_block = reciprocal.parent_block()
            function_block.insert_op_before(scalar, reciprocal)
            function_block.insert_op_before(tensor_scale, reciprocal)
            scale = tensor_scale.result
            reciprocal.operands = (scale, *reciprocal.operands[1:])
        function = next(op for op in m.walk() if op.name == "func.func")
        returned = next(op for op in function.body.block.ops if op.name == "func.return")
        returned.operands = (scale, codes)
        function.function_type = FunctionType.from_lists(list(function.function_type.inputs), [scale.type, codes.type])
        native = native_model(m, tmp_path / name)
        actual_q, actual_s = np.zeros(x.shape, np.int8), np.zeros((), np.float32)
        b = np.zeros((k, 1), np.float32)
        native([(v.ctypes.data, v.shape) for v in (x, b, actual_s, actual_q)])
        q, s, _, _ = quantize_operand(torch.from_numpy(x), **QUANT)
        if name in fixed_scales:
            s = torch.tensor(fixed_scales[name], dtype=torch.float32)
            q = quantize_codes(torch.from_numpy(x), s, quant_min=SPEC.quant_min, quant_max=SPEC.quant_max)
        mismatches[name] = dict(
            samples=x.size,
            codes=int(np.count_nonzero(actual_q != q.numpy())),
            scale=int(actual_s.tobytes() != s.numpy().tobytes()),
        )
    import json

    (tmp_path / "quantizer-sweep.json").write_text(json.dumps(mismatches, indent=2))
    assert all(v["codes"] == v["scale"] == 0 for v in mismatches.values()), mismatches


def test_catalog_selection_screens_actual_emitted_k(tmp_path, monkeypatch):
    import hashlib
    import json
    from types import SimpleNamespace

    from xdsl.ir import Dialect

    from merlin.common.paths import repo_root
    from merlin.targetgen import core_aten_numeric, plugins, software_spec, target_experiment

    spec = numeric_spec()
    spec["numerical_semantics"]["internal_arithmetic"]["primitive_k_observation"] = dict(
        operation="fixture.mac", extent_attribute="k", dialect_module="fixture", dialect_symbol="D"
    )
    monkeypatch.setattr(target_experiment, "descriptor_for", lambda target: "fixture")
    monkeypatch.setattr(
        target_experiment, "load_target_experiment", lambda descriptor: SimpleNamespace(numeric_profile="fixture")
    )
    monkeypatch.setattr(
        software_spec,
        "software_spec_path_for_recipe",
        lambda path: repo_root() / "examples/gemmini/target/software-spec.yaml",
    )
    monkeypatch.setattr(software_spec, "load_software_spec", lambda path, target: spec)
    monkeypatch.setattr(plugins, "load_module", lambda *args, **kwargs: SimpleNamespace(D=Dialect("fixture", [], [])))
    completed = []
    monkeypatch.setattr(core_aten_numeric, "complete_w8a8_reference", lambda *args: completed.append(args[-1]))
    report = dict(cases=[dict(status="bundled", w8a8_selected_ordinals=[0])])

    def builder(prepared, workdir):
        ir = workdir / "kernel.mlir"
        ir.write_text(
            'builtin.module { llvm.func @kernel() { "fixture.mac"() {k = 64 : i64} : () -> () llvm.return } }'
        )
        obj = workdir / "kernel.o"
        obj.write_bytes(b"fixture")
        manifest = workdir / "catalog.json"
        manifest.write_text(
            json.dumps(
                dict(
                    compilation=dict(target_ir_sha256=hashlib.sha256(ir.read_bytes()).hexdigest()),
                    kernels=[dict(symbol="kernel", dimensions=dict(k=64))],
                    bindings=[dict(symbol="kernel", region="w8a8_contraction_0")],
                )
            )
        )
        return manifest, obj

    checked = core_aten_numeric.checked_w8a8_builder(
        builder, target="fixture", package=tmp_path, directory=tmp_path, report=report
    )
    with pytest.raises(ValueError, match="primitive K"):
        checked(tmp_path / "source.mlir", tmp_path)
    assert completed == []
