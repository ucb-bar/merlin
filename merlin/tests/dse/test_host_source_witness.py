"""Independent source-chain witness semantics; no candidate or target dependencies."""
import numpy as np
import pytest

from merlin.perf.host_source_witness import extract_pointwise_chain, evaluate_pointwise_source


@pytest.mark.parametrize("mutation", [None, "missing", "top_only", "stale", "partial", "duplicate", "unknown"])
def test_cached_host_task_activity_requires_complete_bound_rows_without_parser(monkeypatch, mutation):
    from merlin.frontends import linalg_mlir
    from merlin.perf.host_region_qualifier import cached_host_task_activity

    def forbidden(*args, **kwargs):
        raise AssertionError("short task selection must not parse a full model")

    monkeypatch.setattr(linalg_mlir, "parse_mlir_text", forbidden)
    rows = [{"task": str(index), "load_payload_bytes": index, "store_payload_bytes": 1}
            for index in range(8)]
    plan = {"status": "verified", "tasks": 8, "candidate_lowered_sha256": "a" * 64,
            "host_activity": {"status": "derived", "artifact_sha256": "a" * 64,
                "task_activity_coverage": "complete", "tasks": rows,
                "top_tasks_by_scalar_memory_payload": rows[:5]}}
    if mutation == "missing":
        del plan["host_activity"]["tasks"]
    elif mutation == "top_only":
        plan["host_activity"]["tasks"] = rows[:5]
    elif mutation == "stale":
        plan["host_activity"]["artifact_sha256"] = "b" * 64
    elif mutation == "partial":
        plan["host_activity"]["task_activity_coverage"] = "partial"
    elif mutation == "duplicate":
        rows.append(dict(rows[0]))
    elif mutation == "unknown":
        rows[-1]["load_payload_bytes"] = None
    if mutation:
        with pytest.raises(ValueError, match="cached host task"):
            cached_host_task_activity(plan, lowered_sha256="a" * 64)
    else:
        assert len(cached_host_task_activity(plan, lowered_sha256="a" * 64)) == 8


def quantized_readout_source(extra_multiply=False):
    extra = "%one = arith.constant 1.0 : f32\n%scaled = arith.mulf %product, %one : f32" if extra_multiply else ""
    result = "%scaled" if extra_multiply else "%product"
    return '''builtin.module { func.func @forward(%acc:tensor<4xi32>, %scale:tensor<f32>) -> tensor<4xi8> {
%ef = tensor.empty() : tensor<4xf32>
%f = linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>, affine_map<(d0)->()>, affine_map<(d0)->(d0)>], iterator_types=["parallel"]} ins(%acc, %scale:tensor<4xi32>,tensor<f32>) outs(%ef:tensor<4xf32>) {
^bb0(%a:i32,%s:f32,%unused:f32):
%cast = arith.sitofp %a : i32 to f32
%product = arith.mulf %cast, %s : f32
''' + extra + '\nlinalg.yield ' + result + ''' : f32
} -> tensor<4xf32>
%ei = tensor.empty() : tensor<4xi8>
%q = linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>], iterator_types=["parallel"]} ins(%f:tensor<4xf32>) outs(%ei:tensor<4xi8>) {
^bb1(%v:f32,%unused:i8):
%rounded = math.roundeven %v : f32
%lo = arith.constant -128.0 : f32
%hi = arith.constant 127.0 : f32
%lower = arith.maximumf %rounded, %lo : f32
%upper = arith.minimumf %lower, %hi : f32
%out = arith.fptosi %upper : f32 to i8
linalg.yield %out : i8
} -> tensor<4xi8>
func.return %q : tensor<4xi8>
}}'''


def readout_capability():
    return {"schema": "scalar_narrow_readout_contract_v1", "accumulator_dtype": "i32",
            "output_dtype": "i8", "scale_dtype": "f32", "clamp_min": -128, "clamp_max": 127,
            "provenance": {"fixture": "explicit typed scalar instruction contract"}}


def test_quantized_epilogue_opt_in_preserves_even_ties_and_saturation():
    source = quantized_readout_source()
    with pytest.raises(ValueError, match="no supported"):
        extract_pointwise_chain(source, range(4))
    probe, receipt = extract_pointwise_chain(source, range(4), max_extent=4, mechanism="quantized_epilogue")
    assert receipt["source_indices"] == [1, 3]
    values = np.asarray([1, 3, -1, -3], dtype=np.int32)
    result = evaluate_pointwise_source(probe, [values, np.asarray(.5, dtype=np.float32)],
                                       allow_quantized_epilogue=True)[0]
    np.testing.assert_array_equal(result, [0, 2, 0, -2])
    saturated = evaluate_pointwise_source(probe, [np.asarray([1000, -1000, 127, -128], dtype=np.int32),
                                                 np.asarray(1, dtype=np.float32)],
                                          allow_quantized_epilogue=True)[0]
    np.testing.assert_array_equal(saturated, [127, -128, 127, -128])


def test_readout_eligibility_matches_def_use_and_refuses_reassociation_and_stale_source():
    from merlin.perf.host_source_witness import assess_narrow_readout_equivalence
    source = quantized_readout_source()
    probe, receipt = extract_pointwise_chain(source, range(4), max_extent=1, mechanism="quantized_epilogue")
    result = assess_narrow_readout_equivalence(probe, receipt, capability=readout_capability())
    assert result["status"] == "eligible_exact_scalar_DAG"
    assert result["performance_promotion"] is False
    with pytest.raises(ValueError, match="stale"):
        assess_narrow_readout_equivalence(probe + "\n", receipt, capability=readout_capability())
    changed, binding = extract_pointwise_chain(quantized_readout_source(True), range(4),
                                                max_extent=1, mechanism="quantized_epilogue")
    refusal = assess_narrow_readout_equivalence(changed, binding, capability=readout_capability())
    assert refusal["status"] == "refused_exact_native_readout"
    assert "multiple_rounded_float_scalings_cannot_be_reassociated" in refusal["reasons"]
    # Same opcode sequence, wrong multiplication operand: a sequence-only matcher would pass.
    reordered = source.replace("arith.mulf %cast, %s", "arith.mulf %s, %cast")
    changed, binding = extract_pointwise_chain(reordered, range(4), max_extent=1, mechanism="quantized_epilogue")
    assert assess_narrow_readout_equivalence(changed, binding, capability=readout_capability())["status"] == "refused_exact_native_readout"


def bounded_gather_model():
    return '''builtin.module { func.func @forward(%input: tensor<5xf32>) -> tensor<f32> {
%zero = arith.constant 0.0 : f32
%pad = tensor.splat %zero : tensor<7xf32>
%insert = "tensor.insert_slice"(%input, %pad) <{static_offsets = array<i64: 1>, static_sizes = array<i64: 5>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (tensor<5xf32>, tensor<7xf32>) -> tensor<7xf32>
%e = tensor.empty() : tensor<2x3xf32>
%g = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel", "parallel"]} outs(%e:tensor<2x3xf32>) {
^bb0(%unused:f32):
%i = linalg.index 0 : index
%j = linalg.index 1 : index
%two = arith.constant 2 : index
%jj = arith.muli %j, %two : index
%ij = arith.addi %i, %jj : index
%v = tensor.extract %insert[%ij] : tensor<7xf32>
linalg.yield %v : f32
} -> tensor<2x3xf32>
%te = tensor.empty() : tensor<3x2xf32>
%t = linalg.transpose ins(%g:tensor<2x3xf32>) outs(%te:tensor<3x2xf32>) permutation = [1, 0]
%pe = tensor.empty() : tensor<3x2xf32>
%p = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel", "parallel"]} ins(%t:tensor<3x2xf32>) outs(%pe:tensor<3x2xf32>) {
^bb1(%x:f32, %unused:f32):
%v = arith.addf %x, %x : f32
linalg.yield %v : f32
} -> tensor<3x2xf32>
%reduce_zero = arith.constant 0.0 : f32
%initial = tensor.splat %reduce_zero : tensor<f32>
%sum = linalg.reduce ins(%p:tensor<3x2xf32>) outs(%initial:tensor<f32>) dimensions = [0,1]
(%x:f32, %acc:f32) { %v = arith.addf %x, %acc : f32
linalg.yield %v : f32 }
func.return %sum : tensor<f32>
}}'''


def test_bounded_gather_is_proper_source_subgraph_with_original_padding_and_layout():
    from merlin.perf.host_source_witness import extract_bounded_gather
    probe, record = extract_bounded_gather(bounded_gather_model(), range(12))
    values = np.asarray([-0., 1., -2., 3., 4.], dtype=np.float32)
    padded = np.pad(values, (1,1))
    gathered = np.asarray([[padded[i+2*j] for j in range(3)] for i in range(2)]).T
    expected = np.add(gathered, gathered, dtype=np.float32)
    actual = evaluate_pointwise_source(probe, [values])[0]
    assert actual.tobytes() == expected.tobytes()
    assert record["gather_source_index"] == 4
    assert record["inputs"] == [{"shape": [5], "dtype": "f32"}]
    assert record["all_source_intermediate_uses_preserved"]
    assert "linalg.reduce" not in probe


def test_bounded_gather_refuses_large_or_whole_source_and_external_fanout():
    from merlin.perf.host_source_witness import extract_bounded_gather
    source = bounded_gather_model()
    with pytest.raises(ValueError, match="no bounded proper"):
        extract_bounded_gather(source, range(12), max_elements=4)
    # Even an otherwise unrelated outside use of a cloned pad scalar must not disappear.
    # The valid fixture uses a separate reduction initializer constant.
    fanout = source.replace("%initial = tensor.splat %reduce_zero", "%initial = tensor.splat %zero")
    with pytest.raises(ValueError, match="no bounded proper"):
        extract_bounded_gather(fanout, range(12))
    whole = source.partition("%reduce_zero")[0].replace(
        "-> tensor<f32> {", "-> tensor<3x2xf32> {", 1) + "func.return %p : tensor<3x2xf32>\n}}"
    with pytest.raises(ValueError, match="no bounded proper"):
        extract_bounded_gather(whole, range(9))


def test_insert_slice_witness_preserves_source_derived_padding_and_values():
    from merlin.perf.host_source_witness import extract_insert_slice
    probe, receipt = extract_insert_slice(bounded_gather_model(), range(12), max_extent=3)
    assert receipt["source_indices"] == [2]
    assert receipt["source_source_shape"] == [5]
    assert receipt["source_destination_shape"] == [7]
    assert receipt["probe_source_shape"] == [3]
    assert receipt["probe_destination_shape"] == [5]
    assert receipt["probe_offsets"] == [1]
    assert receipt["probe_inserted_payload_bytes"] == 12
    values = np.asarray([-2.0, 3.0, -0.0], dtype=np.float32)
    expected = np.asarray([0.0, -2.0, 3.0, -0.0, 0.0], dtype=np.float32)
    assert evaluate_pointwise_source(probe, [values])[0].tobytes() == expected.tobytes()


def test_insert_slice_witness_refuses_nonunit_stride_and_unowned_source_index():
    from merlin.perf.host_source_witness import extract_insert_slice
    source = bounded_gather_model()
    with pytest.raises(ValueError, match="no supported static insert slice"):
        extract_insert_slice(source.replace("static_strides = array<i64: 1>",
                                            "static_strides = array<i64: 2>"), range(12))
    with pytest.raises(ValueError, match="outside source function"):
        extract_insert_slice(source, [-1])


def chain(dtype, first, second):
    operations = []
    for index, scalar in enumerate((first, second)):
        incoming = "%input" if index == 0 else "%p0"
        operations.append(f'''%e{index} = tensor.empty() : tensor<3x3x{dtype}>
%p{index} = linalg.generic {{indexing_maps = [affine_map<(d0,d1)->(d1,d0)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel", "parallel"]}} ins({incoming}:tensor<3x3x{dtype}>) outs(%e{index}:tensor<3x3x{dtype}>) {{
^bb0(%x: {dtype}, %unused: {dtype}):
{scalar}
linalg.yield %v : {dtype}
}} -> tensor<3x3x{dtype}>''')
    return (f"builtin.module {{ func.func @forward(%input:tensor<3x3x{dtype}>) -> tensor<3x3x{dtype}> {{\n"
            + "\n".join(operations) + f"\nfunc.return %p1:tensor<3x3x{dtype}>\n}}}}")


def test_modular_signed_chain_and_composed_maps():
    source = chain("i8", "%c = arith.constant 3 : i8\n%v = arith.muli %x, %c : i8",
                   "%c = arith.constant 5 : i8\n%v = arith.addi %x, %c : i8")
    probe, receipt = extract_pointwise_chain(source, [0, 1, 2, 3])
    values = np.asarray([127, -128, -1, 85, -86, 0, 42, -42, 1], dtype=np.int8).reshape(3, 3)
    expected = (((values.astype(np.int32)*3 + 5 + 128) % 256)-128).astype(np.int8)
    np.testing.assert_array_equal(evaluate_pointwise_source(probe, [values])[0], expected)
    assert receipt["source_indices"] == [1, 3]
    assert receipt["source_sha256"] != receipt["probe_source_sha256"]


def test_f32_rounding_preserved_between_regions():
    source = chain("f32", "%c = arith.constant 1.000000e+00 : f32\n%v = arith.addf %x, %c : f32",
                   "%c = arith.constant 1.000000e+00 : f32\n%v = arith.subf %x, %c : f32")
    probe, _ = extract_pointwise_chain(source, [0, 1, 2, 3])
    values = np.asarray([2**-25, 2**-24, -2**-24, 0, 1, -1, .5, -.5, .75], dtype=np.float32).reshape(3, 3)
    expected = (values + np.float32(1)).astype(np.float32)-np.float32(1)
    actual = evaluate_pointwise_source(probe, [values])[0]
    np.testing.assert_array_equal(actual, expected)
    assert actual[0, 0] == 0 and values[0, 0] != 0


def test_extraction_refuses_unowned_source_indices_and_reduction():
    source = chain("i8", "%v = arith.addi %x, %x : i8", "%v = arith.addi %x, %x : i8")
    with pytest.raises(ValueError, match="outside"):
        extract_pointwise_chain(source, [-1, 20])
    with pytest.raises(ValueError, match="no supported"):
        extract_pointwise_chain(source.replace('["parallel", "parallel"]', '["parallel", "reduction"]'), [0, 1, 2, 3])


def test_fanout_preserves_all_consumers_and_multiple_outputs():
    source = chain("i8", "%v = arith.addi %x, %x : i8", "%v = arith.addi %x, %x : i8")
    source = source.replace("-> tensor<3x3xi8> {", "-> (tensor<3x3xi8>, tensor<3x3xi8>) {", 1)
    branch = '''%e2 = tensor.empty() : tensor<3x3xi8>
%p2 = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d1,d0)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel", "parallel"]} ins(%p0:tensor<3x3xi8>) outs(%e2:tensor<3x3xi8>) {
^bb0(%x:i8, %unused:i8):
%c = arith.constant 1 : i8
%v = arith.subi %x, %c : i8
linalg.yield %v : i8
} -> tensor<3x3xi8>
func.return %p1, %p2 : tensor<3x3xi8>, tensor<3x3xi8>'''
    source = source.replace("func.return %p1:tensor<3x3xi8>", branch)
    probe, record = extract_pointwise_chain(source, list(range(6)), mechanism="fanout")
    assert record["fanout"]["uses"] == [[3, 0], [5, 0]]
    assert record["fanout"]["all_source_root_uses_preserved"]
    values = np.asarray([127, -128, -1, 85, -86, 0, 42, -42, 1], dtype=np.int8).reshape(3, 3)
    observed = evaluate_pointwise_source(probe, [values])
    wrap = lambda value: (((value+128) % 256)-128).astype(np.int8)
    np.testing.assert_array_equal(observed[0], wrap(values.astype(np.int32)*4))
    np.testing.assert_array_equal(observed[1], wrap(values.astype(np.int32)*2-1))
    with pytest.raises(ValueError, match="no supported complete-use"):
        extract_pointwise_chain(source, [0, 1, 2, 3], mechanism="fanout")


POINTWISE_CONCAT = '''builtin.module {
  func.func @forward(%left: tensor<5x7xf32>, %right: tensor<5x2xf32>) -> tensor<5x9xf32> {
    %empty = tensor.empty() : tensor<5x7xf32>
    %negative = linalg.generic {
        indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>],
        iterator_types = ["parallel", "parallel"]}
      ins(%left : tensor<5x7xf32>) outs(%empty : tensor<5x7xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %negated = arith.negf %value : f32
      linalg.yield %negated : f32
    } -> tensor<5x7xf32>
    %joined = "tensor.concat"(%negative, %right) <{dim = 1 : i64}>
      : (tensor<5x7xf32>, tensor<5x2xf32>) -> tensor<5x9xf32>
    func.return %joined : tensor<5x9xf32>
  }
}'''


POINTWISE_IDENTITY_CONCAT = '''builtin.module {
  func.func @forward(%input: tensor<5x7xf32>) -> tensor<5x7xf32> {
    %empty = tensor.empty() : tensor<5x7xf32>
    %negative = linalg.generic {
        indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>],
        iterator_types = ["parallel", "parallel"]}
      ins(%input : tensor<5x7xf32>) outs(%empty : tensor<5x7xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %negated = arith.negf %value : f32
      linalg.yield %negated : f32
    } -> tensor<5x7xf32>
    %joined = "tensor.concat"(%negative) <{dim = 1 : i64}>
      : (tensor<5x7xf32>) -> tensor<5x7xf32>
    func.return %joined : tensor<5x7xf32>
  }
}'''


def test_pointwise_concat_clones_actual_axis_segments_and_exact_scalar_semantics():
    from merlin.perf.host_source_witness import extract_pointwise_concat

    probe, receipt = extract_pointwise_concat(
        POINTWISE_CONCAT, [0, 1, 2], max_extent=3, max_elements=64)

    assert receipt["source_indices"] == [1, 2]
    assert receipt["concat_axis"] == 1
    assert receipt["concat_operand_count"] == 2
    assert receipt["identity_concat"] is False
    assert receipt["source_operand_shapes"] == [[5, 7], [5, 2]]
    assert receipt["probe_operand_shapes"] == [[3, 3], [3, 2]]
    assert receipt["probe_result_shape"] == [3, 5]
    assert receipt["probe_intermediate_payload_bytes"] == 36
    assert receipt["bounded_tensor_elements"] == 39
    assert receipt["all_source_producer_uses_preserved"] is True
    left = np.asarray([
        [0.0, -0.0, 2**-24],
        [1.0, -1.0, 2**20],
        [-2**-20, 127.0, -127.0],
    ], dtype=np.float32)
    right = np.asarray([[4.0, -4.0], [5.0, -5.0], [6.0, -6.0]], dtype=np.float32)
    expected = np.concatenate((np.negative(left, dtype=np.float32), right), axis=1)
    actual = evaluate_pointwise_source(probe, [left, right])[0]
    assert actual.tobytes() == expected.tobytes()


def test_pointwise_identity_concat_clones_and_evaluates_exact_scalar_semantics():
    from merlin.perf.host_source_witness import extract_pointwise_concat

    probe, receipt = extract_pointwise_concat(
        POINTWISE_IDENTITY_CONCAT, [0, 1, 2], max_extent=3, max_elements=32)

    assert receipt["source_indices"] == [1, 2]
    assert receipt["concat_axis"] == 1
    assert receipt["concat_operand_count"] == 1
    assert receipt["identity_concat"] is True
    assert receipt["source_operand_shapes"] == [[5, 7]]
    assert receipt["probe_operand_shapes"] == [[3, 3]]
    assert receipt["probe_result_shape"] == [3, 3]
    assert receipt["probe_intermediate_payload_bytes"] == 36
    assert receipt["bounded_tensor_elements"] == 27
    values = np.asarray([
        [0.0, -0.0, 2**-24],
        [1.0, -1.0, 2**20],
        [-2**-20, 127.0, -127.0],
    ], dtype=np.float32)
    actual = evaluate_pointwise_source(probe, [values])[0]
    assert actual.tobytes() == np.negative(values, dtype=np.float32).tobytes()


def test_pointwise_concat_refuses_shared_producer_and_oversized_probe():
    from merlin.perf.host_source_witness import extract_pointwise_concat

    shared = POINTWISE_CONCAT.replace(
        "tensor<5x9xf32>", "tensor<5x16xf32>").replace(
        '"tensor.concat"(%negative, %right)',
        '"tensor.concat"(%negative, %negative, %right)').replace(
        ": (tensor<5x7xf32>, tensor<5x2xf32>) -> tensor<5x16xf32>",
        ": (tensor<5x7xf32>, tensor<5x7xf32>, tensor<5x2xf32>) -> tensor<5x16xf32>")
    with pytest.raises(ValueError, match="sole-use pointwise-to-concat"):
        extract_pointwise_concat(shared, [0, 1, 2])
    with pytest.raises(ValueError, match="tensor-element budget"):
        extract_pointwise_concat(
            POINTWISE_CONCAT, [0, 1, 2], max_extent=3, max_elements=38)
DEQUANT_CONTRACTION = '''builtin.module {
  func.func @forward(%w: tensor<3x4xi8>, %s: tensor<4xf32>, %z: tensor<4xi32>,
                     %a: tensor<1x3xf32>, %initial: tensor<1x4xf32>) -> tensor<1x4xf32> {
    %dq = "quant_ext.dequantize_per_channel"(%w, %s, %z) <{axis = 1 : i64, input_dtype = "i8"}>
      : (tensor<3x4xi8>, tensor<4xf32>, tensor<4xi32>) -> tensor<3x4xf32>
    %out = linalg.matmul ins(%a, %dq : tensor<1x3xf32>, tensor<3x4xf32>)
      outs(%initial : tensor<1x4xf32>) -> tensor<1x4xf32>
    func.return %out : tensor<1x4xf32>
  }
}'''


def test_actual_dequant_contraction_keeps_no_recomputation_geometry_and_f32_order():
    from merlin.perf.host_source_witness import extract_dequant_contraction, evaluate_pointwise_source
    import numpy as np
    text, info = extract_dequant_contraction(DEQUANT_CONTRACTION, [0, 1])
    assert info["source_indices"] == [0, 1]
    assert info["source_geometry_mkn"] == [1, 3, 4]
    assert info["probe_geometry_mkn"] == [1, 3, 3]
    assert info["recomputation_factor"] == 1
    inputs = [np.array([[-128, 1, 127], [0, 3, -3], [17, -1, 8]], dtype=np.int8),
              np.array([.1, 2**-24, -.25], dtype=np.float32),
              np.array([128, -3, 127], dtype=np.int32),
              np.array([[-.5, 2**20, 2**-24]], dtype=np.float32),
              np.array([[-.25, 17, -128]], dtype=np.float32)]
    wanted = inputs[-1].copy()
    for j in range(3):
        for k in range(3):
            dq = np.float32(np.float32(np.float32(inputs[0][k, j]) - np.float32(inputs[2][j])) * inputs[1][j])
            wanted[0, j] = np.float32(wanted[0, j] + np.float32(inputs[3][0, k] * dq))
    assert evaluate_pointwise_source(text, inputs)[0].tobytes() == wanted.tobytes()


def test_dequant_source_extractor_refuses_amplification_and_unsigned_reinterpretation():
    import pytest
    from merlin.perf.host_source_witness import extract_dequant_contraction
    for text in (DEQUANT_CONTRACTION.replace("1x3xf32", "2x3xf32").replace("1x4xf32", "2x4xf32"),
                 DEQUANT_CONTRACTION.replace('input_dtype = "i8"', 'input_dtype = "ui8"')):
        with pytest.raises(ValueError, match="no exact sole-use"):
            extract_dequant_contraction(text, [0, 1])


POINTWISE_REDUCTION = '''builtin.module {
  func.func @forward(%input: tensor<5x4xf32>) -> tensor<5xf32> {
    %empty = tensor.empty() : tensor<5x4xf32>
    %squared = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,
        affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel", "parallel"]}
      ins(%input: tensor<5x4xf32>) outs(%empty: tensor<5x4xf32>) {
    ^bb0(%x: f32, %unused: f32):
      %square = arith.mulf %x, %x : f32
      linalg.yield %square : f32
    } -> tensor<5x4xf32>
    %zero = arith.constant 0.000000e+00 : f32
    %initial = tensor.splat %zero : tensor<5xf32>
    %result = linalg.reduce ins(%squared: tensor<5x4xf32>) outs(%initial: tensor<5xf32>) dimensions = [1]
      (%value: f32, %acc: f32) {
        %sum = arith.addf %value, %acc : f32
        linalg.yield %sum : f32
      }
    func.return %result : tensor<5xf32>
  }
}'''


def test_pointwise_reduction_preserves_bodies_initialization_and_iteration_order():
    from merlin.perf.host_source_witness import extract_pointwise_reduction
    text, receipt = extract_pointwise_reduction(POINTWISE_REDUCTION, list(range(5)))
    assert receipt["source_indices"] == [1, 4]
    assert receipt["source_shape"] == [5, 4] and receipt["probe_shape"] == [3, 3]
    assert receipt["producer_result_scalar_operation"] == "arith.mulf"
    assert receipt["reduction_result_scalar_operation"] == "arith.addf"
    assert receipt["reduction_dimensions"] == [1]
    assert receipt["probe_intermediate_payload_bytes"] == 36
    values = np.array([[1, 2**-12, -1], [-2, .5, 2**-20], [127, 2**-10, -127]], dtype=np.float32)
    expected = np.zeros(3, dtype=np.float32)
    for i in range(3):
        for j in range(3):
            expected[i] = np.float32(np.float32(values[i,j]*values[i,j])+expected[i])
    assert evaluate_pointwise_source(text, [values])[0].tobytes() == expected.tobytes()


GENERIC_REDUCTION = '''builtin.module {
  func.func @forward(%lhs: tensor<5x1x4xf32>, %rhs: tensor<5x4x6xf32>) -> tensor<5x1x6xf32> {
    %zero = arith.constant 0.000000e+00 : f32
    %initial = tensor.splat %zero : tensor<5x1x6xf32>
    %result = linalg.generic {indexing_maps = [
        affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>,
        affine_map<(d0,d1,d2,d3)->(d0,d3,d2)>,
        affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>],
        iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%lhs, %rhs : tensor<5x1x4xf32>, tensor<5x4x6xf32>)
      outs(%initial : tensor<5x1x6xf32>) {
    ^bb0(%left: f32, %right: f32, %acc: f32):
      %product = arith.mulf %left, %right : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<5x1x6xf32>
    func.return %result : tensor<5x1x6xf32>
  }
}'''


NAMED_REDUCTION = '''builtin.module {
  func.func @forward(%input: tensor<5x4xi8>) -> tensor<5xi8> {
    %minimum = arith.constant -128 : i8
    %initial = tensor.splat %minimum : tensor<5xi8>
    %result = linalg.reduce ins(%input : tensor<5x4xi8>)
      outs(%initial : tensor<5xi8>) dimensions = [1]
      (%value: i8, %acc: i8) {
        %maximum = arith.maxsi %value, %acc : i8
        linalg.yield %maximum : i8
      }
    func.return %result : tensor<5xi8>
  }
}'''


def test_named_reduction_preserves_initializer_body_and_source_dimension_order():
    from merlin.perf.host_source_witness import extract_named_reduction
    text, receipt = extract_named_reduction(NAMED_REDUCTION, [0, 1, 2])
    assert receipt["source_indices"] == [2]
    assert receipt["source_input_shape"] == [5, 4]
    assert receipt["probe_input_shape"] == [3, 3]
    assert receipt["reduction_dimensions"] == [1]
    assert receipt["probe_output_payload_bytes"] == 3
    values = np.asarray([
        [-128, -7, 1], [127, -1, 0], [-12, -11, -10]], dtype=np.int8)
    np.testing.assert_array_equal(
        evaluate_pointwise_source(text, [values])[0], np.max(values, axis=1))


def test_named_reduction_refuses_unordered_dimensions_and_unsupported_body():
    from merlin.perf.host_source_witness import extract_named_reduction
    unordered = NAMED_REDUCTION.replace("tensor<5x4xi8>", "tensor<5x4x3xi8>").replace(
        "tensor<5xi8>", "tensor<5xi8>").replace("dimensions = [1]", "dimensions = [2, 1]")
    with pytest.raises(ValueError, match="no supported direct named reduction"):
        extract_named_reduction(unordered, [0, 1, 2])
    unsupported = NAMED_REDUCTION.replace("arith.maxsi %value, %acc", "arith.divsi %value, %acc")
    with pytest.raises(ValueError, match="no supported direct named reduction"):
        extract_named_reduction(unsupported, [0, 1, 2])


def test_generic_reduction_preserves_maps_initializer_and_lexicographic_accumulation():
    from merlin.perf.host_source_witness import extract_generic_reduction
    text, receipt = extract_generic_reduction(GENERIC_REDUCTION, [0, 1, 2])
    assert receipt["source_indices"] == [2]
    assert receipt["source_iteration_shape"] == [5, 1, 6, 4]
    assert receipt["probe_iteration_shape"] == [3, 1, 3, 3]
    assert receipt["parallel_dimensions"] == [0, 1, 2]
    assert receipt["reduction_dimensions"] == [3]
    assert receipt["probe_output_payload_bytes"] == 36
    lhs = np.asarray([
        [[1, 2**-12, -1]], [[-2, .5, 2**-20]], [[127, 2**-10, -127]]], dtype=np.float32)
    rhs = np.asarray([
        [[.5, -1, 2], [3, 2**-20, -4], [-.25, 5, 1]],
        [[2, -2, .5], [1, 3, -1], [2**-12, 4, -3]],
        [[-1, 2, 3], [.25, -4, 2], [1, .5, -2]]], dtype=np.float32)
    expected = np.zeros((3, 1, 3), dtype=np.float32)
    for d0 in range(3):
        for d1 in range(1):
            for d2 in range(3):
                for d3 in range(3):
                    expected[d0, d1, d2] = np.float32(
                        expected[d0, d1, d2] + np.float32(lhs[d0, d1, d3] * rhs[d0, d3, d2]))
    assert evaluate_pointwise_source(text, [lhs, rhs])[0].tobytes() == expected.tobytes()


def test_unchanged_dequant_probe_is_not_changed_region_qualification():
    from merlin.perf.host_region_qualifier import reduced_materialization_change
    activity = {"status": "derived", "static_allocation_payload_bytes": 12,
                "load_payload_bytes": 80, "store_payload_bytes": 40}
    extraction = {"mechanism": "dequant_contraction", "probe_intermediate_payload_bytes": 36}
    assert reduced_materialization_change(activity, dict(activity), extraction)["status"] == "NO_RELEVANT_REDUCED_CHANGE"
    before = {**activity, "static_allocation_payload_bytes": 48, "load_payload_bytes": 152, "store_payload_bytes": 76}
    proof = reduced_materialization_change(before, activity, {**extraction, "mechanism": "pointwise_reduction"})
    assert proof["status"] == "changed_reduced_materialization"
    assert proof["deleted_payload_bytes"]["static_allocation_payload_bytes"] == 36
    assert reduced_materialization_change({**before, "load_payload_bytes": None}, activity, extraction)["status"] == "UNKNOWN"


@pytest.mark.parametrize("mutation,expected", [
    (None, "changed_reduced_integer_codegen"),
    ("same_artifact", "NO_RELEVANT_REDUCED_CHANGE"),
    ("missing_counts", "NO_RELEVANT_REDUCED_CHANGE"),
    ("floating_change", "NO_RELEVANT_REDUCED_CHANGE"),
])
def test_integer_only_reduced_change_can_exercise_source_semantics(mutation, expected):
    from copy import deepcopy
    from merlin.perf.host_region_qualifier import reduced_materialization_change

    before = {"status": "derived", "static_allocation_payload_bytes": 0,
        "load_payload_bytes": 24, "store_payload_bytes": 12,
        "dynamic_operations": {"integer_arithmetic": 40, "floating_arithmetic": 3}}
    after = deepcopy(before)
    after["dynamic_operations"]["integer_arithmetic"] = 12
    after_sha = "b" * 64
    if mutation == "same_artifact":
        after_sha = "a" * 64
    elif mutation == "missing_counts":
        del after["dynamic_operations"]["integer_arithmetic"]
    elif mutation == "floating_change":
        after["dynamic_operations"]["floating_arithmetic"] = 1
    evidence = reduced_materialization_change(before, after, {"mechanism": "chain"},
        before_lowered_sha256="a" * 64, after_lowered_sha256=after_sha)
    assert evidence["status"] == expected
    if mutation is None:
        assert evidence["integer_operations_before_after"] == [40, 12]
        assert "not machine work" in evidence["scope"]


def test_host_task_relevance_follows_source_not_renumbered_task_ids():
    from merlin.perf.host_region_qualifier import changed_host_task_candidates

    before = [{"kind": "host", "task_index": 0, "source_op_indices": [2, 3]}]
    after = [{"kind": "host", "task_index": 7, "source_op_indices": [2, 3]}]
    row = {"load_payload_bytes": 8, "store_payload_bytes": 4,
           "dynamic_operations": {"integer_arithmetic": 40}}
    tables = [{"0": row}, {"7": {**row, "dynamic_operations": {"integer_arithmetic": 12}}}]
    choices, unresolved = changed_host_task_candidates(before, after, tables)
    assert not unresolved
    assert choices[0][1] == {**after[0], "comparison_prior_task_indices": [0]}
    assert choices[0][2] == [12, 12]
    assert choices[0][0] == (0, True, 28)
    # The same numeric task ID cannot attribute a different source region.
    after[0].update(task_index=0, source_op_indices=[4, 5])
    choices, unresolved = changed_host_task_candidates(before, after, [tables[0], tables[0]])
    assert not choices and unresolved[0]["source_op_indices"] == [4, 5]


def test_changed_region_extraction_exhausts_best_task_mechanisms_before_next_task():
    """A large model must not be reparsed for every task under one extractor first.

    The V14 qualifier grouped its search by extractor kind.  Forty-five changed TinyLlama
    tasks therefore paid a full 4.66 MB parse for every unsupported kind before the likely
    pointwise-chain extractor was reached, turning a 60-second action into a 784-second stall.
    """
    from merlin.perf import host_region_qualifier as qualifier

    best = ((100, True, 20), {"task_index": 7}, [200, 100])
    second = ((10, True, 5), {"task_index": 8}, [20, 10])
    options = list(qualifier._ranked_source_witness_options([second, best]))
    kinds = [kind for kind, _ in options[:9]]

    assert [row[1][1]["task_index"] for row in options[:9]] == [7] * 9
    assert kinds == [
        "pointwise_concat", "insert_slice", "bounded_gather", "named_reduction", "generic_reduction",
        "pointwise_reduction", "dequant_contraction", "fanout", "chain",
    ]
    assert options[9][1][1]["task_index"] == 8


def test_changed_region_extraction_budget_stops_after_highest_ranked_task():
    """An unsupported large task must produce a bounded refusal, not parse every task."""
    from merlin.perf import host_region_qualifier as qualifier

    best = ((100, True, 20), {"task_index": 7}, [200, 100])
    second = ((10, True, 5), {"task_index": 8}, [20, 10])
    options = list(qualifier._bounded_source_witness_options([second, best]))

    assert len(options) == len(qualifier._SOURCE_WITNESS_KINDS)
    assert {row[1]["task_index"] for _, row in options} == {7}


@pytest.mark.parametrize("mutation", [None, "partial_overlap", "mixed_lane"])
def test_host_task_relevance_handles_complete_task_fusion_not_partial_overlap(mutation):
    from merlin.perf.host_region_qualifier import changed_host_task_candidates

    before = [{"kind": "host", "task_index": i, "source_op_indices": [i]} for i in (0, 1)]
    after = [{"kind": "host", "task_index": 0, "source_op_indices": [0, 1]}]
    row = {"load_payload_bytes": 8, "store_payload_bytes": 8}
    if mutation == "partial_overlap":
        before[1]["source_op_indices"].append(2)
    elif mutation == "mixed_lane":
        before[1]["kind"] = "contraction"
    candidates, unresolved = changed_host_task_candidates(before, after, [{"0": row, "1": row}, {"0": row}])
    if mutation is None:
        assert not unresolved
        assert candidates[0][1]["comparison_prior_task_indices"] == [0, 1]
        assert candidates[0][2] == [32, 16]
        assert candidates[0][0] == (16, False, None)  # Missing integer counts do not become zero.
    else:
        assert not candidates and unresolved


def test_generic_reduction_codegen_change_tracks_result_buffer_not_total_stack_traffic():
    from merlin.perf.host_region_qualifier import reduced_materialization_change
    before = {"status": "derived", "static_allocation_payload_bytes": 36,
              "load_payload_bytes": 360, "store_payload_bytes": 180,
              "dynamic_operations": {"floating_arithmetic": 54, "conversion": 0},
              "buffer_payload": {
                  "alloca:3": {"load_payload_bytes": 144, "store_payload_bytes": 144}}}
    after = {"status": "derived", "static_allocation_payload_bytes": 76,
             "load_payload_bytes": 396, "store_payload_bytes": 252,
             "dynamic_operations": {"floating_arithmetic": 54, "conversion": 0},
             "buffer_payload": {
                 "alloca:3": {"load_payload_bytes": 0, "store_payload_bytes": 36},
                 "alloca:7": {"load_payload_bytes": 36, "store_payload_bytes": 36},
                 "alloca:8": {"load_payload_bytes": 144, "store_payload_bytes": 144}}}
    extraction = {"mechanism": "generic_reduction", "probe_output_payload_bytes": 36,
                  "probe_reduction_steps_per_output": 3}
    proof = reduced_materialization_change(before, after, extraction,
        before_lowered_sha256="before", after_lowered_sha256="after")
    assert proof["status"] == "changed_reduced_codegen"
    assert proof["result_buffer_accumulator_loads_removed"] is True
    assert proof["total_preoptimization_payload_reduced"] is False
    assert reduced_materialization_change(before, after, extraction,
        before_lowered_sha256="same", after_lowered_sha256="same")["status"] == "NO_RELEVANT_REDUCED_CHANGE"


def test_named_reduction_codegen_change_requires_one_final_result_store():
    from merlin.perf.host_region_qualifier import reduced_materialization_change
    before = {"status": "derived", "static_allocation_payload_bytes": 12,
              "load_payload_bytes": 36, "store_payload_bytes": 36,
              "dynamic_operations": {"floating_arithmetic": 9, "conversion": 0},
              "buffer_payload": {
                  "alloca:3": {"load_payload_bytes": 36, "store_payload_bytes": 36}}}
    after = {"status": "derived", "static_allocation_payload_bytes": 12,
             "load_payload_bytes": 12, "store_payload_bytes": 12,
             "dynamic_operations": {"floating_arithmetic": 9, "conversion": 0},
             "buffer_payload": {
                 "alloca:3": {"load_payload_bytes": 12, "store_payload_bytes": 12}}}
    extraction = {"mechanism": "named_reduction", "probe_output_payload_bytes": 12,
                  "probe_reduction_steps_per_output": 3}
    proof = reduced_materialization_change(before, after, extraction,
        before_lowered_sha256="before", after_lowered_sha256="after")
    assert proof["status"] == "changed_reduced_codegen"
    assert proof["result_buffer_accumulator_loads_removed"] is True
    assert proof["ordered_accumulator_representation"] == "ssa"
    assert reduced_materialization_change(before, after, extraction,
        before_lowered_sha256="same", after_lowered_sha256="same")["status"] == "NO_RELEVANT_REDUCED_CHANGE"


def test_insert_slice_codegen_change_requires_deleted_overwritten_payload():
    from merlin.perf.host_region_qualifier import reduced_materialization_change
    before = {"status": "derived", "static_allocation_payload_bytes": 20,
              "load_payload_bytes": 32, "store_payload_bytes": 32}
    after = {"status": "derived", "static_allocation_payload_bytes": 20,
             "load_payload_bytes": 20, "store_payload_bytes": 20}
    extraction = {"mechanism": "insert_slice", "probe_inserted_payload_bytes": 12}
    proof = reduced_materialization_change(before, after, extraction)
    assert proof["status"] == "changed_reduced_materialization"
    assert proof["deleted_overwritten_payload_bytes"] == 12
    assert reduced_materialization_change(
        before, {**after, "load_payload_bytes": 32}, extraction)["status"] == "changed_reduced_materialization"
    assert reduced_materialization_change(
        before, {**after, "store_payload_bytes": 24}, extraction)["status"] == "NO_RELEVANT_REDUCED_CHANGE"


def test_pointwise_concat_change_requires_exact_producer_materialization_deletion():
    from merlin.perf.host_region_qualifier import reduced_materialization_change

    before = {"status": "derived", "static_allocation_payload_bytes": 96,
              "load_payload_bytes": 180, "store_payload_bytes": 156}
    after = {"status": "derived", "static_allocation_payload_bytes": 60,
             "load_payload_bytes": 144, "store_payload_bytes": 120}
    extraction = {"mechanism": "pointwise_concat", "probe_intermediate_payload_bytes": 36}
    proof = reduced_materialization_change(before, after, extraction)
    assert proof["status"] == "changed_reduced_materialization"
    assert proof["expected_producer_payload_bytes"] == 36
    assert reduced_materialization_change(
        before, {**after, "static_allocation_payload_bytes": 64}, extraction
    )["status"] == "NO_RELEVANT_REDUCED_CHANGE"


def test_pointwise_concat_emission_requires_direct_scalar_to_concat_output():
    from merlin.perf.host_region_qualifier import pointwise_concat_emission_evidence

    extraction = {"schema": "actual_source_pointwise_concat_witness_v1",
                  "mechanism": "pointwise_concat",
                  "producer_result_scalar_operation": "arith.negf",
                  "scalar_region_sha256": "a" * 64,
                  "concat_axis": 1,
                  "concat_operand_count": 2,
                  "identity_concat": False,
                  "probe_operand_shapes": [[3, 3], [3, 2]],
                  "probe_result_shape": [3, 5],
                  "probe_intermediate_payload_bytes": 36,
                  "all_source_producer_uses_preserved": True}
    before = {"status": "derived", "dynamic_operations": {
        "floating_arithmetic": 9, "conversion": 0}}
    after = {"status": "derived", "dynamic_operations": {
        "floating_arithmetic": 9, "conversion": 0}}
    delta = {"status": "changed_reduced_materialization",
             "deleted_payload_bytes": {"static_allocation_payload_bytes": 36,
                                       "load_payload_bytes": 36,
                                       "store_payload_bytes": 36}}
    evidence = pointwise_concat_emission_evidence(
        extraction, before, after, emitted_operation_names=["llvm.fneg", "llvm.store"],
        direct_output=True, materialization_delta=delta)
    assert evidence["status"] == "demonstrated_changed_pointwise_in_concat"
    assert evidence["actual_result_scalar_operations"] == 1
    assert evidence["floating_arithmetic_and_conversion_counts_preserved"] is True
    assert pointwise_concat_emission_evidence(
        extraction, before, after, emitted_operation_names=["llvm.store"],
        direct_output=True, materialization_delta=delta)["status"] == "NOT_DEMONSTRATED"
    malformed_identity = {**extraction, "concat_operand_count": 1,
                          "identity_concat": True, "probe_operand_shapes": []}
    assert pointwise_concat_emission_evidence(
        malformed_identity, before, after,
        emitted_operation_names=["llvm.fneg", "llvm.store"],
        direct_output=True, materialization_delta=delta)["status"] == "NOT_DEMONSTRATED"
