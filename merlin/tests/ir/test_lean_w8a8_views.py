"""Prove and execute sliced/concatenated W8A8 inputs without inventing numerical facts."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest
from test_lean_w8a8_bounded import contract, embedding_source, qualify
from test_lean_w8a8_byte_model import evaluate
from test_lean_w8a8_native_recipe import dynamic_matmul_reference

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.passes_quant_int import lower_contraction_int8
from merlin.verify.lean import differential as diff

pytestmark = pytest.mark.usefixtures("lean_oracle")


def tensor(shape):
    return "tensor<" + "".join(f"{dim}x" for dim in shape) + "f32>"


def matmul(left, shape, columns=2):
    rows, reduction = shape
    output = tensor((rows, columns))
    return f"""%empty = tensor.empty() : {output}
      %zero = arith.constant 0.0 : f32
      %filled = linalg.fill ins(%zero : f32) outs(%empty : {output}) -> {output}
      %result = linalg.matmul ins({left}, %weights : {tensor(shape)}, {tensor((reduction, columns))})
        outs(%filled : {output}) -> {output}
      return %result : {output}"""


@dataclass(frozen=True)
class SliceCase:
    source: tuple[int, ...]
    offsets: tuple[int, ...]
    sizes: tuple[int, ...]
    strides: tuple[int, ...]
    result: tuple[int, int]

    def text(self):
        def array(values):
            return "[" + ", ".join(str(value) for value in values) + "]"

        return f"""func.func @forward(%input: {tensor(self.source)}, %weights: {tensor((self.result[1], 2))})
            -> {tensor((self.result[0], 2))} {{
          %view = tensor.extract_slice %input{array(self.offsets)}{array(self.sizes)}{array(self.strides)}
            : {tensor(self.source)} to {tensor(self.result)}
          {matmul("%view", self.result)}
        }}"""

    def reference(self, values):
        indices = tuple(
            slice(offset, offset + size * stride, stride)
            for offset, size, stride in zip(self.offsets, self.sizes, self.strides, strict=True)
        )
        return values[indices].reshape(self.result)


SLICES = [
    SliceCase((4, 8), (1, 1), (2, 3), (1, 2), (2, 3)),
    SliceCase((1, 4, 8), (0, 1, 1), (1, 2, 3), (1, 1, 2), (2, 3)),
    SliceCase((4, 1, 8), (1, 0, 1), (2, 1, 3), (1, 1, 2), (2, 3)),
    SliceCase((4, 8, 1), (1, 1, 0), (2, 3, 1), (1, 2, 1), (2, 3)),
    SliceCase((4, 8), (3, 1), (0, 3), (1, 2), (0, 3)),
    SliceCase((4, 8), (1, 7), (2, 0), (1, 1), (2, 0)),
]


def contracts(count):
    return [{"function": "forward", "intervals": [None] * count}]


def checked_rewrite(before, count, tmp_path, tools, *, index_bits=64, mutate=None):
    after = before.clone()
    assert lower_contraction_int8(after) == 1
    report, documents = qualify(before, after, contracts(count), tmp_path, tools, index_bits=index_bits, mutate=mutate)
    return report, documents


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("case", SLICES)
def test_actual_slices_feed_conditionally_initialized_w8a8(tmp_path, lean_tools, case, index_bits):
    before = parse_mlir_text(case.text())
    report, _ = checked_rewrite(before, 2, tmp_path, lean_tools, index_bits=index_bits)
    assert report["conditional_initialized_policy_refinement"], report
    assert report["functions"][0]["source"]["unbounded_initialized"]
    assert report["functions"][0]["reference"]["unbounded_initialized"]


def concat_source(shapes=((1, 3), (2, 3)), axis=0, *, constant_first=False, square_last=False):
    output_shape = list(shapes[0])
    output_shape[axis] = sum(shape[axis] for shape in shapes)
    output_shape = tuple(output_shape)
    inputs = [(f"%p{index}", shape) for index, shape in enumerate(shapes) if not (index == 0 and constant_first)]
    parameters = ", ".join(f"{name}: {tensor(shape)}" for name, shape in inputs)
    prefix = ""
    if constant_first:
        prefix = f"%one = arith.constant 1.0 : f32\n%p0 = tensor.splat %one : {tensor(shapes[0])}"
    if square_last:
        last = len(shapes) - 1
        prefix += f"""\n%dest = tensor.empty() : {tensor(shapes[last])}
          %squared = linalg.generic {{
            indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
            iterator_types = ["parallel", "parallel"]
          }} ins(%p{last} : {tensor(shapes[last])}) outs(%dest : {tensor(shapes[last])}) {{
          ^bb0(%x: f32, %old: f32):
            %r = arith.mulf %x, %x : f32
            linalg.yield %r : f32
          }} -> {tensor(shapes[last])}
        """
    operands = [f"%p{index}" for index in range(len(shapes))]
    if square_last:
        operands[-1] = "%squared"
    return parse_mlir_text(f"""func.func @forward({parameters}, %weights: {tensor((output_shape[1], 2))})
        -> {tensor((output_shape[0], 2))} {{
      {prefix}
      %view = tensor.concat dim({axis}) {", ".join(operands)}
        : ({", ".join(tensor(shape) for shape in shapes)}) -> {tensor(output_shape)}
      {matmul("%view", output_shape)}
    }}""")


CONCATS = [(((1, 3), (2, 3)), 0), (((2, 1), (2, 0), (2, 2)), 1), (((0, 3), (2, 3), (0, 3)), 0)]


@pytest.mark.parametrize("shapes,axis", CONCATS)
@pytest.mark.parametrize("constant_first", [False, True])
def test_concat_preserves_checked_common_finiteness(tmp_path, lean_tools, shapes, axis, constant_first):
    before = concat_source(shapes, axis, constant_first=constant_first)
    count = len(shapes) + (not constant_first)
    report, _ = checked_rewrite(before, count, tmp_path, lean_tools)
    assert report["conditional_initialized_policy_refinement"], report
    assert report["functions"][0]["reference"]["unbounded_initialized"]


def test_concat_cannot_copy_first_inputs_finiteness_over_unproved_arithmetic(tmp_path, lean_tools):
    before = concat_source(square_last=True)
    report, _ = checked_rewrite(before, 3, tmp_path, lean_tools)
    assert not report["conditional_initialized_policy_refinement"]
    assert not report["functions"][0]["reference"]["initialized"]


def compare_outputs(document, samples, expected, work, fidelity_tools):
    proc, lean = evaluate(document, samples, work)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    inputs, outputs = diff.signature(json.loads(document.read_text()), "forward")
    diff.assert_equal(lean, expected, outputs, samples)
    actual = diff.execute_mlir(
        work / "actual.mlir", samples, work, fidelity_tools, inputs=inputs, outputs=outputs, function="forward"
    )
    diff.assert_equal(lean, actual["results"], outputs, samples)
    assert actual["results"] == expected
    (work / "ordered-view-recipe-results.json").write_text(json.dumps(expected))


def words(values):
    import numpy as np

    return [str(int(word)) for word in values.view(np.uint32).flat]


@pytest.mark.parametrize("case", SLICES)
def test_slice_w8a8_against_native_lean_and_independent_recipe(tmp_path, lean_tools, fidelity_tools, case):
    import numpy as np

    before = parse_mlir_text(case.text())
    report, documents = checked_rewrite(before, 2, tmp_path, lean_tools)
    assert report["conditional_initialized_policy_refinement"], report
    rng = np.random.default_rng(20261022)
    samples, expected = [], []
    for _ in range(3):
        values = rng.uniform(-5, 5, case.source).astype(np.float32)
        weights = rng.uniform(-3, 3, (case.result[1], 2)).astype(np.float32)
        samples.append([words(values), words(weights)])
        expected.append(words(dynamic_matmul_reference(case.reference(values), weights)))
    compare_outputs(documents[1], samples, expected, tmp_path / "after", fidelity_tools)


def concat_samples(shapes, axis):
    import numpy as np

    rng = np.random.default_rng(20261022)
    samples, expected = [], []
    for _ in range(3):
        parts = [rng.uniform(-5, 5, shape).astype(np.float32) for shape in shapes]
        joined = np.concatenate(parts, axis=axis)
        weights = rng.uniform(-3, 3, (joined.shape[1], 2)).astype(np.float32)
        samples.append([*(words(part) for part in parts), words(weights)])
        expected.append(words(dynamic_matmul_reference(joined, weights)))
    return samples, expected


@pytest.mark.parametrize("shapes,axis", CONCATS[:2])
def test_concat_w8a8_against_native_lean_and_independent_recipe(tmp_path, lean_tools, fidelity_tools, shapes, axis):
    before = concat_source(shapes, axis)
    report, documents = checked_rewrite(before, len(shapes) + 1, tmp_path, lean_tools)
    assert report["conditional_initialized_policy_refinement"], report
    samples, expected = concat_samples(shapes, axis)
    compare_outputs(documents[1], samples, expected, tmp_path / "after", fidelity_tools)


def test_native_concat_with_empty_trailing_part_reports_bufferization_refusal(tmp_path, lean_tools, fidelity_tools):
    shapes, axis = CONCATS[2]
    before = concat_source(shapes, axis)
    report, documents = checked_rewrite(before, len(shapes) + 1, tmp_path, lean_tools)
    assert report["conditional_initialized_policy_refinement"], report
    samples, expected = concat_samples(shapes, axis)
    work = tmp_path / "after"
    proc, lean = evaluate(documents[1], samples, work)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    inputs, outputs = diff.signature(json.loads(documents[1].read_text()), "forward")
    diff.assert_equal(lean, expected, outputs, samples)
    with pytest.raises(RuntimeError, match="offset 0 is out-of-bounds: 2 >= 2") as refused:
        diff.execute_mlir(
            work / "actual.mlir", samples, work, fidelity_tools, inputs=inputs, outputs=outputs, function="forward"
        )
    assert "memref.subview" in str(refused.value)
    (work / "native-concat-lowering-refusal.txt").write_text(str(refused.value))
    (work / "ordered-view-recipe-results.json").write_text(json.dumps({"native_compared": False, "expected": expected}))


@pytest.mark.parametrize("change", ["offset", "sizes", "rank", "negative_offset", "unmodelled"])
def test_exported_slice_geometry_is_rechecked(tmp_path, lean_tools, change):
    before = parse_mlir_text(SLICES[0].text())

    def mutate(document):
        ops = document["module"]["ops"][0]["regions"][0]["blocks"][0]["ops"]
        view = next(op for op in ops if op["name"] == "tensor.extract_slice")
        attrs = view["attrs"] if "static_offsets" in view["attrs"] else view["props"]
        if change in {"offset", "negative_offset"}:
            attrs["static_offsets"]["values"][0] = "4" if change == "offset" else "-1"
        elif change == "sizes":
            attrs["static_sizes"]["values"][1] = "4"
        elif change == "rank":
            attrs["static_strides"]["values"].pop()
        else:
            attrs["unreviewed_slice_mode"] = {"kind": "unit"}

    report, _ = checked_rewrite(before, 2, tmp_path, lean_tools, mutate=mutate)
    assert not report["functions"][0]["source"]["initialized"]


@pytest.mark.parametrize("change", ["negative_axis", "outside_rank", "extent", "unmodelled"])
def test_exported_concat_geometry_is_rechecked(tmp_path, lean_tools, change):
    before = concat_source()

    def mutate(document):
        ops = document["module"]["ops"][0]["regions"][0]["blocks"][0]["ops"]
        view = next(op for op in ops if op["name"] == "tensor.concat")
        attrs = view["attrs"] if "dim" in view["attrs"] else view["props"]
        if change in {"negative_axis", "outside_rank"}:
            attrs["dim"]["value"] = "-1" if change == "negative_axis" else "2"
        elif change == "extent":
            view["results"][0]["type"]["shape"][0] += 1
        else:
            attrs["unreviewed_concat_mode"] = {"kind": "unit"}

    report, _ = checked_rewrite(before, 3, tmp_path, lean_tools, mutate=mutate)
    assert not report["functions"][0]["source"]["initialized"]


def reshape_source(rows=2):
    source, flat = tensor((rows, 1, 3)), tensor((rows, 3))
    return parse_mlir_text(f"""func.func @forward(%input: {source}, %weights: tensor<3x2xf32>) -> {tensor((rows, 2))} {{
      %collapsed = tensor.collapse_shape %input [[0, 1], [2]] : {source} into {flat}
      %expanded = tensor.expand_shape %collapsed [[0, 1], [2]] output_shape [{rows}, 1, 3] : {flat} into {source}
      %collapsed_again = tensor.collapse_shape %expanded [[0, 1], [2]] : {source} into {flat}
      %view = tensor.cast %collapsed_again : {flat} to {flat}
      {matmul("%view", (rows, 3))}
    }}""")


@pytest.mark.parametrize("rows", [0, 2])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_reshape_and_cast_preserve_finiteness_through_w8a8(tmp_path, lean_tools, rows, index_bits):
    report, _ = checked_rewrite(reshape_source(rows), 2, tmp_path, lean_tools, index_bits=index_bits)
    assert report["conditional_initialized_policy_refinement"], report
    assert report["functions"][0]["reference"]["unbounded_initialized"]


def test_reshape_w8a8_against_native_lean_and_independent_recipe(tmp_path, lean_tools, fidelity_tools):
    import numpy as np

    report, documents = checked_rewrite(reshape_source(), 2, tmp_path, lean_tools)
    assert report["conditional_initialized_policy_refinement"], report
    rng = np.random.default_rng(20261022)
    samples, expected = [], []
    for _ in range(3):
        values = rng.uniform(-5, 5, (2, 1, 3)).astype(np.float32)
        weights = rng.uniform(-3, 3, (3, 2)).astype(np.float32)
        samples.append([words(values), words(weights)])
        expected.append(words(dynamic_matmul_reference(values.reshape((2, 3)), weights)))
    compare_outputs(documents[1], samples, expected, tmp_path / "after", fidelity_tools)


@pytest.mark.parametrize("index_bits", [32, 64])
def test_integer_intervals_survive_actual_token_reshape_chain(tmp_path, lean_tools, index_bits):
    before = embedding_source(token_view=True)
    after = before.clone()
    assert lower_contraction_int8(after) == 1
    report, _ = qualify(before, after, contract(7), tmp_path, lean_tools, index_bits=index_bits)
    assert report["conditional_initialized_policy_refinement"], report
    assert not report["functions"][0]["source"]["unbounded_initialized"]


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("vocabulary,accepted", [(7, True), (3, False)])
def test_concat_selects_an_interval_covering_every_input(tmp_path, lean_tools, index_bits, vocabulary, accepted):
    before = parse_mlir_text(f"""func.func @forward(%left: tensor<1xi64>, %right: tensor<1xi64>,
        %table: tensor<{vocabulary}x3xf32>, %weights: tensor<3x2xf32>) -> tensor<2x2xf32> {{
      %tokens = tensor.concat dim(0) %left, %right : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
      %destination = tensor.empty() : tensor<2x3xf32>
      %view = linalg.generic {{
        indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      }} ins(%tokens : tensor<2xi64>) outs(%destination : tensor<2x3xf32>) {{
      ^bb0(%token: i64, %old: f32):
        %row = arith.index_cast %token : i64 to index
        %column = linalg.index 1 : index
        %value = tensor.extract %table[%row, %column] : tensor<{vocabulary}x3xf32>
        linalg.yield %value : f32
      }} -> tensor<2x3xf32>
      {matmul("%view", (2, 3))}
    }}""")
    after = before.clone()
    assert lower_contraction_int8(after) == 1
    # The second argument's interval covers both inputs. Copying the first argument's
    # smaller interval would make the output contract false for valid right-hand tokens.
    inputs = [{"function": "forward", "intervals": [{"lower": 0, "upper": 3}, {"lower": 0, "upper": 7}, None, None]}]
    report, _ = qualify(before, after, inputs, tmp_path, lean_tools, index_bits=index_bits)
    assert report["conditional_initialized_policy_refinement"] is accepted, report


@pytest.mark.parametrize("change", ["reassociation", "output_shape", "unmodelled"])
def test_exported_reshape_geometry_is_rechecked(tmp_path, lean_tools, change):
    def mutate(document):
        ops = document["module"]["ops"][0]["regions"][0]["blocks"][0]["ops"]
        op = next(op for op in ops if op["name"] == "tensor.collapse_shape")
        attrs = op["attrs"] if "reassociation" in op["attrs"] else op["props"]
        if change == "reassociation":
            attrs["reassociation"]["elements"][0]["elements"][0]["value"] = "2"
        elif change == "output_shape":
            op["results"][0]["type"]["shape"][0] = 3
        else:
            attrs["unreviewed_reshape_mode"] = {"kind": "unit"}

    report, _ = checked_rewrite(reshape_source(), 2, tmp_path, lean_tools, mutate=mutate)
    assert not report["functions"][0]["source"]["initialized"]


@pytest.mark.parametrize("kind", ["slice", "concat"])
def test_public_receipt_qualifies_view_fed_sequence(tmp_path, lean_tools, kind):
    from test_lean_w8a8_chain import report
    from test_lean_w8a8_dynamic_receipts import evidence
    from test_lean_w8a8_receipts import stage_at

    from merlin.verify.lean import policy

    before = parse_mlir_text(SLICES[0].text()) if kind == "slice" else concat_source()
    stage = stage_at(tmp_path, before)
    receipt = policy.check_w8a8_draft(stage, m2m_python=lean_tools.m2m_python, scope="all")
    numerical = evidence(receipt, chain=False)
    assert numerical["conditional_initialized_policy_refinement"]
    assert numerical["boundary_qualification_passed"]
    assert report(receipt)["whole_pass_verified"] is False
