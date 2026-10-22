"""Producer-proved integer coordinates feeding actual embedding and W8A8 lowering."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest
from test_lean_w8a8_native_recipe import dynamic_matmul_reference
from test_lean_w8a8_views import checked_rewrite, compare_outputs, matmul, tensor, words

from merlin.frontends.linalg_mlir import parse_mlir_text

pytestmark = pytest.mark.usefixtures("lean_oracle")


@dataclass(frozen=True)
class Coordinates:
    rows: int = 4
    width: int = 64
    step: int = 2
    start: int = 1
    vocabulary: int = 11
    mode: str = "scalar"

    def source(self, *, reduction=False, combine="addi"):
        integer = f"i{self.width}"
        vector = f"tensor<{self.rows}x{integer}>"
        start = self.start if self.mode == "scalar" else 0
        prefix = f"""%range_empty = tensor.empty() : {vector}
          %range = linalg.generic {{
            indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]
          }} outs(%range_empty : {vector}) {{
          ^bb0(%old: {integer}):
            %index = linalg.index 0 : index
            %converted = arith.index_cast %index : index to {integer}
            %step = arith.constant {self.step} : {integer}
            %product = arith.muli %converted, %step : {integer}
            %start = arith.constant {start} : {integer}
            %coordinate = arith.{combine} %product, %start : {integer}
            linalg.yield %coordinate : {integer}
          }} -> {vector}
        """
        coords = "%range"
        if self.mode in {"splat", "concat"}:
            prefix += f"""%offset = arith.constant {self.start} : {integer}
              %offsets = tensor.splat %offset : {vector}
              %sum_empty = tensor.empty() : {vector}
              %shifted = linalg.generic {{
                indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                                 affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]
              }} ins(%range, %offsets : {vector}, {vector}) outs(%sum_empty : {vector}) {{
              ^bb0(%x: {integer}, %y: {integer}, %unused: {integer}):
                %sum = arith.addi %x, %y : {integer}
                linalg.yield %sum : {integer}
              }} -> {vector}
            """
            coords = "%shifted"
        rows = self.rows
        if self.mode == "concat":
            rows *= 2
            prefix += f"""%joined = tensor.concat dim(0) %shifted, %shifted
                : ({vector}, {vector}) -> tensor<{rows}x{integer}>
            """
            coords = "%joined"
        result = tensor((rows, 2))
        embedded = tensor((rows, 3))
        table = tensor((self.vocabulary, 3))
        tail = matmul("%embedding", (rows, 3))
        # Exercise an initialized reduction after a proved singleton splat. Its sum need
        # not remain in the initializer's interval; checking execution must still succeed.
        if reduction:
            scalar = f"tensor<{integer}>"
            prefix += f"""%initial = arith.constant 0 : {integer}
              %initial_tensor = tensor.splat %initial : {scalar}
              %total = linalg.generic {{
                indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
                iterator_types = ["reduction"]
              }} ins(%range : {vector}) outs(%initial_tensor : {scalar}) {{
              ^bb0(%item: {integer}, %acc: {integer}):
                %next = arith.addi %acc, %item : {integer}
                linalg.yield %next : {integer}
              }} -> {scalar}
            """
            tail = tail.replace(f"return %result : {result}", f"return %result, %total : {result}, {scalar}")
            result = f"({result}, {scalar})"
        return parse_mlir_text(f"""func.func @forward(%table: {table}, %weights: tensor<3x2xf32>) -> {result} {{
          {prefix}
          %destination = tensor.empty() : {embedded}
          %embedding = linalg.generic {{
            indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>],
            iterator_types = ["parallel", "parallel"]
          }} ins({coords} : tensor<{rows}x{integer}>) outs(%destination : {embedded}) {{
          ^bb0(%token: {integer}, %old_value: f32):
            %row = arith.index_cast %token : {integer} to index
            %column = linalg.index 1 : index
            %value = tensor.extract %table[%row, %column] : {table}
            linalg.yield %value : f32
          }} -> {embedded}
          {tail}
        }}""")

    def indices(self):
        values = [row * self.step + self.start for row in range(self.rows)]
        return values * (2 if self.mode == "concat" else 1)


CASES = [
    Coordinates(),
    Coordinates(width=8),
    Coordinates(width=16),
    Coordinates(width=32),
    Coordinates(rows=0),
    Coordinates(rows=1, width=8, start=127, vocabulary=128),
    Coordinates(rows=64, width=8, step=2, start=1, vocabulary=128),
    Coordinates(mode="splat"),
    Coordinates(mode="concat"),
]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("index_bits", [32, 64])
def test_integer_coordinates_establish_bounds_without_input_assumptions(tmp_path, lean_tools, case, index_bits):
    report, _ = checked_rewrite(case.source(), 2, tmp_path, lean_tools, index_bits=index_bits)
    assert report["conditional_initialized_policy_refinement"], report
    assert report["functions"][0]["source"]["initialized"]
    assert report["functions"][0]["reference"]["initialized"]


@pytest.mark.parametrize("index_bits", [32, 64])
def test_reduction_does_not_require_preserving_its_initializer_interval(tmp_path, lean_tools, index_bits):
    report, _ = checked_rewrite(Coordinates().source(reduction=True), 2, tmp_path, lean_tools, index_bits=index_bits)
    assert report["conditional_initialized_policy_refinement"], report


@pytest.mark.parametrize("combine", ["addi", "subi", "muli", "andi", "ori", "xori"])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_singleton_arithmetic_uses_actual_integer_bits(tmp_path, lean_tools, combine, index_bits):
    # Both operands are singletons; subtraction uses zero so its signed result is
    # nonnegative. No range proof for arbitrary bitwise operands is claimed.
    case = Coordinates(rows=1, width=8, start=0 if combine == "subi" else 1)
    report, _ = checked_rewrite(case.source(combine=combine), 2, tmp_path, lean_tools, index_bits=index_bits)
    assert report["conditional_initialized_policy_refinement"], report


@pytest.mark.parametrize(
    "case",
    [
        Coordinates(vocabulary=7),
        Coordinates(start=-1),
        Coordinates(step=-1),
        Coordinates(rows=64, width=8, step=3, vocabulary=256),
        Coordinates(rows=64, width=8, step=2, start=2, vocabulary=256),
        Coordinates(rows=1, width=8, start=128, vocabulary=256),
        Coordinates(start=12, mode="splat"),
    ],
)
def test_actual_oob_or_signed_wrapping_coordinates_are_refused(tmp_path, lean_tools, case):
    report, _ = checked_rewrite(case.source(), 2, tmp_path, lean_tools)
    assert not report["functions"][0]["source"]["initialized"]
    assert not report["conditional_initialized_policy_refinement"]


@pytest.mark.parametrize("change", ["flags", "unmodelled", "operand", "capture", "constant_bits"])
def test_coordinate_evidence_is_recomputed_from_actual_operations(tmp_path, lean_tools, change):
    def mutate(document):
        ops = document["module"]["ops"][0]["regions"][0]["blocks"][0]["ops"]
        body = next(op for op in ops if op["name"] == "linalg.generic")["regions"][0]["blocks"][0]
        multiply = next(op for op in body["ops"] if op["name"] == "arith.muli")
        if change == "flags":
            multiply["attrs"]["overflowFlags"] = {
                "kind": "enum",
                "dialect": "arith",
                "name": "overflow",
                "value": "nsw",
            }
        elif change == "unmodelled":
            multiply["attrs"]["unreviewed_integer_mode"] = {"kind": "unit"}
        elif change == "operand":
            multiply["operands"][0] = body["args"][0]["id"]
        elif change == "capture":
            embedding = [op for op in ops if op["name"] == "linalg.generic"][-1]
            block = embedding["regions"][0]["blocks"][0]
            extract = next(op for op in block["ops"] if op["name"] == "tensor.extract")
            block["args"][0]["id"] = extract["operands"][0]
        else:
            constant = next(op for op in body["ops"] if op["name"] == "arith.constant")
            constant["attrs"]["value"]["value"] = "127"

    report, _ = checked_rewrite(Coordinates(width=8).source(), 2, tmp_path, lean_tools, mutate=mutate)
    assert not report["functions"][0]["source"]["initialized"]
    assert not report["conditional_initialized_policy_refinement"]


@pytest.mark.parametrize("case", [Coordinates(width=8), Coordinates(mode="splat"), Coordinates(mode="concat")])
def test_coordinate_embedding_and_w8a8_match_native_lean_and_independent_recipe(
    tmp_path, lean_tools, fidelity_tools, case
):
    import numpy as np

    report, documents = checked_rewrite(case.source(), 2, tmp_path, lean_tools)
    assert report["conditional_initialized_policy_refinement"], report
    rng = np.random.default_rng(20261023)
    samples, expected = [], []
    for scale in (0, 1, 100):
        table = (rng.uniform(-5, 5, (case.vocabulary, 3)) * scale).astype(np.float32)
        weights = rng.uniform(-3, 3, (3, 2)).astype(np.float32)
        samples.append([words(table), words(weights)])
        expected.append(words(dynamic_matmul_reference(table[case.indices()], weights)))
    compare_outputs(documents[1], samples, expected, tmp_path / "after", fidelity_tools)
    (tmp_path / "coordinate-recipe-results.json").write_text(
        json.dumps({"indices": case.indices(), "results": expected})
    )


def test_integer_reduction_and_w8a8_outputs_match_native_and_lean(tmp_path, lean_tools, fidelity_tools):
    import numpy as np

    case = Coordinates(width=8, rows=64, step=2, start=1, vocabulary=128)
    report, documents = checked_rewrite(case.source(reduction=True), 2, tmp_path, lean_tools)
    assert report["conditional_initialized_policy_refinement"], report
    table = np.arange(case.vocabulary * 3, dtype=np.float32).reshape((case.vocabulary, 3))
    weights = np.array([[1, -2], [3, 1], [-1, 2]], dtype=np.float32)
    samples = [[words(table), words(weights)]]
    # The returned sum is modular i8 arithmetic, independently of the index width.
    expected = [words(dynamic_matmul_reference(table[case.indices()], weights)) + [str(sum(case.indices()) % 256)]]
    compare_outputs(documents[1], samples, expected, tmp_path / "after", fidelity_tools)
