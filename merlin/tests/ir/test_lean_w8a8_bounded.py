"""Check embedding-fed W8A8 on explicit token intervals, with native numerical comparisons."""

from __future__ import annotations

import json
import subprocess

import pytest
from test_lean_w8a8_body import exported
from test_lean_w8a8_byte_model import evaluate
from test_lean_w8a8_native_recipe import dynamic_matmul_reference

from merlin.common.paths import repo_root
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.passes_quant_int import lower_contraction_int8
from merlin.verify.lean import differential as diff

pytestmark = pytest.mark.usefixtures("lean_oracle")


def embedding_source(rows=2, vocabulary=7, width=3, columns=4, *, token_width=64, contraction=True, token_view=False):
    tokens, table = f"tensor<{rows}xi{token_width}>", f"tensor<{vocabulary}x{width}xf32>"
    embedded, weights, output = (
        f"tensor<{rows}x{width}xf32>",
        f"tensor<{width}x{columns}xf32>",
        f"tensor<{rows}x{columns}xf32>",
    )
    tail = f"return %embedding : {embedded}"
    token_prefix = ""
    token_value = "%tokens"
    if token_view:
        token_value = "%tokens_view"
        expanded = f"tensor<1x{rows}xi{token_width}>"
        token_prefix = f"""%tokens_expanded = tensor.expand_shape %tokens [[0, 1]] output_shape [1, {rows}]
            : {tokens} into {expanded}
          %tokens_collapsed = tensor.collapse_shape %tokens_expanded [[0, 1]] : {expanded} into {tokens}
          %tokens_view = tensor.cast %tokens_collapsed : {tokens} to {tokens}
        """
    if contraction:
        tail = f"""%empty = tensor.empty() : {output}
          %zero = arith.constant 0.0 : f32
          %filled = linalg.fill ins(%zero : f32) outs(%empty : {output}) -> {output}
          %result = linalg.matmul ins(%embedding, %weights : {embedded}, {weights})
            outs(%filled : {output}) -> {output}
          return %result : {output}"""
    return parse_mlir_text(f"""func.func @forward(%tokens: {tokens}, %table: {table}, %weights: {weights})
        -> {output if contraction else embedded} {{
      {token_prefix}
      %destination = tensor.empty() : {embedded}
      %embedding = linalg.generic {{
        indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      }} ins({token_value} : {tokens}) outs(%destination : {embedded}) {{
      ^bb0(%token: i{token_width}, %old: f32):
        %row = arith.index_cast %token : i{token_width} to index
        %column = linalg.index 1 : index
        %value = tensor.extract %table[%row, %column] : {table}
        linalg.yield %value : f32
      }} -> {embedded}
      {tail}
    }}""")


def qualify(before, after, contracts, work, tools, *, index_bits=64, mutate=None):
    documents = []
    for name, program in [("before", before), ("after", after)]:
        directory = work / name
        directory.mkdir()
        documents.append(exported(program, directory, tools, generic=True, allow_unregistered=True))
    if mutate is not None:
        document = json.loads(documents[0].read_text())
        mutate(document)
        documents[0].write_text(json.dumps(document))
    contract_path = work / "input-contracts.json"
    contract_path.write_text(json.dumps(contracts))
    proc = subprocess.run(
        [
            "lake",
            "env",
            "lean",
            "--run",
            str(repo_root() / "merlin/tests/fixtures/lean_w8a8_bounded_policy.lean"),
            str(documents[0]),
            str(documents[1]),
            str(contract_path),
            str(index_bits),
            "all",
        ],
        cwd=repo_root() / "merlin/lean",
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode in (0, 1), proc.stdout + proc.stderr
    report = json.loads(proc.stdout)
    assert report["input_domain"] == "finite_f32_and_declared_integer_intervals_per_function"
    assert report["theorem"] == "MerlinVerify.Check.PolicyW8A8BoundedQualified.check_sound"
    assert report["index_bits"] == index_bits
    for field in ("whole_pass_verified", "policy_reviewed", "primitive_model_laws_verified"):
        assert report[field] is False
    assert (proc.returncode == 0) == report["conditional_initialized_policy_refinement"]
    (work / "bounded-policy-report.json").write_text(json.dumps(report, indent=2))
    return report, documents


def contract(vocabulary, *, lower=0):
    return [{"function": "forward", "intervals": [{"lower": lower, "upper": vocabulary}, None, None]}]


@pytest.mark.parametrize("index_bits,token_width", [(64, 64), (32, 64), (64, 32)])
@pytest.mark.parametrize("shape", [(2, 7, 3, 4), (3, 11, 8, 2), (0, 5, 3, 2)])
def test_actual_embedding_fed_dynamic_w8a8_is_conditionally_initialized(
    tmp_path, lean_tools, shape, index_bits, token_width
):
    before = embedding_source(*shape, token_width=token_width)
    after = before.clone()
    assert lower_contraction_int8(after) == 1
    report, _ = qualify(before, after, contract(shape[1]), tmp_path, lean_tools, index_bits=index_bits)
    assert report["conditional_initialized_policy_refinement"], report
    assert report["functions"][0]["source"]["initialized"]
    assert report["functions"][0]["reference"]["initialized"]
    if shape[0]:
        assert not report["functions"][0]["source"]["unbounded_initialized"]


@pytest.mark.parametrize(
    "change", ["unknown", "out_of_bounds", "empty", "reversed", "wrong_arity", "wrong_function", "floating_interval"]
)
def test_declared_input_domain_is_checked_without_inventing_bounds(tmp_path, lean_tools, change):
    before = embedding_source()
    after = before.clone()
    assert lower_contraction_int8(after) == 1
    contracts = contract(7)
    if change == "unknown":
        contracts[0]["intervals"][0] = None
    elif change == "out_of_bounds":
        contracts[0]["intervals"][0]["upper"] = 8
    elif change in {"empty", "reversed"}:
        contracts[0]["intervals"][0]["lower"] = 7 if change == "empty" else 8
    elif change == "wrong_arity":
        contracts[0]["intervals"].pop()
    elif change == "wrong_function":
        contracts[0]["function"] = "another_function"
    else:
        contracts[0]["intervals"][1] = {"lower": 0, "upper": 7}
    report, _ = qualify(before, after, contracts, tmp_path, lean_tools)
    assert not report["conditional_initialized_policy_refinement"]


@pytest.mark.parametrize(
    "change", ["swapped_indices", "old_destination", "wrong_table", "arithmetic_before_quantization"]
)
def test_real_mlir_mutations_do_not_forge_embedding_finiteness(tmp_path, lean_tools, change):
    before = embedding_source()
    generic = next(op for op in before.walk() if op.name == "linalg.generic")
    body = list(generic.body.block.ops)
    if change == "swapped_indices":
        # The token interval fits the table row dimension but cannot fit its columns.
        body[2].operands = [body[2].operands[0], body[1].results[0], body[0].results[0]]
    elif change == "old_destination":
        body[-1].operands = [generic.body.block.args[-1]]
    elif change == "wrong_table":
        function = next(op for op in before.walk() if op.name == "func.func")
        body[2].operands = [function.body.block.args[2], *body[2].operands[1:]]
    else:
        from xdsl.dialects.arith import MulfOp

        multiplied = MulfOp(body[2].results[0], body[2].results[0])
        generic.body.block.insert_op_before(multiplied, body[-1])
        body[-1].operands = [multiplied.results[0]]
    after = before.clone()
    assert lower_contraction_int8(after) == 1
    report, _ = qualify(before, after, contract(7), tmp_path, lean_tools)
    assert not report["conditional_initialized_policy_refinement"]


@pytest.mark.parametrize("shape", [(2, 7, 3, 4), (3, 11, 8, 2)])
def test_actual_embedding_and_w8a8_against_lean_native_and_independent_recipe(
    tmp_path, lean_tools, fidelity_tools, shape
):
    import numpy as np

    rows, vocabulary, width, columns = shape
    before = embedding_source(*shape)
    after = before.clone()
    assert lower_contraction_int8(after) == 1
    report, documents = qualify(before, after, contract(vocabulary), tmp_path, lean_tools)
    assert report["conditional_initialized_policy_refinement"], report
    rng = np.random.default_rng(20261021)
    samples, expected = [], []
    for tokens in (
        np.zeros(rows, dtype=np.int64),
        np.full(rows, vocabulary - 1, dtype=np.int64),
        rng.integers(0, vocabulary, rows),
    ):
        table = rng.uniform(-5, 5, (vocabulary, width)).astype(np.float32)
        weights = rng.uniform(-3, 3, (width, columns)).astype(np.float32)
        samples.append(
            [
                [str(int(token)) for token in tokens],
                [str(int(word)) for word in table.view(np.uint32).flat],
                [str(int(word)) for word in weights.view(np.uint32).flat],
            ]
        )
        result = dynamic_matmul_reference(table[tokens], weights)
        expected.append([str(int(word)) for word in result.view(np.uint32).flat])
    target = tmp_path / "after"
    proc, lean = evaluate(documents[1], samples, target)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    inputs, outputs = diff.signature(json.loads(documents[1].read_text()), "forward")
    native = diff.execute_mlir(
        target / "actual.mlir", samples, target, fidelity_tools, inputs=inputs, outputs=outputs, function="forward"
    )
    diff.assert_equal(lean, native["results"], outputs, samples)
    assert native["results"] == expected


@pytest.mark.parametrize("swap", [False, True])
def test_input_intervals_are_bound_to_each_function(tmp_path, lean_tools, swap):
    from xdsl.dialects.builtin import StringAttr

    before = embedding_source(vocabulary=7)
    other = embedding_source(vocabulary=11)
    function = next(op for op in other.walk() if op.name == "func.func")
    function.properties["sym_name"] = StringAttr("other")
    before.body.block.add_op(function.clone())
    after = before.clone()
    assert lower_contraction_int8(after) == 2
    first, second = (11, 7) if swap else (7, 11)
    contracts = contract(first) + [{"function": "other", "intervals": contract(second)[0]["intervals"]}]
    report, _ = qualify(before, after, contracts, tmp_path, lean_tools)
    assert report["conditional_initialized_policy_refinement"] is not swap
    assert len(report["functions"]) == 2


@pytest.mark.parametrize("index_bits,vocabulary,accepted", [(32, 2**31, True), (32, 2**31 + 1, False), (2, 7, False)])
def test_whole_interval_must_fit_actual_signed_index_encoding(tmp_path, lean_tools, index_bits, vocabulary, accepted):
    before = embedding_source(vocabulary=vocabulary)
    after = before.clone()
    assert lower_contraction_int8(after) == 1
    # This is a symbolic proof check; huge embedding tables are never allocated/executed.
    report, _ = qualify(before, after, contract(vocabulary), tmp_path, lean_tools, index_bits=index_bits)
    assert report["conditional_initialized_policy_refinement"] is accepted


@pytest.mark.parametrize("change", ["negative_dim", "missing_dim", "outside_loop", "capture_shadow"])
def test_export_protocol_cannot_forge_loop_or_capture_evidence(tmp_path, lean_tools, change):
    before = embedding_source()
    after = before.clone()
    assert lower_contraction_int8(after) == 1

    def mutate(document):
        ops = document["module"]["ops"][0]["regions"][0]["blocks"][0]["ops"]
        generic = next(op for op in ops if op["name"] == "linalg.generic")
        body = generic["regions"][0]["blocks"][0]
        index = body["ops"][1]
        if change in {"negative_dim", "missing_dim"}:
            attrs = index["attrs"] if "dim" in index["attrs"] else index["props"]
            attrs["dim"]["value"] = "-1" if change == "negative_dim" else "2"
        elif change == "outside_loop":
            body["ops"].remove(index)
            ops.insert(1, index)
        else:
            # The loop block argument shadows the captured table's binding. The checker
            # must use that binding's actual scalar type, not a previous tensor fact.
            body["args"][0]["id"] = body["ops"][2]["operands"][0]

    report, _ = qualify(before, after, contract(7), tmp_path, lean_tools, mutate=mutate)
    assert not report["functions"][0]["source"]["initialized"]
    assert not report["conditional_initialized_policy_refinement"]
