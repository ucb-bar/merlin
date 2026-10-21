"""Inactive candidate qualification against authoritative MLIR exports and native execution.

The initialized mask/value/seed cut is explicit. These samples do not establish that the
rest of SmolVLA produces that cut domain, or that no other full-model obligations remain.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.verify.lean import differential as diff
from merlin.verify.lean.export import export_file
from merlin.xdsl_dialects._common import text as to_text


def candidate(source: str, samples: list, work: Path, tools, *, limit=1024) -> tuple[list, dict, Path]:
    work.mkdir(parents=True, exist_ok=True)
    path, exported = work / "subject.mlir", work / "subject.json"
    path.write_text(source)
    export_file(path, exported, backend="mlir", m2m_python=tools.compiler_python)
    document = json.loads(exported.read_text())
    request = work / "candidate-request.json"
    request.write_text(
        json.dumps(
            {
                "export": exported.read_text(),
                "function": "subject",
                "index_bits": 64,
                "limit": limit,
                "samples": samples,
            }
        )
    )
    fixture = merlin_dir() / "tests/fixtures/lean_frontend_mask_runtime.lean"
    result = subprocess.run(
        [str(tools.lake), "env", "lean", "--run", str(fixture), str(request)],
        cwd=merlin_dir() / "lean",
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    answer = json.loads(result.stdout)
    (work / "candidate-results.json").write_text(json.dumps(answer, indent=2) + "\n")
    return answer, document, path


def independent_selection(n: int) -> str:
    return f"""builtin.module {{
func.func @subject(%mask: tensor<{n}xi1>, %values: tensor<{n}xi64>, %seed: tensor<{n}xi64>)
    -> (tensor<{n}xi64>, index) {{
  %empty = tensor.empty() : tensor<{n}xi64>
  %ints = linalg.generic {{indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
    iterator_types = ["parallel"]}} ins(%mask : tensor<{n}xi1>) outs(%empty : tensor<{n}xi64>) {{
    ^bb0(%b: i1, %old: i64): %v = arith.extui %b : i1 to i64 linalg.yield %v : i64
  }} -> tensor<{n}xi64>
  %zero64 = arith.constant 0 : i64
  %init = tensor.splat %zero64 : tensor<i64>
  %sum = linalg.reduce ins(%ints : tensor<{n}xi64>) outs(%init : tensor<i64>) dimensions = [0]
    (%v: i64, %acc: i64) {{ %a = arith.addi %v, %acc : i64 linalg.yield %a : i64 }}
  %size64 = tensor.extract %sum[] : tensor<i64>
  %size = arith.index_cast %size64 : i64 to index
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %n = arith.constant {n} : index
  %buf = tensor.empty(%size) : tensor<?xi64>
  %g:2 = scf.for %i = %zero to %n step %one iter_args(%dst = %buf, %pos = %zero)
      -> (tensor<?xi64>, index) {{
    %b = tensor.extract %mask[%i] : tensor<{n}xi1>
    %r:2 = scf.if %b -> (tensor<?xi64>, index) {{
      %v = tensor.extract %values[%i] : tensor<{n}xi64>
      %next = tensor.insert %v into %dst[%pos] : tensor<?xi64>
      %p = arith.addi %pos, %one : index
      scf.yield %next, %p : tensor<?xi64>, index
    }} else {{ scf.yield %dst, %pos : tensor<?xi64>, index }}
    scf.yield %r#0, %r#1 : tensor<?xi64>, index
  }}
  %s:2 = scf.for %i = %zero to %n step %one iter_args(%dst = %seed, %pos = %zero)
      -> (tensor<{n}xi64>, index) {{
    %b = tensor.extract %mask[%i] : tensor<{n}xi1>
    %r:2 = scf.if %b -> (tensor<{n}xi64>, index) {{
      %v = tensor.extract %g#0[%pos] : tensor<?xi64>
      %next = tensor.insert %v into %dst[%i] : tensor<{n}xi64>
      %p = arith.addi %pos, %one : index
      scf.yield %next, %p : tensor<{n}xi64>, index
    }} else {{ scf.yield %dst, %pos : tensor<{n}xi64>, index }}
    scf.yield %r#0, %r#1 : tensor<{n}xi64>, index
  }}
  return %s#0, %s#1 : tensor<{n}xi64>, index
}} }}"""


def actual_selection() -> str:
    """Clone actual capture operations, preserving their SSA uses, types, regions and attributes."""
    from xdsl.dialects import func
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region

    capture = merlin_dir() / "contract/capsules/model/SY_model_smolvla/capsule.interface.mlir"
    module = parse_mlir_text(capture.read_text())
    forward = next(o for o in module.body.block.ops if isinstance(o, func.FuncOp))
    operations = list(forward.body.block.ops)
    # Identify by actual def/use structure; the Python slice is only an extraction hint.
    gather = next(o for o in operations if o.name == "scf.for")
    gather_index = operations.index(gather)
    first = gather_index - 15
    last = next(i for i in range(gather_index + 1, len(operations)) if operations[i].name == "scf.for") + 1
    selected = operations[first : last + 1]
    defined = {r for o in selected for r in o.results}
    free = list(dict.fromkeys(v for o in selected for v in o.operands if v not in defined))
    types = [v.type for v in free]
    block = Block(arg_types=types)
    mapping = dict(zip(free, block.args, strict=True))
    for operation in selected:
        block.add_op(operation.clone(value_mapper=mapping))
    cloned = list(block.ops)
    result, count = cloned[-1].results[0], cloned[-2].results[1]
    block.add_op(func.ReturnOp(result, count))
    isolated = ModuleOp([func.FuncOp.from_region("subject", types, [result.type, count.type], Region(block))])
    isolated.verify()
    return to_text(isolated, generic=True)


def observe_compact(source: str, n: int) -> str:
    """Add an oracle observation of the dynamic compact result, zero padding its static ABI.

    The original selection operations and regions are retained. This observation loop is
    test instrumentation, not a hypothesis used by the selection proof.
    """
    from xdsl.dialects import func
    from xdsl.dialects.builtin import FunctionType

    module = parse_mlir_text(source)
    subject = next(o for o in module.body.block.ops if isinstance(o, func.FuncOp))
    gather = next(o for o in subject.body.block.ops if o.name == "scf.for")
    returned = subject.body.block.last_op
    helper_source = f"""builtin.module {{func.func @observe(%compact: tensor<?xi64>, %count: index)
      -> tensor<{n}xi64> {{
      %zero = arith.constant 0 : index %one = arith.constant 1 : index
      %init = arith.constant dense<0> : tensor<{n}xi64>
      %out = scf.for %i = %zero to %count step %one iter_args(%dst = %init) -> (tensor<{n}xi64>) {{
        %v = tensor.extract %compact[%i] : tensor<?xi64>
        %next = tensor.insert %v into %dst[%i] : tensor<{n}xi64>
        scf.yield %next : tensor<{n}xi64>
      }}
      return %out : tensor<{n}xi64>
    }} }}"""
    helper_module = parse_mlir_text(helper_source)
    helper = next(iter(helper_module.body.block.ops)).clone()
    observation_type = helper.function_type.outputs.data[0]
    call = func.CallOp("observe", [gather.results[0], returned.operands[1]], [observation_type])
    subject.body.block.insert_op_before(call, returned)
    replacement = func.ReturnOp(*returned.operands, call.results[0])
    subject.body.block.insert_op_before(replacement, returned)
    subject.body.block.erase_op(returned)
    subject.properties["function_type"] = FunctionType.from_lists(
        list(subject.function_type.inputs), list(subject.function_type.outputs) + [observation_type]
    )
    module.body.block.add_op(helper)
    module.verify()
    return to_text(module, generic=True)


def compare_selection(source: str, n: int, work: Path, tools) -> None:
    masks = [[0] * n, [1] * n, [i % 2 for i in range(n)], [int(i % 7 == 2) for i in range(n)]]
    values = [((1 << 63) + 19 * i) % (1 << 64) for i in range(n)]
    seed = [(17 + 31 * i) % (1 << 64) for i in range(n)]
    samples = [[[str(x) for x in m], [str(x) for x in values], [str(x) for x in seed]] for m in masks]
    lean, document, path = candidate(observe_compact(source, n), samples, work, tools, limit=max(n, 1))
    inputs, outputs = diff.signature(document, "subject")
    native = diff.execute_mlir(path, samples, work / "native", tools, inputs=inputs, outputs=outputs)
    diff.assert_equal(lean, native["results"], outputs, samples)
    for mask, result in zip(masks, lean, strict=True):
        expected = [str(v if b else s) for b, v, s in zip(mask, values, seed, strict=True)]
        assert result["values"][0]["bits"] == expected
        assert result["values"][1]["bits"] == [str(sum(mask))]
        selected = [str(v) for b, v in zip(mask, values, strict=True) if b]
        assert result["values"][2]["bits"] == selected + ["0"] * (n - sum(mask))


@pytest.mark.parametrize("n", [0, 1, 8, 1024])
def test_independent_native_selection(tmp_path, fidelity_tools, n):
    compare_selection(independent_selection(n), n, tmp_path, fidelity_tools)


def test_actual_smolvla_selection_native(tmp_path, fidelity_tools):
    compare_selection(actual_selection(), 1024, tmp_path, fidelity_tools)


@pytest.mark.parametrize(
    "args,limit,reason",
    [
        ([[None], ["3"], ["1"]], 16, "controls must be initialized"),
        ([["0"], [str(2**64 - 1)], ["1"]], 16, "controls must be initialized"),
        ([["0"], ["3"], ["0"]], 16, "step must be positive"),
        ([["0"], ["17"], ["1"]], 16, "iteration budget"),
        ([[str(2**63 - 2)], [str(2**63 - 1)], ["2"]], 16, "signed advancement"),
    ],
)
def test_runtime_control_refusals(tmp_path, fidelity_tools, args, limit, reason):
    source = """builtin.module {func.func @subject(%lo: index, %hi: index, %step: index) -> index {
      %zero = arith.constant 0 : index
      %r = scf.for %i = %lo to %hi step %step iter_args(%s = %zero) -> index {
        %v = arith.addi %s, %step : index scf.yield %v : index }
      return %r : index }}"""
    lean, _, _ = candidate(source, [args], tmp_path, fidelity_tools, limit=limit)
    assert lean[0]["status"] == "stuck"
    assert reason in lean[0]["reason"]


def test_uninitialized_condition_is_refused(tmp_path, fidelity_tools):
    source = """builtin.module {func.func @subject(%b: i1) -> i64 {
      %z = arith.constant 0 : i64 %o = arith.constant 1 : i64
      %r = scf.if %b -> i64 {scf.yield %o : i64} else {scf.yield %z : i64}
      return %r : i64 }}"""
    lean, _, _ = candidate(source, [[[None]], [["0"]], [["1"]]], tmp_path, fidelity_tools)
    assert lean[0]["status"] == "stuck" and "initialized i1" in lean[0]["reason"]
    assert [v["values"][0]["bits"] for v in lean[1:]] == [["0"], ["1"]]


def inspect_candidate(source: str, work: Path, tools) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    mlir, exported = work / "subject.mlir", work / "subject.json"
    mlir.write_text(source)
    export_file(mlir, exported, backend="mlir", m2m_python=tools.compiler_python)
    request = work / "inspect-request.json"
    request.write_text(json.dumps({"export": exported.read_text(), "inspect": True}))
    fixture = merlin_dir() / "tests/fixtures/lean_frontend_mask_runtime.lean"
    proc = subprocess.run(
        [str(tools.lake), "env", "lean", "--run", str(fixture), str(request)],
        cwd=merlin_dir() / "lean",
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    answer = json.loads(proc.stdout)
    (work / "body-checks.json").write_text(json.dumps(answer, indent=2) + "\n")
    return answer


def test_actual_and_independent_bodies_reach_lean_checker(tmp_path, fidelity_tools):
    for name, source in [("independent", independent_selection(8)), ("actual-cut", actual_selection())]:
        result = inspect_candidate(source, tmp_path / name, fidelity_tools)
        assert [(x["gather_body"], x["scatter_body"]) for x in result["loops"]] == [(True, False), (False, True)]
        assert all(x["controls"] and x["attributes"] for x in result["loops"])
        assert result["count_providers"] == [True, False], name
        assert result["mask_alias"], name
        assert result["whole_cut"] is (name == "actual-cut")


def test_valid_body_mutations_are_rejected_in_lean(tmp_path, fidelity_tools):
    source = independent_selection(8)
    cases = {
        "wrong_gather_read": source.replace("%values[%i]", "%values[%pos]"),
        "wrong_scatter_read": source.replace("%g#0[%pos]", "%g#0[%i]"),
        "wrong_scatter_write": source.replace("%v into %dst[%i]", "%v into %dst[%pos]"),
        "retained_position": source.replace("scf.yield %next, %p", "scf.yield %next, %pos"),
        "wrong_false_branch": source.replace("scf.yield %dst, %pos", "scf.yield %dst, %one"),
        "wrong_increment_operand": source.replace("arith.addi %pos, %one", "arith.addi %pos, %pos"),
    }
    for name, mutated in cases.items():
        assert mutated != source
        result = inspect_candidate(mutated, tmp_path / name, fidelity_tools)
        assert not all(x["gather_body"] or x["scatter_body"] for x in result["loops"]), name


def test_dynamic_empty_size_refusals(tmp_path, fidelity_tools):
    source = """builtin.module {func.func @subject(%n: index) -> tensor<?xi64> {
      %r = tensor.empty(%n) : tensor<?xi64> return %r : tensor<?xi64> }}"""
    lean, _, _ = candidate(source, [[[None]], [[str(2**64 - 1)]], [["0"]], [["3"]]], tmp_path, fidelity_tools)
    assert [x["status"] for x in lean] == ["stuck", "stuck", "ok", "ok"]
    assert lean[2]["values"][0] == {"shape": [0], "bits": []}
    # Allocation alone initializes no elements. Poison is not accepted as native evidence.
    assert lean[3]["values"][0] == {"shape": [3], "bits": [None, None, None]}


def test_valid_count_and_mask_mutations_are_refused(tmp_path, fidelity_tools):
    source = independent_selection(8)
    cases = {
        "nonzero_initial_count": source.replace("constant 0 : i64", "constant 1 : i64"),
        "signed_extension": source.replace("arith.extui", "arith.extsi"),
        "subtraction_reduction": source.replace("arith.addi %v, %acc", "arith.subi %v, %acc"),
        "wrong_allocation_size": source.replace("tensor.empty(%size)", "tensor.empty(%one)"),
    }
    for name, mutated in cases.items():
        result = inspect_candidate(mutated, tmp_path / name, fidelity_tools)
        assert result["count_providers"][0] is False, name
    other = source.replace("%empty =", "%other = arith.constant dense<false> : tensor<8xi1> %empty =")
    head, tail = other.rsplit("%mask[%i]", 1)
    result = inspect_candidate(head + "%other[%i]" + tail, tmp_path / "wrong_scatter_mask", fidelity_tools)
    assert result["mask_alias"] is False


def test_pinned_torch_selection_oracle(tmp_path):
    import os

    python = os.environ.get("MERLIN_TORCH_PYTHON")
    if not python:
        if os.environ.get("MERLIN_LEAN_REQUIRED"):
            pytest.fail("MERLIN_TORCH_PYTHON is required for the pinned CPU oracle")
        pytest.skip("MERLIN_TORCH_PYTHON is not set")
    samples = []
    for n in (0, 1, 8, 1024):
        values = [((1 << 63) + i * 19) % (1 << 64) for i in range(n)]
        seed = [(17 + i * 31) % (1 << 64) for i in range(n)]
        for mask in ([0] * n, [1] * n, [i % 2 for i in range(n)], [int(i % 7 == 2) for i in range(n)]):
            samples.append([mask, values, seed])
    program = """import json, sys, numpy as np, torch
assert torch.__version__ == '2.10.0+cpu', torch.__version__
results = []
for mask, values, seed in json.load(sys.stdin):
    m = torch.tensor(mask, dtype=torch.bool)
    v = torch.from_numpy(np.array(values, dtype=np.uint64).view(np.int64))
    out = torch.from_numpy(np.array(seed, dtype=np.uint64).view(np.int64)).clone()
    compact = v[m]
    out[m] = compact
    padded = torch.zeros(len(mask), dtype=torch.int64)
    padded[:len(compact)] = compact
    results.append([out.numpy().view(np.uint64).tolist(), int(m.sum()), padded.numpy().view(np.uint64).tolist()])
print(json.dumps({'torch': torch.__version__, 'numpy': np.__version__, 'results': results}))
"""
    proc = subprocess.run(
        [python, "-c", program], input=json.dumps(samples), capture_output=True, text=True, check=True, timeout=60
    )
    result = json.loads(proc.stdout)
    for (mask, values, seed), (out, count, padded) in zip(samples, result["results"], strict=True):
        assert out == [v if b else s for b, v, s in zip(mask, values, seed, strict=True)]
        assert count == sum(mask)
        assert padded == [v for b, v in zip(mask, values, strict=True) if b] + [0] * (len(mask) - count)
    (tmp_path / "pinned-torch-results.json").write_text(json.dumps(result, indent=2) + "\n")


def test_unsigned_for_is_explicitly_unsupported(tmp_path, fidelity_tools):
    source = """builtin.module {func.func @subject(%n: index) -> index {
      %z = arith.constant 0 : index %o = arith.constant 1 : index
      %r = scf.for unsigned %i = %z to %n step %o iter_args(%s = %z) -> index {
        %v = arith.addi %s, %o : index scf.yield %v : index}
      return %r : index }}"""
    lean, _, _ = candidate(source, [[["4"]]], tmp_path, fidelity_tools)
    assert lean[0]["status"] == "stuck" and "unsignedCmp" in lean[0]["reason"]


def test_actual_whole_cut_valid_mutations_reach_lean(tmp_path, fidelity_tools):
    from xdsl.dialects import arith, func
    from xdsl.dialects.builtin import IntegerAttr

    original = actual_selection()
    assert inspect_candidate(original, tmp_path / "baseline", fidelity_tools)["whole_cut"]
    for mutation in (
        "wrong_root_values",
        "wrong_gather_read",
        "wrong_scatter_write",
        "wrong_increment",
        "extra_operation",
        "nonzero_count",
    ):
        module = parse_mlir_text(original)
        subject = next(o for o in module.body.block.ops if isinstance(o, func.FuncOp))
        operations = list(subject.body.block.ops)
        loops = [o for o in operations if o.name == "scf.for"]
        gather, scatter = loops
        if mutation == "wrong_root_values":
            operations[2].operands = [subject.body.block.args[2]]
        elif mutation == "wrong_gather_read":
            branch = next(o for o in gather.regions[0].block.ops if o.name == "scf.if")
            read = next(o for o in branch.regions[0].block.ops if o.name == "tensor.extract")
            read.operands = [read.operands[0], gather.regions[0].block.args[2]]
        elif mutation == "wrong_scatter_write":
            branch = next(o for o in scatter.regions[0].block.ops if o.name == "scf.if")
            insert = next(o for o in branch.regions[0].block.ops if o.name == "tensor.insert")
            insert.operands = [*insert.operands[:2], scatter.regions[0].block.args[2]]
        elif mutation == "wrong_increment":
            branch = next(o for o in gather.regions[0].block.ops if o.name == "scf.if")
            increment = next(o for o in branch.regions[0].block.ops if o.name == "arith.addi")
            increment.operands = [increment.operands[0], increment.operands[0]]
        elif mutation == "extra_operation":
            subject.body.block.insert_op_before(arith.ConstantOp.from_int_and_width(17, 64), gather)
        elif mutation == "nonzero_count":
            operations[6].properties["value"] = IntegerAttr(1, 64)
        module.verify()
        result = inspect_candidate(to_text(module, generic=True), tmp_path / mutation, fidelity_tools)
        assert result["whole_cut"] is False, mutation


def test_authoritative_full_boundaries_check_the_actual_projection(tmp_path, fidelity_tools):
    from merlin.llvmlower.passes_xdsl import fix_bool_fptosi

    capture = merlin_dir() / "contract/capsules/model/SY_model_smolvla/capsule.interface.mlir"
    module = parse_mlir_text(capture.read_text())
    before = to_text(module, generic=True)
    assert fix_bool_fptosi(module) == 1
    for side, source in (("before", before), ("after", to_text(module, generic=True))):
        result = inspect_candidate(source, tmp_path / side, fidelity_tools)
        projection = result["projections"][0]
        assert projection["accepted"], (side, projection)
        assert projection["selected_operations"] == 25
        assert result["whole_cut"] is False  # This is not the whole-model cut signature.
        assert result["count_providers"] == [True, False]
        assert result["mask_alias"]
