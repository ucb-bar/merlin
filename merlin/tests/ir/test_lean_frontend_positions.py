"""Actual position-reduction bounds and composed frontend execution.

Counter bounds quantify arbitrary comparison results. Host-float/native/PyTorch
samples qualify executions and do not supply a universal floating comparison law.
"""

from __future__ import annotations

import copy
import json
import os
import struct
import subprocess
from pathlib import Path

import pytest
from test_lean_frontend_runtime_checkpoint import driver, exported

from merlin.common.paths import merlin_dir
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.passes_xdsl import fix_bool_fptosi
from merlin.verify.lean import differential as diff
from merlin.verify.lean import frontend_runtime as runtime
from merlin.xdsl_dialects._common import text as to_text


def fbits(x: float) -> str:
    return str(struct.unpack("<I", struct.pack("<f", x))[0])


def independent_count(rows: int, red: int, *, seed=0, one=1, flags="", fastmath="") -> str:
    return f"""builtin.module {{func.func @subject(%x: tensor<{rows}xf32>, %thresholds: tensor<{red}xf32>)
      -> tensor<{rows}xi64> {{
      %zero = arith.constant {seed} : i64
      %one = arith.constant {one} : i64
      %initial = tensor.splat %zero : tensor<{rows}xi64>
      %counts = linalg.generic {{indexing_maps = [affine_map<(d0,d1)->(d0)>, affine_map<(d0,d1)->(d1)>,
        affine_map<(d0,d1)->(d0)>], iterator_types = ["parallel", "reduction"]}}
        ins(%x, %thresholds : tensor<{rows}xf32>, tensor<{red}xf32>) outs(%initial : tensor<{rows}xi64>) {{
          ^bb0(%row: f32, %threshold: f32, %acc: i64):
          %b = arith.cmpf ole, %threshold, %row{fastmath} : f32
          %v = arith.select %b, %one, %zero : i64
          %next = arith.addi %acc, %v{flags} : i64
          linalg.yield %next : i64 }} -> tensor<{rows}xi64>
      return %counts : tensor<{rows}xi64> }} }}"""


def actual_slice(start: int, count: int) -> str:
    from xdsl.dialects import func
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region

    capture = merlin_dir() / "contract/capsules/model/SY_model_smolvla/capsule.interface.mlir"
    module = parse_mlir_text(capture.read_text())
    forward = next(o for o in module.body.block.ops if isinstance(o, func.FuncOp))
    selected = list(forward.body.block.ops)[start : start + count]
    nested = [child for op in selected for child in op.walk()]
    local = {v for op in nested for v in op.results}
    local.update(a for op in nested for r in op.regions for b in r.blocks for a in b.args)
    free = list(dict.fromkeys(v for op in nested for v in op.operands if v not in local))
    block = Block(arg_types=[v.type for v in free])
    mapping = dict(zip(free, block.args, strict=True))
    for operation in selected:
        block.add_op(operation.clone(value_mapper=mapping))
    out = list(block.ops)[-1].results[0]
    block.add_op(func.ReturnOp(out))
    isolated = ModuleOp([func.FuncOp.from_region("subject", [v.type for v in free], [out.type], Region(block))])
    isolated.verify()
    return to_text(isolated, generic=True)


def initial_boundary(n: int = 157) -> tuple[str, str]:
    from xdsl.dialects import func
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region

    capture = merlin_dir() / "contract/capsules/model/SY_model_smolvla/capsule.interface.mlir"
    module = parse_mlir_text(capture.read_text())
    result = []
    for repair in (False, True):
        if repair:
            assert fix_bool_fptosi(module) == 1
        forward = next(o for o in module.body.block.ops if isinstance(o, func.FuncOp))
        block = Block(arg_types=forward.function_type.inputs)
        mapping = dict(zip(forward.body.block.args, block.args, strict=True))
        for op in list(forward.body.block.ops)[:n]:
            block.add_op(op.clone(value_mapper=mapping))
        ops = list(block.ops)
        outputs = [ops[115].results[0], ops[114].results[1], *ops[-1].results]
        block.add_op(func.ReturnOp(*outputs))
        isolated = ModuleOp(
            [
                func.FuncOp.from_region(
                    "subject", list(forward.function_type.inputs), [v.type for v in outputs], Region(block)
                )
            ]
        )
        isolated.verify()
        result.append(to_text(isolated, generic=True))
    return tuple(result)


def checked(source: str, work: Path, tools, *, mode="output_ranges", upper=32, **fields) -> dict:
    path = exported(source, work, tools)
    return driver(
        {"mode": mode, "function": "subject", "after_export": str(path), "upper": upper, **fields}, work, tools
    )


def candidate_float(source: str, samples: list, work: Path, tools) -> tuple[list, dict, Path]:
    path = exported(source, work, tools)
    request = work / "oracle-request.json"
    request.write_text(
        json.dumps(
            {
                "export": path.read_text(),
                "function": "subject",
                "index_bits": 64,
                "frontend_runtime": True,
                "limit": 1024,
                "samples": samples,
            }
        )
    )
    proc = subprocess.run(
        [str(tools.lake), "env", "lean", "--run", str(merlin_dir() / "lean/Differential.lean"), str(request)],
        cwd=merlin_dir() / "lean",
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    answer = json.loads(proc.stdout)
    (work / "lean-results.json").write_text(json.dumps(answer, indent=2) + "\n")
    return answer, json.loads(path.read_text()), work / "subject.mlir"


def count_samples(rows: int, red: int) -> list:
    edges = [
        0x00000000,
        0x80000000,
        0x7F800000,
        0xFF800000,
        0x7FC12345,
        0x7FA12345,
        int(fbits(-0.75)),
        int(fbits(0.75)),
        int(fbits(1.25)),
        int(fbits(-1.25)),
        int(fbits(0.5)),
    ]
    thresholds = [fbits((i + 1) / (red + 1)) for i in range(red)]
    samples = [[[str(edges[(i + shift) % len(edges)]) for i in range(rows)], thresholds] for shift in (0, 2, 4, 6, 8)]
    if red:
        special = thresholds.copy()
        special[0], special[-1] = str(0x7FC12345), str(0x7F800000)
        samples.append([[str(edges[i % len(edges)]) for i in range(rows)], special])
    return samples


def compare_count(source: str, rows: int, red: int, work: Path, tools) -> None:
    samples = count_samples(rows, red)
    lean, document, path = candidate_float(source, samples, work / "execute", tools)
    inputs, outputs = diff.signature(document, "subject")
    native = diff.execute_mlir(path, samples, work / "native", tools, inputs=inputs, outputs=outputs)
    diff.assert_equal(lean, native["results"], outputs, samples)
    for sample, result in zip(samples, lean, strict=True):
        x, thresholds = [[struct.unpack("<f", struct.pack("<I", int(v)))[0] for v in a] for a in sample]
        assert result["values"][0]["bits"] == [str(sum(t <= value for t in thresholds)) for value in x]


@pytest.mark.parametrize("rows,red", [(0, 31), (2, 0), (1, 1), (2, 31)])
def test_independent_reduction_bounds_and_native(tmp_path, fidelity_tools, rows, red):
    source = independent_count(rows, red)
    assert checked(source, tmp_path / "proof", fidelity_tools, upper=red + 1)["accepted"]
    compare_count(source, rows, red, tmp_path, fidelity_tools)


@pytest.mark.parametrize("start", [71, 75])
def test_actual_bucket_bounds_and_native(tmp_path, fidelity_tools, start):
    source = actual_slice(start, 4)
    assert checked(source, tmp_path / "proof", fidelity_tools)["accepted"]
    compare_count(source, 32, 31, tmp_path, fidelity_tools)


@pytest.mark.parametrize(
    "change", [{"seed": 1}, {"one": 2}, {"one": -1}, {"flags": " overflow<nsw>"}, {"fastmath": " fastmath<nnan>"}]
)
def test_compiler_valid_counter_mutations_rejected(tmp_path, fidelity_tools, change):
    assert not checked(independent_count(2, 31, **change), tmp_path, fidelity_tools)["accepted"]


def test_static_trace_budget_is_explicit_refusal(tmp_path, fidelity_tools):
    assert not checked(independent_count(1, 2049), tmp_path, fidelity_tools, upper=2050)["accepted"]


@pytest.mark.parametrize("n", [157, 239])
def test_actual_prefix_and_composed_continuation(tmp_path, fidelity_tools, n):
    before, after = initial_boundary(n)
    assert checked(after, tmp_path / "prefix", fidelity_tools, mode="prefix_ranges", upper=1024, start=91)["accepted"]
    answer = checked(after, tmp_path / "scope", fidelity_tools, mode="position_scope", start=91, count=n - 116)
    assert answer["accepted"] and answer["initial_operations"] == n and not answer["whole_function_defined"]
    assert not checked(before, tmp_path / "raw-before", fidelity_tools, mode="position_scope", start=91, count=n - 116)[
        "accepted"
    ]


def test_actual_first_nonboolean_conversion_remains_unproved(tmp_path, fidelity_tools):
    _, after = initial_boundary(241)
    # Operation 240 maps f32 to i8. Its initialized input does not establish that
    # FloatModel.toInt returns a value; the approved UNE model adds no such law.
    answer = checked(after, tmp_path, fidelity_tools, mode="position_scope", start=91, count=125)
    assert not answer["accepted"] and not answer["whole_function_defined"]


@pytest.mark.parametrize("change", ["seed", "increment", "predicate", "bypass-selection"])
def test_actual_continuation_mutations(tmp_path, fidelity_tools, change):
    from xdsl.dialects import arith, builtin, func

    before, after = initial_boundary()
    module = parse_mlir_text(after)
    forward = next(op for op in module.body.block.ops if isinstance(op, func.FuncOp))
    ops = list(forward.body.block.ops)
    if change in ("seed", "increment"):
        constant = ops[71 if change == "seed" else 72]
        constant.properties["value"] = builtin.IntegerAttr(1 if change == "seed" else 2, builtin.i64)
    elif change == "predicate":
        cmp = next(op for op in ops[74].walk() if isinstance(op, arith.CmpfOp))
        cmp.properties["predicate"] = builtin.IntegerAttr(4, builtin.i64)
    else:
        # Both tensors have the same type and independently bounded elements.
        # Bounds alone cannot justify changing the actual selected SSA operand.
        ops[117].operands[0] = ops[90].results[0]
    text = to_text(module, generic=True)
    scope = checked(text, tmp_path / "scope", fidelity_tools, mode="position_scope", start=91, count=41)
    assert scope["accepted"] == (change in ("predicate", "bypass-selection"))
    src = exported(before, tmp_path / "source", fidelity_tools)
    tgt = exported(text, tmp_path / "target", fidelity_tools)
    answer = driver(
        {
            "mode": "boundary",
            "function": "subject",
            "start": 91,
            "continuation_count": 41,
            "before_export": str(src),
            "after_export": str(tgt),
            "suffix_exports": [],
        },
        tmp_path / "boundary",
        fidelity_tools,
    )
    assert not answer["accepted"] and not answer["repair"]


def test_extended_boundary_receipt_and_integrity(tmp_path, fidelity_tools):
    before, after = initial_boundary()
    paths = [tmp_path / "before.mlir", tmp_path / "after.mlir"]
    for path, text in zip(paths, (before, after), strict=True):
        path.write_text(text)
    receipt = runtime.verify_candidate(
        *paths,
        tmp_path / "receipt",
        function="subject",
        start=91,
        compiler_python=fidelity_tools.compiler_python,
        lake=fidelity_tools.lake,
        continuation_count=41,
        bounds_slices=((122, 35),),
        suffix=(paths[1],),
    )
    assert receipt["status"] == "verified_candidate", receipt.get("reason")
    assert receipt["answer"]["continuation"] == {"accepted": True, "count": 41, "initial_operations": 157}
    runtime.validate_receipt(receipt)
    wrong = copy.deepcopy(receipt)
    wrong["continuation_count"] = 42
    with pytest.raises(ValueError, match="receipt order or checkpoint"):
        runtime.validate_receipt(wrong)


def test_pinned_torch_count_oracle(tmp_path):
    python = os.environ.get("MERLIN_TORCH_PYTHON")
    if not python:
        if os.environ.get("MERLIN_LEAN_REQUIRED"):
            pytest.fail("pinned PyTorch interpreter required")
        pytest.skip("pinned PyTorch interpreter unavailable")
    samples = count_samples(32, 31)
    code = """import json,sys,numpy as np,torch
assert torch.__version__ == '2.10.0+cpu', torch.__version__
results=[]
for x,t in json.load(sys.stdin):
    x=torch.from_numpy(np.array([int(v) for v in x],dtype=np.uint32).view(np.float32))
    t=torch.from_numpy(np.array([int(v) for v in t],dtype=np.uint32).view(np.float32))
    results.append((t[None,:] <= x[:,None]).sum(dim=1).tolist())
print(json.dumps({'torch':torch.__version__,'numpy':np.__version__,'results':results}))
"""
    proc = subprocess.run(
        [python, "-c", code], input=json.dumps(samples), capture_output=True, text=True, check=True, timeout=60
    )
    result = json.loads(proc.stdout)
    for (x, thresholds), counts in zip(samples, result["results"], strict=True):
        xx, tt = [[struct.unpack("<f", struct.pack("<I", int(v)))[0] for v in a] for a in (x, thresholds)]
        assert counts == [sum(t <= value for t in tt) for value in xx]
    (tmp_path / "pinned-torch-counts.json").write_text(json.dumps(result, indent=2) + "\n")
