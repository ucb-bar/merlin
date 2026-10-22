"""Actual position gather: universal bounds check and native/pinned-Torch observations.

The large table is produced inside MLIR, keeping the native ABI small. Native
samples are interpretation tests; the Lean checker proves initialized execution
symbolically and does not evaluate the large table through its literal evaluator.
"""

from __future__ import annotations

import json
import os
import subprocess

import pytest
from test_lean_frontend_ranges import checked
from test_lean_frontend_runtime_checkpoint import exported

from merlin.common.paths import merlin_dir
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.verify.lean import differential as diff
from merlin.xdsl_dialects._common import text as to_text

SPECIAL = [0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0xBF400000, 0x3F000000, 0xBFA00000]
OBSERVED = [(i, 0) for i in range(8)] + [(1023, 767), (511, 13), (32, 31), (0, 767)]


def embedding_observer(*, offset=0) -> str:
    from xdsl.dialects import builtin, func

    cases = []
    selected = "%ordinary"
    for i, bits in enumerate(SPECIAL):
        cases.append(
            f"%r{i} = arith.constant {i} : index\n%f{i} = arith.constant 0x{bits:08X} : f32\n"
            f"%is{i} = arith.cmpi eq, %row, %r{i} : index\n"
            f"%s{i} = arith.select %is{i}, %f{i}, {selected} : f32"
        )
        selected = f"%s{i}"
    source = """builtin.module {func.func @subject(%mask: tensor<1x1024xi1>)
      -> (tensor<1x1024xi64>, tensor<1024x768xf32>) {
      %z = arith.constant 0 : i64
      %offset = arith.constant OFFSET : i64
      %pe = tensor.empty() : tensor<1x1024xi64>
      %positions = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,
        affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel", "parallel"]}
        ins(%mask : tensor<1x1024xi1>) outs(%pe : tensor<1x1024xi64>) {
        ^bb0(%bit: i1, %old: i64):
          %i = linalg.index 1 : index
          %j = arith.index_cast %i : index to i64
          %shifted = arith.addi %j, %offset : i64
          %v = arith.select %bit, %shifted, %z : i64
          linalg.yield %v : i64 } -> tensor<1x1024xi64>
      %te = tensor.empty() : tensor<1024x768xf32>
      %table = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>],
        iterator_types = ["parallel", "parallel"]} outs(%te : tensor<1024x768xf32>) {
        ^bb0(%old: f32):
          %row = linalg.index 0 : index
          %col = linalg.index 1 : index
          %ri = arith.index_cast %row : index to i64
          %ci = arith.index_cast %col : index to i64
          %rf = arith.sitofp %ri : i64 to f32
          %cf = arith.sitofp %ci : i64 to f32
          %eighth = arith.constant 0.125 : f32
          %scaled = arith.mulf %cf, %eighth : f32
          %ordinary = arith.addf %rf, %scaled : f32
          CASES
          %zero = arith.constant 0 : index
          %first = arith.cmpi eq, %col, %zero : index
          %v = arith.select %first, SELECTED, %ordinary : f32
          linalg.yield %v : f32 } -> tensor<1024x768xf32>
      return %positions, %table : tensor<1x1024xi64>, tensor<1024x768xf32> }}"""
    source = source.replace("OFFSET", str(offset)).replace("CASES", "\n".join(cases)).replace("SELECTED", selected)
    module = parse_mlir_text(source)
    fn = next(op for op in module.body.block.ops if isinstance(op, func.FuncOp))
    block = fn.body.block
    ret = block.last_op
    positions, table = ret.operands
    block.erase_op(ret)
    capture = merlin_dir() / "contract/capsules/model/SY_model_smolvla/capsule.interface.mlir"
    original = parse_mlir_text(capture.read_text())
    forward = next(op for op in original.body.block.ops if isinstance(op, func.FuncOp))
    original_ops = list(forward.body.block.ops)
    gather = original_ops[117]
    mapping = {gather.operands[0]: positions, forward.body.block.args[2]: table}
    for op in original_ops[116:118]:
        block.add_op(op.clone(value_mapper=mapping))
    result = list(block.ops)[-1].results[0]
    lines, values = [], []
    for i, (row, col) in enumerate(OBSERVED):
        lines += [
            f"%r{i} = arith.constant {row} : index",
            f"%c{i} = arith.constant {col} : index",
            f"%v{i} = tensor.extract %embedding[%zero, %r{i}, %c{i}] : tensor<1x1024x768xf32>",
        ]
        values.append(f"%v{i}")
    observe = parse_mlir_text(
        "builtin.module {func.func @observe(%embedding: tensor<1x1024x768xf32>) -> ("
        + ",".join(["f32"] * len(values))
        + ") {%zero = arith.constant 0 : index\n"
        + "\n".join(lines)
        + "\nreturn "
        + ",".join(values)
        + " : "
        + ",".join(["f32"] * len(values))
        + "}}"
    )
    tail = next(op for op in observe.body.block.ops if isinstance(op, func.FuncOp))
    mapping = {tail.body.block.args[0]: result}
    for op in tail.body.block.ops:
        block.add_op(op.clone(value_mapper=mapping))
    fn.properties["function_type"] = builtin.FunctionType.from_lists(
        list(fn.function_type.inputs), [builtin.f32] * len(values)
    )
    module.verify()
    return to_text(module, generic=True)


def oracle(samples, work) -> dict:
    python = os.environ.get("MERLIN_TORCH_PYTHON")
    if not python:
        if os.environ.get("MERLIN_LEAN_REQUIRED"):
            pytest.fail("pinned PyTorch interpreter required")
        pytest.skip("pinned PyTorch interpreter unavailable")
    code = """import json,sys,numpy as np,torch
assert torch.__version__ == '2.10.0+cpu', torch.__version__
spec=json.load(sys.stdin)
table=torch.arange(1024,dtype=torch.float32)[:,None]+torch.arange(768,dtype=torch.float32)[None,:]*0.125
table[:8,0]=torch.from_numpy(np.array(spec['special'],dtype=np.uint32).view(np.float32))
results=[]
for [mask] in spec['samples']:
    positions=torch.where(torch.tensor([int(x) for x in mask],dtype=torch.bool),torch.arange(1024),0)
    embedded=table[positions]
    values=[int(embedded[row,col].numpy().view(np.uint32)) for row,col in spec['observed']]
    results.append(values)
print(json.dumps({'torch':torch.__version__,'numpy':np.__version__,'results':results}))
"""
    proc = subprocess.run(
        [python, "-c", code],
        input=json.dumps({"special": SPECIAL, "samples": samples, "observed": OBSERVED}),
        text=True,
        capture_output=True,
        timeout=60,
        check=True,
    )
    result = json.loads(proc.stdout)
    (work / "pinned-torch-embedding.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def test_actual_embedding_bounds_native_and_pinned_oracle(tmp_path, fidelity_tools):
    source = embedding_observer()
    assert checked(source, tmp_path / "proof", fidelity_tools)["accepted"]
    document = json.loads(exported(source, tmp_path / "native-input", fidelity_tools).read_text())
    inputs, outputs = diff.signature(document, "subject")
    samples = [[["1"] * 1024], [["0"] * 1024], [[str(i % 2) for i in range(1024)]]]
    native = diff.execute_mlir(
        tmp_path / "native-input/subject.mlir",
        samples,
        tmp_path / "native",
        fidelity_tools,
        inputs=inputs,
        outputs=outputs,
    )
    reference = oracle(samples, tmp_path)
    for result, expected in zip(native["results"], reference["results"], strict=True):
        assert [int(v) for v in result] == expected


@pytest.mark.parametrize("offset", [-1, 1])
def test_actual_embedding_with_invalid_position_producer_refuses(tmp_path, fidelity_tools, offset):
    assert not checked(embedding_observer(offset=offset), tmp_path, fidelity_tools)["accepted"]
