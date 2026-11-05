# Library calls for quantized lowering

Every block below runs as written against the editable library (put `submission/compiler_library`
on `PYTHONPATH`); a test executes each one, so the calls and results shown are current. Two names
are yours to supply: `source_text`, the text of an MLIR module you read (a case's interface or a
file under `public_sources/`), and `output_path`, the `output_epilogue.output_path` mapping of
`target_contract.yaml` (`quant_params.py target --json` prints it).

## Reference arithmetic: check a stage on values you choose

```python
from merlin.runtime.commandbuffer import apply_declared_acc_scale
from merlin.runtime.tensor import Tensor

acc = Tensor((1, 4), [-40, -24, 8, 2047], "i32")
scaled = apply_declared_acc_scale(acc, {"acc_scale": 0.0625}, op="check")
assert scaled.to_i8().data == [-2, -2, 0, 127]          # half to even, then saturate
assert scaled.relu().to_i8().data == [0, 0, 0, 127]
assert Tensor((1, 3), [4, 12, -4], "i32").requant(3).data == [1, 2, 0]   # shift: ties round up
lhs, rhs = Tensor((1, 3), [3, -3, 1], "i8"), Tensor((1, 3), [1, 1, 0], "i8")
assert lhs.residual_add_scaled(rhs, 0.5, 0.5).data == [2, -1, 0]        # one rounding after the sum
```

## Fold or keep a chain of source scales

```python
from merlin.llvmlower.requantization import prove_scale, prove_scale_bound, quantized, synthesize_bias

proof = prove_scale([0.5, 0.0625], -10000, 10000)      # exact over the whole stated domain
assert proof["scale"] == 0.03125 and proof["rounding"] == "nearest_even"
assert all(quantized(a, [0.5, 0.0625]) == quantized(a, [proof["scale"]]) for a in range(-10000, 10001))
try:
    prove_scale([0.1, 0.1], -100000, 100000)            # folding would change an output
except ValueError as refused:
    assert "-12650" in str(refused)                      # the witness accumulator
bound = prove_scale_bound([0.1, 0.1], -100000, 100000)
assert bound["max_output_lsb_error"] == 1 and not bound["exact"]
bias = synthesize_bias([0.5, 0.0625], [0, 1, 0.01], 1, -10000, 10000)
assert bias["integer_bias"] == [0, 32, None]             # None: no exact integer bias for that channel
```

`prove_scale_bound` only reports an error; using a non-exact fold is a numerical policy decision the
case's tolerance has to allow.

## Exact i32 -> i8 readout: thresholds, multiplier/shift, column runs, host C

```python
from merlin.llvmlower.integer_readout import (
    derive, emit_readout, evaluate, exact_rational_candidate, fixedpoint_candidate, partition_column_scales,
)
from merlin.llvmlower.requantization import quantized

proof = derive([0.0625])                                 # whole i32 domain; derive([s], relu=True) clamps at 0
assert evaluate(100, proof) == quantized(100, [0.0625]) == 6
rational = exact_rational_candidate([0.0625])            # exact integer multiply/shift, or None
assert (rational["multiplier"], rational["shift"]) == (1, 4)
assert exact_rational_candidate([0.1]) is None           # then use thresholds or a fixed-point estimate
estimate = fixedpoint_candidate(proof)                    # off by at most one step, corrected by thresholds
assert estimate["max_estimate_output_error"] <= 1
runs = partition_column_scales([0.5] * 16 + [0.125] * 16, 32, relu=True)
assert [(r["first_column"], r["columns"]) for r in runs["regions"]] == [(0, 16), (16, 16)]
code = emit_readout(proof, "readout_q", fixedpoint=True, saturation_first=True, packet=8)
assert "void readout_q(const int32_t*acc,int8_t*out,size_t count)" in code
```

## Accumulator bounds

```python
from merlin.llvmlower.integer_producer_range import IntegerSumProductsRange
from merlin.llvmlower.integer_readout import derive, emit_readout

domain = IntegerSumProductsRange(terms=2304, lhs_min=-128, lhs_max=127, rhs_min=-128, rhs_max=127)
assert domain.interval() == (-37453824, 37748736)        # every prefix checked against i32
try:
    IntegerSumProductsRange(200000, -128, 127, -128, 127).interval()
except ValueError as refused:
    assert "overflow" in str(refused)
proof = derive([0.125, 0.75], *domain.interval())
code = emit_readout(proof, "bounded", producer_range=domain)   # the bound replaces the per-element trap
assert "guaranteed by the bound producer contract" in code
```

## Read a captured requantization and decide what the readout can absorb

```python
from merlin.common import mlir_query as mq
from merlin.llvmlower.integer_gemm import match_integer_gemm
from merlin.llvmlower.requantization_match import match
from merlin.targetgen import output_path as OP
from merlin.xdsl_dialects.lowering import output_fusion as OF

module = mq.parse(source_text)
root = next(op for op in module.walk() if match_integer_gemm(op) is not None)
chain = OF.source_chain(root, OF.Values())               # OF.Values(capture_dir) also reads stored tensors
print([stage.kind for stage in chain.stages], chain.unread)
dims = match_integer_gemm(root)
extents = {id(root): (dims.batch * dims.m, dims.k, dims.n, 2)}
decision = OF.decide(chain, OP.validate(output_path), OF.Values(), extents)
print(decision.cut, decision.stages, [(r["cut"], r["code"]) for r in decision.refusals])
for epilogue in (op for op in module.walk() if mq.op_name(op) == "linalg.generic"):
    try:
        fused = match(epilogue)                          # one pointwise i32 -> i8 scale/round/saturate region
    except ValueError as refused:                        # any other form, with the reason
        continue
    print(fused.dimensions, fused.integer_bias, fused.proof["scale"])
```

`decision.cut` is `pool`, `requant`, `accumulator` or `bare` (contraction only); every refusal
carries its reason. In a compiler, `OF.decisions(groups, module, path, values)` applies the same
rule to every accelerator group (skill `fusion`). `match` refuses anything but one canonical
pointwise region; `merlin.llvmlower.captured_requant.inspect_chain(root)` reads the multi-op
captured form (two scales, a channel bias, optional ReLU, then quantize).
