"""Neutral original provider for tests; this qualifies no target instruction.

The test runtime defines a pure persistent setter, synchronous work reading that
state, and an unknown clobber. Only this reviewed synthetic ABI is projected.
"""

from __future__ import annotations

import inspect
import json
import subprocess
from pathlib import Path

from xdsl.dialects import llvm
from xdsl.parser import Parser

from merlin.common.dense_float_ir import DenseFloatPrinter
from merlin.compile.scheduling.configuration import (
    ConfigurationAdapter,
    ConfigurationEffects,
    eliminate_redundant_configuration,
    verify_configuration_rewrite,
)
from merlin.llvmlower import toolchain
from merlin.llvmlower.host_transform_chain import (
    HostLLVMArtifact,
    HostLLVMEmission,
    HostLLVMStageContract,
    HostLLVMTransformChain,
    HostLLVMTransformStage,
)
from merlin.sched.ir import Call, Const, Kernel
from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context
from merlin.xdsl_dialects.lowering.integer_constant_eval import constant_integer

CLANG = toolchain.clang
TRANSLATOR = toolchain.mlir_translate
SOURCE = """declare void @set_register(i32, i32)
declare void @work(ptr, ptr, ptr, i64, i32)
declare void @unknown()
define void @kernel(ptr %x, ptr %history, ptr %state, i64 %n, i1 %choose) {
entry:
  call void @set_register(i32 0, i32 3)
  call void @set_register(i32 0, i32 3)
  call void @work(ptr %x, ptr %history, ptr %state, i64 %n, i32 3)
  call void @unknown()
  call void @set_register(i32 0, i32 3)
  call void @set_register(i32 0, i32 3)
  %second = getelementptr i32, ptr %history, i64 %n
  call void @work(ptr %x, ptr %second, ptr %state, i64 %n, i32 3)
  br i1 %choose, label %left, label %right
left:
  call void @set_register(i32 0, i32 5)
  br label %join
right:
  call void @set_register(i32 0, i32 5)
  br label %join
join:
  call void @set_register(i32 0, i32 5)
  %twice = mul i64 %n, 2
  %third = getelementptr i32, ptr %history, i64 %twice
  call void @work(ptr %x, ptr %third, ptr %state, i64 %n, i32 5)
  ret void
}
"""
RUNTIME = """#include <stdint.h>
#include <stdlib.h>
static int32_t scale;
void set_register(int32_t field,int32_t value){if(field)abort();scale=value;}
void unknown(void){scale=71;}
void work(const int32_t *x,uint32_t *out,uint32_t *state,uint64_t n,int32_t expected){
 if(scale!=expected)abort();
 for(uint64_t i=0;i<n;i++){out[i]=((uint32_t)x[i]+state[i])*(uint32_t)scale;state[i]++;}
}
"""


def run(argv):
    subprocess.run(list(map(str, argv)), check=True, capture_output=True, text=True, timeout=120)


def imported(source, work, *, context=None):
    path = work / "imported.mlir"
    run([TRANSLATOR(), "--import-llvm", "--mlir-print-op-generic", source, "-o", path])
    return Parser(context or make_llvm_context(), path.read_text()).parse_module()


def _call(op):
    if type(op) is llvm.ConstantOp:
        return Call("constant", ())
    if type(op) is llvm.CallOp and not op.attributes and not op.results and op.callee is not None:
        name = op.callee.root_reference.data
        if name == "set_register" and len(op.operands) == 2:
            values = tuple(constant_integer(value) for value in op.operands)
            if all(type(value) is int for value in values) and values[0] == 0:
                return Call(
                    "setter", tuple((key, Const(value)) for key, value in zip(("field", "value"), values, strict=True))
                )
        if name == "work" and len(op.operands) == 5:
            value = constant_integer(op.operands[-1])
            if type(value) is int:
                return Call("work", (("required", Const(value)),))
    return Call("opaque", ())


def _effects(call):
    if call.instr == "setter":
        return ConfigurationEffects(writes=(("scale", call.arg("value").value),), ordered=True, pure_write=True)
    if call.instr == "work":
        return ConfigurationEffects(reads=(("scale", call.arg("required").value),), ordered=True)
    return ConfigurationEffects(ordered=True) if call.instr == "constant" else None


ADAPTER = ConfigurationAdapter(_effects, "neutral-c-setter-abi-v1", "synthetic-runtime-source-only")


def blocks(module):
    return [block for op in module.body.block.ops if isinstance(op, llvm.FuncOp) for block in op.body.blocks]


def _kernel(ops):
    return Kernel("block", (), tuple(_call(op) for op in ops))


def verify(original, actual):
    """Independent omissions plus full SSA/CFG/body equivalence, not plan equality."""
    expected = original.clone()
    clones = dict(zip(original.walk(), expected.walk(), strict=True))
    before_blocks, after_blocks = blocks(original), blocks(actual)
    if len(before_blocks) != len(after_blocks):
        raise ValueError("actual LLVM control flow changed")
    removed = 0
    for before, after in zip(before_blocks, after_blocks, strict=True):
        source, output = list(before.ops), list(after.ops)
        cursor = 0
        for op in source:
            if cursor < len(output) and _call(op) == _call(output[cursor]):
                cursor += 1
            elif _call(op).instr == "setter":
                clones[op].parent.erase_op(clones[op])
            else:
                raise ValueError("actual LLVM changed an unknown or nonsetter operation")
        if cursor != len(output):
            raise ValueError("actual LLVM inserted operations")
        report = verify_configuration_rewrite(_kernel(source), _kernel(output), ADAPTER)
        removed += report["removed_calls"]
    if not expected.is_structurally_equivalent(actual):
        raise ValueError("actual LLVM SSA, CFG, attributes or body changed")
    return removed


def chain(root, *, enabled=True):
    root.mkdir(parents=True)
    runtime = root / "runtime.c"
    runtime.write_text(RUNTIME)

    def transform(source, work):
        module = imported(source, work)
        for block in blocks(module):
            ops = list(block.ops)
            rewritten = eliminate_redundant_configuration(_kernel(ops), ADAPTER, enabled=enabled).kernel
            cursor = 0
            for op, call in zip(ops, _kernel(ops).body, strict=True):
                if cursor < len(rewritten.body) and call == rewritten.body[cursor]:
                    cursor += 1
                else:
                    block.erase_op(op)
        typed, selected = work / "selected.mlir", work / "selected.ll"
        with typed.open("w") as stream:
            DenseFloatPrinter(stream=stream, print_generic_format=True).print_op(module)
        run([TRANSLATOR(), "--mlir-to-llvmir", typed, "-o", selected])
        return selected

    def check(source, selected, work):
        original_dir, actual_dir = work / "original", work / "actual"
        original_dir.mkdir()
        actual_dir.mkdir()
        context = make_llvm_context()
        removed = verify(
            imported(source, original_dir, context=context), imported(selected, actual_dir, context=context)
        )
        evidence = work / "actual-emission.json"
        evidence.write_text(json.dumps({"actual_removed_setters": removed, "scope": "neutral synthetic ABI only"}))
        return HostLLVMEmission(
            HostLLVMArtifact.capture(source), HostLLVMArtifact.capture(selected), (HostLLVMArtifact.capture(evidence),)
        )

    sources = tuple(
        HostLLVMArtifact.capture(path.resolve())
        for path in dict.fromkeys(
            (
                *(
                    Path(inspect.getsourcefile(owner))
                    for owner in (
                        transform,
                        ConfigurationAdapter,
                        eliminate_redundant_configuration,
                        verify_configuration_rewrite,
                        make_llvm_context,
                        constant_integer,
                        DenseFloatPrinter,
                        Parser,
                        llvm.FuncOp,
                        Kernel,
                    )
                ),
                CLANG(),
                TRANSLATOR(),
            )
        )
    )
    contract = HostLLVMStageContract("neutral_configuration", sources, (HostLLVMArtifact.capture(runtime),))
    stage = HostLLVMTransformStage(contract, transform, check)
    return HostLLVMTransformChain((contract,), (stage,)), runtime
