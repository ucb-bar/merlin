"""Small public integration seam: checked source schedule -> ordinary LLVM.

The caller first obtains DenseSourceResult with schedule_source_dense, using its
selected facts/software, assigned storage and independently justified target
adapter. No target implementation, preferred block size or schedule is provided.

    result = schedule_source_dense(module, contraction, ...)
    function, receipt = emit_dense_function(result, lower_instruction, symbol="kernel")
    # Add the function and target declarations to the ordinary emission module,
    # verify that module, translate/link it, then run independent target checks.

lower_instruction(builder, pointers, instruction, operands) receives the original
logical pointer arguments and actual i64 SSA operands of each checked Call.
It owns target encoding/completion. It must not rebuild outer loops, infer tail
sizes, reseed reductions, move stores, or claim numeric qualification from this
receipt. Exceptions invalidate the partial output; do not reuse that builder.
"""

from xdsl.dialects import llvm

from merlin.llvmlower.host_builder import FnBuilder, PTR


def emit_dense_function(result, lower_instruction, *, symbol):
    if not callable(lower_instruction):
        raise ValueError("a target instruction implementation is required")
    arguments = result.compiled.kernel.args
    builder = FnBuilder([PTR] * len(arguments))
    pointers = {argument.name: value for argument, value in zip(arguments, builder.entry.args, strict=True)}
    report = result.emit_llvm(
        builder,
        lambda instruction, operands: lower_instruction(builder, pointers, instruction, operands),
    )
    function = llvm.FuncOp(symbol, llvm.LLVMFunctionType([PTR] * len(arguments), llvm.LLVMVoidType()),
                           linkage=llvm.LinkageAttr("external"), body=builder.finish())
    # Symbol/callee verification belongs to the containing emission module;
    # a detached function cannot resolve the target's external declarations.
    return function, report
