"""Preserve MLIR number min/max for signaling NaNs at LLVM finalization.

MLIR arith.maxnumf/minnumf select the number when only one input is NaN.
LLVM's older maxnum/minnum intrinsics may instead propagate a signaling NaN.
Explicit NaN selection restores the source rule. Keep the original intrinsic
for two numeric operands, preserving its signed-zero behavior and fast flags.
This is target-independent scalar/vector value legalization, without storage,
effect, rounding or source-provenance permissions.
"""

RUNNER_PRELUDE = r'''
def _merlin_number_minmax(module):
    from torch_mlir import ir as _nmm_ir
    from torch_mlir.dialects import llvm as _nmm_llvm

    def walk(operation):
        for region in operation.regions:
            for block in region.blocks:
                for child in list(block.operations):
                    yield child.operation
                    yield from walk(child.operation)

    candidates = [op for op in walk(module.operation)
                  if op.name in ('llvm.intr.maxnum', 'llvm.intr.minnum')]
    with module.context:
        for old in candidates:
            if len(old.operands) != 2 or len(old.results) != 1 or old.regions:
                raise ValueError('number min/max requires the actual binary LLVM intrinsic')
            lhs, rhs = old.operands
            if lhs.type != rhs.type or old.results[0].type != lhs.type:
                raise ValueError('number min/max changed its typed value interface')
            with old.location, _nmm_ir.InsertionPoint(old):
                original = _nmm_ir.Operation.create(old.name, results=[old.results[0].type],
                    operands=[lhs, rhs], attributes={key: old.attributes[key] for key in old.attributes})
                left_nan = _nmm_llvm.FCmpOp(_nmm_llvm.FCmpPredicate.uno, lhs, lhs).result
                right_nan = _nmm_llvm.FCmpOp(_nmm_llvm.FCmpPredicate.uno, rhs, rhs).result
                right_selected = _nmm_llvm.SelectOp(right_nan, lhs, original.results[0]).result
                selected = _nmm_llvm.SelectOp(left_nan, rhs, right_selected).result
            old.results[0].replace_all_uses_with(selected)
            old.erase()
        module.operation.verify()
    print('OK number_minmax', len(candidates))
'''
