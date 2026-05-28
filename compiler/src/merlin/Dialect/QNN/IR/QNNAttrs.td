#ifndef IREE_MERLIN_COMPILER_DIALECT_QNN_IR_QNNATTRS_TD_
#define IREE_MERLIN_COMPILER_DIALECT_QNN_IR_QNNATTRS_TD_

include "mlir/IR/AttrTypeBase.td"
include "QNNDialect.td"

// Phase 1: no custom attributes. Per-tensor quantization rides on tensor
// types via MLIR's builtin `quant.uniform` encoding. Per-channel quant
// (axis-scale-offset) gets a custom attribute when Phase 2's conversion
// patterns need to round-trip it from the source linalg IR; deferred until
// then to avoid storing un-hashable types in the attribute storage.
// (TableGen generates hash_combine calls that don't accept double or
// APFloat; the right encoding is bytes-of-double cast to int64_t which
// we'll add when needed.)

#endif // IREE_MERLIN_COMPILER_DIALECT_QNN_IR_QNNATTRS_TD_
