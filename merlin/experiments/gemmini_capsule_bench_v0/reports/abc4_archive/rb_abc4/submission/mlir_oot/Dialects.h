#ifndef GEMMINI_OOT_DIALECTS_H
#define GEMMINI_OOT_DIALECTS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// ---- merlin_iface dialect ----
#include "IfaceDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "IfaceTypes.h.inc"

#define GET_OP_CLASSES
#include "IfaceOps.h.inc"

// ---- gemmini dialect ----
#include "GemminiDialect.h.inc"

#define GET_OP_CLASSES
#include "GemminiOps.h.inc"

#endif // GEMMINI_OOT_DIALECTS_H
