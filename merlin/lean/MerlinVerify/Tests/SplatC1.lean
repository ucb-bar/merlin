/- Whole-program checks include producer removal, escaping full tensors, actual payload identity,
stale dependencies and target freshness. Raw AST probes cover malformed SSA rejected by MLIR. -/
import MerlinVerify.Check.SplatC1
import MerlinVerify.Tests.Semantics

namespace MerlinVerify.Tests.SplatC1

open MerlinVerify MerlinVerify.Check MerlinVerify.Tests.Semantics
open MerlinVerify.Sem.Ops.Linalg

def shaped : Ty := tensorTy [2] i32
def zeroRank : Ty := tensorTy [] i32

def splat (result payload : Nat) (ty : Ty) : Op :=
  mkOp "tensor.splat" [v result ty] [payload]

def generic (input : Nat) (compact : Bool) (capture : Bool := false) : Op :=
  .mk "linalg.generic" [v 4 shaped] [input, 1]
    [("indexing_maps", .array [
        .affineMap (if compact then { numDims := 1, numSyms := 0, results := [] }
          else identityMap 1), .affineMap (identityMap 1)]),
      ("iterator_types", .array [.enum "linalg" "iterator_type" "parallel"]),
      ("operandSegmentSizes", .denseArray (.int 32) [1, 1])]
    [] [.mk [.mk [v 5 i32, v 6 i32]
      [mkOp "linalg.yield" [] [if capture then 2 else 5]]]] []

def source (escape : Bool := false) (capture : Bool := false) : Module :=
  { ops := [func "f" [v 0 i32, v 1 shaped, v 2 i32]
    [splat 3 0 shaped, generic 3 false capture, ret (if escape then [4, 3] else [4])]] }

def target (escape : Bool := false) (capture : Bool := false) (payload : Nat := 0)
    (temporary : Nat := 7) : Module :=
  { ops := [func "f" [v 0 i32, v 1 shaped, v 2 i32]
    ((if escape then [splat 3 0 shaped] else []) ++
      [splat temporary payload zeroRank, generic temporary true capture,
       ret (if escape then [4, 3] else [4])])] }

def accepts (s t : Module) (escape : Bool := false) : Bool :=
  Check.SplatC1.check 64 s t "f" [.scalar i32, .tensor [2] i32, .scalar i32]
    (if escape then [.tensor [2] i32, .tensor [2] i32] else [.tensor [2] i32])

#guard accepts source target
#guard accepts (source true) (target true) true
#guard accepts (source true true) (target true true) true
#guard !accepts source (target (payload := 2))
#guard !accepts source (target (temporary := 1))
#guard !accepts (source true) target true
#guard !accepts (source true true) (target true false) true

-- A tensor fact depends on the actual scalar binding at production time. Rebinding that scalar
-- cannot retroactively change the old full tensor's payload.
def shadowed : Module :=
  { ops := [func "f" [v 0 i32, v 1 shaped, v 2 i32]
    [splat 3 0 shaped, mkOp "arith.constant" [v 0 i32] [] [("value", .int 9 i32)],
      generic 3 false, ret [4]]] }
#guard !accepts shadowed target

-- Bare scalar canonical forms are internal proof objects. The exporter and native differential
-- oracle receive only the actual shaped source and rank-zero target programs.
#guard (SplatNormalize.inline? [(0, .scalar i32), (1, .tensor [2] i32), (3, .tensor [2] i32)]
  [⟨3, 0, [2], i32⟩] (generic 3 false)).isSome
#guard (SplatNormalize.inline? [(0, .scalar i32), (1, .tensor [2] i32), (3, .tensor [2] i32)]
  [⟨3, 2, [2], i32⟩] (generic 3 false)).map (·.operands) == some [3, 1]

end MerlinVerify.Tests.SplatC1
