/- Rewrite assumptions must be explicit. An IR annotation does not constrain arbitrary runtime
values in the current universal C1 theorem, and external calls have arbitrary declared meanings. -/
import MerlinVerify.Tests.Semantics

namespace MerlinVerify.Tests.RewriteDomains

open MerlinVerify MerlinVerify.Tests.Semantics

def observer : Op := mkOp "func.func" [] [] [] [("sym_name", .str "observe")]

def program (chained : Bool) : Module :=
  let casts := if chained then
    [mkOp "arith.extsi" [v 1 (.int 16)] [0], mkOp "arith.extsi" [v 2 i32] [1]]
    else [mkOp "arith.extsi" [v 2 i32] [0]]
  { ops := [observer, func "f" [v 0 i8]
    (casts ++ [mkOp "func.call" [v 3 i32] [2] [] [("callee", .symbolRef "observe" [])], ret [3]])] }

/-- The current external-function model is arbitrary; it supplies no poison-refinement law. -/
def observingCtx : Ctx := { ctx with ext := fun _ args =>
  match args with
  | [.scalar .poison] => .ok [.scalar (.some (.bits 32 7))]
  | _ => .ok [.scalar (.some (.bits 32 8))] }

-- This runtime argument deliberately violates the function's declared i8 type.
def wrongWidth : RVal := .scalar (.some (.bits 24 1))
#guard !(Scalar.conforms 64 i8 (.bits 24 1))

-- i24 -> i16 is poison; i24 -> i32 succeeds. The external observer can distinguish them.
#guard readInts (Sem.Ops.denote observingCtx (program true) "f" [wrongWidth]) == some [[some 7]]
#guard readInts (Sem.Ops.denote observingCtx (program false) "f" [wrongWidth]) == some [[some 8]]

-- On correctly typed inputs, this particular distinction disappears.
#guard readInts (Sem.Ops.denote observingCtx (program true) "f" [.scalar (.some (.bits 8 1))])
  == readInts (Sem.Ops.denote observingCtx (program false) "f" [.scalar (.some (.bits 8 1))])

end MerlinVerify.Tests.RewriteDomains
