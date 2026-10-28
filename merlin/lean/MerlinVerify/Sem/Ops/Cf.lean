/-
Op family: `cf.assert` (`mlir/include/mlir/Dialect/ControlFlow/IR/ControlFlowOps.td`).

MLIR: "If the argument is true this operation has no effect. Otherwise, the program execution will
abort." An abort is modelled as `ub` (ledger A1): every claim is conditional on the source's
assertions passing on the input. A poison condition is treated like a failing one. Branching `cf`
ops (multi-block control flow) are outside the evaluator's single-block scope and `stuck`.
-/
import MerlinVerify.Sem.Support

namespace MerlinVerify.Sem.Ops.Cf

open MerlinVerify

def sem : OpSem := fun _ _ op args _ =>
  match op.name, args with
  | "cf.assert", [.scalar c] =>
    match c with
    | .some (.bits 1 b) => if b = 1 then .ok [] else .ub
    | .poison => .ub
    | _ => .stuck "cf.assert condition is not i1"
  | name, _ => .stuck s!"unsupported op {name}"

end MerlinVerify.Sem.Ops.Cf
