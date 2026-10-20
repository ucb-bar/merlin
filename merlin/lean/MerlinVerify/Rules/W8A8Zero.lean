/- Matrix i32 specialization of the shared checked literal initializer. -/
import MerlinVerify.Rules.W8A8Fill

namespace MerlinVerify.Rules.W8A8Zero

open MerlinVerify MerlinVerify.Sem.Ops.Linalg

def zero : Val := .some (.bits 32 0#32)

theorem zero_eq : zero = .some (.bits 32 0) := by decide

def matrixTy (m n : Nat) : Ty := W8A8Fill.tensorTy [m, n] (.int 32)

def fillRegion : Region := W8A8Fill.fillRegion (.int 32)

def operations (m n : Nat) : List Op :=
  W8A8Fill.operations [m, n] (.int 32) (.int 0 (.int 32))

def returned (id : Nat) : Op := W8A8Fill.returned id

theorem fillRegion_sound (ctx : Ctx) (call : CallSem) (env : Env) (value old : Val) :
    evalRegion Sem.Ops.opSem ctx call env fillRegion [.scalar value, .scalar old] =
      .ok [.scalar value] := W8A8Fill.fillRegion_sound ctx call env (.int 32) value old

theorem fill_sound (ctx : Ctx) (call : CallSem) (env : Env) (m n : Nat) :
    genericSem ctx [{ numDims := 2, numSyms := 0, results := [] }, identityMap 2]
      [.parallel, .parallel] [.scalar zero] [.tensor (Tensor.full [m, n] (.int 32) .poison)]
      (fun c => evalRegion Sem.Ops.opSem c call env fillRegion) =
      .ok [.tensor (Tensor.full [m, n] (.int 32) zero)] :=
  W8A8Fill.fill_sound ctx call env [m, n] (.int 32) zero
    (Tensor.full [m, n] (.int 32) .poison).get

theorem operations_sound (ctx : Ctx) (call : CallSem) (env : Env) (m n : Nat) :
    evalOps Sem.Ops.opSem ctx call env (operations m n ++ [returned 2]) =
      .ok [.tensor (Tensor.full [m, n] (.int 32) zero)] :=
  W8A8Fill.operations_sound ctx call env [m, n] (.int 32) (.int 0 (.int 32)) (.bits 32 0#32) rfl

#print axioms fill_sound
#print axioms operations_sound

end MerlinVerify.Rules.W8A8Zero
