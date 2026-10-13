/-
G0-FROZEN dispatch. The concrete op semantics `opSem` routes each op to its family by name.
Families live in `Sem/Ops/*.lean`, one owner each, so semantics work parallelizes without editing a
shared file. Adding a family requires an interface change request.
-/
import MerlinVerify.Sem.Ops.Arith
import MerlinVerify.Sem.Ops.Linalg
import MerlinVerify.Sem.Ops.LinalgNamed
import MerlinVerify.Sem.Ops.Tensor
import MerlinVerify.Sem.Ops.Func

namespace MerlinVerify.Sem.Ops

/-- The dialect prefix of an op name (`"arith"` for `"arith.addi"`). -/
def dialectOf (name : String) : String := (name.splitOn ".").headD ""

def opSem : OpSem := fun ctx call op args regions =>
  match dialectOf op.name with
  | "arith" | "math" => Arith.sem ctx call op args regions
  | "linalg" =>
    if op.name == "linalg.generic" || op.name == "linalg.fill" || op.name == "linalg.index" then
      Linalg.sem ctx call op args regions
    else LinalgNamed.sem ctx call op args regions
  | "tensor" => Tensor.sem ctx call op args regions
  | "func" => Func.sem ctx call op args regions
  | _ => .stuck s!"no semantics for dialect of {op.name}"

/-- The meaning of function `name` in module `m` under Merlin's op semantics. -/
def denote (ctx : Ctx) (m : Module) (name : String) (args : List RVal) : Outcome (List RVal) :=
  MerlinVerify.denote opSem ctx m name args

end MerlinVerify.Sem.Ops
