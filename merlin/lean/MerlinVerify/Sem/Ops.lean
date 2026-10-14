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

/-- The op with its SSA identity erased: name, attributes, properties and result types only.
Operands and regions reach the semantics as values and region meanings, never as syntax. -/
def _root_.MerlinVerify.Op.blind (op : Op) : Op :=
  .mk op.name (op.results.map fun r => { r with id := 0 }) [] op.attrs op.props [] op.successors

/-- Dispatch on the blinded op, so `opSem` cannot depend on SSA identifiers by construction
(`opSem_blind`): renaming values never changes an op's meaning. -/
def dispatch : OpSem := fun ctx call op args regions =>
  match dialectOf op.name with
  | "arith" | "math" => Arith.sem ctx call op args regions
  | "linalg" =>
    if op.name == "linalg.generic" || op.name == "linalg.fill" || op.name == "linalg.index" then
      Linalg.sem ctx call op args regions
    else LinalgNamed.sem ctx call op args regions
  | "tensor" => Tensor.sem ctx call op args regions
  | "func" => Func.sem ctx call op args regions
  | _ => .stuck s!"no semantics for dialect of {op.name}"

/-- The flat `callee` symbol of an op, if it has one. -/
def calleeOf (op : Op) : Option String :=
  match op.attr? "callee" with
  | some (.symbolRef root []) => some root
  | _ => none

/-- `call` restricted to the op's own callee: an op can only invoke the function it names. -/
def restrictCall (callee : Option String) (call : CallSem) : CallSem := fun name args =>
  if callee = some name then call name args else .stuck s!"op may not call @{name}"

/-- Op meaning: dispatch on the blinded op with `call` restricted to its callee. By construction
the meaning cannot depend on SSA identifiers (`opSem_blind`) or on functions the op does not name
(`opSem_call_local`). -/
def opSem : OpSem := fun ctx call op args regions =>
  dispatch ctx (restrictCall (calleeOf op.blind) call) op.blind args regions

theorem opSem_blind {ctx : Ctx} {call : CallSem} {op op' : Op} {args : List RVal}
    {regions : List (Ctx → RegionSem)} (h : op.blind = op'.blind) :
    opSem ctx call op args regions = opSem ctx call op' args regions := by
  simp only [opSem, h]

theorem opSem_call_local {ctx : Ctx} {call call' : CallSem} {op : Op} {args : List RVal}
    {regions : List (Ctx → RegionSem)}
    (h : ∀ c, calleeOf op.blind = some c → call c = call' c) :
    opSem ctx call op args regions = opSem ctx call' op args regions := by
  simp only [opSem]
  congr 1
  funext name xs
  simp only [restrictCall]
  split
  · next hc => rw [h name hc]
  · rfl

/-- The meaning of function `name` in module `m` under Merlin's op semantics. -/
def denote (ctx : Ctx) (m : Module) (name : String) (args : List RVal) : Outcome (List RVal) :=
  MerlinVerify.denote opSem ctx m name args

end MerlinVerify.Sem.Ops
