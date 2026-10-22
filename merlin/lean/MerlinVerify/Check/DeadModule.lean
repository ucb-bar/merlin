/- Exact dead-producer normalization closes across internal calls. Only facts inferred
from actual prefix producers establish totality; arbitrary function inputs stay arbitrary. -/
import MerlinVerify.Check.InferredDead
import MerlinVerify.Check.ModuleBodyEquality

namespace MerlinVerify.Check.DeadModule

open MerlinVerify Sem.Ops

def region (ib : Nat) : Region → Region
  | .mk blocks => .mk (blocks.map fun
      | .mk args ops => .mk args (InferredDead.normalize (Definedness.book ib)
          (args.map fun value => (value.id, VTy.scalar Ty.index)) [] ops))

theorem region_equal (ib : Nat) (ctx : Ctx) (width : ctx.indexBits = ib) (call : CallSem)
    (env : Env) (body : Region) :
    evalRegion opSem ctx call env (region ib body) = evalRegion opSem ctx call env body := by
  cases body with
  | mk blocks =>
    cases blocks with
    | nil => rfl
    | cons block rest =>
      cases rest with
      | cons _ _ => rfl
      | nil =>
        cases block with
        | mk args ops =>
          funext values
          simp only [region, List.map_cons, List.map_nil, evalRegion, evalBlocks, evalBlock]
          congr 1
          funext bound
          exact InferredDead.normalize_equal (Definedness.book ib) _ [] ctx call bound ops width
            (by intro _ _ impossible; cases impossible)

def bodies (ib : Nat) : ModuleBodyEquality.Bodies opSem (fun ctx => ctx.indexBits = ib) where
  check left right := alphaRegion [] (region ib left) (region ib right)
  sound {left right} checked ctx width call env := by
    rw [← region_equal ib ctx width call env left, ← region_equal ib ctx width call env right]
    exact alphaRegion_sound call call [] _ _ ctx env env checked
      (by intro _ _ impossible; simp [Rho.find, List.lookup] at impossible) (fun _ _ => rfl)

/-- Retain every module and function header field except the known unit-valued C-interface
request. The surrounding policy chain separately accounts for that ABI annotation. -/
def headers (m : Module) : Module :=
  { m with ops := m.ops.map fun op => match op with
      | .mk name results operands attrs props regions successors =>
        if name = "func.func" then
          let keep := fun (entry : String × Attr) => !(entry.1 == "llvm.emit_c_interface" && entry.2 == .unit)
          .mk name results operands (attrs.filter keep) (props.filter keep) [] successors
        else .mk name results operands attrs props regions successors }

def check (ib : Nat) (before after : Module) : Bool :=
  ExactSyntax.modules (headers before) (headers after) && ModuleBodyEquality.check (bodies ib) before after

theorem check_sound {ib : Nat} {before after : Module} (checked : check ib before after = true) :
    ∀ ctx, ctx.indexBits = ib → ∀ fn args, denote opSem ctx before fn args = denote opSem ctx after fn args := by
  simp only [check, Bool.and_eq_true] at checked
  exact ModuleBodyEquality.check_sound (bodies ib) checked.2

#print axioms region_equal
#print axioms bodies
#print axioms check_sound

end MerlinVerify.Check.DeadModule
