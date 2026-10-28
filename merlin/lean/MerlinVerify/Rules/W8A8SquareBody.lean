/- Exact source square-region execution. A whole-region check establishes repeated SSA
reads of the same argument; equal numerical classes alone do not establish equal operands. -/
import MerlinVerify.Check.PolicyW8A8Scalar

namespace MerlinVerify.Rules.W8A8SquareBody

open MerlinVerify MerlinVerify.Check

def body (unusedConstant : Bool) : List Op :=
  (if unusedConstant then
    [.mk "arith.constant" [{ id := 2, ty := .float .f32 }] []
      [("value", .float 0x40000000 (.float .f32))] [] [] []]
  else []) ++
  [.mk "arith.mulf" [{ id := 3, ty := .float .f32 }] [0, 0]
    [("fastmath", .enum "arith" "fastmath" "none")] [] [] [],
   .mk "linalg.yield" [] [3] [] [] [] []]

def region (unusedConstant : Bool) : Region :=
  .mk [.mk [{ id := 0, ty := .float .f32 }, { id := 1, ty := .float .f32 }] (body unusedConstant)]

theorem execution (unusedConstant : Bool) (ctx : Ctx) (call : CallSem) (env : Env)
    (x : FBits .f32) (initial : Val) :
    evalRegion Sem.Ops.opSem ctx call env (region unusedConstant)
      [.scalar (.some (.flt .f32 x)), .scalar initial] =
      .ok [.scalar (.some (.flt .f32 (ctx.M.op (Spec.Dequant.defaultKey "arith.mulf") .f32 [x, x])))] := by
  cases unusedConstant <;> unfold region body <;> w8_scalar_simp

def check (actual : Region) : Bool :=
  alphaRegion [] (region false) actual || alphaRegion [] (region true) actual

theorem check_sound {actual : Region} (checked : check actual = true)
    (ctx : Ctx) (call : CallSem) (env : Env) (x : FBits .f32) (initial : Val) :
    evalRegion Sem.Ops.opSem ctx call env actual [.scalar (.some (.flt .f32 x)), .scalar initial] =
      .ok [.scalar (.some (.flt .f32 (ctx.M.op (Spec.Dequant.defaultKey "arith.mulf") .f32 [x, x])))] := by
  simp only [check, Bool.or_eq_true] at checked
  rcases checked with minimal | extra
  · have same := alphaRegion_sound call call [] (region false) actual ctx env env minimal
      (by intro a b found; simp [Rho.find, List.lookup] at found) (fun _ _ => rfl)
    rw [← same]
    exact execution false ctx call env x initial
  · have same := alphaRegion_sound call call [] (region true) actual ctx env env extra
      (by intro a b found; simp [Rho.find, List.lookup] at found) (fun _ _ => rfl)
    rw [← same]
    exact execution true ctx call env x initial

#print axioms execution
#print axioms check_sound

end MerlinVerify.Rules.W8A8SquareBody
