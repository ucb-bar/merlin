/- Preserve legacy definedness while adding independently proved value-dependent bounds. -/
import MerlinVerify.Check.DefinedBounds

namespace MerlinVerify.Check.DefinedExtended

open MerlinVerify MerlinVerify.Check

def check (ib : Nat) (m : Module) (fn : String) (argTys resTys : List VTy) : Bool :=
  Definedness.check (Definedness.book ib) m fn argTys resTys ||
    Definedness.Bounds.check ib m fn argTys resTys

/-- Both routes prove the same typed, initialized-input domain. Neither success flag is trusted
without its own totality theorem, and the legacy obligation remains independently replayable. -/
theorem check_sound {ib : Nat} {m : Module} {fn : String} {argTys resTys : List VTy}
    (h : check ib m fn argTys resTys = true) :
    FuncDefined Sem.Ops.opSem (fun ctx => ctx.indexBits = ib)
      (fun args => HasTys ib args argTys ∧ ResultsDefined args) m fn := by
  simp only [check, Bool.or_eq_true] at h
  rcases h with hlegacy | hbounds
  · exact Definedness.definedCheck_sound hlegacy
  · exact Definedness.Bounds.check_sound hbounds

end MerlinVerify.Check.DefinedExtended
