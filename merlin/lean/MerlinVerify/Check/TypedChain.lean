/- Composition with one explicit input domain shared by universal and typed boundaries. -/
import MerlinVerify.Check.Chain

namespace MerlinVerify.Check

open MerlinVerify MerlinVerify.Sem.Ops

theorem chain_typed_refines {ib : Nat} {m₀ : Module} {ms : List Module} {f : String}
    {argTys resTys : List VTy}
    (hchain : ChainRefinesC opSem (fun ctx => ctx.indexBits = ib)
      (fun args => HasTys ib args argTys) f (m₀ :: ms))
    (hsupp : supportCheck (merlinBook ib) m₀ f argTys resTys = true) :
    FuncRefines opSem (fun ctx => ctx.indexBits = ib) (fun args => HasTys ib args argTys)
      ((m₀ :: ms).getLast (List.cons_ne_nil _ _)) f m₀ f :=
  FuncRefinesC.strict (chain_refinesC m₀ ms hchain) (supportCheck_sound (merlinBook ib) hsupp)

theorem chain_typed_defined_refines {ib : Nat} {m₀ : Module} {ms : List Module} {f : String}
    {argTys resTys : List VTy}
    (hchain : ChainRefinesC opSem (fun ctx => ctx.indexBits = ib)
      (fun args => HasTys ib args argTys) f (m₀ :: ms))
    (hdefined : Definedness.check (Definedness.book ib) m₀ f argTys resTys = true) :
    let D := fun args => HasTys ib args argTys ∧ ResultsDefined args
    let target := (m₀ :: ms).getLast (List.cons_ne_nil _ _)
    FuncRefines opSem (fun ctx => ctx.indexBits = ib) D target f m₀ f ∧
      FuncDefined opSem (fun ctx => ctx.indexBits = ib) D target f := by
  have hs := Definedness.definedCheck_sound hdefined
  have hc := chain_refinesC m₀ ms hchain
  have hr := FuncRefinesC.strict (fun ctx hctx args ha => hc ctx hctx args ha.1) hs.notStuck
  exact ⟨hr, hr.defined hs⟩

end MerlinVerify.Check
