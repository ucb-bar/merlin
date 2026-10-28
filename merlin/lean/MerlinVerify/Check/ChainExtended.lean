/- Source definedness composes with conditional boundaries independently of the checker that
established it. Extended receipts name corollaries with the actual extended-check premise. -/
import MerlinVerify.Check.TypedChain
import MerlinVerify.Check.DefinedExtended

namespace MerlinVerify.Check

open MerlinVerify MerlinVerify.Sem.Ops

/-- The common semantic composition needs a proved defined source, never a receipt verdict. -/
theorem chain_defined_of {sem : OpSem} {ctxOk : Ctx → Prop} {D : List RVal → Prop}
    {m₀ : Module} {ms : List Module} {f : String}
    (hchain : ChainRefinesC sem ctxOk D f (m₀ :: ms))
    (hdefined : FuncDefined sem ctxOk D m₀ f) :
    let target := (m₀ :: ms).getLast (List.cons_ne_nil _ _)
    FuncRefines sem ctxOk D target f m₀ f ∧ FuncDefined sem ctxOk D target f := by
  have hr := FuncRefinesC.strict (chain_refinesC m₀ ms hchain) hdefined.notStuck
  exact ⟨hr, hr.defined hdefined⟩

/-- Universal boundaries apply on the narrower typed, initialized-input definedness domain.
The hypothesis names the exact extended checker replayed by the receipt composer. -/
theorem chain_extended_defined_refines {ib : Nat} {m₀ : Module} {ms : List Module} {f : String}
    {argTys resTys : List VTy}
    (hchain : ChainRefinesC opSem (fun _ => True) (fun _ => True) f (m₀ :: ms))
    (hdefined : DefinedExtended.check ib m₀ f argTys resTys = true) :
    let D := fun args => HasTys ib args argTys ∧ ResultsDefined args
    let target := (m₀ :: ms).getLast (List.cons_ne_nil _ _)
    FuncRefines opSem (fun ctx => ctx.indexBits = ib) D target f m₀ f ∧
      FuncDefined opSem (fun ctx => ctx.indexBits = ib) D target f := by
  have hs := DefinedExtended.check_sound hdefined
  have hc := chain_refinesC m₀ ms hchain
  have hr := FuncRefinesC.strict (fun ctx _ args _ => hc ctx trivial args trivial) hs.notStuck
  exact ⟨hr, hr.defined hs⟩

/-- Typed boundaries and extended definedness use the same index width and runtime types.
Initialization restricts the input domain without changing any boundary's conditional claim. -/
theorem chain_typed_extended_defined_refines {ib : Nat} {m₀ : Module} {ms : List Module} {f : String}
    {argTys resTys : List VTy}
    (hchain : ChainRefinesC opSem (fun ctx => ctx.indexBits = ib)
      (fun args => HasTys ib args argTys) f (m₀ :: ms))
    (hdefined : DefinedExtended.check ib m₀ f argTys resTys = true) :
    let D := fun args => HasTys ib args argTys ∧ ResultsDefined args
    let target := (m₀ :: ms).getLast (List.cons_ne_nil _ _)
    FuncRefines opSem (fun ctx => ctx.indexBits = ib) D target f m₀ f ∧
      FuncDefined opSem (fun ctx => ctx.indexBits = ib) D target f := by
  have hs := DefinedExtended.check_sound hdefined
  have hc := chain_refinesC m₀ ms hchain
  have hr := FuncRefinesC.strict (fun ctx hctx args ha => hc ctx hctx args ha.1) hs.notStuck
  exact ⟨hr, hr.defined hs⟩

end MerlinVerify.Check
