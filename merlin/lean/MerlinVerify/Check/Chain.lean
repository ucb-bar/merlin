/-
Composition of per-boundary claims (design C10).

A compilation is a chain of recorded boundaries `m₀ → m₁ → … → mₙ`, each `after` the next
`before`. Every boundary's checker proves `FuncRefinesC mₖ₊₁ mₖ` for the function it covers
(`moduleCheck_sound`, `c1Check_sound`). `ChainRefinesC` is their conjunction; `chain_refinesC`
composes it into `FuncRefinesC mₙ m₀`, and `chain_refines` adds the support of the capture `m₀`
(`supportCheck_sound`) to conclude strict refinement on typed inputs.
-/
import MerlinVerify.Check.Rulebook

namespace MerlinVerify.Check

open MerlinVerify MerlinVerify.Sem.Ops

variable {sem : OpSem} {ctxOk : Ctx → Prop} {D : List RVal → Prop}

/-- Every boundary of the chain `m₀, m₁, …` is a conditional refinement of its predecessor. -/
def ChainRefinesC (sem : OpSem) (ctxOk : Ctx → Prop) (D : List RVal → Prop) (f : String) :
    List Module → Prop
  | a :: b :: rest => FuncRefinesC sem ctxOk D b f a f ∧ ChainRefinesC sem ctxOk D f (b :: rest)
  | _ => True

theorem RefinesC.refl {α : Type} {R : α → α → Prop} (hR : ∀ a, R a a) :
    ∀ o : Outcome α, RefinesC R o o
  | .ub => Or.inr trivial
  | .stuck r => Or.inl ⟨r, rfl⟩
  | .ok a => Or.inr (hR a)

theorem FuncRefinesC.refl (m : Module) (f : String) : FuncRefinesC sem ctxOk D m f m f :=
  fun _ _ _ _ => RefinesC.refl ResultsRefine.refl _

/-- **Composition.** A chain of conditional refinements composes into one, from the last module to
the first. -/
theorem chain_refinesC {f : String} :
    ∀ (m₀ : Module) (ms : List Module), ChainRefinesC sem ctxOk D f (m₀ :: ms) →
      FuncRefinesC sem ctxOk D ((m₀ :: ms).getLast (List.cons_ne_nil _ _)) f m₀ f
  | m₀, [], _ => FuncRefinesC.refl m₀ f
  | m₀, m₁ :: ms, ⟨h₀, hrest⟩ => by
    rw [List.getLast_cons (List.cons_ne_nil _ _)]
    exact FuncRefinesC.trans (chain_refinesC m₁ ms hrest) h₀

/-- **End to end.** A chain from the capture `m₀` whose boundaries are all conditional refinements,
with `m₀` inside the proved fragment, makes the last module a strict refinement of the capture on
typed inputs. -/
theorem chain_refines {ib : Nat} {m₀ : Module} {ms : List Module} {f : String}
    {argTys resTys : List VTy}
    (hchain : ChainRefinesC opSem (fun _ => True) (fun _ => True) f (m₀ :: ms))
    (hsupp : supportCheck (merlinBook ib) m₀ f argTys resTys = true) :
    FuncRefines opSem (fun ctx => ctx.indexBits = ib) (fun args => HasTys ib args argTys)
      ((m₀ :: ms).getLast (List.cons_ne_nil _ _)) f m₀ f :=
  funcRefines_of_support (chain_refinesC m₀ ms hchain) hsupp

end MerlinVerify.Check
