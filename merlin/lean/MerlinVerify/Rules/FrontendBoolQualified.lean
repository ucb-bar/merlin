/- Independent support and definedness qualification of the direct source-policy theorem.
No claim about the original numeric-cast semantics is made. -/
import MerlinVerify.Rules.FrontendBoolSimulation
import MerlinVerify.Check.Rulebook
import MerlinVerify.Check.TypedModule
import MerlinVerify.Rules.FrontendBoolDefined

namespace MerlinVerify.Rules.FrontendBoolQualified
open MerlinVerify MerlinVerify.Check

/-- Domains come from the actual source signature and must match the actual after signature.
A signature alone is insufficient: the existing body safety checker must also accept. -/
def support (ib : Nat) (src tgt : Module) (fn : String) : Bool :=
  match TypedModule.signature? ib src fn with
  | none => false
  | some (a, r) => TypedModule.signature? ib tgt fn == some (a, r) &&
      supportCheck (merlinBook ib) tgt fn a r

def defined (ib : Nat) (src tgt : Module) (fn : String) : Bool :=
  match TypedModule.signature? ib src fn with
  | none => false
  | some (a, r) => TypedModule.signature? ib tgt fn == some (a, r) &&
      Definedness.check (FrontendBoolDefined.book ib) tgt fn a r

/-- Acceptance of repair and support derives policy support on the source's declared domain.
No original-capture support receipt is substituted for this proof. -/
theorem support_sound {ib : Nat} {src tgt : Module} {fn : String}
    (hr : FrontendBoolSimulation.check src tgt = true) (hs : support ib src tgt fn = true) :
    ∃ a r, TypedModule.signature? ib src fn = some (a, r) ∧
      ∀ ctx, ctx.indexBits = ib → FrontendBoolFptosi.UneCorrect ctx.M →
      ∀ args, HasTys ib args a →
        (FrontendBoolSimulation.policyDenote src ctx fn args).NotStuck ∧
        Refines ResultsRefine (denote Sem.Ops.opSem ctx tgt fn args)
          (FrontendBoolSimulation.policyDenote src ctx fn args) := by
  unfold support at hs
  cases hsig : TypedModule.signature? ib src fn with
  | none => simp [hsig] at hs
  | some pair =>
    obtain ⟨a, r⟩ := pair
    simp only [hsig, Bool.and_eq_true] at hs
    refine ⟨a, r, rfl, fun ctx hib hu args ha => ?_⟩
    have hn := supportCheck_sound (merlinBook ib) hs.2 ctx hib args ha
    have heq := FrontendBoolSimulation.check_sound hr ctx hu fn args
    rw [heq]
    refine ⟨hn, ?_⟩
    cases ho : denote Sem.Ops.opSem ctx tgt fn args with
    | ub => trivial
    | stuck reason => simp [ho, Outcome.NotStuck] at hn
    | ok vs => exact ResultsRefine.refl vs

/-- Successful nonpoison target execution is also successful nonpoison source-policy execution,
by exact program simulation, on typed nonpoison inputs. -/
theorem defined_sound {ib : Nat} {src tgt : Module} {fn : String}
    (hr : FrontendBoolSimulation.check src tgt = true) (hd : defined ib src tgt fn = true) :
    ∃ a r, TypedModule.signature? ib src fn = some (a, r) ∧
      ∀ ctx, ctx.indexBits = ib → FrontendBoolFptosi.UneCorrect ctx.M →
      ∀ args, HasTys ib args a → ResultsDefined args →
        ∃ results, FrontendBoolSimulation.policyDenote src ctx fn args = .ok results ∧
          denote Sem.Ops.opSem ctx tgt fn args = .ok results ∧ ResultsDefined results := by
  unfold defined at hd
  cases hsig : TypedModule.signature? ib src fn with
  | none => simp [hsig] at hd
  | some pair =>
    obtain ⟨a, r⟩ := pair
    simp only [hsig, Bool.and_eq_true] at hd
    refine ⟨a, r, rfl, fun ctx hib hu args ha hi => ?_⟩
    obtain ⟨results, hexec, hdefined⟩ := Definedness.check_sound (FrontendBoolDefined.book ib) hd.2 ctx hib args ⟨ha, hi⟩
    exact ⟨results, (FrontendBoolSimulation.check_sound hr ctx hu fn args).trans hexec,
      hexec, hdefined⟩

#print axioms support_sound
#print axioms defined_sound
end MerlinVerify.Rules.FrontendBoolQualified
