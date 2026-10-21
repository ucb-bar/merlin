/- Composition at an explicit frontend repair checkpoint. No source-refinement claim crosses
   the checkpoint: the reference is Intent(repair.before), at that exact recorded boundary. -/
import MerlinVerify.Rules.FrontendBoolProgram
import MerlinVerify.Check.Chain

namespace MerlinVerify.Rules.FrontendBoolChain

open MerlinVerify MerlinVerify.Check

theorem accepted_intent {before after : Module} (h : FrontendBoolProgram.check before after = true) :
    ∃ intended, FrontendBoolProgram.intent? before = some intended ∧
      ∀ f ∈ Module.definedNames intended,
        FuncRefinesC Sem.Ops.opSem (fun _ => True) (fun _ => True) after f intended f := by
  cases hi : FrontendBoolProgram.intent? before with
  | none => simp [FrontendBoolProgram.check, hi] at h
  | some intended => exact ⟨intended, rfl, (FrontendBoolProgram.check_sound hi h).2.2⟩

/-- Accepted repair followed by any chain of proved ordinary refinements. The context and input
domain are explicit, so typed suffixes compose without claiming a universal domain. The reference
is the reconstructed policy program at `before`, never the original potentially poison cast.
The IEEE/A2 interpretation of the repaired sites remains the separate `UneCorrect` obligation. -/
theorem chain_after_repair_refinesC {before after : Module} (suffix : List Module)
    (hrepair : FrontendBoolProgram.check before after = true)
    (ctxOk : Ctx → Prop) (D : List RVal → Prop) :
    ∃ intended, FrontendBoolProgram.intent? before = some intended ∧
      ∀ f ∈ Module.definedNames intended,
        ChainRefinesC Sem.Ops.opSem ctxOk D f (after :: suffix) →
        FuncRefinesC Sem.Ops.opSem ctxOk D
          ((after :: suffix).getLast (List.cons_ne_nil _ _)) f intended f := by
  obtain ⟨intended, hi, hr⟩ := accepted_intent hrepair
  refine ⟨intended, hi, fun f hf hs => ?_⟩
  have repair : FuncRefinesC Sem.Ops.opSem ctxOk D after f intended f :=
    fun ctx _ args _ => hr f hf ctx trivial args trivial
  exact FuncRefinesC.trans (chain_refinesC after suffix hs) repair

/-- Strict refinement additionally requires support of the *policy program*, on the same domain.
Support of the captured before program is not a substitute for this premise. -/
theorem chain_after_repair_strict {before after intended : Module} {suffix : List Module}
    {ctxOk : Ctx → Prop} {D : List RVal → Prop} {f : String}
    (hi : FrontendBoolProgram.intent? before = some intended)
    (hrepair : FrontendBoolProgram.check before after = true)
    (hf : f ∈ Module.definedNames intended)
    (hsuffix : ChainRefinesC Sem.Ops.opSem ctxOk D f (after :: suffix))
    (hsupport : FuncNotStuck Sem.Ops.opSem ctxOk D intended f) :
    FuncRefines Sem.Ops.opSem ctxOk D
      ((after :: suffix).getLast (List.cons_ne_nil _ _)) f intended f := by
  obtain ⟨reference, href, h⟩ := chain_after_repair_refinesC suffix hrepair ctxOk D
  have heq : reference = intended := Option.some.inj (href.symm.trans hi)
  subst reference
  exact FuncRefinesC.strict (h f hf hsuffix) hsupport

#print axioms chain_after_repair_refinesC
#print axioms chain_after_repair_strict

end MerlinVerify.Rules.FrontendBoolChain
