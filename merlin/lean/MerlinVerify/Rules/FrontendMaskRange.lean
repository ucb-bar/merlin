/- Selection preserves a proved interval on its value and seed tensors. This is
an execution postcondition for the actual accepted cut, not a shape assertion. -/
import MerlinVerify.Rules.FrontendMaskTyped
import MerlinVerify.Rules.FrontendTensorRanges

namespace MerlinVerify.Rules.FrontendMaskRange
open MerlinVerify MerlinVerify.Check Sem.Ops
open FrontendMaskWhole FrontendMaskTyped FrontendMaskProgram

/-- Decoding an arbitrary initialized tensor does not change its accessible words. -/
theorem words_property {shape : List Nat} {t : Tensor} (P : Val → Prop)
    (ht : RVal.HasTy 64 (.tensor t) (.tensor shape (.int 64))) (hd : RVal.Defined (.tensor t))
    (hp : FrontendMaskProperties.Elements P t) {i : Nat} (hi : i < shapeSize shape) :
    P (.some (.bits 64 (words t i))) := by
  have hpt := delinearize_inBounds shape i hi
  obtain ⟨x, hx⟩ := initialized_bits ht hd hpt
  have hp' := hp (Tensor.delinearize shape i) (ht.1.symm ▸ hpt)
  simpa only [words, ht.1, hx, word] using hp'

/-- The template executes with typed initialized inputs and preserves every
shared source/seed interval, including an empty compact tensor. -/
theorem body_exec {r : Definedness.IndexRanges.Interval} (limit : Nat) (ctx : Ctx) (call : CallSem)
    (mask values seed : Tensor)
    (hm : RVal.HasTy 64 (.tensor mask) (.tensor [1,32,32] (.int 1)))
    (hv : RVal.HasTy 64 (.tensor values) (.tensor [1,1024] (.int 64)))
    (hs : RVal.HasTy 64 (.tensor seed) (.tensor [1,1024] (.int 64)))
    (hdm : RVal.Defined (.tensor mask)) (hdv : RVal.Defined (.tensor values)) (hds : RVal.Defined (.tensor seed))
    (hrv : FrontendTensorRanges.Holds r (.tensor values)) (hrs : FrontendTensorRanges.Holds r (.tensor seed))
    (hib : ctx.indexBits = 64) (hu : FrontendBoolFptosi.UneCorrect ctx.M) (hl : 1024 ≤ limit) :
    ∃ out, evalRegion (FrontendMaskRuntime.sem limit) ctx call [] FrontendMaskTemplate.body
        [.tensor mask, .tensor values, .tensor seed] =
          .ok [.tensor out, index (FrontendMaskBounds.countFrom (bits mask) 1024 0)] ∧
      RVal.HasTy 64 (.tensor out) (.tensor [1,1024] (.int 64)) ∧
      RVal.Defined (.tensor out) ∧ FrontendTensorRanges.Holds r (.tensor out) := by
  obtain ⟨out, he, _, hd, ht, hp⟩ := body_inputs_exec_properties limit ctx call
    (bits mask) (words values) (words seed) mask values seed
    (mask_represents hm hdm) (word_represents hv hdv) (word_represents hs hds) hib hu hl
  exact ⟨out, he, ht, hd, hp (fun v => r.Holds (.scalar v))
    (fun i hi => words_property (fun v => r.Holds (.scalar v)) hv hdv hrv (by simpa [shapeSize] using hi))
    (fun i hi => words_property (fun v => r.Holds (.scalar v)) hs hds hrs (by simpa [shapeSize] using hi))⟩

/-- Actual complete-cut acceptance transports the interval postcondition through
all checked operations and uses. Bounds on the cut inputs remain explicit. -/
theorem check_sound {actual : Module} (h : FrontendMaskWhole.check actual = true)
    {r : Definedness.IndexRanges.Interval} (limit : Nat) (base : FloatModel) (ctx : Ctx)
    (mask values seed : Tensor)
    (hm : RVal.HasTy 64 (.tensor mask) (.tensor [1,32,32] (.int 1)))
    (hv : RVal.HasTy 64 (.tensor values) (.tensor [1,1024] (.int 64)))
    (hs : RVal.HasTy 64 (.tensor seed) (.tensor [1,1024] (.int 64)))
    (hdm : RVal.Defined (.tensor mask)) (hdv : RVal.Defined (.tensor values)) (hds : RVal.Defined (.tensor seed))
    (hrv : FrontendTensorRanges.Holds r (.tensor values)) (hrs : FrontendTensorRanges.Holds r (.tensor seed))
    (hib : ctx.indexBits = 64) (hl : 1024 ≤ limit) :
    ∃ out, denote (FrontendMaskRuntime.sem limit) {ctx with M := FrontendF32Comparison.model base}
        actual "subject" [.tensor mask, .tensor values, .tensor seed] =
          .ok [.tensor out, index (FrontendMaskBounds.countFrom (bits mask) 1024 0)] ∧
      RVal.HasTy 64 (.tensor out) (.tensor [1,1024] (.int 64)) ∧
      RVal.Defined (.tensor out) ∧ FrontendTensorRanges.Holds r (.tensor out) := by
  simp only [FrontendMaskWhole.check, Bool.and_eq_true] at h
  have he := FrontendBoolSimulationWith.evalFunc_sound (FrontendMaskRuntime.sem limit)
    (FrontendBoolSimulationWith.candidateLaws limit) h.2
    {ctx with M := FrontendF32Comparison.model base} (FrontendF32Comparison.model_correct base)
  rw [FrontendBoolSimulationWith.policy_nil] at he
  have hlen := FrontendBoolSimulation.functions_length FrontendMaskTemplate.subject.funcs actual.funcs h.2
  obtain ⟨out, hr, ht, hd, hp⟩ := body_exec limit {ctx with M := FrontendF32Comparison.model base}
    (evalFunc (FrontendMaskRuntime.sem limit) {ctx with M := FrontendF32Comparison.model base}
      FrontendMaskTemplate.subject 1) mask values seed hm hv hs hdm hdv hds hrv hrs hib
    (FrontendF32Comparison.model_correct base) hl
  have hsource : denote (FrontendMaskRuntime.sem limit) {ctx with M := FrontendF32Comparison.model base}
      FrontendMaskTemplate.subject "subject" [.tensor mask, .tensor values, .tensor seed] =
        .ok [.tensor out, index (FrontendMaskBounds.countFrom (bits mask) 1024 0)] := by
    rw [denote, show FrontendMaskTemplate.subject.funcs.length = 1 from by cbv, evalFunc, subject_find]
    exact hr
  refine ⟨out, ?_, ht, hd, hp⟩
  simpa only [denote, he, hlen] using hsource

#print axioms check_sound
end MerlinVerify.Rules.FrontendMaskRange
