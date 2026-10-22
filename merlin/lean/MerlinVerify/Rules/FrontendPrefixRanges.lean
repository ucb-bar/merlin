/- Actual-prefix position bounds under the existing candidate interpretation.
Inputs receive only their declared types and initialization, never range annotations. -/
import MerlinVerify.Rules.FrontendRangeDefined
import MerlinVerify.Rules.FrontendRangeFacts
import MerlinVerify.Rules.FrontendPrefixProjection

namespace MerlinVerify.Rules.FrontendPrefixRanges
open MerlinVerify MerlinVerify.Check
open FrontendTensorRanges

def expected (upper : Nat) : List Fact :=
  [⟨⟨.tensor [1,32,32] (.int 1), true⟩, none⟩,
   ⟨⟨.tensor [1,1024] (.int 64), true⟩, some ⟨0,upper⟩⟩,
   ⟨⟨.tensor [1,1024] (.int 64), true⟩, some ⟨0,upper⟩⟩]

def check (m : Module) (name : String) (stop upper : Nat) : Bool :=
  match FrontendPrefixProjection.project? m name stop with
  | none => false
  | some p =>
    match p.cut.block.args.mapM (fun v => vtyOf 64 v.ty) with
    | none => false
    | some a => FrontendRuntimeBridge.regions [p.body] &&
      (p.cut.block.ops.take stop).all (fun op => !op.isTerminator) &&
      match FrontendRangeFrame.inferRegion (FrontendRangeDefined.frameBook 64 []) [] p.body
        (a.map fun ty => Definedness.Refined.emptyFact model ⟨ty,true⟩) with
      | some inferred => FrontendRangeFacts.fits inferred (expected upper)
      | none => false

/-- Acceptance proves actual prefix execution and the requested bounds from
ordinary initialized function inputs, without assuming any intermediate value bound. -/
theorem check_sound {m : Module} {name : String} {stop upper : Nat}
    (h : check m name stop upper = true) :
    ∃ p : FrontendPrefixProjection.Prefix m name stop,
      FrontendPrefixProjection.project? m name stop = some p ∧
      ∃ a, p.cut.block.args.mapM (fun v => vtyOf 64 v.ty) = some a ∧
      ∀ (limit : Nat) (ctx : Ctx) (call : CallSem) (args : List RVal), ctx.indexBits = 64 →
        HasTys 64 args a → ResultsDefined args →
        ∃ values, evalRegion (FrontendMaskRuntime.sem limit) ctx call [] p.body args = .ok values ∧
          Definedness.Refined.FactsHold model 64 values (expected upper) := by
  unfold check at h
  split at h
  · cases h
  · next p hp =>
    split at h
    · cases h
    · next a ha =>
      simp only [Bool.and_eq_true] at h
      obtain ⟨⟨hbridge, _⟩, h⟩ := h
      split at h
      · next inferred hinfer =>
        refine ⟨p, hp, a, ha, ?_⟩
        intro limit ctx call args hib ht hd
        obtain ⟨values, hv, hf⟩ := FrontendRangeFrame.inferRegion_sound
          (ctx := ctx) (call := call) (env := []) (FrontendRangeDefined.frameBook 64 []) hinfer hib
          (.inl rfl) (by intro _ _ h; cases h) (Definedness.Refined.inputFacts model ht hd)
        have hb : FrontendRuntimeBridge.blocksCheck p.body.blocks = true := by
          simpa only [FrontendPrefixProjection.Prefix.body, FrontendRuntimeBridge.regions,
            Region.blocks, Bool.and_true] using hbridge
        have he := FrontendRuntimeBridge.blocks_sound limit ctx call [] p.body.blocks args hb
        change evalRegion (FrontendMaskRuntime.sem limit) ctx call [] p.body args =
          evalRegion Sem.Ops.opSem ctx call [] p.body args at he
        exact ⟨values, he.trans hv, FrontendRangeFacts.fits_sound h hf⟩
      · cases h

#print axioms check_sound
end MerlinVerify.Rules.FrontendPrefixRanges
