/- Integer postconditions inferred from an actual function, with no numeric input facts. -/
import MerlinVerify.Rules.FrontendRangeDefined
import MerlinVerify.Rules.FrontendRangeFacts
import MerlinVerify.Rules.FrontendProgramFrames
import MerlinVerify.Check.TypedModule

namespace MerlinVerify.Rules.FrontendFunctionRanges
open MerlinVerify MerlinVerify.Check
open FrontendTensorRanges

def expected (types : List VTy) (upper : Nat) : List Fact :=
  types.map fun ty => ⟨⟨ty,true⟩, some ⟨0,upper⟩⟩

def check (m : Module) (name : String) (upper : Nat) : Bool :=
  match m.func? name, TypedModule.signature? 64 m name with
  | some f, some (a,r) => !f.isDecl && FrontendRuntimeBridge.regions [f.body] &&
      match FrontendRangeFrame.inferRegion (FrontendRangeDefined.frameBook 64 []) [] f.body
        (a.map fun ty => Definedness.Refined.emptyFact model ⟨ty,true⟩) with
      | some facts => FrontendRangeFacts.fits facts (expected r upper)
      | none => false
  | _, _ => false

/-- Requested output bounds must follow from the actual checked producers.
The formal statement quantifies every FloatModel, including every comparison result. -/
theorem check_sound {m : Module} {name : String} {upper : Nat} (h : check m name upper = true) :
    ∃ a r, TypedModule.signature? 64 m name = some (a,r) ∧
      ∀ (limit : Nat) (ctx : Ctx) (args : List RVal), ctx.indexBits = 64 →
        HasTys 64 args a → ResultsDefined args →
        ∃ values, denote (FrontendMaskRuntime.sem limit) ctx m name args = .ok values ∧
          Definedness.Refined.FactsHold model 64 values (expected r upper) := by
  unfold check at h
  split at h
  · next f a r hf hsig =>
    simp only [Bool.and_eq_true, Bool.not_eq_eq_eq_not, Bool.not_true] at h
    obtain ⟨⟨hdecl, hbridge⟩, h⟩ := h
    split at h
    · next facts hinfer =>
      refine ⟨a,r,hsig,?_⟩
      intro limit ctx args hib ht hd
      obtain ⟨values, hv, hvalues⟩ := FrontendRangeFrame.inferRegion_sound
        (ctx := ctx) (call := evalFunc (FrontendMaskRuntime.sem limit) ctx m m.funcs.length) (env := [])
        (FrontendRangeDefined.frameBook 64 []) hinfer hib (.inl rfl)
        (by intro _ _ h; cases h) (Definedness.Refined.inputFacts model ht hd)
      have hb : FrontendRuntimeBridge.blocksCheck f.body.blocks = true := by
        cases hbody : f.body with
        | mk blocks => simpa only [hbody, FrontendRuntimeBridge.regions, Region.blocks, Bool.and_true] using hbridge
      have he := FrontendProgramFrames.region_eq limit ctx
        (evalFunc (FrontendMaskRuntime.sem limit) ctx m m.funcs.length) [] f.body args hb
      refine ⟨values, ?_, FrontendRangeFacts.fits_sound h hvalues⟩
      simpa only [denote, evalFunc, hf, hdecl, Bool.false_eq_true, ↓reduceIte] using he.trans hv
    · cases h
  · cases h

#print axioms check_sound
end MerlinVerify.Rules.FrontendFunctionRanges
