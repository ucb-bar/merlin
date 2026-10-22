/- Program composition of actual position-producing prefix, selection and continuation. -/
import MerlinVerify.Rules.FrontendPositionScope
import MerlinVerify.Rules.FrontendMaskRange

namespace MerlinVerify.Rules.FrontendPositionExecution
open MerlinVerify MerlinVerify.Check
open FrontendTensorRanges FrontendPositionScope FrontendProgramSegments

def maskFact : Fact := ⟨⟨.tensor [1,32,32] (.int 1), true⟩, none⟩
def positionFact : Fact := ⟨⟨.tensor [1,1024] (.int 64), true⟩, some ⟨0,1024⟩⟩

/-- Expose the three proved tensor roots while retaining every additional observation. -/
theorem roots {vs : List RVal} {extraFacts : List Fact}
    (h : Definedness.Refined.FactsHold model 64 vs (FrontendPrefixRanges.expected 1024 ++ extraFacts)) :
    ∃ mask values seed extras,
      vs = [.tensor mask,.tensor values,.tensor seed] ++ extras ∧
      Definedness.Refined.Fact.Holds model 64 (.tensor mask) maskFact ∧
      Definedness.Refined.Fact.Holds model 64 (.tensor values) positionFact ∧
      Definedness.Refined.Fact.Holds model 64 (.tensor seed) positionFact ∧
      Definedness.Refined.FactsHold model 64 extras extraFacts := by
  cases vs with
  | nil => cases h
  | cons mask rest =>
    obtain _ | ⟨hm, hrest⟩ := h
    cases rest with
    | nil => cases hrest
    | cons values rest =>
      obtain _ | ⟨hv, hrest⟩ := hrest
      cases rest with
      | nil => cases hrest
      | cons seed extras =>
        obtain _ | ⟨hs, he⟩ := hrest
        cases mask with
        | scalar _ => cases hm.1.1
        | tensor mask =>
          cases values with
          | scalar _ => cases hv.1.1
          | tensor values =>
            cases seed with
            | scalar _ => cases hs.1.1
            | tensor seed => exact ⟨mask,values,seed,extras,rfl,hm,hv,hs,he⟩

/-- Strong selection execution for arbitrary initialized tensor representations. -/
theorem selection {m : Module} {name : String} {stop count : Nat}
    (s : Scope m name stop count) {limit : Nat} {base : FloatModel} {ctx : Ctx} {call : CallSem}
    {env : Env} {mask values seed : Tensor}
    (hm : Definedness.Refined.Fact.Holds model 64 (.tensor mask) maskFact)
    (hv : Definedness.Refined.Fact.Holds model 64 (.tensor values) positionFact)
    (hs : Definedness.Refined.Fact.Holds model 64 (.tensor seed) positionFact)
    (hib : ctx.indexBits = 64) (hl : 1024 ≤ limit)
    (hread : env.getAll [s.selection.mask.id,s.selection.values.id,s.selection.seed.id] =
      .ok [.tensor mask,.tensor values,.tensor seed])
    (hcut : FrontendBoolSimulation.region []
      [(50,s.selection.mask.id),(51,s.selection.values.id),(52,s.selection.seed.id)]
      FrontendRuntimeCheckpoint.bareSource s.selection.bare = true) :
    ∃ results, evalRegion (FrontendMaskRuntime.sem limit) {ctx with M := FrontendF32Comparison.model base}
      call env s.selection.bare [] = .ok results ∧
      Definedness.Refined.FactsHold model 64 results selectedFacts := by
  let modelCtx := {ctx with M := FrontendF32Comparison.model base}
  obtain ⟨out, hbody, ht, hd, hp⟩ := FrontendMaskRange.body_exec limit modelCtx call mask values seed
    hm.1.1 hv.1.1 hs.1.1 (hm.1.2 rfl) (hv.1.2 rfl) (hs.1.2 rfl) hv.2 hs.2 hib
    (FrontendF32Comparison.model_correct base) hl
  have hc := roots_corr env s.selection.mask.id s.selection.values.id s.selection.seed.id
    (.tensor mask) (.tensor values) (.tensor seed) hread
  have he := FrontendBoolSimulationWith.region_sound (FrontendMaskRuntime.sem limit)
    (FrontendBoolSimulationWith.candidateLaws limit) [] call _ FrontendRuntimeCheckpoint.bareSource
    s.selection.bare modelCtx [(50,.tensor mask),(51,.tensor values),(52,.tensor seed)] env hcut
    (FrontendF32Comparison.model_correct base) hc
  rw [FrontendBoolSimulationWith.policy_nil] at he
  have hsource : evalRegion (FrontendMaskRuntime.sem limit) modelCtx call
      [(50,.tensor mask),(51,.tensor values),(52,.tensor seed)] FrontendRuntimeCheckpoint.bareSource [] =
      .ok [.tensor out, FrontendMaskProgram.index (FrontendMaskBounds.countFrom (FrontendMaskTyped.bits mask) 1024 0)] := by
    simpa only [FrontendRuntimeCheckpoint.bareSource, FrontendMaskTemplate.body, evalRegion,
      evalBlocks, evalBlock, Env.bindAll, List.length_cons, List.length_nil, ↓reduceIte,
      List.map_cons, List.map_nil, List.zip_cons_cons, List.zip_nil_left, List.cons_append,
      List.nil_append, Outcome.bind_ok] using hbody
  refine ⟨_, (congrFun he []).symm.trans hsource, .cons ⟨⟨ht, fun _ => hd⟩, hp⟩ (.cons ?_ .nil)⟩
  exact ⟨⟨by simp [FrontendMaskProgram.index, RVal.HasTy, Val.HasTy, Scalar.conforms],
    by simp [FrontendMaskProgram.index, RVal.Defined]⟩, trivial⟩

/-- Successful execution of the actual prefix, selection and entire continuation,
composed through the evaluator's real SSA environments and checked capture frames. -/
theorem scope_sound {m : Module} {name : String} {stop count : Nat}
    (s : Scope m name stop count) (h : s.check = true) : s.Executes := by
  unfold Scope.check at h
  simp only [Bool.and_eq_true] at h
  obtain ⟨⟨⟨⟨⟨⟨_, hcut⟩, hpreNon⟩, hselNon⟩, _⟩, hbridge⟩, h⟩ := h
  split at h
  · next a fs rs ha hfs hrs =>
    split at h
    · next prefixFacts resultFacts hpre htail =>
      simp only [Bool.and_eq_true] at h
      obtain ⟨⟨hpreFits, hpreserve⟩, htailFits⟩ := h
      unfold Scope.preFacts? at hfs
      cases hext : s.extras.mapM (fun v => vtyOf 64 v.ty) with
      | none => simp [hext] at hfs
      | some extraTypes =>
        simp only [hext, Option.map_some, Option.some.injEq] at hfs
        subst fs
        refine ⟨a, rs, ha, hrs, ?_⟩
        intro limit base ctx call args hib hl ht hd
        let modelCtx := {ctx with M := FrontendF32Comparison.model base}
        let sem := FrontendMaskRuntime.sem limit
        have hb : FrontendRuntimeBridge.blocksCheck s.preBody.blocks = true ∧
            FrontendRuntimeBridge.blocksCheck s.tailBody.blocks = true := by
          simpa only [Scope.preBody, Scope.tailBody, FrontendRuntimeBridge.regions,
            Region.blocks, Bool.and_true, Bool.and_eq_true] using hbridge
        obtain ⟨observations, hpreEval, hpreFacts⟩ := FrontendRangeFrame.inferRegion_sound
          (ctx := modelCtx) (call := call) (env := []) (FrontendRangeDefined.frameBook 64 []) hpre hib
          (.inl rfl) (by intro _ _ h; cases h) (Definedness.Refined.inputFacts model ht hd)
        have hf := FrontendRangeFacts.fits_sound hpreFits hpreFacts
        have he := (FrontendProgramFrames.region_eq limit modelCtx call [] s.preBody args hb.1).trans hpreEval
        cases hbind : Env.bindAll [] s.selection.p.cut.block.args args with
        | ub => simp [Scope.preBody, evalRegion, evalBlocks, evalBlock, hbind] at he
        | stuck why => simp [Scope.preBody, evalRegion, evalBlocks, evalBlock, hbind] at he
        | ok initial =>
          simp only [Scope.preBody, evalRegion, evalBlocks, evalBlock, hbind, Outcome.bind_ok] at he
          obtain ⟨middle, hprefix, hread⟩ := prefix_state
            (fun op hop => by simpa using List.all_eq_true.mp hpreNon op hop) he
          have hΓ := FrontendProgramFrames.of_getAll middle _ observations _ hread hf
          obtain ⟨mask, values, seed, extraValues, rfl, hm, hv, hs, _⟩ := roots hf
          have hroots : middle.getAll [s.selection.mask.id,s.selection.values.id,s.selection.seed.id] =
              .ok [.tensor mask,.tensor values,.tensor seed] := by
            have h := FrontendProgramFrames.getAll_take middle _ _ hread 3
            simpa only [Scope.observed, s.selection.roots, List.map_append, List.map_cons, List.map_nil,
              List.cons_append, List.nil_append, List.take_succ_cons, List.take_zero] using h
          obtain ⟨selectedValues, hselected, hselectedFacts⟩ := selection s
            (base := base) (ctx := ctx) (call := call) hm hv hs hib hl hroots hcut
          simp only [FrontendRuntimeCheckpoint.Scope.bare, evalRegion, evalBlocks, evalBlock,
            Env.bindAll, List.length_nil, ↓reduceIte, List.map_nil, List.zip_nil_left,
            List.nil_append, Outcome.bind_ok] at hselected
          obtain ⟨afterSelection, hexec, hselectedRead⟩ := prefix_state
            (fun op hop => by simpa using List.all_eq_true.mp hselNon op hop) hselected
          have hkept := FrontendProgramFrames.exec_preserves s.selection.p.cut.selected middle afterSelection hΓ
            (fun op hop => by simpa using List.all_eq_true.mp hselNon op hop) hpreserve hexec
          have hnew := FrontendProgramFrames.of_getAll afterSelection _ selectedValues _ hselectedRead hselectedFacts
          have hpost := FrontendProgramFrames.append hnew hkept
          obtain ⟨results, hresult, hresultFacts⟩ := FrontendRangeFrame.inferRegion_sound
            (ctx := modelCtx) (call := call) (env := afterSelection)
            (FrontendRangeDefined.frameBook 64 []) htail hib (.inl rfl) hpost List.Forall₂.nil
          have htailEval := (FrontendProgramFrames.region_eq limit modelCtx call afterSelection s.tailBody [] hb.2).trans hresult
          simp only [Scope.tailBody, evalRegion, evalBlocks, evalBlock, Env.bindAll,
            List.length_nil, ↓reduceIte, List.map_nil, List.zip_nil_left, List.nil_append, Outcome.bind_ok] at htailEval
          refine ⟨results, ?_, FrontendRangeFacts.fits_sound htailFits hresultFacts⟩
          rw [s.body_segments]
          simp only [evalRegion, evalBlocks, evalBlock, hbind, Outcome.bind_ok, List.append_assoc]
          rw [evalOps_append sem modelCtx call initial _ _
            (fun op hop => by simpa using List.all_eq_true.mp hpreNon op hop), hprefix, Outcome.bind_ok]
          rw [evalOps_append sem modelCtx call middle _ _
            (fun op hop => by simpa using List.all_eq_true.mp hselNon op hop), hexec, Outcome.bind_ok]
          exact htailEval
    · cases h
  · cases h

theorem check_sound {m : Module} {name : String} {stop count : Nat}
    (h : FrontendPositionScope.check m name stop count = true) :
    ∃ s : Scope m name stop count, FrontendPositionScope.project? m name stop count = some s ∧ s.Executes := by
  unfold FrontendPositionScope.check at h
  split at h
  · next s hs => exact ⟨s, hs, scope_sound s h⟩
  · cases h

#print axioms scope_sound
#print axioms check_sound

#print axioms selection
end MerlinVerify.Rules.FrontendPositionExecution
