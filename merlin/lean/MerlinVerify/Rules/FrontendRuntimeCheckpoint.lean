/- Actual-prefix plus actual-selection execution, composed through intermediate SSA state. -/
import MerlinVerify.Rules.FrontendPrefixProjection
import MerlinVerify.Rules.FrontendMaskTyped
import MerlinVerify.Rules.FrontendProgramSegments

namespace MerlinVerify.Rules.FrontendRuntimeCheckpoint
open MerlinVerify MerlinVerify.Check MerlinVerify.Check.Definedness
open FrontendProgramSegments

structure Scope (m : Module) (name : String) (stop : Nat) where
  p : FrontendPrefixProjection.Prefix m name stop
  mask : Value
  values : Value
  seed : Value
  roots : p.cut.args = [mask, values, seed]

def project? (m : Module) (name : String) (stop : Nat) : Option (Scope m name stop) := do
  let p ← FrontendPrefixProjection.project? m name stop
  match h : p.cut.args with
  | [mask, values, seed] => some ⟨p, mask, values, seed, h⟩
  | _ => none

def bareSource : Region := .mk [.mk [] FrontendMaskTemplate.operations]

def Scope.bare {m : Module} {name : String} {stop : Nat} (s : Scope m name stop) : Region :=
  .mk [.mk [] (s.p.cut.selected ++ [returned (s.p.cut.outputs.map (·.id))])]

def Scope.body {m : Module} {name : String} {stop : Nat} (s : Scope m name stop) : Region :=
  .mk [.mk s.p.cut.block.args ((s.p.cut.block.ops.take stop) ++
    s.p.cut.selected ++ [returned (s.p.cut.outputs.map (·.id))])]

/-- The checked region contains the actual initial operations, in their original order.
Only the observation return is added; no source operation is replaced by the projection. -/
theorem Scope.body_actual {m : Module} {name : String} {stop : Nat} (s : Scope m name stop) :
    s.body = .mk [.mk s.p.cut.block.args ((s.p.cut.block.ops.take (stop + 25)) ++
      [returned (s.p.cut.outputs.map (·.id))])] := by
  simp only [Scope.body, s.p.cut.slice, ← List.take_add]

def Scope.program {m : Module} {name : String} {stop : Nat} (s : Scope m name stop) : Module :=
  {ops := [.mk "func.func" [] []
    [("function_type", .type (.function (s.p.cut.block.args.map (·.ty)) (s.p.cut.outputs.map (·.ty)))),
     ("sym_name", .str "subject")] [] [s.body] []]}

def Scope.check {m : Module} {name : String} {stop : Nat} (s : Scope m name stop)
    (limit : Nat) (a : List VTy) : Bool :=
  FrontendMaskWhole.check s.p.cut.program &&
  (s.p.cut.block.ops.take stop).all (fun op => !op.isTerminator) &&
  Definedness.checkRegion (FrontendPrefixDefined.candidateBook limit 64) [] s.p.body (clean a)
    FrontendMaskTyped.inputTypes &&
  FrontendBoolSimulation.region [] [(50,s.mask.id),(51,s.values.id),(52,s.seed.id)] bareSource s.bare

/-- Acceptance connects the full model's declared initialized inputs to actual prefix/selection
execution. It does not assume the intermediate cut domain or replace the operations by an oracle. -/
theorem scope_sound {m : Module} {name : String} {stop limit : Nat} {s : Scope m name stop}
    {a : List VTy} (h : s.check limit a = true) (base : FloatModel) (ctx : Ctx) (call : CallSem)
    (args : List RVal) (ha : HasTys 64 args a) (hd : ResultsDefined args)
    (hib : ctx.indexBits = 64) (hl : 1024 ≤ limit) :
    ∃ results, evalRegion (FrontendMaskRuntime.sem limit) {ctx with M := FrontendF32Comparison.model base}
      call [] s.body args = .ok results ∧
      HasTys 64 results FrontendMaskTyped.resultTypes ∧ ResultsDefined results := by
  simp only [Scope.check, Bool.and_eq_true] at h
  obtain ⟨⟨⟨_, hnonterm⟩, hpre⟩, hcut⟩ := h
  let modelCtx := {ctx with M := FrontendF32Comparison.model base}
  let sem := FrontendMaskRuntime.sem limit
  obtain ⟨mid, hmid, ht, hi⟩ := Definedness.checkRegion_sound
    (ctx := modelCtx) (call := call) (env := [])
    (FrontendPrefixDefined.candidateBook limit 64) hpre hib
    (by intro _ _ h; cases h) (inputFacts ha hd)
  change evalRegion sem modelCtx call [] s.p.body args = .ok mid at hmid
  cases hb : Env.bindAll [] s.p.cut.block.args args with
  | ub => simp [FrontendPrefixProjection.Prefix.body, evalRegion, evalBlocks, evalBlock, hb] at hmid
  | stuck why => simp [FrontendPrefixProjection.Prefix.body, evalRegion, evalBlocks, evalBlock, hb] at hmid
  | ok initial =>
    simp only [FrontendPrefixProjection.Prefix.body, evalRegion, evalBlocks, evalBlock, hb,
      Outcome.bind_ok] at hmid
    obtain ⟨middle, hmiddle, hread⟩ := prefix_state (sem := sem) (ctx := modelCtx) (call := call)
      (env := initial) (pre := s.p.cut.block.ops.take stop)
      (fun op hop => by simpa using List.all_eq_true.mp hnonterm op hop) hmid
    obtain _ | ⟨hm, ht⟩ := ht
    obtain _ | ⟨hv, ht⟩ := ht
    obtain _ | ⟨hs, ht⟩ := ht
    cases ht
    rename_i mask values seed
    have htyped : HasTys 64 [mask,values,seed] FrontendMaskTyped.inputTypes := .cons hm (.cons hv (.cons hs .nil))
    obtain ⟨results, hbody, hr, hd⟩ := FrontendMaskTyped.body_exec limit modelCtx call _ htyped hi hib
      (FrontendF32Comparison.model_correct base) hl
    have hc := roots_corr middle s.mask.id s.values.id s.seed.id mask values seed
      (by simpa only [s.roots, List.map_cons, List.map_nil] using hread)
    have he := FrontendBoolSimulationWith.region_sound sem (FrontendBoolSimulationWith.candidateLaws limit)
      [] call _ bareSource s.bare modelCtx [(50,mask),(51,values),(52,seed)] middle hcut
      (FrontendF32Comparison.model_correct base) hc
    rw [FrontendBoolSimulationWith.policy_nil] at he
    have hsource : evalRegion sem modelCtx call [(50,mask),(51,values),(52,seed)] bareSource [] = .ok results := by
      simpa only [bareSource, FrontendMaskTemplate.body, evalRegion, evalBlocks, evalBlock,
        Env.bindAll, List.length_cons, List.length_nil, ↓reduceIte, List.map_cons, List.map_nil,
        List.zip_cons_cons, List.zip_nil_left, List.cons_append, List.nil_append, Outcome.bind_ok] using hbody
    have htarget := (congrFun he []).symm.trans hsource
    simp only [Scope.bare, evalRegion, evalBlocks, evalBlock, Env.bindAll, List.length_nil,
      ↓reduceIte, List.map_nil, List.zip_nil_left, List.nil_append, Outcome.bind_ok] at htarget
    refine ⟨results, ?_, hr, hd⟩
    simp only [Scope.body, evalRegion, evalBlocks, evalBlock, hb, Outcome.bind_ok]
    rw [List.append_assoc, evalOps_append sem modelCtx call initial _ _
      (fun op hop => by simpa using List.all_eq_true.mp hnonterm op hop), hmiddle, Outcome.bind_ok]
    exact htarget

def check (m : Module) (name : String) (stop limit : Nat) : Bool :=
  match project? m name stop with
  | none => false
  | some s => match s.p.cut.block.args.mapM (fun v => vtyOf 64 v.ty) with
    | none => false
    | some a => s.check limit a

/-- Every accepted checkpoint is a slice of the actual block and executes from its declared
initialized inputs, including the actual SSA state between the prefix and selection. -/
theorem check_sound {m : Module} {name : String} {stop limit : Nat}
    (h : check m name stop limit = true) :
    ∃ s : Scope m name stop, project? m name stop = some s ∧
      ∃ a, s.p.cut.block.args.mapM (fun v => vtyOf 64 v.ty) = some a ∧
        ∀ (base : FloatModel) (ctx : Ctx) (call : CallSem) (args : List RVal), ctx.indexBits = 64 → 1024 ≤ limit →
          HasTys 64 args a → ResultsDefined args →
          ∃ results, evalRegion (FrontendMaskRuntime.sem limit) {ctx with M := FrontendF32Comparison.model base}
            call [] s.body args = .ok results ∧ HasTys 64 results FrontendMaskTyped.resultTypes ∧
              ResultsDefined results := by
  unfold check at h
  split at h
  · cases h
  · next s hp =>
    split at h
    · cases h
    · next a ha =>
      exact ⟨s, hp, a, ha, fun base ctx call args hib hl ht hd => scope_sound h base ctx call args ht hd hib hl⟩

#print axioms scope_sound
#print axioms check_sound
end MerlinVerify.Rules.FrontendRuntimeCheckpoint
