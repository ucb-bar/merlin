/- Lift exact uniform-input normalization through checked generic bodies. Captured facts survive
only when neither region block arguments nor actual source operations rebind their dependencies. -/
import MerlinVerify.Check.SplatNormalize
import MerlinVerify.Rules.StructuredBody

namespace MerlinVerify.Check.SplatNested

open MerlinVerify MerlinVerify.Sem.Ops MerlinVerify.Sem.Ops.Linalg

variable {ib : Nat}

/-- A reusable exact block normalizer with explicitly checked types, loop depth and SSA facts. -/
structure Normalizer (ib : Nat) where
  run : Nat → TEnv → List UniformFacts.Fact → List Op → List VTy → Option (List Op)
  sound : ∀ d Γ facts ops normalized yields ctx call env,
    checkOps (merlinBook ib) d Γ ops yields = true →
    run d Γ facts ops yields = some normalized →
    EnvTyped ib Γ env → BindingFacts.Valid UniformFacts.model facts env →
    ctx.indexBits = ib → d ≤ ctx.loopIndex.length →
    evalOps opSem ctx call env normalized = evalOps opSem ctx call env ops

def base (ib : Nat) : Normalizer ib where
  run _ Γ facts ops _ := SplatNormalize.normalize ib Γ facts ops
  sound d Γ facts ops normalized yields ctx call env hc hn he hf hi hd :=
    SplatNormalize.normalize_sound (merlinBook ib) d Γ facts ops normalized yields ctx call env
      hc hn he hf hi hd

def region (N : Normalizer ib) (Γ : TEnv) (facts : List UniformFacts.Fact)
    (r : Region) (sig : RegionSig) : Option Region :=
  match r with
  | .mk [.mk args ops] =>
    if checkRegion (merlinBook ib) Γ r sig then do
      let nested ← N.run sig.depth ((args.map (·.id)).zip sig.args ++ Γ)
        (BindingFacts.forget UniformFacts.model (args.map (·.id)) facts) ops sig.yields
      pure (.mk [.mk args nested])
    else none
  | _ => none

/-- Region binding masks captured facts at every rebinding, even on raw ASTs with shadowed ids.
Both regions retain exactly the same block arguments and malformed-arity behavior. -/
theorem region_sound (N : Normalizer ib) {Γ : TEnv} {facts : List UniformFacts.Fact}
    {source target : Region} {sig : RegionSig} (hn : region N Γ facts source sig = some target)
    {env : Env} (he : EnvTyped ib Γ env)
    (hf : BindingFacts.Valid UniformFacts.model facts env) (call : CallSem) :
    RegionSafe ib (fun ctx => evalRegion opSem ctx call env source) sig ∧
      Rules.StructuredBody.Equal ib sig (fun ctx => evalRegion opSem ctx call env source)
        (fun ctx => evalRegion opSem ctx call env target) := by
  unfold region at hn
  split at hn
  · next r args ops =>
    split at hn
    · next hcheck =>
      cases ht : N.run sig.depth ((args.map (·.id)).zip sig.args ++ Γ)
          (BindingFacts.forget UniformFacts.model (args.map (·.id)) facts) ops sig.yields with
      | none => simp [ht] at hn
      | some nested =>
        simp only [ht, bind, Option.bind, Option.pure_def, Option.some.injEq] at hn
        subst target
        refine ⟨checkRegion_safe (merlinBook ib) Γ _ sig call env hcheck he, ?_⟩
        intro ctx hib hd values hv
        have hc := hcheck
        simp only [checkRegion, Bool.and_eq_true, beq_iff_eq] at hc
        obtain ⟨ha, hc⟩ := hc
        have hl : args.length = values.length :=
          (optionMapM_length _ _ _ ha).symm.trans (HasTys.length hv).symm
        simp only [evalRegion, evalBlocks, evalBlock, bindAll_eq_ok hl, Outcome.bind_ok]
        exact (N.sound sig.depth _ _ ops nested sig.yields ctx call _ hc ht
          (he.extend _ values sig.args hv)
          (BindingFacts.forget_sound UniformFacts.model hf (args.map (·.id)) values) hib hd).symm
    · cases hn
  · cases hn

def withRegions : Op → List Region → Op
  | .mk name results operands attrs props _ successors, regions =>
    .mk name results operands attrs props regions successors

@[simp] theorem withRegions_operands (op : Op) (regions : List Region) :
    (withRegions op regions).operands = op.operands := by cases op; rfl
@[simp] theorem withRegions_results (op : Op) (regions : List Region) :
    (withRegions op regions).results = op.results := by cases op; rfl
@[simp] theorem withRegions_regions (op : Op) (regions : List Region) :
    (withRegions op regions).regions = regions := by cases op; rfl
@[simp] theorem withRegions_blind (op : Op) (regions : List Region) :
    (withRegions op regions).blind = op.blind := by cases op; rfl
@[simp] theorem withRegions_terminator (op : Op) (regions : List Region) :
    (withRegions op regions).isTerminator = op.isTerminator := by cases op; rfl

theorem semantic_equal {op : Op} {tys inTys outTys : List VTy} {maps : List AffineMap}
    {iters : List IterKind} {ranges : List Nat}
    (hn : op.name = "linalg.generic") (hm : unmodelledAttr? op.blind = none)
    (hsplit : splitInsOuts op tys = some (inTys, outTys))
    (hmaps : attrAffineMaps? op "indexing_maps" = some maps) (hiters : iterKinds? op = some iters)
    (hlayout : structuredRanges? maps iters.length (inTys ++ outTys) = some ranges)
    (hout : ∀ ty ∈ outTys, ty.isTensor = true)
    {ctx : Ctx} (hib : ctx.indexBits = ib) (call : CallSem)
    {args : List RVal} (ha : HasTys ib args tys) {source target : Ctx → RegionSem}
    (hb : RegionSafe ib source
      ⟨(inTys ++ outTys).map VTy.eltScalar, outTys.map VTy.eltScalar, iters.length⟩)
    (he : Rules.StructuredBody.Equal ib
      ⟨(inTys ++ outTys).map VTy.eltScalar, outTys.map VTy.eltScalar, iters.length⟩ source target) :
    opSem ctx call op args [source] = opSem ctx call op args [target] := by
  obtain ⟨ins, outs, hs, hi, ho⟩ := splitInsOuts_typed hsplit ha
  obtain ⟨ts, rfl⟩ := tensors_of_typed ho hout
  simp_rw [linalg_sem_of hm (by rw [hn]; cbv) (.inl hn)]
  simp only [Linalg.sem, Op.blind_name, hn, splitInsOuts_blind, hs,
    attrAffineMaps?_blind, hmaps, iterKinds?_blind, hiters]
  rw [Rules.StructuredBody.genericSem_equal hib (structuredRanges?_spec hlayout) hi ho hb he]

/-- Only supported generic regions are entered; every other operation is retained verbatim. -/
def liftOp? (N : Normalizer ib) (Γ : TEnv) (facts : List UniformFacts.Fact) (op : Op) : Option Op := do
  let tys ← op.operands.mapM (Γ.lookup ·)
  let (inTys, outTys) ← splitInsOuts op tys
  let maps ← attrAffineMaps? op "indexing_maps"
  let iters ← iterKinds? op
  let _ ← structuredRanges? maps iters.length (inTys ++ outTys)
  if op.name = "linalg.generic" ∧ unmodelledAttr? op.blind = none ∧
      (∀ ty ∈ outTys, ty.isTensor = true) then
    match op.regions with
    | [source] =>
      let sig : RegionSig :=
        ⟨(inTys ++ outTys).map VTy.eltScalar, outTys.map VTy.eltScalar, iters.length⟩
      let target ← region N Γ facts source sig
      pure (withRegions op [target])
    | _ => none
  else none

def liftOp (N : Normalizer ib) (Γ : TEnv) (facts : List UniformFacts.Fact) (op : Op) : Op :=
  (liftOp? N Γ facts op).getD op

theorem liftOp_spec (N : Normalizer ib) (Γ : TEnv) (facts : List UniformFacts.Fact) (op : Op) :
    (liftOp N Γ facts op).isTerminator = op.isTerminator ∧
    (liftOp N Γ facts op).results = op.results ∧
    ∀ ctx call env, EnvTyped ib Γ env → BindingFacts.Valid UniformFacts.model facts env →
      ctx.indexBits = ib → stepOp opSem ctx call env (liftOp N Γ facts op) =
        stepOp opSem ctx call env op := by
  unfold liftOp
  cases hl : liftOp? N Γ facts op with
  | none => exact ⟨rfl, rfl, fun _ _ _ _ _ _ => rfl⟩
  | some normalized =>
    unfold liftOp? at hl
    cases ha : op.operands.mapM (Γ.lookup ·) with
    | none => simp [ha] at hl
    | some tys =>
      simp only [ha, bind, Option.bind] at hl
      cases hs : splitInsOuts op tys with
      | none => simp [hs] at hl
      | some pair =>
        obtain ⟨inTys, outTys⟩ := pair
        simp only [hs] at hl
        cases hm : attrAffineMaps? op "indexing_maps" with
        | none => simp [hm] at hl
        | some maps =>
          simp only [hm] at hl
          cases hi : iterKinds? op with
          | none => simp [hi] at hl
          | some iters =>
            simp only [hi] at hl
            cases hg : structuredRanges? maps iters.length (inTys ++ outTys) with
            | none => simp [hg] at hl
            | some ranges =>
              simp only [hg] at hl
              split at hl
              · next hc =>
                split at hl
                · next source hrs =>
                  cases hr : region N Γ facts source
                      ⟨(inTys ++ outTys).map VTy.eltScalar, outTys.map VTy.eltScalar, iters.length⟩ with
                  | none => rw [hr] at hl; cases hl
                  | some target =>
                    simp only [hr, Option.pure_def, Option.some.injEq] at hl
                    subst normalized
                    refine ⟨by simp, by simp, ?_⟩
                    intro ctx call env he hf hib
                    obtain ⟨args, hargs, htys⟩ := he.getAll _ _ ha
                    obtain ⟨hbody, hbodyEq⟩ := region_sound N hr he hf call
                    simp only [Option.getD_some, stepOp, withRegions_operands, withRegions_results,
                      withRegions_regions, hrs, hargs, Outcome.bind_ok, evalRegions]
                    rw [opSem_blind (op' := op) (withRegions_blind _ _)]
                    rw [← semantic_equal hc.1 hc.2.1 hs hm hi hg hc.2.2 hib call htys hbody hbodyEq]
                · cases hl
              · cases hl

def liftOps (N : Normalizer ib) (Γ : TEnv) (facts : List UniformFacts.Fact) :
    List Op → Option (List Op)
  | [] => some []
  | op :: ops =>
    if op.isTerminator then some (op :: ops) else do
      let _ ← op.operands.mapM (Γ.lookup ·)
      let rs ← op.results.mapM (fun r => vtyOf ib r.ty)
      let tail ← liftOps N ((op.results.map (·.id)).zip rs ++ Γ)
        (UniformFacts.advance facts op) ops
      pure (liftOp N Γ facts op :: tail)

theorem liftOps_sound (N : Normalizer ib) (d : Nat) :
    ∀ Γ facts ops normalized yields ctx call env,
      checkOps (merlinBook ib) d Γ ops yields = true →
      liftOps N Γ facts ops = some normalized →
      EnvTyped ib Γ env → BindingFacts.Valid UniformFacts.model facts env →
      ctx.indexBits = ib → d ≤ ctx.loopIndex.length →
      evalOps opSem ctx call env normalized = evalOps opSem ctx call env ops
  | _, _, [], _, _, _, _, _, h, _, _, _, _, _ => by simp [checkOps] at h
  | Γ, facts, op :: ops, normalized, yields, ctx, call, env, h, hn, he, hf, hib, hd => by
    by_cases hterm : op.isTerminator = true
    · simp only [liftOps, hterm, ↓reduceIte, Option.some.injEq] at hn
      subst normalized; rfl
    · have ht : op.isTerminator = false := by simpa using hterm
      cases op with
      | mk name results operands attrs props regions successors =>
        simp only [checkOps] at h
        split at h
        · cases h
        · next tys hl =>
          rw [ite_eq_right hterm] at h
          split at h
          · cases h
          · next rs hr =>
            split at h
            · cases h
            · next sigs hR =>
              simp only [Bool.and_eq_true] at h
              obtain ⟨hregs, hrest⟩ := h
              simp only [liftOps, ht, Bool.false_eq_true, ↓reduceIte, hl, hr,
                Op.operands, Op.results, bind, Option.bind] at hn
              cases htail : liftOps N ((results.map (·.id)).zip rs ++ Γ)
                  (UniformFacts.advance facts (.mk name results operands attrs props regions successors)) ops with
              | none => simp [htail] at hn
              | some tail =>
                simp only [htail, Option.pure_def, Option.some.injEq] at hn
                subst normalized
                have hspec := liftOp_spec N Γ facts (.mk name results operands attrs props regions successors)
                rw [evalOps_cons_step _ _ _ _ _ _ ht,
                  evalOps_cons_step _ _ _ _ _ _ (hspec.1.trans ht), hspec.2.2 ctx call env he hf hib]
                cases hs : stepOp opSem ctx call env
                    (.mk name results operands attrs props regions successors) with
                | ub => rfl
                | stuck _ => rfl
                | ok next =>
                  exact liftOps_sound N d _ _ ops tail yields ctx call next hrest htail
                    (step_envTyped (merlinBook ib) (op := .mk name results operands attrs props regions successors)
                      hl hr hR hregs hib hd he hs)
                    (UniformFacts.advance_sound hf hs) hib hd

/-- Recheck the intermediate canonical program before applying local normalization. Its types
and nested scopes are obtained from the real transformed AST, never an assumed signature. -/
def liftRun (N : Normalizer ib) (d : Nat) (Γ : TEnv) (facts : List UniformFacts.Fact)
    (ops : List Op) (yields : List VTy) : Option (List Op) := do
  let lifted ← liftOps N Γ facts ops
  if checkOps (merlinBook ib) d Γ lifted yields then
    SplatNormalize.normalize ib Γ facts lifted
  else none

theorem liftRun_sound (N : Normalizer ib) (d : Nat) (Γ : TEnv) (facts : List UniformFacts.Fact)
    (ops normalized : List Op) (yields : List VTy) (ctx : Ctx) (call : CallSem) (env : Env)
    (hc : checkOps (merlinBook ib) d Γ ops yields = true)
    (hn : liftRun N d Γ facts ops yields = some normalized)
    (he : EnvTyped ib Γ env) (hf : BindingFacts.Valid UniformFacts.model facts env)
    (hi : ctx.indexBits = ib) (hd : d ≤ ctx.loopIndex.length) :
    evalOps opSem ctx call env normalized = evalOps opSem ctx call env ops := by
  unfold liftRun at hn
  cases hl : liftOps N Γ facts ops with
  | none => simp [hl] at hn
  | some lifted =>
    simp only [hl, bind, Option.bind] at hn
    split at hn
    · next hcheck =>
      exact (SplatNormalize.normalize_sound (merlinBook ib) d Γ facts lifted normalized yields
        ctx call env hcheck hn he hf hi hd).trans
        (liftOps_sound N d Γ facts ops lifted yields ctx call env hc hl he hf hi hd)
    · cases hn

def lift (N : Normalizer ib) : Normalizer ib := ⟨liftRun N, liftRun_sound N⟩

/-- Each layer enters one more enclosing generic body. A finite caller-selected budget affects
completeness only: unmodified deeper regions must still pass the final alpha comparison. -/
def normalizer (ib : Nat) : Nat → Normalizer ib
  | 0 => base ib
  | depth + 1 => lift (normalizer ib depth)

end MerlinVerify.Check.SplatNested
