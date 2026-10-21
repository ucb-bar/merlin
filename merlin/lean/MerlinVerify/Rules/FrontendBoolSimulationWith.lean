/- Reuse the actual frontend checker under an additive operation interpretation.
The new interpretation must prove SSA blindness, locality of region contexts, and unchanged
scalar arith execution. The original frozen-instance theorem and definitions remain unchanged. -/
import MerlinVerify.Rules.FrontendBoolSimulation
import MerlinVerify.Rules.FrontendMaskRuntime

namespace MerlinVerify.Rules.FrontendBoolSimulationWith
open MerlinVerify MerlinVerify.Check
open FrontendBoolFptosi FrontendBoolProgram FrontendBoolPolicy FrontendBoolSimulation

structure Laws (base : OpSem) : Prop where
  blind : ∀ {ctx call op other args regions}, op.blind = other.blind →
    base ctx call op args regions = base ctx call other args regions
  regions : ∀ ctx call op args left right,
    left.map (FrontendBoolContext.pin ctx) = right.map (FrontendBoolContext.pin ctx) →
    base ctx call op args left = base ctx call op args right
  arith_eq : ∀ ctx call op, Sem.Ops.dialectOf op.name = "arith" → ∀ args regions,
    base ctx call op args regions = Sem.Ops.opSem ctx call op args regions

def policySem (base : OpSem) (types : Types) : OpSem := fun ctx call op args regions =>
  match legalSource types op with
  | none => base ctx call op args regions
  | some _ => match args with
    | [x] => castValue x
    | _ => .stuck "frontend f32 bool cast: expected one operand"

private theorem zero_sem (base : OpSem) (laws : Laws base) (ctx : Ctx) (call : CallSem) (z : Nat) :
    base ctx call (FrontendBoolExecution.zero z) [] [] = .ok [.scalar (.some (.flt .f32 0))] :=
  (laws.arith_eq ctx call _ (by cbv) [] []).trans (FrontendBoolExecution.zero_sem ctx call z)

private theorem cmpValue (base : OpSem) (laws : Laws base) (ctx : Ctx) (call : CallSem) (op : Op)
    (hm : Sem.Ops.unmodelledAttr? op.blind = none)
    (hd : Sem.Ops.dialectOf op.name = "arith") (hn : op.name = "arith.cmpf")
    (hr : Sem.Ops.Arith.resultElt op.blind = some (.int 1))
    (hk : Sem.Ops.Arith.floatKey op.blind = uneKey)
    (hu : UneCorrect ctx.M) (x : RVal) :
    base ctx call op [x, .scalar (.some (.flt .f32 0))] [] = castValue x :=
  (laws.arith_eq ctx call op hd _ []).trans (FrontendBoolPolicy.cmpValue ctx call op hm hd hn hr hk hu x)

theorem site_step (base : OpSem) (laws : Laws base) {types : Types} {ρ next : Rho} {source : Nat} {result : Value}
    {s zero cmp : Op} (hl : legalSource types s = some (source, result))
    (hsite : site ρ source result zero cmp = some next)
    (ctx : Ctx) (hune : UneCorrect ctx.M) (callS callT : CallSem)
    {es et : Env} (hc : Corr ρ es et) :
    EnvRel next (stepOp (policySem base types) ctx callS es s)
      (execOps base ctx callT et [zero, cmp]) := by
  obtain ⟨hsoperands, hsresults, _, _⟩ := legal_facts hl
  obtain ⟨z, r, lhs, zid, hzresults, hrresults, hoperands, hf, rfl⟩ := site_spec hsite
  rcases hf with ⟨rfl, hmap, hhead, hzoperands, hzregions, hregions, hfacts, hfresh⟩
  rcases hfacts with ⟨hm, hd, hn, hr, hk⟩
  have hzterm : zero.isTerminator = false :=
    (isTerminator_of_opHeadEq hhead).symm.trans (by rfl)
  have hcterm : cmp.isTerminator = false := by simp [Op.isTerminator, hn, terminatorNames]
  have hzsem : base ctx callT zero [] [] =
      .ok [.scalar (.some (.flt .f32 0))] :=
    (laws.blind (blind_of_opHeadEq hhead).symm).trans
      (zero_sem base laws ctx callT z.id)
  obtain ⟨_, hfr⟩ := freshIds_sound hfresh
  let ez : Env := (z.id, .scalar (.some (.flt .f32 0))) :: et
  have hcz : Corr ρ es ez := push_target hc (hfr z.id (by simp)) _
  have hfres : freshIds ρ [r.id] = true := by
    unfold freshIds
    simp only [Bool.and_eq_true]
    refine ⟨by simp, List.all_eq_true.mpr ?_⟩
    intro y hy
    have heq : y = r.id := by simpa using hy
    subst y
    exact List.all_eq_true.mpr fun p hp => by simpa using hfr r.id (by simp) p hp
  have hsstep : stepOp (policySem base types) ctx callS es s =
      (Outcome.ofOption "unbound SSA value" (es.get? source)).bind fun x =>
        (castValue x).bind fun rs => es.bindAll [result] rs := by
    simp only [stepOp, hsoperands, Env.getAll, Outcome.mapM]
    cases hx : es.get? source <;>
      simp [Outcome.ofOption, policySem, hl, hsresults]
  have hread : ez.getAll cmp.operands =
      (Outcome.ofOption "unbound SSA value" (es.get? source)).bind
        (fun x => .ok [x, .scalar (.some (.flt .f32 0))]) := by
    rw [hoperands]
    simp only [Env.getAll, Outcome.mapM]
    rw [← hcz source lhs hmap]
    cases hx : es.get? source <;> simp [ez, Env.get?, List.lookup, Outcome.ofOption]
  have htstep : execOps base ctx callT et [zero, cmp] =
      (Outcome.ofOption "unbound SSA value" (es.get? source)).bind fun x =>
        (castValue x).bind fun rs => ez.bindAll [r] rs := by
    rw [execOps_cons_step _ _ _ _ _ _ hzterm]
    simp only [stepOp, hzoperands, hzregions, Env.getAll, Outcome.mapM, evalRegions,
      Outcome.bind_ok, hzsem, hzresults]
    change ((et.bindAll [z] [.scalar (.some (.flt .f32 0))]).bind
      (fun next => execOps base ctx callT next [cmp])) = _
    simp only [Env.bindAll, List.length_cons, List.length_nil, ↓reduceIte, List.map_cons,
      List.map_nil, List.zip_cons_cons, List.zip_nil_left, List.cons_append, List.nil_append,
      Outcome.bind_ok]
    change execOps base ctx callT ez [cmp] = _
    rw [execOps_cons_step _ _ _ _ _ _ hcterm]
    simp only [stepOp, hread, hregions, evalRegions]
    cases hx : es.get? source with
    | none => simp [Outcome.ofOption]
    | some x =>
      simp only [Outcome.ofOption, Outcome.bind_ok]
      rw [cmpValue base laws ctx callT cmp hm hd hn hr hk hune x]
      simp only [hrresults]
      cases ho : castValue x <;>
        simp only [Outcome.bind_ok, Outcome.bind_ub, Outcome.bind_stuck, execOps, bind_right_id]
      rfl
  rw [hsstep, htstep]
  cases hx : es.get? source with
  | none => rfl
  | some x =>
    simp only [Outcome.ofOption, Outcome.bind_ok]
    cases ho : castValue x with
    | ub => trivial
    | stuck reason => rfl
    | ok rs =>
      simpa using bind_corr hcz [result] [r] rs rfl hfres

mutual
  /-- Exact execution simulation of the complete block, with no supplied site-occurrence,
source-value, freshness or initializedness premises. All SSA facts come from the checker. -/
  theorem ops_sound (base : OpSem) (laws : Laws base) (types : Types) (call : CallSem) :
      ∀ (ρ : Rho) (ss ts : List Op) (ctx : Ctx) (es et : Env),
        ops types ρ ss ts = true → UneCorrect ctx.M → Corr ρ es et →
        evalOps (policySem base types) ctx call es ss =
          evalOps base ctx call et ts
    | _, [], [], _, _, _, _, _, _ => rfl
    | ρ, (.mk ns ss os ats ps rs us) :: rest, ts, ctx, es, et, h, hu, hc => by
      unfold ops at h
      split at h
      · next source result hl =>
        cases ts with
        | nil => simp at h
        | cons zero tail =>
          cases tail with
          | nil => simp at h
          | cons cmp targetRest =>
            simp only at h
            split at h
            · next next hs =>
              obtain ⟨_, _, _, hterm⟩ := legal_facts hl
              obtain ⟨hzterm, hcterm⟩ := site_nonterm hs
              rw [evalOps_cons_step _ _ _ _ _ _ hterm]
              change _ = evalOps base ctx call et ([zero, cmp] ++ targetRest)
              rw [evalOps_append _ _ _ _ _ _ (by simpa using And.intro hzterm hcterm)]
              exact envRel_bind (site_step base laws hl hs ctx hu call call hc) _ _
                (fun es' et' hc' => ops_sound base laws types call next rest targetRest ctx es' et' h hu hc')
            · simp at h
      · next hl =>
        cases ts with
        | nil => simp at h
        | cons t targetRest =>
          simp only at h
          split at h
          · next next ho =>
            obtain ⟨hhead, hmap, hregions, hfresh, rfl⟩ := operation_spec ho
            have hterm := isTerminator_of_opHeadEq hhead
            have hblind := blind_of_opHeadEq hhead
            have hlen := results_length_of_opHeadEq hhead
            have hread := Corr.getAll hc _ _ hmap
            cases t with
            | mk nt st ot att pt rt ut =>
              dsimp only [Op.operands] at hread
              rw [evalOps.eq_2, evalOps.eq_2]
              by_cases ht : (Op.mk nt st ot att pt rt ut).isTerminator = true
              · rw [ite_eq_left (hterm.trans ht), ite_eq_left ht, ops_isEmpty h, hread]
              · rw [ite_eq_right (by rwa [hterm]), ite_eq_right ht, hread]
                cases et.getAll ot with
                | ub => rfl
                | stuck reason => rfl
                | ok args =>
                  simp only [Outcome.bind_ok, policySem, hl]
                  have hr := regions_sound base laws types call ρ rs rt ctx es et hregions hu hc
                  rw [laws.regions ctx call _ args _ _ hr,
                    laws.blind hblind]
                  cases base ctx call (.mk nt st ot att pt rt ut) args
                      (evalRegions base call et rt) with
                  | ub => rfl
                  | stuck reason => rfl
                  | ok results =>
                    simp only [Outcome.bind_ok]
                    exact envRel_bind (bind_corr hc _ _ results hlen hfresh) _ _
                      (fun es' et' hc' => ops_sound base laws types call _ rest targetRest ctx es' et' h hu hc')
          · simp at h
    | _, [], _ :: _, _, _, _, h, _, _ => by simp [ops] at h

  theorem regions_sound (base : OpSem) (laws : Laws base) (types : Types) (call : CallSem) :
      ∀ (ρ : Rho) (rs ts : List Region) (ctx : Ctx) (es et : Env),
        regions types ρ rs ts = true → UneCorrect ctx.M → Corr ρ es et →
        (evalRegions (policySem base types) call es rs).map (FrontendBoolContext.pin ctx) =
          (evalRegions base call et ts).map (FrontendBoolContext.pin ctx)
    | _, [], [], _, _, _, _, _, _ => rfl
    | ρ, r :: rs, t :: ts, ctx, es, et, h, hu, hc => by
      simp only [regions, Bool.and_eq_true] at h
      simp only [evalRegions, List.map_cons, List.cons.injEq]
      refine ⟨?_, regions_sound base laws types call ρ rs ts ctx es et h.2 hu hc⟩
      funext inner args
      exact congrFun (region_sound base laws types call ρ r t
        { ctx with loopIndex := inner.loopIndex } es et h.1 hu hc) args
    | _, [], _ :: _, _, _, _, h, _, _ => by simp [regions] at h
    | _, _ :: _, [], _, _, _, h, _, _ => by simp [regions] at h

  theorem region_sound (base : OpSem) (laws : Laws base) (types : Types) (call : CallSem) :
      ∀ (ρ : Rho) (s t : Region) (ctx : Ctx) (es et : Env),
        region types ρ s t = true → UneCorrect ctx.M → Corr ρ es et →
        evalRegion (policySem base types) ctx call es s =
          evalRegion base ctx call et t
    | ρ, .mk [.mk argsS ss], .mk [.mk argsT ts], ctx, es, et, h, hu, hc => by
      simp only [region, Bool.and_eq_true, beq_iff_eq] at h
      obtain ⟨⟨hty, hfresh⟩, hops⟩ := h
      have hl : argsS.length = argsT.length := by simpa using congrArg List.length hty
      simp only [evalRegion, evalBlocks, evalBlock]
      funext vals
      exact envRel_bind (bind_corr hc argsS argsT vals hl hfresh) _ _
        (fun es' et' hc' => ops_sound base laws types call _ ss ts ctx es' et' hops hu hc')
    | _, .mk [], .mk [], _, _, _, _, _, _ => rfl
    | _, .mk [], .mk (_ :: _ :: _), _, _, _, h, _, _ => by simp [region] at h
    | _, .mk (_ :: _ :: _), .mk [], _, _, _, h, _, _ => by simp [region] at h
    | _, .mk (_ :: _ :: _), .mk (_ :: _ :: _), _, _, _, h, _, _ => by simp [region] at h
    | _, .mk [], .mk [_], _, _, _, h, _, _ => by simp [region] at h
    | _, .mk [_], .mk [], _, _, _, h, _, _ => by simp [region] at h
    | _, .mk [_], .mk (_ :: _ :: _), _, _, _, h, _, _ => by simp [region] at h
    | _, .mk (_ :: _ :: _), .mk [_], _, _, _, h, _, _ => by simp [region] at h
end
theorem evalFunc_sound (base : OpSem) (laws : Laws base) {types : Types} {src tgt : Module}
    (h : functions types src.funcs tgt.funcs = true) (ctx : Ctx) (hu : UneCorrect ctx.M) :
    ∀ fuel, evalFunc (policySem base types) ctx src fuel =
      evalFunc base ctx tgt fuel := by
  intro fuel
  induction fuel with
  | zero => rfl
  | succ fuel ih =>
    funext name args
    have hf := functions_find src.funcs tgt.funcs h name
    change Found types (src.func? name) (tgt.func? name) at hf
    cases hs : src.func? name with
    | none =>
      cases ht : tgt.func? name with
      | none => simp [evalFunc, hs, ht]
      | some t => simp [hs, ht, Found] at hf
    | some s =>
      cases ht : tgt.func? name with
      | none => simp [hs, ht, Found] at hf
      | some t =>
        simp only [hs, ht, Found] at hf
        obtain ⟨_, hd, hr⟩ := function_spec hf
        simp only [evalFunc, hs, ht, hd]
        split
        · rfl
        · rw [ih]
          exact congrFun (region_sound base laws types (evalFunc base ctx tgt fuel)
            [] s.body t.body ctx [] [] hr hu (by intro x y hxy; simp [Rho.find] at hxy)) args

/-- The same source-only policy, with the reviewed primitive and an explicit base interpretation. -/
def policyDenote (base : OpSem) (src : Module) (ctx : Ctx) (name : String) (args : List RVal) :
    Outcome (List RVal) :=
  match defsOps 64 src.ops with
  | none => .stuck "frontend source inventory exceeds supported nesting"
  | some defs => denote (policySem base (sourceTypes defs)) ctx src name args

/-- The frozen-instance source reference is definitionally retained. -/
theorem original_policy (src : Module) (ctx : Ctx) (name : String) (args : List RVal) :
    policyDenote Sem.Ops.opSem src ctx name args = FrontendBoolSimulation.policyDenote src ctx name args := rfl

/-- Acceptance of the unchanged actual-AST checker implies exact candidate-program execution.
Every interpretation-specific obligation is proved through Laws, rather than asserted by a flag. -/
theorem check_sound (base : OpSem) (laws : Laws base) {src tgt : Module}
    (h : FrontendBoolSimulation.check src tgt = true) (ctx : Ctx) (hu : UneCorrect ctx.M)
    (name : String) (args : List RVal) :
    policyDenote base src ctx name args = denote base ctx tgt name args := by
  simp only [FrontendBoolSimulation.check, Bool.and_eq_true] at h
  cases hd : defsOps 64 src.ops with
  | none => simp [hd] at h
  | some defs =>
    simp only [hd] at h
    have heq := evalFunc_sound base laws h.2 ctx hu
    have hlen := functions_length src.funcs tgt.funcs h.2
    simp only [policyDenote, hd, denote]
    rw [heq, hlen]

/-- With no eligible source types the existing checker is ordinary alpha correspondence. -/
theorem legalSource_nil (op : Op) : FrontendBoolProgram.legalSource [] op = none := by
  unfold FrontendBoolProgram.legalSource
  repeat' first | split | simp_all

theorem policy_nil (base : OpSem) : policySem base [] = base := by
  funext ctx call op args regions
  simp [policySem, legalSource_nil]

theorem candidateLaws (limit : Nat) : Laws (FrontendMaskRuntime.sem limit) where
  blind := FrontendMaskRuntime.sem_blind
  regions := FrontendMaskRuntime.sem_regions limit
  arith_eq := FrontendMaskRuntime.arith_eq limit

/-- The bool repair remains exact under the inactive runtime candidate and the already approved
IEEE f32 instance. This does not activate or approve the candidate runtime interpretation. -/
theorem check_sound_candidate {src tgt : Module} (h : FrontendBoolSimulation.check src tgt = true)
    (limit : Nat) (base : FloatModel) (ctx : Ctx) (name : String) (args : List RVal) :
    policyDenote (FrontendMaskRuntime.sem limit) src { ctx with M := FrontendF32Comparison.model base } name args =
      denote (FrontendMaskRuntime.sem limit) { ctx with M := FrontendF32Comparison.model base } tgt name args :=
  check_sound _ (candidateLaws limit) h _ (FrontendF32Comparison.model_correct base) name args

#print axioms check_sound_candidate
end MerlinVerify.Rules.FrontendBoolSimulationWith
