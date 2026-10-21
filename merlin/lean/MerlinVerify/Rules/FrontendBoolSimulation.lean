/- Executable source-to-materialization simulation. The source interpretation executes A2's
primitive directly. This is additive proof machinery, not a reviewed policy declaration. -/
import MerlinVerify.Rules.FrontendBoolPolicy

namespace MerlinVerify.Rules.FrontendBoolSimulation
open MerlinVerify MerlinVerify.Check
open FrontendBoolFptosi FrontendBoolProgram FrontendBoolPolicy

abbrev Types := List (ValueId × Ty)

def CmpFacts (op : Op) : Prop :=
  Sem.Ops.unmodelledAttr? op.blind = none ∧
  Sem.Ops.dialectOf op.name = "arith" ∧ op.name = "arith.cmpf" ∧
  Sem.Ops.Arith.resultElt op.blind = some (.int 1) ∧
  Sem.Ops.Arith.floatKey op.blind = uneKey
instance (op : Op) : Decidable (CmpFacts op) := inferInstanceAs (Decidable (_ ∧ _))

def SiteFacts (ρ : Rho) (source : Nat) (zero cmp : Op) (z r : Value) (lhs zid : Nat) : Prop :=
  zid = z.id ∧ ρ.find source = some lhs ∧
  opHeadEq (FrontendBoolExecution.zero z.id) zero = true ∧
  zero.operands = [] ∧ zero.regions = [] ∧ cmp.regions = [] ∧
  CmpFacts cmp ∧ freshIds ρ [z.id, r.id] = true
instance (ρ : Rho) (source : Nat) (zero cmp : Op) (z r : Value) (lhs zid : Nat) :
    Decidable (SiteFacts ρ source zero cmp z r lhs zid) :=
  inferInstanceAs (Decidable (_ ∧ _))

/-- All facts are recomputed from the two actual operations and current SSA correspondence. -/
def site (ρ : Rho) (source : Nat) (result : Value) (zero cmp : Op) : Option Rho :=
  match zero.results, cmp.results, cmp.operands with
  | [z], [r], [lhs, zid] =>
    if SiteFacts ρ source zero cmp z r lhs zid then some ((result.id, r.id) :: ρ) else none
  | _, _, _ => none

mutual
  def ops (types : Types) (ρ : Rho) : List Op → List Op → Bool
    | [], [] => true
    | s :: ss, ts =>
      match legalSource types s with
      | some (source, result) =>
        match ts with
        | zero :: cmp :: rest =>
          match site ρ source result zero cmp with
          | some next => ops types next ss rest
          | none => false
        | _ => false
      | none =>
        match ts with
        | t :: rest => match operation types ρ s t with
          | some next => ops types next ss rest
          | none => false
        | [] => false
    | _, _ => false

  def operation (types : Types) (ρ : Rho) : Op → Op → Option Rho
    | s@(.mk _ resS opsS _ _ regsS _), t@(.mk _ resT opsT _ _ regsT _) =>
      if opHeadEq s t && opsS.mapM ρ.find == some opsT && regions types ρ regsS regsT &&
          freshIds ρ (resT.map (·.id))
      then some ((resS.map (·.id)).zip (resT.map (·.id)) ++ ρ) else none

  def regions (types : Types) (ρ : Rho) : List Region → List Region → Bool
    | [], [] => true
    | r :: rs, t :: ts => region types ρ r t && regions types ρ rs ts
    | _, _ => false

  def region (types : Types) (ρ : Rho) : Region → Region → Bool
    | .mk [.mk argsS ss], .mk [.mk argsT ts] =>
      argsS.map (·.ty) == argsT.map (·.ty) && freshIds ρ (argsT.map (·.id)) &&
        ops types ((argsS.map (·.id)).zip (argsT.map (·.id)) ++ ρ) ss ts
    | .mk [], .mk [] => true
    | _, _ => false
end

theorem legal_facts {types : Types} {op : Op} {source : Nat} {result : Value}
    (h : legalSource types op = some (source, result)) :
    op.operands = [source] ∧ op.results = [result] ∧ result.ty = .int 1 ∧
    op.isTerminator = false := by
  unfold legalSource at h
  repeat' first | (split at h <;> simp_all [Op.isTerminator, terminatorNames])
  rcases h with ⟨⟨hty, _⟩, _, _, hs, hr⟩
  subst_vars
  by_cases hn : op.name = "arith.fptosi" <;> simp_all

theorem site_spec {ρ next : Rho} {source : Nat} {result : Value} {zero cmp : Op}
    (h : site ρ source result zero cmp = some next) :
    ∃ z r lhs zid, zero.results = [z] ∧ cmp.results = [r] ∧
      cmp.operands = [lhs, zid] ∧ SiteFacts ρ source zero cmp z r lhs zid ∧
      next = (result.id, r.id) :: ρ := by
  unfold site at h
  split at h
  · next z r lhs zid hz hr hop =>
    split at h
    · next hf => exact ⟨z, r, lhs, zid, hz, hr, hop, hf, (Option.some.inj h).symm⟩
    · simp at h
  · simp at h

theorem bind_right_id {α : Type} (o : Outcome α) : o.bind Outcome.ok = o := by
  cases o <;> rfl

def EnvRel (ρ : Rho) : Outcome Env → Outcome Env → Prop
  | .ub, .ub => True
  | .stuck s, .stuck t => s = t
  | .ok s, .ok t => Corr ρ s t
  | _, _ => False

theorem envRel_bind {ρ : Rho} {s t : Outcome Env}
    (h : EnvRel ρ s t) (fs ft : Env → Outcome (List RVal))
    (hk : ∀ es et, Corr ρ es et → fs es = ft et) : s.bind fs = t.bind ft := by
  cases s <;> cases t <;> simp_all [EnvRel, Outcome.bind]
  exact hk _ _ h

theorem push_target {ρ : Rho} {es et : Env} (hc : Corr ρ es et)
    {z : Nat} (hz : Rho.notImage ρ z) (v : RVal) : Corr ρ es ((z, v) :: et) := by
  intro x y hxy
  obtain ⟨p, hp, hpy⟩ := lookup_mem_image hxy
  have hy : y ≠ z := by rw [← hpy]; exact hz p hp
  simpa [Env.get?, List.lookup, beq_eq_false_iff_ne.mpr hy] using hc x y hxy

theorem bind_corr {ρ : Rho} {es et : Env} (hc : Corr ρ es et)
    (ss ts : List Value) (rs : List RVal) (hl : ss.length = ts.length)
    (hf : freshIds ρ (ts.map (·.id)) = true) :
    EnvRel ((ss.map (·.id)).zip (ts.map (·.id)) ++ ρ)
      (es.bindAll ss rs) (et.bindAll ts rs) := by
  by_cases hk : ss.length = rs.length
  · rw [bindAll_eq_ok hk, bindAll_eq_ok (hl ▸ hk)]
    obtain ⟨hnd, hfr⟩ := freshIds_sound hf
    exact Corr.extend hc _ _ rs (by simpa using hk) (by simpa [← hl] using hk) hnd hfr
  · rw [bindAll_eq_stuck hk, bindAll_eq_stuck (fun e => hk (hl ▸ e))]
    rfl

/-- One checked repair steps in exact lockstep, including failed operand reads, poison and
ill-typed runtime values. The source value and freshness facts are derived here, not supplied. -/
theorem site_step {types : Types} {ρ next : Rho} {source : Nat} {result : Value}
    {s zero cmp : Op} (hl : legalSource types s = some (source, result))
    (hsite : site ρ source result zero cmp = some next)
    (ctx : Ctx) (hune : UneCorrect ctx.M) (callS callT : CallSem)
    {es et : Env} (hc : Corr ρ es et) :
    EnvRel next (stepOp (FrontendBoolPolicy.sem types) ctx callS es s)
      (execOps Sem.Ops.opSem ctx callT et [zero, cmp]) := by
  obtain ⟨hsoperands, hsresults, _, _⟩ := legal_facts hl
  obtain ⟨z, r, lhs, zid, hzresults, hrresults, hoperands, hf, rfl⟩ := site_spec hsite
  rcases hf with ⟨rfl, hmap, hhead, hzoperands, hzregions, hregions, hfacts, hfresh⟩
  rcases hfacts with ⟨hm, hd, hn, hr, hk⟩
  have hzterm : zero.isTerminator = false :=
    (isTerminator_of_opHeadEq hhead).symm.trans (by rfl)
  have hcterm : cmp.isTerminator = false := by simp [Op.isTerminator, hn, terminatorNames]
  have hzsem : Sem.Ops.opSem ctx callT zero [] [] =
      .ok [.scalar (.some (.flt .f32 0))] :=
    (Sem.Ops.opSem_blind (blind_of_opHeadEq hhead).symm).trans
      (FrontendBoolExecution.zero_sem ctx callT z.id)
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
  have hsstep : stepOp (FrontendBoolPolicy.sem types) ctx callS es s =
      (Outcome.ofOption "unbound SSA value" (es.get? source)).bind fun x =>
        (castValue x).bind fun rs => es.bindAll [result] rs := by
    simp only [stepOp, hsoperands, Env.getAll, Outcome.mapM]
    cases hx : es.get? source <;>
      simp [Outcome.ofOption, FrontendBoolPolicy.sem, hl, hsresults]
  have hread : ez.getAll cmp.operands =
      (Outcome.ofOption "unbound SSA value" (es.get? source)).bind
        (fun x => .ok [x, .scalar (.some (.flt .f32 0))]) := by
    rw [hoperands]
    simp only [Env.getAll, Outcome.mapM]
    rw [← hcz source lhs hmap]
    cases hx : es.get? source <;> simp [ez, Env.get?, List.lookup, Outcome.ofOption]
  have htstep : execOps Sem.Ops.opSem ctx callT et [zero, cmp] =
      (Outcome.ofOption "unbound SSA value" (es.get? source)).bind fun x =>
        (castValue x).bind fun rs => ez.bindAll [r] rs := by
    rw [execOps_cons_step _ _ _ _ _ _ hzterm]
    simp only [stepOp, hzoperands, hzregions, Env.getAll, Outcome.mapM, evalRegions,
      Outcome.bind_ok, hzsem, hzresults]
    change ((et.bindAll [z] [.scalar (.some (.flt .f32 0))]).bind
      (fun next => execOps Sem.Ops.opSem ctx callT next [cmp])) = _
    simp only [Env.bindAll, List.length_cons, List.length_nil, ↓reduceIte, List.map_cons,
      List.map_nil, List.zip_cons_cons, List.zip_nil_left, List.cons_append, List.nil_append,
      Outcome.bind_ok]
    change execOps Sem.Ops.opSem ctx callT ez [cmp] = _
    rw [execOps_cons_step _ _ _ _ _ _ hcterm]
    simp only [stepOp, hread, hregions, evalRegions]
    cases hx : es.get? source with
    | none => simp [Outcome.ofOption]
    | some x =>
      simp only [Outcome.ofOption, Outcome.bind_ok]
      rw [FrontendBoolPolicy.cmpValue ctx callT cmp hm hd hn hr hk hune x]
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

theorem site_nonterm {ρ next : Rho} {source : Nat} {result : Value} {zero cmp : Op}
    (h : site ρ source result zero cmp = some next) :
    zero.isTerminator = false ∧ cmp.isTerminator = false := by
  obtain ⟨z, r, lhs, zid, _, _, _, hf, _⟩ := site_spec h
  exact ⟨(isTerminator_of_opHeadEq hf.2.2.1).symm.trans (by rfl),
    by simp [Op.isTerminator, hf.2.2.2.2.2.2.1.2.2.1, terminatorNames]⟩

theorem operation_spec {types : Types} {ρ next : Rho} {s t : Op}
    (h : operation types ρ s t = some next) :
    opHeadEq s t = true ∧ s.operands.mapM ρ.find = some t.operands ∧
    regions types ρ s.regions t.regions = true ∧ freshIds ρ (t.results.map (·.id)) = true ∧
    next = (s.results.map (·.id)).zip (t.results.map (·.id)) ++ ρ := by
  cases s with
  | mk ns ss os as ps rs us =>
    cases t with
    | mk nt st ot attrsT pt rt ut =>
      simp only [operation] at h
      split at h
      · next hc =>
        simp only [Option.some.injEq] at h
        simp only [Bool.and_eq_true, beq_iff_eq] at hc
        exact ⟨hc.1.1.1, hc.1.1.2, hc.1.2, hc.2, h.symm⟩
      · simp at h

theorem ops_isEmpty {types : Types} {ρ : Rho} {ss ts : List Op}
    (h : ops types ρ ss ts = true) : ss.isEmpty = ts.isEmpty := by
  cases ss with
  | nil => cases ts <;> simp_all [ops]
  | cons s ss =>
    cases ts with
    | cons t ts => rfl
    | nil => cases hl : legalSource types s <;> simp_all [ops]

mutual
  /-- Exact execution simulation of the complete block, with no supplied site-occurrence,
source-value, freshness or initializedness premises. All SSA facts come from the checker. -/
  theorem ops_sound (types : Types) (call : CallSem) :
      ∀ (ρ : Rho) (ss ts : List Op) (ctx : Ctx) (es et : Env),
        ops types ρ ss ts = true → UneCorrect ctx.M → Corr ρ es et →
        evalOps (FrontendBoolPolicy.sem types) ctx call es ss =
          evalOps Sem.Ops.opSem ctx call et ts
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
              change _ = evalOps Sem.Ops.opSem ctx call et ([zero, cmp] ++ targetRest)
              rw [evalOps_append _ _ _ _ _ _ (by simpa using And.intro hzterm hcterm)]
              exact envRel_bind (site_step hl hs ctx hu call call hc) _ _
                (fun es' et' hc' => ops_sound types call next rest targetRest ctx es' et' h hu hc')
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
                  simp only [Outcome.bind_ok, FrontendBoolPolicy.sem, hl]
                  have hr := regions_sound types call ρ rs rt ctx es et hregions hu hc
                  rw [FrontendBoolContext.opSem_regions ctx call _ args _ _ hr,
                    Sem.Ops.opSem_blind hblind]
                  cases Sem.Ops.opSem ctx call (.mk nt st ot att pt rt ut) args
                      (evalRegions Sem.Ops.opSem call et rt) with
                  | ub => rfl
                  | stuck reason => rfl
                  | ok results =>
                    simp only [Outcome.bind_ok]
                    exact envRel_bind (bind_corr hc _ _ results hlen hfresh) _ _
                      (fun es' et' hc' => ops_sound types call _ rest targetRest ctx es' et' h hu hc')
          · simp at h
    | _, [], _ :: _, _, _, _, h, _, _ => by simp [ops] at h

  theorem regions_sound (types : Types) (call : CallSem) :
      ∀ (ρ : Rho) (rs ts : List Region) (ctx : Ctx) (es et : Env),
        regions types ρ rs ts = true → UneCorrect ctx.M → Corr ρ es et →
        (evalRegions (FrontendBoolPolicy.sem types) call es rs).map (FrontendBoolContext.pin ctx) =
          (evalRegions Sem.Ops.opSem call et ts).map (FrontendBoolContext.pin ctx)
    | _, [], [], _, _, _, _, _, _ => rfl
    | ρ, r :: rs, t :: ts, ctx, es, et, h, hu, hc => by
      simp only [regions, Bool.and_eq_true] at h
      simp only [evalRegions, List.map_cons, List.cons.injEq]
      refine ⟨?_, regions_sound types call ρ rs ts ctx es et h.2 hu hc⟩
      funext inner args
      exact congrFun (region_sound types call ρ r t
        { ctx with loopIndex := inner.loopIndex } es et h.1 hu hc) args
    | _, [], _ :: _, _, _, _, h, _, _ => by simp [regions] at h
    | _, _ :: _, [], _, _, _, h, _, _ => by simp [regions] at h

  theorem region_sound (types : Types) (call : CallSem) :
      ∀ (ρ : Rho) (s t : Region) (ctx : Ctx) (es et : Env),
        region types ρ s t = true → UneCorrect ctx.M → Corr ρ es et →
        evalRegion (FrontendBoolPolicy.sem types) ctx call es s =
          evalRegion Sem.Ops.opSem ctx call et t
    | ρ, .mk [.mk argsS ss], .mk [.mk argsT ts], ctx, es, et, h, hu, hc => by
      simp only [region, Bool.and_eq_true, beq_iff_eq] at h
      obtain ⟨⟨hty, hfresh⟩, hops⟩ := h
      have hl : argsS.length = argsT.length := by simpa using congrArg List.length hty
      simp only [evalRegion, evalBlocks, evalBlock]
      funext vals
      exact envRel_bind (bind_corr hc argsS argsT vals hl hfresh) _ _
        (fun es' et' hc' => ops_sound types call _ ss ts ctx es' et' hops hu hc')
    | _, .mk [], .mk [], _, _, _, _, _, _ => rfl
    | _, .mk [], .mk (_ :: _ :: _), _, _, _, h, _, _ => by simp [region] at h
    | _, .mk (_ :: _ :: _), .mk [], _, _, _, h, _, _ => by simp [region] at h
    | _, .mk (_ :: _ :: _), .mk (_ :: _ :: _), _, _, _, h, _, _ => by simp [region] at h
    | _, .mk [], .mk [_], _, _, _, h, _, _ => by simp [region] at h
    | _, .mk [_], .mk [], _, _, _, h, _, _ => by simp [region] at h
    | _, .mk [_], .mk (_ :: _ :: _), _, _, _, h, _, _ => by simp [region] at h
    | _, .mk (_ :: _ :: _), .mk [_], _, _, _, h, _, _ => by simp [region] at h
end

def function (types : Types) (s t : Func) : Bool :=
  s.name == t.name && s.isDecl == t.isDecl && region types [] s.body t.body

def functions (types : Types) : List Func → List Func → Bool
  | [], [] => true
  | s :: ss, t :: ts => function types s t && functions types ss ts
  | _, _ => false

theorem function_spec {types : Types} {s t : Func} (h : function types s t = true) :
    s.name = t.name ∧ s.isDecl = t.isDecl ∧ region types [] s.body t.body = true := by
  simpa only [function, Bool.and_eq_true, beq_iff_eq, and_assoc] using h

def Found (types : Types) : Option Func → Option Func → Prop
  | none, none => True
  | some s, some t => function types s t = true
  | _, _ => False

theorem functions_find {types : Types} : ∀ (ss ts : List Func),
    functions types ss ts = true → ∀ name,
      Found types (ss.find? (·.name == name)) (ts.find? (·.name == name))
  | [], [], _, _ => trivial
  | s :: ss, t :: ts, h, name => by
    simp only [functions, Bool.and_eq_true] at h
    have hn := (function_spec h.1).1
    simp only [List.find?_cons, hn]
    split
    · exact h.1
    · exact functions_find ss ts h.2 name
  | [], _ :: _, h, _ => by simp [functions] at h
  | _ :: _, [], h, _ => by simp [functions] at h

theorem functions_length {types : Types} : ∀ (ss ts : List Func),
    functions types ss ts = true → ss.length = ts.length
  | [], [], _ => rfl
  | _ :: ss, _ :: ts, h => by
    simp only [functions, Bool.and_eq_true] at h
    simpa using functions_length ss ts h.2
  | [], _ :: _, h => by simp [functions] at h
  | _ :: _, [], h => by simp [functions] at h

/-- Fuel, symbol lookup, external declarations and calls are preserved. Region execution is
proved using the actual evaluator and the checked SSA correspondence, including tensor bodies. -/
theorem evalFunc_sound {types : Types} {src tgt : Module}
    (h : functions types src.funcs tgt.funcs = true) (ctx : Ctx) (hu : UneCorrect ctx.M) :
    ∀ fuel, evalFunc (FrontendBoolPolicy.sem types) ctx src fuel =
      evalFunc Sem.Ops.opSem ctx tgt fuel := by
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
          exact congrFun (region_sound types (evalFunc Sem.Ops.opSem ctx tgt fuel)
            [] s.body t.body ctx [] [] hr hu (by intro x y hxy; simp [Rho.find] at hxy)) args

/-- Source-only repair denotation. A2 is called at eligible casts in the original program;
this definition neither constructs nor consults the after program or comparison materialization. -/
def policyDenote (src : Module) (ctx : Ctx) (name : String) (args : List RVal) :
    Outcome (List RVal) :=
  match defsOps 64 src.ops with
  | none => .stuck "frontend source inventory exceeds supported nesting"
  | some defs => denote (FrontendBoolPolicy.sem (sourceTypes defs)) ctx src name args

/-- Preserve the complete materialization, attributes and SSA accounting. Function execution is
proved by the source-policy simulation, including internal calls; the older C1 module filter is
not needed as a second execution proof. -/
def structural (src tgt : Module) : Bool :=
  match intent? src with
  | none => false
  | some intended => repairsReady 64 intended.ops && alphaOps [] intended.ops tgt.ops &&
      (outsideChanges intended tgt).unverified.isEmpty

/-- Both independent source-driven structure and actual execution correspondence are required. -/
def check (src tgt : Module) : Bool :=
  structural src tgt &&
    match defsOps 64 src.ops with
    | none => false
    | some defs => functions (sourceTypes defs) src.funcs tgt.funcs

/-- Acceptance to complete program execution against A2's source interpretation. The only
float premise is explicit. No site membership, environment, freshness or use premise is supplied. -/
theorem check_sound {src tgt : Module} (h : check src tgt = true)
    (ctx : Ctx) (hu : UneCorrect ctx.M) (name : String) (args : List RVal) :
    policyDenote src ctx name args = denote Sem.Ops.opSem ctx tgt name args := by
  simp only [check, Bool.and_eq_true] at h
  cases hd : defsOps 64 src.ops with
  | none => simp [hd] at h
  | some defs =>
    simp only [hd] at h
    have heq := evalFunc_sound h.2 ctx hu
    have hlen := functions_length src.funcs tgt.funcs h.2
    simp only [policyDenote, hd, denote]
    rw [heq, hlen]

/-- All initialized f32 bit patterns, NaNs and poison, lifted to the complete accepted program
under the proposed IEEE instance. Human specification/model review is still a separate decision. -/
theorem check_sound_ieee {src tgt : Module} (h : check src tgt = true)
    (base : FloatModel) (ctx : Ctx) (name : String) (args : List RVal) :
    policyDenote src { ctx with M := FrontendF32Comparison.model base } name args =
      denote Sem.Ops.opSem { ctx with M := FrontendF32Comparison.model base } tgt name args :=
  check_sound h _ (FrontendF32Comparison.model_correct base) name args

#print axioms ops_sound
#print axioms check_sound_ieee
end MerlinVerify.Rules.FrontendBoolSimulation
