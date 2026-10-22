/- Complete-program correspondence for fix_bool_sitofp's f32 frontend intent.
Every changed site is recomputed from the source AST; all remaining syntax, SSA, nested regions,
function lookup and calls are checked. This proves a local repair policy, not original signed-i1
cast equivalence or defined execution of every unchanged model operation. Contract admission is pending. -/
import MerlinVerify.Rules.BoolToFloat

namespace MerlinVerify.Rules.BoolToFloatProgram
open MerlinVerify MerlinVerify.Check
open FrontendBoolProgram (defsOps sourceTypes)
open BoolToFloat
open FrontendBoolSimulation (EnvRel envRel_bind bind_corr)

/-- Single-site acceptance checks the unsigned key, nonNeg, types, operands and SSA freshness. -/
def site (ρ : Rho) (source : Nat) (result : Value) (target : Op) : Option Rho :=
  match target.results, target.operands with
  | [r], [lhs] =>
    if ρ.find source = some lhs ∧ target.regions = [] ∧ CastFacts target ∧
        r.ty = .float .f32 ∧ freshIds ρ [r.id] = true
    then some ((result.id, r.id) :: ρ) else none
  | _, _ => none

mutual
  def ops (types : Types) (ρ : Rho) : List Op → List Op → Bool
    | [], [] => true
    | s :: ss, ts =>
      match legalSource types s with
      | some (source, result) =>
        match ts with
        | t :: rest =>
          match site ρ source result t with
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
    op.operands = [source] ∧ op.results = [result] ∧ result.ty = .float .f32 ∧
    op.isTerminator = false := by
  unfold legalSource at h
  repeat' first | (split at h <;> simp_all [Op.isTerminator, terminatorNames])
  all_goals rcases h with ⟨⟨hty, _⟩, _, _, hs, hr⟩; subst_vars; assumption

theorem site_spec {ρ next : Rho} {source : Nat} {result : Value} {target : Op}
    (h : site ρ source result target = some next) :
    ∃ r lhs, target.results = [r] ∧ target.operands = [lhs] ∧
      ρ.find source = some lhs ∧ target.regions = [] ∧ CastFacts target ∧
      r.ty = .float .f32 ∧ freshIds ρ [r.id] = true ∧ next = (result.id, r.id) :: ρ := by
  unfold site at h
  split at h
  · next r lhs hr hop =>
    split at h
    · next hf => exact ⟨r, lhs, hr, hop, hf.1, hf.2.1, hf.2.2.1, hf.2.2.2.1,
        hf.2.2.2.2, (Option.some.inj h).symm⟩
    · simp at h
  · simp at h

theorem site_nonterm {ρ next : Rho} {source : Nat} {result : Value} {target : Op}
    (h : site ρ source result target = some next) : target.isTerminator = false := by
  obtain ⟨_, _, _, _, _, _, hf, _, _, _⟩ := site_spec h
  simp [Op.isTerminator, hf.2.2.1, terminatorNames]

theorem site_step {types : Types} {ρ next : Rho} {source : Nat} {result : Value}
    {s t : Op} (hl : legalSource types s = some (source, result))
    (hs : site ρ source result t = some next)
    (ctx : Ctx) (hu : UnsignedBoolCorrect ctx.M) (callS callT : CallSem)
    {es et : Env} (hc : Corr ρ es et) :
    EnvRel next (stepOp (BoolToFloat.sem types) ctx callS es s)
      (stepOp Sem.Ops.opSem ctx callT et t) := by
  obtain ⟨hsoperands, hsresults, _, _⟩ := legal_facts hl
  obtain ⟨r, lhs, htresults, htoperands, hmap, hregs, hf, _, hfresh, rfl⟩ := site_spec hs
  have hread : es.getAll [source] = et.getAll [lhs] :=
    Corr.getAll hc _ _ (by simp [hmap])
  simp only [stepOp, hsoperands, htoperands, hsresults, htresults]
  simp only [BoolToFloat.sem, hl]
  rw [hread]
  cases he : et.getAll [lhs] with
  | ub => trivial
  | stuck reason => rfl
  | ok args =>
    simp only [Outcome.bind_ok]
    rw [cast_exec ctx callT t hf hu args]
    cases hv : cast ctx.M args with
    | ub => trivial
    | stuck reason => rfl
    | ok rs => simpa using bind_corr hc [result] [r] rs rfl hfresh

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
        ops types ρ ss ts = true → UnsignedBoolCorrect ctx.M → Corr ρ es et →
        evalOps (BoolToFloat.sem types) ctx call es ss =
          evalOps Sem.Ops.opSem ctx call et ts
    | _, [], [], _, _, _, _, _, _ => rfl
    | ρ, (.mk ns ss os ats ps rs us) :: rest, ts, ctx, es, et, h, hu, hc => by
      unfold ops at h
      split at h
      · next source result hl =>
        cases ts with
        | nil => simp at h
        | cons t targetRest =>
          simp only at h
          split at h
          · next next hs =>
            obtain ⟨_, _, _, hterm⟩ := legal_facts hl
            have htterm := site_nonterm hs
            rw [evalOps_cons_step _ _ _ _ _ _ hterm,
              evalOps_cons_step _ _ _ _ _ _ htterm]
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
                  simp only [Outcome.bind_ok, BoolToFloat.sem, hl]
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
        regions types ρ rs ts = true → UnsignedBoolCorrect ctx.M → Corr ρ es et →
        (evalRegions (BoolToFloat.sem types) call es rs).map (FrontendBoolContext.pin ctx) =
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
        region types ρ s t = true → UnsignedBoolCorrect ctx.M → Corr ρ es et →
        evalRegion (BoolToFloat.sem types) ctx call es s =
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
    (h : functions types src.funcs tgt.funcs = true) (ctx : Ctx) (hu : UnsignedBoolCorrect ctx.M) :
    ∀ fuel, evalFunc (BoolToFloat.sem types) ctx src fuel =
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

/-- Source-driven exact materialization used solely for complete syntax accounting.
The source execution policy above invokes A2 directly and never consults this materialization. -/
def replacement (o : Op) : Op :=
  let kept : List (String × Attr) := o.attrs.filter fun (k, _) => k.startsWith "prov."
  let previous := match kept.lookup "prov.transforms" with
    | some (Attr.str s) => s.splitOn "," | _ => []
  let chain := previous.filter (· != "") ++ ["bool_sitofp_uitofp"]
  let attrs := (kept.filter (fun (k, _) => k != "prov.transforms") ++
    [("prov.transforms", Attr.str (String.intercalate "," chain))]).mergeSort (fun a b => a.1 ≤ b.1)
  .mk "arith.uitofp" o.results o.operands attrs [] [] []

def rewriteOps (types : Types) : Nat → List Op → StateT Nat Option (List Op)
  | 0, _ => failure
  | fuel + 1, os => os.mapM fun o => do
    if o.name == "arith.sitofp" && o.operands.any (fun x => types.lookup x == some (.int 1)) then
      let some _ := legalSource types o | failure
      modify (· + 1)
      return replacement o
    let rs ← o.regions.mapM fun r => do
      let bs ← r.blocks.mapM fun b => do
        let children ← rewriteOps types fuel b.ops
        pure (.mk b.args children : Block)
      pure (.mk bs : Region)
    match o with
    | .mk n res args attrs props _ successors => pure (.mk n res args attrs props rs successors)

def intent? (src : Module) : Option Module := do
  let defs ← defsOps 64 src.ops
  if !decide (defs.map (·.id)).Nodup then none else
  let (ops, count) ← (rewriteOps (sourceTypes defs) 64 src.ops).run 0
  if count == 0 then none else pure {src with ops}

def structural (src tgt : Module) : Bool :=
  match intent? src with
  | none => false
  | some intended => decide (intended.attrs = tgt.attrs ∧ intended.props = tgt.props) &&
      alphaOps [] intended.ops tgt.ops &&
      (outsideChanges intended tgt).unverified.isEmpty

def check (src tgt : Module) : Bool :=
  structural src tgt &&
    match defsOps 64 src.ops with
    | none => false
    | some defs => functions (sourceTypes defs) src.funcs tgt.funcs

/-- No construction or reference to target syntax occurs in the source policy denotation. -/
def policyDenote (src : Module) (ctx : Ctx) (name : String) (args : List RVal) :
    Outcome (List RVal) :=
  match defsOps 64 src.ops with
  | none => .stuck "bool-to-float source inventory exceeds supported nesting"
  | some defs => denote (BoolToFloat.sem (sourceTypes defs)) ctx src name args

/-- Complete-program execution of the declared repair policy, with the conversion assumption
explicit. The result covers internal calls, poison, tensor-region lifting, UB and stuck outcomes;
it makes no initialized-return or original signed-cast equivalence claim. -/
theorem check_sound {src tgt : Module} (h : check src tgt = true)
    (ctx : Ctx) (hu : UnsignedBoolCorrect ctx.M) (name : String) (args : List RVal) :
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

/-- The only float premise is discharged by the pre-existing IEEE ofNat interpretation.
The f32 bool-to-float contract remains provisional pending its independent approval. -/
theorem check_sound_ieee {src tgt : Module} (h : check src tgt = true)
    (base : FloatModel) (ctx : Ctx) (name : String) (args : List RVal) :
    policyDenote src {ctx with M := model base} name args =
      denote Sem.Ops.opSem {ctx with M := model base} tgt name args :=
  check_sound h _ (model_correct base) name args

#print axioms check_sound_ieee
end MerlinVerify.Rules.BoolToFloatProgram
