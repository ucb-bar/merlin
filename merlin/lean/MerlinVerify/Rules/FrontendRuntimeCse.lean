/- Exact closed-expression sharing under the same additive runtime interpretation.
No failing operation is discarded: both sides execute the first closed expression. -/
import MerlinVerify.Rules.FrontendBoolSimulationWith
import MerlinVerify.Rules.Cse

namespace MerlinVerify.Rules.FrontendRuntimeCse
open MerlinVerify MerlinVerify.Check FrontendBoolSimulation

/-- Unlike refinement-only CSE, this statement preserves UB and stuck reasons as
well as successful results. The operation interpreter must ignore SSA identifiers. -/
theorem site_sound (sem : OpSem) (laws : FrontendBoolSimulationWith.Laws sem)
    {ρ next : Rho} {a b t : Op} (h : Cse.check ρ [a,b] [t] = some next)
    (ctx : Ctx) (call : CallSem) {es et : Env} (hc : Corr ρ es et) :
    EnvRel next (execOps sem ctx call es [a,b]) (execOps sem ctx call et [t]) := by
  simp only [Cse.check] at h
  split at h
  · next ra rb rt ha hb ht =>
    split at h
    · next checks =>
      obtain ⟨hat, _, hab, hao, hbo, hto, har, hbr, htr, _⟩ := checks
      cases halpha : alphaOp ρ a t with
      | none => simp [halpha] at h
      | some ρ1 =>
        simp only [halpha, Option.map_some, Option.some.injEq] at h
        subst next
        obtain ⟨hhead, _, _, hfr, hρ⟩ := alphaOp_spec halpha
        have hbt : b.isTerminator = false := (isTerminator_of_opHeadEq hab) ▸ hat
        have htt : t.isTerminator = false := (isTerminator_of_opHeadEq hhead) ▸ hat
        have hbsem : sem ctx call b [] [] = sem ctx call a [] [] :=
          laws.blind (blind_of_opHeadEq hab).symm
        have htsem : sem ctx call t [] [] = sem ctx call a [] [] :=
          laws.blind (blind_of_opHeadEq hhead).symm
        rw [execOps_cons_step _ _ _ _ _ _ hat, Cse.closedStep hao har,
          execOps_cons_step _ _ _ _ _ _ htt, Cse.closedStep hto htr, htsem]
        cases he : sem ctx call a [] [] with
        | ub => trivial
        | stuck why => rfl
        | ok values =>
          cases values with
          | nil => simp [Outcome.bind_ok, Env.bindAll, ha, ht, EnvRel]
          | cons v values =>
            cases values with
            | cons w ws => simp [Outcome.bind_ok, Env.bindAll, ha, ht, EnvRel]
            | nil =>
              have hs : stepOp sem ctx call ((ra.id,v)::es) b = .ok ((rb.id,v)::(ra.id,v)::es) := by
                rw [Cse.closedStep hbo hbr, hbsem, he]
                simp [Env.bindAll, hb]
              have hc1 : Corr ρ1 ((ra.id,v)::es) ((rt.id,v)::et) := by
                obtain ⟨_, hfresh⟩ := freshIds_sound hfr
                rw [hρ]
                have hbind := Corr.extend hc [ra.id] [rt.id] [v] rfl rfl
                  (by simp) (by simpa [ht] using hfresh)
                simpa [ha, ht] using hbind
              simp only [Outcome.bind_ok]
              have hfirst : es.bindAll a.results [v] = .ok ((ra.id,v)::es) := by simp [Env.bindAll, ha]
              have htarget : et.bindAll t.results [v] = .ok ((rt.id,v)::et) := by simp [Env.bindAll, ht]
              rw [hfirst, htarget, Outcome.bind_ok, Outcome.bind_ok,
                execOps_cons_step _ _ _ _ _ _ hbt, hs]
              exact Cse.Corr.bindAlias hc1 (by simp [Env.get?, List.lookup])
    · cases h
  · cases h

/-- This first suffix family shares modelled arith.constant operations only.
Other expressions and unknown attributes refuse. All chunk facts come from the AST. -/
def shared (ρ : Rho) (a : Op) (rest : List Op) (t : Op) : Option (Rho × List Op) :=
  match rest with
  | b :: more =>
    if a.name == "arith.constant" && (Sem.Ops.unmodelledAttr? a.blind).isNone then
      (Cse.check ρ [a,b] [t]).map fun next => (next,more)
    else none
  | [] => none

theorem shared_spec {ρ next : Rho} {a t : Op} {rest more : List Op}
    (h : shared ρ a rest t = some (next,more)) :
    ∃ b, rest = b :: more ∧ Cse.check ρ [a,b] [t] = some next := by
  cases rest with
  | nil => cases h
  | cons b tail =>
    simp only [shared] at h
    split at h
    · cases hs : Cse.check ρ [a,b] [t] with
      | none => simp [hs] at h
      | some r =>
        simp only [hs, Option.map_some, Option.some.injEq, Prod.mk.injEq] at h
        rcases h with ⟨rfl,rfl⟩
        exact ⟨b,rfl,hs⟩
    · cases h

/-- The ordinary case checks every operation field and descends into regions. -/
def operation (regions : Rho → List Region → List Region → Bool) (ρ : Rho) (s t : Op) : Option Rho :=
  if opHeadEq s t && s.operands.mapM ρ.find == some t.operands && regions ρ s.regions t.regions &&
      freshIds ρ (t.results.map (·.id))
  then some ((s.results.map (·.id)).zip (t.results.map (·.id)) ++ ρ) else none

/- Fuel bounds checker work only. Exhaustion refuses; it does not bound program execution. -/
mutual
  def ops : Nat → Rho → List Op → List Op → Bool
    | 0, _, _, _ => false
    | _+1, _, [], [] => true
    | n+1, ρ, a::rest, t::tail =>
      match shared ρ a rest t with
      | some (next,more) => ops n next more tail
      | none => match operation (regions n) ρ a t with
        | some next => ops n next rest tail
        | none => false
    | _+1, _, _, _ => false
  def regions : Nat → Rho → List Region → List Region → Bool
    | 0, _, _, _ => false
    | _+1, _, [], [] => true
    | n+1, ρ, r::rs, t::ts => region n ρ r t && regions n ρ rs ts
    | _+1, _, _, _ => false
  def region : Nat → Rho → Region → Region → Bool
    | 0, _, _, _ => false
    | n+1, ρ, .mk [.mk argsS ss], .mk [.mk argsT ts] =>
      argsS.map (·.ty) == argsT.map (·.ty) && freshIds ρ (argsT.map (·.id)) &&
        ops n ((argsS.map (·.id)).zip (argsT.map (·.id)) ++ ρ) ss ts
    | _+1, _, .mk [], .mk [] => true
    | _+1, _, _, _ => false
end

theorem operation_spec {R : Rho → List Region → List Region → Bool} {ρ next : Rho} {s t : Op}
    (h : operation R ρ s t = some next) :
    opHeadEq s t = true ∧ s.operands.mapM ρ.find = some t.operands ∧
      R ρ s.regions t.regions = true ∧ freshIds ρ (t.results.map (·.id)) = true ∧
      next = (s.results.map (·.id)).zip (t.results.map (·.id)) ++ ρ := by
  unfold operation at h
  split at h
  · next hc =>
    simp only [Option.some.injEq] at h
    simp only [Bool.and_eq_true, beq_iff_eq] at hc
    exact ⟨hc.1.1.1,hc.1.1.2,hc.1.2,hc.2,h.symm⟩
  · cases h

theorem ops_isEmpty {n : Nat} {ρ : Rho} {ss ts : List Op} (h : ops n ρ ss ts = true) :
    ss.isEmpty = ts.isEmpty := by
  cases n with
  | zero => cases h
  | succ n => cases ss <;> cases ts <;> simp_all [ops]

mutual
  theorem ops_sound (sem : OpSem) (laws : FrontendBoolSimulationWith.Laws sem) (call : CallSem) :
      ∀ n ρ ss ts ctx es et, ops n ρ ss ts = true → Corr ρ es et →
        evalOps sem ctx call es ss = evalOps sem ctx call et ts
    | 0, _, _, _, _, _, _, h, _ => by cases h
    | _+1, _, [], [], _, _, _, _, _ => rfl
    | n+1, ρ, a::rest, t::tail, ctx, es, et, h, hc => by
      simp only [ops] at h
      cases hs : shared ρ a rest t with
      | some pair =>
        obtain ⟨next,more⟩ := pair
        simp only [hs] at h
        obtain ⟨b,rfl,hsite⟩ := shared_spec hs
        have hn := (Cse.check_sound ρ [a,b] [t] next hsite)
        change evalOps sem ctx call es ([a,b] ++ more) = evalOps sem ctx call et ([t] ++ tail)
        rw [evalOps_append _ _ _ _ _ _ hn.1, evalOps_append _ _ _ _ _ _ hn.2.1]
        exact envRel_bind (site_sound sem laws hsite ctx call hc) _ _
          (fun es' et' hc' => ops_sound sem laws call n next more tail ctx es' et' h hc')
      | none =>
        simp only [hs] at h
        cases ho : operation (regions n) ρ a t with
        | none => simp [ho] at h
        | some next =>
          simp only [ho] at h
          obtain ⟨hhead,hmap,hregions,hfresh,rfl⟩ := operation_spec ho
          have hterm := isTerminator_of_opHeadEq hhead
          have hblind := blind_of_opHeadEq hhead
          have hlen := results_length_of_opHeadEq hhead
          have hread := Corr.getAll hc _ _ hmap
          cases a with
          | mk na sa oa aa pa ra ua =>
            cases t with
            | mk nt st ot att pt rt ut =>
              dsimp only [Op.operands] at hread
              rw [evalOps.eq_2, evalOps.eq_2]
              by_cases ht : (Op.mk nt st ot att pt rt ut).isTerminator = true
              · rw [ite_eq_left (hterm.trans ht), ite_eq_left ht, ops_isEmpty h, hread]
              · rw [ite_eq_right (by rwa [hterm]), ite_eq_right ht, hread]
                cases et.getAll ot with
                | ub => rfl
                | stuck why => rfl
                | ok args =>
                  simp only [Outcome.bind_ok]
                  have hr := regions_sound sem laws call n ρ ra rt ctx es et hregions hc
                  rw [laws.regions ctx call _ args _ _ hr, laws.blind hblind]
                  cases sem ctx call (.mk nt st ot att pt rt ut) args (evalRegions sem call et rt) with
                  | ub => rfl
                  | stuck why => rfl
                  | ok results =>
                    simp only [Outcome.bind_ok]
                    exact envRel_bind (bind_corr hc _ _ results hlen hfresh) _ _
                      (fun es' et' hc' => ops_sound sem laws call n _ rest tail ctx es' et' h hc')
    | _+1, _, [], _::_, _, _, _, h, _ => by simp [ops] at h
    | _+1, _, _::_, [], _, _, _, h, _ => by simp [ops] at h

  theorem regions_sound (sem : OpSem) (laws : FrontendBoolSimulationWith.Laws sem) (call : CallSem) :
      ∀ n ρ rs ts ctx es et, regions n ρ rs ts = true → Corr ρ es et →
        (evalRegions sem call es rs).map (FrontendBoolContext.pin ctx) =
          (evalRegions sem call et ts).map (FrontendBoolContext.pin ctx)
    | 0, _, _, _, _, _, _, h, _ => by cases h
    | _+1, _, [], [], _, _, _, _, _ => rfl
    | n+1, ρ, r::rs, t::ts, ctx, es, et, h, hc => by
      simp only [regions, Bool.and_eq_true] at h
      simp only [evalRegions, List.map_cons, List.cons.injEq]
      refine ⟨?_, regions_sound sem laws call n ρ rs ts ctx es et h.2 hc⟩
      funext inner args
      exact congrFun (region_sound sem laws call n ρ r t
        {ctx with loopIndex := inner.loopIndex} es et h.1 hc) args
    | _+1, _, [], _::_, _, _, _, h, _ => by simp [regions] at h
    | _+1, _, _::_, [], _, _, _, h, _ => by simp [regions] at h

  theorem region_sound (sem : OpSem) (laws : FrontendBoolSimulationWith.Laws sem) (call : CallSem) :
      ∀ n ρ s t ctx es et, region n ρ s t = true → Corr ρ es et →
        evalRegion sem ctx call es s = evalRegion sem ctx call et t
    | 0, _, _, _, _, _, _, h, _ => by cases h
    | n+1, ρ, .mk [.mk argsS ss], .mk [.mk argsT ts], ctx, es, et, h, hc => by
      simp only [region, Bool.and_eq_true, beq_iff_eq] at h
      obtain ⟨⟨hty,hfresh⟩,hops⟩ := h
      have hl : argsS.length = argsT.length := by simpa using congrArg List.length hty
      simp only [evalRegion, evalBlocks, evalBlock]
      funext vals
      exact envRel_bind (bind_corr hc argsS argsT vals hl hfresh) _ _
        (fun es' et' hc' => ops_sound sem laws call n _ ss ts ctx es' et' hops hc')
    | _+1, _, .mk [], .mk [], _, _, _, _, _ => rfl
    | _+1, _, .mk [], .mk ((.mk _ _)::_::_ ), _, _, _, h, _ => by cases h
    | _+1, _, .mk ((.mk _ _)::_::_ ), .mk [], _, _, _, h, _ => by cases h
    | _+1, _, .mk ((.mk _ _)::_::_ ), .mk ((.mk _ _)::_::_ ), _, _, _, h, _ => by cases h
    | _+1, _, .mk [], .mk [.mk _ _], _, _, _, h, _ => by cases h
    | _+1, _, .mk [.mk _ _], .mk [], _, _, _, h, _ => by cases h
    | _+1, _, .mk [.mk _ _], .mk ((.mk _ _)::_::_ ), _, _, _, h, _ => by cases h
    | _+1, _, .mk ((.mk _ _)::_::_ ), .mk [.mk _ _], _, _, _, h, _ => by cases h
end

def function (fuel : Nat) (s t : Func) : Bool :=
  s.name == t.name && s.isDecl == t.isDecl && region fuel [] s.body t.body

def functions (fuel : Nat) : List Func → List Func → Bool
  | [], [] => true
  | s::ss,t::ts => function fuel s t && functions fuel ss ts
  | _,_ => false

theorem function_spec {fuel : Nat} {s t : Func} (h : function fuel s t = true) :
    s.name = t.name ∧ s.isDecl = t.isDecl ∧ region fuel [] s.body t.body = true := by
  simpa only [function, Bool.and_eq_true, beq_iff_eq, and_assoc] using h

def Found (fuel : Nat) : Option Func → Option Func → Prop
  | none,none => True
  | some s,some t => function fuel s t = true
  | _,_ => False

theorem functions_find {fuel : Nat} : ∀ ss ts, functions fuel ss ts = true → ∀ name,
    Found fuel (ss.find? (·.name == name)) (ts.find? (·.name == name))
  | [],[],_,_ => trivial
  | s::ss,t::ts,h,name => by
    simp only [functions, Bool.and_eq_true] at h
    have hn := (function_spec h.1).1
    simp only [List.find?_cons,hn]
    split
    · exact h.1
    · exact functions_find ss ts h.2 name
  | [],_::_,h,_ => by simp [functions] at h
  | _::_,[],h,_ => by simp [functions] at h

theorem functions_length {fuel : Nat} : ∀ ss ts, functions fuel ss ts = true → ss.length = ts.length
  | [],[],_ => rfl
  | _::ss,_::ts,h => by
    simp only [functions, Bool.and_eq_true] at h
    simpa using functions_length ss ts h.2
  | [],_::_,h => by simp [functions] at h
  | _::_,[],h => by simp [functions] at h

theorem evalFunc_sound (sem : OpSem) (laws : FrontendBoolSimulationWith.Laws sem)
    {budget : Nat} {src tgt : Module} (h : functions budget src.funcs tgt.funcs = true) (ctx : Ctx) :
    ∀ fuel, evalFunc sem ctx src fuel = evalFunc sem ctx tgt fuel := by
  intro fuel
  induction fuel with
  | zero => rfl
  | succ fuel ih =>
    funext name args
    have hf := functions_find src.funcs tgt.funcs h name
    change Found budget (src.func? name) (tgt.func? name) at hf
    cases hs : src.func? name with
    | none =>
      cases ht : tgt.func? name with
      | none => simp [evalFunc,hs,ht]
      | some t => simp [hs,ht,Found] at hf
    | some s =>
      cases ht : tgt.func? name with
      | none => simp [hs,ht,Found] at hf
      | some t =>
        simp only [hs,ht,Found] at hf
        obtain ⟨_,hd,hr⟩ := function_spec hf
        simp only [evalFunc,hs,ht,hd]
        split
        · rfl
        · rw [ih]
          exact congrFun (region_sound sem laws (evalFunc sem ctx tgt fuel) budget []
            s.body t.body ctx [] [] hr (by intro x y hxy; simp [Rho.find] at hxy)) args

/-- Module metadata and symbol structure are independently retained. The function
checker then accounts for every changed site and every remaining operation/use. -/
def check (src tgt : Module) : Bool :=
  (outsideChanges src tgt).unverified.isEmpty && functions 262144 src.funcs tgt.funcs

theorem check_sound {src tgt : Module} (h : check src tgt = true)
    (limit : Nat) (ctx : Ctx) (name : String) (args : List RVal) :
    denote (FrontendMaskRuntime.sem limit) ctx src name args =
      denote (FrontendMaskRuntime.sem limit) ctx tgt name args := by
  simp only [check, Bool.and_eq_true] at h
  have he := evalFunc_sound _ (FrontendBoolSimulationWith.candidateLaws limit) h.2 ctx
  have hl := functions_length src.funcs tgt.funcs h.2
  simp only [denote]
  rw [he,hl]

#print axioms site_sound
#print axioms check_sound
end MerlinVerify.Rules.FrontendRuntimeCse
