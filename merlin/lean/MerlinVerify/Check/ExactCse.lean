/- Exact common-subexpression elimination inside one block, as upstream `cse` does it in one
forward walk. A later operation with the same head as an earlier one, operands that hold the same
values, and regions that read the same outside values computes the earlier operation's value in
the same run, so its uses are renamed to the earlier result. Two kinds of fact justify it, both
learned from actual execution under the run's context: an earlier operation together with the
value it computes, and an alias (a removed result holds the same value as the one it was merged
into), so merges cascade without a fixed pass count. Nothing is assumed about an operation's
meaning beyond the model being a function of it. Only operations the DCE rule treats as
effect-free are merged. -/
import MerlinVerify.Check.CtxNormalize
import MerlinVerify.Check.TypedAlias
import MerlinVerify.Check.Synth

namespace MerlinVerify.Check.ExactCse

open MerlinVerify MerlinVerify.Sem.Ops

/-- What `a` computes in `env`, before its results are bound. -/
def compute (ctx : Ctx) (call : CallSem) (env : Env) (a : Op) : Outcome (List RVal) :=
  (env.getAll a.operands).bind fun args => opSem ctx call a args (evalRegions opSem call env a.regions)

theorem stepOp_compute (ctx : Ctx) (call : CallSem) (env : Env) (a : Op) :
    stepOp opSem ctx call env a = (compute ctx call env a).bind (env.bindAll a.results) := by
  simp only [stepOp, compute, Outcome.bind_assoc]

theorem mem_of_lookup : ∀ (ρ : Rho) (x y : ValueId), List.lookup x ρ = some y → (x, y) ∈ ρ
  | [], _, _, h => by simp at h
  | (a, b) :: ρ, x, y, h => by
    simp only [List.lookup] at h
    split at h
    · next he =>
      have hx : x = a := by simpa using he
      simp only [Option.some.injEq] at h
      subst hx h
      exact List.mem_cons_self
    · exact List.mem_cons_of_mem _ (mem_of_lookup ρ x y h)

/-- A correspondence every pair of which looks up equal values. -/
theorem corr_pairs {ρ : Rho} {envS envT : Env} (h : ∀ p ∈ ρ, envS.get? p.1 = envT.get? p.2) :
    Corr ρ envS envT := fun x y hf => h (x, y) (mem_of_lookup ρ x y hf)

def diagonal (ids : List ValueId) : Rho := ids.map fun x => (x, x)

theorem corr_diagonal {ids : List ValueId} {envS envT : Env}
    (h : ∀ x ∈ ids, envS.get? x = envT.get? x) : Corr (diagonal ids) envS envT :=
  corr_pairs fun p hp => by
    obtain ⟨x, hx, rfl⟩ := List.mem_map.1 hp
    exact h x hx

theorem getAll_map {env : Env} {f : ValueId → ValueId} : ∀ (ids : List ValueId),
    (∀ x ∈ ids, env.get? (f x) = env.get? x) → env.getAll (ids.map f) = env.getAll ids
  | [], _ => rfl
  | x :: xs, h => by
    have hx := h x List.mem_cons_self
    have ih := getAll_map xs (fun y hy => h y (List.mem_cons_of_mem _ hy))
    simp only [List.map_cons, Env.getAll, Outcome.mapM] at ih ⊢
    rw [hx, ih]

theorem getAll_congr {env env' : Env} (ids : List ValueId)
    (h : ∀ x ∈ ids, env'.get? x = env.get? x) : env'.getAll ids = env.getAll ids := by
  have := getAll_map (env := env') (f := id) ids (fun x _ => rfl)
  induction ids with
  | nil => rfl
  | cons x xs ih =>
    have ih' := ih (fun y hy => h y (List.mem_cons_of_mem _ hy)) (getAll_map (f := id) xs (fun _ _ => rfl))
    simp only [Env.getAll, Outcome.mapM] at ih' ⊢
    rw [h x List.mem_cons_self, ih']

inductive Fact where
  /-- `a` ran earlier; its regions read outside values only among `caps`. -/
  | op (a : Op) (caps : List ValueId)
  /-- `x` holds the same value as `y`. -/
  | same (x y : ValueId)

def Holds (ctx : Ctx) (call : CallSem) (env : Env) : Fact → Prop
  | .op a caps => ∃ r v, a.results = [r] ∧ env.get? r.id = some v ∧
      alphaRegions (diagonal caps) a.regions a.regions = true ∧ compute ctx call env a = .ok [v]
  | .same x y => env.get? x = env.get? y

def Valid (facts : List Fact) (ctx : Ctx) (call : CallSem) (env : Env) : Prop :=
  ∀ f ∈ facts, Holds ctx call env f

/-- The id an earlier merge made `x` an alias of, else `x`. -/
def resolve : List Fact → ValueId → ValueId
  | [], x => x
  | .same x' y :: facts, x => if x' == x then y else resolve facts x
  | .op _ _ :: facts, x => resolve facts x

theorem resolve_get {facts : List Fact} {ctx : Ctx} {call : CallSem} {env : Env}
    (hf : Valid facts ctx call env) (x : ValueId) : env.get? (resolve facts x) = env.get? x := by
  induction facts with
  | nil => rfl
  | cons f facts ih =>
    have rest : Valid facts ctx call env := fun g hg => hf g (List.mem_cons_of_mem _ hg)
    cases f with
    | op a caps => exact ih rest
    | same x' y =>
      simp only [resolve]
      split
      · next he =>
        have hx : x' = x := by simpa using he
        subst hx
        exact (hf _ List.mem_cons_self).symm
      · exact ih rest

mutual
  /-- Ids bound inside regions: block arguments and the results of nested operations. -/
  def boundRegions : List Region → List ValueId
    | [] => []
    | (.mk bs) :: rs => boundBlocks bs ++ boundRegions rs

  def boundBlocks : List Block → List ValueId
    | [] => []
    | (.mk args ops) :: bs => args.map (·.id) ++ boundOps ops ++ boundBlocks bs

  def boundOps : List Op → List ValueId
    | [] => []
    | (.mk _ results _ _ _ regions _) :: ops => results.map (·.id) ++ boundRegions regions ++ boundOps ops
end

/-- The outside ids an operation's regions read: uses not bound inside. Only a proposal; the
alpha check in `candidate` and `earlier` decides, and refuses a region that reads anything else. -/
def captures (a : Op) : List ValueId :=
  let bound := boundRegions a.regions
  (usesRegions a.regions).filter fun x => !bound.contains x

/-- An operation whose value may be reused: effect-free, one result that it does not read, and
regions that read outside values only among `captures`. -/
def candidate (a : Op) : Bool :=
  !a.isTerminator && Rules.Dce.erasable a &&
    (match a.results with
     | [r] => !a.operands.contains r.id && !(captures a).contains r.id
     | _ => false) &&
    alphaRegions (diagonal (captures a)) a.regions a.regions

/-- Region correspondence for matching `s` against an earlier operation: each id `s` reads maps
from the id it is an alias of. -/
def outside (facts : List Fact) (s : Op) : Rho := (captures s).map fun y => (resolve facts y, y)

theorem corr_outside {facts : List Fact} {ctx : Ctx} {call : CallSem} {env : Env}
    (hf : Valid facts ctx call env) (s : Op) : Corr (outside facts s) env env :=
  corr_pairs fun p hp => by
    obtain ⟨y, _, rfl⟩ := List.mem_map.1 hp
    exact resolve_get hf y

/-- The earlier operation `s` repeats, with its result, if any. -/
def earlier : List Fact → List Fact → Op → Option (Op × Value)
  | _, [], _ => none
  | all, .op a _ :: facts, s =>
    match a.results with
    | [ra] =>
      if opHeadEq a s && a.operands == s.operands.map (resolve all) &&
          alphaRegions (outside all s) a.regions s.regions then some (a, ra)
      else earlier all facts s
    | _ => earlier all facts s
  | all, .same _ _ :: facts, s => earlier all facts s

theorem earlier_spec : ∀ {all facts : List Fact} {s a : Op} {ra : Value},
    earlier all facts s = some (a, ra) →
    (∃ caps, Fact.op a caps ∈ facts) ∧ a.results = [ra] ∧ opHeadEq a s = true ∧
      a.operands = s.operands.map (resolve all) ∧ alphaRegions (outside all s) a.regions s.regions = true
  | _, [], _, _, _, h => by simp [earlier] at h
  | all, .op b caps :: facts, s, a, ra, h => by
    simp only [earlier] at h
    split at h
    · next rb hrb =>
      split at h
      · next hc =>
        simp only [Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl⟩ := h
        simp only [Bool.and_eq_true, beq_iff_eq] at hc
        exact ⟨⟨caps, List.mem_cons_self⟩, hrb, hc.1.1, hc.1.2, hc.2⟩
      · obtain ⟨⟨c, hc⟩, rest⟩ := earlier_spec h
        exact ⟨⟨c, List.mem_cons_of_mem _ hc⟩, rest⟩
    · obtain ⟨⟨c, hc⟩, rest⟩ := earlier_spec h
      exact ⟨⟨c, List.mem_cons_of_mem _ hc⟩, rest⟩
  | all, .same x y :: facts, s, a, ra, h => by
    simp only [earlier] at h
    obtain ⟨⟨c, hc⟩, rest⟩ := earlier_spec h
    exact ⟨⟨c, List.mem_cons_of_mem _ hc⟩, rest⟩

/-- A repeat computes exactly what the earlier operation computes in this run. -/
theorem repeat_value {facts : List Fact} {ctx : Ctx} {call : CallSem} {env : Env} {s a : Op}
    {ra : Value} (hf : Valid facts ctx call env) (he : earlier facts facts s = some (a, ra)) :
    a.results = [ra] ∧ ∃ v, env.get? ra.id = some v ∧ compute ctx call env s = .ok [v] := by
  obtain ⟨⟨caps, hmem⟩, hra, hhead, hops, hregs⟩ := earlier_spec he
  obtain ⟨ra', v, hra', hv, _, hcomp⟩ := hf _ hmem
  have : ra' = ra := by simpa [hra] using hra'.symm
  subst this
  refine ⟨hra, v, hv, ?_⟩
  rw [← hcomp]
  simp only [compute]
  rw [hops, getAll_map s.operands (fun x _ => resolve_get hf x),
    alphaRegions_sound call call _ a.regions s.regions env env hregs (corr_outside hf s) (fun _ _ => rfl)]
  simp only [opSem_blind (blind_of_opHeadEq hhead)]

/-- A fact survives a binding that touches none of its dependencies. -/
def untouched (written : List ValueId) : Fact → Bool
  | .op a caps => (a.operands ++ caps ++ a.results.map (·.id)).all fun x => !written.contains x
  | .same x y => !written.contains x && !written.contains y

theorem lookup_fresh {x : ValueId} {env : Env} {ids : List ValueId} {vals : List RVal}
    (h : ids.contains x = false) : Env.get? (ids.zip vals ++ env) x = env.get? x :=
  BindingFacts.lookup_zip_append ids vals (by simpa using h)

theorem holds_extend {ctx : Ctx} {call : CallSem} {env : Env} {f : Fact} (hh : Holds ctx call env f)
    {ids : List ValueId} {vals : List RVal} (hu : untouched ids f = true) :
    Holds ctx call (ids.zip vals ++ env) f := by
  cases f with
  | op a caps =>
    obtain ⟨r, v, hr, hv, hc, hcomp⟩ := hh
    simp only [untouched, List.all_eq_true, List.mem_append, Bool.not_eq_eq_eq_not,
      Bool.not_true] at hu
    refine ⟨r, v, hr, ?_, hc, ?_⟩
    · rw [lookup_fresh (hu r.id (Or.inr (by simp [hr])))]; exact hv
    · rw [← hcomp]
      simp only [compute]
      rw [getAll_congr a.operands (fun x hx => lookup_fresh (hu x (Or.inl (Or.inl hx)))),
        alphaRegions_sound call call _ a.regions a.regions _ env hc
          (corr_diagonal fun x hx => lookup_fresh (hu x (Or.inl (Or.inr hx)))) (fun _ _ => rfl)]
  | same x y =>
    simp only [untouched, Bool.and_eq_true, Bool.not_eq_eq_eq_not, Bool.not_true] at hu
    simp only [Holds] at hh ⊢
    rw [lookup_fresh hu.1, lookup_fresh hu.2]
    exact hh

theorem compute_ok {ctx : Ctx} {call : CallSem} {env env' : Env} {a : Op}
    (h : stepOp opSem ctx call env a = .ok env') :
    ∃ results, compute ctx call env a = .ok results ∧ env.bindAll a.results results = .ok env' := by
  rw [stepOp_compute] at h
  cases hc : compute ctx call env a with
  | ub => rw [hc] at h; cases h
  | stuck r => rw [hc] at h; cases h
  | ok results => rw [hc] at h; exact ⟨results, rfl, h⟩

/-- The new fact `s` contributes after it ran: an alias when it repeats an earlier operation, else
the operation itself when something in `rest` reads it. Like upstream `cse`, which erases an
operation that is unused when it reaches it, nothing is merged into an operation nothing reads. -/
def learn (facts : List Fact) (s : Op) (rest : List Op) : List Fact :=
  if candidate s then
    match s.results, earlier facts facts s with
    | [r], some (_, ra) => if r.id == ra.id then [] else [.same r.id ra.id]
    | [r], none => if (usesOps rest).contains r.id then [.op s (captures s)] else []
    | _, _ => []
  else []

def advance (_ib : Nat) (_Γ : TEnv) (facts : List Fact) (s : Op) (rest : List Op) : List Fact :=
  learn facts s rest ++ facts.filter (untouched (s.results.map (·.id)))

theorem advance_sound (ib : Nat) (Γ : TEnv) (facts : List Fact) (s : Op) (rest : List Op) (ctx : Ctx)
    (call : CallSem)
    (env env' : Env) (_ : EnvTyped ib Γ env) (hf : Valid facts ctx call env) (_ : ctx.indexBits = ib)
    (hs : stepOp opSem ctx call env s = .ok env') :
    Valid (advance ib Γ facts s rest) ctx call env' := by
  obtain ⟨vals, henv⟩ := BindingFacts.step_bindings hs
  have hkept : Valid (facts.filter (untouched (s.results.map (·.id)))) ctx call env' := by
    intro f hmem
    obtain ⟨hmem, hu⟩ := List.mem_filter.1 hmem
    rw [henv]
    exact holds_extend (hf f hmem) hu
  intro f hmem
  rcases List.mem_append.1 hmem with hnew | hold
  · unfold learn at hnew
    split at hnew
    · next hcand =>
      simp only [candidate, Bool.and_eq_true, Bool.not_eq_eq_eq_not, Bool.not_true] at hcand
      obtain ⟨⟨_, hres⟩, hclosed⟩ := hcand
      obtain ⟨results, hcomp, hbind⟩ := compute_ok hs
      split at hnew
      · next r a ra hr he =>
        split at hnew
        · cases hnew
        · next hne =>
          simp only [List.mem_singleton] at hnew
          subst hnew
          obtain ⟨_, v, hv, hcv⟩ := repeat_value hf he
          rw [hcomp] at hcv
          cases hcv
          rw [hr, bindAll_ok rfl] at hbind
          cases hbind
          have hne' : (ra.id == r.id) = false := beq_eq_false_iff_ne.2 fun h => hne (by simp [h])
          simp only [Holds, List.map_cons, List.map_nil, List.zip_cons_cons, List.zip_nil_left,
            List.cons_append, List.nil_append, Env.get?, List.lookup, beq_self_eq_true, hne']
          exact hv.symm
      · next r hr _ =>
        split at hnew
        · simp only [List.mem_singleton] at hnew
          subst hnew
          simp only [hr, Bool.and_eq_true, Bool.not_eq_eq_eq_not] at hres
          simp only [Env.bindAll, hr] at hbind
          split at hbind
          · next hlen =>
            obtain ⟨v, rfl⟩ := List.length_eq_one_iff.1 hlen.symm
            simp only [Outcome.ok.injEq] at hbind
            subst hbind
            refine ⟨r, v, hr, by simp [Env.get?], hclosed, ?_⟩
            rw [← hcomp]
            simp only [compute, List.map_cons, List.map_nil, List.zip_cons_cons, List.zip_nil_left,
              List.cons_append, List.nil_append]
            have fresh : ∀ x, (x == r.id) = true → x = r.id := fun x h => by simpa using h
            have hop : ∀ x ∈ s.operands, Env.get? ((r.id, v) :: env) x = env.get? x := by
              intro x hx
              have hb : (x == r.id) = false := by
                cases hxr : (x == r.id)
                · rfl
                · have := fresh x hxr; subst this; simp_all
              simp [Env.get?, List.lookup, hb]
            have hcap : ∀ x ∈ captures s, Env.get? ((r.id, v) :: env) x = env.get? x := by
              intro x hx
              have hb : (x == r.id) = false := by
                cases hxr : (x == r.id)
                · rfl
                · have := fresh x hxr; subst this; simp_all
              simp [Env.get?, List.lookup, hb]
            rw [getAll_congr s.operands hop,
              alphaRegions_sound call call _ s.regions s.regions _ env hclosed (corr_diagonal hcap)
                (fun _ _ => rfl)]
          · cases hbind
        · cases hnew
      · cases hnew
    · cases hnew
  · exact hkept f hold

def eliminate (Γ : TEnv) (facts : List Fact) (s : Op) (tail : List Op) : Option (List Op) := do
  if candidate s then
    let [r] := s.results | none
    let (_, ra) ← earlier facts facts s
    TypedAlias.rename? Γ r.id ra.id tail
  else none

theorem eliminate_sound {Γ : TEnv} {facts : List Fact} {s : Op} {tail renamed : List Op}
    (h : eliminate Γ facts s tail = some renamed) {ctx : Ctx} {call : CallSem} {env : Env}
    (hf : Valid facts ctx call env) :
    ∃ env', stepOp opSem ctx call env s = .ok env' ∧
      evalOps opSem ctx call env renamed = evalOps opSem ctx call env' tail := by
  unfold eliminate at h
  split at h
  · split at h
    · next r hr =>
      cases he : earlier facts facts s with
      | none => simp [he] at h
      | some p =>
        obtain ⟨a, ra⟩ := p
        simp only [he, bind, Option.bind] at h
        obtain ⟨_, v, hv, hcomp⟩ := repeat_value hf he
        have hstep : stepOp opSem ctx call env s = .ok ((r.id, v) :: env) := by
          rw [stepOp_compute, hcomp, Outcome.bind_ok, hr, bindAll_ok rfl]
          rfl
        exact ⟨_, hstep, TypedAlias.rename_sound h hv⟩
    · cases h
  · cases h

def finish (_ib : Nat) (Γ : TEnv) (facts : List Fact) (_hints : List Op) (s : Op) (tail : List Op) :
    List Op :=
  (eliminate Γ facts s tail).getD (s :: tail)

theorem finish_sound (ib : Nat) (Γ : TEnv) (facts : List Fact) (hints : List Op) (s : Op)
    (tail : List Op) (ctx : Ctx) (call : CallSem) (env : Env) (ht : s.isTerminator = false)
    (_ : EnvTyped ib Γ env) (hf : Valid facts ctx call env) (_ : ctx.indexBits = ib) :
    evalOps opSem ctx call env (finish ib Γ facts hints s tail) =
      evalOps opSem ctx call env (s :: tail) := by
  unfold finish
  cases he : eliminate Γ facts s tail with
  | none => rfl
  | some renamed =>
    obtain ⟨env', hs, heq⟩ := eliminate_sound he hf
    simp only [Option.getD_some]
    rw [evalOps_cons_step _ _ _ _ _ _ ht, hs, Outcome.bind_ok]
    exact heq

def family : CtxNormalize.Family (List Fact) Valid where
  initial := []
  initial_valid _ _ _ f hf := by cases hf
  advance := advance
  advance_sound := advance_sound
  finish := finish
  finish_sound := finish_sound

/-- One forward walk: aliases let a merge make later operations repeats in the same walk. -/
def base (ib : Nat) (hints : List Op) : SplatNormalizer.Normalizer ib :=
  CtxNormalize.base family ib hints

#print axioms advance_sound
#print axioms finish_sound

end MerlinVerify.Check.ExactCse
