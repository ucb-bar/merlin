/- Context-aware definedness for loop bodies, without changing the frozen facts or evaluator. -/
import MerlinVerify.Rules.FrontendBoolDefined
import MerlinVerify.Rules.FrontendRuntimeBridge

namespace MerlinVerify.Rules.FrontendDefinedFrame
open MerlinVerify MerlinVerify.Check MerlinVerify.Check.Definedness MerlinVerify.Sem.Ops
variable {sem : OpSem} {ib depth : Nat}

/-- The frame states only that linalg.index coordinates exist. It supplies no comparison law,
value bound, initialization fact or certificate. -/
structure BookAt (sem : OpSem) (ib depth : Nat) where
  rule : FEnv → Op → List Fact → Option (List Fact)
  sound : ∀ Γ op fs rs, rule Γ op fs = some rs → ∀ ctx call env args,
    ctx.indexBits = ib → depth ≤ ctx.loopIndex.length → EnvHolds ib Γ env → FactsHold ib args fs →
    ∃ vs, sem ctx call op args (evalRegions sem call env op.regions) = .ok vs ∧ FactsHold ib vs rs

def lift (old : Book sem ib) (depth : Nat) : BookAt sem ib depth where
  rule := old.rule
  sound Γ op fs rs h ctx call env args hib _ hΓ ha := old.sound Γ op fs rs h ctx call env args hib hΓ ha

def orElse (first next : BookAt sem ib depth) : BookAt sem ib depth where
  rule Γ op fs := (first.rule Γ op fs).orElse fun _ => next.rule Γ op fs
  sound Γ op fs rs h ctx call env args hib hd hΓ ha := by
    cases hf : first.rule Γ op fs with
    | some ys =>
      simp only [hf, Option.orElse_some, Option.some.injEq] at h
      subst h
      exact first.sound Γ op fs ys hf ctx call env args hib hd hΓ ha
    | none =>
      simp only [hf, Option.orElse_none] at h
      exact next.sound Γ op fs rs h ctx call env args hib hd hΓ ha

def checkOps {sem : OpSem} {ib depth : Nat} (R : BookAt sem ib depth) (Γ : FEnv) : List Op → List VTy → Bool
  | [], _ => false
  | op :: rest, yields =>
    match op.operands.mapM (fun x => Γ.lookup x) with
    | none => false
    | some fs =>
      if op.isTerminator then
        rest.isEmpty && fs.all (·.initialized) && fs.map (·.ty) == yields
      else if op.successors.isEmpty then
        match R.rule Γ op fs with
        | none => false
        | some rs => rs.length == op.results.length &&
          checkOps R ((op.results.map (·.id)).zip rs ++ Γ) rest yields
      else false

theorem checkOps_sound {sem : OpSem} {ib depth : Nat} (R : BookAt sem ib depth) :
    ∀ Γ ops yields ctx call env, checkOps R Γ ops yields = true → EnvHolds ib Γ env →
      ctx.indexBits = ib → depth ≤ ctx.loopIndex.length → ∃ vs, evalOps sem ctx call env ops = .ok vs ∧
        HasTys ib vs yields ∧ ResultsDefined vs
  | _, [], _, _, _, _, h, _, _, _ => by cases h
  | Γ, (.mk n results operands attrs props regions successors) :: rest, yields, ctx, call, env,
      h, hΓ, hib, hdepth => by
    simp only [checkOps] at h
    split at h
    · cases h
    · next fs hfs =>
      obtain ⟨args, hargs, hfacts⟩ := hΓ.getAll operands fs hfs
      rw [evalOps.eq_2]
      split
      · next ht =>
        rw [ite_eq_left ht] at h
        simp only [Bool.and_eq_true, List.isEmpty_iff, beq_iff_eq] at h
        obtain ⟨⟨rfl, hi⟩, hty⟩ := h
        simp only [List.isEmpty_nil, ↓reduceIte, hargs]
        exact ⟨args, rfl, hty ▸ hfacts.typed, hfacts.defined (List.all_eq_true.1 hi)⟩
      · next ht =>
        rw [ite_eq_right ht] at h
        split at h
        · next hempty =>
          split at h
          · cases h
          · next rs hrule =>
            simp only [Bool.and_eq_true, beq_iff_eq] at h
            obtain ⟨hlen, hrest⟩ := h
            obtain ⟨vs, hop, hv⟩ := R.sound Γ _ fs rs hrule ctx call env args hib hdepth hΓ hfacts
            simp only [Op.regions] at hop
            simp only [hargs, Outcome.bind_ok, hop, Op.results]
            have hb : results.length = vs.length := hlen.symm.trans (List.Forall₂.length_eq hv).symm
            rw [bindAll_ok hb, Outcome.bind_ok]
            exact checkOps_sound R _ rest yields ctx call _ hrest (hΓ.extend _ vs rs hv) hib hdepth
        · cases h


def checkRegion (R : BookAt sem ib depth) (Γ : FEnv) (region : Region)
    (args : List Fact) (yields : List VTy) : Bool :=
  match region with
  | .mk [.mk bargs ops] => bargs.mapM (fun a => vtyOf ib a.ty) == some (args.map (·.ty)) &&
    checkOps R ((bargs.map (·.id)).zip args ++ Γ) ops yields
  | _ => false

theorem checkRegion_sound (R : BookAt sem ib depth) {Γ : FEnv} {region : Region}
    {argFacts : List Fact} {yields : List VTy} (h : checkRegion R Γ region argFacts yields = true)
    {ctx : Ctx} {call : CallSem} {env : Env} (hib : ctx.indexBits = ib)
    (hd : depth ≤ ctx.loopIndex.length) (hΓ : EnvHolds ib Γ env) {args : List RVal}
    (ha : FactsHold ib args argFacts) : ∃ vs, evalRegion sem ctx call env region args = .ok vs ∧
      HasTys ib vs yields ∧ ResultsDefined vs := by
  cases region with
  | mk blocks =>
    cases blocks with
    | nil => simp [checkRegion] at h
    | cons b rest =>
      cases rest with
      | cons _ _ => simp [checkRegion] at h
      | nil =>
        cases b with
        | mk params body =>
          simp only [checkRegion, Bool.and_eq_true, beq_iff_eq] at h
          have hlen : params.length = args.length := by
            rw [← optionMapM_length _ _ _ h.1]
            simpa using (List.Forall₂.length_eq ha).symm
          simp only [evalRegion, evalBlocks, evalBlock, bindAll_ok hlen, Outcome.bind_ok]
          exact checkOps_sound R _ body yields ctx call _ h.2 (hΓ.extend _ args argFacts ha) hib hd

/-- A depth-zero book plugs into the existing whole-function definedness checker unchanged. -/
def root (book : BookAt sem ib 0) : Book sem ib where
  rule := book.rule
  sound Γ op fs rs h ctx call env args hib hΓ ha :=
    book.sound Γ op fs rs h ctx call env args hib (Nat.zero_le _) hΓ ha

end MerlinVerify.Rules.FrontendDefinedFrame
