/- A bounds-aware region walker. Inferred facts are recomputed from real SSA bindings;
terminators return proved facts, rather than discarding element intervals. -/
import MerlinVerify.Rules.FrontendTensorRanges

namespace MerlinVerify.Rules.FrontendRangeFrame
open MerlinVerify MerlinVerify.Check
open FrontendTensorRanges

/-- Root books need no enclosing loop frame. Nested books get the actual coordinates. -/
def Frame (ranges : List Nat) (ctx : Ctx) : Prop :=
  ranges = [] ∨ Idx.InBounds ranges ctx.loopIndex

structure BookAt (ib : Nat) (ranges : List Nat) where
  rule : FEnv → Op → List Fact → Option (List Fact)
  sound : ∀ Γ op fs rs, rule Γ op fs = some rs → ∀ ctx call env args,
    ctx.indexBits = ib → Frame ranges ctx → Definedness.Refined.EnvHolds model ib Γ env →
    Definedness.Refined.FactsHold model ib args fs →
    ∃ vs, Sem.Ops.opSem ctx call op args (evalRegions Sem.Ops.opSem call env op.regions) = .ok vs ∧
      Definedness.Refined.FactsHold model ib vs rs

def lift {ib : Nat} (old : Definedness.Refined.Book model Sem.Ops.opSem ib) (ranges : List Nat) : BookAt ib ranges where
  rule := old.rule
  sound Γ op fs rs h ctx call env args hib _ hΓ ha := old.sound Γ op fs rs h ctx call env args hib hΓ ha

def orElse {ib : Nat} {ranges : List Nat} (first next : BookAt ib ranges) : BookAt ib ranges where
  rule Γ op fs := (first.rule Γ op fs).orElse fun _ => next.rule Γ op fs
  sound Γ op fs rs h ctx call env args hib hf hΓ ha := by
    cases hp : first.rule Γ op fs with
    | none =>
      simp only [hp, Option.orElse_none] at h
      exact next.sound Γ op fs rs h ctx call env args hib hf hΓ ha
    | some ys =>
      simp only [hp, Option.orElse_some, Option.some.injEq] at h
      subst rs
      exact first.sound Γ op fs ys hp ctx call env args hib hf hΓ ha

/-- Inferred result facts describe the actual terminator operands, including all uses. -/
def inferOps {ib : Nat} {ranges : List Nat} (R : BookAt ib ranges) (Γ : FEnv) :
    List Op → Option (List Fact)
  | [] => none
  | op :: rest =>
    match op.operands.mapM (fun x => Γ.lookup x) with
    | none => none
    | some fs =>
      if op.isTerminator then
        if rest.isEmpty && fs.all (·.base.initialized) then some fs else none
      else if op.successors.isEmpty then
        match R.rule Γ op fs with
        | none => none
        | some rs => if rs.length == op.results.length then
            inferOps R ((op.results.map (·.id)).zip rs ++ Γ) rest
          else none
      else none

theorem inferOps_sound {ib : Nat} {ranges : List Nat} (R : BookAt ib ranges) :
    ∀ Γ ops rs ctx call env, inferOps R Γ ops = some rs → Definedness.Refined.EnvHolds model ib Γ env →
      ctx.indexBits = ib → Frame ranges ctx →
      ∃ vs, evalOps Sem.Ops.opSem ctx call env ops = .ok vs ∧ Definedness.Refined.FactsHold model ib vs rs
  | _, [], _, _, _, _, h, _, _, _ => by cases h
  | Γ, (.mk n results operands attrs props regions successors) :: rest, rs, ctx, call, env,
      h, hΓ, hib, hframe => by
    simp only [inferOps] at h
    split at h
    · cases h
    · next fs hfs =>
      obtain ⟨args, hargs, hfacts⟩ := hΓ.getAll operands fs hfs
      rw [evalOps.eq_2]
      split
      · next ht =>
        rw [ite_eq_left ht] at h
        split at h
        · next hc =>
          simp only [Bool.and_eq_true, List.isEmpty_iff] at hc
          obtain ⟨rfl, _⟩ := hc
          cases h
          simp only [List.isEmpty_nil, ↓reduceIte, hargs]
          exact ⟨args, rfl, hfacts⟩
        · cases h
      · next ht =>
        rw [ite_eq_right ht] at h
        split at h
        · next hempty =>
          split at h
          · cases h
          · next ys hrule =>
            split at h
            · next hlen =>
              obtain ⟨vs, hop, hv⟩ := R.sound Γ _ fs ys hrule ctx call env args hib hframe hΓ hfacts
              simp only [Op.regions] at hop
              simp only [hargs, Outcome.bind_ok, hop, Op.results]
              have hb : results.length = vs.length :=
                (beq_iff_eq.mp hlen).symm.trans (List.Forall₂.length_eq hv).symm
              rw [bindAll_ok hb, Outcome.bind_ok]
              exact inferOps_sound R _ rest rs ctx call _ h (hΓ.extend _ vs ys hv) hib hframe
            · cases h
        · cases h

def inferRegion {ib : Nat} {ranges : List Nat} (R : BookAt ib ranges) (Γ : FEnv)
    (region : Region) (args : List Fact) : Option (List Fact) :=
  match region with
  | .mk [.mk bargs ops] =>
    if bargs.mapM (fun a => vtyOf ib a.ty) = some (args.map (·.base.ty)) then
      inferOps R ((bargs.map (·.id)).zip args ++ Γ) ops
    else none
  | _ => none

theorem inferRegion_sound {ib : Nat} {ranges : List Nat} (R : BookAt ib ranges)
    {Γ : FEnv} {region : Region} {fs rs : List Fact}
    (h : inferRegion R Γ region fs = some rs) {ctx : Ctx} {call : CallSem} {env : Env}
    (hib : ctx.indexBits = ib) (hf : Frame ranges ctx) (hΓ : Definedness.Refined.EnvHolds model ib Γ env)
    {args : List RVal} (ha : Definedness.Refined.FactsHold model ib args fs) :
    ∃ vs, evalRegion Sem.Ops.opSem ctx call env region args = .ok vs ∧ Definedness.Refined.FactsHold model ib vs rs := by
  cases region with
  | mk blocks =>
    cases blocks with
    | nil => cases h
    | cons b rest =>
      cases rest with
      | cons _ _ => cases b; cases h
      | nil =>
        cases b with
        | mk params ops =>
          simp only [inferRegion] at h
          split at h
          · next ht =>
            have hlen : params.length = args.length := by
              rw [← optionMapM_length _ _ _ ht]
              simpa using (List.Forall₂.length_eq ha).symm
            simp only [evalRegion, evalBlocks, evalBlock, bindAll_ok hlen, Outcome.bind_ok]
            exact inferOps_sound R _ ops rs ctx call _ h (hΓ.extend _ args fs ha) hib hf
          · cases h

def root {ib : Nat} (R : BookAt ib []) : Definedness.Refined.Book model Sem.Ops.opSem ib where
  rule := R.rule
  sound Γ op fs rs h ctx call env args hib hΓ ha := R.sound Γ op fs rs h ctx call env args hib (.inl rfl) hΓ ha

#print axioms inferRegion_sound
end MerlinVerify.Rules.FrontendRangeFrame
