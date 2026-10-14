/-
Progress: a statically checked program never gets `stuck` on well-typed inputs (design §4.5).

The checker is generic in a per-op typing `rule`: given an op and its operand types, the rule
either refuses (`none`) or names the signature each of the op's regions must have. A rule is sound
for an op semantics if, whenever it accepts, the op is *safe*: on operands of the given types and
regions meeting their signatures, the op is not stuck and its results have the declared types.

`checkOps_progress`: if `checkOps` accepts a block under a typing environment that describes the
runtime environment, the block's evaluation is not stuck, and what it yields has the expected
types. The concrete rule for Merlin's op semantics, and its soundness, live in `Check/Rules*.lean`.
-/
import MerlinVerify.Check.Typing
import MerlinVerify.Check.ProgramLemmas

namespace MerlinVerify.Check

open MerlinVerify

/-- What a region must satisfy: its block argument types and the types its terminator yields. -/
structure RegionSig where
  args : List VTy
  yields : List VTy

/-- A region meaning is safe for a signature: on typed arguments it is never stuck, and whatever it
yields has the signature's types. Quantified over every context at index width `ib`. -/
def RegionSafe (ib : Nat) (r : Ctx → RegionSem) (sig : RegionSig) : Prop :=
  ∀ ctx, ctx.indexBits = ib → ∀ vals, HasTys ib vals sig.args →
    (r ctx vals).NotStuck ∧ ∀ ys, r ctx vals = .ok ys → HasTys ib ys sig.yields

/-- An op is safe at the given operand/result types and region signatures. -/
def OpSafe (sem : OpSem) (ib : Nat) (op : Op) (argTys resTys : List VTy) (sigs : List RegionSig) :
    Prop :=
  ∀ ctx call args regions, ctx.indexBits = ib → HasTys ib args argTys →
    List.Forall₂ (RegionSafe ib) regions sigs →
    (sem ctx call op args regions).NotStuck ∧
      ∀ rs, sem ctx call op args regions = .ok rs → HasTys ib rs resTys

/-- A typing rule and its soundness for an op semantics. -/
structure Rulebook (sem : OpSem) (ib : Nat) where
  rule : Op → List VTy → List VTy → Option (List RegionSig)
  sound : ∀ op argTys resTys sigs, rule op argTys resTys = some sigs → OpSafe sem ib op argTys resTys sigs

variable {sem : OpSem} {ib : Nat}

mutual
  /-- Check a block's ops under `Γ`; its terminator must yield exactly `yields`. -/
  def checkOps (R : Rulebook sem ib) (Γ : TEnv) : List Op → List VTy → Bool
    | [], _ => false
    | (op@(.mk _ results operands _ _ regions _)) :: rest, yields =>
      match operands.mapM (fun x => List.lookup x Γ) with
      | none => false
      | some argTys =>
        if op.isTerminator then rest.isEmpty && argTys == yields
        else
          match results.mapM (fun (r : Value) => vtyOf ib r.ty) with
          | none => false
          | some resTys =>
            match R.rule op argTys resTys with
            | none => false
            | some sigs =>
              checkRegions R Γ regions sigs &&
                checkOps R ((results.map (fun (r : Value) => r.id)).zip resTys ++ Γ) rest yields

  def checkRegions (R : Rulebook sem ib) (Γ : TEnv) : List Region → List RegionSig → Bool
    | [], [] => true
    | r :: rs, s :: ss => checkRegion R Γ r s && checkRegions R Γ rs ss
    | _, _ => false

  def checkRegion (R : Rulebook sem ib) (Γ : TEnv) : Region → RegionSig → Bool
    | .mk [.mk args ops], sig =>
      args.mapM (fun (a : Value) => vtyOf ib a.ty) == some sig.args &&
        checkOps R ((args.map (fun (a : Value) => a.id)).zip sig.args ++ Γ) ops sig.yields
    | .mk _, _ => false
end

theorem HasTys.length {vs : List RVal} {ts : List VTy} (h : HasTys ib vs ts) : vs.length = ts.length :=
  List.Forall₂.length_eq h

/-- Reading typed operands from a typed environment succeeds, with values of those types. -/
theorem EnvTyped.getAll {Γ : TEnv} {env : Env} (h : EnvTyped ib Γ env) :
    ∀ (xs : List ValueId) (ts : List VTy), xs.mapM (fun x => List.lookup x Γ) = some ts →
      ∃ vs, env.getAll xs = .ok vs ∧ HasTys ib vs ts
  | [], ts, hm => by
    simp only [List.mapM_nil, Option.pure_def, Option.some.injEq] at hm
    subst hm
    exact ⟨[], rfl, List.Forall₂.nil⟩
  | x :: xs, ts, hm => by
    simp only [List.mapM_cons] at hm
    cases hx : List.lookup x Γ with
    | none => simp [hx] at hm
    | some t =>
      cases hxs : xs.mapM (fun x => List.lookup x Γ) with
      | none => simp [hx, hxs] at hm
      | some ts' =>
        simp only [hx, hxs, Option.pure_def, bind, Option.bind, Option.some.injEq] at hm
        subst hm
        obtain ⟨v, hv, hvt⟩ := h x t hx
        obtain ⟨vs, hvs, hvst⟩ := EnvTyped.getAll h xs ts' hxs
        refine ⟨v :: vs, ?_, List.Forall₂.cons hvt hvst⟩
        have hvs' : Outcome.mapM (fun v => Outcome.ofOption "unbound SSA value" (env.get? v)) xs =
            .ok vs := hvs
        show Outcome.mapM (fun v => Outcome.ofOption "unbound SSA value" (env.get? v)) (x :: xs) = _
        rw [Outcome.mapM.eq_2, hvs']
        simp only [hv, Outcome.ofOption, Outcome.bind_ok]

/-- Binding values of types `ts` to ids extends a typed environment by those ids' types. -/
theorem EnvTyped.extend {Γ : TEnv} {env : Env} (h : EnvTyped ib Γ env) :
    ∀ (ids : List ValueId) (vs : List RVal) (ts : List VTy), HasTys ib vs ts →
      EnvTyped ib (ids.zip ts ++ Γ) (ids.zip vs ++ env)
  | [], _, _, _ => by simpa using h
  | _ :: _, [], [], _ => by simpa using h
  | x :: xs, v :: vs, t :: ts, hv => by
    obtain _ | ⟨hvt, hrest⟩ := hv
    have ih := EnvTyped.extend h xs vs ts hrest
    intro y u hy
    simp only [List.zip_cons_cons, List.cons_append, List.lookup] at hy
    simp only [List.zip_cons_cons, List.cons_append, Env.get?, List.lookup]
    by_cases hyx : y = x
    · subst hyx
      simp only [beq_self_eq_true, ↓reduceIte, Option.some.injEq] at hy ⊢
      subst hy
      exact ⟨v, rfl, hvt⟩
    · have hb : (y == x) = false := by simpa using hyx
      simp only [hb, Bool.false_eq_true, ↓reduceIte] at hy ⊢
      exact ih y u hy
  | _ :: _, [], _ :: _, hv => by cases hv
  | _ :: _, _ :: _, [], hv => by cases hv

theorem optionMapM_length {α β : Type} (f : α → Option β) :
    ∀ (l : List α) (r : List β), l.mapM f = some r → r.length = l.length
  | [], r, h => by simp only [List.mapM_nil, Option.pure_def, Option.some.injEq] at h; subst h; rfl
  | a :: l, r, h => by
    simp only [List.mapM_cons] at h
    cases hfa : f a with
    | none => simp [hfa] at h
    | some b =>
      cases hl : l.mapM f with
      | none => simp [hfa, hl] at h
      | some bs =>
        simp only [hfa, hl, Option.pure_def, bind, Option.bind, Option.some.injEq] at h
        subst h
        simp [optionMapM_length f l bs hl]

theorem bindAll_ok {env : Env} {vs : List Value} {rs : List RVal} (h : vs.length = rs.length) :
    env.bindAll vs rs = .ok ((vs.map (·.id)).zip rs ++ env) := by
  simp [Env.bindAll, h]

mutual
  theorem checkOps_progress (R : Rulebook sem ib) :
      ∀ (Γ : TEnv) (ops : List Op) (yields : List VTy) (ctx : Ctx) (call : CallSem) (env : Env),
        checkOps R Γ ops yields = true → EnvTyped ib Γ env → ctx.indexBits = ib →
        (evalOps sem ctx call env ops).NotStuck ∧
          ∀ rs, evalOps sem ctx call env ops = .ok rs → HasTys ib rs yields
    | _, [], _, _, _, _, h, _, _ => by simp [checkOps] at h
    | Γ, (.mk n results operands as ps regions succ) :: rest, yields, ctx, call, env, h, hΓ, hib => by
      simp only [checkOps] at h
      split at h
      · simp at h
      · next argTys hargs =>
        obtain ⟨vs, hvs, hvt⟩ := hΓ.getAll operands argTys hargs
        rw [evalOps.eq_2]
        split
        · next ht =>
          rw [if_pos ht] at h
          simp only [Bool.and_eq_true, List.isEmpty_iff, beq_iff_eq] at h
          obtain ⟨rfl, rfl⟩ := h
          simp only [List.isEmpty_nil, ↓reduceIte, hvs]
          exact ⟨trivial, fun rs hrs => by cases hrs; exact hvt⟩
        · next ht =>
          rw [if_neg ht] at h
          split at h
          · simp at h
          · next resTys hres =>
            split at h
            · simp at h
            · next sigs hrule =>
              simp only [Bool.and_eq_true] at h
              obtain ⟨hregs, hrest⟩ := h
              have hR := checkRegions_safe R Γ regions sigs call env hregs hΓ
              have hsafe := R.sound _ argTys resTys sigs hrule ctx call vs
                (evalRegions sem call env regions) hib hvt hR
              rw [hvs, Outcome.bind_ok]
              cases hop : sem ctx call (Op.mk n results operands as ps regions succ) vs
                  (evalRegions sem call env regions) with
              | ub => exact ⟨trivial, fun rs hrs => by cases hrs⟩
              | stuck r => exact absurd (hop ▸ hsafe.1) (by simp [Outcome.NotStuck])
              | ok rs =>
                have hrs := hsafe.2 rs hop
                have hlen : results.length = rs.length := by
                  rw [← optionMapM_length _ _ _ hres]; exact (HasTys.length hrs).symm
                simp only [Outcome.bind_ok, Op.results]
                rw [bindAll_ok hlen, Outcome.bind_ok]
                exact checkOps_progress R _ rest yields ctx call _ hrest
                  (hΓ.extend _ rs resTys hrs) hib

  theorem checkRegions_safe (R : Rulebook sem ib) :
      ∀ (Γ : TEnv) (rs : List Region) (sigs : List RegionSig) (call : CallSem) (env : Env),
        checkRegions R Γ rs sigs = true → EnvTyped ib Γ env →
        List.Forall₂ (RegionSafe ib) (evalRegions sem call env rs) sigs
    | _, [], [], _, _, _, _ => List.Forall₂.nil
    | Γ, r :: rs, s :: ss, call, env, h, hΓ => by
      simp only [checkRegions, Bool.and_eq_true] at h
      simp only [evalRegions]
      exact List.Forall₂.cons (checkRegion_safe R Γ r s call env h.1 hΓ)
        (checkRegions_safe R Γ rs ss call env h.2 hΓ)
    | _, [], _ :: _, _, _, h, _ => by simp [checkRegions] at h
    | _, _ :: _, [], _, _, h, _ => by simp [checkRegions] at h

  theorem checkRegion_safe (R : Rulebook sem ib) :
      ∀ (Γ : TEnv) (r : Region) (sig : RegionSig) (call : CallSem) (env : Env),
        checkRegion R Γ r sig = true → EnvTyped ib Γ env →
        RegionSafe ib (fun ctx => evalRegion sem ctx call env r) sig
    | Γ, .mk [.mk args ops], sig, call, env, h, hΓ => by
      simp only [checkRegion, Bool.and_eq_true, beq_iff_eq] at h
      obtain ⟨hargs, hops⟩ := h
      intro ctx hib vals hvals
      have hlen : args.length = vals.length := by
        rw [← optionMapM_length _ _ _ hargs]; exact (HasTys.length hvals).symm
      simp only [evalRegion, evalBlocks, evalBlock]
      rw [bindAll_ok hlen, Outcome.bind_ok]
      exact checkOps_progress R _ ops sig.yields ctx call _ hops (hΓ.extend _ vals sig.args hvals) hib
    | _, .mk [], _, _, _, h, _ => by cases h
    | _, .mk ((.mk _ _) :: _ :: _), _, _, _, h, _ => by cases h
end

end MerlinVerify.Check
