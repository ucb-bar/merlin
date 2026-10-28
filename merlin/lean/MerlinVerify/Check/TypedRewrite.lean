/- Typed local normalization. Runtime facts come from checked source execution, including each
intermediate SSA binding. The universal C1 interface is unchanged. -/
import MerlinVerify.Check.Rulebook
import MerlinVerify.Check.C1

namespace MerlinVerify.Check

open MerlinVerify MerlinVerify.Sem.Ops

/-- An operation expansion with an explicit typed-input domain. Region closures, operands and
result bindings stay unchanged; no runtime type fact is inferred from a rewrite annotation. -/
structure TypedExpansion where
  expand : Nat → Op → List VTy → Option Op
  sound : ∀ ib s tys v, expand ib s tys = some v →
    v.isTerminator = s.isTerminator ∧ v.operands = s.operands ∧
    v.results = s.results ∧ v.regions = s.regions ∧
    ∀ ctx call args regions, ctx.indexBits = ib → HasTys ib args tys →
      OutLE (opSem ctx call v args regions) (opSem ctx call s args regions)

namespace TypedExpansion

def rewrite (E : TypedExpansion) (ib : Nat) (s : Op) (tys : List VTy) : Op :=
  (E.expand ib s tys).getD s

theorem rewrite_sound (E : TypedExpansion) (ib : Nat) (s : Op) (tys : List VTy) :
    (E.rewrite ib s tys).isTerminator = s.isTerminator ∧
    (E.rewrite ib s tys).operands = s.operands ∧
    (E.rewrite ib s tys).results = s.results ∧
    (E.rewrite ib s tys).regions = s.regions ∧
    ∀ ctx call args regions, ctx.indexBits = ib → HasTys ib args tys →
      OutLE (opSem ctx call (E.rewrite ib s tys) args regions) (opSem ctx call s args regions) := by
  unfold rewrite
  cases he : E.expand ib s tys with
  | none => exact ⟨rfl, rfl, rfl, rfl, fun _ _ _ _ _ _ => .inr (.inr rfl)⟩
  | some v => exact E.sound ib s tys v he

theorem step (E : TypedExpansion) {ib : Nat} {s : Op} {tys : List VTy}
    {Γ : TEnv} {env : Env} (hΓ : EnvTyped ib Γ env)
    (hl : s.operands.mapM (Γ.lookup ·) = some tys) (ctx : Ctx) (call : CallSem)
    (hib : ctx.indexBits = ib) :
    stepOp opSem ctx call env s = .ub ∨
    (∃ r, stepOp opSem ctx call env s = .stuck r) ∨
    stepOp opSem ctx call env (E.rewrite ib s tys) = stepOp opSem ctx call env s := by
  obtain ⟨_, ho, hr, hregs, hsem⟩ := E.rewrite_sound ib s tys
  obtain ⟨args, ha, ht⟩ := hΓ.getAll _ _ hl
  unfold stepOp
  rw [ho, hr, hregs, ha]
  simp only [Outcome.bind_ok]
  rcases hsem ctx call args _ hib ht with hu | ⟨r, hs⟩ | he
  · rw [hu]; exact .inl rfl
  · rw [hs]; exact .inr (.inl ⟨_, rfl⟩)
  · rw [he]; exact .inr (.inr rfl)

def normalize (E : TypedExpansion) (ib : Nat) (Γ : TEnv) : List Op → Option (List Op)
  | [] => some []
  | s :: ss =>
    if s.isTerminator then some (s :: ss) else do
      let tys ← s.operands.mapM (Γ.lookup ·)
      let rs ← s.results.mapM (fun r => vtyOf ib r.ty)
      let tail ← E.normalize ib ((s.results.map (·.id)).zip rs ++ Γ) ss
      pure (E.rewrite ib s tys :: tail)

end TypedExpansion

/-- A successful checked source step establishes the types of its actual SSA results. This also
handles captures through the existing region safety proof and permits modeled UB/poison. -/
theorem step_envTyped {ib d : Nat} (R : Rulebook opSem ib) {Γ : TEnv} {op : Op}
    {tys rs : List VTy} {sigs : List RegionSig}
    (hl : op.operands.mapM (Γ.lookup ·) = some tys)
    (hr : op.results.mapM (fun r => vtyOf ib r.ty) = some rs)
    (hR : R.rule d op tys rs = some sigs) (hregs : checkRegions R Γ op.regions sigs = true)
    {ctx : Ctx} {call : CallSem} {env env' : Env}
    (hib : ctx.indexBits = ib) (hd : d ≤ ctx.loopIndex.length) (hΓ : EnvTyped ib Γ env)
    (hs : stepOp opSem ctx call env op = .ok env') :
    EnvTyped ib ((op.results.map (·.id)).zip rs ++ Γ) env' := by
  obtain ⟨args, ha, ht⟩ := hΓ.getAll _ _ hl
  have hsafe := R.sound d op tys rs sigs hR ctx call args _ hib hd ht
    (checkRegions_safe R Γ op.regions sigs call env hregs hΓ)
  unfold stepOp at hs
  rw [ha, Outcome.bind_ok] at hs
  cases hv : opSem ctx call op args (evalRegions opSem call env op.regions) with
  | ub => rw [hv] at hs; cases hs
  | stuck r => rw [hv] at hs; cases hs
  | ok vs =>
    have hvs := hsafe.2 vs hv
    have hlen : op.results.length = vs.length :=
      (optionMapM_length _ _ _ hr).symm.trans (HasTys.length hvs).symm
    rw [hv, Outcome.bind_ok, bindAll_ok hlen] at hs
    cases hs
    exact hΓ.extend _ vs rs hvs

theorem TypedExpansion.normalize_sound (E : TypedExpansion) {ib : Nat}
    (R : Rulebook opSem ib) (d : Nat) :
    ∀ Γ ss ns yields ctx call env,
      checkOps R d Γ ss yields = true → E.normalize ib Γ ss = some ns →
      EnvTyped ib Γ env → ctx.indexBits = ib → d ≤ ctx.loopIndex.length →
      OutLE (evalOps opSem ctx call env ns) (evalOps opSem ctx call env ss)
  | _, [], _, _, _, _, _, h, _, _, _, _ => by simp [checkOps] at h
  | Γ, s :: ss, ns, yields, ctx, call, env, h, hn, hΓ, hib, hd => by
    by_cases hterm : s.isTerminator = true
    · simp only [normalize, hterm, ↓reduceIte, Option.some.injEq] at hn
      subst ns
      exact .inr (.inr rfl)
    · have ht : s.isTerminator = false := by simpa using hterm
      cases s with
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
              simp only [normalize, ht, Bool.false_eq_true, ↓reduceIte, hl, hr,
                Op.operands, Op.results, bind, Option.bind] at hn
              cases htail : E.normalize ib ((results.map (·.id)).zip rs ++ Γ) ss with
              | none => simp [htail] at hn
              | some tail =>
                simp only [htail, Option.pure_def, Option.some.injEq] at hn
                subst ns
                have hvterm := (E.rewrite_sound ib _ tys).1.trans ht
                rw [evalOps_cons_step _ _ _ _ _ _ ht,
                  evalOps_cons_step _ _ _ _ _ _ hvterm]
                rcases E.step (s := .mk name results operands attrs props regions successors)
                    (tys := tys) hΓ hl ctx call hib with hu | ⟨r, hs⟩ | he
                · rw [hu]; exact .inl rfl
                · rw [hs]; exact .inr (.inl ⟨r, rfl⟩)
                · rw [he]
                  cases hstep : stepOp opSem ctx call env (.mk name results operands attrs props regions successors) with
                  | ub => exact .inl rfl
                  | stuck r => exact .inr (.inl ⟨r, rfl⟩)
                  | ok env' =>
                    simp only [Outcome.bind_ok]
                    exact E.normalize_sound R d _ ss tail yields ctx call env' hrest htail
                      (step_envTyped R (op := .mk name results operands attrs props regions successors)
                        hl hr hR hregs hib hd hΓ hstep) hib hd

end MerlinVerify.Check
