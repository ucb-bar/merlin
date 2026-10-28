/- Exact removal of unused total scalar casts. The shared TypedDead helper checks every
continuation use, including nested captures, and protects prior bindings from shadowing. -/
import MerlinVerify.Check.TypedDead
import MerlinVerify.Check.TypedNormalizer
import MerlinVerify.Rules.IntegerCast

namespace MerlinVerify.Rules.IntegerCastDead

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops

def total (Γ : TEnv) (s : Op) : Bool := (IntegerCast.read Γ s).isSome

theorem total_exec {ib : Nat} {Γ : TEnv} {s : Op} (h : total Γ s = true)
    {env : Env} (he : EnvTyped ib Γ env) (ctx : Ctx) (call : CallSem) :
    ∃ r value, s.results = [r] ∧ stepOp opSem ctx call env s = .ok ((r.id, value) :: env) := by
  unfold total at h
  cases hr : IntegerCast.read Γ s with
  | none => simp [hr] at h
  | some f =>
    have hl := IntegerCast.read_legal hr
    obtain ⟨_, _, _, _, hresults, _, hinput, _, _, _, _, _⟩ := IntegerCast.read_legal hr
    obtain ⟨value, hv, ht⟩ := he _ _ hinput
    cases value with
    | tensor _ => cases ht
    | scalar value =>
      refine ⟨f.result, .scalar (f.kind.apply f.outputWidth value), hresults, ?_⟩
      exact IntegerCast.cast_exec hl ctx call env hv (IntegerCast.int_type_change ht)

def drop? (Γ : TEnv) (s : Op) (tail : List Op) : Bool :=
  total Γ s && TypedDead.unused Γ s tail

theorem drop_sound {ib : Nat} {Γ : TEnv} {s : Op} {tail : List Op}
    (h : drop? Γ s tail = true) {env : Env} (he : EnvTyped ib Γ env)
    (ctx : Ctx) (call : CallSem) :
    ∃ env', stepOp opSem ctx call env s = .ok env' ∧
      evalOps opSem ctx call env' tail = evalOps opSem ctx call env tail := by
  simp only [drop?, Bool.and_eq_true] at h
  obtain ⟨ht, hu⟩ := h
  exact TypedDead.drop_sound hu (total_exec ht he ctx call)

def normalize (ib : Nat) (Γ : TEnv) : List Op → Option (List Op)
  | [] => some []
  | s :: ss =>
    if s.isTerminator then some (s :: ss) else do
      let _ ← s.operands.mapM (Γ.lookup ·)
      let rs ← s.results.mapM (fun r => vtyOf ib r.ty)
      let tail ← normalize ib ((s.results.map (·.id)).zip rs ++ Γ) ss
      pure (if drop? Γ s tail then tail else s :: tail)

/-- Cleanup is exact for poison as well as initialized inputs: only a proved-total cast whose
result is absent from the complete checked continuation may be removed. -/
theorem normalize_sound {ib : Nat} (R : Rulebook opSem ib) (d : Nat) :
    ∀ Γ ss ns yields ctx call env,
      checkOps R d Γ ss yields = true → normalize ib Γ ss = some ns →
      EnvTyped ib Γ env → ctx.indexBits = ib → d ≤ ctx.loopIndex.length →
      evalOps opSem ctx call env ns = evalOps opSem ctx call env ss
  | _, [], _, _, _, _, _, h, _, _, _, _ => by simp [checkOps] at h
  | Γ, s :: ss, ns, yields, ctx, call, env, h, hn, he, hib, hd => by
    by_cases hterm : s.isTerminator = true
    · simp only [normalize, hterm, ↓reduceIte, Option.some.injEq] at hn
      subst ns; rfl
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
              cases htail : normalize ib ((results.map (·.id)).zip rs ++ Γ) ss with
              | none => simp [htail] at hn
              | some tail =>
                simp only [htail, Option.pure_def, Option.some.injEq] at hn
                subst ns
                have ih : ∀ env', stepOp opSem ctx call env
                    (.mk name results operands attrs props regions successors) = .ok env' →
                    evalOps opSem ctx call env' tail = evalOps opSem ctx call env' ss := by
                  intro env' hs
                  exact normalize_sound R d _ ss tail yields ctx call env' hrest htail
                    (step_envTyped R (op := .mk name results operands attrs props regions successors)
                      hl hr hR hregs hib hd he hs) hib hd
                split
                · next hdropped =>
                  obtain ⟨env', hs, heq⟩ := drop_sound hdropped he ctx call
                  rw [evalOps_cons_step _ _ _ _ _ _ ht, hs, Outcome.bind_ok]
                  exact heq.symm.trans (ih env' hs)
                · rw [evalOps_cons_step _ _ _ _ _ _ ht, evalOps_cons_step _ _ _ _ _ _ ht]
                  cases hs : stepOp opSem ctx call env
                      (.mk name results operands attrs props regions successors) with
                  | ub => rfl
                  | stuck _ => rfl
                  | ok env' => exact ih env' hs

def normalizer : TypedNormalizer where
  run ib _ Γ ss _ _ := normalize ib Γ ss
  sound ib d Γ ss _hints ns yields ctx call env hc hn he hi hd :=
    normalize_sound (merlinBook ib) d Γ ss ns yields ctx call env hc hn he hi hd

end MerlinVerify.Rules.IntegerCastDead
