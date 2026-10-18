/- Exact typed local normalization with continuation-sensitive edits. Families justify alias
substitution, total deletion, or a changed prefix; one induction owns source facts and typing. -/
import MerlinVerify.Check.BindingFacts
import MerlinVerify.Check.BodyNormalize

namespace MerlinVerify.Check.TailNormalize

open MerlinVerify MerlinVerify.Sem.Ops

structure Family (F : Type) (model : BindingFacts.Model F) where
  advance : Nat → TEnv → List F → Op → List F
  advance_sound : ∀ ib Γ facts s ctx call env env',
    EnvTyped ib Γ env → BindingFacts.Valid model facts env → ctx.indexBits = ib →
    stepOp opSem ctx call env s = .ok env' →
    BindingFacts.Valid model (advance ib Γ facts s) env'
  finish : Nat → TEnv → List F → List Op → Op → List Op → List Op
  finish_sound : ∀ ib Γ facts hints s tail ctx call env,
    s.isTerminator = false → EnvTyped ib Γ env → BindingFacts.Valid model facts env →
    ctx.indexBits = ib →
    evalOps opSem ctx call env (finish ib Γ facts hints s tail) =
      evalOps opSem ctx call env (s :: tail)

variable {F : Type} {model : BindingFacts.Model F}

def normalize (E : Family F model) (ib : Nat) (Γ : TEnv) (facts : List F) (hints : List Op) :
    List Op → Option (List Op)
  | [] => some []
  | s :: ss =>
    if s.isTerminator then some (s :: ss) else do
      let _ ← s.operands.mapM (Γ.lookup ·)
      let rs ← s.results.mapM (fun r => vtyOf ib r.ty)
      let tail ← normalize E ib ((s.results.map (·.id)).zip rs ++ Γ)
        (E.advance ib Γ facts s) hints ss
      pure (E.finish ib Γ facts hints s tail)

/-- Every fact is justified by successful original execution. Continuation edits must be exact
for the actual environment, so unknown source failures cannot silently become total. -/
theorem normalize_sound (E : Family F model) {ib : Nat} (R : Rulebook opSem ib) (d : Nat) :
    ∀ Γ facts hints ss ns yields ctx call env,
      checkOps R d Γ ss yields = true → normalize E ib Γ facts hints ss = some ns →
      EnvTyped ib Γ env → BindingFacts.Valid model facts env →
      ctx.indexBits = ib → d ≤ ctx.loopIndex.length →
      evalOps opSem ctx call env ns = evalOps opSem ctx call env ss
  | _, _, _, [], _, _, _, _, _, h, _, _, _, _, _ => by simp [checkOps] at h
  | Γ, facts, hints, s :: ss, ns, yields, ctx, call, env, h, hn, he, hf, hib, hd => by
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
              cases htail : normalize E ib ((results.map (·.id)).zip rs ++ Γ)
                  (E.advance ib Γ facts (.mk name results operands attrs props regions successors)) hints ss with
              | none => simp [htail] at hn
              | some tail =>
                simp only [htail, Option.pure_def, Option.some.injEq] at hn
                subst ns
                rw [E.finish_sound ib Γ facts hints _ tail ctx call env ht he hf hib,
                  evalOps_cons_step _ _ _ _ _ _ ht, evalOps_cons_step _ _ _ _ _ _ ht]
                cases hs : stepOp opSem ctx call env
                    (.mk name results operands attrs props regions successors) with
                | ub => rfl
                | stuck _ => rfl
                | ok env' =>
                  exact normalize_sound E R d _ _ hints ss tail yields ctx call env' hrest htail
                    (step_envTyped R (op := .mk name results operands attrs props regions successors)
                      hl hr hR hregs hib hd he hs)
                    (E.advance_sound ib Γ facts _ ctx call env env' he hf hib hs) hib hd

/-- Start every block without invented facts. BodyNormalize supplies the separate structured
region traversal and actual local typing; family facts are not inferred from captures. -/
def base (E : Family F model) (ib : Nat) (hints : List Op) : SplatNormalizer.Normalizer ib where
  run _ Γ _ ss _ := normalize E ib Γ [] hints ss
  sound d Γ _ ss ns yields ctx call env hc hn he _ hi hd :=
    normalize_sound E (merlinBook ib) d Γ [] hints ss ns yields ctx call env hc hn he
      (by intro f hf; cases hf) hi hd

end MerlinVerify.Check.TailNormalize
