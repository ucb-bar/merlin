/- `StateNormalize` with facts that may depend on the run's context and call semantics. A block
runs under one context and one call semantics, so a fact learned from an earlier operation's
actual execution, such as the value a floating-point body produced, stays usable later in the
same block. The induction is `StateNormalize`'s with those two parameters held fixed; `advance`
also sees the rest of the source block, as syntax. -/
import MerlinVerify.Check.BodyNormalize

namespace MerlinVerify.Check.CtxNormalize

open MerlinVerify MerlinVerify.Sem.Ops

structure Family (State : Type) (Valid : State → Ctx → CallSem → Env → Prop) where
  initial : State
  initial_valid : ∀ ctx call env, Valid initial ctx call env
  /-- `rest`: the operations after `s` in the source block (syntax only, no semantic premise). -/
  advance : Nat → TEnv → State → Op → List Op → State
  advance_sound : ∀ ib Γ facts s rest ctx call env env',
    EnvTyped ib Γ env → Valid facts ctx call env → ctx.indexBits = ib →
    stepOp opSem ctx call env s = .ok env' →
    Valid (advance ib Γ facts s rest) ctx call env'
  finish : Nat → TEnv → State → List Op → Op → List Op → List Op
  finish_sound : ∀ ib Γ facts hints s tail ctx call env,
    s.isTerminator = false → EnvTyped ib Γ env → Valid facts ctx call env →
    ctx.indexBits = ib →
    evalOps opSem ctx call env (finish ib Γ facts hints s tail) =
      evalOps opSem ctx call env (s :: tail)

variable {State : Type} {Valid : State → Ctx → CallSem → Env → Prop}

def normalize (E : Family State Valid) (ib : Nat) (Γ : TEnv) (facts : State) (hints : List Op) :
    List Op → Option (List Op)
  | [] => some []
  | s :: ss =>
    if s.isTerminator then some (s :: ss) else do
      let _ ← s.operands.mapM (Γ.lookup ·)
      let rs ← s.results.mapM (fun r => vtyOf ib r.ty)
      let tail ← normalize E ib ((s.results.map (·.id)).zip rs ++ Γ)
        (E.advance ib Γ facts s ss) hints ss
      pure (E.finish ib Γ facts hints s tail)

/-- Every fact is justified by successful original execution. Continuation edits must be exact
for the actual environment, so unknown source failures cannot silently become total. -/
theorem normalize_sound (E : Family State Valid) {ib : Nat} (R : Rulebook opSem ib) (d : Nat) :
    ∀ Γ facts hints ss ns yields ctx call env,
      checkOps R d Γ ss yields = true → normalize E ib Γ facts hints ss = some ns →
      EnvTyped ib Γ env → Valid facts ctx call env →
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
                  (E.advance ib Γ facts (.mk name results operands attrs props regions successors) ss) hints ss with
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
                    (E.advance_sound ib Γ facts _ ss ctx call env env' he hf hib hs) hib hd

/-- Start every block with the family's universally valid initial state. No fact may be
invented from annotations or an unchecked capture. BodyNormalize supplies the separate
structured region traversal and actual local typing. -/
def base (E : Family State Valid) (ib : Nat) (hints : List Op) : SplatNormalizer.Normalizer ib where
  run _ Γ _ ss _ := normalize E ib Γ E.initial hints ss
  sound d Γ _ ss ns yields ctx call env hc hn he _ hi hd :=
    normalize_sound E (merlinBook ib) d Γ E.initial hints ss ns yields ctx call env hc hn he
      (E.initial_valid ctx call env) hi hd

end MerlinVerify.Check.CtxNormalize
