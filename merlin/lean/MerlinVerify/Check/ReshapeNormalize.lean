/- Typed reshape provenance permits mixed compositions that are not universally valid from
result annotations alone. Target headers are merely candidates; every endpoint is rechecked. -/
import MerlinVerify.Check.ReshapeFacts
import MerlinVerify.Check.ReshapeDead
import MerlinVerify.Check.TypedNormalizer

namespace MerlinVerify.Check.ReshapeNormalize

open MerlinVerify MerlinVerify.Sem.Ops

/-- Both modeled reshape steps are total on the exact source-provenance values. -/
def Legal (f : ReshapeFacts.Fact) (s v : Op) : Prop :=
  Rules.ReshapeCompose.Names s ∧ Rules.ReshapeCompose.Names v ∧
  s.operands = [f.result] ∧ v.operands = [f.root] ∧
  match Sem.Ops.Tensor.resultTensorTy s, Sem.Ops.Tensor.resultTensorTy v,
      Sem.Ops.Tensor.reassociation? s, Sem.Ops.Tensor.reassociation? v with
  | some (dst, e), some (dst', e'), some gs, some gv =>
    dst = dst' ∧ e = f.element ∧ e' = f.element ∧
    reshapeValid s gs f.shape dst = true ∧ reshapeValid v gv f.original dst = true
  | _, _, _, _ => False

instance (f : ReshapeFacts.Fact) (s v : Op) : Decidable (Legal f s v) := by
  unfold Legal
  split <;> infer_instance

theorem meaning {f : ReshapeFacts.Fact} {s v : Op} (h : Legal f s v) (hresults : s.results = v.results)
    {env : Env} (hf : ReshapeFacts.Holds f env) (ctx : Ctx) (call : CallSem) :
    stepOp opSem ctx call env v = stepOp opSem ctx call env s := by
  obtain ⟨hs, hv, hso, hvo, h⟩ := h
  split at h
  · next dst e dst' e' gs gv hst hvt hsg hvg =>
    obtain ⟨rfl, rfl, rfl, hvs, hvv⟩ := h
    obtain ⟨t, hr, hshape, helt, hi⟩ := hf
    have hsemS := Rules.ReshapeCompose.sem_defined hs hst hsg ctx call
      (Rules.ReshapeIndex.reshape t f.shape f.element) (by simpa [Rules.ReshapeIndex.reshape, Tensor.canon] using hvs)
      (by simp [Rules.ReshapeIndex.reshape, Tensor.canon])
    have hsemV := Rules.ReshapeCompose.sem_defined hv hvt hvg ctx call t
      (hshape ▸ hvv) helt.symm
    have hcompose := Rules.ReshapeIndex.compose t f.shape dst f.element (reshapeValid_size hvs)
    unfold stepOp
    rw [hso, hvo, hs.2.2.2.1, hv.2.2.2.1]
    simp only [Env.getAll, Outcome.mapM, hr, hi, Outcome.ofOption, Outcome.bind_ok,
      evalRegions, hsemS, hsemV, hcompose, hresults]
  · contradiction

/-- Copy only a candidate's semantic header; source result names/types and proven root are fixed. -/
def candidate (f : ReshapeFacts.Fact) (s hint : Op) : Op :=
  .mk hint.name s.results [f.root] hint.attrs hint.props [] []

/-- Search lazily: irrelevant target operations never allocate a source candidate. -/
def tryHints (f : ReshapeFacts.Fact) (s : Op) : List Op → Option Op
  | [] => none
  | hint :: hints =>
    if Sem.Ops.Tensor.resultTensorTy hint = Sem.Ops.Tensor.resultTensorTy s then
      let v := candidate f s hint
      if Legal f s v then some v else tryHints f s hints
    else tryHints f s hints

theorem tryHints_spec {f : ReshapeFacts.Fact} {s : Op} : ∀ hints v,
    tryHints f s hints = some v → Legal f s v ∧ v.results = s.results
  | [], _, h => by cases h
  | hint :: hints, v, h => by
    simp only [tryHints] at h
    split at h
    · split at h
      · next hl => cases h; exact ⟨hl, rfl⟩
      · exact tryHints_spec hints v h
    · exact tryHints_spec hints v h

/-- A fact is relevant only to its actual current result operand. Producers are retained. -/
def select (hints : List Op) (s : Op) : List ReshapeFacts.Fact → Op
  | [] => s
  | f :: facts =>
    if s.operands = [f.result] then
      match tryHints f s hints with
      | some v => v
      | none => select hints s facts
    else select hints s facts

theorem select_spec (hints : List Op) (s : Op) : ∀ facts,
    (select hints s facts).isTerminator = s.isTerminator ∧
    (select hints s facts).results = s.results ∧
    ∀ env, BindingFacts.Valid ReshapeFacts.model facts env →
      ∀ ctx call, stepOp opSem ctx call env (select hints s facts) = stepOp opSem ctx call env s
  | [] => ⟨rfl, rfl, fun _ _ _ _ => rfl⟩
  | f :: facts => by
    simp only [select]
    have ih := select_spec hints s facts
    split
    · cases hh : tryHints f s hints with
      | none =>
        simp only
        exact ⟨ih.1, ih.2.1, fun env hf ctx call =>
          ih.2.2 env (fun g hg => hf g (by simp [hg])) ctx call⟩
      | some v =>
        obtain ⟨hl, hr⟩ := tryHints_spec hints v hh
        exact ⟨(Rules.ReshapeCompose.nonterm hl.2.1).trans (Rules.ReshapeCompose.nonterm hl.1).symm,
          hr, fun env hf ctx call => meaning hl hr.symm (hf f (by simp)) ctx call⟩
    · exact ⟨ih.1, ih.2.1, fun env hf ctx call =>
        ih.2.2 env (fun g hg => hf g (by simp [hg])) ctx call⟩

def rewrite (facts : List ReshapeFacts.Fact) (hints : List Op) (s : Op) : Op :=
  if Rules.ReshapeCompose.Names s then select hints s facts else s

theorem rewrite_spec (facts : List ReshapeFacts.Fact) (hints : List Op) (s : Op) :
    (rewrite facts hints s).isTerminator = s.isTerminator ∧
    (rewrite facts hints s).results = s.results ∧
    ∀ env, BindingFacts.Valid ReshapeFacts.model facts env →
      ∀ ctx call, stepOp opSem ctx call env (rewrite facts hints s) = stepOp opSem ctx call env s := by
  unfold rewrite
  split
  · exact select_spec hints s facts
  · exact ⟨rfl, rfl, fun _ _ _ _ => rfl⟩

def normalize (ib : Nat) (Γ : TEnv) (facts : List ReshapeFacts.Fact) (hints : List Op) :
    List Op → Option (List Op)
  | [] => some []
  | s :: ss =>
    if s.isTerminator then some (s :: ss) else do
      let _ ← s.operands.mapM (Γ.lookup ·)
      let rs ← s.results.mapM (fun r => vtyOf ib r.ty)
      let tail ← normalize ib ((s.results.map (·.id)).zip rs ++ Γ)
        (ReshapeFacts.advance Γ facts s) hints ss
      pure (if ReshapeDead.drop? Γ s tail then tail else rewrite facts hints s :: tail)

/-- Every prefix fact is learned from an actually successful source step and invalidated when
any dependency is shadowed. Exact equality retains poison and all source failures. -/
theorem normalize_sound {ib : Nat} (d : Nat) :
    ∀ Γ facts hints ss ns yields ctx call env,
      checkOps (merlinBook ib) d Γ ss yields = true → normalize ib Γ facts hints ss = some ns →
      EnvTyped ib Γ env → BindingFacts.Valid ReshapeFacts.model facts env →
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
              cases htail : normalize ib ((results.map (·.id)).zip rs ++ Γ)
                  (ReshapeFacts.advance Γ facts (.mk name results operands attrs props regions successors)) hints ss with
              | none => simp [htail] at hn
              | some tail =>
                simp only [htail, Option.pure_def, Option.some.injEq] at hn
                subst ns
                have ih : ∀ env', stepOp opSem ctx call env
                    (.mk name results operands attrs props regions successors) = .ok env' →
                    evalOps opSem ctx call env' tail = evalOps opSem ctx call env' ss := by
                  intro env' hs
                  exact normalize_sound d _ _ hints ss tail yields ctx call env' hrest htail
                    (step_envTyped (merlinBook ib) (op := .mk name results operands attrs props regions successors)
                      hl hr hR hregs hib hd he hs)
                    (ReshapeFacts.advance_sound he hf hs) hib hd
                split
                · next hdrop =>
                  obtain ⟨env', hs, heq⟩ := ReshapeDead.drop_sound hdrop he ctx call
                  rw [evalOps_cons_step _ _ _ _ _ _ ht, hs, Outcome.bind_ok]
                  exact heq.symm.trans (ih env' hs)
                · have hvterm := (rewrite_spec facts hints
                      (.mk name results operands attrs props regions successors)).1.trans ht
                  rw [evalOps_cons_step _ _ _ _ _ _ ht,
                    evalOps_cons_step _ _ _ _ _ _ hvterm,
                    (rewrite_spec facts hints (.mk name results operands attrs props regions successors)).2.2
                      env hf ctx call]
                  cases hs : stepOp opSem ctx call env
                      (.mk name results operands attrs props regions successors) with
                  | ub => rfl
                  | stuck _ => rfl
                  | ok env' => exact ih env' hs

def normalizer : TypedNormalizer where
  run ib _ Γ ss hints _ := normalize ib Γ [] (hints.filter (fun op => decide (Rules.ReshapeCompose.Names op))) ss
  sound ib d Γ ss hints ns yields ctx call env h hn he hib hd :=
    normalize_sound d Γ [] _ ss ns yields ctx call env h hn he (by intro f h; cases h) hib hd

end MerlinVerify.Check.ReshapeNormalize
