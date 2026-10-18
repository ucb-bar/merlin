/- Rank-zero dense extraction uses actual closed constant values. Opposite-side scalar
constants are untrusted materialization candidates and must reproduce the exact decoded bits. -/
import MerlinVerify.Check.ConstantFacts
import MerlinVerify.Check.TypedDead
import MerlinVerify.Check.BodyNormalize
import MerlinVerify.Check.RegionDepth
import MerlinVerify.Check.TypedNormalizeC1

namespace MerlinVerify.Check.DenseExtract

open MerlinVerify MerlinVerify.Sem.Ops

/-- Restrict this family before evaluating a payload: unrelated weight tensors and scalar
constants need neither decoding nor retained facts during a rank-zero extraction scan. -/
def producer (s : Op) : Bool :=
  s.name == "arith.constant" && match Sem.Ops.Tensor.resultTensorTy s with
  | some ([], _) => true
  | _ => false

def advance (ib : Nat) (facts : List ConstantFacts.Fact) (s : Op) : List ConstantFacts.Fact :=
  if producer s then ConstantFacts.advance ib facts s
  else BindingFacts.forget ConstantFacts.model (s.results.map (·.id)) facts

theorem advance_sound {ib : Nat} {facts : List ConstantFacts.Fact} {s : Op}
    {ctx : Ctx} {call : CallSem} {env env' : Env}
    (hf : BindingFacts.Valid ConstantFacts.model facts env) (hi : ctx.indexBits = ib)
    (hs : stepOp opSem ctx call env s = .ok env') :
    BindingFacts.Valid ConstantFacts.model (advance ib facts s) env' := by
  unfold advance
  split
  · exact ConstantFacts.advance_sound hf hi hs
  · exact BindingFacts.forget_step ConstantFacts.model hf hs

def scalar? (ib : Nat) (s : Op) : Option Val :=
  match ConstantFacts.value? ib s with
  | some (.scalar value) => some value
  | _ => none

theorem scalar_spec {ib : Nat} {s : Op} {value : Val} (h : scalar? ib s = some value) :
    ConstantFacts.value? ib s = some (.scalar value) := by
  unfold scalar? at h
  split at h
  · next v hv => cases h; exact hv
  · cases h

def Legal (ib : Nat) (f : ConstantFacts.Fact) (s v : Op) : Prop :=
  s.name = "tensor.extract" ∧ unmodelledAttr? s.blind = none ∧
    s.regions = [] ∧ s.successors = [] ∧ s.operands = [f.id] ∧
    match f.value with
    | .tensor t => t.shape = [] ∧ scalar? ib v = some (t.get [])
    | _ => False

instance (ib : Nat) (f : ConstantFacts.Fact) (s v : Op) : Decidable (Legal ib f s v) := by
  unfold Legal; split <;> infer_instance

theorem meaning {ib : Nat} {f : ConstantFacts.Fact} {s v : Op}
    (h : Legal ib f s v) (hr : v.results = s.results) {env : Env}
    (hf : ConstantFacts.model.holds f env) {ctx : Ctx} (hi : ctx.indexBits = ib) (call : CallSem) :
    stepOp opSem ctx call env v = stepOp opSem ctx call env s := by
  obtain ⟨hn, hm, hregs, _, ho, h⟩ := h
  cases hfval : f.value with
  | scalar _ => simp only [hfval] at h
  | tensor t =>
    simp only [hfval] at h
    obtain ⟨hshape, hv⟩ := h
    have hlookup : env.get? f.id = some (.tensor t) := by simpa [ConstantFacts.model, hfval] using hf
    obtain ⟨he, hsem⟩ := ConstantFacts.value_spec (scalar_spec hv)
    have ht : opSem ctx call v [] [] = .ok [.scalar (t.get [])] :=
      (ConstantFacts.context_independent he hi call).trans hsem
    have hd : dialectOf s.name = "tensor" := by rw [hn]; cbv
    unfold stepOp
    rw [he.2.2.1, he.2.2.2.1, ho, hregs]
    simp only [Env.getAll, Outcome.mapM, hlookup, Outcome.ofOption, Outcome.bind_ok,
      evalRegions, ht, tensor_sem_of hm hd]
    simp [Sem.Ops.Tensor.sem, Op.blind_name, hn, hshape, hr]

def candidate (s hint : Op) : Op := .mk hint.name s.results [] hint.attrs hint.props [] []

def tryHints (ib : Nat) (f : ConstantFacts.Fact) (s : Op) : List Op → Option Op
  | [] => none
  | hint :: hints =>
    if (hint.results.map (·.ty)) == (s.results.map (·.ty)) then
      let v := candidate s hint
      if Legal ib f s v then some v else tryHints ib f s hints
    else tryHints ib f s hints

theorem tryHints_spec {ib : Nat} {f : ConstantFacts.Fact} {s : Op} : ∀ hints v,
    tryHints ib f s hints = some v → Legal ib f s v ∧ v.results = s.results
  | [], _, h => by cases h
  | hint :: hints, v, h => by
    simp only [tryHints] at h
    split at h
    · split at h
      · next hl => cases h; exact ⟨hl, rfl⟩
      · exact tryHints_spec hints v h
    · exact tryHints_spec hints v h

def select (ib : Nat) (hints : List Op) (s : Op) : List ConstantFacts.Fact → Op
  | [] => s
  | f :: facts =>
    if s.operands = [f.id] then
      match tryHints ib f s hints with
      | some v => v
      | none => select ib hints s facts
    else select ib hints s facts

theorem select_spec (ib : Nat) (hints : List Op) (s : Op) : ∀ facts,
    (select ib hints s facts).isTerminator = s.isTerminator ∧
    (select ib hints s facts).results = s.results ∧
    ∀ env, BindingFacts.Valid ConstantFacts.model facts env → ∀ ctx call, ctx.indexBits = ib →
      stepOp opSem ctx call env (select ib hints s facts) = stepOp opSem ctx call env s
  | [] => ⟨rfl, rfl, fun _ _ _ _ _ => rfl⟩
  | f :: facts => by
    simp only [select]
    have ih := select_spec ib hints s facts
    split
    · cases hh : tryHints ib f s hints with
      | none => exact ⟨ih.1, ih.2.1, fun env hf ctx call hi =>
          ih.2.2 env (fun g hg => hf g (by simp [hg])) ctx call hi⟩
      | some v =>
        obtain ⟨hl, hr⟩ := tryHints_spec hints v hh
        have hvname : v.name = "arith.constant" := by
          obtain ⟨_, _, _, _, _, h⟩ := hl
          cases heq : f.value with
          | scalar _ => simp only [heq] at h
          | tensor _ =>
            simp only [heq] at h
            exact (ConstantFacts.value_spec (scalar_spec h.2)).1.1
        refine ⟨?_, hr, fun env hf ctx call hi => meaning hl hr (hf f (by simp)) hi call⟩
        simp [Op.isTerminator, terminatorNames, hvname, hl.1]
    · exact ⟨ih.1, ih.2.1, fun env hf ctx call hi =>
        ih.2.2 env (fun g hg => hf g (by simp [hg])) ctx call hi⟩

def rewrite (ib : Nat) (facts : List ConstantFacts.Fact) (hints : List Op) (s : Op) : Op :=
  if s.name = "tensor.extract" then select ib hints s facts else s

theorem rewrite_spec (ib : Nat) (facts : List ConstantFacts.Fact) (hints : List Op) (s : Op) :
    (rewrite ib facts hints s).isTerminator = s.isTerminator ∧
    (rewrite ib facts hints s).results = s.results ∧
    ∀ env, BindingFacts.Valid ConstantFacts.model facts env → ∀ ctx call, ctx.indexBits = ib →
      stepOp opSem ctx call env (rewrite ib facts hints s) = stepOp opSem ctx call env s := by
  unfold rewrite
  split
  · exact select_spec ib hints s facts
  · exact ⟨rfl, rfl, fun _ _ _ _ _ => rfl⟩

def drop? (ib : Nat) (Γ : TEnv) (s : Op) (tail : List Op) : Bool :=
  producer s && (ConstantFacts.value? ib s).isSome && TypedDead.unused Γ s tail

theorem drop_sound {ib : Nat} {Γ : TEnv} {s : Op} {tail : List Op}
    (h : drop? ib Γ s tail = true) {ctx : Ctx} (hi : ctx.indexBits = ib) (call : CallSem) (env : Env) :
    ∃ env', stepOp opSem ctx call env s = .ok env' ∧
      evalOps opSem ctx call env' tail = evalOps opSem ctx call env tail := by
  simp only [drop?, Bool.and_eq_true] at h
  obtain ⟨value, hv⟩ := Option.isSome_iff_exists.1 h.1.2
  obtain ⟨r, hr, hs⟩ := ConstantFacts.value_exec hv hi call env
  exact TypedDead.drop_sound h.2 ⟨r, value, hr, hs⟩

def normalize (ib : Nat) (Γ : TEnv) (facts : List ConstantFacts.Fact) (hints : List Op) :
    List Op → Option (List Op)
  | [] => some []
  | s :: ss =>
    if s.isTerminator then some (s :: ss) else do
      let _ ← s.operands.mapM (Γ.lookup ·)
      let rs ← s.results.mapM (fun r => vtyOf ib r.ty)
      let tail ← normalize ib ((s.results.map (·.id)).zip rs ++ Γ)
        (advance ib facts s) hints ss
      pure (if drop? ib Γ s tail then tail else rewrite ib facts hints s :: tail)

/-- Every prefix fact is learned from an actually successful source step and invalidated when
any dependency is shadowed. Exact equality retains poison and all source failures. -/
theorem normalize_sound {ib : Nat} (d : Nat) :
    ∀ Γ facts hints ss ns yields ctx call env,
      checkOps (merlinBook ib) d Γ ss yields = true → normalize ib Γ facts hints ss = some ns →
      EnvTyped ib Γ env → BindingFacts.Valid ConstantFacts.model facts env →
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
                  (advance ib facts (.mk name results operands attrs props regions successors)) hints ss with
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
                    (advance_sound hf hib hs) hib hd
                split
                · next hdrop =>
                  obtain ⟨env', hs, heq⟩ := drop_sound hdrop hib call env
                  rw [evalOps_cons_step _ _ _ _ _ _ ht, hs, Outcome.bind_ok]
                  exact heq.symm.trans (ih env' hs)
                · have hvterm := (rewrite_spec ib facts hints
                      (.mk name results operands attrs props regions successors)).1.trans ht
                  rw [evalOps_cons_step _ _ _ _ _ _ ht,
                    evalOps_cons_step _ _ _ _ _ _ hvterm,
                    (rewrite_spec ib facts hints (.mk name results operands attrs props regions successors)).2.2
                      env hf ctx call hib]
                  cases hs : stepOp opSem ctx call env
                      (.mk name results operands attrs props regions successors) with
                  | ub => rfl
                  | stuck _ => rfl
                  | ok env' => exact ih env' hs

mutual
  def hintOps : List Op → List Op
    | [] => []
    | (.mk name results operands attrs props regions successors) :: rest =>
      let s := Op.mk name results operands attrs props regions successors
      (if name = "arith.constant" then [s] else []) ++ hintRegions regions ++ hintOps rest
  def hintRegions : List Region → List Op
    | [] => []
    | .mk blocks :: rest => hintBlocks blocks ++ hintRegions rest
  def hintBlocks : List Block → List Op
    | [] => []
    | .mk _ ops :: rest => hintOps ops ++ hintBlocks rest
end

/-- Constants are learned afresh in each block. Captured tensors are never treated as constants
from their types, and region arguments cannot inherit a stale binding with the same name. -/
def base (ib : Nat) (headers : List Op) : SplatNormalizer.Normalizer ib where
  run _ Γ _ ops _ := normalize ib Γ [] headers ops
  sound d Γ _ ops normalized yields ctx call env hc hn he _ hi hd :=
    normalize_sound d Γ [] headers ops normalized yields ctx call env hc hn he
      (by intro f hf; cases hf) hi hd

def normalizer : TypedNormalizer where
  run ib d Γ ops hints yields :=
    (BodyNormalize.recursive (base ib (hintOps hints))
      (max (RegionDepth.ops ops) (RegionDepth.ops hints))).run d Γ [] ops yields
  sound ib d Γ ops hints normalized yields ctx call env hc hn he hi hd :=
    (BodyNormalize.recursive (base ib (hintOps hints))
      (max (RegionDepth.ops ops) (RegionDepth.ops hints))).sound
      d Γ [] ops normalized yields ctx call env hc hn he (by intro f hf; cases hf) hi hd

def check (ib : Nat) (src tgt : Module) (fn : String) (argTys resTys : List VTy) : Bool :=
  TypedNormalizeC1.check normalizer ib src tgt fn argTys resTys

theorem check_sound {ib : Nat} {src tgt : Module} {fn : String} {argTys resTys : List VTy}
    (h : check ib src tgt fn argTys resTys = true) :
    FuncRefinesC opSem (fun ctx => ctx.indexBits = ib) (fun args => HasTys ib args argTys)
      tgt fn src fn := TypedNormalizeC1.check_sound normalizer h

end MerlinVerify.Check.DenseExtract
