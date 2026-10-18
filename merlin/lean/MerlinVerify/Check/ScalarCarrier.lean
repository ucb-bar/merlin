/- Exact scalar-carrier normalization. Only actual source producer execution establishes a
uniform rank-zero tensor, and alias substitution independently checks every continuation use. -/
import MerlinVerify.Rules.ScalarCarrierKernel
import MerlinVerify.Check.TypedAlias
import MerlinVerify.Check.BodyNormalize
import MerlinVerify.Check.RegionDepth
import MerlinVerify.Check.TypedNormalizeC1

namespace MerlinVerify.Check.ScalarCarrier

open MerlinVerify MerlinVerify.Sem.Ops

/-- A canonical internal spelling lets the existing uniform-fact learner and exact dead-splat
normalizer share the constructor proof. Exported source and target syntax remain unchanged. -/
def splat (s : Op) : Op := .mk "tensor.splat" s.results s.operands [] [] [] []

def Constructor (Γ : TEnv) (s : Op) : Prop :=
  s.name = "tensor.from_elements" ∧ unmodelledAttr? s.blind = none ∧ s.regions = [] ∧ s.successors = [] ∧
  match s.operands, s.results, Sem.Ops.Tensor.resultTensorTy s with
  | [payload], [_], some ([], element) => Γ.lookup payload = some (.scalar element)
  | _, _, _ => False

instance (Γ : TEnv) (s : Op) : Decidable (Constructor Γ s) := by
  unfold Constructor; split <;> infer_instance

def rewrite (Γ : TEnv) (s : Op) : Op := if Constructor Γ s then splat s else s

theorem rewrite_spec (Γ : TEnv) (s : Op) :
    (rewrite Γ s).isTerminator = s.isTerminator ∧ (rewrite Γ s).results = s.results ∧
    ∀ ib env, EnvTyped ib Γ env → ∀ ctx call,
      stepOp opSem ctx call env (rewrite Γ s) = stepOp opSem ctx call env s := by
  unfold rewrite
  split
  · next hc =>
    obtain ⟨hn, hm, hregs, _, h⟩ := hc
    split at h
    · next payload result element ho hr ht =>
      have hs : s.isTerminator = false := by simp [Op.isTerminator, terminatorNames, hn]
      have htgt : (splat s).isTerminator = false := rfl
      refine ⟨htgt.trans hs.symm, rfl, ?_⟩
      intro ib env he ctx call
      obtain ⟨value, hv, hty⟩ := he _ _ h
      cases value with
      | tensor _ => cases hty
      | scalar value =>
        have htarget := UniformFacts.splat_exec (op := splat s) rfl rfl ho hr
          (by simpa [splat, Op.results, Sem.Ops.Tensor.resultTensorTy] using ht) ctx call env hv
        rw [htarget]
        unfold stepOp
        rw [ho, hregs]
        simp only [Env.getAll, Outcome.mapM, hv, Outcome.ofOption, Outcome.bind_ok, evalRegions,
          Rules.ScalarCarrier.from_elements_sem hn hm ht ctx call value]
        simp [Env.bindAll, hr]
    · contradiction
  · exact ⟨rfl, rfl, fun _ _ _ _ _ => rfl⟩

def advance (Γ : TEnv) (facts : List UniformFacts.Fact) (s : Op) : List UniformFacts.Fact :=
  UniformFacts.advance facts (rewrite Γ s)

theorem advance_sound {ib : Nat} {Γ : TEnv} {facts : List UniformFacts.Fact} {s : Op}
    {ctx : Ctx} {call : CallSem} {env env' : Env}
    (he : EnvTyped ib Γ env) (hf : BindingFacts.Valid UniformFacts.model facts env)
    (hs : stepOp opSem ctx call env s = .ok env') :
    BindingFacts.Valid UniformFacts.model (advance Γ facts s) env' := by
  apply UniformFacts.advance_sound hf
  rw [(rewrite_spec Γ s).2.2 ib env he ctx call]
  exact hs

private def valueDecEq (a b : Value) : Decidable (a = b) :=
  decidable_of_iff (a.id = b.id ∧ a.ty = b.ty) (by cases a; cases b; simp)
local instance : DecidableEq Value := valueDecEq

def Legal (Γ : TEnv) (s : Op) (f : UniformFacts.Fact) (r : Value) : Prop :=
  s.name = "tensor.extract" ∧ unmodelledAttr? s.blind = none ∧ s.regions = [] ∧ s.successors = [] ∧
  s.operands = [f.tensor] ∧ s.results = [r] ∧ r.ty = f.element ∧ f.shape = [] ∧
  Γ.lookup f.tensor = some (.tensor [] f.element) ∧ Γ.lookup f.scalar = some (.scalar f.element)

instance (Γ : TEnv) (s : Op) (f : UniformFacts.Fact) (r : Value) : Decidable (Legal Γ s f r) := by
  unfold Legal; infer_instance

def eliminate (Γ : TEnv) (facts : List UniformFacts.Fact) (s : Op) (tail : List Op) : Option (List Op) := do
  if s.name = "tensor.extract" then
    let [r] := s.results | none
    let f ← facts.find? (fun f => decide (Legal Γ s f r))
    TypedAlias.rename? Γ r.id f.scalar tail
  else none

theorem eliminate_sound {Γ : TEnv} {facts : List UniformFacts.Fact} {s : Op} {tail renamed : List Op}
    (h : eliminate Γ facts s tail = some renamed) {ctx : Ctx} {call : CallSem} {env : Env}
    (hf : BindingFacts.Valid UniformFacts.model facts env) :
    s.isTerminator = false ∧ ∃ env', stepOp opSem ctx call env s = .ok env' ∧
      evalOps opSem ctx call env renamed = evalOps opSem ctx call env' tail := by
  unfold eliminate at h
  split at h
  · next hn =>
    split at h
    · next r hr =>
      cases hm : facts.find? (fun f => decide (Legal Γ s f r)) with
      | none => simp [hm] at h
      | some f =>
        simp only [hm, bind, Option.bind] at h
        have hl : Legal Γ s f r := by simpa using List.find?_some hm
        obtain ⟨_, hmodel, hregs, _, ho, _, _, hshape, _, _⟩ := hl
        obtain ⟨value, hp, ht⟩ := hf f (List.mem_of_find?_eq_some hm)
        rw [hshape] at ht
        have hs : stepOp opSem ctx call env s = .ok ((r.id, .scalar value) :: env) := by
          unfold stepOp
          rw [ho, hregs]
          simp only [Env.getAll, Outcome.mapM, ht, Outcome.ofOption, Outcome.bind_ok, evalRegions,
            Rules.ScalarCarrier.extract_sem hn hmodel ctx call value]
          simp [Env.bindAll, hr]
        exact ⟨by simp [Op.isTerminator, terminatorNames, hn], _, hs, TypedAlias.rename_sound h hp⟩
    · cases h
  · cases h

def normalize (ib : Nat) (Γ : TEnv) (facts : List UniformFacts.Fact) : List Op → Option (List Op)
  | [] => some []
  | s :: ss =>
    if s.isTerminator then some (s :: ss) else do
      let _ ← s.operands.mapM (Γ.lookup ·)
      let rs ← s.results.mapM (fun r => vtyOf ib r.ty)
      let tail ← normalize ib ((s.results.map (·.id)).zip rs ++ Γ) (advance Γ facts s) ss
      pure ((eliminate Γ facts s tail).getD (rewrite Γ s :: tail))

/-- Source progress establishes both actual intermediate types and uniform facts. Alias checks
then preserve the entire normalized continuation, including every poison and failure outcome. -/
theorem normalize_sound {ib : Nat} (R : Rulebook opSem ib) (d : Nat) :
    ∀ Γ facts ss ns yields ctx call env,
      checkOps R d Γ ss yields = true → normalize ib Γ facts ss = some ns →
      EnvTyped ib Γ env → BindingFacts.Valid UniformFacts.model facts env →
      ctx.indexBits = ib → d ≤ ctx.loopIndex.length →
      evalOps opSem ctx call env ns = evalOps opSem ctx call env ss
  | _, _, [], _, _, _, _, _, h, _, _, _, _, _ => by simp [checkOps] at h
  | Γ, facts, s :: ss, ns, yields, ctx, call, env, h, hn, he, hf, hib, hd => by
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
                  (advance Γ facts (.mk name results operands attrs props regions successors)) ss with
              | none => simp [htail] at hn
              | some tail =>
                simp only [htail, Option.pure_def, Option.some.injEq] at hn
                subst ns
                have ih : ∀ env', stepOp opSem ctx call env
                    (.mk name results operands attrs props regions successors) = .ok env' →
                    evalOps opSem ctx call env' tail = evalOps opSem ctx call env' ss := by
                  intro env' hs
                  exact normalize_sound R d _ _ ss tail yields ctx call env' hrest htail
                    (step_envTyped R (op := .mk name results operands attrs props regions successors)
                      hl hr hR hregs hib hd he hs)
                    (advance_sound he hf hs) hib hd
                cases helim : eliminate Γ facts (.mk name results operands attrs props regions successors) tail with
                | some renamed =>
                  simp only [Option.getD_some]
                  obtain ⟨_, env', hs, heq⟩ := eliminate_sound helim hf (ctx := ctx) (call := call)
                  rw [evalOps_cons_step _ _ _ _ _ _ ht, hs, Outcome.bind_ok]
                  exact heq.trans (ih env' hs)
                | none =>
                  simp only [Option.getD_none]
                  have hvterm := (rewrite_spec Γ
                    (.mk name results operands attrs props regions successors)).1.trans ht
                  rw [evalOps_cons_step _ _ _ _ _ _ ht,
                    evalOps_cons_step _ _ _ _ _ _ hvterm,
                    (rewrite_spec Γ (.mk name results operands attrs props regions successors)).2.2
                      ib env he ctx call]
                  cases hs : stepOp opSem ctx call env
                      (.mk name results operands attrs props regions successors) with
                  | ub => rfl
                  | stuck _ => rfl
                  | ok env' => exact ih env' hs

def raw (ib : Nat) : SplatNormalizer.Normalizer ib where
  run _ Γ facts ops _ := normalize ib Γ facts ops
  sound d Γ facts ops normalized yields ctx call env hc hn he hf hi hd :=
    normalize_sound (merlinBook ib) d Γ facts ops normalized yields ctx call env hc hn he hf hi hd

/-- After aliases are substituted, the existing exact splat pass can remove only those internal
scalar carriers whose tensor results no longer escape, including nested captures. -/
def base (ib : Nat) : SplatNormalizer.Normalizer ib :=
  BodyNormalize.sequential (raw ib) (SplatNormalizer.base ib)

def normalizer : TypedNormalizer where
  run ib d Γ ops hints yields :=
    (BodyNormalize.recursive (base ib) (max (RegionDepth.ops ops) (RegionDepth.ops hints))).run
      d Γ [] ops yields
  sound ib d Γ ops hints normalized yields ctx call env hc hn he hi hd :=
    (BodyNormalize.recursive (base ib) (max (RegionDepth.ops ops) (RegionDepth.ops hints))).sound
      d Γ [] ops normalized yields ctx call env hc hn he (by intro f hf; cases hf) hi hd

def check (ib : Nat) (src tgt : Module) (fn : String) (argTys resTys : List VTy) : Bool :=
  TypedNormalizeC1.check normalizer ib src tgt fn argTys resTys

theorem check_sound {ib : Nat} {src tgt : Module} {fn : String} {argTys resTys : List VTy}
    (h : check ib src tgt fn argTys resTys = true) :
    FuncRefinesC opSem (fun ctx => ctx.indexBits = ib) (fun args => HasTys ib args argTys)
      tgt fn src fn := TypedNormalizeC1.check_sound normalizer h

end MerlinVerify.Check.ScalarCarrier
