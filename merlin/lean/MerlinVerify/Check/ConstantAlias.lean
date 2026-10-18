/- Exact scalar-constant aliases are established by actual semantic values at the checked
index width. Literal spelling is irrelevant; float payloads are compared as exact bits only. -/
import MerlinVerify.Check.ConstantFacts
import MerlinVerify.Check.TypedAlias
import MerlinVerify.Check.BodyNormalize
import MerlinVerify.Check.RegionDepth
import MerlinVerify.Check.TypedNormalizeC1

namespace MerlinVerify.Check.ConstantAlias

open MerlinVerify MerlinVerify.Sem.Ops

/-- Avoid evaluating unrelated tensor weights. This is only a candidate filter; actual
constant execution below establishes the scalar payload and totality used by the proof. -/
def producer (s : Op) : Bool :=
  s.name == "arith.constant" && match s.results with
  | [r] => match r.ty with
    | .int _ | .index | .float _ => true
    | _ => false
  | _ => false

def scalar? (ib : Nat) (s : Op) : Option Val :=
  if producer s then
    match ConstantFacts.value? ib s with
    | some (.scalar value) => some value
    | _ => none
  else none

theorem scalar_spec {ib : Nat} {s : Op} {value : Val} (h : scalar? ib s = some value) :
    ConstantFacts.value? ib s = some (.scalar value) := by
  unfold scalar? at h
  split at h
  · split at h
    · next v hv => cases h; exact hv
    · cases h
  · cases h

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

def matching (ib : Nat) (Γ : TEnv) (r : Value) (value : Val) (f : ConstantFacts.Fact) : Bool :=
  (Γ.lookup f.id == vtyOf ib r.ty) && match f.value with
  | .scalar previous => previous == value
  | _ => false

theorem matching_value {ib : Nat} {Γ : TEnv} {r : Value} {value : Val} {f : ConstantFacts.Fact}
    (h : matching ib Γ r value f = true) : f.value = .scalar value := by
  simp only [matching, Bool.and_eq_true] at h
  cases hv : f.value with
  | tensor _ => simp [hv] at h
  | scalar previous => simpa [hv] using h.2

def eliminate (ib : Nat) (Γ : TEnv) (facts : List ConstantFacts.Fact) (s : Op)
    (tail : List Op) : Option (List Op) := do
  let value ← scalar? ib s
  let [r] := s.results | none
  let f ← facts.find? (matching ib Γ r value)
  TypedAlias.rename? Γ r.id f.id tail

theorem eliminate_sound {ib : Nat} {Γ : TEnv} {facts : List ConstantFacts.Fact}
    {s : Op} {tail renamed : List Op} (h : eliminate ib Γ facts s tail = some renamed)
    {ctx : Ctx} {call : CallSem} {env : Env}
    (hf : BindingFacts.Valid ConstantFacts.model facts env) (hi : ctx.indexBits = ib) :
    s.isTerminator = false ∧ ∃ env', stepOp opSem ctx call env s = .ok env' ∧
      evalOps opSem ctx call env renamed = evalOps opSem ctx call env' tail := by
  unfold eliminate at h
  cases hv : scalar? ib s with
  | none => simp [hv] at h
  | some value =>
    simp only [hv, bind, Option.bind] at h
    split at h
    · next r hr =>
      cases hm : facts.find? (matching ib Γ r value) with
      | none => simp [hm] at h
      | some f =>
        simp only [hm] at h
        have hvalue := matching_value (List.find?_some hm)
        have hlookup : env.get? f.id = some (.scalar value) := by
          simpa [ConstantFacts.model, hvalue] using hf f (List.mem_of_find?_eq_some hm)
        obtain ⟨r', hr', hs⟩ := ConstantFacts.value_exec (scalar_spec hv) hi call env
        have : r' = r := by simpa [hr] using hr'.symm
        subst r'
        have hn := (ConstantFacts.value_spec (scalar_spec hv)).1.1
        exact ⟨by simp [Op.isTerminator, terminatorNames, hn], _, hs,
          TypedAlias.rename_sound h hlookup⟩
    · cases h

def normalize (ib : Nat) (Γ : TEnv) (facts : List ConstantFacts.Fact) : List Op → Option (List Op)
  | [] => some []
  | s :: ss =>
    if s.isTerminator then some (s :: ss) else do
      let _ ← s.operands.mapM (Γ.lookup ·)
      let rs ← s.results.mapM (fun r => vtyOf ib r.ty)
      let tail ← normalize ib ((s.results.map (·.id)).zip rs ++ Γ) (advance ib facts s) ss
      pure ((eliminate ib Γ facts s tail).getD (s :: tail))

/-- Actual constant execution establishes the retained current-binding values. Checked aliases
then preserve the entire continuation, including captures and every poison or failure outcome. -/
theorem normalize_sound {ib : Nat} (R : Rulebook opSem ib) (d : Nat) :
    ∀ Γ facts ss ns yields ctx call env,
      checkOps R d Γ ss yields = true → normalize ib Γ facts ss = some ns →
      EnvTyped ib Γ env → BindingFacts.Valid ConstantFacts.model facts env →
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
                  (advance ib facts (.mk name results operands attrs props regions successors)) ss with
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
                    (advance_sound hf hib hs) hib hd
                cases helim : eliminate ib Γ facts (.mk name results operands attrs props regions successors) tail with
                | some renamed =>
                  simp only [Option.getD_some]
                  obtain ⟨_, env', hs, heq⟩ := eliminate_sound helim hf hib (ctx := ctx) (call := call)
                  rw [evalOps_cons_step _ _ _ _ _ _ ht, hs, Outcome.bind_ok]
                  exact heq.trans (ih env' hs)
                | none =>
                  simp only [Option.getD_none]
                  rw [evalOps_cons_step _ _ _ _ _ _ ht, evalOps_cons_step _ _ _ _ _ _ ht]
                  cases hs : stepOp opSem ctx call env
                      (.mk name results operands attrs props regions successors) with
                  | ub => rfl
                  | stuck _ => rfl
                  | ok env' => exact ih env' hs

/-- Each block learns only actual local constants; captured values never become constant
facts from their annotations. The checked alias walk still updates captured uses of local results. -/
def base (ib : Nat) : SplatNormalizer.Normalizer ib where
  run _ Γ _ ops _ := normalize ib Γ [] ops
  sound d Γ _ ops normalized yields ctx call env hc hn he _ hi hd :=
    normalize_sound (merlinBook ib) d Γ [] ops normalized yields ctx call env hc hn he
      (by intro f hf; cases hf) hi hd

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

end MerlinVerify.Check.ConstantAlias
