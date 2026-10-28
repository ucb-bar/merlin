/- Exact, typed normalization of uniform structured inputs. Facts are learned only from actual
source steps; the same checker is applied independently to both sides of a boundary. -/
import MerlinVerify.Rules.SplatScalar
import MerlinVerify.Check.UniformFacts
import MerlinVerify.Check.TypedRewrite

namespace MerlinVerify.Check.SplatNormalize

open MerlinVerify MerlinVerify.Sem.Ops MerlinVerify.Sem.Ops.Linalg
open MerlinVerify.Rules.SplatInputs

structure Input where
  tensor : ValueId
  payload : ValueId
  slot : Slot

def Input.legal (Γ : TEnv) (facts : List UniformFacts.Fact) (b : Input) : Prop :=
  Γ.lookup b.tensor = some (b.slot.ty false) ∧
  Γ.lookup b.payload = some b.slot.payloadTy ∧
  match b.slot with
  | .keep _ _ => b.tensor = b.payload
  | .splat shape elt _ => (⟨b.tensor, b.payload, shape, elt⟩ : UniformFacts.Fact) ∈ facts

instance (Γ : TEnv) (facts : List UniformFacts.Fact) (b : Input) :
    Decidable (b.legal Γ facts) := by unfold Input.legal; cases b.slot <;> infer_instance

def select (Γ : TEnv) (facts : List UniformFacts.Fact) (id : ValueId) (ty : VTy)
    (map : AffineMap) : Input :=
  match facts.find? (fun f => f.tensor == id && ty == .tensor f.shape f.element &&
      Γ.lookup f.scalar == some (.scalar f.element)) with
  | some f => ⟨id, f.scalar, .splat f.shape f.element map⟩
  | none => ⟨id, id, .keep ty map⟩

theorem input_value {ib : Nat} {Γ : TEnv} {facts : List UniformFacts.Fact} {env : Env}
    (he : EnvTyped ib Γ env) (hf : BindingFacts.Valid UniformFacts.model facts env)
    {b : Input} (hb : b.legal Γ facts) :
    ∃ value, env.get? b.payload = some value ∧ RVal.HasTy ib value b.slot.payloadTy ∧
      env.get? b.tensor = some (b.slot.value false value) := by
  obtain ⟨value, hv, ht⟩ := he _ _ hb.2.1
  refine ⟨value, hv, ht, ?_⟩
  cases hs : b.slot with
  | keep ty map =>
    have hi : b.tensor = b.payload := by simpa [Input.legal, hs] using hb.2.2
    simpa [hs, hi, Slot.value] using hv
  | splat shape elt map =>
    have hm : (⟨b.tensor, b.payload, shape, elt⟩ : UniformFacts.Fact) ∈ facts := by
      simpa [Input.legal, hs] using hb.2.2
    obtain ⟨scalar, hv', hx⟩ := hf _ hm
    have heq : value = .scalar scalar := Option.some.inj (hv.symm.trans hv')
    simpa [hs, heq, Slot.value] using hx

theorem input_values {ib : Nat} {Γ : TEnv} {facts : List UniformFacts.Fact} {env : Env}
    (he : EnvTyped ib Γ env) (hf : BindingFacts.Valid UniformFacts.model facts env) :
    ∀ bs : List Input, (∀ b ∈ bs, b.legal Γ facts) →
      ∃ values, env.getAll (bs.map Input.payload) = .ok values ∧
        HasTys ib values (bs.map (fun b => b.slot.payloadTy)) ∧
        env.getAll (bs.map Input.tensor) = .ok (inputs false (bs.map Input.slot) values)
  | [], _ => ⟨[], rfl, .nil, rfl⟩
  | b :: bs, hb => by
    obtain ⟨v, hv, ht, hx⟩ := input_value he hf (hb b (by simp))
    obtain ⟨vs, hvs, hts, hxs⟩ := input_values he hf bs (fun b hm => hb b (by simp [hm]))
    refine ⟨v :: vs, ?_, .cons ht hts, ?_⟩
    · simpa only [List.map_cons, Env.getAll, Outcome.mapM, hv, Outcome.ofOption,
        Outcome.bind_ok] using congrArg (fun out => out.bind (fun xs => .ok (v :: xs))) hvs
    · simpa only [List.map_cons, inputs, List.zip_cons_cons, List.map_cons, Env.getAll,
        Outcome.mapM, hx, Outcome.ofOption, Outcome.bind_ok] using
        congrArg (fun out => out.bind (fun xs => .ok (b.slot.value false v :: xs))) hxs

def replace (s : Op) (operands : List ValueId) (maps : List AffineMap) : Op :=
  let keep := fun (p : String × Attr) => p.1 != "indexing_maps"
  .mk s.name s.results operands
    ((("indexing_maps", .array (maps.map Attr.affineMap)) :: s.attrs.filter keep).mergeSort
      (fun a b => a.1 ≤ b.1))
    (s.props.filter keep) s.regions s.successors

def Legal (Γ : TEnv) (facts : List UniformFacts.Fact) (s v : Op) (bs : List Input)
    (outIds : List ValueId) (outTys : List VTy) (outMaps : List AffineMap)
    (iters : List IterKind) : Prop :=
  s.name = "linalg.generic" ∧ v.name = "linalg.generic" ∧
  unmodelledAttr? s.blind = none ∧ unmodelledAttr? v.blind = none ∧
  s.operands = bs.map Input.tensor ++ outIds ∧ v.operands = bs.map Input.payload ++ outIds ∧
  attrDenseArray? s "operandSegmentSizes" = some [Int.ofNat bs.length, Int.ofNat outIds.length] ∧
  attrDenseArray? v "operandSegmentSizes" = some [Int.ofNat bs.length, Int.ofNat outIds.length] ∧
  attrAffineMaps? s "indexing_maps" = some (bs.map (fun b => b.slot.map false) ++ outMaps) ∧
  attrAffineMaps? v "indexing_maps" = some (bs.map (fun b => b.slot.map true) ++ outMaps) ∧
  iterKinds? s = some iters ∧ iterKinds? v = some iters ∧
  outIds.mapM (Γ.lookup ·) = some outTys ∧ (∀ ty ∈ outTys, ty.isTensor = true) ∧
  bs.all (fun b => decide (b.legal Γ facts)) = true

instance (Γ : TEnv) (facts : List UniformFacts.Fact) (s v : Op) (bs : List Input)
    (outIds : List ValueId) (outTys : List VTy) (outMaps : List AffineMap) (iters : List IterKind) :
    Decidable (Legal Γ facts s v bs outIds outTys outMaps iters) := by
  unfold Legal; infer_instance

theorem meaning {ib : Nat} {Γ : TEnv} {facts : List UniformFacts.Fact} {s v : Op}
    {bs : List Input} {outIds : List ValueId} {outTys : List VTy} {outMaps : List AffineMap}
    {iters : List IterKind} {ranges : List Nat}
    (h : Legal Γ facts s v bs outIds outTys outMaps iters)
    (hres : s.results = v.results) (hregs : s.regions = v.regions)
    (hl : Rules.SplatScalar.checkLayout (bs.map Input.slot) outMaps iters outTys = some ranges)
    {env : Env} (he : EnvTyped ib Γ env)
    (hf : BindingFacts.Valid UniformFacts.model facts env) (ctx : Ctx) (call : CallSem) :
    stepOp opSem ctx call env v = stepOp opSem ctx call env s := by
  obtain ⟨hsn, hvn, hsm, hvm, hso, hvo, hss, hvs, hsmap, hvmap,
    hsi, hvi, hout, hot, hbs⟩ := h
  obtain ⟨payloads, hp, ht, hi⟩ := input_values he hf bs (by simpa using List.all_eq_true.1 hbs)
  obtain ⟨outValues, ho, hto⟩ := he.getAll _ _ hout
  obtain ⟨outs, rfl⟩ := tensors_of_typed hto hot
  have hiLen : (inputs false (bs.map Input.slot) payloads).length = bs.length := by
    simpa using inputs_length (slots := bs.map Input.slot) (by simpa only [List.map_map, Function.comp_def] using ht) false
  have hpLen : payloads.length = bs.length := by simpa using HasTys.length ht
  have hoLen : (outs.map RVal.tensor).length = outIds.length :=
    (HasTys.length hto).trans (optionMapM_length _ _ _ hout)
  have hsa : env.getAll s.operands = .ok (inputs false (bs.map Input.slot) payloads ++ outs.map RVal.tensor) := by
    rw [hso]
    simp only [Env.getAll, Rules.DequantSite.mapM_append]
    change (env.getAll (bs.map Input.tensor)).bind _ = _
    rw [hi, Outcome.bind_ok]
    change ((env.getAll outIds).bind _) = _
    rw [ho]; rfl
  have hva : env.getAll v.operands = .ok (payloads ++ outs.map RVal.tensor) := by
    rw [hvo]
    simp only [Env.getAll, Rules.DequantSite.mapM_append]
    change (env.getAll (bs.map Input.payload)).bind _ = _
    rw [hp, Outcome.bind_ok]
    change ((env.getAll outIds).bind _) = _
    rw [ho]; rfl
  have hsplitS : splitInsOuts s.blind
      (inputs false (bs.map Input.slot) payloads ++ outs.map RVal.tensor) =
      some (inputs false (bs.map Input.slot) payloads, outs.map RVal.tensor) := by
    rw [splitInsOuts_blind]; simp [splitInsOuts, hss, ← hiLen, hoLen]
  have hsplitT : splitInsOuts v.blind (payloads ++ outs.map RVal.tensor) =
      some (payloads, outs.map RVal.tensor) := by
    rw [splitInsOuts_blind]; simp [splitInsOuts, hvs, ← hpLen, hoLen]
  unfold stepOp
  rw [hsa, hva, Outcome.bind_ok, Outcome.bind_ok, ← hregs, ← hres]
  rw [linalg_sem_of hsm (by rw [hsn]; cbv) (.inl hsn),
    linalg_sem_of hvm (by rw [hvn]; cbv) (.inl hvn)]
  simp only [Linalg.sem, Op.blind_name, hsn, hvn, hsplitS, hsplitT,
    attrAffineMaps?_blind, iterKinds?_blind, hsmap, hvmap, hsi, hvi]
  cases evalRegions opSem call env s.regions with
  | nil => rfl
  | cons body rest =>
    cases rest with
    | cons _ _ => rfl
    | nil =>
      simp only [Op.blind_results, List.length_map]
      have heq := Rules.SplatScalar.checkLayout_sound hl
        (by simpa only [List.map_map, Function.comp_def] using ht) hto ctx body
      simp only [List.map_map, Function.comp_def] at heq
      rw [← heq, hres]

def inline? (Γ : TEnv) (facts : List UniformFacts.Fact) (s : Op) : Option Op := do
  let (ins, outs) ← splitInsOuts s s.operands
  let maps ← attrAffineMaps? s "indexing_maps"
  let iters ← iterKinds? s
  let tys ← ins.mapM (Γ.lookup ·)
  let outTys ← outs.mapM (Γ.lookup ·)
  let bs := ((ins.zip tys).zip (maps.take ins.length)).map fun ((id, ty), map) =>
    select Γ facts id ty map
  let outMaps := maps.drop ins.length
  let v := replace s (bs.map Input.payload ++ outs) (bs.map (fun b => b.slot.map true) ++ outMaps)
  if Legal Γ facts s v bs outs outTys outMaps iters then
    let _ ← Rules.SplatScalar.checkLayout (bs.map Input.slot) outMaps iters outTys
    pure v
  else none

def rewrite (Γ : TEnv) (facts : List UniformFacts.Fact) (s : Op) : Op :=
  (inline? Γ facts s).getD s

theorem rewrite_spec (Γ : TEnv) (facts : List UniformFacts.Fact) (s : Op) :
    (rewrite Γ facts s).isTerminator = s.isTerminator ∧
    (rewrite Γ facts s).results = s.results ∧
    ∀ ib env, EnvTyped ib Γ env → BindingFacts.Valid UniformFacts.model facts env →
      ∀ ctx call, stepOp opSem ctx call env (rewrite Γ facts s) = stepOp opSem ctx call env s := by
  unfold rewrite
  cases hi : inline? Γ facts s with
  | none => exact ⟨rfl, rfl, fun _ _ _ _ _ _ => rfl⟩
  | some v =>
    unfold inline? at hi
    cases hsplit : splitInsOuts s s.operands with
    | none => simp [hsplit] at hi
    | some pair =>
      obtain ⟨ins, outs⟩ := pair
      simp only [hsplit, bind, Option.bind] at hi
      cases hmaps : attrAffineMaps? s "indexing_maps" with
      | none => simp [hmaps] at hi
      | some maps =>
        simp only [hmaps] at hi
        cases hiters : iterKinds? s with
        | none => simp [hiters] at hi
        | some iters =>
          simp only [hiters] at hi
          cases htys : ins.mapM (Γ.lookup ·) with
          | none => simp [htys] at hi
          | some tys =>
            simp only [htys] at hi
            cases hout : outs.mapM (Γ.lookup ·) with
            | none => simp [hout] at hi
            | some outTys =>
              simp only [hout] at hi
              split at hi
              · next hc =>
                cases hl : Rules.SplatScalar.checkLayout
                    (((ins.zip tys).zip (maps.take ins.length)).map fun ((id, ty), map) =>
                      (select Γ facts id ty map).slot) (maps.drop ins.length) iters outTys with
                | none => simp only [List.map_map, Function.comp_def] at hi hl; rw [hl] at hi; cases hi
                | some ranges =>
                  simp only [List.map_map, Function.comp_def] at hi hl
                  rw [hl] at hi
                  simp only [Option.pure_def, Option.some.injEq] at hi
                  subst v
                  refine ⟨rfl, rfl, ?_⟩
                  intro ib env he hf ctx call
                  simpa only [Option.getD_some, List.map_map, Function.comp_def] using
                    meaning hc rfl rfl (by simpa only [List.map_map, Function.comp_def] using hl) he hf ctx call
              · cases hi

/-- Identity correspondence for every currently bound static name. -/
def diagonal (Γ : TEnv) : Rho := Γ.map fun p => (p.1, p.1)

theorem diagonal_refl (Γ : TEnv) (env : Env) : Corr (diagonal Γ) env env := by
  intro x y h
  have he : x = y := by
    induction Γ with
    | nil => simp [diagonal, Rho.find] at h
    | cons p ps ih =>
      simp only [diagonal, List.map_cons, Rho.find, List.lookup] at h
      split at h
      · simp_all only [beq_iff_eq, Option.some.injEq]
      · exact ih h
  rw [he]

/-- A producer is removable only if its payload is actually typed in scope, its result is fresh,
and the entire canonical continuation checks without that result, including nested captures. -/
def drop? (Γ : TEnv) (s : Op) (tail : List Op) : Bool :=
  match UniformFacts.learn s with
  | some f => Γ.lookup f.scalar == some (.scalar f.element) &&
      freshIds (diagonal Γ) [f.tensor] && alphaOps (diagonal Γ) tail tail
  | none => false

theorem drop_sound {ib : Nat} {Γ : TEnv} {s : Op} {tail : List Op}
    (hd : drop? Γ s tail = true) {env : Env} (he : EnvTyped ib Γ env)
    (ctx : Ctx) (call : CallSem) :
    ∃ env', stepOp opSem ctx call env s = .ok env' ∧
      evalOps opSem ctx call env' tail = evalOps opSem ctx call env tail := by
  unfold drop? at hd
  cases hl : UniformFacts.learn s with
  | none => simp [hl] at hd
  | some fact =>
    simp only [hl, Bool.and_eq_true, beq_iff_eq] at hd
    obtain ⟨⟨ht, hfresh⟩, halpha⟩ := hd
    unfold UniformFacts.learn at hl
    split at hl
    · next hm =>
      split at hl
      · next scalar result shape element ha hr hty =>
        split at hl
        · cases hl
          obtain ⟨value, hv, hvt⟩ := he _ _ ht
          cases value with
          | tensor _ => cases hvt
          | scalar value =>
            have hexec := UniformFacts.splat_exec hm.1 hm.2 ha hr hty ctx call env hv
            refine ⟨_, hexec, ?_⟩
            have hc := Rules.DequantSite.Corr.temporary (diagonal_refl Γ env)
              ((freshIds_sound hfresh).2 result.id (by simp))
              (.tensor (Tensor.full shape element value))
            exact (alphaOps_sound call call _ tail tail ctx _ _ halpha hc
              (fun _ _ => rfl)).symm
        · cases hl
      · cases hl
    · cases hl

def normalize (ib : Nat) (Γ : TEnv) (facts : List UniformFacts.Fact) :
    List Op → Option (List Op)
  | [] => some []
  | s :: ss =>
    if s.isTerminator then some (s :: ss) else do
      let _ ← s.operands.mapM (Γ.lookup ·)
      let rs ← s.results.mapM (fun r => vtyOf ib r.ty)
      let tail ← normalize ib ((s.results.map (·.id)).zip rs ++ Γ)
        (UniformFacts.advance facts s) ss
      pure (if drop? Γ s tail then tail else rewrite Γ facts s :: tail)

/-- Exact equality, including target stuck/UB behavior, is essential when normalization is used
on both sides. Source progress supplies actual intermediate types and producer facts. -/
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
                  (UniformFacts.advance facts (.mk name results operands attrs props regions successors)) ss with
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
                    (UniformFacts.advance_sound hf hs) hib hd
                split
                · next hdrop =>
                  obtain ⟨env', hs, heq⟩ := drop_sound hdrop he ctx call
                  rw [evalOps_cons_step _ _ _ _ _ _ ht, hs, Outcome.bind_ok]
                  exact heq.symm.trans (ih env' hs)
                · have hvterm := (rewrite_spec Γ facts
                    (.mk name results operands attrs props regions successors)).1.trans ht
                  rw [evalOps_cons_step _ _ _ _ _ _ ht,
                    evalOps_cons_step _ _ _ _ _ _ hvterm,
                    (rewrite_spec Γ facts (.mk name results operands attrs props regions successors)).2.2
                      ib env he hf ctx call]
                  cases hs : stepOp opSem ctx call env
                      (.mk name results operands attrs props regions successors) with
                  | ub => rfl
                  | stuck _ => rfl
                  | ok env' => exact ih env' hs

end MerlinVerify.Check.SplatNormalize
