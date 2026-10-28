/- Strict refinement at the return boundary, without an exact alias for a raw tensor. -/
import MerlinVerify.Check.IdentityReturnSyntax
import MerlinVerify.Check.C1

namespace MerlinVerify.Check.IdentityReturnProof

open MerlinVerify MerlinVerify.Sem.Ops MerlinVerify.Sem.Ops.Linalg
open IdentityReturnSyntax

private theorem getAll_cons {env : Env} {id : ValueId} {ids : List ValueId}
    {value : RVal} {values : List RVal}
    (hv : env.get? id = some value) (hs : env.getAll ids = .ok values) :
    env.getAll (id :: ids) = .ok (value :: values) := by
  change ((Outcome.ofOption _ (env.get? id)).bind fun v =>
    (env.getAll ids).bind fun vs => .ok (v :: vs)) = _
  simp [hv, hs, Outcome.ofOption]

theorem selected_value {ib : Nat} {Γ : TEnv} {ρ : Rho} {result input id target : ValueId}
    {envS envT : Env} {original copied : Tensor}
    (hs : returnId Γ ρ result input id = some target)
    (he : EnvTyped ib Γ envS) (hc : Corr ρ envS envT)
    (hi : envS.get? input = some (.tensor original)) (hr : TensorRefines original copied) :
    ∃ sourceValue targetValue,
      Env.get? ((result, RVal.tensor copied) :: envS) id = some sourceValue ∧
      envT.get? target = some targetValue ∧ RValRefines targetValue sourceValue := by
  by_cases hx : id = result
  · subst id
    simp only [returnId, ↓reduceIte] at hs
    refine ⟨.tensor copied, .tensor original, ?_, ?_, hr⟩
    · simp [Env.get?, List.lookup]
    · exact (hc input target hs).symm.trans hi
  · simp only [returnId, hx, ↓reduceIte] at hs
    cases hl : Γ.lookup id with
    | none => simp [hl] at hs
    | some ty =>
      simp only [hl, Option.isSome_some, ↓reduceIte] at hs
      obtain ⟨value, hv, _⟩ := he id ty hl
      refine ⟨value, value, ?_, (hc id target hs).symm.trans hv, RValRefines.refl value⟩
      have hb : (id == result) = false := by simpa using hx
      simpa [Env.get?, List.lookup, hb] using hv

theorem selected_values {ib : Nat} {Γ : TEnv} {ρ : Rho} {result input : ValueId}
    {envS envT : Env} {original copied : Tensor}
    (he : EnvTyped ib Γ envS) (hc : Corr ρ envS envT)
    (hi : envS.get? input = some (.tensor original)) (hr : TensorRefines original copied) :
    ∀ (ids targets : List ValueId), ids.mapM (returnId Γ ρ result input) = some targets →
      ∃ sourceValues targetValues,
        Env.getAll ((result, RVal.tensor copied) :: envS) ids = .ok sourceValues ∧
        envT.getAll targets = .ok targetValues ∧ ResultsRefine targetValues sourceValues
  | [], targets, h => by
    simp only [List.mapM_nil, Option.pure_def, Option.some.injEq] at h
    subst targets
    exact ⟨[], [], rfl, rfl, .nil⟩
  | id :: ids, targets, h => by
    simp only [List.mapM_cons] at h
    cases hv : returnId Γ ρ result input id with
    | none => simp [hv] at h
    | some target =>
      cases hs : ids.mapM (returnId Γ ρ result input) with
      | none => simp [hv, hs] at h
      | some rest =>
        simp only [hv, hs, bind, Option.bind, Option.pure_def, Option.some.injEq] at h
        subst targets
        obtain ⟨vs, vt, hvs, hvt, href⟩ := selected_values he hc hi hr ids rest hs
        obtain ⟨v, w, hsource, htarget, hvalue⟩ := selected_value hv he hc hi hr
        exact ⟨v :: vs, w :: vt, getAll_cons hsource hvs, getAll_cons htarget hvt,
          .cons hvalue href⟩

/-- The source operation succeeds on actual typed tensors, with the destination's metadata.
Only its canonical result is related pointwise to the actual original input. -/
theorem generic_exec {ib : Nat} {Γ : TEnv} {s : Op} {g : Generic}
    (hl : Legal ib Γ s g) (hregions : s.regions = [g.region])
    (ctx : Ctx) (call : CallSem) (env : Env) (he : EnvTyped ib Γ env) :
    ∃ input copied : Tensor, env.get? g.input = some (.tensor input) ∧
      stepOp opSem ctx call env s = .ok ((g.result.id, .tensor copied) :: env) ∧
      TensorRefines input copied := by
  obtain ⟨hn, hm, ha, hr, _, hseg, hmaps, hiters, _, hinput, hdestination,
    _, _, _, _, hyield, hyieldOps⟩ := hl
  obtain ⟨input, hInput, htInput⟩ := he _ _ hinput
  cases input with
  | scalar _ => cases htInput
  | tensor input =>
    obtain ⟨init, hInit, htInit⟩ := he _ _ hdestination
    cases init with
    | scalar _ => cases htInit
    | tensor init =>
      let copied := Tensor.canon g.shape init.elt input.get
      have hloop := Rules.IdentityReturn.copy_loop ctx call env g.shape input init
        g.arg g.old g.term htInput.1 htInit.1 hyield.1 hyieldOps
      have hd : dialectOf s.name = "linalg" := by rw [hn]; cbv
      have hseg' : attrDenseArray? s.blind "operandSegmentSizes" = some [1, 1] := by
        simpa [attrDenseArray?, Op.blind_attr?] using hseg
      have hsem : opSem ctx call s [.tensor input, .tensor init]
          (evalRegions opSem call env s.regions) = .ok [.tensor copied] := by
        rw [hregions]
        simp only [evalRegions]
        rw [linalg_sem_of hm hd (.inl hn)]
        simp [Linalg.sem, Op.blind_name, hn, splitInsOuts, hseg', attrAffineMaps?_blind,
          hmaps, iterKinds?_blind, hiters, Op.blind_results, hr, Generic.region, copied, hloop]
      have hargs : env.getAll s.operands = .ok [.tensor input, .tensor init] := by
        simp [ha, Env.getAll, Outcome.mapM, Outcome.ofOption, hInput, hInit]
      refine ⟨input, copied, hInput, ?_, ?_⟩
      · simp [stepOp, hargs, hsem, Env.bindAll, hr]
      · exact Rules.IdentityReturn.input_refines_copy input g.shape init.elt htInput.1
          (htInput.2.1.trans htInit.2.1.symm)

theorem block_refines {ib : Nat} {Γ : TEnv} {ρ : Rho} {s sourceRet targetRet : Op} {g : Generic}
    (hl : Legal ib Γ s g) (hregions : s.regions = [g.region])
    (hs : DefaultTerm "func.return" sourceRet) (ht : DefaultTerm "func.return" targetRet)
    (hids : sourceRet.operands.mapM (returnId Γ ρ g.result.id g.input) = some targetRet.operands)
    (ctx : Ctx) (callS callT : CallSem) (envS envT : Env)
    (he : EnvTyped ib Γ envS) (hc : Corr ρ envS envT) :
    Refines ResultsRefine (evalOps opSem ctx callT envT [targetRet])
      (evalOps opSem ctx callS envS [s, sourceRet]) := by
  obtain ⟨input, copied, hinput, hstep, href⟩ := generic_exec hl hregions ctx callS envS he
  obtain ⟨vs, vt, hvs, hvt, hvalues⟩ := selected_values he hc hinput href _ _ hids
  have hn : s.isTerminator = false := by simp [Op.isTerminator, terminatorNames, hl.1]
  have hsr : sourceRet.isTerminator = true := by simp [Op.isTerminator, terminatorNames, hs.1]
  have htr : targetRet.isTerminator = true := by simp [Op.isTerminator, terminatorNames, ht.1]
  rw [evalOps_cons_step _ _ _ _ _ _ hn, hstep, Outcome.bind_ok,
    evalOps_cons_term _ _ _ _ _ _ hsr, evalOps_cons_term _ _ _ _ _ _ htr]
  simpa only [List.isEmpty_nil, ↓reduceIte, hvs, hvt, Refines] using hvalues

end MerlinVerify.Check.IdentityReturnProof
