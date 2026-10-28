/- Complete lexical typing checker. Every nested block and call is checked even if an earlier
operation is known to be unsupported. Successful-result preservation is distinct from progress. -/
import MerlinVerify.Rules.ScalarSquareTypingRules

namespace MerlinVerify.Rules.ScalarSquareTyping
open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops

def RegionPreserving (ib : Nat) (region : Ctx → RegionSem) (sig : RegionSig) : Prop :=
  ∀ ctx, ctx.indexBits = ib → ∀ args,
    HasTys ib args sig.args → ∀ results, region ctx args = .ok results → HasTys ib results sig.yields

mutual
  def opsCheck (ib : Nat) (signatures : Signatures) (d : Nat) (Γ : TEnv) : List Op → List VTy → Bool
    | [], _ => false
    | op@(.mk _ results operands _ _ regions successors) :: rest, yields =>
      match operands.mapM (fun x => Γ.lookup x) with
      | none => false
      | some ats =>
        if op.isTerminator then rest.isEmpty && decide (results = [] ∧ regions = [] ∧ successors = []) && ats == yields
        else match results.mapM (fun r => vtyOf ib r.ty) with
          | none => false
          | some rts => match typingRule ib signatures d op ats rts with
            | none => false
            | some sigs => regionsCheck ib signatures Γ regions sigs &&
              opsCheck ib signatures d ((results.map (·.id)).zip rts ++ Γ) rest yields
  def regionsCheck (ib : Nat) (signatures : Signatures) (Γ : TEnv) : List Region → List RegionSig → Bool
    | [], [] => true
    | r :: rs, sig :: sigs => regionCheck ib signatures Γ r sig && regionsCheck ib signatures Γ rs sigs
    | _, _ => false
  def regionCheck (ib : Nat) (signatures : Signatures) (Γ : TEnv) : Region → RegionSig → Bool
    | .mk [.mk args ops], sig =>
      args.mapM (fun a => vtyOf ib a.ty) == some sig.args &&
        opsCheck ib signatures sig.depth ((args.map (·.id)).zip sig.args ++ Γ) ops sig.yields
    | _, _ => false
end

theorem preserving_pairs {ib : Nat} {regions : List (Ctx → RegionSem)} {sigs : List RegionSig}
    (h : List.Forall₂ (RegionPreserving ib) regions sigs) (M : FloatModel) :
    List.Forall₂ (RegionRelated M ib) (regions.map fun r => (r, r)) sigs := by
  induction h with
  | nil => exact .nil
  | cons h _ ih => exact .cons (fun ctx _ hib hd args ha => ⟨rfl, h ctx hib args ha⟩) ih

mutual
  theorem opsCheck_preserves (ib : Nat) (signatures : Signatures) (d : Nat) :
      ∀ Γ ops yields ctx call env,
        opsCheck ib signatures d Γ ops yields = true → EnvTyped ib Γ env →
        ctx.indexBits = ib → CallTyped ib signatures call →
        ∀ results, evalOps opSem ctx call env ops = .ok results → HasTys ib results yields
    | _, [], _, _, _, _, h, _, _, _ => by cases h
    | Γ, (.mk n results operands attrs props regions successors) :: rest, yields,
        ctx, call, env, h, he, hib, hc => by
      simp only [opsCheck] at h
      split at h
      · cases h
      · next ats hats =>
        obtain ⟨args, hargs, ha⟩ := he.getAll operands ats hats
        rw [evalOps.eq_2]
        split
        · next ht =>
          rw [ite_eq_left ht] at h
          simp only [Bool.and_eq_true, List.isEmpty_iff, decide_eq_true_eq, beq_iff_eq] at h
          obtain ⟨⟨rfl, _⟩, rfl⟩ := h
          simp only [List.isEmpty_nil, ↓reduceIte, hargs]
          intro rs eq
          cases eq
          exact ha
        · next ht =>
          rw [ite_eq_right ht] at h
          split at h
          · cases h
          · next rts hrts =>
            split at h
            · cases h
            · next sigs hrule =>
              simp only [Bool.and_eq_true] at h
              have hb := regionsCheck_preserves ib signatures Γ regions sigs call env h.1 he hc
              have hs := typingRule_sound hrule ctx call call args
                ((evalRegions opSem call env regions).map fun r => (r, r)) hib ha
                (preserving_pairs hb ctx.M) (fun _ _ => True) (fun _ _ _ _ _ => trivial)
                (fun _ _ _ => rfl) hc
              simp only [List.map_map, Function.comp_def, List.map_id_fun'] at hs
              rw [hargs, Outcome.bind_ok]
              cases hop : opSem ctx call (.mk n results operands attrs props regions successors) args
                  (evalRegions opSem call env regions) with
              | ub => intro _ impossible; cases impossible
              | stuck _ => intro _ impossible; cases impossible
              | ok values =>
                have hv := hs.2 values hop
                have hlen : results.length = values.length := by
                  rw [← optionMapM_length _ _ _ hrts]; exact (HasTys.length hv).symm
                simp only [Outcome.bind_ok, Op.results]
                rw [bindAll_ok hlen, Outcome.bind_ok]
                exact opsCheck_preserves ib signatures d _ rest yields ctx call _ h.2
                  (he.extend _ values rts hv) hib hc

  theorem regionsCheck_preserves (ib : Nat) (signatures : Signatures) :
      ∀ Γ regions sigs call env, regionsCheck ib signatures Γ regions sigs = true →
        EnvTyped ib Γ env → CallTyped ib signatures call →
        List.Forall₂ (RegionPreserving ib) (evalRegions opSem call env regions) sigs
    | _, [], [], _, _, _, _, _ => .nil
    | Γ, r :: rs, sig :: sigs, call, env, h, he, hc => by
      simp only [regionsCheck, Bool.and_eq_true] at h
      exact .cons (regionCheck_preserves ib signatures Γ r sig call env h.1 he hc)
        (regionsCheck_preserves ib signatures Γ rs sigs call env h.2 he hc)
    | _, [], _ :: _, _, _, h, _, _ => by cases h
    | _, _ :: _, [], _, _, h, _, _ => by cases h

  theorem regionCheck_preserves (ib : Nat) (signatures : Signatures) :
      ∀ Γ region sig call env, regionCheck ib signatures Γ region sig = true →
        EnvTyped ib Γ env → CallTyped ib signatures call →
        RegionPreserving ib (fun ctx => evalRegion opSem ctx call env region) sig
    | Γ, .mk [.mk args ops], sig, call, env, h, he, hc => by
      simp only [regionCheck, Bool.and_eq_true, beq_iff_eq] at h
      intro ctx hib values hv
      have hlen : args.length = values.length := by
        rw [← optionMapM_length _ _ _ h.1]; exact (HasTys.length hv).symm
      simp only [evalRegion, evalBlocks, evalBlock]
      rw [bindAll_ok hlen, Outcome.bind_ok]
      exact opsCheck_preserves ib signatures sig.depth _ ops sig.yields ctx call _ h.2
        (he.extend _ values sig.args hv) hib hc
    | _, .mk [], _, _, _, h, _, _ => by cases h
    | _, .mk ((.mk _ _) :: _ :: _), _, _, _, h, _, _ => by cases h
end

end MerlinVerify.Rules.ScalarSquareTyping
