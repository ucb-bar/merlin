/- Shared typed exact-normalizer interface and whole-function comparison. Both actual programs
are checked; normalization equality preserves UB and stuck outcomes on both sides. -/
import MerlinVerify.Check.SplatNormalize
import MerlinVerify.Check.TypedC1

namespace MerlinVerify.Check.SplatNormalizer

open MerlinVerify MerlinVerify.Sem.Ops

/-- A reusable exact block normalizer with explicitly checked types, loop depth and SSA facts. -/
structure Normalizer (ib : Nat) where
  run : Nat → TEnv → List UniformFacts.Fact → List Op → List VTy → Option (List Op)
  sound : ∀ d Γ facts ops normalized yields ctx call env,
    checkOps (merlinBook ib) d Γ ops yields = true →
    run d Γ facts ops yields = some normalized →
    EnvTyped ib Γ env → BindingFacts.Valid UniformFacts.model facts env →
    ctx.indexBits = ib → d ≤ ctx.loopIndex.length →
    evalOps opSem ctx call env normalized = evalOps opSem ctx call env ops

def base (ib : Nat) : Normalizer ib where
  run _ Γ facts ops _ := SplatNormalize.normalize ib Γ facts ops
  sound d Γ facts ops normalized yields ctx call env hc hn he hf hi hd :=
    SplatNormalize.normalize_sound (merlinBook ib) d Γ facts ops normalized yields ctx call env
      hc hn he hf hi hd

def check (ib : Nat) (N : Normalizer ib) (src tgt : Module) (fn : String) (argTys resTys : List VTy) : Bool :=
  match src.func? fn, tgt.func? fn with
  | some fS, some fT =>
    match fS.body, fT.body with
    | .mk [.mk argsS ss], .mk [.mk argsT ts] =>
      let ΓS := (argsS.map (·.id)).zip argTys
      let ΓT := (argsT.map (·.id)).zip argTys
      let ρ := (argsS.map (·.id)).zip (argsT.map (·.id))
      match N.run 0 ΓS [] ss resTys,
          N.run 0 ΓT [] ts resTys with
      | some ns, some nt =>
        !fS.isDecl && !fT.isDecl &&
        argsS.mapM (fun a => vtyOf ib a.ty) == some argTys &&
        argsT.mapM (fun a => vtyOf ib a.ty) == some argTys &&
        freshIds [] (argsT.map (·.id)) &&
        checkOps (merlinBook ib) 0 ΓS ss resTys &&
        checkOps (merlinBook ib) 0 ΓT ts resTys &&
        (calleesOps ns ++ calleesOps nt).all (fun c => isDeclIn src c && isDeclIn tgt c) &&
        alphaOps ρ ns nt
      | _, _ => false
    | _, _ => false
  | _, _ => false

/-- The actual target refines the actual source on the same checked runtime domain. In
particular, acceptance cannot hide a stuck target behind a conditional target normalization. -/
theorem check_sound {ib : Nat} {N : Normalizer ib} {src tgt : Module} {fn : String} {argTys resTys : List VTy}
    (h : check ib N src tgt fn argTys resTys = true) :
    FuncRefinesC opSem (fun ctx => ctx.indexBits = ib) (fun args => HasTys ib args argTys)
      tgt fn src fn := by
  intro ctx hib args hargs
  unfold check at h
  split at h
  · next fS fT hfS hfT =>
    split at h
    · next argsS ss argsT ts hbS hbT =>
      dsimp only at h
      split at h
      · next ns nt hnS hnT =>
        simp only [Bool.and_eq_true, Bool.not_eq_eq_eq_not, Bool.not_true, beq_iff_eq,
          List.all_eq_true] at h
        obtain ⟨⟨⟨⟨⟨⟨⟨⟨hdS, hdT⟩, haS⟩, haT⟩, hfr⟩, hs⟩, ht⟩, hdecl⟩, halpha⟩ := h
        obtain ⟨hnd, _⟩ := freshIds_sound hfr
        have hlenS : argsS.length = args.length :=
          (optionMapM_length _ _ _ haS).symm.trans (HasTys.length hargs).symm
        have hlenT : argsT.length = args.length :=
          (optionMapM_length _ _ _ haT).symm.trans (HasTys.length hargs).symm
        have hΓS : EnvTyped ib ((argsS.map (·.id)).zip argTys) ((argsS.map (·.id)).zip args) := by
          simpa using (envTyped_nil ib).extend (argsS.map (·.id)) args argTys hargs
        have hΓT : EnvTyped ib ((argsT.map (·.id)).zip argTys) ((argsT.map (·.id)).zip args) := by
          simpa using (envTyped_nil ib).extend (argsT.map (·.id)) args argTys hargs
        have hc : Corr ((argsS.map (·.id)).zip (argsT.map (·.id)))
            ((argsS.map (·.id)).zip args) ((argsT.map (·.id)).zip args) := by
          simpa using Corr.extend Corr.nil _ _ args (by simpa using hlenS)
            (by simpa using hlenT) hnd (fun y hy p hp => by simp at hp)
        have hcall : CallAgree (evalFunc opSem ctx src src.funcs.length)
            (evalFunc opSem ctx tgt tgt.funcs.length) (calleesOps ns) := by
          intro c hc
          have hb := hdecl c (List.mem_append_left _ hc)
          rw [evalFunc_decl _ _ _ _ _ (funcs_length_pos hfS) hb.1,
            evalFunc_decl _ _ _ _ _ (funcs_length_pos hfT) hb.2]
        simp only [denote, evalFunc, hfS, hfT, hdS, hdT, Bool.false_eq_true, ↓reduceIte,
          hbS, hbT, evalRegion, evalBlocks, evalBlock]
        rw [bindAll_eq_ok hlenS, bindAll_eq_ok hlenT]
        simp only [Outcome.bind_ok, List.append_nil]
        apply OutLE.refinesC
        apply Or.inr; apply Or.inr
        have heS := N.sound 0 _ _ ss ns resTys ctx
          (evalFunc opSem ctx src src.funcs.length) _ hs hnS hΓS (by intro f hf; cases hf)
          hib (Nat.zero_le _)
        have heT := N.sound 0 _ _ ts nt resTys ctx
          (evalFunc opSem ctx tgt tgt.funcs.length) _ ht hnT hΓT (by intro f hf; cases hf)
          hib (Nat.zero_le _)
        rw [← heS, ← heT]
        exact (alphaOps_sound _ _ _ ns nt ctx _ _ halpha hc hcall).symm
      · cases h
    · cases h
  · cases h

end MerlinVerify.Check.SplatNormalizer
