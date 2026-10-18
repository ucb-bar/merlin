/- Exact typed normalization of both actual programs, followed by the universal C1 script checker.
Neither program is replaced during export; independent equality proofs justify both normal forms. -/
import MerlinVerify.Check.TypedNormalizer
import MerlinVerify.Check.TypedC1

namespace MerlinVerify.Check.TypedNormalizeC1

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops

def check (E : TypedNormalizer) (ib : Nat) (src tgt : Module) (fn : String)
    (argTys resTys : List VTy) : Bool :=
  match src.func? fn, tgt.func? fn with
  | some fS, some fT =>
    match fS.body, fT.body with
    | .mk [.mk argsS ss], .mk [.mk argsT ts] =>
      let Γ := (argsS.map (·.id)).zip argTys
      let ΓT := (argsT.map (·.id)).zip argTys
      let ρ := (argsS.map (·.id)).zip (argsT.map (·.id))
      match E.run ib 0 Γ ss ts resTys, E.run ib 0 ΓT ts ss resTys with
      | some ns, some nt =>
        !fS.isDecl && !fT.isDecl && argsS.map (·.ty) == argsT.map (·.ty) &&
        argsS.mapM (fun a => vtyOf ib a.ty) == some argTys && freshIds [] (argsT.map (·.id)) &&
        checkOps (merlinBook ib) 0 Γ ss resTys &&
        checkOps (merlinBook ib) 0 ΓT ts resTys &&
        (calleesOps ns ++ calleesOps nt).all (fun c => isDeclIn src c && isDeclIn tgt c) &&
        runScript ρ (synthLocal ρ ns nt) ns nt
      | _, _ => false
    | _, _ => false
  | _, _ => false

/-- Every successful typed check refines the original source on the declared runtime domain.
Runtime ranks/widths are justified by source execution, not by trusting a rewrite certificate. -/
theorem check_sound (E : TypedNormalizer) {ib : Nat} {src tgt : Module} {fn : String}
    {argTys resTys : List VTy} (h : check E ib src tgt fn argTys resTys = true) :
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
      · next ns nt hn hnT =>
        simp only [Bool.and_eq_true, Bool.not_eq_eq_eq_not, Bool.not_true, beq_iff_eq,
          List.all_eq_true] at h
        obtain ⟨⟨⟨⟨⟨⟨⟨⟨hdS, hdT⟩, hty⟩, harg⟩, hfresh⟩, hs⟩, ht⟩, hdecl⟩, hrun⟩ := h
        obtain ⟨hnd, _⟩ := freshIds_sound hfresh
        have hlen : argsS.length = args.length :=
          (optionMapM_length _ _ _ harg).symm.trans (HasTys.length hargs).symm
        have hST : argsS.length = argsT.length := by simpa using congrArg List.length hty
        have hcall : CallAgree (evalFunc opSem ctx src src.funcs.length)
            (evalFunc opSem ctx tgt tgt.funcs.length) (calleesOps ns ++ calleesOps nt) := by
          intro c hc
          have hb := hdecl c hc
          rw [evalFunc_decl _ _ _ _ _ (funcs_length_pos hfS) hb.1,
            evalFunc_decl _ _ _ _ _ (funcs_length_pos hfT) hb.2]
        simp only [denote, evalFunc, hfS, hfT, hdS, hdT, Bool.false_eq_true, ↓reduceIte,
          hbS, hbT, evalRegion, evalBlocks, evalBlock]
        rw [bindAll_eq_ok hlen, bindAll_eq_ok (hST ▸ hlen)]
        simp only [Outcome.bind_ok, List.append_nil]
        apply OutLE.refinesC
        have hΓ : EnvTyped ib ((argsS.map (·.id)).zip argTys) ((argsS.map (·.id)).zip args) := by
          simpa using (envTyped_nil ib).extend (argsS.map (·.id)) args argTys hargs
        have hΓT : EnvTyped ib ((argsT.map (·.id)).zip argTys) ((argsT.map (·.id)).zip args) := by
          simpa using (envTyped_nil ib).extend (argsT.map (·.id)) args argTys hargs
        rw [← E.sound ib 0 _ ss ts ns resTys ctx _ _ hs hn hΓ hib (Nat.zero_le _),
          ← E.sound ib 0 _ ts ss nt resTys ctx _ _ ht hnT hΓT hib (Nat.zero_le _)]
        exact runScript_sound _ _ ctx _ _ ns nt _ _ hrun
          (by simpa using Corr.extend Corr.nil _ _ args (by simpa using hlen)
                (by simpa [← hST] using hlen) hnd (fun y hy p hp => by simp at hp)) hcall
      · cases h
    · cases h
  · cases h

end MerlinVerify.Check.TypedNormalizeC1
