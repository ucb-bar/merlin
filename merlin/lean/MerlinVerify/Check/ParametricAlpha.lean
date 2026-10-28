/- SSA equivalence for explicit program interpretations. The two required semantic laws are
proved by each interpretation, never inferred from tests or a certificate. Operand arity and
region emptiness are retained because the source-contract adapter checks those properties. -/
import MerlinVerify.Check.C1
import MerlinVerify.Check.Progress

namespace MerlinVerify.Check.ParametricAlpha

open MerlinVerify MerlinVerify.Sem.Ops

structure Laws (sem : OpSem) : Prop where
  operation : ∀ {ctx : Ctx} {call : CallSem} {s t : Op} {args : List RVal}
    {regions : List (Ctx → RegionSem)},
    opHeadEq s t = true → s.operands.length = t.operands.length →
    s.regions.isEmpty = t.regions.isEmpty → sem ctx call s args regions = sem ctx call t args regions
  call_local : ∀ {ctx : Ctx} {call call' : CallSem} {op : Op} {args : List RVal}
    {regions : List (Ctx → RegionSem)},
    (∀ c, calleeOf op.blind = some c → call c = call' c) →
    sem ctx call op args regions = sem ctx call' op args regions

theorem alphaRegions_isEmpty {rho : Rho} {left right : List Region}
    (matched : alphaRegions rho left right = true) : left.isEmpty = right.isEmpty := by
  cases left <;> cases right <;> simp_all [alphaRegions]

variable {sem : OpSem}

mutual
  theorem alphaOps_sound (laws : Laws sem) (callS callT : CallSem) :
      ∀ (ρ : Rho) (ss ts : List Op) (ctx : Ctx) (envS envT : Env),
        alphaOps ρ ss ts = true → Corr ρ envS envT → CallAgree callS callT (calleesOps ss) →
        evalOps sem ctx callS envS ss = evalOps sem ctx callT envT ts
    | _, [], [], _, _, _, _, _, _ => by simp [evalOps]
    | ρ, (.mk nS resS opsS aS pS regsS uS) :: ss, (.mk nT resT opsT aT pT regsT uT) :: ts,
        ctx, envS, envT, h, hc, hcall => by
      set s := Op.mk nS resS opsS aS pS regsS uS with hs
      set t := Op.mk nT resT opsT aT pT regsT uT with ht0
      simp only [alphaOps] at h
      split at h
      · next ρ' hop =>
        rw [hs, ht0] at hop
        simp only [alphaOp] at hop
        split at hop
        · next hcond =>
          rw [← hs, ← ht0] at hcond
          simp only [Option.some.injEq] at hop
          subst hop
          simp only [Bool.and_eq_true, beq_iff_eq] at hcond
          obtain ⟨⟨⟨hhead, hmap⟩, hregs⟩, hfresh⟩ := hcond
          have hcallS : CallAgree callS callT (calleesOps ss) := fun c hc' =>
            hcall c (by rw [hs, calleesOps]; exact List.mem_append_right _ hc')
          have hcallR : CallAgree callS callT (calleesRegions regsS) := fun c hc' =>
            hcall c (by rw [hs, calleesOps]
                        exact List.mem_append_left _ (List.mem_append_right _ hc'))
          have hcallOp : ∀ c, calleeOf s.blind = some c → callS c = callT c := fun c hc' =>
            hcall c (by rw [hs, calleesOps]
                        exact List.mem_append_left _ (List.mem_append_left _ (by rw [← hs]; simp [hc'])))
          have hterm := isTerminator_of_opHeadEq hhead
          have harity : s.operands.length = t.operands.length :=
            (optionMapM_length _ _ _ hmap).symm
          have hregions : s.regions.isEmpty = t.regions.isEmpty := alphaRegions_isEmpty hregs
          have hlen := results_length_of_opHeadEq hhead
          obtain ⟨hnd, hfr⟩ := freshIds_sound hfresh
          have hga := Corr.getAll hc opsS opsT hmap
          rw [hs, ht0, evalOps.eq_2, evalOps.eq_2, ← hs, ← ht0]
          by_cases ht : t.isTerminator = true
          · have hs : s.isTerminator = true := by rw [hterm]; exact ht
            rw [ite_eq_left hs, ite_eq_left ht, alphaOps_isEmpty h, hga]
          · have hs : ¬ s.isTerminator = true := by rw [hterm]; exact ht
            rw [ite_eq_right hs, ite_eq_right ht, hga,
              alphaRegions_sound laws callS callT ρ regsS regsT envS envT hregs hc hcallR]
            cases envT.getAll opsT with
            | ub => rfl
            | stuck r => rfl
            | ok args =>
              simp only [Outcome.bind]
              rw [laws.call_local (call' := callT) hcallOp, laws.operation hhead harity hregions]
              cases sem ctx callT t args (evalRegions sem callT envT regsT) with
              | ub => rfl
              | stuck r => rfl
              | ok results =>
                simp only
                have hl : resS.length = resT.length := hlen
                by_cases hk : resS.length = results.length
                · rw [show s.results = resS from rfl, show t.results = resT from rfl,
                    bindAll_eq_ok hk, bindAll_eq_ok (hl ▸ hk)]
                  simp only
                  exact alphaOps_sound laws callS callT _ ss ts ctx _ _ h
                    (Corr.extend hc _ _ results (by simpa using hk) (by simpa [← hl] using hk)
                      hnd hfr) hcallS
                · rw [show s.results = resS from rfl, show t.results = resT from rfl,
                    bindAll_eq_stuck hk, bindAll_eq_stuck (fun e => hk (hl ▸ e))]
        · simp at hop
      · simp at h
    | _, [], _ :: _, _, _, _, h, _, _ => by simp [alphaOps] at h
    | _, _ :: _, [], _, _, _, h, _, _ => by simp [alphaOps] at h

  theorem alphaRegions_sound (laws : Laws sem) (callS callT : CallSem) :
      ∀ (ρ : Rho) (rs rs' : List Region) (envS envT : Env),
        alphaRegions ρ rs rs' = true → Corr ρ envS envT → CallAgree callS callT (calleesRegions rs) →
        evalRegions sem callS envS rs = evalRegions sem callT envT rs'
    | _, [], [], _, _, _, _, _ => rfl
    | ρ, (.mk bs) :: rs, r' :: rs', envS, envT, h, hc, hcall => by
      simp only [alphaRegions, Bool.and_eq_true] at h
      have hr : CallAgree callS callT (calleesBlocks bs) := fun c hc' =>
        hcall c (by simp only [calleesRegions]; exact List.mem_append_left _ hc')
      have hrs : CallAgree callS callT (calleesRegions rs) := fun c hc' =>
        hcall c (by simp only [calleesRegions]; exact List.mem_append_right _ hc')
      simp only [evalRegions, List.cons.injEq]
      exact ⟨funext fun ctx => alphaRegion_sound laws callS callT ρ (.mk bs) r' ctx envS envT h.1 hc hr,
        alphaRegions_sound laws callS callT ρ rs rs' envS envT h.2 hc hrs⟩
    | _, [], _ :: _, _, _, h, _, _ => by simp [alphaRegions] at h
    | _, _ :: _, [], _, _, h, _, _ => by simp [alphaRegions] at h

  theorem alphaRegion_sound (laws : Laws sem) (callS callT : CallSem) :
      ∀ (ρ : Rho) (r r' : Region) (ctx : Ctx) (envS envT : Env),
        alphaRegion ρ r r' = true → Corr ρ envS envT →
        CallAgree callS callT (calleesBlocks r.blocks) →
        evalRegion sem ctx callS envS r = evalRegion sem ctx callT envT r'
    | ρ, .mk [.mk argsS opsS], .mk [.mk argsT opsT], ctx, envS, envT, h, hc, hcall => by
      simp only [alphaRegion, Bool.and_eq_true, beq_iff_eq] at h
      obtain ⟨⟨hty, hfresh⟩, hops⟩ := h
      obtain ⟨hnd, hfr⟩ := freshIds_sound hfresh
      have hl : argsS.length = argsT.length := by simpa using congrArg List.length hty
      have hcallO : CallAgree callS callT (calleesOps opsS) := fun c hc' =>
        hcall c (by simp only [Region.blocks, calleesBlocks]; exact List.mem_append_left _ hc')
      simp only [evalRegion, evalBlocks, evalBlock]
      funext vals
      by_cases hk : argsS.length = vals.length
      · rw [bindAll_eq_ok hk, bindAll_eq_ok (hl ▸ hk)]
        simp only [Outcome.bind]
        exact alphaOps_sound laws callS callT _ opsS opsT ctx _ _ hops
          (Corr.extend hc _ _ vals (by simpa using hk) (by simpa [← hl] using hk) hnd hfr) hcallO
      · rw [bindAll_eq_stuck hk, bindAll_eq_stuck (fun e => hk (hl ▸ e))]
        rfl
    -- Two empty regions: both evaluate to `stuck`.
    | _, .mk [], .mk [], _, _, _, _, _, _ => rfl
    | _, .mk [], .mk (_ :: _ :: _), _, _, _, h, _, _ => by simp [alphaRegion] at h
    | _, .mk (_ :: _ :: _), .mk [], _, _, _, h, _, _ => by simp [alphaRegion] at h
    | _, .mk (_ :: _ :: _), .mk (_ :: _ :: _), _, _, _, h, _, _ => by simp [alphaRegion] at h
    | _, .mk [], .mk [_], _, _, _, h, _, _ => by simp [alphaRegion] at h
    | _, .mk [_], .mk [], _, _, _, h, _, _ => by simp [alphaRegion] at h
    | _, .mk [_], .mk (_ :: _ :: _), _, _, _, h, _, _ => by simp [alphaRegion] at h
    | _, .mk (_ :: _ :: _), .mk [_], _, _, _, h, _, _ => by simp [alphaRegion] at h
end

#print axioms alphaOps_sound
#print axioms alphaRegions_sound
#print axioms alphaRegion_sound

end MerlinVerify.Check.ParametricAlpha
