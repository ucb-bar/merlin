/-
Alpha-equivalence: two op lists that are the same program up to a consistent renaming of SSA
values evaluate identically (design §5.1).

`alphaOps ρ ss ts` checks, op by op: same name, attributes, properties, successors and result types;
operands map through `ρ`; regions alpha-equivalent (block arguments paired); target result ids
fresh (distinct and not already an image of `ρ`). The soundness theorem `alphaOps_sound` is stated
for Merlin's op semantics, which is identifier-blind and call-local by construction
(`Sem.Ops.opSem_blind`, `opSem_call_local`); calls agree on every callee the program names.
-/
import MerlinVerify.Sem.Ops
import MerlinVerify.Check.Corr
import MerlinVerify.Check.ProgramLemmas

namespace MerlinVerify.Check

open MerlinVerify MerlinVerify.Sem.Ops

/-- Distinct ids, none already an image of `ρ`. -/
def freshIds (ρ : Rho) (ys : List ValueId) : Bool :=
  decide ys.Nodup && ys.all (fun y => ρ.all (fun p => p.2 != y))

theorem freshIds_sound {ρ : Rho} {ys : List ValueId} (h : freshIds ρ ys = true) :
    ys.Nodup ∧ ∀ y ∈ ys, Rho.notImage ρ y := by
  simp only [freshIds, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true, bne_iff_ne,
    ne_eq] at h
  exact ⟨h.1, fun y hy p hp => h.2 y hy p hp⟩

/-- Everything an op's meaning can see, except operands and regions, agrees. -/
def opHeadEq (s t : Op) : Bool :=
  s.name == t.name && decide (s.attrs = t.attrs) && decide (s.props = t.props) &&
    s.successors == t.successors && decide (s.results.map (·.ty) = t.results.map (·.ty))

theorem blind_of_opHeadEq {s t : Op} (h : opHeadEq s t = true) : s.blind = t.blind := by
  simp only [opHeadEq, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at h
  obtain ⟨⟨⟨⟨hn, ha⟩, hp⟩, hs⟩, hr⟩ := h
  have hres : s.results.map (fun r => { r with id := 0 }) = t.results.map (fun r => { r with id := 0 }) := by
    have e : ∀ l : List Value, l.map (fun r => { r with id := 0 }) = (l.map (·.ty)).map (fun ty => ({ id := 0, ty } : Value)) := by
      intro l; induction l with
      | nil => rfl
      | cons a l ih => simp [ih]
    rw [e, e, hr]
  simp only [Op.blind, hn, ha, hp, hs, hres]

theorem isTerminator_of_opHeadEq {s t : Op} (h : opHeadEq s t = true) : s.isTerminator = t.isTerminator := by
  simp only [opHeadEq, Bool.and_eq_true, beq_iff_eq] at h
  simp only [Op.isTerminator, h.1.1.1.1]

theorem results_length_of_opHeadEq {s t : Op} (h : opHeadEq s t = true) :
    s.results.length = t.results.length := by
  simp only [opHeadEq, Bool.and_eq_true, decide_eq_true_eq] at h
  have := congrArg List.length h.2
  simpa using this

mutual
  def alphaOps (ρ : Rho) : List Op → List Op → Bool
    | [], [] => true
    | s :: ss, t :: ts =>
      match alphaOp ρ s t with
      | some ρ' => alphaOps ρ' ss ts
      | none => false
    | _, _ => false

  def alphaOp (ρ : Rho) : Op → Op → Option Rho
    | s@(.mk _ resS opsS _ _ regsS _), t@(.mk _ resT opsT _ _ regsT _) =>
      if opHeadEq s t && opsS.mapM ρ.find == some opsT && alphaRegions ρ regsS regsT &&
          freshIds ρ (resT.map (·.id))
      then some ((resS.map (·.id)).zip (resT.map (·.id)) ++ ρ) else none

  def alphaRegions (ρ : Rho) : List Region → List Region → Bool
    | [], [] => true
    | r :: rs, r' :: rs' => alphaRegion ρ r r' && alphaRegions ρ rs rs'
    | _, _ => false

  def alphaRegion (ρ : Rho) : Region → Region → Bool
    | .mk [.mk argsS opsS], .mk [.mk argsT opsT] =>
      argsS.map (·.ty) == argsT.map (·.ty) && freshIds ρ (argsT.map (·.id)) &&
        alphaOps ((argsS.map (·.id)).zip (argsT.map (·.id)) ++ ρ) opsS opsT
    | .mk bs, .mk bs' => bs.length != 1 && bs'.length != 1
end

mutual
  /-- Every function symbol an op list may call (including inside nested regions). -/
  def calleesOps : List Op → List String
    | [] => []
    | (op@(.mk _ _ _ _ _ regs _)) :: rest =>
      (calleeOf op.blind).toList ++ calleesRegions regs ++ calleesOps rest

  def calleesRegions : List Region → List String
    | [] => []
    | (.mk bs) :: rs => calleesBlocks bs ++ calleesRegions rs

  def calleesBlocks : List Block → List String
    | [] => []
    | (.mk _ ops) :: bs => calleesOps ops ++ calleesBlocks bs
end

/-- Two call meanings agree on a list of callees. -/
def CallAgree (callS callT : CallSem) (names : List String) : Prop :=
  ∀ c ∈ names, callS c = callT c

theorem alphaOps_isEmpty {ρ : Rho} {ss ts : List Op} (h : alphaOps ρ ss ts = true) :
    ss.isEmpty = ts.isEmpty := by
  cases ss <;> cases ts <;> simp_all [alphaOps]

theorem evalBlocks_not_single (sem : OpSem) (ctx : Ctx) (call : CallSem) (env : Env)
    (bs : List Block) (h : bs.length ≠ 1) :
    evalBlocks sem ctx call env bs = fun _ => .stuck "only single-block regions are supported" := by
  match bs, h with
  | [], _ => rfl
  | [_], h => simp at h
  | _ :: _ :: _, _ => rfl

theorem bindAll_eq_stuck {env : Env} {vs : List Value} {rs : List RVal} (h : vs.length ≠ rs.length) :
    env.bindAll vs rs = .stuck "arity mismatch binding SSA values" := by
  simp [Env.bindAll, h]

theorem bindAll_eq_ok {env : Env} {vs : List Value} {rs : List RVal} (h : vs.length = rs.length) :
    env.bindAll vs rs = .ok ((vs.map (·.id)).zip rs ++ env) := by
  simp [Env.bindAll, h]

mutual
  theorem alphaOps_sound (callS callT : CallSem) :
      ∀ (ρ : Rho) (ss ts : List Op) (ctx : Ctx) (envS envT : Env),
        alphaOps ρ ss ts = true → Corr ρ envS envT → CallAgree callS callT (calleesOps ss) →
        evalOps opSem ctx callS envS ss = evalOps opSem ctx callT envT ts
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
          have hblind := blind_of_opHeadEq hhead
          have hlen := results_length_of_opHeadEq hhead
          obtain ⟨hnd, hfr⟩ := freshIds_sound hfresh
          have hga := Corr.getAll hc opsS opsT hmap
          rw [hs, ht0, evalOps.eq_2, evalOps.eq_2, ← hs, ← ht0]
          by_cases ht : t.isTerminator = true
          · have hs : s.isTerminator = true := by rw [hterm]; exact ht
            rw [if_pos hs, if_pos ht, alphaOps_isEmpty h, hga]
          · have hs : ¬ s.isTerminator = true := by rw [hterm]; exact ht
            rw [if_neg hs, if_neg ht, hga,
              alphaRegions_sound callS callT ρ regsS regsT envS envT hregs hc hcallR]
            cases envT.getAll opsT with
            | ub => rfl
            | stuck r => rfl
            | ok args =>
              simp only [Outcome.bind]
              rw [opSem_call_local (call' := callT) hcallOp, opSem_blind (op' := t) hblind]
              cases opSem ctx callT t args (evalRegions opSem callT envT regsT) with
              | ub => rfl
              | stuck r => rfl
              | ok results =>
                simp only
                have hl : resS.length = resT.length := hlen
                by_cases hk : resS.length = results.length
                · rw [show s.results = resS from rfl, show t.results = resT from rfl,
                    bindAll_eq_ok hk, bindAll_eq_ok (hl ▸ hk)]
                  simp only
                  exact alphaOps_sound callS callT _ ss ts ctx _ _ h
                    (Corr.extend hc _ _ results (by simpa using hk) (by simpa [← hl] using hk)
                      hnd hfr) hcallS
                · rw [show s.results = resS from rfl, show t.results = resT from rfl,
                    bindAll_eq_stuck hk, bindAll_eq_stuck (fun e => hk (hl ▸ e))]
        · simp at hop
      · simp at h
    | _, [], _ :: _, _, _, _, h, _, _ => by simp [alphaOps] at h
    | _, _ :: _, [], _, _, _, h, _, _ => by simp [alphaOps] at h

  theorem alphaRegions_sound (callS callT : CallSem) :
      ∀ (ρ : Rho) (rs rs' : List Region) (envS envT : Env),
        alphaRegions ρ rs rs' = true → Corr ρ envS envT → CallAgree callS callT (calleesRegions rs) →
        evalRegions opSem callS envS rs = evalRegions opSem callT envT rs'
    | _, [], [], _, _, _, _, _ => rfl
    | ρ, (.mk bs) :: rs, r' :: rs', envS, envT, h, hc, hcall => by
      simp only [alphaRegions, Bool.and_eq_true] at h
      have hr : CallAgree callS callT (calleesBlocks bs) := fun c hc' =>
        hcall c (by simp only [calleesRegions]; exact List.mem_append_left _ hc')
      have hrs : CallAgree callS callT (calleesRegions rs) := fun c hc' =>
        hcall c (by simp only [calleesRegions]; exact List.mem_append_right _ hc')
      simp only [evalRegions, List.cons.injEq]
      exact ⟨funext fun ctx => alphaRegion_sound callS callT ρ (.mk bs) r' ctx envS envT h.1 hc hr,
        alphaRegions_sound callS callT ρ rs rs' envS envT h.2 hc hrs⟩
    | _, [], _ :: _, _, _, h, _, _ => by simp [alphaRegions] at h
    | _, _ :: _, [], _, _, h, _, _ => by simp [alphaRegions] at h

  theorem alphaRegion_sound (callS callT : CallSem) :
      ∀ (ρ : Rho) (r r' : Region) (ctx : Ctx) (envS envT : Env),
        alphaRegion ρ r r' = true → Corr ρ envS envT →
        CallAgree callS callT (calleesBlocks r.blocks) →
        evalRegion opSem ctx callS envS r = evalRegion opSem ctx callT envT r'
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
        exact alphaOps_sound callS callT _ opsS opsT ctx _ _ hops
          (Corr.extend hc _ _ vals (by simpa using hk) (by simpa [← hl] using hk) hnd hfr) hcallO
      · rw [bindAll_eq_stuck hk, bindAll_eq_stuck (fun e => hk (hl ▸ e))]
        rfl
    -- Not both single-block: `alphaRegion` demands neither is, and both evaluate to `stuck`.
    | _, .mk [], .mk [], _, _, _, _, _, _ => rfl
    | _, .mk [], .mk (_ :: _ :: _), _, _, _, _, _, _ => rfl
    | _, .mk (_ :: _ :: _), .mk [], _, _, _, _, _, _ => rfl
    | _, .mk (_ :: _ :: _), .mk (_ :: _ :: _), _, _, _, _, _, _ => rfl
    | _, .mk [], .mk [_], _, _, _, h, _, _ => by simp [alphaRegion] at h
    | _, .mk [_], .mk [], _, _, _, h, _, _ => by simp [alphaRegion] at h
    | _, .mk [_], .mk (_ :: _ :: _), _, _, _, h, _, _ => by simp [alphaRegion] at h
    | _, .mk (_ :: _ :: _), .mk [_], _, _, _, h, _, _ => by simp [alphaRegion] at h
end

end MerlinVerify.Check
