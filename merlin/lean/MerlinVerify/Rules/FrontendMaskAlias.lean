/- Same-mask correspondence for the two captured reshape chains, including provenance. -/
import MerlinVerify.Rules.FrontendMaskExecution

namespace MerlinVerify.Rules.FrontendMaskAlias
open MerlinVerify MerlinVerify.Check
open FrontendMaskRuntime
open FrontendBoolSimulation (envRel_bind bind_corr)

def ReshapeFacts (s t : Op) : Prop :=
  (s.name = "tensor.collapse_shape" ∨ s.name = "tensor.expand_shape") ∧ t.name = s.name ∧
  s.results.map (·.ty) = t.results.map (·.ty) ∧
  Sem.Ops.unmodelledAttr? s.blind = none ∧ Sem.Ops.unmodelledAttr? t.blind = none ∧
  Sem.Ops.Tensor.resultTensorTy s.blind = Sem.Ops.Tensor.resultTensorTy t.blind ∧
  Sem.Ops.Tensor.reassociation? s.blind = Sem.Ops.Tensor.reassociation? t.blind
instance (s t : Op) : Decidable (ReshapeFacts s t) := inferInstanceAs (Decidable (_ ∧ _))

def header (s t : Op) : Bool := opHeadEq s t || decide (ReshapeFacts s t)

theorem header_term {s t : Op} (h : header s t = true) : s.isTerminator = t.isTerminator := by
  simp only [header, Bool.or_eq_true, decide_eq_true_eq] at h
  rcases h with h | ⟨hn, ht, _⟩
  · exact isTerminator_of_opHeadEq h
  · rcases hn with hn | hn <;> simp [Op.isTerminator, terminatorNames, ht, hn]

theorem header_length {s t : Op} (h : header s t = true) : s.results.length = t.results.length := by
  simp only [header, Bool.or_eq_true, decide_eq_true_eq] at h
  rcases h with h | ⟨_, _, hr, _⟩
  · exact results_length_of_opHeadEq h
  · simpa using congrArg List.length hr

/-- Equality follows from precisely the fields the frozen reshape interpretation reads. Unknown
attributes are refused on both sides; provenance need not have identical diagnostic strings. -/
theorem header_sem {s t : Op} (h : header s t = true) (limit : Nat)
    (ctx : Ctx) (call : CallSem) (args : List RVal) :
    sem limit ctx call s args [] = sem limit ctx call t args [] := by
  simp only [header, Bool.or_eq_true, decide_eq_true_eq] at h
  rcases h with h | ⟨hn, ht, _, hms, hmt, hr, hg⟩
  · exact sem_blind (blind_of_opHeadEq h)
  · rcases hn with hn | hn
    all_goals
      have htn := ht.trans hn
      have hd : Sem.Ops.dialectOf s.name = "tensor" := by rw [hn]; cbv
      have hdt : Sem.Ops.dialectOf t.name = "tensor" := by rw [htn]; cbv
      rw [tensor_eq limit ctx call s hd (by rw [hn]; decide),
        tensor_eq limit ctx call t hdt (by rw [htn]; decide)]
      simp only [Sem.Ops.opSem, hms, hmt, Sem.Ops.dispatch, Op.blind_name, hn, htn]
      have hc : Sem.Ops.dialectOf "tensor.collapse_shape" = "tensor" := by cbv
      have he : Sem.Ops.dialectOf "tensor.expand_shape" = "tensor" := by cbv
      simp only [hc, he, Sem.Ops.Tensor.sem, Op.blind_name, hn, htn, hr, hg]

/-- Flat actual-program correspondence. All result definitions and uses are recomputed; only
reshape headers have the explicitly proved attribute-footprint comparison above. -/
def ops (ρ : Rho) : List Op → List Op → Bool
  | [], [] => true
  | s :: ss, t :: ts =>
    header s t && decide (s.regions = [] ∧ t.regions = []) &&
      (s.operands.mapM ρ.find == some t.operands) && freshIds ρ (t.results.map (·.id)) &&
      ops ((s.results.map (·.id)).zip (t.results.map (·.id)) ++ ρ) ss ts
  | _, _ => false

theorem ops_empty {ρ : Rho} {s t : List Op} (h : ops ρ s t = true) : s.isEmpty = t.isEmpty := by
  cases s <;> cases t <;> simp_all [ops]

theorem ops_sound (limit : Nat) (ctx : Ctx) (call : CallSem) :
    ∀ (ρ : Rho) (ss ts : List Op) (es et : Env), ops ρ ss ts = true → Corr ρ es et →
      evalOps (sem limit) ctx call es ss = evalOps (sem limit) ctx call et ts
  | _, [], [], _, _, _, _ => rfl
  | ρ, (.mk ns sr so ats ps rs us) :: ss, (.mk nt tr targetOperands att pt rt ut) :: ts, es, et, h, hc => by
    let s : Op := .mk ns sr so ats ps rs us
    let t : Op := .mk nt tr targetOperands att pt rt ut
    simp only [ops, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at h
    obtain ⟨⟨⟨⟨hh, ⟨hrs, hrt⟩⟩, hm⟩, hf⟩, htail⟩ := h
    have ht := header_term hh
    have hl := header_length hh
    have hread := Corr.getAll hc _ _ hm
    dsimp only [Op.operands] at hread
    rw [evalOps.eq_2, evalOps.eq_2]
    by_cases hterm : t.isTerminator = true
    · rw [ite_eq_left (ht.trans hterm), ite_eq_left hterm, ops_empty htail, hread]
    · rw [ite_eq_right (by rwa [ht]), ite_eq_right hterm, hread]
      cases et.getAll targetOperands with
      | ub => rfl
      | stuck reason => rfl
      | ok args =>
        simp only [Outcome.bind_ok]
        have hssem := header_sem hh limit ctx call args
        change rs = [] at hrs
        change rt = [] at hrt
        simp only [hrs, hrt, evalRegions]
        simp only [hrs, hrt] at hssem
        rw [hssem]
        cases sem limit ctx call (.mk nt tr targetOperands att pt [] ut) args [] with
        | ub => rfl
        | stuck reason => rfl
        | ok results =>
          simp only [Outcome.bind_ok]
          exact envRel_bind (bind_corr hc _ _ results hl hf) _ _
            (fun es' et' he => ops_sound limit ctx call _ ss ts es' et' htail he)
  | _, [], _ :: _, _, _, h, _ => by simp [ops] at h
  | _, _ :: _, [], _, _, h, _ => by simp [ops] at h

def check (root : Nat) : Region → Region → Bool
  | .mk [.mk [] left], .mk [.mk [] right] => ops [(root, root)] left right
  | _, _ => false

theorem check_sound {root : Nat} {left right : Region} (h : check root left right = true)
    (limit : Nat) (ctx : Ctx) (call : CallSem) (env : Env) :
    evalRegion (sem limit) ctx call env left [] = evalRegion (sem limit) ctx call env right [] := by
  have hc : Corr [(root, root)] env env := by
    intro x y hxy
    by_cases hx : x = root
    · subst x
      simp [Rho.find, List.lookup] at hxy
      subst y
      rfl
    · simp only [Rho.find, List.lookup, beq_eq_false_iff_ne.mpr hx] at hxy
      cases hxy
  unfold check at h
  split at h
  · next ss ts =>
    simpa [evalRegion, evalBlocks, evalBlock, Env.bindAll] using ops_sound limit ctx call _ ss ts env env h hc
  · simp at h

#print axioms check_sound
end MerlinVerify.Rules.FrontendMaskAlias
