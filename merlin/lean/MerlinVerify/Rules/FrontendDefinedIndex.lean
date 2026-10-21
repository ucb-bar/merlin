/- Initialized linalg.index values under a checked enclosing iteration frame. -/
import MerlinVerify.Rules.FrontendDefinedFrame

namespace MerlinVerify.Rules.FrontendDefinedIndex
open MerlinVerify MerlinVerify.Check MerlinVerify.Check.Definedness MerlinVerify.Sem.Ops
open FrontendDefinedFrame

def rule (depth : Nat) : Definedness.Rule := fun op fs =>
  if unmodelledAttr? op.blind = none ∧ dialectOf op.name = "linalg" ∧ op.name = "linalg.index" ∧
      fs = [] ∧ op.results.map (·.ty) = [.index] ∧ op.regions = [] then
    match attrInt? op "dim" with
    | some d => if 0 ≤ d ∧ d.toNat < depth then some [⟨.scalar .index, true⟩] else none
    | none => none
  else none

def book (ib depth : Nat) : BookAt opSem ib depth where
  rule _ := rule depth
  sound Γ op fs rs h ctx call env args hib hdepth _ ha := by
    unfold rule at h
    split at h
    · next hc =>
      obtain ⟨hm, hd, hn, rfl, _, _⟩ := hc
      split at h
      · next d hdim =>
        split at h
        · next hc =>
          cases h
          cases ha
          have hlt : d.toNat < ctx.loopIndex.length := by omega
          have hdim' : attrInt? op.blind "dim" = some d := by simpa [attrInt?] using hdim
          have hi : ctx.loopIndex[d.toNat]? = some ctx.loopIndex[d.toNat] := List.getElem?_eq_getElem hlt
          rw [opSem_modelled hm]
          have hd' : dialectOf "linalg.index" = "linalg" := by cbv
          simp only [dispatch, Op.blind_name, hd, hn, hd', beq_self_eq_true, Bool.or_true, ↓reduceIte]
          simp only [Linalg.sem, Op.blind_name, hn]
          simp only [hdim', hi, hc.1, ↓reduceIte]
          refine ⟨_, rfl, .cons ?_ .nil⟩
          constructor
          · simp [RVal.HasTy, Val.HasTy, Scalar.conforms, hib]
          · simp [RVal.Defined]
        · cases h
      · cases h
    · cases h

#print axioms book
end MerlinVerify.Rules.FrontendDefinedIndex

namespace MerlinVerify.Rules.FrontendDefinedIndex
open MerlinVerify MerlinVerify.Check MerlinVerify.Check.Definedness MerlinVerify.Sem.Ops

theorem index_int_fact (ib : Nat) (v : RVal) :
    Fact.Holds ib v ⟨.scalar .index, true⟩ ↔ Fact.Holds ib v ⟨.scalar (.int ib), true⟩ := by
  cases v with
  | tensor _ => simp [Fact.Holds, RVal.HasTy]
  | scalar v =>
    cases v with
    | poison => simp [Fact.Holds, RVal.HasTy, Val.HasTy, RVal.Defined]
    | some s => cases s <;> simp [Fact.Holds, RVal.HasTy, Val.HasTy, Scalar.conforms]

/-- Equal-width signed index casts preserve initialized bits in either direction. -/
def castRule (ib : Nat) : Definedness.Rule := fun op fs =>
  match fs, op.results with
  | [⟨.scalar ty, true⟩], [r] =>
    if dialectOf op.name = "arith" ∧ op.name = "arith.index_cast" ∧
        ((ty = .index ∧ r.ty = .int ib) ∨ (ty = .int ib ∧ r.ty = .index)) then
      some [⟨.scalar r.ty, true⟩]
    else none
  | _, _ => none

theorem castRule_sound (ib : Nat) : Definedness.RuleSound ib (castRule ib) := by
  intro op fs rs hm h ctx call args hib ha
  unfold castRule at h
  split at h
  · next ty r hres =>
    split at h
    · next hc =>
      obtain ⟨hd, hn, hdir⟩ := hc
      cases h
      obtain _ | ⟨hv, hnil⟩ := ha
      cases hnil
      rcases hdir with ⟨rfl, hr⟩ | ⟨rfl, hr⟩
      all_goals
        first | have hv' := (index_int_fact ib _).mp hv | have hv' := hv
        obtain ⟨x, rfl⟩ := initialized_int hv'
        have hresult : Arith.resultElt op.blind = some r.ty := by
          simp [Arith.resultElt, Op.blind_results, hres, hr]
        rw [opSem_modelled hm]
        simp only [dispatch, Op.blind_name, hd]
        simp only [Arith.sem, Op.blind_name, hn, Arith.widthOf, hresult, hr, Ty.bitWidth?, hib,
          Outcome.ofOption, Outcome.bindOp_eq, Outcome.bind_ok, Arith.unOp, Arith.pureN]
        rw [show [.scalar (.some (.bits ib x))] =
          ([.some (.bits ib x)] : List Val).map RVal.scalar from rfl, liftEager_scalars]
        simp only [Arith.intUn, Nat.le_refl, ↓reduceIte, Outcome.bind_ok, Arith.bv]
        exact ⟨_, rfl, .cons ⟨by simp [hr, RVal.HasTy, Val.HasTy, Scalar.conforms],
          by simp [RVal.Defined]⟩ .nil⟩
    · cases h
  · cases h

def castBook (ib : Nat) : Definedness.Book opSem ib where
  rule _ op fs := if unmodelledAttr? op.blind = none ∧ op.regions = [] then castRule ib op fs else none
  sound Γ op fs rs h ctx call env args hib _ ha := by
    split at h
    · next hc =>
      simpa only [hc.2, evalRegions] using castRule_sound ib op fs rs hc.1 h ctx call args hib ha
    · cases h

#print axioms castRule_sound
end MerlinVerify.Rules.FrontendDefinedIndex
