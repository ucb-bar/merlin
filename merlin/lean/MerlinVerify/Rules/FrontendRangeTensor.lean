/- Static tensor views and splats transport proved element intervals. -/
import MerlinVerify.Rules.FrontendRangeGeneric

namespace MerlinVerify.Rules.FrontendRangeTensor
open MerlinVerify MerlinVerify.Check Sem.Ops
open FrontendTensorRanges

def splatRule (_ : FEnv) (op : Op) (fs : List Fact) : Option (List Fact) :=
  if unmodelledAttr? op.blind = none ∧ op.regions = [] ∧ dialectOf op.name = "tensor" ∧ op.name = "tensor.splat" then
    match fs, Tensor.resultTensorTy op with
    | [⟨⟨.scalar t, true⟩, info⟩], some (sh, e) =>
      if t = e then some [⟨⟨.tensor sh e, true⟩, info⟩] else none
    | _, _ => none
  else none

def splatBook (ib : Nat) : Definedness.Refined.Book model opSem ib where
  rule := splatRule
  sound Γ op fs rs h ctx call env args hib _ ha := by
    unfold splatRule at h
    split at h
    · next hc =>
      obtain ⟨hm, hregs, hd, hn⟩ := hc
      split at h
      · next _ _ _ scalarTy info shape elt hr =>
        split at h
        · next hte =>
          subst elt
          cases h
          obtain _ | ⟨hv, hnil⟩ := ha
          cases hnil
          rename_i v
          cases v with
          | tensor _ => cases hv.1.1
          | scalar x =>
            refine ⟨[.tensor (MerlinVerify.Tensor.full shape scalarTy x)], ?_, .cons ⟨⟨?_, ?_⟩, ?_⟩ .nil⟩
            · rw [hregs, evalRegions, tensor_sem_of hm hd]
              simp only [Tensor.sem, Op.blind_name, hn, resultTensorTy_blind, hr]
            · exact hasTy_full hv.1.1
            · intro _ i hi
              change Idx.InBounds shape i at hi
              simpa only [RVal.Defined, MerlinVerify.Tensor.full, MerlinVerify.Tensor.canon, hi, ↓reduceIte] using hv.1.2 rfl
            · cases info with
              | none => trivial
              | some r =>
                intro p hp
                change Idx.InBounds shape p at hp
                simpa [MerlinVerify.Tensor.full, MerlinVerify.Tensor.canon, hp, model, Holds] using hv.2
        · cases h
      · cases h
    · cases h

def reshapeRule (_ : FEnv) (op : Op) (fs : List Fact) : Option (List Fact) :=
  if unmodelledAttr? op.blind = none ∧ op.regions = [] ∧ dialectOf op.name = "tensor" ∧
      (op.name = "tensor.collapse_shape" ∨ op.name = "tensor.expand_shape") then
    match fs, Tensor.resultTensorTy op, Tensor.reassociation? op with
    | [⟨⟨.tensor sh e, true⟩, info⟩], some (shape, elt), some groups =>
      if e = elt ∧ reshapeValid op groups sh shape = true then
        some [⟨⟨.tensor shape elt, true⟩, info⟩]
      else none
    | _, _, _ => none
  else none

def reshapeBook (ib : Nat) : Definedness.Refined.Book model opSem ib where
  rule := reshapeRule
  sound Γ op fs rs h ctx call env args hib _ ha := by
    unfold reshapeRule at h
    split at h
    · next hc =>
      obtain ⟨hm, hregs, hd, hn⟩ := hc
      split at h
      · next _ _ _ sh e info shape elt groups hr hg =>
        split at h
        · next hc =>
          obtain ⟨he, hvalid⟩ := hc
          subst elt
          cases h
          obtain _ | ⟨hv, hnil⟩ := ha
          cases hnil
          rename_i a
          cases a with
          | scalar _ => cases hv.1.1
          | tensor t =>
            obtain ⟨hts, hte, htg⟩ := hv.1.1
            have hdefined := hv.1.2 rfl
            subst sh
            subst e
            have hsize := reshapeValid_size hvalid
            rw [hregs, evalRegions, tensor_sem_of hm hd]
            have hsem : Tensor.sem ctx (restrictCall (calleeOf op.blind) call) op.blind [.tensor t] [] =
                .ok [.tensor (ReshapeIndex.reshape t shape t.elt)] := by
              unfold reshapeValid at hvalid
              rcases hn with hn | hn <;>
                simp_all [Tensor.sem, Op.blind_name, resultTensorTy_blind, reassociation?_blind, ReshapeIndex.reshape]
            refine ⟨_, hsem, .cons ⟨⟨⟨rfl, rfl, ?_⟩, ?_⟩, ?_⟩ .nil⟩
            · intro i hi
              simp only [ReshapeIndex.reshape, MerlinVerify.Tensor.canon, hi, ↓reduceIte]
              exact htg _ (delinearize_inBounds _ _ (hsize ▸ linearIndex_lt _ _ hi))
            · intro _ i hi
              change Idx.InBounds shape i at hi
              simp only [ReshapeIndex.reshape, MerlinVerify.Tensor.canon, hi, ↓reduceIte]
              exact hdefined _ (delinearize_inBounds _ _ (hsize ▸ linearIndex_lt _ _ hi))
            · cases info with
              | none => trivial
              | some r => exact reshape_holds shape t.elt hv.2 hsize.symm
        · cases h
      · cases h
    · cases h

#print axioms reshapeBook
end MerlinVerify.Rules.FrontendRangeTensor
