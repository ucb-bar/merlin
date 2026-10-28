/- Pointwise semantics of rank-one scalar construction and checked extraction. No scalar
algebra is used: selected values retain poison and their exact integer/floating bit patterns. -/
import MerlinVerify.Check.RulesTensor

namespace MerlinVerify.Rules.FromElements

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops

def tensor (element : Ty) (values : List Val) : Tensor :=
  Tensor.canon [values.length] element fun i => values.toArray[linearIndex [values.length] i]?.getD .poison

theorem scalar_args : ∀ (args : List RVal) (values : List Val),
    args.mapM (fun | .scalar v => some v | _ => none) = some values →
    args = values.map RVal.scalar
  | [], values, h => by simpa using h.symm
  | .tensor _ :: args, values, h => by simp [List.mapM_cons] at h
  | .scalar value :: args, values, h => by
    simp only [List.mapM_cons, Option.pure_def, bind, Option.bind] at h
    cases ht : args.mapM (fun | .scalar v => some v | _ => none) with
    | none => simp [ht] at h
    | some rest =>
      simp only [ht, Option.some.injEq] at h
      subst values
      simp only [List.map_cons, scalar_args args rest ht]

theorem tensor_get {element : Ty} {values : List Val} {coordinate : Nat}
    (h : coordinate < values.length) :
    (tensor element values).get [coordinate] = values[coordinate] := by
  have hi : Idx.InBounds [values.length] [coordinate] := by
    exact (Idx.inBounds_cons_cons _ _ _ _).2 ⟨h, (Idx.inBounds_nil_iff []).2 rfl⟩
  simp [tensor, Tensor.canon, hi, linearIndex, List.getElem?_toArray, List.getElem?_eq_getElem h]

theorem constructor_sem {s : Op} {size : Nat} {element : Ty}
    (hn : s.name = "tensor.from_elements") (hm : unmodelledAttr? s.blind = none)
    (ht : Sem.Ops.Tensor.resultTensorTy s = some ([size], element))
    (values : List Val) (hlen : values.length = size) (ctx : Ctx) (call : CallSem) :
    opSem ctx call s (values.map RVal.scalar) [] = .ok [.tensor (tensor element values)] := by
  have hd : dialectOf s.name = "tensor" := by rw [hn]; cbv
  rw [tensor_sem_of hm hd]
  simp only [Sem.Ops.Tensor.sem, Op.blind_name, hn, resultTensorTy_blind, ht]
  split
  · next shape elt vs htype hargs =>
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj htype)
    have heq : values = vs := Option.some.inj
      ((mapM_scalars (fun | .scalar v => some v | _ => none) (fun _ => rfl) values).symm.trans hargs)
    subst vs
    simp [shapeSize, hlen, tensor]
  · next hbad =>
    exact (hbad [size] element values rfl
      (mapM_scalars (fun | .scalar v => some v | _ => none) (fun _ => rfl) values)).elim

/-- Success of the actual constructor, including its count check, establishes the concrete
scalar sequence. Result annotations alone cannot establish this relation. -/
theorem constructor_success {s : Op} {size : Nat} {element : Ty} {args values : List RVal}
    (hn : s.name = "tensor.from_elements") (hm : unmodelledAttr? s.blind = none)
    (ht : Sem.Ops.Tensor.resultTensorTy s = some ([size], element))
    {ctx : Ctx} {call : CallSem} (hs : opSem ctx call s args [] = .ok values) :
    ∃ scalars, args = scalars.map RVal.scalar ∧ scalars.length = size ∧
      values = [.tensor (tensor element scalars)] := by
  have hd : dialectOf s.name = "tensor" := by rw [hn]; cbv
  rw [tensor_sem_of hm hd] at hs
  simp only [Sem.Ops.Tensor.sem, Op.blind_name, hn, resultTensorTy_blind, ht] at hs
  split at hs
  · next shape elt scalars htype hargs =>
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj htype)
    split at hs
    · next hlen =>
      have hl : scalars.length = size := by simpa [shapeSize] using hlen
      refine ⟨scalars, scalar_args args scalars hargs, hl, ?_⟩
      simpa [tensor, hl] using (Outcome.ok.inj hs).symm
    · cases hs
  · cases hs

theorem extract_sem {s : Op} (hn : s.name = "tensor.extract")
    (hm : unmodelledAttr? s.blind = none) {t : Tensor} {size coordinate : Nat}
    (hshape : t.shape = [size]) (hbound : coordinate < size) {index : Val}
    (hi : Sem.Ops.Tensor.indexVal? index = some (Int.ofNat coordinate)) (ctx : Ctx) (call : CallSem) :
    opSem ctx call s [.tensor t, .scalar index] [] = .ok [.scalar (t.get [coordinate])] := by
  have hd : dialectOf s.name = "tensor" := by rw [hn]; cbv
  rw [tensor_sem_of hm hd]
  simp [Sem.Ops.Tensor.sem, Op.blind_name, hn, hi, hshape, hbound]

end MerlinVerify.Rules.FromElements
