/- Tensor insertion establishes an exact pointwise value, including poison. The runtime
coordinate decoder and complete bounds premise match the frozen tensor semantics. -/
import MerlinVerify.Check.RulesTensor

namespace MerlinVerify.Rules.InsertExtract

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops

/-- Decode the actual scalar coordinate values, never their source annotations. -/
def coordinates (args : List RVal) : Option (List Int) :=
  args.mapM fun | .scalar value => Sem.Ops.Tensor.indexVal? value | _ => none

/-- The source operation is total at these actual decoded in-bounds coordinates. -/
theorem insert_sem {s : Op} {t : Tensor} {indices : List RVal} {coords : List Int}
    (hn : s.name = "tensor.insert") (hm : unmodelledAttr? s.blind = none)
    (hi : coordinates indices = some coords) (hl : coords.length = t.shape.length)
    (hb : (coords.zip t.shape).all (fun (x, size) => 0 ≤ x ∧ x < size) = true)
    (ctx : Ctx) (call : CallSem) (value : Val) :
    opSem ctx call s (.scalar value :: .tensor t :: indices) [] =
      .ok [.tensor (t.set (coords.map Int.toNat) value).canonize] := by
  have hd : dialectOf s.name = "tensor" := by rw [hn]; cbv
  rw [tensor_sem_of hm hd]
  simp only [Sem.Ops.Tensor.sem, Op.blind_name, hn]
  change (match coordinates indices with
    | some is => if is.length = t.shape.length ∧
        (is.zip t.shape).all (fun (x, size) => 0 ≤ x ∧ x < size) then
        Outcome.ok [RVal.tensor (t.set (is.map Int.toNat) value).canonize] else Outcome.ub
    | none => Outcome.ub) = _
  rw [hi]
  exact ite_eq_left ⟨hl, hb⟩

/-- Canonicalization of the written tensor preserves its newly written in-bounds point. -/
theorem inserted_get (t : Tensor) (value : Val) (coordinate : Idx)
    (hb : Idx.InBounds t.shape coordinate) :
    (t.set coordinate value).canonize.get coordinate = value := by
  simp [Tensor.canonize, Tensor.canon, Tensor.set, hb]

/-- Extraction at an independently decoded matching coordinate returns the stored payload. -/
theorem extract_sem {s : Op} {t : Tensor} {indices : List RVal} {coords : List Int}
    (hn : s.name = "tensor.extract") (hm : unmodelledAttr? s.blind = none)
    (hi : coordinates indices = some coords) (hl : coords.length = t.shape.length)
    (hb : (coords.zip t.shape).all (fun (x, size) => 0 ≤ x ∧ x < size) = true)
    (ctx : Ctx) (call : CallSem) :
    opSem ctx call s (.tensor t :: indices) [] = .ok [.scalar (t.get (coords.map Int.toNat))] := by
  have hd : dialectOf s.name = "tensor" := by rw [hn]; cbv
  rw [tensor_sem_of hm hd]
  simp only [Sem.Ops.Tensor.sem, Op.blind_name, hn]
  change (match coordinates indices with
    | some is => if is.length = t.shape.length ∧
        (is.zip t.shape).all (fun (x, size) => 0 ≤ x ∧ x < size) then
        Outcome.ok [RVal.scalar (t.get (is.map Int.toNat))] else Outcome.ub
    | none => Outcome.ub) = _
  rw [hi]
  exact ite_eq_left ⟨hl, hb⟩

end MerlinVerify.Rules.InsertExtract
