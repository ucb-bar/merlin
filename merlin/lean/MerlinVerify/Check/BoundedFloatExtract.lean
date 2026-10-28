/- A bounded read preserves the tensor's proved floating and elementwise integer facts.
Only the actual current index facts may justify an access. -/
import MerlinVerify.Check.BoundedFloatFacts
import MerlinVerify.Check.DefinedBounds
import MerlinVerify.Rules.W8A8SourceFree

namespace MerlinVerify.Check.BoundedFloatFacts.Extract

open MerlinVerify Definedness Rules Sem.Ops

local notation "sourceSem" => W8A8SourceSemantics.opSem

def rule (_ : Refined.FEnv Info) (op : Op) (facts : List (Refined.Fact Info)) :
    Option (List (Refined.Fact Info)) :=
  if unmodelledAttr? op.blind = none ∧ dialectOf op.name = "tensor" ∧ op.name = "tensor.extract" ∧
      op.regions = [] then
    match facts, op.results with
    | ⟨⟨.tensor shape elt, true⟩, info⟩ :: indices, [result] =>
      if result.ty = elt ∧ Bounds.within (indices.map scalarRange) shape = true then
        some [⟨⟨.scalar elt, true⟩, info⟩]
      else none
    | _, _ => none
  else none

theorem rule_sound (ib : Nat) : ∀ Γ op facts results, rule Γ op facts = some results →
    ∀ ctx call env args, ctx.indexBits = ib → Refined.EnvHolds model ib Γ env →
    Refined.FactsHold model ib args facts → ∃ values,
      sourceSem ctx call op args (evalRegions sourceSem call env op.regions) = .ok values ∧
      Refined.FactsHold model ib values results := by
  intro Γ op facts results checked ctx call env args _ _ inputs
  unfold rule at checked
  split at checked
  · next header =>
    obtain ⟨modelled, dialect, named, regions⟩ := header
    split at checked
    · next shape elt info indices result arity =>
      split at checked
      · next conditions =>
        obtain ⟨_, within⟩ := conditions
        cases checked
        obtain _ | ⟨head, tail⟩ := inputs
        rename_i tensor indicesValues
        cases tensor with
        | scalar _ => cases head.1.1
        | tensor tensor =>
          obtain ⟨coords, decoded, count, bounded⟩ := Bounds.within_sound ((toScalarRanges ib).facts tail) within
          have inBounds : Idx.InBounds tensor.shape (coords.map Int.toNat) := by
            rw [head.1.1.1]
            exact coords_inBounds coords shape count bounded
          refine ⟨[.scalar (tensor.get (coords.map Int.toNat))], ?_,
            .cons ⟨Fact.element head.1 inBounds, (elementwise _ _).1 head.2 _ inBounds⟩ .nil⟩
          have ordinary : W8A8SourceSemantics.isSource op.name = false := by
            simp [W8A8SourceSemantics.isSource, named, W8A8SourceContract.name]
          rw [regions, evalRegions, W8A8SourceSemantics.unchanged ordinary, tensor_sem_of modelled dialect]
          simp only [Sem.Ops.Tensor.sem, Op.blind_name, named]
          change (match indicesValues.mapM IndexRanges.value? with
            | some is => if is.length = tensor.shape.length ∧
                (is.zip tensor.shape).all (fun (x, size) => 0 ≤ x ∧ x < size) then
                Outcome.ok [RVal.scalar (tensor.get (is.map Int.toNat))] else Outcome.ub
            | none => Outcome.ub) = Outcome.ok [RVal.scalar (tensor.get (coords.map Int.toNat))]
          rw [decoded]
          exact ite_eq_left ⟨by simpa only [head.1.1.1] using count,
            by simpa only [head.1.1.1] using bounded⟩
      · cases checked
    · cases checked
  · cases checked

def book (ib : Nat) : Refined.Book model sourceSem ib where
  rule := rule
  sound := rule_sound ib

#print axioms rule_sound
#print axioms book

end MerlinVerify.Check.BoundedFloatFacts.Extract
