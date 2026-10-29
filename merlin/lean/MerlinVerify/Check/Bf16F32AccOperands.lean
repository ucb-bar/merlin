/- Operand interfaces for reconstructed MACC sites. Repeated uses of one SSA
tensor remain one input, with one value and type. Packing supplies no numerical
or physical-alias premise: it preserves the source's existing SSA identity. -/
import MerlinVerify.Sem.Program
import MerlinVerify.Core.Eq

namespace MerlinVerify.Check.Bf16F32AccOperands

open MerlinVerify

/-- Keep operand order; merge an identical SSA operand only at an identical type. -/
def pack? (left right : Value) : Option (List Value) :=
  if left.id = right.id then
    if left.ty = right.ty then some [left] else none
  else some [left, right]

theorem pack_cases {left right : Value} {packed : List Value}
    (accepted : pack? left right = some packed) :
    (left.id = right.id ∧ left.ty = right.ty ∧ packed = [left]) ∨
    (left.id ≠ right.id ∧ packed = [left, right]) := by
  unfold pack? at accepted
  split at accepted
  · next shared =>
    split at accepted
    · next typed => exact Or.inl ⟨shared, typed, (Option.some.inj accepted).symm⟩
    · cases accepted
  · next distinct => exact Or.inr ⟨distinct, (Option.some.inj accepted).symm⟩

theorem pack_nodup {left right : Value} {packed : List Value}
    (accepted : pack? left right = some packed) : (packed.map (·.id)).Nodup := by
  rcases pack_cases accepted with ⟨_, _, rfl⟩ | ⟨distinct, rfl⟩
  · simp
  · simp [distinct]

/-- A repeated source identity consumes one argument, rather than two independent values. -/
theorem pack_length {left right : Value} {packed : List Value}
    (accepted : pack? left right = some packed) : packed.length = 1 ↔ left.id = right.id := by
  rcases pack_cases accepted with ⟨shared, _, rfl⟩ | ⟨distinct, rfl⟩
  · simp [shared]
  · simp [distinct]

theorem same_sharing {left right targetLeft targetRight : Value} {source target : List Value}
    (sourcePacked : pack? left right = some source)
    (targetPacked : pack? targetLeft targetRight = some target)
    (arity : source.length = target.length) :
    left.id = right.id ↔ targetLeft.id = targetRight.id := by
  rw [← pack_length sourcePacked, ← pack_length targetPacked, arity]

/-- The frozen evaluator retrieves exactly the two original operand values from
the packed argument environment. Shared source IDs require the same value, as
they already do in any SSA environment; distinct IDs stay independent. -/
theorem bind_sound {left right : Value} {packed : List Value}
    (accepted : pack? left right = some packed) (lv rv : RVal)
    (shared : left.id = right.id → lv = rv) :
    ∃ values env, Env.bindAll [] packed values = .ok env ∧
      env.getAll [left.id, right.id] = .ok [lv, rv] := by
  rcases pack_cases accepted with ⟨same, _, rfl⟩ | ⟨distinct, rfl⟩
  · have values := shared same
    subst rv
    refine ⟨[lv], [(left.id, lv)], ?_, ?_⟩
    · simp [Env.bindAll]
    · simp [Env.getAll, Env.get?, Outcome.mapM, Outcome.ofOption, same]
  · refine ⟨[lv, rv], [(left.id, lv), (right.id, rv)], ?_, ?_⟩
    · simp [Env.bindAll]
    · have different : (right.id == left.id) = false := by
        cases h : (right.id == left.id) with
        | false => rfl
        | true =>
          have same : right.id = left.id := by simpa only [beq_iff_eq] using h
          exact False.elim (distinct same.symm)
      simp [Env.getAll, Env.get?, Outcome.mapM, Outcome.ofOption, List.lookup_cons, different]

#print axioms pack_nodup
#print axioms same_sharing
#print axioms bind_sound

end MerlinVerify.Check.Bf16F32AccOperands
