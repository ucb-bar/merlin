/- Whole-block preservation, including arbitrary partial effects and nested
lexical regions. Only the intrinsic cast/dimension operations are deleted or
materialized. No complete-normalizer or whole-program agreement is assumed. -/
import Merlin.GraphLoopsShapeCode

set_option autoImplicit false

namespace Merlin.GraphLoopsShape

theorem preservation {Value World : Type} (meaning : Interpretation Value World) (bits : Nat) (code : Code) :
    ∀ expected context aliases, Valid bits expected context aliases code →
      ∀ source target, Agreement context aliases source target → ∀ before outputs after,
        Eval meaning false aliases code source before outputs after ↔
          Eval meaning true aliases code target before outputs after := by
  induction code with
  | returned values =>
    intro expected context aliases valid source target same before outputs after
    have equal := same.read_arguments values valid.2.1
    simp only [Eval]
    rw [equal]
  | cast origin result sourceType targetType rest induction =>
    intro expected context aliases valid source target same before outputs after
    obtain ⟨scope, ⟨member, originKind, rfl, resultKind, known, fresh⟩, tailValid⟩ := valid
    have typed : (source origin.name).kind = .view sourceType := (same.1 origin member).1.trans originKind
    have computed := cast_result typed
    have resultHolds : result.Holds (source origin.name) := by
      refine ⟨typed.trans resultKind.symm, ?_⟩
      intro index equal
      rw [known] at equal
      cases equal
    have nextSame := same.alias member result fresh resultHolds
    have tail := induction _ _ _ tailValid _ _ nextSame before outputs after
    simp only [Eval, Bool.false_eq_true, ↓reduceIte]
    constructor
    · rintro ⟨value, calculated, run⟩
      have equal : value = source origin.name := Option.some.inj (calculated.symm.trans computed)
      subst value
      exact tail.mp run
    · intro run
      exact ⟨source origin.name, computed, tail.mpr run⟩
  | dimension view axis result type axisIndex value rest induction =>
    intro expected context aliases valid source target same before outputs after
    obtain ⟨scope, ⟨viewMember, axisMember, viewKind, knownAxis, resultKind, knownResult, dimension, fresh⟩, tailValid⟩ := valid
    have viewTyped : (source view.name).kind = .view type := (same.1 view viewMember).1.trans viewKind
    have axisValue := known_index (same.1 axis axisMember) knownAxis
    have computed : dimensionResult type (source view.name) (source axis.name) = some value := by
      rw [axisValue]
      exact dimension_result viewTyped dimension
    have resultHolds := result.holds_index (Value := Value) value resultKind knownResult
    have nextSame := same.define scope result fresh (.index value) resultHolds
    have tail := induction _ _ _ tailValid _ _ nextSame before outputs after
    simp only [Eval, Bool.false_eq_true, ↓reduceIte]
    constructor
    · rintro ⟨actual, calculated, run⟩
      have equal : actual = value := Option.some.inj (calculated.symm.trans computed)
      subst actual
      exact tail.mp run
    · intro run
      exact ⟨value, computed, tail.mpr run⟩
  | constant result value rest induction =>
    intro expected context aliases valid source target same before outputs after
    obtain ⟨scope, ⟨kind, known, _fits, fresh⟩, tailValid⟩ := valid
    have typed := result.holds_index (Value := Value) value kind known
    exact induction _ _ _ tailValid _ _ (same.define scope result fresh (.index value) typed) before outputs after
  | action key operands results rest induction =>
    intro expected context aliases valid source target same before outputs after
    obtain ⟨scope, bound, fresh, _plain, tailValid⟩ := valid
    have inputs := same.read_arguments operands bound
    simp only [Eval]
    constructor
    · rintro ⟨values, middle, run, typed, remaining⟩
      refine ⟨values, middle, ?_, typed, ?_⟩
      · rw [← inputs]; exact run
      · exact (induction _ _ _ tailValid _ _ (same.define_many scope typed fresh) middle outputs after).mp remaining
    · rintro ⟨values, middle, run, typed, remaining⟩
      refine ⟨values, middle, ?_, typed, ?_⟩
      · rw [inputs]; exact run
      · exact (induction _ _ _ tailValid _ _ (same.define_many scope typed fresh) middle outputs after).mpr remaining
  | region key operands parameters results body rest bodyInduction restInduction =>
    intro expected context aliases valid source target same before outputs after
    obtain ⟨scope, bound, freshParams, _plainParams, bodyValid, freshResults, _plainResults, restValid⟩ := valid
    have inputs := same.read_arguments operands bound
    have regions :
        (fun arguments initial returned final => ValuesHold parameters arguments ∧
          Eval meaning false (extendAliases aliases parameters) body (assign source parameters arguments) initial returned final) =
        (fun arguments initial returned final => ValuesHold parameters arguments ∧
          Eval meaning true (extendAliases aliases parameters) body (assign target parameters arguments) initial returned final) := by
      funext arguments initial returned final
      apply propext
      constructor
      · rintro ⟨typed, run⟩
        exact ⟨typed, (bodyInduction _ _ _ bodyValid _ _ (same.define_many scope typed freshParams) initial returned final).mp run⟩
      · rintro ⟨typed, run⟩
        exact ⟨typed, (bodyInduction _ _ _ bodyValid _ _ (same.define_many scope typed freshParams) initial returned final).mpr run⟩
    simp only [Eval]
    constructor
    · rintro ⟨values, middle, run, typed, remaining⟩
      refine ⟨values, middle, ?_, typed, ?_⟩
      · rw [← inputs, ← regions]; exact run
      · exact (restInduction _ _ _ restValid _ _ (same.define_many scope typed freshResults) middle outputs after).mp remaining
    · rintro ⟨values, middle, run, typed, remaining⟩
      refine ⟨values, middle, ?_, typed, ?_⟩
      · rw [inputs, regions]; exact run
      · exact (restInduction _ _ _ restValid _ _ (same.define_many scope typed freshResults) middle outputs after).mpr remaining

/-- All possible COMPLETE returned-value/world outcomes, under one declared
operation/region meaning. Entry descriptors are typed values, not arbitrary
metadata. Unknown primitive errors/divergence/native conformance remain separate. -/
theorem entry_equivalence {Value World : Type} (meaning : Interpretation Value World) (bits : Nat) (code : Code)
    (expected : List Kind) (parameters : Context) (valid : Valid bits expected parameters [] code)
    (source target : Env Value) (typed : Typed parameters source)
    (sameInputs : ∀ binding ∈ parameters, source binding.name = target binding.name)
    (before : World) (outputs : List (Runtime Value)) (after : World) :
    Eval meaning false [] code source before outputs after ↔ Eval meaning true [] code target before outputs after := by
  exact preservation meaning bits code expected parameters [] valid source target ⟨typed, sameInputs⟩ before outputs after

end Merlin.GraphLoopsShape
