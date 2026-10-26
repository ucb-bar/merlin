/- Witness constructors for the DECLARED operational interpretation. These are
kernel proof helpers, not a replacement producer or a native runtime evaluator. -/
import Merlin.GraphLoopsShapeModel

set_option autoImplicit false

namespace Merlin.GraphLoopsShape

theorem Binding.holds_plain {Value : Type} (binding : Binding) (value : Runtime Value)
    (kind : value.kind = binding.kind) (plain : binding.known = none) : binding.Holds value := by
  refine ⟨kind, ?_⟩
  intro index equal
  rw [plain] at equal
  cases equal

theorem Eval.source_cast {Value World : Type} {meaning : Interpretation Value World} {aliases : Aliases}
    {origin result : Binding} {source target : ViewType} {rest : Code} {env : Env Value} {before after : World}
    {outputs : List (Runtime Value)} (value : Runtime Value) (computed : castResult source target (env origin.name) = some value)
    (remaining : Eval meaning false ((result.name, resolve aliases origin.name) :: aliases) rest (put env result.name value) before outputs after) :
    Eval meaning false aliases (.cast origin result source target rest) env before outputs after := ⟨value, computed, remaining⟩

theorem Eval.source_dimension {Value World : Type} {meaning : Interpretation Value World} {aliases : Aliases}
    {view axis result : Binding} {type : ViewType} {axisIndex value actual : Nat} {rest : Code} {env : Env Value} {before after : World}
    {outputs : List (Runtime Value)} (computed : dimensionResult type (env view.name) (env axis.name) = some actual)
    (remaining : Eval meaning false ((result.name, result.name) :: aliases) rest (put env result.name (.index actual)) before outputs after) :
    Eval meaning false aliases (.dimension view axis result type axisIndex value rest) env before outputs after := ⟨actual, computed, remaining⟩

theorem Eval.source_constant {Value World : Type} {meaning : Interpretation Value World} {aliases : Aliases}
    {result : Binding} {value : Nat} {rest : Code} {env : Env Value} {before after : World} {outputs : List (Runtime Value)}
    (remaining : Eval meaning false ((result.name, result.name) :: aliases) rest (put env result.name (.index value)) before outputs after) :
    Eval meaning false aliases (.constant result value rest) env before outputs after := remaining

theorem Eval.source_action {Value World : Type} {meaning : Interpretation Value World} {aliases : Aliases}
    {key : String} {operands results : List Binding} {rest : Code} {env : Env Value} {before after : World} {outputs : List (Runtime Value)}
    (values : List (Runtime Value)) (middle : World)
    (run : meaning.action key (readArguments false aliases env operands) before values middle)
    (typed : ValuesHold results values)
    (remaining : Eval meaning false (extendAliases aliases results) rest (assign env results values) middle outputs after) :
    Eval meaning false aliases (.action key operands results rest) env before outputs after := ⟨values, middle, run, typed, remaining⟩

theorem Eval.source_region {Value World : Type} {meaning : Interpretation Value World} {aliases : Aliases}
    {key : String} {operands parameters results : List Binding} {body rest : Code} {env : Env Value}
    {before after : World} {outputs : List (Runtime Value)} (values : List (Runtime Value)) (middle : World)
    (run : meaning.region key (readArguments false aliases env operands)
      (fun inputs initial returned final => ValuesHold parameters inputs ∧
        Eval meaning false (extendAliases aliases parameters) body (assign env parameters inputs) initial returned final)
      before values middle)
    (typed : ValuesHold results values)
    (remaining : Eval meaning false (extendAliases aliases results) rest (assign env results values) middle outputs after) :
    Eval meaning false aliases (.region key operands parameters results body rest) env before outputs after := ⟨values, middle, run, typed, remaining⟩

theorem Load.ready {Value Public : Type} (type : ViewType) (descriptor : Descriptor) (valid : Conforms type descriptor)
    (coordinates : List Nat) (value : Value) (memory : MemoryWorld Value Public)
    (inside : Merlin.GraphLoopsReuse.Geometry.Coordinate descriptor.shape coordinates)
    (read : memory.heap descriptor.allocation (descriptor.alignedBase + descriptor.offset + linearOffset descriptor.strides coordinates) = some value) :
    Load (Runtime.view type descriptor valid :: coordinates.map Runtime.index) memory [Runtime.scalar type.element value] memory :=
  ⟨type, descriptor, valid, coordinates, value, rfl, inside, read, rfl, rfl⟩

theorem Store.ready {Value Public : Type} (type : ViewType) (descriptor : Descriptor) (valid : Conforms type descriptor)
    (coordinates : List Nat) (value : Value) (memory : MemoryWorld Value Public)
    (inside : Merlin.GraphLoopsReuse.Geometry.Coordinate descriptor.shape coordinates) :
    Store (Runtime.scalar type.element value :: Runtime.view type descriptor valid :: coordinates.map Runtime.index) memory []
      ⟨memory.observed, Merlin.GraphLoopsReuse.replaceCell memory.heap descriptor.allocation
        (descriptor.alignedBase + descriptor.offset + linearOffset descriptor.strides coordinates) value⟩ :=
  ⟨type, descriptor, valid, coordinates, value, rfl, inside, rfl, rfl⟩

theorem SCF.once {Value World : Type} (body : RegionMeaning Value World) (before after : World)
    (run : body [.index 0] before [] after) : SCF body [.index 0, .index 1, .index 1] before [] after := by
  refine ⟨0, 1, 1, [], rfl, by decide, [0], ?_⟩
  exact .more (middle := ([], after)) (by decide) run (.done _ (by decide))

end Merlin.GraphLoopsShape
