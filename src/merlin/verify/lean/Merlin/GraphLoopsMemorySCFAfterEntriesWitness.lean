/- UNCOMPILED successor. Constructed typed input/caller-heap witnesses, plus
an explicit Unit-valued execution for the retained literal/binary body. Unit
relations illustrate nonempty abstract outcomes; they are not integer numerics.
-/
import Merlin.GraphLoopsMemorySCFAfterEntriesChain

set_option autoImplicit false

namespace Merlin.GraphLoopsMemorySCF.After

open GraphLoopsShape

def sampleRuntime (object : Nat) : Kind → Runtime Unit
  | .index => .index 0
  | .scalar element => .scalar element ()
  | .view type => .view type
      ⟨object, 0, type.offset, type.shape, type.strides, type.element, type.space⟩
      ⟨rfl, rfl, rfl, rfl, rfl⟩

theorem sample_kind (object : Nat) (kind : Kind) : (sampleRuntime object kind).kind = kind := by cases kind <;> rfl

def sampleInputs (object : Nat) (parameters : List Binding) : List (Runtime Unit) :=
  parameters.map (fun binding => sampleRuntime object binding.kind)

theorem sample_typed (object : Nat) (parameters : List Binding) (plain : Plain parameters) :
    ValuesHold parameters (sampleInputs object parameters) := by
  induction parameters with
  | nil => exact .nil
  | cons binding rest induction =>
    exact .cons (Binding.holds_plain binding _ (sample_kind object binding.kind) (plain binding List.mem_cons_self))
      (induction (fun item member => plain item (List.mem_cons_of_mem _ member)))

theorem sample_external (bound : Nat) (parameters : List Binding) :
    ∀ input ∈ sampleInputs bound parameters, ExternalInput bound input := by
  intro input member
  obtain ⟨binding, _, rfl⟩ := List.mem_map.mp member
  cases binding.kind with
  | index => trivial
  | scalar => trivial
  | view type => exact Nat.lt_irrefl bound

def sampleHeap : Heap Unit := fun _ _ => some ()

theorem sample_cell_initialized (object index : Nat) : sampleHeap object index = some () := rfl

/-- Execute the literal and binary ACTIONS with explicit partial relations
chosen to be inhabited. No source execution is assumed. The ordinary Eval
definition supplies both SSA writes and the final return.
-/
theorem literal_binary_execution {World : Type} (meaning : GraphLoopsShape.Interpretation Unit World)
    (literalKey binaryKey element : String) (input literalResult result : Binding)
    (literalKind : literalResult.kind = .scalar element) (literalPlain : literalResult.known = none)
    (resultKind : result.kind = .scalar element) (resultPlain : result.known = none)
    (separate : input.name ≠ literalResult.name)
    (literal : meaning.action literalKey = LiteralAction element literalKey (fun _ _ => True))
    (binary : meaning.action binaryKey = Epoch.Binary element (fun _ _ _ => True))
    (aliases : Aliases) (env : GraphLoopsShape.Env Unit) (read : env input.name = .scalar element ()) (world : World) :
    Eval meaning false aliases
      (.action literalKey [] [literalResult]
        (.action binaryKey [input, literalResult] [result] (.returned [result])))
      env world [.scalar element ()] world := by
  refine ⟨[.scalar element ()], world, ?_,
    .cons (Binding.holds_plain literalResult _ literalKind.symm literalPlain) .nil, ?_⟩
  · rw [literal]
    exact ⟨rfl, (), trivial, rfl, rfl⟩
  · refine ⟨[.scalar element ()], world, ?_,
      .cons (Binding.holds_plain result _ resultKind.symm resultPlain) .nil, ?_⟩
    · rw [binary]
      refine ⟨(), (), (), ?_, trivial, rfl, rfl⟩
      simp only [readArguments, List.map_cons, List.map_nil, Bool.false_eq_true, ↓reduceIte,
        GraphLoopsShape.assign, GraphLoopsShape.put, separate, read]
    · exact ⟨by simp only [readArguments, List.map_cons, List.map_nil, Bool.false_eq_true,
        ↓reduceIte, GraphLoopsShape.assign, GraphLoopsShape.put], rfl⟩

/-- Construct a complete finite-depth entry execution from its actual body and
primitive selections. No whole-run or successful-execution premise is used. -/
theorem literal_binary_complete {World : Type} (meaning : GraphLoopsShape.Interpretation Unit World)
    (module : Module) (function : Function) (found : functionAt module function.name = some function)
    (literalKey binaryKey element : String) (input literalResult result : Binding)
    (parameters : function.parameters = [input])
    (body : function.code = .action literalKey [] [literalResult]
      (.action binaryKey [input, literalResult] [result] (.returned [result])))
    (inputKind : input.kind = .scalar element) (inputPlain : input.known = none)
    (literalKind : literalResult.kind = .scalar element) (literalPlain : literalResult.known = none)
    (resultKind : result.kind = .scalar element) (resultPlain : result.known = none)
    (separate : input.name ≠ literalResult.name)
    (literal : (withCalls meaning module (GraphLoopsShape.invoke meaning module false 0)).action literalKey =
      LiteralAction element literalKey (fun _ _ => True))
    (binary : (withCalls meaning module (GraphLoopsShape.invoke meaning module false 0)).action binaryKey =
      Epoch.Binary element (fun _ _ _ => True)) (world : World) :
    complete meaning module false function.name [.scalar element ()] world [.scalar element ()] world := by
  refine ⟨1, ?_⟩
  simp only [GraphLoopsShape.invoke, found, Function.invoke, parameters, body]
  refine ⟨.cons (Binding.holds_plain input _ inputKind.symm inputPlain) .nil, ?_⟩
  apply literal_binary_execution _ literalKey binaryKey element input literalResult result literalKind literalPlain
    resultKind resultPlain separate literal binary
  simp only [GraphLoopsShape.assign, GraphLoopsShape.put, ↓reduceIte]

end Merlin.GraphLoopsMemorySCF.After
