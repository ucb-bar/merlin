/- A concrete caller-memory observer: index constant, load, unchanged scalar
binary operation, store and return. The effect summary is derived from these
operations and actual scoped bindings, not supplied as a whole-callee premise. -/
import Merlin.GraphLoopsShapeEpochAccess
import Merlin.GraphLoopsShapeEpochPublic

set_option autoImplicit false

namespace Merlin.GraphLoopsShape.Epoch

structure Touch where
  loadKey : String
  binaryKey : String
  storeKey : String
  input : Binding
  buffer : Binding
  zero : Binding
  loaded : Binding
  result : Binding
  element : String
  deriving DecidableEq, Repr

def Touch.code (program : Touch) : Code :=
  .constant program.zero 0
    (.action program.loadKey [program.buffer, program.zero] [program.loaded]
      (.action program.binaryKey [program.loaded, program.input] [program.result]
        (.action program.storeKey [program.result, program.buffer, program.zero] []
          (.returned [program.result]))))

def Touch.Facts (program : Touch) : Prop :=
  program.input.kind = .scalar program.element ∧ program.input.known = none ∧
  program.buffer.kind = .view (type [1] program.element) ∧ program.buffer.known = none ∧
  program.zero.kind = .index ∧ program.zero.known = some 0 ∧
  program.loaded.kind = .scalar program.element ∧ program.loaded.known = none ∧
  program.result.kind = .scalar program.element ∧ program.result.known = none ∧
  program.input.name ≠ program.zero.name ∧ program.input.name ≠ program.loaded.name ∧
  program.buffer.name ≠ program.zero.name ∧ program.buffer.name ≠ program.loaded.name ∧
  program.buffer.name ≠ program.result.name ∧ program.zero.name ≠ program.loaded.name ∧
  program.zero.name ≠ program.result.name

instance touchFactsDecidable (program : Touch) : Decidable program.Facts := by
  unfold Touch.Facts
  infer_instance

theorem Touch.eval {Value User : Type} (program : Touch) (checked : decide program.Facts = true)
    (meaning : Interpretation Value (MemoryWorld Value User)) (operation : Value → Value → Value → Prop)
    (load : meaning.action program.loadKey = Load)
    (binary : meaning.action program.binaryKey = Binary program.element operation)
    (store : meaning.action program.storeKey = Store)
    (aliases : Aliases) (env : Env Value) (object : Nat) (value : Value)
    (input : env program.input.name = .scalar program.element value)
    (buffer : env program.buffer.name = view object [1] program.element)
    (before after : MemoryWorld Value User) (outputs : List (Runtime Value)) :
    Eval meaning false aliases program.code env before outputs after ↔
      ∃ loaded result, before.heap object 0 = some loaded ∧ operation loaded value result ∧
        outputs = [.scalar program.element result] ∧
        after = ⟨before.observed, GraphLoopsReuse.replaceCell before.heap object 0 result⟩ := by
  rcases of_decide_eq_true checked with ⟨_, _, _, _, _, _, loadedKind, loadedPlain, resultKind, resultPlain,
    inputZero, inputLoaded, bufferZero, bufferLoaded, bufferResult, zeroLoaded, zeroResult⟩
  let initial := put env program.zero.name (.index 0)
  let firstAliases := (program.zero.name, program.zero.name) :: aliases
  change Eval meaning false firstAliases _ initial before outputs after ↔ _
  have loadArguments : readArguments false firstAliases initial [program.buffer, program.zero] =
      view object [1] program.element :: [0].map Runtime.index := by
    simp only [readArguments, List.map_cons, List.map_nil, Bool.false_eq_true, ↓reduceIte,
      initial, put, bufferZero, buffer]
  rw [eval_load meaning program.loadKey load firstAliases initial _ program.loaded _ object [1] [0]
    program.element loadArguments loadedKind loadedPlain]
  simp only [GraphLoopsReuse.Geometry.Coordinate, Nat.zero_lt_succ, and_self, true_and,
    GraphLoopsReuse.Geometry.flatten, GraphLoopsReuse.Geometry.volume, Nat.zero_mul, Nat.zero_add]
  constructor
  · rintro ⟨loaded, read, suffix⟩
    let loadedEnv := put initial program.loaded.name (.scalar program.element loaded)
    let loadedAliases := extendAliases firstAliases [program.loaded]
    have binaryArguments : readArguments false loadedAliases loadedEnv [program.loaded, program.input] =
        [.scalar program.element loaded, .scalar program.element value] := by
      simp only [readArguments, List.map_cons, List.map_nil, Bool.false_eq_true, ↓reduceIte,
        loadedEnv, initial, put, inputLoaded, inputZero, input]
    obtain ⟨result, run, suffix⟩ := (eval_binary meaning program.binaryKey program.element operation binary
      loadedAliases loadedEnv _ program.result _ loaded value binaryArguments resultKind resultPlain before after outputs).mp suffix
    let resultEnv := put loadedEnv program.result.name (.scalar program.element result)
    have storeArguments : readArguments false (extendAliases loadedAliases [program.result]) resultEnv
        [program.result, program.buffer, program.zero] =
        .scalar program.element result :: view object [1] program.element :: [0].map Runtime.index := by
      simp only [readArguments, List.map_cons, List.map_nil, Bool.false_eq_true, ↓reduceIte,
        resultEnv, loadedEnv, initial, put, bufferResult, bufferLoaded, bufferZero, zeroResult, zeroLoaded, buffer]
    obtain ⟨_, suffix⟩ := (eval_store meaning program.storeKey store _ resultEnv _ _ object [1] [0]
      program.element result storeArguments before after outputs).mp suffix
    have final : outputs = [.scalar program.element result] ∧
        after = ⟨before.observed, GraphLoopsReuse.replaceCell before.heap object 0 result⟩ := by
      simpa only [Eval, readArguments, List.map_cons, List.map_nil, Bool.false_eq_true, ↓reduceIte,
        resultEnv, put, GraphLoopsReuse.Geometry.flatten, GraphLoopsReuse.Geometry.volume,
        Nat.zero_mul, Nat.zero_add] using suffix
    exact ⟨loaded, result, read, run, final⟩
  · rintro ⟨loaded, result, read, run, outputs, after⟩
    refine ⟨loaded, read, ?_⟩
    let loadedEnv := put initial program.loaded.name (.scalar program.element loaded)
    let loadedAliases := extendAliases firstAliases [program.loaded]
    have binaryArguments : readArguments false loadedAliases loadedEnv [program.loaded, program.input] =
        [.scalar program.element loaded, .scalar program.element value] := by
      simp only [readArguments, List.map_cons, List.map_nil, Bool.false_eq_true, ↓reduceIte,
        loadedEnv, initial, put, inputLoaded, inputZero, input]
    apply (eval_binary meaning program.binaryKey program.element operation binary loadedAliases loadedEnv _ program.result _
      loaded value binaryArguments resultKind resultPlain before _ _).mpr
    refine ⟨result, run, ?_⟩
    let resultEnv := put loadedEnv program.result.name (.scalar program.element result)
    have storeArguments : readArguments false (extendAliases loadedAliases [program.result]) resultEnv
        [program.result, program.buffer, program.zero] =
        .scalar program.element result :: view object [1] program.element :: [0].map Runtime.index := by
      simp only [readArguments, List.map_cons, List.map_nil, Bool.false_eq_true, ↓reduceIte,
        resultEnv, loadedEnv, initial, put, bufferResult, bufferLoaded, bufferZero, zeroResult, zeroLoaded, buffer]
    apply (eval_store meaning program.storeKey store _ resultEnv _ _ object [1] [0] program.element result
      storeArguments before _ _).mpr
    refine ⟨by simp [GraphLoopsReuse.Geometry.Coordinate], ?_⟩
    simpa only [Eval, readArguments, List.map_cons, List.map_nil, Bool.false_eq_true, ↓reduceIte,
      resultEnv, put, GraphLoopsReuse.Geometry.flatten, GraphLoopsReuse.Geometry.volume,
      Nat.zero_mul, Nat.zero_add] using And.intro outputs after

/-- The observer has an explicitly derived caller-only effect. All private
objects and all other caller cells are retained by concrete heap reconstruction. -/
theorem Touch.caller_effect {Value User : Type} (program : Touch) (checked : decide program.Facts = true)
    (bound object : Nat) (outside : ¬ object < bound)
    (meaning : Interpretation Value (MemoryWorld Value User)) (operation : Value → Value → Value → Prop)
    (load : meaning.action program.loadKey = Load)
    (binary : meaning.action program.binaryKey = Binary program.element operation)
    (store : meaning.action program.storeKey = Store)
    (aliases : Aliases) (env : Env Value) (value : Value)
    (input : env program.input.name = .scalar program.element value)
    (buffer : env program.buffer.name = view object [1] program.element)
    (before : GraphLoopsReuse.Memory Value (PublicState Value User))
    (after : MemoryWorld Value User) (outputs : List (Runtime Value)) :
    Eval meaning false aliases program.code env (fullWorld bound before) outputs after ↔
      ∃ loaded result, before.observed.callerHeap object 0 = some loaded ∧ operation loaded value result ∧
        outputs = [.scalar program.element result] ∧
        after = fullWorld bound { before with observed := { before.observed with
          callerHeap := GraphLoopsReuse.replaceCell before.observed.callerHeap object 0 result } } := by
  rw [program.eval checked meaning operation load binary store aliases env object value input buffer]
  change (∃ loaded result, mergeHeap bound before.heap before.observed.callerHeap object 0 = some loaded ∧ _) ↔ _
  rw [read_caller bound object 0 before.heap before.observed.callerHeap outside]
  apply exists_congr; intro loaded
  apply exists_congr; intro result
  rw [caller_store_world bound object 0 result before outside]
  rfl

def Touch.function (program : Touch) (name : String) : Function :=
  ⟨name, [program.input, program.buffer], [.scalar program.element], program.code⟩

/-- The complete callee entry uses its actual typed parameters and local SSA
stack. No private caller register environment is handed to the arithmetic. -/
theorem Touch.invoke {Value User : Type} (program : Touch) (checked : decide program.Facts = true)
    (name : String) (separate : program.input.name ≠ program.buffer.name)
    (meaning : Interpretation Value (MemoryWorld Value User)) (operation : Value → Value → Value → Prop)
    (load : meaning.action program.loadKey = Load)
    (binary : meaning.action program.binaryKey = Binary program.element operation)
    (store : meaning.action program.storeKey = Store)
    (object : Nat) (value : Value) (before after : MemoryWorld Value User) (outputs : List (Runtime Value)) :
    (program.function name).invoke meaning false [.scalar program.element value, view object [1] program.element]
      before outputs after ↔
      ∃ loaded result, before.heap object 0 = some loaded ∧ operation loaded value result ∧
        outputs = [.scalar program.element result] ∧
        after = ⟨before.observed, GraphLoopsReuse.replaceCell before.heap object 0 result⟩ := by
  have facts := of_decide_eq_true checked
  have typed : ValuesHold [program.input, program.buffer]
      [.scalar program.element value, view object [1] program.element] :=
    .cons (Binding.holds_plain program.input _ facts.1.symm facts.2.1)
      (.cons (Binding.holds_plain program.buffer _ facts.2.2.1.symm facts.2.2.2.1) .nil)
  change (ValuesHold [program.input, program.buffer] _ ∧ _) ↔ _
  simp only [typed, true_and]
  have inputRead : assign emptyEnv [program.input, program.buffer]
      [.scalar program.element value, view object [1] program.element] program.input.name =
      Runtime.scalar program.element value := by
    simp only [assign, put, separate, ↓reduceIte]
  have bufferRead : assign emptyEnv [program.input, program.buffer]
      [.scalar program.element value, view object [1] program.element] program.buffer.name =
      view object [1] program.element := by
    simp only [assign, put, ↓reduceIte]
  exact program.eval checked meaning operation load binary store _ _ object value inputRead bufferRead before after outputs

/-- Internal call dispatch unfolds the ACTUAL observer function for every
positive call depth. Each executed opcode is excluded from recursive dispatch
by its checked call-table entry; there is no whole-call semantic premise. -/
theorem Touch.call {Value User : Type} (program : Touch) (checked : decide program.Facts = true)
    (module : Module) (name : String) (found : functionAt module name = some (program.function name))
    (separate : program.input.name ≠ program.buffer.name)
    (loadNotCall : callTarget module.calls program.loadKey = none)
    (binaryNotCall : callTarget module.calls program.binaryKey = none)
    (storeNotCall : callTarget module.calls program.storeKey = none)
    (meaning : Interpretation Value (MemoryWorld Value User)) (operation : Value → Value → Value → Prop)
    (load : meaning.action program.loadKey = Load)
    (binary : meaning.action program.binaryKey = Binary program.element operation)
    (store : meaning.action program.storeKey = Store)
    (depth object : Nat) (value : Value) (before after : MemoryWorld Value User) (outputs : List (Runtime Value)) :
    GraphLoopsShape.invoke meaning module false (depth + 1) name
      [.scalar program.element value, view object [1] program.element] before outputs after ↔
      ∃ loaded result, before.heap object 0 = some loaded ∧ operation loaded value result ∧
        outputs = [.scalar program.element result] ∧
        after = ⟨before.observed, GraphLoopsReuse.replaceCell before.heap object 0 result⟩ := by
  simp only [GraphLoopsShape.invoke, found]
  apply program.invoke checked name separate _ operation
  · simp only [withCalls, loadNotCall, load]
  · simp only [withCalls, binaryNotCall, binary]
  · simp only [withCalls, storeNotCall, store]

end Merlin.GraphLoopsShape.Epoch
