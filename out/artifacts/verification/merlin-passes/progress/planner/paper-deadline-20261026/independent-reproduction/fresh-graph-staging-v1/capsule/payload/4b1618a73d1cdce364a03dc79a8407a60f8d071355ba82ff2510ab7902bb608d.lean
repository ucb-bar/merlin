/- Local view/SSA adapter to the sealed epoch memory semantics.
Canonical private descriptors identify logical source objects at base zero;
this is not a claim about native pointers or reused physical identities. -/
import Merlin.GraphLoopsShapeCountermodels

set_option autoImplicit false

namespace Merlin.GraphLoopsShape.Epoch

abbrev Geometry := Merlin.GraphLoopsReuse.Geometry.Coordinate

def strides : List Nat → List Nat
  | [] => []
  | _ :: rest => Merlin.GraphLoopsReuse.Geometry.volume rest :: strides rest

def type (shape : List Nat) (element : String) : ViewType :=
  ⟨shape, 0, strides shape, element, 0⟩

def descriptor (object : Nat) (shape : List Nat) (element : String) : Descriptor :=
  ⟨object, 0, 0, shape, strides shape, element, 0⟩

theorem conforms (object : Nat) (shape : List Nat) (element : String) :
    Conforms (type shape element) (descriptor object shape element) := by
  exact ⟨rfl, rfl, rfl, rfl, rfl⟩

def view {Value : Type} (object : Nat) (shape : List Nat) (element : String) : Runtime Value :=
  .view (type shape element) (descriptor object shape element) (conforms object shape element)

/-- Every rank uses the old row-major element address, with no volume expansion. -/
theorem offset (shape coordinates : List Nat) :
    linearOffset (strides shape) coordinates = Merlin.GraphLoopsReuse.Geometry.flatten shape coordinates := by
  induction shape generalizing coordinates with
  | nil => rfl
  | cons extent rest induction =>
    cases coordinates with
    | nil => rfl
    | cons index indices =>
      simp only [strides, linearOffset, Merlin.GraphLoopsReuse.Geometry.flatten, induction, Nat.mul_comm]

theorem address (object : Nat) (shape coordinates : List Nat) (element : String) :
    GraphLoopsShape.address (descriptor object shape element) coordinates =
      (object, Merlin.GraphLoopsReuse.Geometry.flatten shape coordinates) := by
  simp only [GraphLoopsShape.address, descriptor, Nat.zero_add, offset]

/-- Erasing a typed scalar into the initializer environment never turns a view
into a scalar or guesses an undefined variable's value. -/
def erase {Value : Type} : Runtime Value → Option (GraphLoopsMemorySCF.Datum Value)
  | .index value => some (.index value)
  | .scalar _ value => some (.scalar value)
  | .view _ _ _ => none

def eraseEnv {Value : Type} (env : Env Value) : GraphLoopsMemorySCF.Env Value :=
  fun name => erase (env name)

theorem erase_put {Value : Type} (env : Env Value) (name : Nat) (value : Runtime Value)
    (datum : GraphLoopsMemorySCF.Datum Value) (scalar : erase value = some datum) :
    eraseEnv (put env name value) = GraphLoopsMemorySCF.put (eraseEnv env) name datum := by
  funext other
  by_cases same : other = name
  · simp only [eraseEnv, put, GraphLoopsMemorySCF.put, same, ↓reduceIte, scalar]
  · simp only [eraseEnv, put, GraphLoopsMemorySCF.put, same, ↓reduceIte]

def eraseValues {Value : Type} : List (Runtime Value) → Option (List (GraphLoopsMemorySCF.Datum Value))
  | [] => some []
  | value :: rest => do
      let first ← erase value
      let remaining ← eraseValues rest
      pure (first :: remaining)

theorem erase_read {Value : Type} (env : Env Value) (bindings : List Binding) :
    GraphLoopsMemorySCF.read (eraseEnv env) (names bindings) =
      eraseValues (readArguments false [] env bindings) := by
  induction bindings with
  | nil => rfl
  | cons binding rest induction =>
    simp only [names, List.map_cons, GraphLoopsMemorySCF.read, readArguments, Bool.false_eq_true,
      ↓reduceIte, eraseValues, eraseEnv] at induction ⊢
    rw [induction]

/-- Actual read/store descriptors determine the same old element-cell address.
The input's allocation label is kept, not replaced merely because types match. -/
theorem store_ready {Value Public : Type} (object : Nat) (shape coordinates : List Nat) (element : String)
    (value : Value) (before : MemoryWorld Value Public)
    (inside : Merlin.GraphLoopsReuse.Geometry.Coordinate shape coordinates) :
    Store (.scalar element value :: view object shape element :: coordinates.map Runtime.index) before []
      ⟨before.observed, Merlin.GraphLoopsReuse.replaceCell before.heap object
        (Merlin.GraphLoopsReuse.Geometry.flatten shape coordinates) value⟩ := by
  have result := Store.ready (type shape element) (descriptor object shape element)
    (conforms object shape element) coordinates value before inside
  simpa only [view, type, descriptor, Nat.zero_add, offset] using result

theorem load_ready {Value Public : Type} (object : Nat) (shape coordinates : List Nat) (element : String)
    (value : Value) (before : MemoryWorld Value Public)
    (inside : Merlin.GraphLoopsReuse.Geometry.Coordinate shape coordinates)
    (read : before.heap object (Merlin.GraphLoopsReuse.Geometry.flatten shape coordinates) = some value) :
    Load (view object shape element :: coordinates.map Runtime.index) before [.scalar element value] before := by
  apply Load.ready (type shape element) (descriptor object shape element)
    (conforms object shape element) coordinates value before inside
  simpa only [descriptor, Nat.zero_add, offset] using read

theorem indices_injective {Value : Type} (first second : List Nat)
    (same : first.map (Runtime.index (Value := Value)) = second.map Runtime.index) : first = second := by
  induction first generalizing second with
  | nil => cases second <;> simp_all
  | cons index rest induction =>
    cases second with
    | nil => cases same
    | cons other tail =>
      have parts := List.cons.inj same
      have heads := Runtime.index.inj parts.1
      rw [heads, induction tail parts.2]

theorem store_iff {Value Public : Type} (object : Nat) (shape coordinates : List Nat) (element : String)
    (value : Value) (before after : MemoryWorld Value Public) (outputs : List (Runtime Value)) :
    Store (.scalar element value :: view object shape element :: coordinates.map Runtime.index) before outputs after ↔
      Merlin.GraphLoopsReuse.Geometry.Coordinate shape coordinates ∧ outputs = [] ∧
      after = ⟨before.observed, Merlin.GraphLoopsReuse.replaceCell before.heap object
        (Merlin.GraphLoopsReuse.Geometry.flatten shape coordinates) value⟩ := by
  constructor
  · rintro ⟨actualType, actualDescriptor, valid, actualCoordinates, actualValue, inputs, inside, output, final⟩
    have scalarEq := (List.cons.inj inputs).1
    have viewEq := (List.cons.inj (List.cons.inj inputs).2).1
    have indicesEq := (List.cons.inj (List.cons.inj inputs).2).2
    change Runtime.view (type shape element) (descriptor object shape element) (conforms object shape element) =
      Runtime.view actualType actualDescriptor valid at viewEq
    cases viewEq
    cases scalarEq
    have coordinatesEq : coordinates = actualCoordinates :=
      indices_injective coordinates actualCoordinates indicesEq
    subst actualCoordinates
    refine ⟨inside, output, ?_⟩
    simpa only [descriptor, Nat.zero_add, offset] using final
  · rintro ⟨inside, rfl, rfl⟩
    exact store_ready object shape coordinates element value before inside

theorem load_iff {Value Public : Type} (object : Nat) (shape coordinates : List Nat) (element : String)
    (before after : MemoryWorld Value Public) (outputs : List (Runtime Value)) :
    Load (view object shape element :: coordinates.map Runtime.index) before outputs after ↔
      ∃ value, Merlin.GraphLoopsReuse.Geometry.Coordinate shape coordinates ∧
        before.heap object (Merlin.GraphLoopsReuse.Geometry.flatten shape coordinates) = some value ∧
        outputs = [.scalar element value] ∧ after = before := by
  constructor
  · rintro ⟨actualType, actualDescriptor, valid, actualCoordinates, value, inputs, inside, read, output, final⟩
    have viewEq := (List.cons.inj inputs).1
    have indicesEq := (List.cons.inj inputs).2
    change Runtime.view (type shape element) (descriptor object shape element) (conforms object shape element) =
      Runtime.view actualType actualDescriptor valid at viewEq
    cases viewEq
    have coordinatesEq := indices_injective coordinates actualCoordinates indicesEq
    subst actualCoordinates
    refine ⟨value, inside, ?_, output, final⟩
    simpa only [descriptor, Nat.zero_add, offset] using read
  · rintro ⟨value, inside, read, rfl, rfl⟩
    exact load_ready object shape coordinates element value _ inside read

theorem erase_assign {Value : Type} (env : Env Value) (bindings : List Binding)
    (values : List (Runtime Value)) (datums : List (GraphLoopsMemorySCF.Datum Value))
    (scalar : eraseValues values = some datums) :
    eraseEnv (assign env bindings values) = GraphLoopsMemorySCF.assign (eraseEnv env) (names bindings) datums := by
  induction bindings generalizing env values datums with
  | nil => rfl
  | cons binding rest induction =>
    cases values with
    | nil =>
      have same : datums = [] := (Option.some.inj scalar).symm
      subst datums
      rfl
    | cons value values =>
      cases first : erase value with
      | none => simp [eraseValues, first] at scalar
      | some datum =>
        cases remaining : eraseValues values with
        | none => simp [eraseValues, first, remaining] at scalar
        | some tail =>
          have same : datums = datum :: tail := by simpa [eraseValues, first, remaining] using scalar.symm
          subst datums
          change eraseEnv (assign (put env binding.name value) rest values) =
            GraphLoopsMemorySCF.assign (GraphLoopsMemorySCF.put (eraseEnv env) binding.name datum) (names rest) tail
          rw [induction _ values tail remaining, erase_put env binding.name value datum first]

end Merlin.GraphLoopsShape.Epoch
