/- Real initialized tensor arguments construct the source entry relation and the
graph's typed initial store. Future cells are explicit zero defaults; actual
tensor.empty operations are separately evaluated to poison by the source proof. -/
import Merlin.GraphRegionsOperationalProgram

namespace Merlin.GraphRegionsSource
open MerlinVerify

def TensorArguments (types : TensorTypes) : List Value :=
  types.map fun (id, type) => ⟨id, type.ty⟩

def InputsDomain (types : TensorTypes) (values : List Tensor) : Prop :=
  List.Forall₂ (fun pair tensor => pair.2.initialized tensor) types values

def tensorInputs : TensorTypes → List Tensor → List (Nat × Tensor)
  | (id, _) :: types, tensor :: tensors => (id, tensor) :: tensorInputs types tensors
  | _, _ => []

def entryEnvironment (types : TensorTypes) (values : List Tensor) : Env :=
  (tensorInputs types values).map fun (id, tensor) => (id, RVal.tensor tensor)

def entryState (types : Nat → TensorType) (arguments : TensorTypes) (values : List Tensor) : Nat → Tensor :=
  fun id => ((tensorInputs arguments values).lookup id).getD (defaultTensor (types id))

theorem input_lookup {arguments : TensorTypes} {values : List Tensor} (domain : InputsDomain arguments values)
    (id : Nat) :
    match arguments.lookup id with
    | none => (tensorInputs arguments values).lookup id = none
    | some type => ∃ tensor, (tensorInputs arguments values).lookup id = some tensor ∧ type.initialized tensor := by
  induction domain with
  | nil => rfl
  | @cons pair tensor arguments values head tail ih =>
    by_cases same : id = pair.1
    · subst id
      simp [List.lookup, tensorInputs, head]
    · have different : (id == pair.1) = false := beq_eq_false_iff_ne.mpr same
      simpa [List.lookup, tensorInputs, different] using ih

theorem input_environment_lookup (pairs : List (Nat × Tensor)) (id : Nat) :
    (pairs.map (fun (key, tensor) => (key, RVal.tensor tensor))).lookup id =
      (pairs.lookup id).map RVal.tensor := by
  induction pairs with
  | nil => rfl
  | cons pair pairs ih =>
    cases equal : (id == pair.1) <;> simp [List.lookup, equal, ih]

theorem entry_related {arguments : TensorTypes} {values : List Tensor}
    (domain : InputsDomain arguments values) (types : Nat → TensorType) :
    EnvRel arguments (entryEnvironment arguments values) (entryState types arguments values) := by
  intro id type found
  have lookup := input_lookup domain id
  rw [found] at lookup
  obtain ⟨tensor, bound, initialized⟩ := lookup
  exact ⟨by simp [entryEnvironment, Env.get?, input_environment_lookup, entryState, bound],
    by simpa [entryState, bound] using initialized⟩

theorem lookup_binding_member {arguments : TensorTypes} {id : Nat} {type : TensorType}
    (found : arguments.lookup id = some type) : (id, type) ∈ arguments := by
  induction arguments with
  | nil => simp [List.lookup] at found
  | cons pair arguments ih =>
    by_cases same : id = pair.1
    · have typed : type = pair.2 := by simpa [List.lookup, same] using found.symm
      simp [same, typed]
    · have different : (id == pair.1) = false := beq_eq_false_iff_ne.mpr same
      exact List.mem_cons_of_mem _ (ih (by simpa [List.lookup, different] using found))

def InputTypesBound (types : Nat → TensorType) (arguments : TensorTypes) : Prop :=
  ∀ pair ∈ arguments, types pair.1 = pair.2

/-- The global store typing required by the earlier graph theorem is constructed
from the input-only domain and finite checked type bindings, not assumed. -/
theorem entry_typed {arguments : TensorTypes} {values : List Tensor}
    (domain : InputsDomain arguments values) (types : Nat → TensorType)
    (bound : InputTypesBound types arguments) :
    GraphRegions.WellTyped (fun slot tensor => (types slot).initialized tensor) (entryState types arguments values) := by
  intro id
  have lookup := input_lookup domain id
  cases found : arguments.lookup id with
  | none =>
    rw [found] at lookup
    simpa [entryState, lookup] using default_initialized (types id)
  | some type =>
    rw [found] at lookup
    obtain ⟨tensor, actual, initialized⟩ := lookup
    have agrees : types id = type := bound (id, type) (lookup_binding_member found)
    simpa [entryState, actual, agrees] using initialized

theorem bind_function_inputs {arguments : TensorTypes} {values : List Tensor}
    (domain : InputsDomain arguments values) :
    Env.bindAll [] (TensorArguments arguments) (values.map RVal.tensor) =
      .ok (entryEnvironment arguments values) := by
  have arity : arguments.length = values.length := domain.length_eq
  simp only [Env.bindAll, TensorArguments, List.length_map, arity, ↓reduceIte]
  congr 1
  induction domain with
  | nil => rfl
  | @cons pair tensor arguments values head tail ih =>
    simp only [List.map_cons, List.zip_cons_cons, List.append_nil, entryEnvironment, tensorInputs]
    congr 1
    simpa only [entryEnvironment, List.append_nil] using ih (by simpa using arity)

end Merlin.GraphRegionsSource
