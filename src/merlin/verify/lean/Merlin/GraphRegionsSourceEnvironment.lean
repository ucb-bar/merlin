/- Direct source-boundary environment relation. Logical tensors are initialized;
allocation bindings are tracked separately as poison and never asserted initialized.
This is local to the source bridge, not a generic state-transport framework. -/
import Merlin.GraphRegionsOperationalAction

namespace Merlin.GraphRegionsSource
open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops

abbrev TensorTypes := List (Nat × TensorType)

def EnvRel (types : TensorTypes) (environment : Env) (state : Nat → Tensor) : Prop :=
  ∀ id type, types.lookup id = some type →
    environment.get? id = some (.tensor (state id)) ∧ type.initialized (state id)

def poisonTensor (type : TensorType) : Tensor := Tensor.full type.shape (.int type.width) .poison

def PoisonRel (types : TensorTypes) (environment : Env) : Prop :=
  ∀ id type, types.lookup id = some type → environment.get? id = some (.tensor (poisonTensor type))

theorem lookup_key_member {types : List (Nat × α)} {id : Nat} {type : α}
    (found : types.lookup id = some type) : id ∈ types.map Prod.fst := by
  induction types with
  | nil => simp [List.lookup] at found
  | cons pair types ih =>
    by_cases same : id = pair.1
    · simp [same]
    · have different : (id == pair.1) = false := beq_eq_false_iff_ne.mpr same
      have old : types.lookup id = some type := by simpa [List.lookup, different] using found
      exact List.mem_cons_of_mem _ (ih old)

theorem envRel_fresh {types : TensorTypes} {environment : Env} {state : Nat → Tensor}
    (related : EnvRel types environment state) {result : Nat} (fresh : result ∉ types.map Prod.fst)
    (value : RVal) : EnvRel types ((result, value) :: environment) state := by
  intro id type found
  have distinct : id ≠ result := fun equal => fresh (equal ▸ lookup_key_member found)
  have different : (id == result) = false := beq_eq_false_iff_ne.mpr distinct
  exact ⟨by simpa [Env.get?, List.lookup, different] using (related id type found).1,
    (related id type found).2⟩

theorem poisonRel_fresh {types : TensorTypes} {environment : Env}
    (related : PoisonRel types environment) {result : Nat} (fresh : result ∉ types.map Prod.fst)
    (value : RVal) : PoisonRel types ((result, value) :: environment) := by
  intro id type found
  have distinct : id ≠ result := fun equal => fresh (equal ▸ lookup_key_member found)
  have different : (id == result) = false := beq_eq_false_iff_ne.mpr distinct
  simpa [Env.get?, List.lookup, different] using related id type found

theorem envRel_put {types : TensorTypes} {environment : Env} {state : Nat → Tensor}
    (related : EnvRel types environment state) (result : Nat) (type : TensorType) (value : Tensor)
    (initialized : type.initialized value) :
    EnvRel ((result, type) :: types) ((result, .tensor value) :: environment) (put state result value) := by
  intro id ty found
  by_cases same : id = result
  · subst id
    have equal : ty = type := by simpa [List.lookup] using found.symm
    subst ty
    exact ⟨by simp [Env.get?, List.lookup, put], by simpa [put] using initialized⟩
  · have different : (id == result) = false := beq_eq_false_iff_ne.mpr same
    have old : types.lookup id = some ty := by simpa [List.lookup, different] using found
    exact ⟨by simpa [Env.get?, List.lookup, put, same, different] using (related id ty old).1,
      by simpa [put, same] using (related id ty old).2⟩

theorem poisonRel_put {types : TensorTypes} {environment : Env}
    (related : PoisonRel types environment) (result : Nat) (type : TensorType) :
    PoisonRel ((result, type) :: types) ((result, .tensor (poisonTensor type)) :: environment) := by
  intro id ty found
  by_cases same : id = result
  · subst id
    have equal : ty = type := by simpa [List.lookup] using found.symm
    subst ty
    simp [Env.get?, List.lookup]
  · have different : (id == result) = false := beq_eq_false_iff_ne.mpr same
    have old : types.lookup id = some ty := by simpa [List.lookup, different] using found
    simpa [Env.get?, List.lookup, different] using related id ty old

def Stage.InputsTyped (types : TensorTypes) (stage : Stage) : Prop :=
  List.Forall₂ (fun id scalar => types.lookup id = some (⟨stage.shape, scalar.2⟩ : TensorType))
    stage.inputs stage.body.inputs

theorem source_inputs_initialized {types : TensorTypes} {stage : Stage} {environment : Env} {state : Nat → Tensor}
    (related : EnvRel types environment state) (typed : stage.InputsTyped types) :
    InputsInitialized stage.body.inputs stage.shape (stage.inputs.map state) := by
  unfold Stage.InputsTyped at typed
  generalize stage.inputs = ids at typed ⊢
  generalize stage.body.inputs = scalars at typed ⊢
  induction typed with
  | nil => exact .nil
  | @cons id scalar ids scalars head tail ih =>
    exact .cons (related id _ head).2 ih

theorem source_inputs_bound {types : TensorTypes} {stage : Stage} {environment : Env} {state : Nat → Tensor}
    (related : EnvRel types environment state) (typed : stage.InputsTyped types) :
    environment.getAll stage.inputs = .ok ((stage.inputs.map state).map RVal.tensor) := by
  unfold Stage.InputsTyped at typed
  generalize stage.inputs = ids at typed ⊢
  generalize stage.body.inputs = scalars at typed ⊢
  induction typed with
  | nil => rfl
  | @cons id scalar ids scalars head tail ih =>
    change (Outcome.ofOption "unbound SSA value" (environment.get? id)).bind
      (fun value => (environment.getAll ids).bind (fun rest => .ok (value :: rest))) = _
    rw [(related id _ head).1]
    simp only [Outcome.ofOption, Outcome.bind_ok, ih, List.map_cons]

theorem getAll_append (environment : Env) (first second : List Nat) :
    environment.getAll (first ++ second) =
      (environment.getAll first).bind fun left =>
      (environment.getAll second).bind fun right => .ok (left ++ right) := by
  induction first with
  | nil =>
    change environment.getAll second = (environment.getAll second).bind (fun right => .ok right)
    cases environment.getAll second <;> rfl
  | cons id ids ih =>
    change (Outcome.ofOption "unbound SSA value" (environment.get? id)).bind
      (fun value => (environment.getAll (ids ++ second)).bind (fun rest => .ok (value :: rest))) = _
    rw [ih]
    cases found : environment.get? id <;>
      simp [Env.getAll, Outcome.mapM, Outcome.ofOption, Outcome.bind_assoc, Outcome.bind_ok, found]

end Merlin.GraphRegionsSource
