/- Exact outer integer constants, explicit lexical availability and scalar
closure agreement. Constants are not modeled as initialized tensor allocations. -/
import Merlin.GraphRegionsSourceBody

namespace Merlin.GraphRegionsSource
open MerlinVerify

def ConstantsRel (constants : List CapturedConstant) (environment : Env) : Prop :=
  ∀ constant ∈ constants, environment.get? constant.id = some (scalar constant.width constant.value)

def CapturesAvailable (available used : List CapturedConstant) : Prop :=
  ∀ constant ∈ used, constant ∈ available

theorem constant_lookup {constants : List CapturedConstant} {id width : Nat}
    (found : (constantTypes constants).lookup id = some width) :
    ∃ constant ∈ constants, constant.id = id ∧ constant.width = width ∧ constantState constants id = constant.value := by
  induction constants with
  | nil => simp [constantTypes, List.lookup] at found
  | cons constant constants ih =>
    by_cases same : id = constant.id
    · have type : width = constant.width := by simpa [constantTypes, List.lookup, same] using found.symm
      exact ⟨constant, by simp, same.symm, type.symm, by simp [constantState, List.lookup, same]⟩
    · have different : (id == constant.id) = false := beq_eq_false_iff_ne.mpr same
      have tail : (constantTypes constants).lookup id = some width :=
        by simpa [constantTypes, List.lookup, different] using found
      obtain ⟨selected, member, sourceId, sourceType, value⟩ := ih tail
      exact ⟨selected, List.mem_cons_of_mem _ member, sourceId, sourceType,
        by simpa [constantState, List.lookup, different] using value⟩

theorem constants_agree {available used : List CapturedConstant} {environment : Env}
    (related : ConstantsRel available environment) (captured : CapturesAvailable available used) :
    ScalarAgree (constantTypes used) environment (constantState used) := by
  intro id width found
  obtain ⟨constant, member, rfl, rfl, value⟩ := constant_lookup found
  simpa [value] using related constant (captured constant member)

theorem scalar_lookup_key {types : ScalarTypes} {id width : Nat}
    (found : types.lookup id = some width) : id ∈ types.map Prod.fst := by
  induction types with
  | nil => simp [List.lookup] at found
  | cons pair types ih =>
    by_cases same : id = pair.1
    · simp [same]
    · have different : (id == pair.1) = false := beq_eq_false_iff_ne.mpr same
      exact List.mem_cons_of_mem _ (ih (by simpa [List.lookup, different] using found))

theorem scalar_agree_fresh {types : ScalarTypes} {environment : Env} {values : ScalarState}
    (agree : ScalarAgree types environment values) {result : Nat} (fresh : result ∉ types.map Prod.fst)
    (value : RVal) : ScalarAgree types ((result, value) :: environment) values := by
  intro id width found
  have distinct : id ≠ result := fun equal => fresh (equal ▸ scalar_lookup_key found)
  have different : (id == result) = false := beq_eq_false_iff_ne.mpr distinct
  simpa [Env.get?, List.lookup, different] using agree id width found

theorem constants_fresh {constants : List CapturedConstant} {environment : Env}
    (related : ConstantsRel constants environment) {result : Nat}
    (fresh : result ∉ constants.map CapturedConstant.id) (value : RVal) :
    ConstantsRel constants ((result, value) :: environment) := by
  intro constant member
  have distinct : constant.id ≠ result := by
    intro equal
    apply fresh
    exact equal ▸ List.mem_map_of_mem member
  have different : (constant.id == result) = false := beq_eq_false_iff_ne.mpr distinct
  simpa [Env.get?, List.lookup, different] using related constant member

theorem constants_cons {constants : List CapturedConstant} {environment : Env}
    (related : ConstantsRel constants environment) (constant : CapturedConstant)
    (fresh : constant.id ∉ constants.map CapturedConstant.id) :
    ConstantsRel (constant :: constants) ((constant.id, scalar constant.width constant.value) :: environment) := by
  intro selected member
  rcases List.mem_cons.mp member with rfl | member
  · simp [Env.get?, List.lookup]
  · exact constants_fresh related fresh _ selected member

end Merlin.GraphRegionsSource
