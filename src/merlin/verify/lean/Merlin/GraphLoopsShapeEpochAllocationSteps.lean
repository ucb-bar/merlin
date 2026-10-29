/- Per-allocation/free correspondence, with explicit source allocation order and
prebound logical descriptors. No arbitrary site identifier is handed to an
allocation opcode; its private clock determines the next logical object. -/
import Merlin.GraphLoopsShapeEpochScalarSteps

set_option autoImplicit false

namespace Merlin.GraphLoopsShape.Epoch

theorem put_existing {Value : Type} (env : Env Value) (name : Nat) (value : Runtime Value)
    (existing : env name = value) : put env name value = env := by
  funext other
  by_cases equal : other = name
  · subst other; simp only [put, ↓reduceIte, existing]
  · exact ite_eq_right equal

theorem allocate_step_iff {Value Public : Type} (objects : GraphLoopsReuse.Objects)
    (configs : List GraphLoopsMemorySCF.Configuration) (meaning : GraphLoopsMemorySCF.Interpretation Value Public)
    (object : Nat) (retained : Bool) (before after : GraphLoopsReuse.Memory Value Public) :
    GraphLoopsMemorySCF.Step objects configs meaning false false (.allocate object retained) before after ↔
      after = GraphLoopsReuse.clearMemory objects false object retained before := by
  constructor
  · intro run
    cases run with
    | ordinary ordinary run => cases run; rfl
  · rintro rfl
    exact .ordinary (by trivial) (.allocate object retained before)

theorem release_step_iff {Value Public : Type} (objects : GraphLoopsReuse.Objects)
    (configs : List GraphLoopsMemorySCF.Configuration) (meaning : GraphLoopsMemorySCF.Interpretation Value Public)
    (object : Nat) (retained : Bool) (before after : GraphLoopsReuse.Memory Value Public) :
    GraphLoopsMemorySCF.Step objects configs meaning false false (.release object retained) before after ↔
      after = GraphLoopsReuse.clearMemory objects false object retained before := by
  constructor
  · intro run
    cases run with
    | ordinary ordinary run => cases run; rfl
  · rintro rfl
    exact .ordinary (by trivial) (.release object retained before)

theorem allocate_atom {Value User : Type} (bound object : Nat) (inside : object < bound)
    (objects : GraphLoopsReuse.Objects) (configs : List GraphLoopsMemorySCF.Configuration)
    (old : GraphLoopsMemorySCF.Interpretation Value (PublicState Value User)) (retained : Bool)
    (key : String) (result : Binding) (shape : List Nat) (element : String)
    (kind : result.kind = .view (type shape element)) (plain : result.known = none)
    (meaning : Interpretation Value (MemoryWorld Value (AllocationPublic User)))
    (allocator : meaning.action key = Allocate shape element)
    (before : GraphLoopsReuse.Memory Value (PublicState Value User))
    (pinned : before.observed.registers result.name = view object shape element)
    (env : Env Value) (world : MemoryWorld Value (AllocationPublic User)) :
    (Atom.action key [] [result]).step meaning before.observed.registers (clockWorld bound object before) env world ↔
      ∃ after, GraphLoopsMemorySCF.Step objects configs old false false (.allocate object retained) before after ∧
        env = after.observed.registers ∧ world = clockWorld bound (object + 1) after := by
  simp only [allocate_step_iff, exists_eq_left]
  change (∃ values, meaning.action key [] (clockWorld bound object before) values world ∧
    ValuesHold [result] values ∧ env = assign before.observed.registers [result] values) ↔ _
  rw [allocator]
  simp only [allocation_epoch bound object inside shape element before objects retained]
  have clearObserved : (GraphLoopsReuse.clearMemory objects false object retained before).observed = before.observed := by
    simp only [GraphLoopsReuse.clearMemory, Bool.false_and, Bool.false_eq_true, ↓reduceIte]
  simp only [clearObserved]
  have existing := put_existing before.observed.registers result.name (view object shape element) pinned
  constructor
  · rintro ⟨values, ⟨rfl, world⟩, _, envEqual⟩
    have sameEnv : env = before.observed.registers := by simpa only [assign, existing] using envEqual
    exact ⟨sameEnv, world⟩
  · rintro ⟨env, world⟩
    refine ⟨[view object shape element], ⟨rfl, world⟩,
      .cons (Binding.holds_plain result _ kind.symm plain) .nil, ?_⟩
    simpa only [assign, existing] using env

theorem release_atom {Value User : Type} (bound next object : Nat) (inside : object < bound)
    (objects : GraphLoopsReuse.Objects) (configs : List GraphLoopsMemorySCF.Configuration)
    (old : GraphLoopsMemorySCF.Interpretation Value (PublicState Value User)) (retained : Bool)
    (key : String) (buffer : Binding) (shape : List Nat) (element : String)
    (meaning : Interpretation Value (MemoryWorld Value (AllocationPublic User)))
    (release : meaning.action key = Release)
    (before : GraphLoopsReuse.Memory Value (PublicState Value User))
    (pinned : before.observed.registers buffer.name = view object shape element)
    (env : Env Value) (world : MemoryWorld Value (AllocationPublic User)) :
    (Atom.action key [buffer] []).step meaning before.observed.registers (clockWorld bound next before) env world ↔
      ∃ after, GraphLoopsMemorySCF.Step objects configs old false false (.release object retained) before after ∧
        env = after.observed.registers ∧ world = clockWorld bound next after := by
  simp only [release_step_iff, exists_eq_left]
  change (∃ values, meaning.action key [before.observed.registers buffer.name]
    (clockWorld bound next before) values world ∧ ValuesHold [] values ∧ env = assign before.observed.registers [] values) ↔ _
  rw [release, pinned]
  simp only [release_epoch bound next object inside shape element before objects retained]
  constructor
  · rintro ⟨values, ⟨rfl, world⟩, _, env⟩
    exact ⟨env, world⟩
  · rintro ⟨env, world⟩
    exact ⟨[], ⟨rfl, world⟩, .nil, env⟩

end Merlin.GraphLoopsShape.Epoch
