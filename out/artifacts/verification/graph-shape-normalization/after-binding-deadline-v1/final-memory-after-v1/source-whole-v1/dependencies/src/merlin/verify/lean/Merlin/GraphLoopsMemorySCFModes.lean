/- Exact-middle composition inside ONE operational memory/SSA interpretation.
Segmentation preserves the complete terminal memory, not just public outcomes.
The reuse edge separately relates allocation-changing worlds. -/
import Merlin.GraphLoopsMemorySCFComposition

set_option autoImplicit false

namespace Merlin.GraphLoopsMemorySCF

theorem step_modes {Value Public : Type} {objects : GraphLoopsReuse.Objects} {configs : List Configuration}
    {meaning : Interpretation Value Public} {candidate sourceMode targetMode : Bool}
    {instruction : GraphLoopsReuse.Instruction} {before after : GraphLoopsReuse.Memory Value Public}
    (valid : ∀ config ∈ configs, config.Valid objects)
    (run : Step objects configs meaning candidate sourceMode instruction before after) :
    Step objects configs meaning candidate targetMode instruction before after := by
  cases run with
  | ordinary ordinary run => exact .ordinary ordinary run
  | @initialized config member before after run =>
    have configValid := valid config member
    have log := (heap_run_iff_log_mode _ _ _ _ _ _ _ _ configValid.2.1 configValid.2.2.2).mp run
    exact .initialized member before ((heap_run_iff_log_mode _ _ _ _ _ _ _ _ configValid.2.1 configValid.2.2.2).mpr log)

/-- Whole-list congruence applies to ANY number of changed initializers, with
identical surrounding observers, effects, allocations and releases. -/
theorem execution_modes {Value Public : Type} {objects : GraphLoopsReuse.Objects} {configs : List Configuration}
    {meaning : Interpretation Value Public} {candidate sourceMode targetMode : Bool}
    {instructions : List GraphLoopsReuse.Instruction} {before after : GraphLoopsReuse.Memory Value Public}
    (valid : ∀ config ∈ configs, config.Valid objects)
    (run : Execution objects configs meaning candidate sourceMode instructions before after) :
    Execution objects configs meaning candidate targetMode instructions before after := by
  induction run with
  | done before => exact .done before
  | more first remaining induction => exact .more (step_modes valid first) induction

theorem outcomes_modes {Value Public : Type} (objects : GraphLoopsReuse.Objects) (configs : List Configuration)
    (valid : ∀ config ∈ configs, config.Valid objects) (meaning : Interpretation Value Public)
    (candidate sourceMode targetMode : Bool) (instructions : List GraphLoopsReuse.Instruction)
    (before : GraphLoopsReuse.Memory Value Public) (outcome : Public) :
    outcomes objects configs meaning candidate sourceMode instructions before outcome ↔
      outcomes objects configs meaning candidate targetMode instructions before outcome := by
  constructor <;> rintro ⟨after, run, observed⟩ <;> exact ⟨after, execution_modes valid run, observed⟩

/-- Explicit adjacent composition: source→EXACT reused middle→segmented final.
The middle is the SAME instruction sequence, interpretation and initial memory.
There is no evaluator bridge, whole-program equality hypothesis or numeric law. -/
theorem checked_reuse_then_segment {Value Public : Type} (objects : GraphLoopsReuse.Objects) (configs : List Configuration)
    (instructions : List GraphLoopsReuse.Instruction) (valid : ∀ config ∈ configs, config.Valid objects)
    (accepted : GraphLoopsReuse.checkProgram objects GraphLoopsReuse.emptyStatus instructions = true)
    (meaning : Interpretation Value Public) (source middle : GraphLoopsReuse.Memory Value Public)
    (same : source.observed = middle.observed) (outcome : Public) :
    (outcomes objects configs meaning false false instructions source outcome ↔
      outcomes objects configs meaning true false instructions middle outcome) ∧
    (outcomes objects configs meaning true false instructions middle outcome ↔
      outcomes objects configs meaning true true instructions middle outcome) ∧
    (outcomes objects configs meaning false false instructions source outcome ↔
      outcomes objects configs meaning true true instructions middle outcome) := by
  have reuse := checked_complete_observations objects configs instructions valid accepted meaning false false source middle same outcome
  have segment := outcomes_modes objects configs valid meaning true false true instructions middle outcome
  exact ⟨reuse, segment, reuse.trans segment⟩

/-- Existing atomic-reuse root instantiated with the same INDUCED initializer
meaning. The operational theorem above is stronger about stores, not an assumed
identity of atomic heaps (which differ outside the valid allocation domain). -/
theorem checked_induced_atomic {Value Public : Type} (objects : GraphLoopsReuse.Objects) (configs : List Configuration)
    (instructions : List GraphLoopsReuse.Instruction)
    (accepted : GraphLoopsReuse.checkProgram objects GraphLoopsReuse.emptyStatus instructions = true)
    (meaning : Interpretation Value Public) (source target : GraphLoopsReuse.Memory Value Public)
    (same : source.observed = target.observed) (outcome : Public) :
    (∃ after, GraphLoopsReuse.Execution objects (inducedPrimitives meaning configs) false instructions source after ∧ after.observed = outcome) ↔
    (∃ after, GraphLoopsReuse.Execution objects (inducedPrimitives meaning configs) true instructions target after ∧ after.observed = outcome) :=
  GraphLoopsReuse.checked_observations objects instructions accepted (inducedPrimitives meaning configs) source target same outcome

end Merlin.GraphLoopsMemorySCF
