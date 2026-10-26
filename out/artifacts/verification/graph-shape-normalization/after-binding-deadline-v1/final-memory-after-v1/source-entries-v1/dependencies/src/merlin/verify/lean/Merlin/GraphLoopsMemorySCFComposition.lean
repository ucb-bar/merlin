/- Whole-program preservation for explicit initializer SCF plus private reuse.
Unchanged instructions use the sealed old memory semantics. Initializers execute
actual scoped operations and stores. No whole-program agreement is assumed. -/
import Merlin.GraphLoopsMemorySCFSimulation

set_option autoImplicit false

namespace Merlin.GraphLoopsMemorySCF

structure Configuration where
  object : Nat
  key : String
  program : Initializer
  tile : Nat
  deriving DecidableEq, Repr

def Configuration.Valid (objects : GraphLoopsReuse.Objects) (config : Configuration) : Prop :=
  config.object < objects.length ∧ config.program.WellFormed ∧
  config.program.shape = (GraphLoopsReuse.objectAt objects config.object).shape ∧ 0 < config.tile

instance (objects : GraphLoopsReuse.Objects) (config : Configuration) : Decidable (config.Valid objects) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _))

structure Interpretation (Value Public : Type) where
  pure : PureMeaning Value
  captures : Public → Env Value
  retained : GraphLoopsReuse.Primitives Value Public

/-- The old atomic primitive is INDUCED from actual scalar/loop/store execution.
No initializer key receives an independently postulated whole-result meaning. -/
def AtomicInitialization {Value Public : Type} (meaning : Interpretation Value Public) (configs : List Configuration)
    (key : String) (observed : Public) (values : Nat → Value) : Prop :=
  ∃ config ∈ configs, config.key = key ∧ ∃ writes,
    Run meaning.pure config.program config.tile false (meaning.captures observed) writes ∧
    ∀ cells index, index < GraphLoopsReuse.Geometry.volume config.program.shape →
      applyWrites writes cells index = some (values index)

def inducedPrimitives {Value Public : Type} (meaning : Interpretation Value Public) (configs : List Configuration) :
    GraphLoopsReuse.Primitives Value Public :=
  { meaning.retained with initialization := AtomicInitialization meaning configs }

theorem initialization_induced {Value Public : Type} (meaning : Interpretation Value Public) (configs : List Configuration)
    {config : Configuration} (member : config ∈ configs) (valid : config.program.WellFormed)
    {observed : Public} {writes : Log Value}
    (run : Run meaning.pure config.program config.tile false (meaning.captures observed) writes) :
    ∃ values, (inducedPrimitives meaning configs).initialization config.key observed values := by
  obtain ⟨values, initialized⟩ := run.materializes valid
  exact ⟨values, config, member, rfl, writes, run, initialized⟩

def NonInitializer : GraphLoopsReuse.Instruction → Prop
  | .initialize _ _ => False
  | _ => True

/-- Allocation, public effects, private accesses and release retain precisely the
old semantics. No allocator-failure or pointer-identity observation is added. -/
inductive Step {Value Public : Type} (objects : GraphLoopsReuse.Objects) (configs : List Configuration)
    (meaning : Interpretation Value Public) (candidate segmented : Bool) :
    GraphLoopsReuse.Instruction → GraphLoopsReuse.Memory Value Public → GraphLoopsReuse.Memory Value Public → Prop where
  | ordinary {instruction : GraphLoopsReuse.Instruction} {before after : GraphLoopsReuse.Memory Value Public}
      (ordinary : NonInitializer instruction)
      (run : GraphLoopsReuse.Step objects (inducedPrimitives meaning configs) candidate instruction before after) :
      Step objects configs meaning candidate segmented instruction before after
  | initialized {config : Configuration} (member : config ∈ configs) (before : GraphLoopsReuse.Memory Value Public)
      {after : Heap Value}
      (run : HeapRun meaning.pure config.program config.tile (GraphLoopsReuse.addressOf objects candidate config.object)
        segmented (meaning.captures before.observed) before.heap after) :
      Step objects configs meaning candidate segmented (.initialize config.object config.key) before ⟨before.observed, after⟩

theorem step_forward {Value Public : Type} {objects : GraphLoopsReuse.Objects} {configs : List Configuration}
    {meaning : Interpretation Value Public} {sourceMode targetMode : Bool} {status : GraphLoopsReuse.Status}
    {instruction : GraphLoopsReuse.Instruction} {source target after : GraphLoopsReuse.Memory Value Public}
    (valid : ∀ config ∈ configs, config.Valid objects)
    (admitted : GraphLoopsReuse.Admitted objects status instruction)
    (related : GraphLoopsReuse.Related (GraphLoopsReuse.slotOf objects) (GraphLoopsReuse.extentOf objects) status.ready source target)
    (run : Step objects configs meaning false sourceMode instruction source after) :
    ∃ next, Step objects configs meaning true targetMode instruction target next ∧
      GraphLoopsReuse.Related (GraphLoopsReuse.slotOf objects) (GraphLoopsReuse.extentOf objects)
        (GraphLoopsReuse.next objects status instruction).ready after next := by
  cases run with
  | ordinary ordinary run =>
    obtain ⟨next, translated, relatedNext⟩ := GraphLoopsReuse.step_forward admitted related run
    exact ⟨next, .ordinary ordinary translated, relatedNext⟩
  | @initialized config member before after run =>
    have configValid := valid config member
    obtain ⟨writes, logRun, rfl⟩ := (heap_run_iff_log_mode _ _ _ _ _ _ _ _ configValid.2.1 configValid.2.2.2).mp run
    have targetLog : Run meaning.pure config.program config.tile false (meaning.captures target.observed) writes :=
      related.1 ▸ logRun
    have targetRun := (heap_run_iff_log_mode meaning.pure config.program config.tile (GraphLoopsReuse.slotOf objects config.object)
      targetMode (meaning.captures target.observed) target.heap _ configValid.2.1 configValid.2.2.2).mpr ⟨writes, targetLog, rfl⟩
    refine ⟨⟨target.observed, _⟩, .initialized member target targetRun, ?_⟩
    exact related_initialize_logs related admitted.initialize_separate configValid.2.1 configValid.2.2.1 logRun

theorem step_backward {Value Public : Type} {objects : GraphLoopsReuse.Objects} {configs : List Configuration}
    {meaning : Interpretation Value Public} {sourceMode targetMode : Bool} {status : GraphLoopsReuse.Status}
    {instruction : GraphLoopsReuse.Instruction} {source target after : GraphLoopsReuse.Memory Value Public}
    (valid : ∀ config ∈ configs, config.Valid objects)
    (admitted : GraphLoopsReuse.Admitted objects status instruction)
    (related : GraphLoopsReuse.Related (GraphLoopsReuse.slotOf objects) (GraphLoopsReuse.extentOf objects) status.ready source target)
    (run : Step objects configs meaning true targetMode instruction target after) :
    ∃ next, Step objects configs meaning false sourceMode instruction source next ∧
      GraphLoopsReuse.Related (GraphLoopsReuse.slotOf objects) (GraphLoopsReuse.extentOf objects)
        (GraphLoopsReuse.next objects status instruction).ready next after := by
  cases run with
  | ordinary ordinary run =>
    obtain ⟨next, translated, relatedNext⟩ := GraphLoopsReuse.step_backward admitted related run
    exact ⟨next, .ordinary ordinary translated, relatedNext⟩
  | @initialized config member before after run =>
    have configValid := valid config member
    obtain ⟨writes, logRun, rfl⟩ := (heap_run_iff_log_mode _ _ _ _ _ _ _ _ configValid.2.1 configValid.2.2.2).mp run
    have sourceLog : Run meaning.pure config.program config.tile false (meaning.captures source.observed) writes :=
      related.1.symm ▸ logRun
    have sourceRun := (heap_run_iff_log_mode meaning.pure config.program config.tile config.object
      sourceMode (meaning.captures source.observed) source.heap _ configValid.2.1 configValid.2.2.2).mpr ⟨writes, sourceLog, rfl⟩
    refine ⟨⟨source.observed, _⟩, .initialized member source sourceRun, ?_⟩
    exact related_initialize_logs related admitted.initialize_separate configValid.2.1 configValid.2.2.1 sourceLog

inductive Execution {Value Public : Type} (objects : GraphLoopsReuse.Objects) (configs : List Configuration)
    (meaning : Interpretation Value Public) (candidate segmented : Bool) :
    List GraphLoopsReuse.Instruction → GraphLoopsReuse.Memory Value Public → GraphLoopsReuse.Memory Value Public → Prop where
  | done (before : GraphLoopsReuse.Memory Value Public) : Execution objects configs meaning candidate segmented [] before before
  | more {instruction : GraphLoopsReuse.Instruction} {rest : List GraphLoopsReuse.Instruction}
      {before middle after : GraphLoopsReuse.Memory Value Public}
      (first : Step objects configs meaning candidate segmented instruction before middle)
      (remaining : Execution objects configs meaning candidate segmented rest middle after) :
      Execution objects configs meaning candidate segmented (instruction :: rest) before after

theorem execution_forward {Value Public : Type} {objects : GraphLoopsReuse.Objects} {configs : List Configuration}
    {meaning : Interpretation Value Public} {sourceMode targetMode : Bool} {status : GraphLoopsReuse.Status}
    {instructions : List GraphLoopsReuse.Instruction} (valid : ∀ config ∈ configs, config.Valid objects)
    (safe : GraphLoopsReuse.Safe objects status instructions) :
    ∀ source target after, GraphLoopsReuse.Related (GraphLoopsReuse.slotOf objects) (GraphLoopsReuse.extentOf objects) status.ready source target →
      Execution objects configs meaning false sourceMode instructions source after →
      ∃ next, Execution objects configs meaning true targetMode instructions target next ∧ after.observed = next.observed := by
  induction safe with
  | done =>
    intro source target after related run
    cases run
    exact ⟨target, .done target, related.1⟩
  | step admitted remaining induction =>
    intro source target after related run
    cases run with
    | more first suffix =>
      obtain ⟨middle, translated, same⟩ := step_forward valid admitted related first
      obtain ⟨next, translatedRest, observed⟩ := induction _ middle _ same suffix
      exact ⟨next, .more translated translatedRest, observed⟩

theorem execution_backward {Value Public : Type} {objects : GraphLoopsReuse.Objects} {configs : List Configuration}
    {meaning : Interpretation Value Public} {sourceMode targetMode : Bool} {status : GraphLoopsReuse.Status}
    {instructions : List GraphLoopsReuse.Instruction} (valid : ∀ config ∈ configs, config.Valid objects)
    (safe : GraphLoopsReuse.Safe objects status instructions) :
    ∀ source target after, GraphLoopsReuse.Related (GraphLoopsReuse.slotOf objects) (GraphLoopsReuse.extentOf objects) status.ready source target →
      Execution objects configs meaning true targetMode instructions target after →
      ∃ next, Execution objects configs meaning false sourceMode instructions source next ∧ next.observed = after.observed := by
  induction safe with
  | done =>
    intro source target after related run
    cases run
    exact ⟨source, .done source, related.1⟩
  | step admitted remaining induction =>
    intro source target after related run
    cases run with
    | more first suffix =>
      obtain ⟨middle, translated, same⟩ := step_backward valid admitted related first
      obtain ⟨next, translatedRest, observed⟩ := induction middle _ _ same suffix
      exact ⟨next, .more translated translatedRest, observed⟩

def outcomes {Value Public : Type} (objects : GraphLoopsReuse.Objects) (configs : List Configuration)
    (meaning : Interpretation Value Public) (candidate segmented : Bool) (instructions : List GraphLoopsReuse.Instruction)
    (before : GraphLoopsReuse.Memory Value Public) (outcome : Public) : Prop :=
  ∃ after, Execution objects configs meaning candidate segmented instructions before after ∧ after.observed = outcome

/-- ALL possible COMPLETE public outcomes for the entire logical sequence,
including arbitrarily many changed initializers and retained observers. Does not
identify categorized failures/divergence or infer primitive/allocator totality. -/
theorem checked_complete_observations {Value Public : Type} (objects : GraphLoopsReuse.Objects) (configs : List Configuration)
    (instructions : List GraphLoopsReuse.Instruction) (valid : ∀ config ∈ configs, config.Valid objects)
    (accepted : GraphLoopsReuse.checkProgram objects GraphLoopsReuse.emptyStatus instructions = true)
    (meaning : Interpretation Value Public) (sourceMode targetMode : Bool)
    (source target : GraphLoopsReuse.Memory Value Public) (same : source.observed = target.observed) (outcome : Public) :
    outcomes objects configs meaning false sourceMode instructions source outcome ↔
      outcomes objects configs meaning true targetMode instructions target outcome := by
  have safe := GraphLoopsReuse.checkProgram_sound objects instructions _ accepted
  have related : GraphLoopsReuse.Related (GraphLoopsReuse.slotOf objects) (GraphLoopsReuse.extentOf objects)
      GraphLoopsReuse.emptyStatus.ready source target := ⟨same, by simp [GraphLoopsReuse.emptyStatus]⟩
  constructor
  · rintro ⟨after, run, observed⟩
    obtain ⟨next, translated, equal⟩ := execution_forward valid safe source target after related run
    exact ⟨next, translated, equal.symm.trans observed⟩
  · rintro ⟨after, run, observed⟩
    obtain ⟨next, translated, equal⟩ := execution_backward valid safe source target after related run
    exact ⟨next, translated, equal.trans observed⟩

end Merlin.GraphLoopsMemorySCF
