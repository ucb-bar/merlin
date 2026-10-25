/- Complete per-period memory/completion admission for actual projected effects.
Effect truth, initialized external inputs and defined writes are explicit semantic
premises. This checks safety, not external-output coverage or value preservation
for changed allocations. Physical completion tokens may be reused AFTER drain. -/
import Merlin.GraphLoops

set_option autoImplicit false

namespace Merlin.GraphLoops.Storage

structure Region where
  space : Nat
  address : Nat
  extent : Nat
  deriving DecidableEq, Repr

def Contains (outer inner : Region) : Prop :=
  outer.space = inner.space ∧ 0 < inner.extent ∧ outer.address ≤ inner.address ∧
    inner.address + inner.extent ≤ outer.address + outer.extent

instance (outer inner : Region) : Decidable (Contains outer inner) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _))

def Separated (left right : Region) : Prop :=
  left.space ≠ right.space ∨ ResourceGeometry.Disjoint left.address left.extent right.address right.extent

instance (left right : Region) : Decidable (Separated left right) :=
  inferInstanceAs (Decidable (_ ∨ _))

structure Access where
  region : Region
  writes : Bool
  deriving DecidableEq, Repr

structure Job where
  token : Nat
  accesses : List Access
  deriving DecidableEq, Repr

inductive Effect where
  | synchronous (accesses : List Access)
  | asynchronous (token : Nat) (accesses : List Access)
  | wait (tokens : List Nat)
  deriving DecidableEq, Repr

structure State where
  initialized : List Region
  pending : List Job
  seen : List Nat
  deriving DecidableEq, Repr

def Covered (ranges : List Region) (region : Region) : Prop := ∃ known ∈ ranges, Contains known region

instance (ranges : List Region) (region : Region) : Decidable (Covered ranges region) :=
  inferInstanceAs (Decidable (∃ known ∈ ranges, Contains known region))

/-- A pending operation reserves its read AND write regions until actual completion. -/
def Ready (windows privateObjects : List Region) (state : State) (accesses : List Access) : Prop :=
  (∀ access ∈ accesses, Covered windows access.region) ∧
  (∀ access ∈ accesses, access.writes = false → Covered state.initialized access.region) ∧
  (∀ access ∈ accesses, ∀ job ∈ state.pending, ∀ earlier ∈ job.accesses,
    access.writes = true ∨ earlier.writes = true → Separated access.region earlier.region) ∧
  (∀ access ∈ accesses, ∀ object ∈ privateObjects, ¬ Separated access.region object →
    Covered state.initialized object ∨ (access.writes = true ∧ access.region = object))

instance (windows privateObjects : List Region) (state : State) (accesses : List Access) :
    Decidable (Ready windows privateObjects state accesses) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _))

def writes (accesses : List Access) : List Region :=
  accesses.filterMap (fun access => if access.writes then some access.region else none)

def release : List Job → Nat → Option (Job × List Job)
  | [], _ => none
  | job :: rest, token =>
    if job.token = token then some (job, rest) else do
      let (completed, remaining) ← release rest token
      return (completed, job :: remaining)

def waitAll : List Nat → State → Option State
  | [], state => some state
  | token :: rest, state => do
    let (completed, remaining) ← release state.pending token
    waitAll rest { state with
      pending := remaining
      initialized := writes completed.accesses ++ state.initialized }

def step (windows privateObjects : List Region) (state : State) : Effect → Option State
  | .synchronous accesses =>
    if Ready windows privateObjects state accesses then
      some { state with initialized := writes accesses ++ state.initialized }
    else none
  | .asynchronous token accesses =>
    if Ready windows privateObjects state accesses ∧ token ∉ state.seen then
      some { state with pending := state.pending ++ [⟨token, accesses⟩], seen := token :: state.seen }
    else none
  | .wait tokens => if tokens ≠ [] ∧ tokens.Nodup then waitAll tokens state else none

def execute (windows privateObjects : List Region) : State → List Effect → Option State
  | state, [] => some state
  | state, effect :: rest => do
    let next ← step windows privateObjects state effect
    execute windows privateObjects next rest

/-- No initial private bytes or overlapping objects are smuggled in as inputs.
All ranges are positive and bounded by independently selected allocation windows. -/
def Domain (windows initialized privateObjects : List Region) : Prop :=
  (∀ window ∈ windows, 0 < window.extent) ∧
  (∀ known ∈ initialized, Covered windows known) ∧
  (∀ object ∈ privateObjects, Covered windows object ∧
    ∀ known ∈ initialized, Separated object known) ∧
  (∀ object ∈ privateObjects, ∀ other ∈ privateObjects, object ≠ other → Separated object other) ∧
  privateObjects.Nodup

instance (windows initialized privateObjects : List Region) : Decidable (Domain windows initialized privateObjects) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _ ∧ _))

def checkBlock (windows initialized privateObjects : List Region) (effects : List Effect) : Bool :=
  if Domain windows initialized privateObjects then
    match execute windows privateObjects ⟨initialized, [], []⟩ effects with
  | none => false
  | some final => decide (final.pending = [] ∧ ∀ object ∈ privateObjects, Covered final.initialized object)
  else false

/-- Every period is reopened from independently initialized external memory and
an empty pending set; no private state or pending work is carried implicitly. -/
def checkPeriods (windows initialized privateObjects : List Region) (extent staticCalls : Nat)
    (blocks : List (List Effect)) : Bool :=
  decide (0 < extent ∧ 0 < staticCalls ∧ blocks.length = extent ∧
    ∀ block ∈ blocks, block.length = staticCalls ∧ checkBlock windows initialized privateObjects block = true)

inductive Trace (windows privateObjects : List Region) : State → List Effect → State → Prop where
  | nil (state) : Trace windows privateObjects state [] state
  | cons {before middle after : State} {effect : Effect} {rest : List Effect}
      (transition : step windows privateObjects before effect = some middle)
      (remaining : Trace windows privateObjects middle rest after) : Trace windows privateObjects before (effect :: rest) after

theorem execute_sound (windows privateObjects : List Region) : ∀ state effects final,
    execute windows privateObjects state effects = some final → Trace windows privateObjects state effects final := by
  intro state effects
  induction effects generalizing state with
  | nil => intro final checked; cases checked; exact .nil state
  | cons effect rest induction =>
    intro final checked
    simp only [execute] at checked
    cases selected : step windows privateObjects state effect with
    | none => simp [selected] at checked
    | some next =>
      simp only [selected, bind, Option.bind] at checked
      exact .cons selected (induction next final checked)

theorem synchronous_ready {windows privateObjects : List Region} {state next : State} {accesses : List Access}
    (accepted : step windows privateObjects state (.synchronous accesses) = some next) : Ready windows privateObjects state accesses := by
  simp only [step] at accepted
  by_cases ready : Ready windows privateObjects state accesses
  · exact ready
  · simp [ready] at accepted

theorem asynchronous_ready {windows privateObjects : List Region} {state next : State} {token : Nat} {accesses : List Access}
    (accepted : step windows privateObjects state (.asynchronous token accesses) = some next) :
    Ready windows privateObjects state accesses ∧ token ∉ state.seen := by
  simp only [step] at accepted
  by_cases ready : Ready windows privateObjects state accesses ∧ token ∉ state.seen
  · exact ready
  · simp [ready] at accepted

theorem block_sound {windows initialized privateObjects : List Region} {effects : List Effect}
    (accepted : checkBlock windows initialized privateObjects effects = true) :
    ∃ final, Trace windows privateObjects ⟨initialized, [], []⟩ effects final ∧
      final.pending = [] ∧ ∀ object ∈ privateObjects, Covered final.initialized object := by
  unfold checkBlock at accepted
  split at accepted
  next domain =>
    cases completed : execute windows privateObjects ⟨initialized, [], []⟩ effects with
    | none => simp [completed] at accepted
    | some final =>
      simp only [completed] at accepted
      exact ⟨final, execute_sound _ _ _ _ _ completed, of_decide_eq_true accepted⟩
  next => cases accepted

theorem block_domain {windows initialized privateObjects : List Region} {effects : List Effect}
    (accepted : checkBlock windows initialized privateObjects effects = true) :
    Domain windows initialized privateObjects := by
  unfold checkBlock at accepted
  split at accepted
  · assumption
  · cases accepted

theorem periods_sound {windows initialized privateObjects : List Region} {extent staticCalls : Nat}
    {blocks : List (List Effect)}
    (accepted : checkPeriods windows initialized privateObjects extent staticCalls blocks = true) :
    blocks.length = extent ∧ ∀ block ∈ blocks,
      block.length = staticCalls ∧ ∃ final, Trace windows privateObjects ⟨initialized, [], []⟩ block final ∧
        final.pending = [] ∧ ∀ object ∈ privateObjects, Covered final.initialized object := by
  have facts := of_decide_eq_true accepted
  exact ⟨facts.2.2.1, fun block member =>
    ⟨(facts.2.2.2 block member).1, block_sound (facts.2.2.2 block member).2⟩⟩

/-- Exact list deletion, rather than a candidate assertion that a token ended. -/
inductive Removal (token : Nat) : List Job → Job → List Job → Prop where
  | head (job : Job) (rest : List Job) (named : job.token = token) :
      Removal token (job :: rest) job rest
  | tail (job : Job) {rest remaining : List Job} {completed : Job}
      (later : Removal token rest completed remaining) :
      Removal token (job :: rest) completed (job :: remaining)

theorem release_sound (token : Nat) : ∀ pending completed remaining,
    release pending token = some (completed, remaining) → Removal token pending completed remaining := by
  intro pending
  induction pending with
  | nil => intro completed remaining checked; cases checked
  | cons job rest induction =>
    intro completed remaining checked
    simp only [release] at checked
    by_cases named : job.token = token
    · simp only [named, ↓reduceIte, Option.some.injEq, Prod.mk.injEq] at checked
      rcases checked with ⟨rfl, rfl⟩
      exact .head job rest named
    · simp only [named, ↓reduceIte] at checked
      cases selected : release rest token with
      | none => simp [selected] at checked
      | some result =>
        rcases result with ⟨found, left⟩
        simp only [selected, bind, Option.bind, pure, Option.some.injEq, Prod.mk.injEq] at checked
        rcases checked with ⟨rfl, rfl⟩
        exact .tail job (induction found left selected)

theorem Removal.present {token : Nat} {pending remaining : List Job} {completed : Job}
    (removed : Removal token pending completed remaining) :
    completed ∈ pending ∧ completed.token = token := by
  induction removed with
  | head _ _ named => exact ⟨List.mem_cons_self, named⟩
  | tail _ _ induction => exact ⟨List.mem_cons_of_mem _ induction.1, induction.2⟩

/-- Writes become initialized only after the exact outstanding operation waits. -/
inductive Waits : State → List Nat → State → Prop where
  | nil (state : State) : Waits state [] state
  | cons {state final : State} {token : Nat} {tokens : List Nat} {job : Job} {remaining : List Job}
      (completed : Removal token state.pending job remaining)
      (rest : Waits { state with
        pending := remaining
        initialized := writes job.accesses ++ state.initialized } tokens final) :
      Waits state (token :: tokens) final

theorem waitAll_sound : ∀ tokens state final,
    waitAll tokens state = some final → Waits state tokens final := by
  intro tokens
  induction tokens with
  | nil => intro state final checked; cases checked; exact .nil state
  | cons token tokens induction =>
    intro state final checked
    simp only [waitAll] at checked
    cases selected : release state.pending token with
    | none => simp [selected] at checked
    | some result =>
      rcases result with ⟨job, remaining⟩
      simp only [selected, bind, Option.bind] at checked
      exact .cons (release_sound _ _ _ _ selected) (induction _ _ checked)

inductive SafeStep (windows privateObjects : List Region) : State → Effect → State → Prop where
  | synchronous {state : State} {accesses : List Access}
      (ready : Ready windows privateObjects state accesses) :
      SafeStep windows privateObjects state (.synchronous accesses)
        { state with initialized := writes accesses ++ state.initialized }
  | asynchronous {state : State} {token : Nat} {accesses : List Access}
      (ready : Ready windows privateObjects state accesses) (fresh : token ∉ state.seen) :
      SafeStep windows privateObjects state (.asynchronous token accesses)
        { state with pending := state.pending ++ [⟨token, accesses⟩], seen := token :: state.seen }
  | wait {state final : State} {tokens : List Nat}
      (nonempty : tokens ≠ []) (unique : tokens.Nodup) (completed : Waits state tokens final) :
      SafeStep windows privateObjects state (.wait tokens) final

theorem step_safe {windows privateObjects : List Region} {state final : State} {effect : Effect}
    (checked : step windows privateObjects state effect = some final) :
    SafeStep windows privateObjects state effect final := by
  cases effect with
  | synchronous accesses =>
    have ready := synchronous_ready checked
    simp only [step, ready, ↓reduceIte, Option.some.injEq] at checked
    subst final
    exact .synchronous ready
  | asynchronous token accesses =>
    have ready := asynchronous_ready checked
    simp [step, ready.1, ready.2] at checked
    subst final
    exact .asynchronous ready.1 ready.2
  | wait tokens =>
    simp only [step] at checked
    by_cases valid : tokens ≠ [] ∧ tokens.Nodup
    · simp [valid.1, valid.2] at checked
      exact .wait valid.1 valid.2 (waitAll_sound _ _ _ checked)
    · simp [valid] at checked

/-- Every event, including events after a long good preceding, has a safe step.
The preceding state is reached by exactly the preceding actual effects. -/
theorem Trace.event {windows privateObjects : List Region} {state final : State}
    {effects : List Effect} (trace : Trace windows privateObjects state effects final) :
    ∀ preceding effect suffix, effects = preceding ++ effect :: suffix →
      ∃ before after, Trace windows privateObjects state preceding before ∧
        SafeStep windows privateObjects before effect after ∧
        Trace windows privateObjects after suffix final := by
  induction trace with
  | nil state =>
    intro preceding effect suffix equal
    have : 0 = preceding.length + (suffix.length + 1) := by simpa using congrArg List.length equal
    omega
  | @cons before middle after current rest transition remaining induction =>
    intro preceding effect suffix equal
    cases preceding with
    | nil =>
      simp only [List.nil_append, List.cons.injEq] at equal
      rcases equal with ⟨rfl, rfl⟩
      exact ⟨before, middle, .nil before, step_safe transition, remaining⟩
    | cons first preceding =>
      simp only [List.cons_append, List.cons.injEq] at equal
      obtain ⟨rfl, equal⟩ := equal
      obtain ⟨pointState, next, earlier, safe, later⟩ := induction preceding effect suffix equal
      exact ⟨pointState, next, .cons transition earlier, safe, later⟩

/-- Interval admission is linked to the existing resource geometry predicate. -/
theorem contained_point {outer inner : Region} (contained : Contains outer inner) {point : Nat}
    (inside : inner.address ≤ point ∧ point < inner.address + inner.extent) :
    outer.address ≤ point ∧ point < outer.address + outer.extent := by
  unfold Contains at contained
  omega

theorem pending_no_overwrite {windows privateObjects : List Region} {state : State} {accesses : List Access}
    (ready : Ready windows privateObjects state accesses) {access : Access} (current : access ∈ accesses)
    {job : Job} (pending : job ∈ state.pending) {earlier : Access} (used : earlier ∈ job.accesses)
    (writes : access.writes = true ∨ earlier.writes = true)
    (sameSpace : access.region.space = earlier.region.space) {point : Nat} :
    ¬ (access.region.address ≤ point ∧ point < access.region.address + access.region.extent ∧
      earlier.region.address ≤ point ∧ point < earlier.region.address + earlier.region.extent) := by
  have separated := ready.2.2.1 access current job pending earlier used writes
  rcases separated with different | disjoint
  · exact (different sameSpace).elim
  · exact ResourceGeometry.disjointRanges _ _ _ _ point disjoint

end Merlin.GraphLoops.Storage
