/- Graham D3 independent witness: the premises of the declared-call root
`actual_complete_observers` (qualification-v1 fresh-replay/0/Check.lean, i8 [3,47]) are
JOINTLY satisfiable by one concrete nonzero instance, so the root is not vacuous.

The sealed query's own `actual_nonzero_witness` only evaluates the logical stages; it
does not exhibit a machine (`read`/`invoke`), packing, native Action and publication
relation satisfying `InstructionContract`, `Refines` and `publication` together. This
file constructs one. It is a mathematical model machine (one tensor per backing,
synchronous, never faulting); it says NOTHING about the real target instruction,
packing or ABI fidelity, which remain reviewed contracts. -/
import Check

set_option autoImplicit false
set_option linter.unusedSimpArgs false

open Merlin MerlinVerify Merlin.GraphRegions Merlin.GraphRegionsDeclared

namespace GrahamD3.DeclaredJoint

abbrev V := Tensor 141 Nat
abbrev Machine := Nat → V

/-- Reads only whole 141-byte windows starting at 0, the only shape the actual calls use. -/
def read (memory : Machine) (window : Window) : Option V :=
  if window.start = 0 ∧ window.bytes = 141 then some (memory window.backing) else none

def programOf (call : BoundCall) : Program :=
  if call = bound0 then program0 else if call = bound1 then program1 else program2

/-- Synchronous, non-faulting model instruction: overwrite exactly the output backing. -/
def invoke (call : BoundCall) (memory : Machine) : Option Machine :=
  some (fun backing => if backing = call.output.backing then
    tensorEvaluate (programOf call) witnessPrimitive (call.reads.map (fun window => memory window.backing))
    else memory backing)

/-- Inverse of the actual `addresses` table on its seven used backings. -/
def slotOfBacking (backing : Nat) : Nat :=
  if backing = 0 then 2 else if backing = 1 then 0 else if backing = 2 then 3 else
  if backing = 6 then 9 else if backing = 3 then 1 else if backing = 4 then 16 else
  if backing = 5 then 21 else 0

def pack (entry : StateEffects.State V) : Machine := fun backing => entry (slotOfBacking backing)

/-- A frame-respecting fallback for unrelated nodes (writes reset to default). -/
def fallback (node : Node) : TypedAction node V (fun _ _ => True) where
  action := {
    step := fun state slot => if slot ∈ node.footprint.writes then default else state slot
    frame := by intro state slot outside; simp [outside]
    locality := by intro first second _ slot inside; simp [inside] }
  preservesTypes := by intros; trivial

abbrev region := (actualGraph.nodes.drop 4).take 3
abbrev suffix := actualGraph.nodes.drop 7
abbrev meaning := registry witnessPrimitive fallback (items.map CallStage.stage)
def native := logicalFusion meaning region suffix actualGraph.observations

theorem map_some_injective {α : Type} : ∀ {first second : List α}, first.map some = second.map some → first = second
  | [], [], _ => rfl
  | [], _ :: _, equal => by simp at equal
  | _ :: _, [], equal => by simp at equal
  | a :: as, b :: bs, equal => by
    simp only [List.map_cons, List.cons.injEq, Option.some.injEq] at equal
    rw [equal.1, map_some_injective equal.2]

theorem reads_shape (call : BoundCall) (memory : Machine) (arguments : List V)
    (whole : ∀ window ∈ call.reads, window.start = 0 ∧ window.bytes = 141)
    (hyp : call.reads.map (read memory) = arguments.map some) :
    arguments = call.reads.map (fun window => memory window.backing) := by
  have same : call.reads.map (read memory) = (call.reads.map (fun window => memory window.backing)).map some := by
    rw [List.map_map]
    apply List.map_congr_left
    intro window member
    simp [read, whole window member]
  rw [same] at hyp
  exact (map_some_injective hyp).symm

theorem contract_of (call : BoundCall) (program : Program)
    (whole : ∀ window ∈ call.reads, window.start = 0 ∧ window.bytes = 141)
    (outputWhole : call.output.start = 0 ∧ call.output.bytes = 141)
    (selects : programOf call = program) :
    InstructionContract read invoke witnessPrimitive call program where
  capture_compute_commit := by
    intro memory arguments hyp
    have args := reads_shape call memory arguments whole hyp
    refine ⟨_, rfl, ?_, ?_⟩
    · simp [read, outputWhole, selects, args]
    · intro window disjoint
      by_cases shaped : window.start = 0 ∧ window.bytes = 141
      · have different : window.backing ≠ call.output.backing := by
          intro equal
          rcases disjoint with backing | before | after
          · exact backing equal
          · omega
          · omega
        simp [read, shaped, different]
      · simp [read, shaped]

theorem contracts : ∀ item ∈ items,
    InstructionContract read invoke witnessPrimitive item.call item.stage.program := by
  intro item member
  simp only [items, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl
  · exact contract_of bound0 program0 (by decide) (by decide) (by decide)
  · exact contract_of bound1 program1 (by decide) (by decide) (by decide)
  · exact contract_of bound2 program2 (by decide) (by decide) (by decide)

theorem initialized : ∀ entry, Refines read addresses initialKnown entry (pack entry) := by
  intro entry slot member
  simp only [initialKnown, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl | rfl <;> rfl

theorem publication : ∀ entry final, runCalls invoke items (pack entry) = some final →
    ∀ slot ∈ liveOuts region suffix actualGraph.observations,
      read final (addresses slot) = some (native.step entry slot) := by
  intro entry final ran slot member
  obtain ⟨final', ran', refined⟩ := sequence_refines witnessPrimitive addresses items initialKnown
    (checkSequence_sound actual_calls_valid) contracts (initialized entry)
  have same : final = final' := Option.some.inj (ran.symm.trans ran')
  subst same
  have covered : slot ∈ finalKnown initialKnown items := by
    apply actual_output_coverage
    simpa only [actual_boundary] using member
  rw [refined slot covered]
  congr 1
  have law := logicalFusion_law meaning region suffix actualGraph.observations entry (fun _ => trivial) slot member
  have stages := executeStages_nodes witnessPrimitive meaning (items.map CallStage.stage)
    (registry_steps witnessPrimitive fallback _ actual_nodes_unique) entry
  simp only [List.map_map, Function.comp_def] at stages
  rw [native, law, stages]
  rw [show List.map (fun x => x.stage.node) items = region from actual_boundary]

/-- All premises of `actual_complete_observers` hold together for one instance. -/
theorem jointly_satisfiable :
    (∀ item ∈ items, InstructionContract read invoke witnessPrimitive item.call item.stage.program) ∧
    (∀ entry, Refines read addresses initialKnown entry (pack entry)) ∧
    (∀ entry final, runCalls invoke items (pack entry) = some final →
      ∀ slot ∈ liveOuts region suffix actualGraph.observations,
        read final (addresses slot) = some (native.step entry slot)) :=
  ⟨contracts, initialized, publication⟩

/-- The sealed root therefore yields a real (non-vacuous) observer equality here.
Stated as a `def` holding the proof so its inferred type is exactly the root's conclusion. -/
def instantiated := actual_complete_observers witnessPrimitive fallback read invoke contracts pack
  initialized native publication witnessInitial

/-- The model machine actually runs all three calls and publishes the nonzero
consumer values 33 and 242 in every element. -/
theorem machine_publishes_nonzero : ∃ final, runCalls invoke items (pack witnessInitial) = some final ∧
    ∀ index : Fin 141, final 4 index = 33 ∧ final 5 index = 242 := by
  refine ⟨_, rfl, ?_⟩
  intro index
  have w := actual_nonzero_witness index
  simp [runCalls, invoke, items, bound0, bound1, bound2, programOf, pack, slotOfBacking,
    tensorEvaluate, Program.evaluate, witnessPrimitive, witnessInitial, stage0, stage1, stage2,
    program0, program1, program2, Merlin.GraphRegionsDeclared.run, Merlin.GraphRegionsDeclared.bind] at w ⊢


/-- Generic form: for ANY machine/packing meeting the per-instruction contracts and the
initialized-packing premise, `publication` holds for the logical fused Action. So, for
that canonical `native`, publication is a consequence of the finer premises and does
not smuggle in region or whole-program equality. -/
theorem publication_derivable {α Mach : Type} [Inhabited α] (primitive : Primitive → List α → α)
    (fallbackα : ∀ node, TypedAction node (Tensor 141 α) (fun _ _ => True))
    (readα : Mach → Window → Option (Tensor 141 α)) (invokeα : BoundCall → Mach → Option Mach)
    (contractsα : ∀ item ∈ items, InstructionContract readα invokeα primitive item.call item.stage.program)
    (packα : StateEffects.State (Tensor 141 α) → Mach)
    (initializedα : ∀ entry, Refines readα addresses initialKnown entry (packα entry)) :
    ∀ entry final, runCalls invokeα items (packα entry) = some final →
      ∀ slot ∈ liveOuts region suffix actualGraph.observations,
        readα final (addresses slot) =
          some ((logicalFusion (registry primitive fallbackα (items.map CallStage.stage))
            region suffix actualGraph.observations).step entry slot) := by
  intro entry final ran slot member
  obtain ⟨final', ran', refined⟩ := sequence_refines primitive addresses items initialKnown
    (checkSequence_sound actual_calls_valid) contractsα (initializedα entry)
  have same : final = final' := Option.some.inj (ran.symm.trans ran')
  subst same
  have covered : slot ∈ finalKnown initialKnown items := by
    apply actual_output_coverage
    simpa only [actual_boundary] using member
  rw [refined slot covered]
  congr 1
  have law := logicalFusion_law (registry primitive fallbackα (items.map CallStage.stage)) region suffix
    actualGraph.observations entry (fun _ => trivial) slot member
  have stages := executeStages_nodes primitive (registry primitive fallbackα (items.map CallStage.stage))
    (items.map CallStage.stage) (registry_steps primitive fallbackα _ actual_nodes_unique) entry
  simp only [List.map_map, Function.comp_def] at stages
  rw [law, stages]
  rw [show List.map (fun x => x.stage.node) items = region from actual_boundary]

end GrahamD3.DeclaredJoint

#print axioms GrahamD3.DeclaredJoint.contracts
#print axioms GrahamD3.DeclaredJoint.initialized
#print axioms GrahamD3.DeclaredJoint.publication
#print axioms GrahamD3.DeclaredJoint.jointly_satisfiable
#print axioms GrahamD3.DeclaredJoint.instantiated
#print axioms GrahamD3.DeclaredJoint.machine_publishes_nonzero
#print axioms GrahamD3.DeclaredJoint.publication_derivable
