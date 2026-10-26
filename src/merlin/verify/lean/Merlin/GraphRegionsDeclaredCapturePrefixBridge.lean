/- Connect actual restricted prefix execution
to region closure: cached captures are read from executed source definitions,
not populated by assuming source success or global typing. Full parent graph
projection, tensor lifting and native qualification are subsequent obligations. -/
import Merlin.GraphRegionsDeclaredCapturePrefix

namespace Merlin.GraphRegionsDeclared.Capture
open MerlinVerify Merlin.GraphRegionsDeclared

def prefixCache {α : Type} (captures names : List Nat) (cells : Prefix.Cells α)
    (arguments : Environment α) : Environment α :=
  fun slot => if slot ∈ captures then (cells (renameSlot names slot)).getD (arguments slot) else arguments slot

theorem ready_congr {α : Type} {meaning : Primitive → List α → α} {captures : List Nat}
    {next : Nat} {instructions : List Instruction} {first second : Environment α}
    (ready : Ready meaning captures next instructions first)
    (same : ∀ slot ∈ captures, first slot = second slot) :
    Ready meaning captures next instructions second := by
  induction instructions generalizing next with
  | nil => trivial
  | cons instruction rest ih =>
    refine ⟨?_, ih ready.2⟩
    intro member
    obtain ⟨operands, literal, value⟩ := ready.1 member
    exact ⟨operands, literal, (same next member).symm.trans value⟩

theorem reference_cache_agrees {α : Type} (meaning : Primitive → List α → α)
    (arguments : Environment α) (initial : Prefix.Cells α)
    {actualPrefix : List Op} {leading : List Prefix.Instruction} {available : List Nat}
    {program : Program} {names : List Nat} {references : List Reference}
    (matchedPrefix : Prefix.Matches actualPrefix leading) (validPrefix : Prefix.Valid available leading)
    (matched : checkReferences actualPrefix program names references = true)
    (eligible : checkEligible (references.map (·.slot)) program.inputs.length program.instructions = true)
    {reference : Reference} (member : reference ∈ references) :
    prefixCache (references.map (·.slot)) names (Prefix.run meaning leading initial) arguments reference.slot =
      cacheLiterals meaning (references.map (·.slot)) program.inputs.length program.instructions arguments reference.slot := by
  obtain ⟨lower, width, literal, instruction, found, named, primitive, empty, definition⟩ :=
    checkReference_sound (checkReferences_each matched reference member)
  have emitted : actualPrefix[reference.prefixIndex]? =
      some ((Prefix.Instruction.literal reference.sourceId width literal).emit) := by
    rw [definition]
    simp only [Instruction.operation, primitive, empty, Primitive.name, Primitive.width, Primitive.attrs,
      Prefix.Instruction.emit]
  have sourceDefinition := Prefix.literal_lookup_of_match matchedPrefix emitted
  have actualValue := Prefix.literal_initialized meaning validPrefix sourceDefinition initial
  have slotMember : reference.slot ∈ references.map (·.slot) := List.mem_map.mpr ⟨reference, member, rfl⟩
  have cacheValue : prefixCache (references.map (·.slot)) names (Prefix.run meaning leading initial) arguments reference.slot =
      meaning (.constant width literal) [] := by
    simp only [prefixCache, slotMember, ↓reduceIte, renameSlot, named, Option.getD_some, actualValue]
  have ready := initialize_ready meaning (references.map (·.slot)) (checkEligible_sound eligible) arguments
  have position : program.inputs.length + (reference.slot - program.inputs.length) = reference.slot := by
    rw [Nat.add_comm]
    exact Nat.sub_add_cancel lower
  have cached := ready_lookup ready found (by simpa only [position] using slotMember)
  rw [position, primitive] at cached
  exact cacheValue.trans cached.symm

theorem prefixCache_inputs {α : Type} {actualPrefix : List Op} {program : Program} {names : List Nat}
    {references : List Reference} (matched : checkReferences actualPrefix program names references = true)
    (cells : Prefix.Cells α) (arguments : Environment α) :
    EnvironmentAgree program.inputs (prefixCache (references.map (·.slot)) names cells arguments) arguments := by
  intro index width found
  have bound := (List.getElem?_eq_some_iff.mp found).choose
  have notCapture : index ∉ references.map (·.slot) := by
    intro member
    obtain ⟨reference, known, equal⟩ := List.mem_map.mp member
    have lower := (checkReference_sound (checkReferences_each matched reference known)).1
    rw [equal] at lower
    exact (Nat.not_le_of_gt bound) lower
  simp only [prefixCache, notCapture, ↓reduceIte]

/-- Static checked prefix syntax/freshness proves the literal input domain; the
runtime formal argument values remain arbitrary initialized mathematical values.
No premise asserting successful execution of the whole source program appears. -/
theorem executed_prefix_closure_result {α : Type} (meaning : Primitive → List α → α)
    (arguments : Environment α) (initial : Prefix.Cells α)
    {actualPrefix : List Op} {leading : List Prefix.Instruction} {available : List Nat}
    {program : Program} {names : List Nat} {references : List Reference}
    (matchedPrefix : Prefix.Matches actualPrefix leading) (validPrefix : Prefix.Valid available leading)
    (matched : checkReferences actualPrefix program names references = true)
    (eligible : checkEligible (references.map (·.slot)) program.inputs.length program.instructions = true)
    (valid : program.Valid) :
    sourceRun meaning (references.map (·.slot)) program.inputs.length program.instructions
      (prefixCache (references.map (·.slot)) names (Prefix.run meaning leading initial) arguments) program.returned =
        program.evaluate meaning arguments := by
  have constructed := initialize_ready meaning (references.map (·.slot)) (checkEligible_sound eligible) arguments
  have ready : Ready meaning (references.map (·.slot)) program.inputs.length program.instructions
      (prefixCache (references.map (·.slot)) names (Prefix.run meaning leading initial) arguments) := by
    apply ready_congr constructed
    intro slot member
    obtain ⟨reference, known, equal⟩ := List.mem_map.mp member
    subst slot
    exact (reference_cache_agrees meaning arguments initial matchedPrefix validPrefix matched eligible known).symm
  exact closed_result meaning (references.map (·.slot)) program valid _ arguments ready
    (prefixCache_inputs matched _ arguments)

end Merlin.GraphRegionsDeclared.Capture
