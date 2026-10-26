/- UNCOMPILED successor. Complete call-free continuation frame law. Runtime
support is constructed at entry and maintained after each SSA write, not a
supplied global intermediate typing invariant. Regions, allocation and calls
are outside this finite grammar. -/
import Merlin.GraphLoopsMemorySCFAfterContextActions

set_option autoImplicit false
namespace Merlin.GraphLoopsMemorySCF.After
open GraphLoopsShape

def PublicSuffix (indices : IndexOpcodes) (opcodes : Epoch.Opcodes) (literals : LiteralOpcodes) : Code → Prop
  | .returned bindings => NoViewBindings bindings
  | .cast _ _ source target rest => source = target ∧ PublicSuffix indices opcodes literals rest
  | .dimension _ _ _ _ _ _ rest | .constant _ _ rest => PublicSuffix indices opcodes literals rest
  | .action key _ results rest => (publicOpcode indices opcodes literals key).isSome = true ∧
      NoViewBindings results ∧ PublicSuffix indices opcodes literals rest
  | .region _ _ _ _ _ _ => False

def publicSuffixDecidable (indices : IndexOpcodes) (opcodes : Epoch.Opcodes) (literals : LiteralOpcodes) :
    (code : Code) → Decidable (PublicSuffix indices opcodes literals code)
  | .returned bindings => inferInstanceAs (Decidable (NoViewBindings bindings))
  | .cast _ _ source target rest =>
      letI := publicSuffixDecidable indices opcodes literals rest
      inferInstanceAs (Decidable (_ ∧ _))
  | .dimension _ _ _ _ _ _ rest | .constant _ _ rest => publicSuffixDecidable indices opcodes literals rest
  | .action key _ results rest =>
      letI := publicSuffixDecidable indices opcodes literals rest
      inferInstanceAs (Decidable (_ ∧ _ ∧ _))
  | .region _ _ _ _ _ _ => isFalse (fun impossible => impossible)

instance (indices : IndexOpcodes) (opcodes : Epoch.Opcodes) (literals : LiteralOpcodes) (code : Code) :
    Decidable (PublicSuffix indices opcodes literals code) := publicSuffixDecidable indices opcodes literals code

def NoSuffixCalls (module : Module) (code : Code) : Prop :=
  ∀ key ∈ actionKeys code, callTarget module.calls key = none

instance (module : Module) (code : Code) : Decidable (NoSuffixCalls module code) := by
  unfold NoSuffixCalls
  infer_instance

/-- Every listed action has a constructorwise frame law from the exact common
interpretation. No arbitrary state-inspecting operation can satisfy the syntax
check merely because it returns a scalar.
-/
theorem suffix_transfer {Value User : Type} (bits bound : Nat) (indices : IndexOpcodes)
    (opcodes : Epoch.Opcodes) (literals : LiteralOpcodes) (scalar : Epoch.BinaryMeaning Value) (literal : LiteralMeaning Value)
    (code : Code) (valid : PublicSuffix indices opcodes literals code)
    (aliases : Aliases) (env : GraphLoopsShape.Env Value) (external : ExternalEnv bound env)
    (before other after : CallerWorld Value User) (same : PublicRelated bound before other)
    (outputs : List (Runtime Value))
    (run : Eval (entryMeaning bits indices opcodes scalar literals literal) false aliases code env before outputs after) :
    ∃ final, Eval (entryMeaning bits indices opcodes scalar literals literal) false aliases code env other outputs final ∧
      PublicRelated bound after final := by
  induction code generalizing aliases env before other after outputs with
  | returned bindings =>
    obtain ⟨returned, rfl⟩ := run
    exact ⟨other, ⟨returned, rfl⟩, same⟩
  | cast origin result source target rest induction =>
    obtain ⟨value, cast, suffix⟩ := run
    have safe := cast_external bound source target (env origin.name) value (external origin.name) cast
    obtain ⟨final, next, related⟩ := induction valid.2 _ _ (external_put bound env external result.name value safe)
      before other after same outputs suffix
    exact ⟨final, ⟨value, cast, next⟩, related⟩
  | dimension buffer axis result type axisIndex value rest induction =>
    obtain ⟨actual, dimension, suffix⟩ := run
    obtain ⟨final, next, related⟩ := induction valid _ _
      (external_put bound env external result.name (.index actual) trivial) before other after same outputs suffix
    exact ⟨final, ⟨actual, dimension, next⟩, related⟩
  | constant result value rest induction =>
    exact induction valid _ _ (external_put bound env external result.name (.index value) trivial)
      before other after same outputs run
  | action key operands results rest induction =>
    obtain ⟨values, middle, action, typed, suffix⟩ := run
    have existsOpcode : ∃ opcode, publicOpcode indices opcodes literals key = some opcode := by
      cases selected : publicOpcode indices opcodes literals key with
      | none =>
        have impossible : false = true := by simpa only [selected, Option.isSome_none] using valid.1
        cases impossible
      | some opcode => exact ⟨opcode, rfl⟩
    obtain ⟨opcode, chosen⟩ := existsOpcode
    have selected := public_action_selected (User := User) bits indices opcodes literals key opcode chosen scalar literal
    rw [selected] at action
    obtain ⟨otherMiddle, otherAction, middleRelated⟩ := opcode.transfer bits bound key scalar literal
      (readArguments false aliases env operands) (external_arguments bound env external aliases operands)
      before other middle same values action
    have safe := external_assign bound env external results values (nonview_values_external bound results values typed valid.2.1)
    obtain ⟨final, next, related⟩ := induction valid.2.2 _ _ safe middle otherMiddle after middleRelated outputs suffix
    refine ⟨final, ⟨values, otherMiddle, ?_, typed, next⟩, related⟩
    rw [selected]
    exact otherAction
  | region => exact False.elim valid

/-- Call-depth dispatch is eliminated only for actual suffix keys checked absent
from the call table; the same primitive interpretation remains in force.
-/
theorem suffix_dispatch {Value World : Type} (meaning : Interpretation Value World) (module : Module)
    (depth : Nat) (code : Code) (noCalls : NoSuffixCalls module code)
    (aliases : Aliases) (env : GraphLoopsShape.Env Value) (before : World) (outputs : List (Runtime Value)) (after : World) :
    Eval (withCalls meaning module (GraphLoopsShape.invoke meaning module false depth)) false aliases code env before outputs after ↔
      Eval meaning false aliases code env before outputs after := by
  apply eval_action_agreement
  · intro key member
    simp only [withCalls, noCalls key member]
  · rfl

end Merlin.GraphLoopsMemorySCF.After
