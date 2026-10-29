/- UNCOMPILED successor. Unchanged scalar literals receive an explicit shared
partial literal interpretation. No value-of-literal/numerical law is assumed or
proved. Literals are effect-free by this disclosed model; fidelity is separate.
Finite closed-support checks derive that selected-kernel execution is unchanged.
-/
import Merlin.GraphLoopsMemorySCFAfterEntriesClosure

set_option autoImplicit false

namespace Merlin.GraphLoopsMemorySCF.After

open GraphLoopsShape

abbrev LiteralOpcodes := List (String × String)
abbrev LiteralMeaning (Value : Type) := String → Value → Prop

def literalAt : LiteralOpcodes → String → Option String
  | [], _ => none
  | (key, element) :: rest, actual => if actual = key then some element else literalAt rest actual

def LiteralAction {Value World : Type} (element key : String) (literal : LiteralMeaning Value)
    (inputs : List (Runtime Value)) (before : World) (outputs : List (Runtime Value)) (after : World) : Prop :=
  inputs = [] ∧ ∃ value, literal key value ∧ outputs = [.scalar element value] ∧ after = before

def withLiterals {Value World : Type} (base : GraphLoopsShape.Interpretation Value World) (opcodes : LiteralOpcodes)
    (literal : LiteralMeaning Value) : GraphLoopsShape.Interpretation Value World where
  action := fun key => match literalAt opcodes key with
    | none => base.action key
    | some element => LiteralAction element key literal
  region := base.region

def LiteralAbsentEntry (module : Module) (literals : LiteralOpcodes) (name : String) : Prop :=
  match functionAt module name with
    | none => True
    | some function => ∀ key ∈ actionKeys function.code, literalAt literals key = none

instance (module : Module) (literals : LiteralOpcodes) (name : String) :
    Decidable (LiteralAbsentEntry module literals name) := by
  unfold LiteralAbsentEntry
  split <;> infer_instance

def ClosedLiteralAbsent (module : Module) (names : List String) (literals : LiteralOpcodes) : Prop :=
  ∀ name ∈ names, LiteralAbsentEntry module literals name

instance (module : Module) (names : List String) (literals : LiteralOpcodes) :
    Decidable (ClosedLiteralAbsent module names literals) := by
  unfold ClosedLiteralAbsent
  infer_instance

def LiteralCodeFacts (literals : LiteralOpcodes) : Code → Prop
  | .returned _ => True
  | .cast _ _ _ _ rest | .dimension _ _ _ _ _ _ rest | .constant _ _ rest => LiteralCodeFacts literals rest
  | .action key operands results rest =>
      (match literalAt literals key with
       | none => True
       | some element => operands = [] ∧ match results with
         | [result] => result.kind = .scalar element ∧ result.known = none
         | _ => False) ∧ LiteralCodeFacts literals rest
  | .region _ _ _ _ body rest => LiteralCodeFacts literals body ∧ LiteralCodeFacts literals rest

def literalCodeDecidable (literals : LiteralOpcodes) : (code : Code) → Decidable (LiteralCodeFacts literals code)
  | .returned _ => isTrue trivial
  | .cast _ _ _ _ rest | .dimension _ _ _ _ _ _ rest | .constant _ _ rest => literalCodeDecidable literals rest
  | .action key operands results rest => by
      letI := literalCodeDecidable literals rest
      unfold LiteralCodeFacts
      split
      · infer_instance
      · cases results with
        | nil => infer_instance
        | cons result rest => cases rest <;> infer_instance
  | .region _ _ _ _ body rest =>
      letI := literalCodeDecidable literals body
      letI := literalCodeDecidable literals rest
      inferInstanceAs (Decidable (_ ∧ _))

instance (literals : LiteralOpcodes) (code : Code) : Decidable (LiteralCodeFacts literals code) := literalCodeDecidable literals code

def LiteralModuleFacts (literals : LiteralOpcodes) (module : Module) : Prop :=
  (literals.map Prod.fst).Nodup ∧
    (∀ function ∈ module.functions, LiteralCodeFacts literals function.code) ∧
    (∀ entry ∈ literals, callTarget module.calls entry.1 = none)

instance (literals : LiteralOpcodes) (module : Module) : Decidable (LiteralModuleFacts literals module) := by
  unfold LiteralModuleFacts
  infer_instance

theorem closed_literal_extension {Value World : Type} (base : GraphLoopsShape.Interpretation Value World)
    (literals : LiteralOpcodes) (literal : LiteralMeaning Value) (module : Module) (names : List String)
    (closed : CallClosed module names) (absent : ClosedLiteralAbsent module names literals)
    (name : String) (member : name ∈ names) (inputs : List (Runtime Value))
    (before : World) (outputs : List (Runtime Value)) (after : World) :
    complete base module false name inputs before outputs after ↔
      complete (withLiterals base literals literal) module false name inputs before outputs after := by
  have actions : ∀ entry ∈ names, ∀ function, functionAt module entry = some function →
      ∀ key ∈ actionKeys function.code, base.action key = (withLiterals base literals literal).action key := by
    intro entry entryMember function found key used
    have missing := absent entry entryMember
    simp only [LiteralAbsentEntry, found] at missing
    simp only [withLiterals, missing key used]
  unfold complete
  apply exists_congr; intro depth
  rw [closed_invoke_extension base (withLiterals base literals literal) module names closed actions rfl depth name member]

def entryMeaning {Value User : Type} (bits : Nat) (indices : IndexOpcodes) (opcodes : Epoch.Opcodes)
    (scalar : Epoch.BinaryMeaning Value) (literals : LiteralOpcodes) (literal : LiteralMeaning Value) :
    GraphLoopsShape.Interpretation Value (MemoryWorld Value (Epoch.AllocationPublic User)) :=
  withLiterals (finalMeaning bits indices opcodes scalar) literals literal

theorem literal_outcomes_extension {Value User : Type} (bound : Nat)
    (base : GraphLoopsShape.Interpretation Value (MemoryWorld Value (Epoch.AllocationPublic User)))
    (literals : LiteralOpcodes) (literal : LiteralMeaning Value) (module : Module) (names : List String)
    (closed : CallClosed module names) (absent : ClosedLiteralAbsent module names literals)
    (name : String) (member : name ∈ names) (inputs : List (Runtime Value)) (heap : Heap Value) (user : User)
    (observed : Epoch.Observation Value User) :
    Epoch.graphOutcomes bound base module name inputs heap user observed ↔
      Epoch.graphOutcomes bound (withLiterals base literals literal) module name inputs heap user observed := by
  unfold Epoch.graphOutcomes
  apply exists_congr; intro returned
  apply exists_congr; intro world
  apply and_congr_left; intro _
  exact closed_literal_extension base literals literal module names closed absent name member inputs _ returned world

end Merlin.GraphLoopsMemorySCF.After
