/- Preserved countermodels for omitting entry/value-result/pure-primitive
contracts. These are limitations of the theorem domain, not a pass refutation
on initialized canonical words with the disclosed exact local interpretation. -/
import Merlin.GraphStateIntegerOperations

namespace Merlin.GraphState.Integer.Countermodels
set_option autoImplicit false
open Scalar

def base : Model Int Nat Int [] where
  default := 0
  primitive := Integer.primitive
  index := id
  iterations := fun _ _ _ => []
  effect := fun _ _ world => .ok ([],world)
  arity := by intro identity arguments world values final success; cases success; simp [signature]

def addZero : Program := .pure (.constant 8 0) [] (.pure (.add 8) [1,0] (.ret [0]))
def unchanged : Program := .ret [0]

theorem noncanonical_entry_refutes_neutral_deletion :
    evalProgram base addZero [256] 7 ≠ evalProgram base unchanged [256] 7 := by
  intro same
  have outputs : (([0],7) : List Int × Nat) = ([256],7) := Except.ok.inj same
  have impossible : (0 : Int) = 256 := congrArg (fun result => result.1.headD 0) outputs
  omega

theorem entry_really_noncanonical : ¬ Canonical (.bits 8) 256 := by
  change (0 : Int) ≠ 256
  decide

def wrongPrimitive : Model Int Nat Int [] := {base with primitive := fun _ _ => 7}

theorem missing_primitive_contract_refutes_deletion :
    evalProgram wrongPrimitive addZero [5] 7 ≠ evalProgram wrongPrimitive unchanged [5] 7 := by
  intro same
  have outputs : (([7],7) : List Int × Nat) = ([5],7) := Except.ok.inj same
  have impossible : (7 : Int) = 5 := congrArg (fun result => result.1.headD 0) outputs
  omega

def registry : List Signature := [{inputs := [],outputs := [.bits 8]}]
def noncanonicalProvider : Model Int Nat Int registry where
  default := 0
  primitive := Integer.primitive
  index := id
  iterations := fun _ _ _ => []
  effect := fun identity _ world => .ok ((signature registry identity).outputs.map (fun _ => 256),world+1)
  arity := by intro identity arguments world values final success; cases success; simp only [List.length_map]

def withProvider : Program := .opaque 0 [] [.bits 8] addZero
def liftedProvider : Program := .opaque 0 [] [.bits 8] unchanged

theorem provider_programs_statically_typed :
    checkTypes registry [] [.bits 8] withProvider = true ∧
    checkTypes registry [] [.bits 8] liftedProvider = true := by decide

theorem missing_local_result_contract_refutes_deletion :
    evalProgram noncanonicalProvider withProvider [] 7 ≠ evalProgram noncanonicalProvider liftedProvider [] 7 := by
  intro same
  have outputs : (([0],8) : List Int × Nat) = ([256],8) := Except.ok.inj same
  have impossible : (0 : Int) = 256 := congrArg (fun result => result.1.headD 0) outputs
  omega

/-- A physical arithmetic instruction with a hidden world update cannot be
represented by the total STATE-FREE Scalar primitive field. Removing it would
change the observed world even when its word result is correct. -/
theorem hidden_effect_countermodel :
    (Except.ok ([5],8) : Outcome Int Nat Int) ≠ .ok ([5],7) := by
  intro same
  have impossible : (8 : Nat) = 7 := congrArg Prod.snd (Except.ok.inj same)
  omega

#print axioms noncanonical_entry_refutes_neutral_deletion
#print axioms missing_local_result_contract_refutes_deletion
#print axioms missing_primitive_contract_refutes_deletion
#print axioms hidden_effect_countermodel
end Merlin.GraphState.Integer.Countermodels
