import Query
namespace Merlin.LoopScopedWitness
open Merlin.GraphState.Scalar Merlin.GraphState.Pipeline
open Merlin.GraphStatePipelineInstance
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false
-- Explicit illustrative external implementation, not a native-delivery premise.
-- Every successful retained call increments the observable world; IV9 faults
-- after three successful loop calls and records its own terminal-world effect.
def run (fail : Bool) (identity : Nat) (arguments : List Int) (world : Nat) :
    Except (Nat × Nat) (List Int × Nat) :=
  if fail && decide (identity = 7) && decide (arguments[1]?.getD 0 = 9) then .error (17,world+5)
  else .ok ((signature registry identity).outputs.map (fun _ => 17),world+1)
def provider (fail : Bool) : External Nat Nat registry := {
  run := run fail
  arity := by
    intro identity arguments world values final success
    simp only [run] at success
    split at success
    · cases success
    · cases success; simp
}
def chosen (p : Prim → List Int → Int) (fail : Bool) : Model Int Nat Nat registry := M p (provider fail)
theorem provider_step (p : Prim → List Int → Int) (fail : Bool) (x0 x1 iv value : Int) (world : Nat) :
    evalProgram (chosen p fail) F2.B2 (iv::value::F2.E2 p x0 x1) world =
      if fail && decide (iv = 9) then .error (.external 17,world+5) else .ok ([17],world+1) := by
  by_cases condition : fail = true ∧ iv = 9
  all_goals simp [F2.B2,F2.E2,chosen,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,
    IndexOp.eval,opaqueStep,evalProgram,reads,provider,run,signature,registry,Except.bind,condition]
def body (p : Prim → List Int → Int) (fail : Bool) (x0 x1 : Int) (iv : Int) (carry : List Int) (world : Nat) : Outcome Int Nat Nat :=
  evalProgram (chosen p fail) F2.B2 (iv::carry++F2.E2 p x0 x1) world
theorem staticIndices : indexIterations 3 27 2 = [3,5,7,9,11,13,15,17,19,21,23,25] :=
  indexIterations_count (lower := 3) (upper := 27) (step := 2) (count := 12) (by decide) (by decide)
theorem loop_success (p : Prim → List Int → Int) (x0 x1 value : Int) (world : Nat) :
    loopEval 1 (body p false x0 x1) (indexIterations 3 27 2) [value] world = .ok ([17],world+12) := by
  rw [staticIndices]
  simp [loopEval,body,provider_step,Nat.add_assoc,Except.bind]
theorem loop_failure (p : Prim → List Int → Int) (x0 x1 value : Int) (world : Nat) :
    loopEval 1 (body p true x0 x1) (indexIterations 3 27 2) [value] world = .error (.external 17,world+8) := by
  rw [staticIndices]
  simp [loopEval,body,provider_step,Nat.add_assoc,Except.bind]
theorem observer (p : Prim → List Int → Int) (fail : Bool) (x0 x1 value : Int) (world : Nat) :
    evalProgram (chosen p fail) F2.N2 (value::F2.E2 p x0 x1) world = .ok ([value],world+1) := by
  simp [F2.N2,F2.E2,chosen,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,
    IndexOp.eval,opaqueStep,evalProgram,reads,provider,run,signature,registry,Except.bind]
theorem source_success (p : Prim → List Int → Int) (x0 x1 : Int) (world : Nat) :
    evalProgram (chosen p false) F2.S0 [x0,x1] world = .ok ([17],world+13) := by
  rw [F2.firstTwo Int Nat Nat (chosen p false) [x0,x1] world rfl]
  change evalProgram (M p (provider false)) F2.S2 [x0,x1] world = _
  rw [F2.form2]
  change (loopEval 1 (body p false x0 x1) (indexIterations 3 27 2) [x0] world).bind _ = _
  rw [loop_success]
  simpa [chosen,Except.bind,Nat.add_assoc] using observer p false x0 x1 17 (world+12)
theorem source_failure (p : Prim → List Int → Int) (x0 x1 : Int) (world : Nat) :
    evalProgram (chosen p true) F2.S0 [x0,x1] world = .error (.external 17,world+8) := by
  rw [F2.firstTwo Int Nat Nat (chosen p true) [x0,x1] world rfl]
  change evalProgram (M p (provider true)) F2.S2 [x0,x1] world = _
  rw [F2.form2]
  change (loopEval 1 (body p true x0 x1) (indexIterations 3 27 2) [x0] world).bind _ = _
  rw [loop_failure]
  rfl
-- Total wrapping interpretation; this witness is not an arithmetic-fidelity theorem.
def wrap (width : Nat) (value : Int) : Int := value % (2^width : Int)
def primitive (op : Prim) (arguments : List Int) : Int :=
  let first := arguments[0]?.getD 0
  let second := arguments[1]?.getD 0
  match op with
  | .constant width value => wrap width value
  | .add width => wrap width (first+second)
  | .sub width => wrap width (first-second)
  | .mul width => wrap width (first*second)
  | .zext _ target | .trunc _ target => wrap target first
  | .sext source target =>
    let unsigned := wrap source first
    wrap target (if unsigned < (2^(source-1) : Int) then unsigned else unsigned-(2^source : Int))
def arguments : List Int := [3,5]
theorem input_arity : arguments.length = F2.inputs.length := by decide
theorem input_nonzero : arguments.all (fun value => decide (value ≠ 0)) = true := by decide
theorem input_word_domain : ∀ value ∈ arguments, 0 ≤ value ∧ value < (2^32 : Int) := by
  intro value member
  simp only [arguments,List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with rfl | rfl <;> decide
theorem actual_source_success : evalProgram (chosen primitive false) F2.S0 arguments 7 = .ok ([17],20) :=
  source_success primitive 3 5 7
theorem actual_final_success : evalProgram (chosen primitive false) F2.S4 arguments 7 = .ok ([17],20) := by
  change evalProgram (M primitive (provider false)) F2.S4 arguments 7 = _
  rw [← F2.actual primitive (provider false) arguments 7 input_arity]
  exact actual_source_success
theorem actual_source_failure : evalProgram (chosen primitive true) F2.S0 arguments 7 = .error (.external 17,15) :=
  source_failure primitive 3 5 7
theorem actual_final_failure : evalProgram (chosen primitive true) F2.S4 arguments 7 = .error (.external 17,15) := by
  change evalProgram (M primitive (provider true)) F2.S4 arguments 7 = _
  rw [← F2.actual primitive (provider true) arguments 7 input_arity]
  exact actual_source_failure
theorem source_initialized : Initialized (chosen primitive false) F2.S0 arguments 7 :=
  typed_operands_initialized (checkTypes_sound _ _ _ F2.typed0) (chosen primitive false) arguments 7 input_arity
theorem final_initialized : Initialized (chosen primitive false) F2.S4 arguments 7 :=
  typed_operands_initialized (checkTypes_sound _ _ _ F2.typed4) (chosen primitive false) arguments 7 input_arity
end Merlin.LoopScopedWitness
