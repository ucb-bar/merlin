import Merlin.GraphStateIntegerOperations
import Merlin.GraphStatePipelineInline
import Merlin.GraphStatePipelineIndex
import Merlin.GraphStatePipelineLiterals
import Merlin.GraphStatePipelineProgram
import Merlin.GraphStateScalar
set_option maxRecDepth 4096
set_option maxHeartbeats 2000000
namespace Merlin.GraphStatePipelineInlineInstance
open Merlin.GraphState.Scalar Merlin.GraphState.Pipeline
def registry : List Signature := [{inputs := [],outputs := [.index]},{inputs := [],outputs := [.index]},{inputs := [],outputs := [.index]},{inputs := [.index,(.bits 8)],outputs := [(.bits 8)]},{inputs := [(.bits 8),(.bits 8)],outputs := [(.bits 8)]}]
def controls : Controls := [(some (.constant (2))),(some (.constant (5))),(some (.constant (6))),none,none]
theorem controlAccepted : checkControl registry controls = true := by decide
def profileBits : Nat := 32
theorem literalsAccepted : checkLiterals profileBits controls = true := by decide
theorem literalsFit (identity : Nat) (value : Int) (found : Merlin.GraphState.Pipeline.controlAt controls identity = some (.constant value)) :
    signedRepresentative profileBits value := checkLiterals_sound literalsAccepted found
def M {World Error : Type} (primitive : Prim → List Int → Int)
    (external : External World Error registry) : Model Int World Error registry :=
  indexModel (checkControl_sound controlAccepted) primitive external
theorem naturalRange (lower upper step : Nat) (positive : 0 < step) :
  indexIterations (Int.ofNat lower) (Int.ofNat upper) (Int.ofNat step) =
    (scfRange lower upper step).map Int.ofNat := by
  change indexIterations (lower : Int) (upper : Int) (step : Int) = _
  have pos : (0 : Int) < (step : Int) := by omega
  simp only [indexIterations,Int.natCast_nonneg,pos,and_self,↓reduceIte,Int.toNat_natCast]
namespace F3
def inputs : List Ty := [(.bits 8),(.bits 8)]
def outputs : List Ty := [(.bits 8)]
def S0 : Program := (.opaque 1 [] [.index] (.opaque 2 [] [.index] (.opaque 0 [] [.index] (.loop 2 1 0 [3] [(.bits 8)] (.pure (.add 8) [1,6] (.opaque 3 [1,0] [(.bits 8)] (.ret [0]))) (.opaque 4 [0,5] [(.bits 8)] (.ret [0]))))))
def S1 : Program := (.opaque 1 [] [.index] (.opaque 2 [] [.index] (.opaque 0 [] [.index] (.opaque 1 [] [.index] (.pure (.add 8) [4,5] (.opaque 3 [1,0] [(.bits 8)] (.opaque 4 [0,7] [(.bits 8)] (.ret [0]))))))))
theorem typed0 : checkTypes registry inputs outputs S0 = true := by decide
theorem typed1 : checkTypes registry inputs outputs S1 = true := by decide
def E (_primitive : Prim → List Int → Int) (x0 x1 : Int) : List Int := [(2 : Int),(6 : Int),(5 : Int),x0,x1]
def B : Program := (.pure (.add 8) [1,6] (.opaque 3 [1,0] [(.bits 8)] (.ret [0])))
def N : Program := (.opaque 4 [0,5] [(.bits 8)] (.ret [0]))
def bounds : Merlin.GraphLoopsUnroll.Bounds := {bits := 32,lower := 5,upper := 6,step := 2,factor := 2,count := 1,groups := 0,remainder := 1}
theorem boundsChecked : bounds.Valid := by decide
theorem isInline : bounds.count ≤ bounds.factor := by decide
theorem inlineIndicesFit (ordinal : Nat) (inside : ordinal < bounds.count) :
    Merlin.GraphLoopsSegment.FitsSigned bounds.bits (bounds.lower+bounds.step*ordinal) ∧
    Merlin.GraphLoopsSegment.FitsSigned bounds.bits (bounds.lower+bounds.step*ordinal+bounds.step) :=
  Merlin.GraphLoopsUnroll.Bounds.inline_control boundsChecked inside
variable {World Error : Type} (primitive : Prim → List Int → Int)
    (external : External World Error registry) (x0 x1 : Int)
theorem source_form (world : World) : evalProgram (M primitive external) S0 [x0,x1] world =
  (loopEval 1 (fun i c w => evalProgram (M primitive external) B (i::c++E primitive x0 x1) w) (indexIterations 5 6 2) [x0] world).bind
    (fun r => evalProgram (M primitive external) N (r.1++E primitive x0 x1) r.2) := rfl
theorem actual_inputs (world : World) : evalProgram (M primitive external) S0 [x0,x1] world =
    evalProgram (M primitive external) S1 [x0,x1] world := by
  rw [source_form]
  rw [Inline.counted_continuation 1 (fun i c w => evalProgram (M primitive external) B (i::c++E primitive x0 x1) w) (fun c w => evalProgram (M primitive external) N (c++E primitive x0 x1) w) 5 6 2 1 (by decide) (by decide)]
  simp only [B,N,E,S1,progression,List.map_cons,List.map_nil,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
  all_goals
    cases step0 : external.run 3 [(5 : Int),(primitive (.add 8) [x0,x1])] world with
    | error failure =>
      rcases failure with ⟨error,terminal⟩
      simp only [step0,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
    | ok result =>
      rcases result with ⟨values,w0⟩
      have size : values.length = 1 := by
        have localArity := external.arity 3 [(5 : Int),(primitive (.add 8) [x0,x1])] world values w0 step0
        simpa only [registry,signature,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some,List.length_cons,List.length_nil] using localArity
      obtain ⟨v0_0,rfl⟩ := List.length_eq_one_iff.mp size
      simp only [step0,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
      all_goals
        cases step1 : external.run 4 [v0_0,x1] w0 with
        | error failure =>
          rcases failure with ⟨error,terminal⟩
          simp only [step1,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
        | ok result =>
          rcases result with ⟨values,w1⟩
          have size : values.length = 1 := by
            have localArity := external.arity 4 [v0_0,x1] w0 values w1 step1
            simpa only [registry,signature,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some,List.length_cons,List.length_nil] using localArity
          obtain ⟨v1_0,rfl⟩ := List.length_eq_one_iff.mp size
          simp only [step1,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
theorem actual (arguments : List Int) (world : World) (arity : arguments.length = inputs.length) :
    evalProgram (M primitive external) S0 arguments world = evalProgram (M primitive external) S1 arguments world := by
  have size : arguments.length = 2 := arity
  obtain ⟨a0,atail0,rfl⟩ := List.exists_cons_of_length_eq_add_one size
  have asize0 : atail0.length = 1 := by
    simp only [List.length_cons] at size
    omega
  obtain ⟨a1,rfl⟩ := List.length_eq_one_iff.mp asize0
  exact actual_inputs primitive external a0 a1 world
end F3
namespace F4
def inputs : List Ty := [(.bits 8)]
def outputs : List Ty := [(.bits 8)]
def S0 : Program := (.ret [0])
def S1 : Program := (.ret [0])
theorem typed0 : checkTypes registry inputs outputs S0 = true := by decide
theorem typed1 : checkTypes registry inputs outputs S1 = true := by decide
theorem actual {World Error : Type} (primitive : Prim → List Int → Int) (external : External World Error registry)
    (arguments : List Int) (world : World) (_arity : arguments.length = inputs.length) :
    evalProgram (M primitive external) S0 arguments world = evalProgram (M primitive external) S1 arguments world := rfl
end F4
def originalEntries : List Entry := [{inputs := F3.inputs,outputs := F3.outputs,body := F3.S0},{inputs := F4.inputs,outputs := F4.outputs,body := F4.S0}]
theorem originalEntriesChecked : checkEntries registry originalEntries = true := by decide
def finalEntries : List Entry := [{inputs := F3.inputs,outputs := F3.outputs,body := F3.S1},{inputs := F4.inputs,outputs := F4.outputs,body := F4.S1}]
theorem finalEntriesChecked : checkEntries registry finalEntries = true := by decide
theorem complete {World Error : Type} (primitive : Prim → List Int → Int) (external : External World Error registry) :
    PreservesEntries (M primitive external) originalEntries finalEntries :=
  entries_cons (M primitive external) rfl rfl (F3.actual primitive external) (entries_cons (M primitive external) rfl rfl (F4.actual primitive external) (entries_nil (M primitive external)))
end Merlin.GraphStatePipelineInlineInstance
theorem merlin_unroll_inline_transform {World Error : Type}
    (primitive : Merlin.GraphState.Scalar.Prim → List Int → Int)
    (external : Merlin.GraphState.Pipeline.External World Error Merlin.GraphStatePipelineInlineInstance.registry) :
    Merlin.GraphState.Pipeline.PreservesEntries (Merlin.GraphStatePipelineInlineInstance.M primitive external)
      Merlin.GraphStatePipelineInlineInstance.originalEntries Merlin.GraphStatePipelineInlineInstance.finalEntries :=
  Merlin.GraphStatePipelineInlineInstance.complete primitive external
namespace Merlin.GraphInlineWitness
open Merlin.GraphState.Scalar Merlin.GraphState.Pipeline
open Merlin.GraphStatePipelineInlineInstance
abbrev wordPrimitive := Merlin.GraphState.Integer.primitive
def chosen (failLane : Option Int) (failObserver : Bool) (failEveryBody : Bool) : External Nat Int registry where
  run := fun identity arguments world =>
    if (identity = 3 ∧ (failEveryBody = true ∨ failLane = some (arguments[0]?.getD 0))) ∨
        (identity = 4 ∧ failObserver = true) then .error (17,world+1)
    else .ok ((signature registry identity).outputs.map
      (fun _ => if identity = 3 then arguments[1]?.getD 0 else arguments.headD 0),world+1)
  arity := by
    intro identity arguments world values final success
    split at success
    · cases success
    · cases success; simp only [List.length_map]
theorem source_success : evalProgram (M wordPrimitive (chosen none false false)) Merlin.GraphStatePipelineInlineInstance.F3.S0 [3,2] 7 = .ok ([5],9) := rfl
theorem target_success : evalProgram (M wordPrimitive (chosen none false false)) Merlin.GraphStatePipelineInlineInstance.F3.S1 [3,2] 7 = .ok ([5],9) := by
  rw [← Merlin.GraphStatePipelineInlineInstance.F3.actual wordPrimitive (chosen none false false) [3,2] 7 rfl]
  exact source_success
theorem source_success_initialized :
    Initialized (M wordPrimitive (chosen none false false)) Merlin.GraphStatePipelineInlineInstance.F3.S0 [3,2] 7 :=
  (check_operands_initialized (show check registry Merlin.GraphStatePipelineInlineInstance.F3.inputs Merlin.GraphStatePipelineInlineInstance.F3.outputs Merlin.GraphStatePipelineInlineInstance.F3.S0 Merlin.GraphStatePipelineInlineInstance.F3.S0 = true by decide)
    (M wordPrimitive (chosen none false false)) [3,2] 7 rfl).1
theorem target_success_initialized :
    Initialized (M wordPrimitive (chosen none false false)) Merlin.GraphStatePipelineInlineInstance.F3.S1 [3,2] 7 :=
  (check_operands_initialized (show check registry Merlin.GraphStatePipelineInlineInstance.F3.inputs Merlin.GraphStatePipelineInlineInstance.F3.outputs Merlin.GraphStatePipelineInlineInstance.F3.S1 Merlin.GraphStatePipelineInlineInstance.F3.S1 = true by decide)
    (M wordPrimitive (chosen none false false)) [3,2] 7 rfl).1
theorem source_observer_failure : evalProgram (M wordPrimitive (chosen none true false)) Merlin.GraphStatePipelineInlineInstance.F3.S0 [3,2] 7 = .error (.external 17,9) := rfl
theorem target_observer_failure : evalProgram (M wordPrimitive (chosen none true false)) Merlin.GraphStatePipelineInlineInstance.F3.S1 [3,2] 7 = .error (.external 17,9) := by
  rw [← Merlin.GraphStatePipelineInlineInstance.F3.actual wordPrimitive (chosen none true false) [3,2] 7 rfl]
  exact source_observer_failure
theorem source_observer_failure_initialized :
    Initialized (M wordPrimitive (chosen none true false)) Merlin.GraphStatePipelineInlineInstance.F3.S0 [3,2] 7 :=
  (check_operands_initialized (show check registry Merlin.GraphStatePipelineInlineInstance.F3.inputs Merlin.GraphStatePipelineInlineInstance.F3.outputs Merlin.GraphStatePipelineInlineInstance.F3.S0 Merlin.GraphStatePipelineInlineInstance.F3.S0 = true by decide)
    (M wordPrimitive (chosen none true false)) [3,2] 7 rfl).1
theorem target_observer_failure_initialized :
    Initialized (M wordPrimitive (chosen none true false)) Merlin.GraphStatePipelineInlineInstance.F3.S1 [3,2] 7 :=
  (check_operands_initialized (show check registry Merlin.GraphStatePipelineInlineInstance.F3.inputs Merlin.GraphStatePipelineInlineInstance.F3.outputs Merlin.GraphStatePipelineInlineInstance.F3.S1 Merlin.GraphStatePipelineInlineInstance.F3.S1 = true by decide)
    (M wordPrimitive (chosen none true false)) [3,2] 7 rfl).1
theorem source_first_body_failure : evalProgram (M wordPrimitive (chosen (some 5) false false)) Merlin.GraphStatePipelineInlineInstance.F3.S0 [3,2] 7 = .error (.external 17,8) := rfl
theorem target_first_body_failure : evalProgram (M wordPrimitive (chosen (some 5) false false)) Merlin.GraphStatePipelineInlineInstance.F3.S1 [3,2] 7 = .error (.external 17,8) := by
  rw [← Merlin.GraphStatePipelineInlineInstance.F3.actual wordPrimitive (chosen (some 5) false false) [3,2] 7 rfl]
  exact source_first_body_failure
theorem source_first_body_failure_initialized :
    Initialized (M wordPrimitive (chosen (some 5) false false)) Merlin.GraphStatePipelineInlineInstance.F3.S0 [3,2] 7 :=
  (check_operands_initialized (show check registry Merlin.GraphStatePipelineInlineInstance.F3.inputs Merlin.GraphStatePipelineInlineInstance.F3.outputs Merlin.GraphStatePipelineInlineInstance.F3.S0 Merlin.GraphStatePipelineInlineInstance.F3.S0 = true by decide)
    (M wordPrimitive (chosen (some 5) false false)) [3,2] 7 rfl).1
theorem target_first_body_failure_initialized :
    Initialized (M wordPrimitive (chosen (some 5) false false)) Merlin.GraphStatePipelineInlineInstance.F3.S1 [3,2] 7 :=
  (check_operands_initialized (show check registry Merlin.GraphStatePipelineInlineInstance.F3.inputs Merlin.GraphStatePipelineInlineInstance.F3.outputs Merlin.GraphStatePipelineInlineInstance.F3.S1 Merlin.GraphStatePipelineInlineInstance.F3.S1 = true by decide)
    (M wordPrimitive (chosen (some 5) false false)) [3,2] 7 rfl).1
theorem source_last_body_failure : evalProgram (M wordPrimitive (chosen (some 5) false false)) Merlin.GraphStatePipelineInlineInstance.F3.S0 [3,2] 7 = .error (.external 17,8) := rfl
theorem target_last_body_failure : evalProgram (M wordPrimitive (chosen (some 5) false false)) Merlin.GraphStatePipelineInlineInstance.F3.S1 [3,2] 7 = .error (.external 17,8) := by
  rw [← Merlin.GraphStatePipelineInlineInstance.F3.actual wordPrimitive (chosen (some 5) false false) [3,2] 7 rfl]
  exact source_last_body_failure
theorem source_last_body_failure_initialized :
    Initialized (M wordPrimitive (chosen (some 5) false false)) Merlin.GraphStatePipelineInlineInstance.F3.S0 [3,2] 7 :=
  (check_operands_initialized (show check registry Merlin.GraphStatePipelineInlineInstance.F3.inputs Merlin.GraphStatePipelineInlineInstance.F3.outputs Merlin.GraphStatePipelineInlineInstance.F3.S0 Merlin.GraphStatePipelineInlineInstance.F3.S0 = true by decide)
    (M wordPrimitive (chosen (some 5) false false)) [3,2] 7 rfl).1
theorem target_last_body_failure_initialized :
    Initialized (M wordPrimitive (chosen (some 5) false false)) Merlin.GraphStatePipelineInlineInstance.F3.S1 [3,2] 7 :=
  (check_operands_initialized (show check registry Merlin.GraphStatePipelineInlineInstance.F3.inputs Merlin.GraphStatePipelineInlineInstance.F3.outputs Merlin.GraphStatePipelineInlineInstance.F3.S1 Merlin.GraphStatePipelineInlineInstance.F3.S1 = true by decide)
    (M wordPrimitive (chosen (some 5) false false)) [3,2] 7 rfl).1
end Merlin.GraphInlineWitness
