import Merlin.GraphStatePipelineInline
import Merlin.GraphStatePipelineIndex
import Merlin.GraphStatePipelineLiterals
import Merlin.GraphStatePipelineProgram
import Merlin.GraphStateScalar
set_option maxRecDepth 4096
set_option maxHeartbeats 2000000
namespace Merlin.GraphStatePipelineInlineInstance
open Merlin.GraphState.Scalar Merlin.GraphState.Pipeline
def registry : List Signature := [{inputs := [],outputs := [.index]},{inputs := [],outputs := [.index]},{inputs := [.index,(.bits 8)],outputs := [(.bits 8)]},{inputs := [(.bits 8)],outputs := [(.bits 8)]}]
def controls : Controls := [(some (.constant (2))),(some (.constant (3))),none,none]
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
def S0 : Program := (.opaque 1 [] [.index] (.opaque 1 [] [.index] (.opaque 0 [] [.index] (.loop 2 1 0 [3,4] [(.bits 8),(.bits 8)] (.pure (.add 8) [1,7] (.opaque 2 [1,0] [(.bits 8)] (.ret [0,4]))) (.opaque 3 [0] [(.bits 8)] (.ret [0]))))))
def S1 : Program := (.opaque 1 [] [.index] (.opaque 1 [] [.index] (.opaque 0 [] [.index] (.opaque 3 [3] [(.bits 8)] (.ret [0])))))
theorem typed0 : checkTypes registry inputs outputs S0 = true := by decide
theorem typed1 : checkTypes registry inputs outputs S1 = true := by decide
def E (_primitive : Prim → List Int → Int) (x0 x1 : Int) : List Int := [(2 : Int),(3 : Int),(3 : Int),x0,x1]
def B : Program := (.pure (.add 8) [1,7] (.opaque 2 [1,0] [(.bits 8)] (.ret [0,4])))
def N : Program := (.opaque 3 [0] [(.bits 8)] (.ret [0]))
def bounds : Merlin.GraphLoopsUnroll.Bounds := {bits := 32,lower := 3,upper := 3,step := 2,factor := 2,count := 0,groups := 0,remainder := 0}
theorem boundsChecked : bounds.Valid := by decide
theorem isInline : bounds.count ≤ bounds.factor := by decide
theorem inlineIndicesFit (ordinal : Nat) (inside : ordinal < bounds.count) :
    Merlin.GraphLoopsSegment.FitsSigned bounds.bits (bounds.lower+bounds.step*ordinal) ∧
    Merlin.GraphLoopsSegment.FitsSigned bounds.bits (bounds.lower+bounds.step*ordinal+bounds.step) :=
  Merlin.GraphLoopsUnroll.Bounds.inline_control boundsChecked inside
variable {World Error : Type} (primitive : Prim → List Int → Int)
    (external : External World Error registry) (x0 x1 : Int)
theorem source_form (world : World) : evalProgram (M primitive external) S0 [x0,x1] world =
  (loopEval 2 (fun i c w => evalProgram (M primitive external) B (i::c++E primitive x0 x1) w) (indexIterations 3 3 2) [x0,x1] world).bind
    (fun r => evalProgram (M primitive external) N (r.1++E primitive x0 x1) r.2) := rfl
theorem empty_body_never_runs (world : World) :
    loopEval 2 (fun i c w => evalProgram (M primitive external) B (i::c++E primitive x0 x1) w) (indexIterations 3 3 2) [x0,x1] world = .ok ([x0,x1],world) :=
  Inline.empty_no_body 2 _ 3 3 2 (by decide) (by decide) _ world rfl
theorem actual_inputs (world : World) : evalProgram (M primitive external) S0 [x0,x1] world =
    evalProgram (M primitive external) S1 [x0,x1] world := by
  rw [source_form]
  rw [empty_body_never_runs]
  simp only [B,N,E,S1,progression,List.map_cons,List.map_nil,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
  cases step0 : external.run 3 [x0] world with
  | error failure =>
    rcases failure with ⟨error,terminal⟩
    simp only [step0,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
  | ok result =>
    rcases result with ⟨values,w0⟩
    have size : values.length = 1 := by
      have localArity := external.arity 3 [x0] world values w0 step0
      simpa only [registry,signature,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some,List.length_cons,List.length_nil] using localArity
    obtain ⟨v0_0,rfl⟩ := List.length_eq_one_iff.mp size
    simp only [step0,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
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
