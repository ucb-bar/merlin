import Merlin.GraphStateIntegerProgram
import Merlin.GraphStateIntegerTrace
import Merlin.GraphStatePipelineIndex
import Merlin.GraphStatePipelineLiterals
import Merlin.GraphStatePipelineScope
import Merlin.GraphStatePipelineLoops
import Merlin.GraphStatePipelineProgram
import Merlin.GraphStateScalar
set_option maxRecDepth 4096
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false
namespace Merlin.GraphStateIntegerInstance.Suffix
open Merlin.GraphState.Scalar Merlin.GraphState.Pipeline
def registry : List Signature := [{inputs := [.index,.index],outputs := [.index]},{inputs := [],outputs := [.index]},{inputs := [],outputs := [.index]},{inputs := [],outputs := [.index]},{inputs := [],outputs := [.index]},{inputs := [],outputs := [.index]},{inputs := [],outputs := [.index]},{inputs := [],outputs := [.index]},{inputs := [.index,.index],outputs := [.index]},{inputs := [(.bits 64),.index],outputs := [(.bits 64)]},{inputs := [(.bits 64),.index],outputs := []},{inputs := [(.bits 64)],outputs := []}]
def controls : Controls := [(some .add),(some (.constant (0))),(some (.constant (12))),(some (.constant (1))),(some (.constant (2))),(some (.constant (3))),(some (.constant (4))),(some (.constant (8))),(some .minui),none,none,none]
theorem controlAccepted : checkControl registry controls = true := by decide
def profileBits : Nat := 64
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
def inputs : List Ty := [(.bits 64),(.bits 64)]
def outputs : List Ty := [(.bits 64)]
def S0 : Program := (.opaque 1 [] [.index] (.opaque 2 [] [.index] (.opaque 3 [] [.index] (.pure (.constant 64 (0)) [] (.pure (.constant 64 (1)) [] (.pure (.constant 64 (-3)) [] (.pure (.constant 64 (3)) [] (.pure (.constant 64 (0)) [] (.pure (.sub 64) [8,9] (.loop 8 7 6 [9] [(.bits 64)] (.pure (.add 64) [12,12] (.pure (.add 64) [13,13] (.pure (.add 64) [3,0] (.pure (.add 64) [0,2] (.opaque 10 [0,4] [] (.opaque 9 [0,4] [(.bits 64)] (.ret [0]))))))) (.opaque 11 [0] [] (.ret [0]))))))))))))
theorem typed0 : checkTypes registry inputs outputs S0 = true := by decide
def S1 : Program := (.opaque 1 [] [.index] (.opaque 2 [] [.index] (.opaque 3 [] [.index] (.loop 2 1 0 [3] [(.bits 64)] (.pure (.add 64) [6,6] (.pure (.add 64) [2,0] (.pure (.add 64) [0,1] (.opaque 10 [0,3] [] (.opaque 9 [0,3] [(.bits 64)] (.ret [0])))))) (.opaque 11 [0] [] (.ret [0]))))))
theorem typed1 : checkTypes registry inputs outputs S1 = true := by decide
def S2 : Program := (.opaque 1 [] [.index] (.opaque 2 [] [.index] (.opaque 3 [] [.index] (.pure (.add 64) [4,4] (.loop 3 2 1 [4] [(.bits 64)] (.pure (.add 64) [1,2] (.pure (.add 64) [0,3] (.opaque 10 [0,2] [] (.opaque 9 [0,2] [(.bits 64)] (.ret [0]))))) (.opaque 11 [0] [] (.ret [0])))))))
theorem typed2 : checkTypes registry inputs outputs S2 = true := by decide
def S3 : Program := (.opaque 1 [] [.index] (.opaque 2 [] [.index] (.opaque 3 [] [.index] (.pure (.add 64) [4,4] (.opaque 6 [] [.index] (.opaque 2 [] [.index] (.loop 5 0 1 [6] [(.bits 64)] (.pure (.add 64) [1,4] (.pure (.add 64) [0,5] (.opaque 10 [0,2] [] (.opaque 9 [0,2] [(.bits 64)] (.opaque 3 [] [.index] (.opaque 0 [4,0] [.index] (.pure (.add 64) [2,9] (.pure (.add 64) [0,10] (.opaque 10 [0,2] [] (.opaque 9 [0,2] [(.bits 64)] (.opaque 4 [] [.index] (.opaque 0 [9,0] [.index] (.pure (.add 64) [2,14] (.pure (.add 64) [0,15] (.opaque 10 [0,2] [] (.opaque 9 [0,2] [(.bits 64)] (.opaque 5 [] [.index] (.opaque 0 [14,0] [.index] (.pure (.add 64) [2,19] (.pure (.add 64) [0,20] (.opaque 10 [0,2] [] (.opaque 9 [0,2] [(.bits 64)] (.ret [0]))))))))))))))))))))))) (.opaque 11 [0] [] (.ret [0])))))))))
theorem typed3 : checkTypes registry inputs outputs S3 = true := by decide
def S4 : Program := (.opaque 1 [] [.index] (.opaque 2 [] [.index] (.opaque 3 [] [.index] (.pure (.add 64) [4,4] (.opaque 6 [] [.index] (.opaque 2 [] [.index] (.opaque 7 [] [.index] (.loop 6 1 0 [7] [(.bits 64)] (.opaque 0 [0,2] [.index] (.opaque 8 [0,4] [.index] (.loop 2 0 6 [3] [(.bits 64)] (.pure (.add 64) [1,9] (.pure (.add 64) [0,10] (.opaque 10 [0,2] [] (.opaque 9 [0,2] [(.bits 64)] (.opaque 3 [] [.index] (.opaque 0 [4,0] [.index] (.pure (.add 64) [2,14] (.pure (.add 64) [0,15] (.opaque 10 [0,2] [] (.opaque 9 [0,2] [(.bits 64)] (.opaque 4 [] [.index] (.opaque 0 [9,0] [.index] (.pure (.add 64) [2,19] (.pure (.add 64) [0,20] (.opaque 10 [0,2] [] (.opaque 9 [0,2] [(.bits 64)] (.opaque 5 [] [.index] (.opaque 0 [14,0] [.index] (.pure (.add 64) [2,24] (.pure (.add 64) [0,25] (.opaque 10 [0,2] [] (.opaque 9 [0,2] [(.bits 64)] (.ret [0]))))))))))))))))))))))) (.ret [0])))) (.opaque 11 [0] [] (.ret [0]))))))))))
theorem typed4 : checkTypes registry inputs outputs S4 = true := by decide
theorem firstAccepted : check registry inputs outputs S0 S1 = true := by decide
theorem secondAccepted : check registry inputs outputs S1 S2 = true := by decide
theorem firstTwo : Preserves registry inputs.length S0 S2 := check_composes firstAccepted secondAccepted
def E2 (primitive : Prim → List Int → Int) (x0 x1 : Int) : List Int := [(primitive (.add 64) [x1,x1]),(1 : Int),(12 : Int),(0 : Int),x0,x1]
def B2 : Program := (.pure (.add 64) [1,2] (.pure (.add 64) [0,3] (.opaque 10 [0,2] [] (.opaque 9 [0,2] [(.bits 64)] (.ret [0])))))
def N2 : Program := (.opaque 11 [0] [] (.ret [0]))
def E3 (primitive : Prim → List Int → Int) (x0 x1 : Int) : List Int := [(12 : Int),(4 : Int),(primitive (.add 64) [x1,x1]),(1 : Int),(12 : Int),(0 : Int),x0,x1]
def B3 : Program := (.pure (.add 64) [1,4] (.pure (.add 64) [0,5] (.opaque 10 [0,2] [] (.opaque 9 [0,2] [(.bits 64)] (.opaque 3 [] [.index] (.opaque 0 [4,0] [.index] (.pure (.add 64) [2,9] (.pure (.add 64) [0,10] (.opaque 10 [0,2] [] (.opaque 9 [0,2] [(.bits 64)] (.opaque 4 [] [.index] (.opaque 0 [9,0] [.index] (.pure (.add 64) [2,14] (.pure (.add 64) [0,15] (.opaque 10 [0,2] [] (.opaque 9 [0,2] [(.bits 64)] (.opaque 5 [] [.index] (.opaque 0 [14,0] [.index] (.pure (.add 64) [2,19] (.pure (.add 64) [0,20] (.opaque 10 [0,2] [] (.opaque 9 [0,2] [(.bits 64)] (.ret [0])))))))))))))))))))))))
def N3 : Program := (.opaque 11 [0] [] (.ret [0]))
def E4 (primitive : Prim → List Int → Int) (x0 x1 : Int) : List Int := [(8 : Int),(12 : Int),(4 : Int),(primitive (.add 64) [x1,x1]),(1 : Int),(12 : Int),(0 : Int),x0,x1]
def B4 : Program := (.opaque 0 [0,2] [.index] (.opaque 8 [0,4] [.index] (.loop 2 0 6 [3] [(.bits 64)] (.pure (.add 64) [1,9] (.pure (.add 64) [0,10] (.opaque 10 [0,2] [] (.opaque 9 [0,2] [(.bits 64)] (.opaque 3 [] [.index] (.opaque 0 [4,0] [.index] (.pure (.add 64) [2,14] (.pure (.add 64) [0,15] (.opaque 10 [0,2] [] (.opaque 9 [0,2] [(.bits 64)] (.opaque 4 [] [.index] (.opaque 0 [9,0] [.index] (.pure (.add 64) [2,19] (.pure (.add 64) [0,20] (.opaque 10 [0,2] [] (.opaque 9 [0,2] [(.bits 64)] (.opaque 5 [] [.index] (.opaque 0 [14,0] [.index] (.pure (.add 64) [2,24] (.pure (.add 64) [0,25] (.opaque 10 [0,2] [] (.opaque 9 [0,2] [(.bits 64)] (.ret [0]))))))))))))))))))))))) (.ret [0]))))
def N4 : Program := (.opaque 11 [0] [] (.ret [0]))
variable {World Error : Type} (primitive : Prim → List Int → Int)
    (external : External World Error registry) (x0 x1 : Int)
theorem form2 (world : World) : evalProgram (M primitive external) S2 [x0,x1] world =
  (loopEval 1 (fun i carry w => evalProgram (M primitive external) B2 (i::carry++E2 primitive x0 x1) w)
    (indexIterations (0 : Int) (12 : Int) (1 : Int)) [x0] world).bind
      (fun result => evalProgram (M primitive external) N2 (result.1++E2 primitive x0 x1) result.2) := by rfl
theorem form3 (world : World) : evalProgram (M primitive external) S3 [x0,x1] world =
  (loopEval 1 (fun i carry w => evalProgram (M primitive external) B3 (i::carry++E3 primitive x0 x1) w)
    (indexIterations (0 : Int) (12 : Int) (4 : Int)) [x0] world).bind
      (fun result => evalProgram (M primitive external) N3 (result.1++E3 primitive x0 x1) result.2) := by rfl
theorem form4 (world : World) : evalProgram (M primitive external) S4 [x0,x1] world =
  (loopEval 1 (fun i carry w => evalProgram (M primitive external) B4 (i::carry++E4 primitive x0 x1) w)
    (indexIterations (0 : Int) (12 : Int) (8 : Int)) [x0] world).bind
      (fun result => evalProgram (M primitive external) N4 (result.1++E4 primitive x0 x1) result.2) := by rfl
theorem body_group (base : Int) (carry : List Int) (world : World) (arity : carry.length = 1) :
  evalProgram (M primitive external) B3 (base::carry++E3 primitive x0 x1) world =
  loopEval 1 (fun i c w => evalProgram (M primitive external) B2 (i::c++E2 primitive x0 x1) w)
    [base,(base+1),(base+2),(base+3)] carry world := by
  obtain ⟨c0,rfl⟩ := List.length_eq_one_iff.mp arity
  simp only [B2,B3,E2,E3,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
  cases step0 : external.run 10 [(primitive (.add 64) [(primitive (.add 64) [c0,(primitive (.add 64) [x1,x1])]),(primitive (.add 64) [x1,x1])]),base] world with
  | error failure =>
    rcases failure with ⟨error,terminal⟩
    simp only [step0,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
  | ok result =>
    rcases result with ⟨values,w0⟩
    have size : values.length = 0 := by
      have localArity := external.arity 10 [(primitive (.add 64) [(primitive (.add 64) [c0,(primitive (.add 64) [x1,x1])]),(primitive (.add 64) [x1,x1])]),base] world values w0 step0
      simpa only [registry,signature,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some,List.length_cons,List.length_nil] using localArity
    have v0_empty : values = [] := List.length_eq_zero_iff.mp size
    subst values
    simp only [step0,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
    cases step1 : external.run 9 [(primitive (.add 64) [(primitive (.add 64) [c0,(primitive (.add 64) [x1,x1])]),(primitive (.add 64) [x1,x1])]),base] w0 with
    | error failure =>
      rcases failure with ⟨error,terminal⟩
      simp only [step1,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
    | ok result =>
      rcases result with ⟨values,w1⟩
      have size : values.length = 1 := by
        have localArity := external.arity 9 [(primitive (.add 64) [(primitive (.add 64) [c0,(primitive (.add 64) [x1,x1])]),(primitive (.add 64) [x1,x1])]),base] w0 values w1 step1
        simpa only [registry,signature,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some,List.length_cons,List.length_nil] using localArity
      obtain ⟨v1_0,rfl⟩ := List.length_eq_one_iff.mp size
      simp only [step1,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
      cases step2 : external.run 10 [(primitive (.add 64) [(primitive (.add 64) [v1_0,(primitive (.add 64) [x1,x1])]),(primitive (.add 64) [x1,x1])]),(base+1)] w1 with
      | error failure =>
        rcases failure with ⟨error,terminal⟩
        simp only [step2,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
      | ok result =>
        rcases result with ⟨values,w2⟩
        have size : values.length = 0 := by
          have localArity := external.arity 10 [(primitive (.add 64) [(primitive (.add 64) [v1_0,(primitive (.add 64) [x1,x1])]),(primitive (.add 64) [x1,x1])]),(base+1)] w1 values w2 step2
          simpa only [registry,signature,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some,List.length_cons,List.length_nil] using localArity
        have v2_empty : values = [] := List.length_eq_zero_iff.mp size
        subst values
        simp only [step2,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
        cases step3 : external.run 9 [(primitive (.add 64) [(primitive (.add 64) [v1_0,(primitive (.add 64) [x1,x1])]),(primitive (.add 64) [x1,x1])]),(base+1)] w2 with
        | error failure =>
          rcases failure with ⟨error,terminal⟩
          simp only [step3,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
        | ok result =>
          rcases result with ⟨values,w3⟩
          have size : values.length = 1 := by
            have localArity := external.arity 9 [(primitive (.add 64) [(primitive (.add 64) [v1_0,(primitive (.add 64) [x1,x1])]),(primitive (.add 64) [x1,x1])]),(base+1)] w2 values w3 step3
            simpa only [registry,signature,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some,List.length_cons,List.length_nil] using localArity
          obtain ⟨v3_0,rfl⟩ := List.length_eq_one_iff.mp size
          simp only [step3,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
          cases step4 : external.run 10 [(primitive (.add 64) [(primitive (.add 64) [v3_0,(primitive (.add 64) [x1,x1])]),(primitive (.add 64) [x1,x1])]),(base+2)] w3 with
          | error failure =>
            rcases failure with ⟨error,terminal⟩
            simp only [step4,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
          | ok result =>
            rcases result with ⟨values,w4⟩
            have size : values.length = 0 := by
              have localArity := external.arity 10 [(primitive (.add 64) [(primitive (.add 64) [v3_0,(primitive (.add 64) [x1,x1])]),(primitive (.add 64) [x1,x1])]),(base+2)] w3 values w4 step4
              simpa only [registry,signature,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some,List.length_cons,List.length_nil] using localArity
            have v4_empty : values = [] := List.length_eq_zero_iff.mp size
            subst values
            simp only [step4,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
            cases step5 : external.run 9 [(primitive (.add 64) [(primitive (.add 64) [v3_0,(primitive (.add 64) [x1,x1])]),(primitive (.add 64) [x1,x1])]),(base+2)] w4 with
            | error failure =>
              rcases failure with ⟨error,terminal⟩
              simp only [step5,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
            | ok result =>
              rcases result with ⟨values,w5⟩
              have size : values.length = 1 := by
                have localArity := external.arity 9 [(primitive (.add 64) [(primitive (.add 64) [v3_0,(primitive (.add 64) [x1,x1])]),(primitive (.add 64) [x1,x1])]),(base+2)] w4 values w5 step5
                simpa only [registry,signature,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some,List.length_cons,List.length_nil] using localArity
              obtain ⟨v5_0,rfl⟩ := List.length_eq_one_iff.mp size
              simp only [step5,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
              cases step6 : external.run 10 [(primitive (.add 64) [(primitive (.add 64) [v5_0,(primitive (.add 64) [x1,x1])]),(primitive (.add 64) [x1,x1])]),(base+3)] w5 with
              | error failure =>
                rcases failure with ⟨error,terminal⟩
                simp only [step6,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
              | ok result =>
                rcases result with ⟨values,w6⟩
                have size : values.length = 0 := by
                  have localArity := external.arity 10 [(primitive (.add 64) [(primitive (.add 64) [v5_0,(primitive (.add 64) [x1,x1])]),(primitive (.add 64) [x1,x1])]),(base+3)] w5 values w6 step6
                  simpa only [registry,signature,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some,List.length_cons,List.length_nil] using localArity
                have v6_empty : values = [] := List.length_eq_zero_iff.mp size
                subst values
                simp only [step6,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
                cases step7 : external.run 9 [(primitive (.add 64) [(primitive (.add 64) [v5_0,(primitive (.add 64) [x1,x1])]),(primitive (.add 64) [x1,x1])]),(base+3)] w6 with
                | error failure =>
                  rcases failure with ⟨error,terminal⟩
                  simp only [step7,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
                | ok result =>
                  rcases result with ⟨values,w7⟩
                  have size : values.length = 1 := by
                    have localArity := external.arity 9 [(primitive (.add 64) [(primitive (.add 64) [v5_0,(primitive (.add 64) [x1,x1])]),(primitive (.add 64) [x1,x1])]),(base+3)] w6 values w7 step7
                    simpa only [registry,signature,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some,List.length_cons,List.length_nil] using localArity
                  obtain ⟨v7_0,rfl⟩ := List.length_eq_one_iff.mp size
                  simp only [step7,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
def I4 : Program := (.pure (.add 64) [1,9] (.pure (.add 64) [0,10] (.opaque 10 [0,2] [] (.opaque 9 [0,2] [(.bits 64)] (.opaque 3 [] [.index] (.opaque 0 [4,0] [.index] (.pure (.add 64) [2,14] (.pure (.add 64) [0,15] (.opaque 10 [0,2] [] (.opaque 9 [0,2] [(.bits 64)] (.opaque 4 [] [.index] (.opaque 0 [9,0] [.index] (.pure (.add 64) [2,19] (.pure (.add 64) [0,20] (.opaque 10 [0,2] [] (.opaque 9 [0,2] [(.bits 64)] (.opaque 5 [] [.index] (.opaque 0 [14,0] [.index] (.pure (.add 64) [2,24] (.pure (.add 64) [0,25] (.opaque 10 [0,2] [] (.opaque 9 [0,2] [(.bits 64)] (.ret [0])))))))))))))))))))))))
def LS : List Expr := [(.var 0),(.var 1),(.var 7),(.var 8),(.var 9),(.var 10),(.var 7),(.var 12),(.var 13),(.var 14)]
def RS : List Expr := [(.var 0),(.var 1),(.var 2),(.var 3),(.var 4),(.var 5),(.var 6),(.var 7),(.var 8),(.var 9),(.var 10),(.var 11),(.var 12),(.var 13),(.var 14)]
theorem innerAccepted : Scope.check registry LS RS B3 I4 = true := by decide
theorem inner_same (iv outer outerCarry0 : Int) (carry : List Int) (world : World) (arity : carry.length = 1) :
  evalProgram (M primitive external) I4 (iv::carry++[(min (outer + (8 : Int)) (12 : Int)),(outer + (8 : Int)),outer,outerCarry0,(8 : Int),(12 : Int),(4 : Int),(primitive (.add 64) [x1,x1]),(1 : Int),(12 : Int),(0 : Int),x0,x1]) world =
  evalProgram (M primitive external) B3 (iv::carry++E3 primitive x0 x1) world := by
  obtain ⟨c0,rfl⟩ := List.length_eq_one_iff.mp arity
  have checked := Scope.check_sound innerAccepted (M primitive external) [iv,c0,(min (outer + (8 : Int)) (12 : Int)),(outer + (8 : Int)),outer,outerCarry0,(8 : Int),(12 : Int),(4 : Int),(primitive (.add 64) [x1,x1]),(1 : Int),(12 : Int),(0 : Int),x0,x1] world
  simpa only [LS,RS,E3,evalExpr,List.cons_append,List.nil_append,List.map_cons,List.map_nil,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some] using checked.symm
def CL23 : List Expr := [(.var 0),(.var 3),(.var 4),(.var 1),(.var 6),(.var 7),(.var 8)]
def CR23 : List Expr := [(.var 0),(.var 1),(.var 2),(.var 3),(.var 4),(.var 5),(.var 6),(.var 7),(.var 8)]
theorem suffixAccepted23 : Scope.check registry CL23 CR23 N2 N3 = true := by decide
theorem suffix_same23 (carry : List Int) (world : World) (arity : carry.length = 1) :
  evalProgram (M primitive external) N2 (carry++E2 primitive x0 x1) world =
  evalProgram (M primitive external) N3 (carry++E3 primitive x0 x1) world := by
  obtain ⟨c0,rfl⟩ := List.length_eq_one_iff.mp arity
  have checked := Scope.check_sound suffixAccepted23 (M primitive external) [c0,(12 : Int),(4 : Int),(primitive (.add 64) [x1,x1]),(1 : Int),(12 : Int),(0 : Int),x0,x1] world
  simpa only [CL23,CR23,E2,E3,evalExpr,List.cons_append,List.nil_append,List.map_cons,List.map_nil,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some] using checked
def CL34 : List Expr := [(.var 0),(.var 2),(.var 3),(.var 4),(.var 5),(.var 2),(.var 7),(.var 8),(.var 9)]
def CR34 : List Expr := [(.var 0),(.var 1),(.var 2),(.var 3),(.var 4),(.var 5),(.var 6),(.var 7),(.var 8),(.var 9)]
theorem suffixAccepted34 : Scope.check registry CL34 CR34 N3 N4 = true := by decide
theorem suffix_same34 (carry : List Int) (world : World) (arity : carry.length = 1) :
  evalProgram (M primitive external) N3 (carry++E3 primitive x0 x1) world =
  evalProgram (M primitive external) N4 (carry++E4 primitive x0 x1) world := by
  obtain ⟨c0,rfl⟩ := List.length_eq_one_iff.mp arity
  have checked := Scope.check_sound suffixAccepted34 (M primitive external) [c0,(8 : Int),(12 : Int),(4 : Int),(primitive (.add 64) [x1,x1]),(1 : Int),(12 : Int),(0 : Int),x0,x1] world
  simpa only [CL34,CR34,E3,E4,evalExpr,List.cons_append,List.nil_append,List.map_cons,List.map_nil,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some] using checked
theorem outer_form (base c0 : Int) (world : World) :
  evalProgram (M primitive external) B4 (base::[c0]++E4 primitive x0 x1) world =
  (loopEval 1 (fun i cs w => evalProgram (M primitive external) I4
    (i::cs++[min (base+8) 12,base+8,base,c0]++E4 primitive x0 x1) w)
      (indexIterations base (min (base+8) 12) 4) [c0] world).bind
        (fun result => evalProgram (M primitive external) (.ret [0])
          (result.1++[min (base+8) 12,base+8,base,c0]++E4 primitive x0 x1) result.2) := by
  simp only [List.append_assoc]
  rfl
theorem outer_same (base : Nat) (carry : List Int) (world : World) (arity : carry.length = 1) :
  evalProgram (M primitive external) B4 (Int.ofNat base::carry++E4 primitive x0 x1) world =
  loopEval 1 (fun i cs w => evalProgram (M primitive external) B3 (i::cs++E3 primitive x0 x1) w)
    ((scfRange base (min (base+4*2) 12) 4).map Int.ofNat) carry world := by
  obtain ⟨c0,rfl⟩ := List.length_eq_one_iff.mp arity
  rw [outer_form]
  have bodies : ∀ i cs w, cs.length = 1 →
      evalProgram (M primitive external) I4 (i::cs++[min (Int.ofNat base+8) 12,Int.ofNat base+8,Int.ofNat base,c0]++E4 primitive x0 x1) w =
      evalProgram (M primitive external) B3 (i::cs++E3 primitive x0 x1) w := by
    intro i cs w size
    simpa only [E4,List.append_assoc,List.cons_append,List.nil_append] using
      inner_same primitive external x0 x1 i (Int.ofNat base) c0 cs w size
  rw [loopEval_congr 1 _ _ bodies]
  have minimum : min (Int.ofNat base+8) 12 = Int.ofNat (min (base+8) 12) := minui_natural (base+8) 12
  have ranges : indexIterations (Int.ofNat base) (Int.ofNat (min (base+8) 12)) 4 =
      (scfRange base (min (base+8) 12) 4).map Int.ofNat := naturalRange base (min (base+8) 12) 4 (by decide)
  rw [minimum,ranges]
  cases run : loopEval 1 (fun i cs w => evalProgram (M primitive external) B3 (i::cs++E3 primitive x0 x1) w)
      ((scfRange base (min (base+8) 12) 4).map Int.ofNat) [c0] world with
  | error failure => rfl
  | ok result =>
    have size := loopEval_arity 1 _ _ _ _ run
    rcases result with ⟨values,final⟩
    obtain ⟨returned0,rfl⟩ := List.length_eq_one_iff.mp size
    rfl
def unrollBounds : Merlin.GraphLoopsUnroll.Bounds :=
  { bits := 64, lower := 0, upper := 12, step := 1, factor := 4, count := 12, groups := 3, remainder := 0 }
def segmentBounds : Merlin.GraphLoopsSegment.Bounds :=
  { bits := 64, lower := 0, upper := 12, step := 4, tile := 2 }
theorem unrollBoundsChecked : unrollBounds.Valid := by decide
theorem segmentBoundsChecked : segmentBounds.Valid := by decide
theorem unroll_actual (world : World) :
    evalProgram (M primitive external) S2 [x0,x1] world = evalProgram (M primitive external) S3 [x0,x1] world := by
  have bodies : ∀ index values w, values.length = 1 →
      evalProgram (M primitive external) B3 (Int.ofNat index::values++E3 primitive x0 x1) w =
      loopEval 1 (fun i c current => evalProgram (M primitive external) B2 (i::c++E2 primitive x0 x1) current)
        ((progression index 1 4).map Int.ofNat) values w := by
    intro index values w size
    simpa [progression,Int.ofNat_eq_natCast,Int.add_assoc] using body_group primitive external x0 x1 (Int.ofNat index) values w size
  have loops : loopEval 1 (fun i c w => evalProgram (M primitive external) B2 (i::c++E2 primitive x0 x1) w)
      (indexIterations 0 12 1) [x0] world =
    loopEval 1 (fun i c w => evalProgram (M primitive external) B3 (i::c++E3 primitive x0 x1) w)
      (indexIterations 0 12 4) [x0] world :=
    unroll_loop 1 _ _ 0 12 1 4 3 (by decide) (by decide) (by decide) bodies [x0] world
  rw [form2,form3,loops]
  cases run : loopEval 1 (fun i c w => evalProgram (M primitive external) B3 (i::c++E3 primitive x0 x1) w)
      (indexIterations 0 12 4) [x0] world with
  | error failure => rfl
  | ok result => exact suffix_same23 primitive external x0 x1 result.1 result.2 (loopEval_arity 1 _ _ _ _ run)
theorem segment_actual (world : World) :
    evalProgram (M primitive external) S3 [x0,x1] world = evalProgram (M primitive external) S4 [x0,x1] world := by
  have r2 : indexIterations 0 12 4 = (scfRange 0 12 4).map Int.ofNat := naturalRange 0 12 4 (by decide)
  have r6 : indexIterations 0 12 8 = (scfRange 0 12 8).map Int.ofNat := naturalRange 0 12 8 (by decide)
  have loops : loopEval 1 (fun i c w => evalProgram (M primitive external) B3 (i::c++E3 primitive x0 x1) w)
      (indexIterations 0 12 4) [x0] world =
    loopEval 1 (fun i c w => evalProgram (M primitive external) B4 (i::c++E4 primitive x0 x1) w)
      (indexIterations 0 12 8) [x0] world := by
    rw [r2,r6]
    exact segment_loop 1 _ _ 0 12 4 2 (by decide) (by decide) (outer_same primitive external x0 x1) [x0] world
  rw [form3,form4,loops]
  cases run : loopEval 1 (fun i c w => evalProgram (M primitive external) B4 (i::c++E4 primitive x0 x1) w)
      (indexIterations 0 12 8) [x0] world with
  | error failure => rfl
  | ok result => exact suffix_same34 primitive external x0 x1 result.1 result.2 (loopEval_arity 1 _ _ _ _ run)
theorem actual_inputs (world : World) :
    evalProgram (M primitive external) S0 [x0,x1] world = evalProgram (M primitive external) S4 [x0,x1] world :=
  (firstTwo Int World Error (M primitive external) [x0,x1] world rfl).trans
    ((unroll_actual primitive external x0 x1 world).trans (segment_actual primitive external x0 x1 world))
theorem actual (arguments : List Int) (world : World) (arity : arguments.length = inputs.length) :
    evalProgram (M primitive external) S0 arguments world = evalProgram (M primitive external) S4 arguments world := by
  cases arguments with
  | nil => simp only [inputs,List.length_nil,List.length_cons] at arity; contradiction
  | cons first rest =>
    cases rest with
    | nil => simp only [inputs,List.length_nil,List.length_cons] at arity; contradiction
    | cons second later =>
      have empty : later = [] := List.length_eq_zero_iff.mp (by
        simp only [inputs,List.length_nil,List.length_cons] at arity
        omega)
      subst later
      exact actual_inputs primitive external first second world
end F3
namespace F4
def inputs : List Ty := [(.bits 64)]
def outputs : List Ty := [(.bits 64)]
def S0 : Program := (.ret [0])
theorem typed0 : checkTypes registry inputs outputs S0 = true := by decide
def S1 : Program := (.ret [0])
theorem typed1 : checkTypes registry inputs outputs S1 = true := by decide
def S2 : Program := (.ret [0])
theorem typed2 : checkTypes registry inputs outputs S2 = true := by decide
def S3 : Program := (.ret [0])
theorem typed3 : checkTypes registry inputs outputs S3 = true := by decide
def S4 : Program := (.ret [0])
theorem typed4 : checkTypes registry inputs outputs S4 = true := by decide
theorem firstAccepted : check registry inputs outputs S0 S1 = true := by decide
theorem secondAccepted : check registry inputs outputs S1 S2 = true := by decide
theorem firstTwo : Preserves registry inputs.length S0 S2 := check_composes firstAccepted secondAccepted
theorem actual {World Error : Type} (primitive : Prim → List Int → Int) (external : External World Error registry)
    (arguments : List Int) (world : World) (arity : arguments.length = inputs.length) :
    evalProgram (M primitive external) S0 arguments world = evalProgram (M primitive external) S4 arguments world :=
  firstTwo Int World Error (M primitive external) arguments world arity
end F4
def originalEntries : List Entry := [{inputs := F3.inputs, outputs := F3.outputs, body := F3.S0},{inputs := F4.inputs, outputs := F4.outputs, body := F4.S0}]
def finalEntries : List Entry := [{inputs := F3.inputs, outputs := F3.outputs, body := F3.S4},{inputs := F4.inputs, outputs := F4.outputs, body := F4.S4}]
theorem originalEntriesChecked : checkEntries registry originalEntries = true := by decide
theorem finalEntriesChecked : checkEntries registry finalEntries = true := by decide
theorem complete {World Error : Type} (primitive : Prim → List Int → Int)
    (external : External World Error registry) : PreservesEntries (M primitive external) originalEntries finalEntries :=
  entries_cons (M primitive external) rfl rfl (F3.actual primitive external) (entries_cons (M primitive external) rfl rfl (F4.actual primitive external) (entries_nil (M primitive external)))
end Merlin.GraphStateIntegerInstance.Suffix
set_option maxRecDepth 4096
set_option maxHeartbeats 2000000
namespace Merlin.GraphStateIntegerInstance
open Merlin.GraphState.Scalar Merlin.GraphState.Integer
abbrev registry := Suffix.registry
theorem originalRegistry : registry = [{inputs := [.index,.index],outputs := [.index]},{inputs := [],outputs := [.index]},{inputs := [],outputs := [.index]},{inputs := [],outputs := [.index]},{inputs := [],outputs := [.index]},{inputs := [],outputs := [.index]},{inputs := [],outputs := [.index]},{inputs := [],outputs := [.index]},{inputs := [.index,.index],outputs := [.index]},{inputs := [(.bits 64),.index],outputs := [(.bits 64)]},{inputs := [(.bits 64),.index],outputs := []},{inputs := [(.bits 64)],outputs := []}] := by decide
namespace F3
def inputs : List Ty := [(.bits 64),(.bits 64)]
def outputs : List Ty := [(.bits 64)]
def before : Program := (.opaque 1 [] [.index] (.opaque 2 [] [.index] (.opaque 3 [] [.index] (.pure (.constant 64 (0)) [] (.pure (.constant 64 (1)) [] (.pure (.constant 64 (-3)) [] (.pure (.constant 64 (3)) [] (.pure (.add 64) [1,0] (.pure (.add 64) [8,0] (.pure (.sub 64) [9,10] (.loop 9 8 7 [1] [(.bits 64)] (.pure (.add 64) [13,13] (.pure (.add 64) [14,14] (.pure (.mul 64) [3,9] (.pure (.add 64) [0,1] (.pure (.add 64) [0,3] (.opaque 10 [0,5] [] (.opaque 9 [0,5] [(.bits 64)] (.pure (.sub 64) [0,14] (.ret [0]))))))))) (.opaque 11 [0] [] (.ret [0])))))))))))))
theorem beforeTyped : checkTypes registry inputs outputs before = true := by decide
def after : Program := (.opaque 1 [] [.index] (.opaque 2 [] [.index] (.opaque 3 [] [.index] (.pure (.constant 64 (0)) [] (.pure (.constant 64 (1)) [] (.pure (.constant 64 (-3)) [] (.pure (.constant 64 (3)) [] (.pure (.constant 64 (0)) [] (.pure (.sub 64) [8,9] (.loop 8 7 6 [9] [(.bits 64)] (.pure (.add 64) [12,12] (.pure (.add 64) [13,13] (.pure (.add 64) [3,0] (.pure (.add 64) [0,2] (.opaque 10 [0,4] [] (.opaque 9 [0,4] [(.bits 64)] (.ret [0]))))))) (.opaque 11 [0] [] (.ret [0]))))))))))))
theorem afterTyped : checkTypes registry inputs outputs after = true := by decide
def first : Code := (.opaque 1 [] [.index] (.opaque 2 [] [.index] (.opaque 3 [] [.index] (.loop (.var 2) (.var 1) (.var 0) [(.two (.add 64) (.var 3) (.two (.add 64) (.zero (.constant 64 (-3))) (.zero (.constant 64 (3)))))] [(.bits 64)] (.opaque 10 [(.two (.add 64) (.two (.add 64) (.two (.mul 64) (.var 1) (.zero (.constant 64 (1)))) (.two (.add 64) (.var 6) (.var 6))) (.two (.add 64) (.var 6) (.var 6))),(.var 0)] [] (.opaque 9 [(.two (.add 64) (.two (.add 64) (.two (.mul 64) (.var 1) (.zero (.constant 64 (1)))) (.two (.add 64) (.var 6) (.var 6))) (.two (.add 64) (.var 6) (.var 6))),(.var 0)] [(.bits 64)] (.ret [(.two (.sub 64) (.var 0) (.zero (.constant 64 (0))))]))) (.opaque 11 [(.var 0)] [] (.ret [(.var 0)]))))))
theorem firstNormalized : normalize registry (vars inputs.length) before = some first := by decide
def second : Code := (.opaque 1 [] [.index] (.opaque 2 [] [.index] (.opaque 3 [] [.index] (.loop (.var 2) (.var 1) (.var 0) [(.var 3)] [(.bits 64)] (.opaque 10 [(.two (.add 64) (.two (.add 64) (.var 1) (.two (.add 64) (.var 6) (.var 6))) (.two (.add 64) (.var 6) (.var 6))),(.var 0)] [] (.opaque 9 [(.two (.add 64) (.two (.add 64) (.var 1) (.two (.add 64) (.var 6) (.var 6))) (.two (.add 64) (.var 6) (.var 6))),(.var 0)] [(.bits 64)] (.ret [(.var 0)]))) (.opaque 11 [(.var 0)] [] (.ret [(.var 0)]))))))
theorem secondNormalized : normalize registry (vars inputs.length) after = some second := by decide
theorem localRewrites : Relation registry inputs outputs first second :=
  (Relation.effect (by decide) (by decide) ExprsTyped.nil (Sames.nil _) (Relation.effect (by decide) (by decide) ExprsTyped.nil (Sames.nil _) (Relation.effect (by decide) (by decide) ExprsTyped.nil (Sames.nil _) (Relation.loop (ExprTyped.var (type := .index) (by decide)) (ExprTyped.var (type := .index) (by decide)) (ExprTyped.var (type := .index) (by decide)) (ExprsTyped.cons (ExprTyped.two (first := (.bits 64)) (second := (.bits 64)) (by decide) (by decide) (ExprTyped.var (type := (.bits 64)) (by decide)) (ExprTyped.two (first := (.bits 64)) (second := (.bits 64)) (by decide) (by decide) (ExprTyped.zero (by decide) (by decide)) (ExprTyped.zero (by decide) (by decide)))) ExprsTyped.nil) (Same.trans (Same.refl _ _) (Same.symm (Same.refl _ _))) (Same.trans (Same.refl _ _) (Same.symm (Same.refl _ _))) (Same.trans (Same.refl _ _) (Same.symm (Same.refl _ _))) (Sames.cons (Same.trans (Same.trans (Same.two (.add 64) (Same.refl _ _) (Same.trans (Same.two (.add 64) (Same.constant _ 64 (-3) (18446744073709551613) (by decide)) (Same.constant _ 64 (3) (3) (by decide))) (Same.fold_add _ 64 (18446744073709551613) (3) (0) (by decide)))) (Same.add_zero 64 (ExprTyped.var (type := (.bits 64)) (by decide)))) (Same.symm (Same.refl _ _))) (Sames.nil _)) (Relation.effect (by decide) (by decide) (ExprsTyped.cons (ExprTyped.two (first := (.bits 64)) (second := (.bits 64)) (by decide) (by decide) (ExprTyped.two (first := (.bits 64)) (second := (.bits 64)) (by decide) (by decide) (ExprTyped.two (first := (.bits 64)) (second := (.bits 64)) (by decide) (by decide) (ExprTyped.var (type := (.bits 64)) (by decide)) (ExprTyped.zero (by decide) (by decide))) (ExprTyped.two (first := (.bits 64)) (second := (.bits 64)) (by decide) (by decide) (ExprTyped.var (type := (.bits 64)) (by decide)) (ExprTyped.var (type := (.bits 64)) (by decide)))) (ExprTyped.two (first := (.bits 64)) (second := (.bits 64)) (by decide) (by decide) (ExprTyped.var (type := (.bits 64)) (by decide)) (ExprTyped.var (type := (.bits 64)) (by decide)))) (ExprsTyped.cons (ExprTyped.var (type := .index) (by decide)) ExprsTyped.nil)) (Sames.cons (Same.trans (Same.two (.add 64) (Same.two (.add 64) (Same.trans (Same.two (.mul 64) (Same.refl _ _) (Same.constant _ 64 (1) (1) (by decide))) (Same.mul_one 64 (ExprTyped.var (type := (.bits 64)) (by decide)))) (Same.two (.add 64) (Same.refl _ _) (Same.refl _ _))) (Same.two (.add 64) (Same.refl _ _) (Same.refl _ _))) (Same.symm (Same.two (.add 64) (Same.two (.add 64) (Same.refl _ _) (Same.two (.add 64) (Same.refl _ _) (Same.refl _ _))) (Same.two (.add 64) (Same.refl _ _) (Same.refl _ _))))) (Sames.cons (Same.trans (Same.refl _ _) (Same.symm (Same.refl _ _))) (Sames.nil _))) (Relation.effect (by decide) (by decide) (ExprsTyped.cons (ExprTyped.two (first := (.bits 64)) (second := (.bits 64)) (by decide) (by decide) (ExprTyped.two (first := (.bits 64)) (second := (.bits 64)) (by decide) (by decide) (ExprTyped.two (first := (.bits 64)) (second := (.bits 64)) (by decide) (by decide) (ExprTyped.var (type := (.bits 64)) (by decide)) (ExprTyped.zero (by decide) (by decide))) (ExprTyped.two (first := (.bits 64)) (second := (.bits 64)) (by decide) (by decide) (ExprTyped.var (type := (.bits 64)) (by decide)) (ExprTyped.var (type := (.bits 64)) (by decide)))) (ExprTyped.two (first := (.bits 64)) (second := (.bits 64)) (by decide) (by decide) (ExprTyped.var (type := (.bits 64)) (by decide)) (ExprTyped.var (type := (.bits 64)) (by decide)))) (ExprsTyped.cons (ExprTyped.var (type := .index) (by decide)) ExprsTyped.nil)) (Sames.cons (Same.trans (Same.two (.add 64) (Same.two (.add 64) (Same.trans (Same.two (.mul 64) (Same.refl _ _) (Same.constant _ 64 (1) (1) (by decide))) (Same.mul_one 64 (ExprTyped.var (type := (.bits 64)) (by decide)))) (Same.two (.add 64) (Same.refl _ _) (Same.refl _ _))) (Same.two (.add 64) (Same.refl _ _) (Same.refl _ _))) (Same.symm (Same.two (.add 64) (Same.two (.add 64) (Same.refl _ _) (Same.two (.add 64) (Same.refl _ _) (Same.refl _ _))) (Same.two (.add 64) (Same.refl _ _) (Same.refl _ _))))) (Sames.cons (Same.trans (Same.refl _ _) (Same.symm (Same.refl _ _))) (Sames.nil _))) (Relation.ret (ExprsTyped.cons (ExprTyped.two (first := (.bits 64)) (second := (.bits 64)) (by decide) (by decide) (ExprTyped.var (type := (.bits 64)) (by decide)) (ExprTyped.zero (by decide) (by decide))) ExprsTyped.nil) (Sames.cons (Same.trans (Same.trans (Same.two (.sub 64) (Same.refl _ _) (Same.constant _ 64 (0) (0) (by decide))) (Same.sub_zero 64 (ExprTyped.var (type := (.bits 64)) (by decide)))) (Same.symm (Same.refl _ _))) (Sames.nil _))))) (Relation.effect (by decide) (by decide) (ExprsTyped.cons (ExprTyped.var (type := (.bits 64)) (by decide)) ExprsTyped.nil) (Sames.cons (Same.trans (Same.refl _ _) (Same.symm (Same.refl _ _))) (Sames.nil _)) (Relation.ret (ExprsTyped.cons (ExprTyped.var (type := (.bits 64)) (by decide)) ExprsTyped.nil) (Sames.cons (Same.trans (Same.refl _ _) (Same.symm (Same.refl _ _))) (Sames.nil _))))))))
theorem actual {World Error : Type} (model : Model Int World Error registry)
    (exact : ExactModel model) (laws : LocalTyping model Canonical)
    (values : List Int) (world : World) (entry : Values Canonical inputs values) :
    evalProgram model before values world = evalProgram model after values world :=
  (programs_sound model exact laws firstNormalized secondNormalized localRewrites values world entry).2
theorem reached {World Error : Type} (model : Model Int World Error registry)
    (laws : LocalTyping model Canonical) (values : List Int) (world : World)
    (entry : Values Canonical inputs values) :
    Reached model Canonical before values world ∧ Reached model Canonical after values world :=
  ⟨typed_reached model laws (checkTypes_sound _ _ _ beforeTyped) values world entry,
   typed_reached model laws (checkTypes_sound _ _ _ afterTyped) values world entry⟩
end F3
namespace F4
def inputs : List Ty := [(.bits 64)]
def outputs : List Ty := [(.bits 64)]
def before : Program := (.ret [0])
theorem beforeTyped : checkTypes registry inputs outputs before = true := by decide
def after : Program := (.ret [0])
theorem afterTyped : checkTypes registry inputs outputs after = true := by decide
def first : Code := (.ret [(.var 0)])
theorem firstNormalized : normalize registry (vars inputs.length) before = some first := by decide
def second : Code := (.ret [(.var 0)])
theorem secondNormalized : normalize registry (vars inputs.length) after = some second := by decide
theorem localRewrites : Relation registry inputs outputs first second :=
  (Relation.ret (ExprsTyped.cons (ExprTyped.var (type := (.bits 64)) (by decide)) ExprsTyped.nil) (Sames.cons (Same.trans (Same.refl _ _) (Same.symm (Same.refl _ _))) (Sames.nil _)))
theorem actual {World Error : Type} (model : Model Int World Error registry)
    (exact : ExactModel model) (laws : LocalTyping model Canonical)
    (values : List Int) (world : World) (entry : Values Canonical inputs values) :
    evalProgram model before values world = evalProgram model after values world :=
  (programs_sound model exact laws firstNormalized secondNormalized localRewrites values world entry).2
theorem reached {World Error : Type} (model : Model Int World Error registry)
    (laws : LocalTyping model Canonical) (values : List Int) (world : World)
    (entry : Values Canonical inputs values) :
    Reached model Canonical before values world ∧ Reached model Canonical after values world :=
  ⟨typed_reached model laws (checkTypes_sound _ _ _ beforeTyped) values world entry,
   typed_reached model laws (checkTypes_sound _ _ _ afterTyped) values world entry⟩
end F4
def originalEntries : List Merlin.GraphState.Pipeline.Entry := [{inputs := F3.inputs, outputs := F3.outputs, body := F3.before},{inputs := F4.inputs, outputs := F4.outputs, body := F4.before}]
def middleEntries : List Merlin.GraphState.Pipeline.Entry := [{inputs := F3.inputs, outputs := F3.outputs, body := F3.after},{inputs := F4.inputs, outputs := F4.outputs, body := F4.after}]
theorem prefix_actual {World Error : Type} (model : Model Int World Error registry)
    (exact : ExactModel model) (laws : LocalTyping model Canonical) :
    PreservesEntriesCanonical model originalEntries middleEntries :=
  (Merlin.GraphState.Integer.entries_cons model rfl rfl (F3.actual model exact laws) (Merlin.GraphState.Integer.entries_cons model rfl rfl (F4.actual model exact laws) (Merlin.GraphState.Integer.entries_nil model)))
theorem exactMiddle : middleEntries = Suffix.originalEntries := by decide
theorem complete {World Error : Type} (external : Merlin.GraphState.Pipeline.External World Error registry)
    (typed : ∀ identity arguments world values final,
      Values Canonical (signature registry identity).inputs arguments →
      external.run identity arguments world = .ok (values,final) →
      Values Canonical (signature registry identity).outputs values) :
    PreservesEntriesCanonical (Suffix.M primitive external) originalEntries Suffix.finalEntries := by
  apply compose_suffix (Suffix.M primitive external) (prefix_actual _ ⟨rfl,rfl⟩
    (index_model_typing (Merlin.GraphState.Pipeline.checkControl_sound Suffix.controlAccepted) external typed))
  rw [exactMiddle]
  exact Suffix.complete primitive external
end Merlin.GraphStateIntegerInstance
