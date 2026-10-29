import Merlin.GraphStatePipelineIndex
import Merlin.GraphStatePipelineLiterals
import Merlin.GraphStatePipelineScope
import Merlin.GraphStatePipelineLoops
import Merlin.GraphStatePipelineProgram
import Merlin.GraphStateScalar
set_option maxRecDepth 4096
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false
namespace Merlin.GraphStatePipelineInstance
open Merlin.GraphState.Scalar Merlin.GraphState.Pipeline
def registry : List Signature := [{inputs := [.index,.index],outputs := [.index]},{inputs := [],outputs := [.index]},{inputs := [],outputs := [.index]},{inputs := [],outputs := [.index]},{inputs := [],outputs := [.index]},{inputs := [],outputs := [.index]},{inputs := [],outputs := [.index]},{inputs := [.index,.index],outputs := [.index]},{inputs := [(.bits 32),.index],outputs := [(.bits 32)]},{inputs := [(.bits 32),(.bits 32)],outputs := []},{inputs := [(.bits 32)],outputs := []}]
def controls : Controls := [(some .add),(some (.constant (2))),(some (.constant (32))),(some (.constant (42))),(some (.constant (4))),(some (.constant (6))),(some (.constant (8))),(some .minui),none,none,none]
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
namespace F4
def inputs : List Ty := [(.bits 32),(.bits 32)]
def outputs : List Ty := [(.bits 32),(.bits 32),(.bits 32)]
def S0 : Program := (.opaque 1 [] [.index] (.opaque 3 [] [.index] (.opaque 1 [] [.index] (.loop 2 1 0 [3,4,3] [(.bits 32),(.bits 32),(.bits 32)] (.pure (.add 32) [1,8] (.opaque 8 [0,1] [(.bits 32)] (.pure (.add 32) [4,10] (.opaque 8 [0,3] [(.bits 32)] (.pure (.add 32) [7,12] (.opaque 8 [0,5] [(.bits 32)] (.opaque 9 [7,14] [] (.ret [4,2,0])))))))) (.opaque 10 [0] [] (.ret [0,1,2]))))))
theorem typed0 : checkTypes registry inputs outputs S0 = true := by decide
def S1 : Program := (.opaque 1 [] [.index] (.opaque 3 [] [.index] (.opaque 1 [] [.index] (.loop 2 1 0 [3,4,3] [(.bits 32),(.bits 32),(.bits 32)] (.pure (.add 32) [1,8] (.opaque 8 [0,1] [(.bits 32)] (.pure (.add 32) [4,10] (.opaque 8 [0,3] [(.bits 32)] (.pure (.add 32) [7,12] (.opaque 8 [0,5] [(.bits 32)] (.opaque 9 [7,14] [] (.ret [4,2,0])))))))) (.opaque 10 [0] [] (.ret [0,1,2]))))))
theorem typed1 : checkTypes registry inputs outputs S1 = true := by decide
def S2 : Program := (.opaque 1 [] [.index] (.opaque 3 [] [.index] (.opaque 1 [] [.index] (.loop 2 1 0 [3,4,3] [(.bits 32),(.bits 32),(.bits 32)] (.pure (.add 32) [1,8] (.opaque 8 [0,1] [(.bits 32)] (.pure (.add 32) [4,10] (.opaque 8 [0,3] [(.bits 32)] (.pure (.add 32) [7,12] (.opaque 8 [0,5] [(.bits 32)] (.opaque 9 [7,14] [] (.ret [4,2,0])))))))) (.opaque 10 [0] [] (.ret [0,1,2]))))))
theorem typed2 : checkTypes registry inputs outputs S2 = true := by decide
def S3 : Program := (.opaque 1 [] [.index] (.opaque 3 [] [.index] (.opaque 1 [] [.index] (.opaque 6 [] [.index] (.opaque 3 [] [.index] (.loop 4 0 1 [5,6,5] [(.bits 32),(.bits 32),(.bits 32)] (.pure (.add 32) [1,10] (.opaque 8 [0,1] [(.bits 32)] (.pure (.add 32) [4,12] (.opaque 8 [0,3] [(.bits 32)] (.pure (.add 32) [7,14] (.opaque 8 [0,5] [(.bits 32)] (.opaque 9 [7,16] [] (.opaque 1 [] [.index] (.opaque 0 [7,0] [.index] (.pure (.add 32) [6,18] (.opaque 8 [0,1] [(.bits 32)] (.pure (.add 32) [6,20] (.opaque 8 [0,3] [(.bits 32)] (.pure (.add 32) [6,22] (.opaque 8 [0,5] [(.bits 32)] (.opaque 9 [12,24] [] (.opaque 4 [] [.index] (.opaque 0 [15,0] [.index] (.pure (.add 32) [6,26] (.opaque 8 [0,1] [(.bits 32)] (.pure (.add 32) [6,28] (.opaque 8 [0,3] [(.bits 32)] (.pure (.add 32) [6,30] (.opaque 8 [0,5] [(.bits 32)] (.opaque 9 [12,32] [] (.opaque 5 [] [.index] (.opaque 0 [23,0] [.index] (.pure (.add 32) [6,34] (.opaque 8 [0,1] [(.bits 32)] (.pure (.add 32) [6,36] (.opaque 8 [0,3] [(.bits 32)] (.pure (.add 32) [6,38] (.opaque 8 [0,5] [(.bits 32)] (.opaque 9 [12,40] [] (.ret [4,2,0]))))))))))))))))))))))))))))))))))) (.opaque 10 [0] [] (.ret [0,1,2]))))))))
theorem typed3 : checkTypes registry inputs outputs S3 = true := by decide
def S4 : Program := (.opaque 1 [] [.index] (.opaque 3 [] [.index] (.opaque 1 [] [.index] (.opaque 6 [] [.index] (.opaque 3 [] [.index] (.opaque 2 [] [.index] (.loop 5 1 0 [6,7,6] [(.bits 32),(.bits 32),(.bits 32)] (.opaque 0 [0,4] [.index] (.opaque 7 [0,6] [.index] (.loop 2 0 8 [3,4,5] [(.bits 32),(.bits 32),(.bits 32)] (.pure (.add 32) [1,17] (.opaque 8 [0,1] [(.bits 32)] (.pure (.add 32) [4,19] (.opaque 8 [0,3] [(.bits 32)] (.pure (.add 32) [7,21] (.opaque 8 [0,5] [(.bits 32)] (.opaque 9 [7,23] [] (.opaque 1 [] [.index] (.opaque 0 [7,0] [.index] (.pure (.add 32) [6,25] (.opaque 8 [0,1] [(.bits 32)] (.pure (.add 32) [6,27] (.opaque 8 [0,3] [(.bits 32)] (.pure (.add 32) [6,29] (.opaque 8 [0,5] [(.bits 32)] (.opaque 9 [12,31] [] (.opaque 4 [] [.index] (.opaque 0 [15,0] [.index] (.pure (.add 32) [6,33] (.opaque 8 [0,1] [(.bits 32)] (.pure (.add 32) [6,35] (.opaque 8 [0,3] [(.bits 32)] (.pure (.add 32) [6,37] (.opaque 8 [0,5] [(.bits 32)] (.opaque 9 [12,39] [] (.opaque 5 [] [.index] (.opaque 0 [23,0] [.index] (.pure (.add 32) [6,41] (.opaque 8 [0,1] [(.bits 32)] (.pure (.add 32) [6,43] (.opaque 8 [0,3] [(.bits 32)] (.pure (.add 32) [6,45] (.opaque 8 [0,5] [(.bits 32)] (.opaque 9 [12,47] [] (.ret [4,2,0]))))))))))))))))))))))))))))))))))) (.ret [0,1,2])))) (.opaque 10 [0] [] (.ret [0,1,2])))))))))
theorem typed4 : checkTypes registry inputs outputs S4 = true := by decide
theorem firstAccepted : check registry inputs outputs S0 S1 = true := by decide
theorem secondAccepted : check registry inputs outputs S1 S2 = true := by decide
theorem firstTwo : Preserves registry inputs.length S0 S2 := check_composes firstAccepted secondAccepted
def E2 (primitive : Prim → List Int → Int) (x0 x1 : Int) : List Int := [(2 : Int),(42 : Int),(2 : Int),x0,x1]
def B2 : Program := (.pure (.add 32) [1,8] (.opaque 8 [0,1] [(.bits 32)] (.pure (.add 32) [4,10] (.opaque 8 [0,3] [(.bits 32)] (.pure (.add 32) [7,12] (.opaque 8 [0,5] [(.bits 32)] (.opaque 9 [7,14] [] (.ret [4,2,0]))))))))
def N2 : Program := (.opaque 10 [0] [] (.ret [0,1,2]))
def E3 (primitive : Prim → List Int → Int) (x0 x1 : Int) : List Int := [(42 : Int),(8 : Int),(2 : Int),(42 : Int),(2 : Int),x0,x1]
def B3 : Program := (.pure (.add 32) [1,10] (.opaque 8 [0,1] [(.bits 32)] (.pure (.add 32) [4,12] (.opaque 8 [0,3] [(.bits 32)] (.pure (.add 32) [7,14] (.opaque 8 [0,5] [(.bits 32)] (.opaque 9 [7,16] [] (.opaque 1 [] [.index] (.opaque 0 [7,0] [.index] (.pure (.add 32) [6,18] (.opaque 8 [0,1] [(.bits 32)] (.pure (.add 32) [6,20] (.opaque 8 [0,3] [(.bits 32)] (.pure (.add 32) [6,22] (.opaque 8 [0,5] [(.bits 32)] (.opaque 9 [12,24] [] (.opaque 4 [] [.index] (.opaque 0 [15,0] [.index] (.pure (.add 32) [6,26] (.opaque 8 [0,1] [(.bits 32)] (.pure (.add 32) [6,28] (.opaque 8 [0,3] [(.bits 32)] (.pure (.add 32) [6,30] (.opaque 8 [0,5] [(.bits 32)] (.opaque 9 [12,32] [] (.opaque 5 [] [.index] (.opaque 0 [23,0] [.index] (.pure (.add 32) [6,34] (.opaque 8 [0,1] [(.bits 32)] (.pure (.add 32) [6,36] (.opaque 8 [0,3] [(.bits 32)] (.pure (.add 32) [6,38] (.opaque 8 [0,5] [(.bits 32)] (.opaque 9 [12,40] [] (.ret [4,2,0])))))))))))))))))))))))))))))))))))
def N3 : Program := (.opaque 10 [0] [] (.ret [0,1,2]))
def E4 (primitive : Prim → List Int → Int) (x0 x1 : Int) : List Int := [(32 : Int),(42 : Int),(8 : Int),(2 : Int),(42 : Int),(2 : Int),x0,x1]
def B4 : Program := (.opaque 0 [0,4] [.index] (.opaque 7 [0,6] [.index] (.loop 2 0 8 [3,4,5] [(.bits 32),(.bits 32),(.bits 32)] (.pure (.add 32) [1,17] (.opaque 8 [0,1] [(.bits 32)] (.pure (.add 32) [4,19] (.opaque 8 [0,3] [(.bits 32)] (.pure (.add 32) [7,21] (.opaque 8 [0,5] [(.bits 32)] (.opaque 9 [7,23] [] (.opaque 1 [] [.index] (.opaque 0 [7,0] [.index] (.pure (.add 32) [6,25] (.opaque 8 [0,1] [(.bits 32)] (.pure (.add 32) [6,27] (.opaque 8 [0,3] [(.bits 32)] (.pure (.add 32) [6,29] (.opaque 8 [0,5] [(.bits 32)] (.opaque 9 [12,31] [] (.opaque 4 [] [.index] (.opaque 0 [15,0] [.index] (.pure (.add 32) [6,33] (.opaque 8 [0,1] [(.bits 32)] (.pure (.add 32) [6,35] (.opaque 8 [0,3] [(.bits 32)] (.pure (.add 32) [6,37] (.opaque 8 [0,5] [(.bits 32)] (.opaque 9 [12,39] [] (.opaque 5 [] [.index] (.opaque 0 [23,0] [.index] (.pure (.add 32) [6,41] (.opaque 8 [0,1] [(.bits 32)] (.pure (.add 32) [6,43] (.opaque 8 [0,3] [(.bits 32)] (.pure (.add 32) [6,45] (.opaque 8 [0,5] [(.bits 32)] (.opaque 9 [12,47] [] (.ret [4,2,0]))))))))))))))))))))))))))))))))))) (.ret [0,1,2]))))
def N4 : Program := (.opaque 10 [0] [] (.ret [0,1,2]))
variable {World Error : Type} (primitive : Prim → List Int → Int)
    (external : External World Error registry) (x0 x1 : Int)
theorem form2 (world : World) : evalProgram (M primitive external) S2 [x0,x1] world =
  (loopEval 3 (fun i carry w => evalProgram (M primitive external) B2 (i::carry++E2 primitive x0 x1) w)
    (indexIterations (2 : Int) (42 : Int) (2 : Int)) [x0,x1,x0] world).bind
      (fun result => evalProgram (M primitive external) N2 (result.1++E2 primitive x0 x1) result.2) := by rfl
theorem form3 (world : World) : evalProgram (M primitive external) S3 [x0,x1] world =
  (loopEval 3 (fun i carry w => evalProgram (M primitive external) B3 (i::carry++E3 primitive x0 x1) w)
    (indexIterations (2 : Int) (42 : Int) (8 : Int)) [x0,x1,x0] world).bind
      (fun result => evalProgram (M primitive external) N3 (result.1++E3 primitive x0 x1) result.2) := by rfl
theorem form4 (world : World) : evalProgram (M primitive external) S4 [x0,x1] world =
  (loopEval 3 (fun i carry w => evalProgram (M primitive external) B4 (i::carry++E4 primitive x0 x1) w)
    (indexIterations (2 : Int) (42 : Int) (32 : Int)) [x0,x1,x0] world).bind
      (fun result => evalProgram (M primitive external) N4 (result.1++E4 primitive x0 x1) result.2) := by rfl
theorem body_group (base : Int) (carry : List Int) (world : World) (arity : carry.length = 3) :
  evalProgram (M primitive external) B3 (base::carry++E3 primitive x0 x1) world =
  loopEval 3 (fun i c w => evalProgram (M primitive external) B2 (i::c++E2 primitive x0 x1) w)
    [base,(base+2),(base+4),(base+6)] carry world := by
  obtain ⟨c0,ctail0,rfl⟩ := List.exists_cons_of_length_eq_add_one arity
  have csize0 : ctail0.length = 2 := by
    simp only [List.length_cons] at arity
    omega
  obtain ⟨c1,ctail1,rfl⟩ := List.exists_cons_of_length_eq_add_one csize0
  have csize1 : ctail1.length = 1 := by
    simp only [List.length_cons] at csize0
    omega
  obtain ⟨c2,rfl⟩ := List.length_eq_one_iff.mp csize1
  simp only [B2,B3,E2,E3,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
  cases step0 : external.run 8 [(primitive (.add 32) [c0,x1]),base] world with
  | error failure =>
    rcases failure with ⟨error,terminal⟩
    simp only [step0,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
  | ok result =>
    rcases result with ⟨values,w0⟩
    have size : values.length = 1 := by
      have localArity := external.arity 8 [(primitive (.add 32) [c0,x1]),base] world values w0 step0
      simpa only [registry,signature,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some,List.length_cons,List.length_nil] using localArity
    obtain ⟨v0_0,rfl⟩ := List.length_eq_one_iff.mp size
    simp only [step0,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
    cases step1 : external.run 8 [(primitive (.add 32) [c1,x1]),base] w0 with
    | error failure =>
      rcases failure with ⟨error,terminal⟩
      simp only [step1,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
    | ok result =>
      rcases result with ⟨values,w1⟩
      have size : values.length = 1 := by
        have localArity := external.arity 8 [(primitive (.add 32) [c1,x1]),base] w0 values w1 step1
        simpa only [registry,signature,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some,List.length_cons,List.length_nil] using localArity
      obtain ⟨v1_0,rfl⟩ := List.length_eq_one_iff.mp size
      simp only [step1,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
      cases step2 : external.run 8 [(primitive (.add 32) [c2,x1]),base] w1 with
      | error failure =>
        rcases failure with ⟨error,terminal⟩
        simp only [step2,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
      | ok result =>
        rcases result with ⟨values,w2⟩
        have size : values.length = 1 := by
          have localArity := external.arity 8 [(primitive (.add 32) [c2,x1]),base] w1 values w2 step2
          simpa only [registry,signature,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some,List.length_cons,List.length_nil] using localArity
        obtain ⟨v2_0,rfl⟩ := List.length_eq_one_iff.mp size
        simp only [step2,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
        cases step3 : external.run 9 [c0,x1] w2 with
        | error failure =>
          rcases failure with ⟨error,terminal⟩
          simp only [step3,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
        | ok result =>
          rcases result with ⟨values,w3⟩
          have size : values.length = 0 := by
            have localArity := external.arity 9 [c0,x1] w2 values w3 step3
            simpa only [registry,signature,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some,List.length_cons,List.length_nil] using localArity
          have v3_empty : values = [] := List.length_eq_zero_iff.mp size
          subst values
          simp only [step3,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
          cases step4 : external.run 8 [(primitive (.add 32) [v0_0,x1]),(base+2)] w3 with
          | error failure =>
            rcases failure with ⟨error,terminal⟩
            simp only [step4,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
          | ok result =>
            rcases result with ⟨values,w4⟩
            have size : values.length = 1 := by
              have localArity := external.arity 8 [(primitive (.add 32) [v0_0,x1]),(base+2)] w3 values w4 step4
              simpa only [registry,signature,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some,List.length_cons,List.length_nil] using localArity
            obtain ⟨v4_0,rfl⟩ := List.length_eq_one_iff.mp size
            simp only [step4,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
            cases step5 : external.run 8 [(primitive (.add 32) [v1_0,x1]),(base+2)] w4 with
            | error failure =>
              rcases failure with ⟨error,terminal⟩
              simp only [step5,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
            | ok result =>
              rcases result with ⟨values,w5⟩
              have size : values.length = 1 := by
                have localArity := external.arity 8 [(primitive (.add 32) [v1_0,x1]),(base+2)] w4 values w5 step5
                simpa only [registry,signature,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some,List.length_cons,List.length_nil] using localArity
              obtain ⟨v5_0,rfl⟩ := List.length_eq_one_iff.mp size
              simp only [step5,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
              cases step6 : external.run 8 [(primitive (.add 32) [v2_0,x1]),(base+2)] w5 with
              | error failure =>
                rcases failure with ⟨error,terminal⟩
                simp only [step6,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
              | ok result =>
                rcases result with ⟨values,w6⟩
                have size : values.length = 1 := by
                  have localArity := external.arity 8 [(primitive (.add 32) [v2_0,x1]),(base+2)] w5 values w6 step6
                  simpa only [registry,signature,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some,List.length_cons,List.length_nil] using localArity
                obtain ⟨v6_0,rfl⟩ := List.length_eq_one_iff.mp size
                simp only [step6,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
                cases step7 : external.run 9 [v0_0,x1] w6 with
                | error failure =>
                  rcases failure with ⟨error,terminal⟩
                  simp only [step7,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
                | ok result =>
                  rcases result with ⟨values,w7⟩
                  have size : values.length = 0 := by
                    have localArity := external.arity 9 [v0_0,x1] w6 values w7 step7
                    simpa only [registry,signature,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some,List.length_cons,List.length_nil] using localArity
                  have v7_empty : values = [] := List.length_eq_zero_iff.mp size
                  subst values
                  simp only [step7,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
                  cases step8 : external.run 8 [(primitive (.add 32) [v4_0,x1]),(base+4)] w7 with
                  | error failure =>
                    rcases failure with ⟨error,terminal⟩
                    simp only [step8,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
                  | ok result =>
                    rcases result with ⟨values,w8⟩
                    have size : values.length = 1 := by
                      have localArity := external.arity 8 [(primitive (.add 32) [v4_0,x1]),(base+4)] w7 values w8 step8
                      simpa only [registry,signature,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some,List.length_cons,List.length_nil] using localArity
                    obtain ⟨v8_0,rfl⟩ := List.length_eq_one_iff.mp size
                    simp only [step8,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
                    cases step9 : external.run 8 [(primitive (.add 32) [v5_0,x1]),(base+4)] w8 with
                    | error failure =>
                      rcases failure with ⟨error,terminal⟩
                      simp only [step9,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
                    | ok result =>
                      rcases result with ⟨values,w9⟩
                      have size : values.length = 1 := by
                        have localArity := external.arity 8 [(primitive (.add 32) [v5_0,x1]),(base+4)] w8 values w9 step9
                        simpa only [registry,signature,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some,List.length_cons,List.length_nil] using localArity
                      obtain ⟨v9_0,rfl⟩ := List.length_eq_one_iff.mp size
                      simp only [step9,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
                      cases step10 : external.run 8 [(primitive (.add 32) [v6_0,x1]),(base+4)] w9 with
                      | error failure =>
                        rcases failure with ⟨error,terminal⟩
                        simp only [step10,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
                      | ok result =>
                        rcases result with ⟨values,w10⟩
                        have size : values.length = 1 := by
                          have localArity := external.arity 8 [(primitive (.add 32) [v6_0,x1]),(base+4)] w9 values w10 step10
                          simpa only [registry,signature,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some,List.length_cons,List.length_nil] using localArity
                        obtain ⟨v10_0,rfl⟩ := List.length_eq_one_iff.mp size
                        simp only [step10,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
                        cases step11 : external.run 9 [v4_0,x1] w10 with
                        | error failure =>
                          rcases failure with ⟨error,terminal⟩
                          simp only [step11,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
                        | ok result =>
                          rcases result with ⟨values,w11⟩
                          have size : values.length = 0 := by
                            have localArity := external.arity 9 [v4_0,x1] w10 values w11 step11
                            simpa only [registry,signature,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some,List.length_cons,List.length_nil] using localArity
                          have v11_empty : values = [] := List.length_eq_zero_iff.mp size
                          subst values
                          simp only [step11,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
                          cases step12 : external.run 8 [(primitive (.add 32) [v8_0,x1]),(base+6)] w11 with
                          | error failure =>
                            rcases failure with ⟨error,terminal⟩
                            simp only [step12,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
                          | ok result =>
                            rcases result with ⟨values,w12⟩
                            have size : values.length = 1 := by
                              have localArity := external.arity 8 [(primitive (.add 32) [v8_0,x1]),(base+6)] w11 values w12 step12
                              simpa only [registry,signature,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some,List.length_cons,List.length_nil] using localArity
                            obtain ⟨v12_0,rfl⟩ := List.length_eq_one_iff.mp size
                            simp only [step12,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
                            cases step13 : external.run 8 [(primitive (.add 32) [v9_0,x1]),(base+6)] w12 with
                            | error failure =>
                              rcases failure with ⟨error,terminal⟩
                              simp only [step13,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
                            | ok result =>
                              rcases result with ⟨values,w13⟩
                              have size : values.length = 1 := by
                                have localArity := external.arity 8 [(primitive (.add 32) [v9_0,x1]),(base+6)] w12 values w13 step13
                                simpa only [registry,signature,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some,List.length_cons,List.length_nil] using localArity
                              obtain ⟨v13_0,rfl⟩ := List.length_eq_one_iff.mp size
                              simp only [step13,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
                              cases step14 : external.run 8 [(primitive (.add 32) [v10_0,x1]),(base+6)] w13 with
                              | error failure =>
                                rcases failure with ⟨error,terminal⟩
                                simp only [step14,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
                              | ok result =>
                                rcases result with ⟨values,w14⟩
                                have size : values.length = 1 := by
                                  have localArity := external.arity 8 [(primitive (.add 32) [v10_0,x1]),(base+6)] w13 values w14 step14
                                  simpa only [registry,signature,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some,List.length_cons,List.length_nil] using localArity
                                obtain ⟨v14_0,rfl⟩ := List.length_eq_one_iff.mp size
                                simp only [step14,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
                                cases step15 : external.run 9 [v8_0,x1] w14 with
                                | error failure =>
                                  rcases failure with ⟨error,terminal⟩
                                  simp only [step15,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
                                | ok result =>
                                  rcases result with ⟨values,w15⟩
                                  have size : values.length = 0 := by
                                    have localArity := external.arity 9 [v8_0,x1] w14 values w15 step15
                                    simpa only [registry,signature,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some,List.length_cons,List.length_nil] using localArity
                                  have v15_empty : values = [] := List.length_eq_zero_iff.mp size
                                  subst values
                                  simp only [step15,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
def I4 : Program := (.pure (.add 32) [1,17] (.opaque 8 [0,1] [(.bits 32)] (.pure (.add 32) [4,19] (.opaque 8 [0,3] [(.bits 32)] (.pure (.add 32) [7,21] (.opaque 8 [0,5] [(.bits 32)] (.opaque 9 [7,23] [] (.opaque 1 [] [.index] (.opaque 0 [7,0] [.index] (.pure (.add 32) [6,25] (.opaque 8 [0,1] [(.bits 32)] (.pure (.add 32) [6,27] (.opaque 8 [0,3] [(.bits 32)] (.pure (.add 32) [6,29] (.opaque 8 [0,5] [(.bits 32)] (.opaque 9 [12,31] [] (.opaque 4 [] [.index] (.opaque 0 [15,0] [.index] (.pure (.add 32) [6,33] (.opaque 8 [0,1] [(.bits 32)] (.pure (.add 32) [6,35] (.opaque 8 [0,3] [(.bits 32)] (.pure (.add 32) [6,37] (.opaque 8 [0,5] [(.bits 32)] (.opaque 9 [12,39] [] (.opaque 5 [] [.index] (.opaque 0 [23,0] [.index] (.pure (.add 32) [6,41] (.opaque 8 [0,1] [(.bits 32)] (.pure (.add 32) [6,43] (.opaque 8 [0,3] [(.bits 32)] (.pure (.add 32) [6,45] (.opaque 8 [0,5] [(.bits 32)] (.opaque 9 [12,47] [] (.ret [4,2,0])))))))))))))))))))))))))))))))))))
def LS : List Expr := [(.var 0),(.var 1),(.var 2),(.var 3),(.var 11),(.var 12),(.var 13),(.var 11),(.var 13),(.var 16),(.var 17)]
def RS : List Expr := [(.var 0),(.var 1),(.var 2),(.var 3),(.var 4),(.var 5),(.var 6),(.var 7),(.var 8),(.var 9),(.var 10),(.var 11),(.var 12),(.var 13),(.var 14),(.var 15),(.var 16),(.var 17)]
theorem innerAccepted : Scope.check registry LS RS B3 I4 = true := by decide
theorem inner_same (iv outer outerCarry0 outerCarry1 outerCarry2 : Int) (carry : List Int) (world : World) (arity : carry.length = 3) :
  evalProgram (M primitive external) I4 (iv::carry++[(min (outer + (32 : Int)) (42 : Int)),(outer + (32 : Int)),outer,outerCarry0,outerCarry1,outerCarry2,(32 : Int),(42 : Int),(8 : Int),(2 : Int),(42 : Int),(2 : Int),x0,x1]) world =
  evalProgram (M primitive external) B3 (iv::carry++E3 primitive x0 x1) world := by
  obtain ⟨c0,ctail0,rfl⟩ := List.exists_cons_of_length_eq_add_one arity
  have csize0 : ctail0.length = 2 := by
    simp only [List.length_cons] at arity
    omega
  obtain ⟨c1,ctail1,rfl⟩ := List.exists_cons_of_length_eq_add_one csize0
  have csize1 : ctail1.length = 1 := by
    simp only [List.length_cons] at csize0
    omega
  obtain ⟨c2,rfl⟩ := List.length_eq_one_iff.mp csize1
  have checked := Scope.check_sound innerAccepted (M primitive external) [iv,c0,c1,c2,(min (outer + (32 : Int)) (42 : Int)),(outer + (32 : Int)),outer,outerCarry0,outerCarry1,outerCarry2,(32 : Int),(42 : Int),(8 : Int),(2 : Int),(42 : Int),(2 : Int),x0,x1] world
  simpa only [LS,RS,E3,evalExpr,List.cons_append,List.nil_append,List.map_cons,List.map_nil,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some] using checked.symm
def CL23 : List Expr := [(.var 0),(.var 1),(.var 2),(.var 5),(.var 3),(.var 5),(.var 8),(.var 9)]
def CR23 : List Expr := [(.var 0),(.var 1),(.var 2),(.var 3),(.var 4),(.var 5),(.var 6),(.var 7),(.var 8),(.var 9)]
theorem suffixAccepted23 : Scope.check registry CL23 CR23 N2 N3 = true := by decide
theorem suffix_same23 (carry : List Int) (world : World) (arity : carry.length = 3) :
  evalProgram (M primitive external) N2 (carry++E2 primitive x0 x1) world =
  evalProgram (M primitive external) N3 (carry++E3 primitive x0 x1) world := by
  obtain ⟨c0,ctail0,rfl⟩ := List.exists_cons_of_length_eq_add_one arity
  have csize0 : ctail0.length = 2 := by
    simp only [List.length_cons] at arity
    omega
  obtain ⟨c1,ctail1,rfl⟩ := List.exists_cons_of_length_eq_add_one csize0
  have csize1 : ctail1.length = 1 := by
    simp only [List.length_cons] at csize0
    omega
  obtain ⟨c2,rfl⟩ := List.length_eq_one_iff.mp csize1
  have checked := Scope.check_sound suffixAccepted23 (M primitive external) [c0,c1,c2,(42 : Int),(8 : Int),(2 : Int),(42 : Int),(2 : Int),x0,x1] world
  simpa only [CL23,CR23,E2,E3,evalExpr,List.cons_append,List.nil_append,List.map_cons,List.map_nil,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some] using checked
def CL34 : List Expr := [(.var 0),(.var 1),(.var 2),(.var 4),(.var 5),(.var 6),(.var 4),(.var 6),(.var 9),(.var 10)]
def CR34 : List Expr := [(.var 0),(.var 1),(.var 2),(.var 3),(.var 4),(.var 5),(.var 6),(.var 7),(.var 8),(.var 9),(.var 10)]
theorem suffixAccepted34 : Scope.check registry CL34 CR34 N3 N4 = true := by decide
theorem suffix_same34 (carry : List Int) (world : World) (arity : carry.length = 3) :
  evalProgram (M primitive external) N3 (carry++E3 primitive x0 x1) world =
  evalProgram (M primitive external) N4 (carry++E4 primitive x0 x1) world := by
  obtain ⟨c0,ctail0,rfl⟩ := List.exists_cons_of_length_eq_add_one arity
  have csize0 : ctail0.length = 2 := by
    simp only [List.length_cons] at arity
    omega
  obtain ⟨c1,ctail1,rfl⟩ := List.exists_cons_of_length_eq_add_one csize0
  have csize1 : ctail1.length = 1 := by
    simp only [List.length_cons] at csize0
    omega
  obtain ⟨c2,rfl⟩ := List.length_eq_one_iff.mp csize1
  have checked := Scope.check_sound suffixAccepted34 (M primitive external) [c0,c1,c2,(32 : Int),(42 : Int),(8 : Int),(2 : Int),(42 : Int),(2 : Int),x0,x1] world
  simpa only [CL34,CR34,E3,E4,evalExpr,List.cons_append,List.nil_append,List.map_cons,List.map_nil,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some] using checked
theorem outer_form (base c0 c1 c2 : Int) (world : World) :
  evalProgram (M primitive external) B4 (base::[c0,c1,c2]++E4 primitive x0 x1) world =
  (loopEval 3 (fun i cs w => evalProgram (M primitive external) I4
    (i::cs++[min (base+32) 42,base+32,base,c0,c1,c2]++E4 primitive x0 x1) w)
      (indexIterations base (min (base+32) 42) 8) [c0,c1,c2] world).bind
        (fun result => evalProgram (M primitive external) (.ret [0,1,2])
          (result.1++[min (base+32) 42,base+32,base,c0,c1,c2]++E4 primitive x0 x1) result.2) := by
  simp only [List.append_assoc]
  rfl
theorem outer_same (base : Nat) (carry : List Int) (world : World) (arity : carry.length = 3) :
  evalProgram (M primitive external) B4 (Int.ofNat base::carry++E4 primitive x0 x1) world =
  loopEval 3 (fun i cs w => evalProgram (M primitive external) B3 (i::cs++E3 primitive x0 x1) w)
    ((scfRange base (min (base+8*4) 42) 8).map Int.ofNat) carry world := by
  obtain ⟨c0,ctail0,rfl⟩ := List.exists_cons_of_length_eq_add_one arity
  have csize0 : ctail0.length = 2 := by
    simp only [List.length_cons] at arity
    omega
  obtain ⟨c1,ctail1,rfl⟩ := List.exists_cons_of_length_eq_add_one csize0
  have csize1 : ctail1.length = 1 := by
    simp only [List.length_cons] at csize0
    omega
  obtain ⟨c2,rfl⟩ := List.length_eq_one_iff.mp csize1
  rw [outer_form]
  have bodies : ∀ i cs w, cs.length = 3 →
      evalProgram (M primitive external) I4 (i::cs++[min (Int.ofNat base+32) 42,Int.ofNat base+32,Int.ofNat base,c0,c1,c2]++E4 primitive x0 x1) w =
      evalProgram (M primitive external) B3 (i::cs++E3 primitive x0 x1) w := by
    intro i cs w size
    simpa only [E4,List.append_assoc,List.cons_append,List.nil_append] using
      inner_same primitive external x0 x1 i (Int.ofNat base) c0 c1 c2 cs w size
  rw [loopEval_congr 3 _ _ bodies]
  have minimum : min (Int.ofNat base+32) 42 = Int.ofNat (min (base+32) 42) := minui_natural (base+32) 42
  have ranges : indexIterations (Int.ofNat base) (Int.ofNat (min (base+32) 42)) 8 =
      (scfRange base (min (base+32) 42) 8).map Int.ofNat := naturalRange base (min (base+32) 42) 8 (by decide)
  rw [minimum,ranges]
  cases run : loopEval 3 (fun i cs w => evalProgram (M primitive external) B3 (i::cs++E3 primitive x0 x1) w)
      ((scfRange base (min (base+32) 42) 8).map Int.ofNat) [c0,c1,c2] world with
  | error failure => rfl
  | ok result =>
    have size := loopEval_arity 3 _ _ _ _ run
    rcases result with ⟨values,final⟩
    obtain ⟨returned0,returnedtail0,rfl⟩ := List.exists_cons_of_length_eq_add_one size
    have returnedsize0 : returnedtail0.length = 2 := by
      simp only [List.length_cons] at size
      omega
    obtain ⟨returned1,returnedtail1,rfl⟩ := List.exists_cons_of_length_eq_add_one returnedsize0
    have returnedsize1 : returnedtail1.length = 1 := by
      simp only [List.length_cons] at returnedsize0
      omega
    obtain ⟨returned2,rfl⟩ := List.length_eq_one_iff.mp returnedsize1
    rfl
def unrollBounds : Merlin.GraphLoopsUnroll.Bounds :=
  { bits := 64, lower := 2, upper := 42, step := 2, factor := 4, count := 20, groups := 5, remainder := 0 }
def segmentBounds : Merlin.GraphLoopsSegment.Bounds :=
  { bits := 64, lower := 2, upper := 42, step := 8, tile := 4 }
theorem unrollBoundsChecked : unrollBounds.Valid := by decide
theorem segmentBoundsChecked : segmentBounds.Valid := by decide
theorem unroll_actual (world : World) :
    evalProgram (M primitive external) S2 [x0,x1] world = evalProgram (M primitive external) S3 [x0,x1] world := by
  have bodies : ∀ index values w, values.length = 3 →
      evalProgram (M primitive external) B3 (Int.ofNat index::values++E3 primitive x0 x1) w =
      loopEval 3 (fun i c current => evalProgram (M primitive external) B2 (i::c++E2 primitive x0 x1) current)
        ((progression index 2 4).map Int.ofNat) values w := by
    intro index values w size
    simpa [progression,Int.ofNat_eq_natCast,Int.add_assoc] using body_group primitive external x0 x1 (Int.ofNat index) values w size
  have loops : loopEval 3 (fun i c w => evalProgram (M primitive external) B2 (i::c++E2 primitive x0 x1) w)
      (indexIterations 2 42 2) [x0,x1,x0] world =
    loopEval 3 (fun i c w => evalProgram (M primitive external) B3 (i::c++E3 primitive x0 x1) w)
      (indexIterations 2 42 8) [x0,x1,x0] world :=
    unroll_loop 3 _ _ 2 42 2 4 5 (by decide) (by decide) (by decide) bodies [x0,x1,x0] world
  rw [form2,form3,loops]
  cases run : loopEval 3 (fun i c w => evalProgram (M primitive external) B3 (i::c++E3 primitive x0 x1) w)
      (indexIterations 2 42 8) [x0,x1,x0] world with
  | error failure => rfl
  | ok result => exact suffix_same23 primitive external x0 x1 result.1 result.2 (loopEval_arity 3 _ _ _ _ run)
theorem segment_actual (world : World) :
    evalProgram (M primitive external) S3 [x0,x1] world = evalProgram (M primitive external) S4 [x0,x1] world := by
  have r2 : indexIterations 2 42 8 = (scfRange 2 42 8).map Int.ofNat := naturalRange 2 42 8 (by decide)
  have r6 : indexIterations 2 42 32 = (scfRange 2 42 32).map Int.ofNat := naturalRange 2 42 32 (by decide)
  have loops : loopEval 3 (fun i c w => evalProgram (M primitive external) B3 (i::c++E3 primitive x0 x1) w)
      (indexIterations 2 42 8) [x0,x1,x0] world =
    loopEval 3 (fun i c w => evalProgram (M primitive external) B4 (i::c++E4 primitive x0 x1) w)
      (indexIterations 2 42 32) [x0,x1,x0] world := by
    rw [r2,r6]
    exact segment_loop 3 _ _ 2 42 8 4 (by decide) (by decide) (outer_same primitive external x0 x1) [x0,x1,x0] world
  rw [form3,form4,loops]
  cases run : loopEval 3 (fun i c w => evalProgram (M primitive external) B4 (i::c++E4 primitive x0 x1) w)
      (indexIterations 2 42 32) [x0,x1,x0] world with
  | error failure => rfl
  | ok result => exact suffix_same34 primitive external x0 x1 result.1 result.2 (loopEval_arity 3 _ _ _ _ run)
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
end F4
namespace F5
def inputs : List Ty := [(.bits 32)]
def outputs : List Ty := [(.bits 32)]
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
end F5
def originalEntries : List Entry := [{inputs := F4.inputs, outputs := F4.outputs, body := F4.S0},{inputs := F5.inputs, outputs := F5.outputs, body := F5.S0}]
def finalEntries : List Entry := [{inputs := F4.inputs, outputs := F4.outputs, body := F4.S4},{inputs := F5.inputs, outputs := F5.outputs, body := F5.S4}]
theorem originalEntriesChecked : checkEntries registry originalEntries = true := by decide
theorem finalEntriesChecked : checkEntries registry finalEntries = true := by decide
theorem complete {World Error : Type} (primitive : Prim → List Int → Int)
    (external : External World Error registry) : PreservesEntries (M primitive external) originalEntries finalEntries :=
  entries_cons (M primitive external) rfl rfl (F4.actual primitive external) (entries_cons (M primitive external) rfl rfl (F5.actual primitive external) (entries_nil (M primitive external)))
end Merlin.GraphStatePipelineInstance
