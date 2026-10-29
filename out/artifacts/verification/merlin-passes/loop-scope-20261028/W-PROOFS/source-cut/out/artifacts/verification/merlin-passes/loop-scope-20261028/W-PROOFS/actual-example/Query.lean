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
def registry : List Signature := [{inputs := [.index,.index],outputs := [.index]},{inputs := [],outputs := [.index]},{inputs := [],outputs := [.index]},{inputs := [],outputs := [.index]},{inputs := [],outputs := [.index]},{inputs := [],outputs := [.index]},{inputs := [.index,.index],outputs := [.index]},{inputs := [(.bits 32),.index],outputs := [(.bits 32)]},{inputs := [(.bits 32)],outputs := []}]
def controls : Controls := [(some .add),(some (.constant (12))),(some (.constant (27))),(some (.constant (2))),(some (.constant (3))),(some (.constant (4))),(some .minui),none,none]
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
namespace F2
def inputs : List Ty := [(.bits 32),(.bits 32)]
def outputs : List Ty := [(.bits 32)]
def S0 : Program := (.opaque 4 [] [.index] (.opaque 2 [] [.index] (.opaque 3 [] [.index] (.loop 2 1 0 [3] [(.bits 32)] (.pure (.add 32) [1,6] (.opaque 7 [0,1] [(.bits 32)] (.ret [0]))) (.opaque 8 [0] [] (.ret [0]))))))
theorem typed0 : checkTypes registry inputs outputs S0 = true := by decide
def S1 : Program := (.opaque 4 [] [.index] (.opaque 2 [] [.index] (.opaque 3 [] [.index] (.loop 2 1 0 [3] [(.bits 32)] (.pure (.add 32) [1,6] (.opaque 7 [0,1] [(.bits 32)] (.ret [0]))) (.opaque 8 [0] [] (.ret [0]))))))
theorem typed1 : checkTypes registry inputs outputs S1 = true := by decide
def S2 : Program := (.opaque 4 [] [.index] (.opaque 2 [] [.index] (.opaque 3 [] [.index] (.loop 2 1 0 [3] [(.bits 32)] (.pure (.add 32) [1,6] (.opaque 7 [0,1] [(.bits 32)] (.ret [0]))) (.opaque 8 [0] [] (.ret [0]))))))
theorem typed2 : checkTypes registry inputs outputs S2 = true := by decide
def S3 : Program := (.opaque 4 [] [.index] (.opaque 2 [] [.index] (.opaque 3 [] [.index] (.opaque 5 [] [.index] (.opaque 2 [] [.index] (.loop 4 0 1 [5] [(.bits 32)] (.pure (.add 32) [1,8] (.opaque 7 [0,1] [(.bits 32)] (.opaque 3 [] [.index] (.opaque 0 [3,0] [.index] (.pure (.add 32) [2,12] (.opaque 7 [0,1] [(.bits 32)] (.ret [0]))))))) (.opaque 8 [0] [] (.ret [0]))))))))
theorem typed3 : checkTypes registry inputs outputs S3 = true := by decide
def S4 : Program := (.opaque 4 [] [.index] (.opaque 2 [] [.index] (.opaque 3 [] [.index] (.opaque 5 [] [.index] (.opaque 2 [] [.index] (.opaque 1 [] [.index] (.loop 5 1 0 [6] [(.bits 32)] (.opaque 0 [0,2] [.index] (.opaque 6 [0,4] [.index] (.loop 2 0 6 [3] [(.bits 32)] (.pure (.add 32) [1,13] (.opaque 7 [0,1] [(.bits 32)] (.opaque 3 [] [.index] (.opaque 0 [3,0] [.index] (.pure (.add 32) [2,17] (.opaque 7 [0,1] [(.bits 32)] (.ret [0]))))))) (.ret [0])))) (.opaque 8 [0] [] (.ret [0])))))))))
theorem typed4 : checkTypes registry inputs outputs S4 = true := by decide
theorem firstAccepted : check registry inputs outputs S0 S1 = true := by decide
theorem secondAccepted : check registry inputs outputs S1 S2 = true := by decide
theorem firstTwo : Preserves registry inputs.length S0 S2 := check_composes firstAccepted secondAccepted
def E2 (primitive : Prim → List Int → Int) (x0 x1 : Int) : List Int := [(2 : Int),(27 : Int),(3 : Int),x0,x1]
def B2 : Program := (.pure (.add 32) [1,6] (.opaque 7 [0,1] [(.bits 32)] (.ret [0])))
def N2 : Program := (.opaque 8 [0] [] (.ret [0]))
def E3 (primitive : Prim → List Int → Int) (x0 x1 : Int) : List Int := [(27 : Int),(4 : Int),(2 : Int),(27 : Int),(3 : Int),x0,x1]
def B3 : Program := (.pure (.add 32) [1,8] (.opaque 7 [0,1] [(.bits 32)] (.opaque 3 [] [.index] (.opaque 0 [3,0] [.index] (.pure (.add 32) [2,12] (.opaque 7 [0,1] [(.bits 32)] (.ret [0])))))))
def N3 : Program := (.opaque 8 [0] [] (.ret [0]))
def E4 (primitive : Prim → List Int → Int) (x0 x1 : Int) : List Int := [(12 : Int),(27 : Int),(4 : Int),(2 : Int),(27 : Int),(3 : Int),x0,x1]
def B4 : Program := (.opaque 0 [0,2] [.index] (.opaque 6 [0,4] [.index] (.loop 2 0 6 [3] [(.bits 32)] (.pure (.add 32) [1,13] (.opaque 7 [0,1] [(.bits 32)] (.opaque 3 [] [.index] (.opaque 0 [3,0] [.index] (.pure (.add 32) [2,17] (.opaque 7 [0,1] [(.bits 32)] (.ret [0]))))))) (.ret [0]))))
def N4 : Program := (.opaque 8 [0] [] (.ret [0]))
variable {World Error : Type} (primitive : Prim → List Int → Int)
    (external : External World Error registry) (x0 x1 : Int)
theorem form2 (world : World) : evalProgram (M primitive external) S2 [x0,x1] world =
  (loopEval 1 (fun i carry w => evalProgram (M primitive external) B2 (i::carry++E2 primitive x0 x1) w)
    (indexIterations (3 : Int) (27 : Int) (2 : Int)) [x0] world).bind
      (fun result => evalProgram (M primitive external) N2 (result.1++E2 primitive x0 x1) result.2) := by rfl
theorem form3 (world : World) : evalProgram (M primitive external) S3 [x0,x1] world =
  (loopEval 1 (fun i carry w => evalProgram (M primitive external) B3 (i::carry++E3 primitive x0 x1) w)
    (indexIterations (3 : Int) (27 : Int) (4 : Int)) [x0] world).bind
      (fun result => evalProgram (M primitive external) N3 (result.1++E3 primitive x0 x1) result.2) := by rfl
theorem form4 (world : World) : evalProgram (M primitive external) S4 [x0,x1] world =
  (loopEval 1 (fun i carry w => evalProgram (M primitive external) B4 (i::carry++E4 primitive x0 x1) w)
    (indexIterations (3 : Int) (27 : Int) (12 : Int)) [x0] world).bind
      (fun result => evalProgram (M primitive external) N4 (result.1++E4 primitive x0 x1) result.2) := by rfl
theorem body_group (base : Int) (carry : List Int) (world : World) (arity : carry.length = 1) :
  evalProgram (M primitive external) B3 (base::carry++E3 primitive x0 x1) world =
  loopEval 1 (fun i c w => evalProgram (M primitive external) B2 (i::c++E2 primitive x0 x1) w)
    [base,(base+2)] carry world := by
  obtain ⟨c0,rfl⟩ := List.length_eq_one_iff.mp arity
  simp only [B2,B3,E2,E3,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
  cases step0 : external.run 7 [(primitive (.add 32) [c0,x1]),base] world with
  | error failure =>
    rcases failure with ⟨error,terminal⟩
    simp only [step0,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
  | ok result =>
    rcases result with ⟨values,w0⟩
    have size : values.length = 1 := by
      have localArity := external.arity 7 [(primitive (.add 32) [c0,x1]),base] world values w0 step0
      simpa only [registry,signature,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some,List.length_cons,List.length_nil] using localArity
    obtain ⟨v0_0,rfl⟩ := List.length_eq_one_iff.mp size
    simp only [step0,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
    cases step1 : external.run 7 [(primitive (.add 32) [v0_0,x1]),(base+2)] w0 with
    | error failure =>
      rcases failure with ⟨error,terminal⟩
      simp only [step1,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
    | ok result =>
      rcases result with ⟨values,w1⟩
      have size : values.length = 1 := by
        have localArity := external.arity 7 [(primitive (.add 32) [v0_0,x1]),(base+2)] w0 values w1 step1
        simpa only [registry,signature,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some,List.length_cons,List.length_nil] using localArity
      obtain ⟨v1_0,rfl⟩ := List.length_eq_one_iff.mp size
      simp only [step1,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
def I4 : Program := (.pure (.add 32) [1,13] (.opaque 7 [0,1] [(.bits 32)] (.opaque 3 [] [.index] (.opaque 0 [3,0] [.index] (.pure (.add 32) [2,17] (.opaque 7 [0,1] [(.bits 32)] (.ret [0])))))))
def LS : List Expr := [(.var 0),(.var 1),(.var 7),(.var 8),(.var 9),(.var 7),(.var 11),(.var 12),(.var 13)]
def RS : List Expr := [(.var 0),(.var 1),(.var 2),(.var 3),(.var 4),(.var 5),(.var 6),(.var 7),(.var 8),(.var 9),(.var 10),(.var 11),(.var 12),(.var 13)]
theorem innerAccepted : Scope.check registry LS RS B3 I4 = true := by decide
theorem inner_same (iv outer outerCarry0 : Int) (carry : List Int) (world : World) (arity : carry.length = 1) :
  evalProgram (M primitive external) I4 (iv::carry++[(min (outer + (12 : Int)) (27 : Int)),(outer + (12 : Int)),outer,outerCarry0,(12 : Int),(27 : Int),(4 : Int),(2 : Int),(27 : Int),(3 : Int),x0,x1]) world =
  evalProgram (M primitive external) B3 (iv::carry++E3 primitive x0 x1) world := by
  obtain ⟨c0,rfl⟩ := List.length_eq_one_iff.mp arity
  have checked := Scope.check_sound innerAccepted (M primitive external) [iv,c0,(min (outer + (12 : Int)) (27 : Int)),(outer + (12 : Int)),outer,outerCarry0,(12 : Int),(27 : Int),(4 : Int),(2 : Int),(27 : Int),(3 : Int),x0,x1] world
  simpa only [LS,RS,E3,evalExpr,List.cons_append,List.nil_append,List.map_cons,List.map_nil,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some] using checked.symm
def CL23 : List Expr := [(.var 0),(.var 3),(.var 1),(.var 5),(.var 6),(.var 7)]
def CR23 : List Expr := [(.var 0),(.var 1),(.var 2),(.var 3),(.var 4),(.var 5),(.var 6),(.var 7)]
theorem suffixAccepted23 : Scope.check registry CL23 CR23 N2 N3 = true := by decide
theorem suffix_same23 (carry : List Int) (world : World) (arity : carry.length = 1) :
  evalProgram (M primitive external) N2 (carry++E2 primitive x0 x1) world =
  evalProgram (M primitive external) N3 (carry++E3 primitive x0 x1) world := by
  obtain ⟨c0,rfl⟩ := List.length_eq_one_iff.mp arity
  have checked := Scope.check_sound suffixAccepted23 (M primitive external) [c0,(27 : Int),(4 : Int),(2 : Int),(27 : Int),(3 : Int),x0,x1] world
  simpa only [CL23,CR23,E2,E3,evalExpr,List.cons_append,List.nil_append,List.map_cons,List.map_nil,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some] using checked
def CL34 : List Expr := [(.var 0),(.var 2),(.var 3),(.var 4),(.var 2),(.var 6),(.var 7),(.var 8)]
def CR34 : List Expr := [(.var 0),(.var 1),(.var 2),(.var 3),(.var 4),(.var 5),(.var 6),(.var 7),(.var 8)]
theorem suffixAccepted34 : Scope.check registry CL34 CR34 N3 N4 = true := by decide
theorem suffix_same34 (carry : List Int) (world : World) (arity : carry.length = 1) :
  evalProgram (M primitive external) N3 (carry++E3 primitive x0 x1) world =
  evalProgram (M primitive external) N4 (carry++E4 primitive x0 x1) world := by
  obtain ⟨c0,rfl⟩ := List.length_eq_one_iff.mp arity
  have checked := Scope.check_sound suffixAccepted34 (M primitive external) [c0,(12 : Int),(27 : Int),(4 : Int),(2 : Int),(27 : Int),(3 : Int),x0,x1] world
  simpa only [CL34,CR34,E3,E4,evalExpr,List.cons_append,List.nil_append,List.map_cons,List.map_nil,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some] using checked
theorem outer_form (base c0 : Int) (world : World) :
  evalProgram (M primitive external) B4 (base::[c0]++E4 primitive x0 x1) world =
  (loopEval 1 (fun i cs w => evalProgram (M primitive external) I4
    (i::cs++[min (base+12) 27,base+12,base,c0]++E4 primitive x0 x1) w)
      (indexIterations base (min (base+12) 27) 4) [c0] world).bind
        (fun result => evalProgram (M primitive external) (.ret [0])
          (result.1++[min (base+12) 27,base+12,base,c0]++E4 primitive x0 x1) result.2) := by
  simp only [List.append_assoc]
  rfl
theorem outer_same (base : Nat) (carry : List Int) (world : World) (arity : carry.length = 1) :
  evalProgram (M primitive external) B4 (Int.ofNat base::carry++E4 primitive x0 x1) world =
  loopEval 1 (fun i cs w => evalProgram (M primitive external) B3 (i::cs++E3 primitive x0 x1) w)
    ((scfRange base (min (base+4*3) 27) 4).map Int.ofNat) carry world := by
  obtain ⟨c0,rfl⟩ := List.length_eq_one_iff.mp arity
  rw [outer_form]
  have bodies : ∀ i cs w, cs.length = 1 →
      evalProgram (M primitive external) I4 (i::cs++[min (Int.ofNat base+12) 27,Int.ofNat base+12,Int.ofNat base,c0]++E4 primitive x0 x1) w =
      evalProgram (M primitive external) B3 (i::cs++E3 primitive x0 x1) w := by
    intro i cs w size
    simpa only [E4,List.append_assoc,List.cons_append,List.nil_append] using
      inner_same primitive external x0 x1 i (Int.ofNat base) c0 cs w size
  rw [loopEval_congr 1 _ _ bodies]
  have minimum : min (Int.ofNat base+12) 27 = Int.ofNat (min (base+12) 27) := minui_natural (base+12) 27
  have ranges : indexIterations (Int.ofNat base) (Int.ofNat (min (base+12) 27)) 4 =
      (scfRange base (min (base+12) 27) 4).map Int.ofNat := naturalRange base (min (base+12) 27) 4 (by decide)
  rw [minimum,ranges]
  cases run : loopEval 1 (fun i cs w => evalProgram (M primitive external) B3 (i::cs++E3 primitive x0 x1) w)
      ((scfRange base (min (base+12) 27) 4).map Int.ofNat) [c0] world with
  | error failure => rfl
  | ok result =>
    have size := loopEval_arity 1 _ _ _ _ run
    rcases result with ⟨values,final⟩
    obtain ⟨returned0,rfl⟩ := List.length_eq_one_iff.mp size
    rfl
def unrollBounds : Merlin.GraphLoopsUnroll.Bounds :=
  { bits := 64, lower := 3, upper := 27, step := 2, factor := 2, count := 12, groups := 6, remainder := 0 }
def segmentBounds : Merlin.GraphLoopsSegment.Bounds :=
  { bits := 64, lower := 3, upper := 27, step := 4, tile := 3 }
theorem unrollBoundsChecked : unrollBounds.Valid := by decide
theorem segmentBoundsChecked : segmentBounds.Valid := by decide
theorem unroll_actual (world : World) :
    evalProgram (M primitive external) S2 [x0,x1] world = evalProgram (M primitive external) S3 [x0,x1] world := by
  have bodies : ∀ index values w, values.length = 1 →
      evalProgram (M primitive external) B3 (Int.ofNat index::values++E3 primitive x0 x1) w =
      loopEval 1 (fun i c current => evalProgram (M primitive external) B2 (i::c++E2 primitive x0 x1) current)
        ((progression index 2 2).map Int.ofNat) values w := by
    intro index values w size
    simpa [progression,Int.ofNat_eq_natCast,Int.add_assoc] using body_group primitive external x0 x1 (Int.ofNat index) values w size
  have loops : loopEval 1 (fun i c w => evalProgram (M primitive external) B2 (i::c++E2 primitive x0 x1) w)
      (indexIterations 3 27 2) [x0] world =
    loopEval 1 (fun i c w => evalProgram (M primitive external) B3 (i::c++E3 primitive x0 x1) w)
      (indexIterations 3 27 4) [x0] world :=
    unroll_loop 1 _ _ 3 27 2 2 6 (by decide) (by decide) (by decide) bodies [x0] world
  rw [form2,form3,loops]
  cases run : loopEval 1 (fun i c w => evalProgram (M primitive external) B3 (i::c++E3 primitive x0 x1) w)
      (indexIterations 3 27 4) [x0] world with
  | error failure => rfl
  | ok result => exact suffix_same23 primitive external x0 x1 result.1 result.2 (loopEval_arity 1 _ _ _ _ run)
theorem segment_actual (world : World) :
    evalProgram (M primitive external) S3 [x0,x1] world = evalProgram (M primitive external) S4 [x0,x1] world := by
  have r2 : indexIterations 3 27 4 = (scfRange 3 27 4).map Int.ofNat := naturalRange 3 27 4 (by decide)
  have r6 : indexIterations 3 27 12 = (scfRange 3 27 12).map Int.ofNat := naturalRange 3 27 12 (by decide)
  have loops : loopEval 1 (fun i c w => evalProgram (M primitive external) B3 (i::c++E3 primitive x0 x1) w)
      (indexIterations 3 27 4) [x0] world =
    loopEval 1 (fun i c w => evalProgram (M primitive external) B4 (i::c++E4 primitive x0 x1) w)
      (indexIterations 3 27 12) [x0] world := by
    rw [r2,r6]
    exact segment_loop 1 _ _ 3 27 4 3 (by decide) (by decide) (outer_same primitive external x0 x1) [x0] world
  rw [form3,form4,loops]
  cases run : loopEval 1 (fun i c w => evalProgram (M primitive external) B4 (i::c++E4 primitive x0 x1) w)
      (indexIterations 3 27 12) [x0] world with
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
end F2
def originalEntries : List Entry := [{inputs := F2.inputs, outputs := F2.outputs, body := F2.S0}]
def finalEntries : List Entry := [{inputs := F2.inputs, outputs := F2.outputs, body := F2.S4}]
theorem originalEntriesChecked : checkEntries registry originalEntries = true := by decide
theorem finalEntriesChecked : checkEntries registry finalEntries = true := by decide
theorem complete {World Error : Type} (primitive : Prim → List Int → Int)
    (external : External World Error registry) : PreservesEntries (M primitive external) originalEntries finalEntries :=
  entries_cons (M primitive external) rfl rfl (F2.actual primitive external) (entries_nil (M primitive external))
end Merlin.GraphStatePipelineInstance
