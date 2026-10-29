import Merlin.GraphStatePipelineTailWitness
import Merlin.GraphStateIntegerOperations
import Merlin.GraphStatePipelineTail
import Merlin.GraphStatePipelineIndex
import Merlin.GraphStatePipelineLiterals
import Merlin.GraphStatePipelineScope
import Merlin.GraphStatePipelineLoops
import Merlin.GraphStatePipelineProgram
import Merlin.GraphStateScalar
set_option maxRecDepth 4096
set_option maxHeartbeats 2000000
namespace Merlin.GraphStatePipelineTailInstance
open Merlin.GraphState.Scalar Merlin.GraphState.Pipeline
def registry : List Signature := [{inputs := [.index,.index],outputs := [.index]},{inputs := [],outputs := [.index]},{inputs := [],outputs := [.index]},{inputs := [],outputs := [.index]},{inputs := [],outputs := [.index]},{inputs := [],outputs := [.index]},{inputs := [.index,(.bits 8)],outputs := [(.bits 8)]},{inputs := [(.bits 8)],outputs := [(.bits 8)]}]
def controls : Controls := [(some .add),(some (.constant (0))),(some (.constant (1))),(some (.constant (2))),(some (.constant (4))),(some (.constant (5))),none,none]
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
def S2 : Program := (.opaque 1 [] [.index] (.opaque 5 [] [.index] (.opaque 2 [] [.index] (.loop 2 1 0 [3] [(.bits 8)] (.opaque 6 [0,1] [(.bits 8)] (.ret [0])) (.opaque 7 [0] [(.bits 8)] (.ret [0]))))))
def S3 : Program := (.opaque 1 [] [.index] (.opaque 5 [] [.index] (.opaque 2 [] [.index] (.opaque 3 [] [.index] (.opaque 4 [] [.index] (.loop 4 0 1 [5] [(.bits 8)] (.opaque 6 [0,1] [(.bits 8)] (.opaque 2 [] [.index] (.opaque 0 [2,0] [.index] (.opaque 6 [0,2] [(.bits 8)] (.ret [0]))))) (.loop 1 4 3 [0] [(.bits 8)] (.opaque 6 [0,1] [(.bits 8)] (.ret [0])) (.opaque 7 [0] [(.bits 8)] (.ret [0])))))))))
theorem typed2 : checkTypes registry inputs outputs S2 = true := by decide
theorem typed3 : checkTypes registry inputs outputs S3 = true := by decide
def E2 (primitive : Prim → List Int → Int) (x0 x1 : Int) : List Int := [(1 : Int),(5 : Int),(0 : Int),x0,x1]
def B2 : Program := (.opaque 6 [0,1] [(.bits 8)] (.ret [0]))
def E3 (primitive : Prim → List Int → Int) (x0 x1 : Int) : List Int := [(4 : Int),(2 : Int),(1 : Int),(5 : Int),(0 : Int),x0,x1]
def B3 : Program := (.opaque 6 [0,1] [(.bits 8)] (.opaque 2 [] [.index] (.opaque 0 [2,0] [.index] (.opaque 6 [0,2] [(.bits 8)] (.ret [0])))))
def BT : Program := (.opaque 6 [0,1] [(.bits 8)] (.ret [0]))
def N2 : Program := (.opaque 7 [0] [(.bits 8)] (.ret [0]))
def N3 : Program := (.loop 1 4 3 [0] [(.bits 8)] (.opaque 6 [0,1] [(.bits 8)] (.ret [0])) (.opaque 7 [0] [(.bits 8)] (.ret [0])))
def NT : Program := (.opaque 7 [0] [(.bits 8)] (.ret [0]))
variable {World Error : Type} (primitive : Prim → List Int → Int)
    (external : External World Error registry) (x0 x1 : Int)
theorem form2 (world : World) : evalProgram (M primitive external) S2 [x0,x1] world =
  (loopEval 1 (fun i c w => evalProgram (M primitive external) B2 (i::c++E2 primitive x0 x1) w)
    (indexIterations 0 5 1) [x0] world).bind
    (fun r => evalProgram (M primitive external) N2 (r.1++E2 primitive x0 x1) r.2) := by rfl
theorem form3 (world : World) : evalProgram (M primitive external) S3 [x0,x1] world =
  (loopEval 1 (fun i c w => evalProgram (M primitive external) B3 (i::c++E3 primitive x0 x1) w)
    (indexIterations 0 4 2) [x0] world).bind
    (fun r => evalProgram (M primitive external) N3 (r.1++E3 primitive x0 x1) r.2) := by rfl
theorem body_group (base : Int) (carry : List Int) (world : World) (arity : carry.length = 1) :
  evalProgram (M primitive external) B3 (base::carry++E3 primitive x0 x1) world =
  loopEval 1 (fun i c w => evalProgram (M primitive external) B2 (i::c++E2 primitive x0 x1) w)
    [base,(base+1)] carry world := by
  obtain ⟨c0,rfl⟩ := List.length_eq_one_iff.mp arity
  simp only [B2,B3,E2,E3,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
  cases step0 : external.run 6 [base,c0] world with
  | error failure =>
    rcases failure with ⟨error,terminal⟩
    simp only [step0,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
  | ok result =>
    rcases result with ⟨values,w0⟩
    have size : values.length = 1 := by
      have localArity := external.arity 6 [base,c0] world values w0 step0
      simpa only [registry,signature,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some,List.length_cons,List.length_nil] using localArity
    obtain ⟨v0_0,rfl⟩ := List.length_eq_one_iff.mp size
    simp only [step0,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
    cases step1 : external.run 6 [(base+1),v0_0] w0 with
    | error failure =>
      rcases failure with ⟨error,terminal⟩
      simp only [step1,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
    | ok result =>
      rcases result with ⟨values,w1⟩
      have size : values.length = 1 := by
        have localArity := external.arity 6 [(base+1),v0_0] w0 values w1 step1
        simpa only [registry,signature,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some,List.length_cons,List.length_nil] using localArity
      obtain ⟨v1_0,rfl⟩ := List.length_eq_one_iff.mp size
      simp only [step1,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte,loopEval]
def tailScopeL : List Expr := [(.var 0),(.var 1),(.var 5),(.var 6),(.var 7),(.var 8),(.var 9)]
def tailScopeR : List Expr := [(.var 0),(.var 1),(.var 2),(.var 3),(.var 4),(.var 5),(.var 6),(.var 7),(.var 8),(.var 9)]
theorem tailScopeChecked : Scope.check registry tailScopeL tailScopeR B2 BT = true := by decide
theorem tail_body (iv : Int) (previous carry : List Int) (world : World) (previousSize : previous.length = 1) (arity : carry.length = 1) :
  evalProgram (M primitive external) B2 (iv::carry++E2 primitive x0 x1) world =
  evalProgram (M primitive external) BT (iv::carry++(previous++E3 primitive x0 x1)) world := by
  obtain ⟨p0,rfl⟩ := List.length_eq_one_iff.mp previousSize
  obtain ⟨c0,rfl⟩ := List.length_eq_one_iff.mp arity
  have checked := Scope.check_sound tailScopeChecked (M primitive external) [iv,c0,p0,(4 : Int),(2 : Int),(1 : Int),(5 : Int),(0 : Int),x0,x1] world
  simpa only [tailScopeL,tailScopeR,E2,E3,evalExpr,List.cons_append,List.nil_append,List.map_cons,List.map_nil,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some] using checked
def suffixScopeL : List Expr := [(.var 0),(.var 4),(.var 5),(.var 6),(.var 7),(.var 8)]
def suffixScopeR : List Expr := [(.var 0),(.var 1),(.var 2),(.var 3),(.var 4),(.var 5),(.var 6),(.var 7),(.var 8)]
theorem suffixScopeChecked : Scope.check registry suffixScopeL suffixScopeR N2 NT = true := by decide
theorem suffix_same (previous carry : List Int) (world : World) (previousSize : previous.length = 1) (arity : carry.length = 1) :
  evalProgram (M primitive external) N2 (carry++E2 primitive x0 x1) world =
  evalProgram (M primitive external) NT (carry++(previous++E3 primitive x0 x1)) world := by
  obtain ⟨p0,rfl⟩ := List.length_eq_one_iff.mp previousSize
  obtain ⟨c0,rfl⟩ := List.length_eq_one_iff.mp arity
  have checked := Scope.check_sound suffixScopeChecked (M primitive external) [c0,p0,(4 : Int),(2 : Int),(1 : Int),(5 : Int),(0 : Int),x0,x1] world
  simpa only [suffixScopeL,suffixScopeR,E2,E3,evalExpr,List.cons_append,List.nil_append,List.map_cons,List.map_nil,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some] using checked
theorem after_bulk (previous : List Int) (world : World) (previousSize : previous.length = 1) :
  evalProgram (M primitive external) N3 (previous++E3 primitive x0 x1) world =
  (loopEval 1 (fun i c w => evalProgram (M primitive external) BT (i::c++(previous++E3 primitive x0 x1)) w)
    (indexIterations 4 5 1) previous world).bind
    (fun r => evalProgram (M primitive external) NT (r.1++(previous++E3 primitive x0 x1)) r.2) := by
  obtain ⟨p0,rfl⟩ := List.length_eq_one_iff.mp previousSize
  rfl
def bounds : Merlin.GraphLoopsUnroll.Bounds := {bits := 32,lower := 0,upper := 5,step := 1,factor := 2,count := 5,groups := 2,remainder := 1}
theorem boundsChecked : bounds.Valid := by decide
theorem constantsFit :
    Merlin.GraphLoopsSegment.FitsSigned bounds.bits bounds.lower ∧
    Merlin.GraphLoopsSegment.FitsSigned bounds.bits bounds.upper ∧
    Merlin.GraphLoopsSegment.FitsSigned bounds.bits bounds.step ∧
    Merlin.GraphLoopsSegment.FitsSigned bounds.bits (bounds.step*bounds.factor) ∧
    Merlin.GraphLoopsSegment.FitsSigned bounds.bits (bounds.lower+bounds.step*(bounds.groups*bounds.factor)) :=
  Merlin.GraphLoopsUnroll.Bounds.constants boundsChecked
theorem lanesFit (group lane : Nat) (inGroup : group < bounds.groups) (inLane : lane < bounds.factor) :
    bounds.lower+(bounds.step*bounds.factor)*group+bounds.step*lane =
      bounds.lower+bounds.step*(group*bounds.factor+lane) ∧
    Merlin.GraphLoopsSegment.FitsSigned bounds.bits (bounds.step*lane) ∧
    Merlin.GraphLoopsSegment.FitsSigned bounds.bits (bounds.lower+(bounds.step*bounds.factor)*group+bounds.step*lane) :=
  Merlin.GraphLoopsUnroll.Bounds.lane_control boundsChecked inGroup inLane
theorem tailIndicesFit (ordinal : Nat) (inside : ordinal < bounds.remainder) :
    Merlin.GraphLoopsSegment.FitsSigned bounds.bits
      (bounds.lower+bounds.step*(bounds.groups*bounds.factor+ordinal)) ∧
    Merlin.GraphLoopsSegment.FitsSigned bounds.bits
      (bounds.lower+bounds.step*(bounds.groups*bounds.factor+ordinal)+bounds.step) := by
  apply Merlin.GraphLoopsUnroll.Bounds.inline_control boundsChecked
  have partition := boundsChecked.2.2.2.2.2.1
  omega
theorem actual_inputs (world : World) : evalProgram (M primitive external) S2 [x0,x1] world = evalProgram (M primitive external) S3 [x0,x1] world := by
  have bodies : ∀ index values w, values.length = 1 →
      evalProgram (M primitive external) B3 (Int.ofNat index::values++E3 primitive x0 x1) w =
      loopEval 1 (fun i c w => evalProgram (M primitive external) B2 (i::c++E2 primitive x0 x1) w) ((progression index 1 2).map Int.ofNat) values w := by
    intro index values w size
    simpa [progression,Int.ofNat_eq_natCast,Int.add_assoc] using body_group primitive external x0 x1 (Int.ofNat index) values w size
  have loops := Tail.ordered_bulk_tail 1 (fun i c w => evalProgram (M primitive external) B2 (i::c++E2 primitive x0 x1) w) (fun i c w => evalProgram (M primitive external) B3 (i::c++E3 primitive x0 x1) w) 0 5 1 2 2 1
    (by decide) (by decide) (by decide) bodies [x0] world
  rw [form2,form3,loops]
  cases bulkRun : loopEval 1 (fun i c w => evalProgram (M primitive external) B3 (i::c++E3 primitive x0 x1) w) (indexIterations 0 4 2) [x0] world with
  | error failure => rfl
  | ok result =>
    rcases result with ⟨previous,afterBulk⟩
    have previousSize := loopEval_arity 1 _ _ _ _ bulkRun
    simp only [Except.bind]
    rw [after_bulk primitive external x0 x1 previous afterBulk previousSize]
    have tails := loopEval_congr 1 (fun i c w => evalProgram (M primitive external) B2 (i::c++E2 primitive x0 x1) w) (fun i c w => evalProgram (M primitive external) BT (i::c++(previous++E3 primitive x0 x1)) w)
      (fun i c w size => tail_body primitive external x0 x1 i previous c w previousSize size)
      (indexIterations 4 5 1) previous afterBulk
    rw [tails]
    cases tailRun : loopEval 1 (fun i c w => evalProgram (M primitive external) BT (i::c++(previous++E3 primitive x0 x1)) w) (indexIterations 4 5 1) previous afterBulk with
    | error failure => rfl
    | ok result =>
      exact suffix_same primitive external x0 x1 previous result.1 result.2 previousSize (loopEval_arity 1 _ _ _ _ tailRun)
theorem actual (arguments : List Int) (world : World) (arity : arguments.length = inputs.length) :
    evalProgram (M primitive external) S2 arguments world = evalProgram (M primitive external) S3 arguments world := by
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
def S2 : Program := (.ret [0])
def S3 : Program := (.ret [0])
theorem typed2 : checkTypes registry inputs outputs S2 = true := by decide
theorem typed3 : checkTypes registry inputs outputs S3 = true := by decide
theorem actual {World Error : Type} (primitive : Prim → List Int → Int) (external : External World Error registry)
    (arguments : List Int) (world : World) (_arity : arguments.length = inputs.length) :
    evalProgram (M primitive external) S2 arguments world = evalProgram (M primitive external) S3 arguments world := rfl
end F4
def originalEntries : List Entry := [{inputs := F3.inputs,outputs := F3.outputs,body := F3.S2},{inputs := F4.inputs,outputs := F4.outputs,body := F4.S2}]
theorem originalEntriesChecked : checkEntries registry originalEntries = true := by decide
def finalEntries : List Entry := [{inputs := F3.inputs,outputs := F3.outputs,body := F3.S3},{inputs := F4.inputs,outputs := F4.outputs,body := F4.S3}]
theorem finalEntriesChecked : checkEntries registry finalEntries = true := by decide
theorem complete {World Error : Type} (primitive : Prim → List Int → Int) (external : External World Error registry) :
    PreservesEntries (M primitive external) originalEntries finalEntries :=
  entries_cons (M primitive external) rfl rfl (F3.actual primitive external) (entries_cons (M primitive external) rfl rfl (F4.actual primitive external) (entries_nil (M primitive external)))
end Merlin.GraphStatePipelineTailInstance
theorem merlin_unroll_tail_transform {World Error : Type}
    (primitive : Merlin.GraphState.Scalar.Prim → List Int → Int)
    (external : Merlin.GraphState.Pipeline.External World Error Merlin.GraphStatePipelineTailInstance.registry) :
    Merlin.GraphState.Pipeline.PreservesEntries (Merlin.GraphStatePipelineTailInstance.M primitive external)
      Merlin.GraphStatePipelineTailInstance.originalEntries Merlin.GraphStatePipelineTailInstance.finalEntries :=
  Merlin.GraphStatePipelineTailInstance.complete primitive external
namespace Merlin.GraphUnrollTailWitness
open Merlin.GraphState.Scalar Merlin.GraphState.Pipeline
open Merlin.GraphStatePipelineTailInstance
abbrev wordPrimitive := Merlin.GraphState.Integer.primitive
def chosen (failAt : Option Int) : External Nat Int registry where
  run := fun identity arguments world =>
    if identity = 6 ∧ failAt = some (arguments[0]?.getD 0) then .error (17,world+1)
    else .ok ((signature registry identity).outputs.map
      (fun _ => if identity = 6 then arguments[1]?.getD 0 else arguments.headD 0),world+1)
  arity := by
    intro identity arguments world values final success
    split at success
    · cases success
    · cases success; simp only [List.length_map]
def body (failAt : Option Int) (i : Int) (values : List Int) (world : Nat) : Outcome Int Nat Int :=
  evalProgram (M wordPrimitive (chosen failAt)) Merlin.GraphStatePipelineTailInstance.F3.B2
    (i::values++Merlin.GraphStatePipelineTailInstance.F3.E2 wordPrimitive 3 5) world
theorem body_actual (failAt : Option Int) (i : Int) (world : Nat) :
    body failAt i [3] world =
      if failAt = some i then .error (.external 17,world+1) else .ok ([3],world+1) := by
  simp [body,Merlin.GraphStatePipelineTailInstance.F3.B2,Merlin.GraphStatePipelineTailInstance.F3.E2,M,indexModel,indexEffect,controls,controlAt,
    chosen,registry,signature,opaqueStep,evalProgram,reads]
theorem source_success : evalProgram (M wordPrimitive (chosen none)) Merlin.GraphStatePipelineTailInstance.F3.S2 [3,5] 7 = .ok ([3],13) := by
  have run := Tail.constant_carry_run 1 [3] (body none) rfl
    ((progression 0 1 5).map Int.ofNat)
    (fun i w _ => by simp only [body_actual,reduceCtorEq,↓reduceIte]) 7
  have counted : indexIterations 0 5 1 = (progression 0 1 5).map Int.ofNat :=
    indexIterations_count (by decide) (by decide)
  rw [Merlin.GraphStatePipelineTailInstance.F3.form2,counted]
  change (loopEval 1 (body none) ((progression 0 1 5).map Int.ofNat) [3] 7).bind _ = _
  rw [run,List.length_map,Tail.progression_length]
  rfl
theorem source_bulk_failure :
    evalProgram (M wordPrimitive (chosen (some 1))) Merlin.GraphStatePipelineTailInstance.F3.S2 [3,5] 7 =
      .error (.external 17,9) := by
  have prefix : ∀ i w, i ∈ ((progression 0 1 1).map Int.ofNat) →
      body (some 1) i [3] w = .ok ([3],w+1) := by
    intro i w member
    obtain ⟨n,inside,rfl⟩ := List.mem_map.mp member
    obtain ⟨k,bound,value⟩ := Tail.progression_member inside
    have different : (some (1 : Int)) ≠ some (Int.ofNat n) := by
      intro same
      have equal := Option.some.inj same
      omega
    simp only [body_actual,different,↓reduceIte]
  have failed := Tail.fault_after_prefix 1 [3] (body (some 1)) rfl
    ((progression 0 1 1).map Int.ofNat)
    ((progression 2 1 3).map Int.ofNat) 1 prefix (.external 17)
    (fun w => by simp only [body_actual,↓reduceIte]) 7
  have split : indexIterations 0 5 1 =
      (progression 0 1 1).map Int.ofNat ++
        (1 : Int)::(progression 2 1 3).map Int.ofNat := by
    rw [indexIterations_count (show 0 < (1 : Nat) by decide)
      (show Merlin.GraphLoopsUnroll.CountCorrect 0 5 1 (1+4) by decide),
      Tail.progression_append,List.map_append]
    rfl
  rw [Merlin.GraphStatePipelineTailInstance.F3.form2,split]
  change (loopEval 1 (body (some 1)) _ [3] 7).bind _ = _
  rw [failed,List.length_map,Tail.progression_length]
  rfl
theorem source_tail_failure :
    evalProgram (M wordPrimitive (chosen (some 4))) Merlin.GraphStatePipelineTailInstance.F3.S2 [3,5] 7 =
      .error (.external 17,12) := by
  have prefix : ∀ i w, i ∈ ((progression 0 1 4).map Int.ofNat) →
      body (some 4) i [3] w = .ok ([3],w+1) := by
    intro i w member
    obtain ⟨n,inside,rfl⟩ := List.mem_map.mp member
    obtain ⟨k,bound,value⟩ := Tail.progression_member inside
    have different : (some (4 : Int)) ≠ some (Int.ofNat n) := by
      intro same
      have equal := Option.some.inj same
      omega
    simp only [body_actual,different,↓reduceIte]
  have failed := Tail.fault_after_prefix 1 [3] (body (some 4)) rfl
    ((progression 0 1 4).map Int.ofNat)
    ((progression 5 1 0).map Int.ofNat) 4 prefix (.external 17)
    (fun w => by simp only [body_actual,↓reduceIte]) 7
  have split : indexIterations 0 5 1 =
      (progression 0 1 4).map Int.ofNat ++
        (4 : Int)::(progression 5 1 0).map Int.ofNat := by
    rw [indexIterations_count (show 0 < (1 : Nat) by decide)
      (show Merlin.GraphLoopsUnroll.CountCorrect 0 5 1 (4+1) by decide),
      Tail.progression_append,List.map_append]
    rfl
  rw [Merlin.GraphStatePipelineTailInstance.F3.form2,split]
  change (loopEval 1 (body (some 4)) _ [3] 7).bind _ = _
  rw [failed,List.length_map,Tail.progression_length]
  rfl
theorem target_success : evalProgram (M wordPrimitive (chosen none)) Merlin.GraphStatePipelineTailInstance.F3.S3 [3,5] 7 = .ok ([3],13) := by
  rw [← Merlin.GraphStatePipelineTailInstance.F3.actual wordPrimitive (chosen none) [3,5] 7 rfl]
  exact source_success
theorem source_success_initialized :
    Initialized (M wordPrimitive (chosen none)) Merlin.GraphStatePipelineTailInstance.F3.S2 [3,5] 7 :=
  (check_operands_initialized (show check registry Merlin.GraphStatePipelineTailInstance.F3.inputs Merlin.GraphStatePipelineTailInstance.F3.outputs Merlin.GraphStatePipelineTailInstance.F3.S2 Merlin.GraphStatePipelineTailInstance.F3.S2 = true by decide)
    (M wordPrimitive (chosen none)) [3,5] 7 rfl).1
theorem target_success_initialized :
    Initialized (M wordPrimitive (chosen none)) Merlin.GraphStatePipelineTailInstance.F3.S3 [3,5] 7 :=
  (check_operands_initialized (show check registry Merlin.GraphStatePipelineTailInstance.F3.inputs Merlin.GraphStatePipelineTailInstance.F3.outputs Merlin.GraphStatePipelineTailInstance.F3.S3 Merlin.GraphStatePipelineTailInstance.F3.S3 = true by decide)
    (M wordPrimitive (chosen none)) [3,5] 7 rfl).1
theorem target_bulk_failure : evalProgram (M wordPrimitive (chosen (some 1))) Merlin.GraphStatePipelineTailInstance.F3.S3 [3,5] 7 = .error (.external 17,9) := by
  rw [← Merlin.GraphStatePipelineTailInstance.F3.actual wordPrimitive (chosen (some 1)) [3,5] 7 rfl]
  exact source_bulk_failure
theorem source_bulk_failure_initialized :
    Initialized (M wordPrimitive (chosen (some 1))) Merlin.GraphStatePipelineTailInstance.F3.S2 [3,5] 7 :=
  (check_operands_initialized (show check registry Merlin.GraphStatePipelineTailInstance.F3.inputs Merlin.GraphStatePipelineTailInstance.F3.outputs Merlin.GraphStatePipelineTailInstance.F3.S2 Merlin.GraphStatePipelineTailInstance.F3.S2 = true by decide)
    (M wordPrimitive (chosen (some 1))) [3,5] 7 rfl).1
theorem target_bulk_failure_initialized :
    Initialized (M wordPrimitive (chosen (some 1))) Merlin.GraphStatePipelineTailInstance.F3.S3 [3,5] 7 :=
  (check_operands_initialized (show check registry Merlin.GraphStatePipelineTailInstance.F3.inputs Merlin.GraphStatePipelineTailInstance.F3.outputs Merlin.GraphStatePipelineTailInstance.F3.S3 Merlin.GraphStatePipelineTailInstance.F3.S3 = true by decide)
    (M wordPrimitive (chosen (some 1))) [3,5] 7 rfl).1
theorem target_tail_failure : evalProgram (M wordPrimitive (chosen (some 4))) Merlin.GraphStatePipelineTailInstance.F3.S3 [3,5] 7 = .error (.external 17,12) := by
  rw [← Merlin.GraphStatePipelineTailInstance.F3.actual wordPrimitive (chosen (some 4)) [3,5] 7 rfl]
  exact source_tail_failure
theorem source_tail_failure_initialized :
    Initialized (M wordPrimitive (chosen (some 4))) Merlin.GraphStatePipelineTailInstance.F3.S2 [3,5] 7 :=
  (check_operands_initialized (show check registry Merlin.GraphStatePipelineTailInstance.F3.inputs Merlin.GraphStatePipelineTailInstance.F3.outputs Merlin.GraphStatePipelineTailInstance.F3.S2 Merlin.GraphStatePipelineTailInstance.F3.S2 = true by decide)
    (M wordPrimitive (chosen (some 4))) [3,5] 7 rfl).1
theorem target_tail_failure_initialized :
    Initialized (M wordPrimitive (chosen (some 4))) Merlin.GraphStatePipelineTailInstance.F3.S3 [3,5] 7 :=
  (check_operands_initialized (show check registry Merlin.GraphStatePipelineTailInstance.F3.inputs Merlin.GraphStatePipelineTailInstance.F3.outputs Merlin.GraphStatePipelineTailInstance.F3.S3 Merlin.GraphStatePipelineTailInstance.F3.S3 = true by decide)
    (M wordPrimitive (chosen (some 4))) [3,5] 7 rfl).1
end Merlin.GraphUnrollTailWitness
