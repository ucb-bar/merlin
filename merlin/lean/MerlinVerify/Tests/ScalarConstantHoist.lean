/- Leading closed constants move only across independently checked body and outer scopes. -/
import MerlinVerify.Check.ScalarConstantHoist
import MerlinVerify.Tests.Semantics

namespace MerlinVerify.Tests.ScalarConstantHoist

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops
open Check.ScalarConstantHoist
open MerlinVerify.Tests.Semantics

def tensorTy (shape : List Nat) (ty : Ty := .int 13) : Ty := .tensor (shape.map some) ty none
def Γ (extent : Nat := 3) (reduction : Bool := false) : TEnv :=
  [(0,.tensor [extent] (.int 13)), (1,.tensor (if reduction then [] else [extent]) (.int 13)),
    (2,.scalar (.int 32))]
def constant (value : Int := 5) (id : Nat := 6) (ty : Ty := .int 13) : Op :=
  .mk "arith.constant" [⟨id,ty⟩] [] [("value",.int value ty)] [] [] []
def ret : Op := .mk "func.return" [] [3,2] [] [] [] []
def body : List Op :=
  [.mk "arith.cmpi" [⟨7,.int 1⟩] [6,4] [("predicate",.int 2 (.int 64))] [] [] [],
   .mk "arith.extui" [⟨8,.int 13⟩] [7] [] [] [] [],
   .mk "arith.xori" [⟨9,.int 13⟩] [8,5] [] [] [] [],
   .mk "linalg.yield" [] [9] [] [] [] []]
def generic (extent : Nat := 3) (reduction : Bool := false) (leading : Op := constant)
    (ops : List Op := body) : Op :=
  .mk "linalg.generic" [⟨3,tensorTy (if reduction then [] else [extent])⟩] [0,1]
    [("indexing_maps",.array [amap 1 [0],amap 1 (if reduction then [] else [0])]),
     ("iterator_types",.array [iter (if reduction then "reduction" else "parallel")]),
     ("operandSegmentSizes",.denseArray (.int 32) [1,1])] []
    [.mk [.mk [⟨4,.int 13⟩,⟨5,.int 13⟩] (leading :: ops)]] []
def source (extent : Nat := 3) (reduction : Bool := false) : List Op := [generic extent reduction,ret]
def normal (extent : Nat := 3) (reduction : Bool := false) := normalize 64 (Γ extent reduction) (source extent reduction)

#guard (normal).map (fun ops => ops.map (·.name)) == some ["arith.constant","linalg.generic","func.return"]
#guard (normal).map (fun ops => (ops.headD ret).results.map (·.id)) == some [6]
#guard (normal).map (fun ops => (ops.getD 1 ret).results.map (·.id)) == some [3]
#guard [0,1,3].all fun n => [false,true].all fun reduction => (normal n reduction).isSome
#guard (checked 32 (Γ) generic [ret]).isSome
#guard (checked 64 (Γ) generic [ret]).isSome

-- Unknown, malformed, nonclosed, non-scalar and index-valued constants cannot justify motion.
def withConstAttrs (attrs : List (String × Attr)) : Op :=
  .mk "arith.constant" [⟨6,.int 13⟩] [] attrs [] [] []
#guard (checked 64 (Γ) (generic 3 false (withConstAttrs [("value",.int 5 (.int 13)),("unknown",.unit)])) [ret]).isNone
#guard (checked 64 (Γ) (generic 3 false (withConstAttrs [("value",.opaque "unsupported")])) [ret]).isNone
#guard (checked 64 (Γ) (generic 3 false (constant 5 6 .index)) [ret]).isNone
#guard (checked 64 (Γ) (generic 3 false (.mk "arith.constant" [⟨6,.int 13⟩] [4]
  [("value",.int 5 (.int 13))] [] [] [])) [ret]).isNone
#guard (checked 64 (Γ) (generic 3 false (.mk "arith.constant" [⟨6,tensorTy []⟩] []
  [("value",.dense (tensorTy []) true [5,0])] [] [] [])) [ret]).isNone

-- A newly outer-visible binding must not shadow inputs, arguments or results, or escape in
-- the enclosing continuation, including captures of a later nested operation.
#guard (checked 64 ((6,.scalar (.int 13)) :: Γ) generic [ret]).isNone
#guard (checked 64 (Γ) (generic 3 false (constant 5 4)) [ret]).isNone
#guard (checked 64 (Γ) (generic 3 false (constant 5 3)) [ret]).isNone
#guard (checked 64 (Γ) generic [.mk "func.return" [] [3,6,2] [] [] [] []]).isNone
#guard (checked 64 (Γ) generic
  [.mk "test.capture" [] [] [] [] [.mk [.mk [] [.mk "func.return" [] [6] [] [] [] []]]] [],ret]).isNone

-- Hoisting stays bounded at one actual leading constant; it does not claim arbitrary motion.
def second : Op := .mk "arith.constant" [⟨10,.int 13⟩] [] [("value",.int 7 (.int 13))] [] [] []
#guard (normalize 64 (Γ) [generic 3 false constant (second :: body),ret]).map
  (fun ops => ops.map (·.name)) == some ["arith.constant","linalg.generic","func.return"]
#guard (checked 64 (Γ) (generic 3 false second body) [ret]).isNone

-- A captured scalar in a nested body remains valid after its binding moves outward.
def inner : Op :=
  .mk "linalg.generic" [⟨13,tensorTy []⟩] [10]
    [("indexing_maps",.array [amap 0 []]),("iterator_types",.array []),
     ("operandSegmentSizes",.denseArray (.int 32) [0,1])] []
    [.mk [.mk [⟨14,.int 13⟩]
      [.mk "arith.xori" [⟨15,.int 13⟩] [6,14] [] [] [] [],.mk "linalg.yield" [] [15] [] [] [] []]]] []
def nested : List Op :=
  [.mk "tensor.splat" [⟨10,tensorTy []⟩] [4] [] [] [] [],inner,
   .mk "tensor.extract" [⟨16,.int 13⟩] [13] [] [] [] [],.mk "linalg.yield" [] [16] [] [] [] []]
#guard (checked 64 (Γ) (generic 3 false constant nested) [ret]).isSome

-- Poison payloads, reduction accumulators and zero-iteration loops retain exact outcomes.
def observe (result : Outcome (List RVal)) : String × Option (List (List (Option Int))) :=
  match result with
  | .ub => ("ub",none)
  | .stuck reason => (reason,none)
  | .ok values => ("ok",readInts (.ok values))
def run (extent : Nat) (reduction : Bool) (input : RVal) (ops : List Op) :=
  observe (evalOps opSem ctx (fun _ _ => .stuck "no calls")
    [(0,input),(1,intTensor (if reduction then [] else [extent]) 13 [7,11,13]),
     (2,.scalar (.some (.bits 32 37)))] ops)
#guard [false,true].all fun reduction =>
  run 3 reduction (.tensor (Tensor.full [3] (.int 13) .poison)) (source 3 reduction) ==
    run 3 reduction (.tensor (Tensor.full [3] (.int 13) .poison)) ((normal 3 reduction).getD [])
#guard [false,true].all fun reduction =>
  run 0 reduction (intTensor [0] 13 []) (source 0 reduction) ==
    run 0 reduction (intTensor [0] 13 []) ((normal 0 reduction).getD [])
#guard (List.range 256).all fun n => [false,true].all fun reduction =>
  let input := intTensor [3] 13 [n,n+1,n+2]
  run 3 reduction input (source 3 reduction) == run 3 reduction input ((normal 3 reduction).getD [])


-- Float constants are only decoded bit payloads; no arithmetic identity is assumed.
def floatConstant : Op := .mk "arith.constant" [⟨6,.float .f32⟩] []
  [("value",.float 2147483648 (.float .f32))] [] [] []
def floatGeneric : Op := .mk "linalg.generic" [⟨3,tensorTy [3] (.float .f32)⟩] [0,1]
  [("indexing_maps",.array [amap 1 [0],amap 1 [0]]),
   ("iterator_types",.array [iter "parallel"]),("operandSegmentSizes",.denseArray (.int 32) [1,1])] []
  [.mk [.mk [⟨4,.float .f32⟩,⟨5,.float .f32⟩]
    [floatConstant,.mk "arith.addf" [⟨7,.float .f32⟩] [4,6] [] [] [] [],
     .mk "linalg.yield" [] [7] [] [] [] []]]] []
#guard (checked 64 [(0,.tensor [3] (.float .f32)),(1,.tensor [3] (.float .f32)),(2,.scalar (.int 32))]
  floatGeneric [ret]).isSome

def division : List Op :=
  [.mk "arith.divui" [⟨7,.int 13⟩] [4,6] [] [] [] [],.mk "linalg.yield" [] [7] [] [] [] []]
#guard let ss := [generic 3 false (constant 0) division,ret]
  run 3 false (intTensor [3] 13 [1,2,3]) ss ==
    run 3 false (intTensor [3] 13 [1,2,3]) ((normalize 64 (Γ) ss).getD [])
#guard let ss := [generic 0 false (constant 0) division,ret]
  run 0 false (intTensor [0] 13 []) ss == run 0 false (intTensor [0] 13 []) ((normalize 64 (Γ 0) ss).getD [])

end MerlinVerify.Tests.ScalarConstantHoist
