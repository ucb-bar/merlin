/- Closed coordinates, precise point dependencies, and total deletion are independently checked. -/
import MerlinVerify.Check.InsertExtract
import MerlinVerify.Check.DefinedInsert
import MerlinVerify.Tests.Semantics

namespace MerlinVerify.Tests.InsertExtract

open MerlinVerify MerlinVerify.Check
open MerlinVerify.Check.InsertExtract
open MerlinVerify.Tests.Semantics

def tensorTy (shape : List Nat) : Ty := .tensor (shape.map some) (.int 13) none
def Γ (shape : List Nat) : TEnv := [(0, .scalar (.int 13)), (1, .tensor shape (.int 13))]
def constant (id : Nat) (n : Int) : Op :=
  .mk "arith.constant" [⟨id, .index⟩] [] [("value", .int n .index)] [] [] []
def write (shape : List Nat) (indices : List Nat) : Op :=
  .mk "tensor.insert" [⟨3, tensorTy shape⟩] (0 :: 1 :: indices) [] [] [] []
def read (indices : List Nat) : Op := .mk "tensor.extract" [⟨4, .int 13⟩] (3 :: indices) [] [] [] []
def ret (ids : List Nat) : Op := .mk "func.return" [] ids [] [] [] []
def source (escape : Bool) : List Op :=
  [constant 2 1, write [3] [2], read [2], ret (if escape then [4, 3] else [4])]
def normalized (ib : Nat) (ops : List Op) := InsertExtract.normalize ib (Γ [3]) ⟨[], []⟩ ops

#guard (normalized 64 (source false)).map (fun ops => ops.map (·.name)) == some ["func.return"]
#guard (normalized 32 (source false)).map (fun ops => ops.map (·.operands)) == some [[0]]
#guard (normalized 64 (source true)).map (fun ops => ops.map (·.name)) ==
  some ["arith.constant", "tensor.insert", "func.return"]
#guard ((normalized 64 (source true)).bind (·.getLast?)).map (·.operands) == some [0, 3]
#guard (InsertExtract.normalize 64 (Γ []) ⟨[], []⟩ [write [] [], read [], ret [4]]).map
  (fun ops => ops.map (·.operands)) == some [[0]]

-- Negative, out-of-bounds, rank-mismatched, empty-dimension and unknown indices never justify deletion.
#guard [(-1 : Int), 3, 4294967295].all fun n =>
  (normalized 64 [constant 2 n, write [3] [2], read [2], ret [4]]).map List.length == some 4
#guard (normalized 64 [write [3] [], read [], ret [4]]).map List.length == some 3
#guard (InsertExtract.normalize 64 (Γ [0]) ⟨[], []⟩
  [constant 2 0, write [0] [2], read [2], ret [4]]).map List.length == some 4
#guard (InsertExtract.normalize 64 ((2, .scalar .index) :: Γ [3]) ⟨[], []⟩
  [write [3] [2], read [2], ret [4]]).map List.length == some 3

-- Actual claim width controls signed decoding; an apparent large literal can wrap to coordinate one.
#guard (normalized 32 [constant 2 4294967297, write [3] [2], read [2], ret [4]]).map
  (fun ops => ops.map (·.operands)) == some [[0]]
#guard (normalized 64 [constant 2 4294967297, write [3] [2], read [2], ret [4]]).map List.length == some 4

-- Scalar/tensor writes invalidate exactly their current point relation.
def scalarOverwrite : Op := .mk "arith.constant" [⟨0, .int 13⟩] [] [("value", .int 7 (.int 13))] [] [] []
def tensorOverwrite : Op := .mk "tensor.empty" [⟨3, tensorTy [3]⟩] [] [] [] [] []
#guard (normalized 64 [constant 2 1, write [3] [2], scalarOverwrite, read [2], ret [4]]).map
  (fun ops => (ops.filter (fun s => s.name == "tensor.extract")).length) == some 1
#guard (normalized 64 [constant 2 1, write [3] [2], tensorOverwrite, read [2], ret [4]]).map
  (fun ops => (ops.filter (fun s => s.name == "tensor.extract")).length) == some 1
#guard (normalized 64 [constant 2 1, write [3] [2], constant 2 2, read [2], ret [4]]).map
  (fun ops => (ops.filter (fun s => s.name == "tensor.extract")).length) == some 1
#guard (normalized 64 [constant 2 1, write [3] [2], constant 2 1, read [2], ret [4]]).map
  (fun ops => ops.map (·.operands)) == some [[0]]

-- Exact payload preservation includes poison and arbitrary source tensor contents.
def run (ops : List Op) (payload : Val) :=
  Semantics.readInts (evalOps Sem.Ops.opSem Semantics.ctx (fun _ _ => .stuck "no calls")
    [(0, .scalar payload), (1, .tensor (Tensor.full [3] (.int 13) .poison))] ops)
#guard run (source false) .poison == run ((normalized 64 (source false)).getD []) .poison
#guard (List.range 16).all fun n =>
  let payload := Val.some (.bits 13 (BitVec.ofNat 13 n))
  run (source false) payload == run ((normalized 64 (source false)).getD []) payload

-- Late aliases expose dead splats; poison payloads remain exact, while escapes and captures retain them.
def fresh : Op := .mk "tensor.splat" [⟨5, tensorTy []⟩] [0] [] [] [] []
def localWrite : Op := .mk "tensor.insert" [⟨3, tensorTy []⟩] [0, 5] [] [] [] []
#guard (InsertExtract.normalize 64 (Γ []) ⟨[], []⟩ [fresh, localWrite, read [], ret [4]]).map
  (fun ops => ops.map (·.name)) == some ["func.return"]
#guard (InsertExtract.normalize 64 (Γ []) ⟨[], []⟩ [fresh, localWrite, read [], ret [4, 5]]).map
  (fun ops => ops.map (·.name)) == some ["tensor.splat", "func.return"]
def captured : Op := .mk "test.region" [] [] [] [] [.mk [.mk [] [ret [5]]]] []
#guard !(drop? (Γ []) ⟨[], []⟩ fresh [captured, ret [0]])
#guard !(drop? [] ⟨[], []⟩ fresh [ret [0]])
def unknownFresh : Op := .mk "tensor.splat" [⟨5, tensorTy []⟩] [0] [("unreviewed.meaning", .unit)] [] [] []
#guard !(drop? (Γ []) ⟨[], []⟩ unknownFresh [ret [0]])
def splatRun (ops : List Op) (payload : Val) :=
  Semantics.readInts (evalOps Sem.Ops.opSem Semantics.ctx (fun _ _ => .stuck "no calls")
    [(0, .scalar payload), (1, .tensor (Tensor.full [] (.int 13) .poison))] ops)
#guard splatRun [fresh, localWrite, read [], ret [4]] .poison ==
  splatRun ((InsertExtract.normalize 64 (Γ []) ⟨[], []⟩ [fresh, localWrite, read [], ret [4]]).getD []) .poison

open MerlinVerify.Check.Definedness
open MerlinVerify.Check.Definedness.IndexRanges

def bounds (low high : Nat) : IndexRanges.Fact := ⟨⟨.scalar .index, true⟩, some ⟨low, high⟩⟩
def inputFacts (shape : List Nat) (clean : Bool) : List IndexRanges.Fact :=
  [⟨⟨.scalar (.int 13), true⟩, none⟩, ⟨⟨.tensor shape (.int 13), clean⟩, none⟩]
#guard (Definedness.Insert.rule [] (write [3] [2]) (inputFacts [3] true ++ [bounds 1 2])).isSome
#guard (Definedness.Insert.rule [] (write [3] [2]) (inputFacts [3] false ++ [bounds 1 2])).isNone
#guard (Definedness.Insert.rule [] (write [] []) (inputFacts [] false)).isSome
#guard (Definedness.Insert.rule [] (write [0] [2]) (inputFacts [0] true ++ [bounds 0 1])).isNone
#guard (Definedness.Insert.rule [] (write [3] [2]) (inputFacts [3] true ++ [bounds 1 4])).isNone
#guard (Definedness.Insert.rule [] (write [3] [2]) (inputFacts [3] true ++ [⟨⟨.scalar .index, true⟩, none⟩])).isNone
#guard (Definedness.Insert.rule [] (write [] [])
  [⟨⟨.scalar (.int 13), false⟩, none⟩, ⟨⟨.tensor [] (.int 13), true⟩, none⟩]).isNone

end MerlinVerify.Tests.InsertExtract
