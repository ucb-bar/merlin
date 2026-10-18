/- Exact neutral aliases use actual scalar width and literal-binding evidence. -/
import MerlinVerify.Rules.IntegerNeutral
import MerlinVerify.Tests.Semantics

namespace MerlinVerify.Tests.IntegerNeutral

open MerlinVerify MerlinVerify.Check MerlinVerify.Rules.IntegerNeutral

def Γ (w : Nat) : TEnv := [(0,.scalar (.int w)),(1,.scalar (.int w))]
def binary (kind : Kind) (rev : Bool) (w : Nat) : Op :=
  .mk kind.name [⟨3,.int w⟩] (Rules.IntegerIdentity.ordered rev 0 2) [] [] [] []
def constant (kind : Kind) (w : Nat) : Op := .mk "arith.constant" [⟨2,.int w⟩] []
  [("value",.int (kind.neutral w).toInt (.int w))] [] [] []
def ret (ids : List Nat) : Op := .mk "func.return" [] ids [] [] [] []
def source (kind : Kind) (rev : Bool) (w : Nat) : List Op :=
  [constant kind w,binary kind rev w,ret [3,2,3,1]]
def normalized (kind : Kind) (rev : Bool) (w : Nat) := normalize 64 (Γ w) [] (source kind rev w)

#guard kinds.all fun k => [0,1,4,13,64].all fun w =>
  (normalized k false w).map (fun ops => ops.map (·.operands)) == some [[],[0,2,0,1]]
#guard (kinds.filter (· != .sub)).all fun k =>
  (normalized k true 13).map (fun ops => ops.map (·.operands)) == some [[],[0,2,0,1]]
#guard (normalized .sub true 13).map List.length == some 3

def facts (kind : Kind) (w : Nat) := ConstantAlias.advance 64 [] (constant kind w)
def localΓ (w : Nat) := (2,.scalar (.int w)) :: Γ w
#guard (eliminate 64 [] (facts .add 13) (binary .add false 13) [ret [3]]).isNone
#guard (eliminate 64 [(0,.scalar (.int 8)),(2,.scalar (.int 13))]
  (facts .add 13) (binary .add false 13) [ret [3]]).isNone
#guard (eliminate 64 [(0,.tensor [3] (.int 13)),(2,.scalar (.int 13))]
  (facts .add 13) (binary .add false 13) [ret [3]]).isNone
#guard (eliminate 64 [(0,.scalar .index),(2,.scalar (.int 13))]
  (facts .add 13) (binary .add false 13) [ret [3]]).isNone
#guard (eliminate 64 (localΓ 13) [] (binary .add false 13) [ret [3]]).isNone
#guard (eliminate 64 (localΓ 13) (facts .mul 13) (binary .add false 13) [ret [3]]).isNone

-- Every promise or unknown semantic modifier is rejected by this exact family.
def withAttr (key : String) (value : Attr) : Op :=
  .mk "arith.addi" [⟨3,.int 13⟩] [0,2] [(key,value)] [] [] []
#guard ["nonNeg","isExact","disjoint","unreviewed.meaning"].all fun key =>
  (eliminate 64 (localΓ 13) (facts .add 13) (withAttr key .unit) [ret [3]]).isNone
#guard (eliminate 64 (localΓ 13) (facts .add 13)
  (withAttr "overflowFlags" (.enum "arith" "overflow" "nsw")) [ret [3]]).isNone
#guard (eliminate 64 (localΓ 13) (facts .add 13)
  (withAttr "fastmath" (.enum "arith" "fastmath" "nnan")) [ret [3]]).isNone
#guard (eliminate 64 (localΓ 13) (facts .add 13)
  (.mk "arith.addi" [⟨0,.int 13⟩] [0,2] [] [] [] []) [ret [0]]).isNone
#guard (eliminate 64 (localΓ 13) (facts .add 13)
  (binary .add false 13) [.mk "arith.addi" [⟨0,.int 13⟩] [1,1] [] [] [] [],ret [3]]).isNone

def nested (arg : Nat) : Op := .mk "test.region" [] [] [] []
  [.mk [.mk [⟨arg,.int 13⟩] [ret [3]]]] []
#guard (eliminate 64 (localΓ 13) (facts .add 13) (binary .add false 13) [nested 4,ret [1]]).isSome
#guard (eliminate 64 (localΓ 13) (facts .add 13) (binary .add false 13) [nested 3,ret [1]]).isNone

-- Source shadowing forgets the old constant binding; a new literal is checked independently.
def overwrite : Op := .mk "arith.constant" [⟨2,.int 13⟩] [] [("value",.int 7 (.int 13))] [] [] []
#guard (eliminate 64 (localΓ 13) (ConstantAlias.advance 64 (facts .add 13) overwrite)
  (binary .add false 13) [ret [3]]).isNone

def run (ops : List Op) (value : Val) :=
  Semantics.readInts (evalOps Sem.Ops.opSem Semantics.ctx (fun _ _ => .stuck "no calls")
    [(0,.scalar value),(1,.scalar (.some (.bits 13 7)))] ops)
#guard kinds.all fun k => run (source k false 13) .poison ==
  run ((normalized k false 13).getD []) .poison
#guard kinds.all fun k => (List.range 16).all fun x =>
  run (source k false 13) (.some (.bits 13 (BitVec.ofNat 13 x))) ==
  run ((normalized k false 13).getD []) (.some (.bits 13 (BitVec.ofNat 13 x)))

-- These annihilators erase poison and must not be offered as neutral aliases.
#guard [.mul,.band,.minU,.maxU,.minS,.maxS].all fun kind =>
  let wrong : ConstantFacts.Fact := ⟨2,.scalar (.some (.bits 13
    (match kind with | .maxU | .maxS => BitVec.allOnes 13 | _ => 0)))⟩
  (eliminate 64 (localΓ 13) [wrong] (binary kind false 13) [ret [3]]).isNone

end MerlinVerify.Tests.IntegerNeutral
