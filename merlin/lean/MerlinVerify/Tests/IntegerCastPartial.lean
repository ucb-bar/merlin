/- Actual-width cast composition, exact poison, escaping intermediates, and stale origins. -/
import MerlinVerify.Rules.IntegerCastPartial
import MerlinVerify.Tests.Semantics

namespace MerlinVerify.Tests.IntegerCastPartial

open MerlinVerify MerlinVerify.Check MerlinVerify.Rules
open IntegerCastPartial

def Γ (a : Nat) : TEnv := [(0,.scalar (.int a)),(1,.scalar (.int 32))]
def cast (kind : IntegerCast.Kind) (input result width : Nat) : Op :=
  .mk kind.name [⟨result,.int width⟩] [input] [] [] [] []
def ret (ids : List Nat) : Op := .mk "func.return" [] ids [] [] [] []
def source (kind : IntegerCast.Kind) (_a b c : Nat) (escape : Bool := false) : List Op :=
  [cast kind 0 2 b,cast .trunc 2 3 c,ret (if escape then [3,2,0,1] else [3,0,1])]
def normalized (kind : IntegerCast.Kind) (a b c : Nat) (escape : Bool := false) :=
  normalize 64 (Γ a) [] (source kind a b c escape)

def above (kind : IntegerCast.Kind) (a b c : Nat) : Bool :=
  (normalized kind a b c).map (fun ops => ops.map (fun op => (op.name,op.operands))) ==
    some [(kind.name,[0]),("func.return",[3,0,1])]
def below (kind : IntegerCast.Kind) (a b c : Nat) : Bool :=
  (normalized kind a b c).map (fun ops => ops.map (fun op => (op.name,op.operands))) ==
    some [("arith.trunci",[0]),("func.return",[3,0,1])]

#guard [IntegerCast.Kind.zero,.sign].all fun kind =>
  [(0,8,3),(1,32,8),(8,32,13),(13,64,32),(32,128,64)].all fun (a,b,c) => above kind a b c
#guard [IntegerCast.Kind.zero,.sign].all fun kind =>
  [(1,8,0),(4,32,1),(13,32,8),(32,128,13),(64,128,32)].all fun (a,b,c) => below kind a b c
#guard (normalized .sign 8 32 13 true).map (fun ops => ops.map (fun op => (op.name,op.operands))) ==
  some [("arith.extsi",[0]),("arith.extsi",[0]),("func.return",[3,2,0,1])]
-- Existing exact original-width cancellation is retained through the CastCancel fallback.
#guard (normalized .zero 8 32 8).map (fun ops => ops.map (fun op => (op.name,op.operands))) ==
  some [("func.return",[0,0,1])]
#guard (normalized .sign 8 32 8 true).map List.length == some 2
#guard (normalize 64 [] [] (source .sign 8 32 13)).isNone
#guard !(decide (Compatible .trunc 13 8 4))
#guard !(decide (Compatible .sign 8 8 4))
#guard !(decide (Compatible .sign 8 32 32))
#guard !(decide (Compatible .zero 8 32 8))

-- Rebinding the root invalidates every fact about its prior runtime value.
def overwrite : Op := .mk "arith.constant" [⟨0,.int 8⟩] [] [("value",.int 7 (.int 8))] [] [] []
#guard (normalize 64 (Γ 8) []
  [cast .sign 0 2 32,overwrite,cast .trunc 2 3 13,ret [3,0,1]]).map List.length == some 4

def flagged (key : String) (attr : Attr) : Op :=
  .mk "arith.extui" [⟨2,.int 32⟩] [0] [(key,attr)] [] [] []
#guard ["nonNeg","isExact","disjoint","unknown.meaning"].all fun key =>
  (normalize 64 (Γ 8) [] [flagged key .unit,cast .trunc 2 3 13,ret [3,0,1]]).map List.length == some 3
#guard (normalize 64 (Γ 8) []
  [flagged "overflowFlags" (.enum "arith" "overflow" "nsw"),cast .trunc 2 3 13,ret [3,0,1]]).map List.length == some 3
#guard (normalize 64 (Γ 8) []
  [cast .zero 0 2 32,.mk "arith.trunci" [⟨3,.int 13⟩] [2] [("unknown.meaning",.unit)] [] [] [],ret [3,0,1]]).map List.length == some 3

open MerlinVerify.Tests.Semantics

def run (_a : Nat) (ops : List Op) (value : Val) :=
  readInts (evalOps Sem.Ops.opSem ctx (fun _ _ => .stuck "no calls")
    [(0,.scalar value),(1,.scalar (.some (.bits 32 37)))] ops)
#guard [IntegerCast.Kind.zero,.sign].all fun kind =>
  [(1,32,8),(4,32,1),(8,32,13),(13,32,8)].all fun (a,b,c) =>
    run a (source kind a b c) .poison == run a ((normalized kind a b c).getD []) .poison
#guard [IntegerCast.Kind.zero,.sign].all fun kind =>
  [(1,32,8),(4,32,1),(8,32,13),(13,32,8)].all fun (a,b,c) =>
    (List.range (2^a)).all fun n =>
      run a (source kind a b c) (.some (.bits a (BitVec.ofNat a n))) ==
      run a ((normalized kind a b c).getD []) (.some (.bits a (BitVec.ofNat a n)))

end MerlinVerify.Tests.IntegerCastPartial
