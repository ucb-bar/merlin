/- Actual exporter defaults are normalized before cast composition. Numerical promises and
unknown headers refuse; both actual sides receive the same proved internal representation. -/
import MerlinVerify.Rules.IntegerCastPartial
import MerlinVerify.Tests.Semantics

namespace MerlinVerify.Tests.IntegerCastHeader

open MerlinVerify MerlinVerify.Check MerlinVerify.Rules

def Γ (a : Nat) : TEnv := [(0, .scalar (.int a)), (1, .scalar (.int 32))]
def cast (name : String) (input result width : Nat) (attrs : List (String × Attr) := [])
    (props : List (String × Attr) := []) : Op :=
  .mk name [⟨result, .int width⟩] [input] attrs props [] []
def ret (ids : List Nat) : Op := .mk "func.return" [] ids [] [] [] []
def header (s : Op) := (s.name, s.results.map (fun v => (v.id, v.ty)), s.operands, s.attrs, s.props)
def noneFlags : List (String × Attr) := [("overflowFlags", .enum "arith" "overflow" "none")]
def ns (a : Nat) (ss : List Op) := IntegerCastPartial.normalize 64 (Γ a) [] ss

def agrees (kind : String) (a b c : Nat) (props : Bool := false) : Bool :=
  let source := [cast kind 0 2 b, cast "arith.trunci" 2 3 c
    (if props then [] else noneFlags) (if props then noneFlags else []), ret [3, 1]]
  let target := [cast (if c < a then "arith.trunci" else kind) 0 2 c, ret [2, 1]]
  match ns a source, ns a target with
  | some left, some right => alphaOps [(0, 0), (1, 1)] left right
  | _, _ => false

#guard ["arith.extui", "arith.extsi"].all fun kind =>
  [(1,32,8), (8,32,13), (13,64,8), (32,128,13)].all fun (a,b,c) =>
    agrees kind a b c && agrees kind a b c true
#guard [false, true].all fun props =>
  let raw := cast "arith.trunci" 0 2 8 (if props then [] else noneFlags) (if props then noneFlags else [])
  header (IntegerCastHeader.operation (Γ 32) raw) == header (cast "arith.trunci" 0 2 8)
#guard header (IntegerCastHeader.operation (Γ 8)
  (cast "arith.extui" 0 2 32 [("prov.fqn", .str "producer")])) == header (cast "arith.extui" 0 2 32)
#guard header (IntegerCastHeader.operation (Γ 8)
  (cast "arith.extsi" 0 2 32 [("fastmath", .enum "arith" "fastmath" "none")])) == header (cast "arith.extsi" 0 2 32)

-- A nonempty overflow promise, malformed effective attribute, unknown key or other promise
-- cannot be mistaken for an absent/default header, even if a shadowed attribute says otherwise.
#guard ["nsw", "nuw", "nsw,nuw", "unreviewed"].all fun flags =>
  let raw := cast "arith.trunci" 0 2 8 [("overflowFlags", .enum "arith" "overflow" flags)]
  header (IntegerCastHeader.operation (Γ 32) raw) == header raw
#guard ["nonNeg", "isExact", "disjoint", "unknown.meaning"].all fun key =>
  let raw := cast "arith.extui" 0 2 32 [(key, .unit)]
  header (IntegerCastHeader.operation (Γ 8) raw) == header raw
#guard let raw := cast "arith.trunci" 0 2 8 noneFlags [("overflowFlags", .enum "arith" "overflow" "nuw")]
  header (IntegerCastHeader.operation (Γ 32) raw) == header raw
#guard let raw := cast "arith.trunci" 0 2 8 noneFlags [("overflowFlags", .str "none")]
  header (IntegerCastHeader.operation (Γ 32) raw) == header raw
#guard let raw := cast "arith.extui" 0 2 32 [("overflowFlags", .enum "arith" "overflow" "none")]
  header (IntegerCastHeader.operation (Γ 8) raw) == header raw
#guard let raw := cast "arith.extsi" 0 2 32 [("fastmath", .enum "arith" "fastmath" "nnan")]
  header (IntegerCastHeader.operation (Γ 8) raw) == header raw

-- Original default-bearing truncation and its canonical header retain exact poison behavior.
def run (ops : List Op) (value : Val) :=
  Semantics.readInts (evalOps Sem.Ops.opSem Semantics.ctx (fun _ _ => .stuck "no calls")
    [(0, .scalar value), (1, .scalar (.some (.bits 32 37)))] ops)
#guard ["arith.extui", "arith.extsi"].all fun kind =>
  let source := [cast kind 0 2 32, cast "arith.trunci" 2 3 8 noneFlags, ret [3, 1]]
  run source .poison == run ((ns 1 source).getD []) .poison
#guard ["arith.extui", "arith.extsi"].all fun kind => (List.range 256).all fun n =>
  let source := [cast kind 0 2 32, cast "arith.trunci" 2 3 13 noneFlags, ret [3, 1]]
  let value := Val.some (.bits 8 (BitVec.ofNat 8 n))
  run source value == run ((ns 8 source).getD []) value

end MerlinVerify.Tests.IntegerCastHeader
