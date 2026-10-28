/- Exact scalar cast round trips, conservative modifiers, SSA invalidation and captures. -/
import MerlinVerify.Rules.CastCancel
import MerlinVerify.Tests.Semantics

namespace MerlinVerify.Tests.CastCancel

open MerlinVerify MerlinVerify.Check MerlinVerify.Rules
open MerlinVerify.Tests.Semantics

def cast (kind : IntegerCast.Kind) (x r width : Nat) (attrs : List (String × Attr) := []) : Op :=
  .mk kind.name [⟨r, .int width⟩] [x] attrs [] [] []
def scope (width : Nat) : TEnv := [(0, .scalar (.int width))]
def chain (kind : IntegerCast.Kind) (width wide : Nat) (uses : List Nat := [2]) : List Op :=
  [cast kind 0 1 wide, cast .trunc 1 2 width, ret uses]
def normalized (kind : IntegerCast.Kind) (width wide : Nat) (uses : List Nat := [2]) :=
  Rules.CastCancel.normalize 64 (scope width) [] (chain kind width wide uses)

#guard (normalized .zero 1 2).map (List.map Op.name) == some ["func.return"]
#guard (normalized .sign 1 128).map (List.map Op.operands) == some [[0]]
#guard (normalized .zero 13 32).map (List.map Op.operands) == some [[0]]
#guard (normalized .sign 32 64).map (List.map Op.operands) == some [[0]]
#guard (normalized .sign 64 128 [2, 0]).map (List.map Op.operands) == some [[0, 0]]
#guard (normalized .zero 8 13 [1, 2, 0]).map (List.map Op.operands) == some [[0], [1, 0, 0]]
#guard (normalized .sign 13 32 [1, 2]).map (List.map Op.name) == some ["arith.extsi", "func.return"]

-- Composed successful-source origins survive multiple strict widening steps.
#guard (Rules.CastCancel.normalize 32 (scope 1) []
  [cast .zero 0 1 8, cast .sign 1 2 13, cast .zero 2 3 64,
   cast .trunc 3 4 1, ret [4]]).map (List.map Op.operands) == some [[0]]
#guard (Rules.CastCancel.normalize 64 (scope 8) []
  [cast .sign 0 1 13, cast .sign 1 2 32, cast .trunc 2 3 8,
   ret [3, 1]]).map (List.map Op.operands) == some [[0], [0, 1]]

-- A partial truncation or a truncation followed by extension is outside this family.
#guard (Rules.CastCancel.normalize 64 (scope 8) []
  [cast .sign 0 1 32, cast .trunc 1 2 13, ret [2]]).map List.length == some 3
#guard (Rules.CastCancel.normalize 64 (scope 32) []
  [cast .trunc 0 1 8, cast .sign 1 2 32, ret [2]]).map List.length == some 3
#guard (Rules.CastCancel.normalize 64 (scope 8) []
  [cast .zero 0 1 8, cast .trunc 1 2 8, ret [2]]).map List.length == some 3
#guard (Rules.CastCancel.normalize 64 [(0, .tensor [2] (.int 8))] []
  (chain .zero 8 32)).map List.length == some 3

def facts : List IntegerCast.Fact := Rules.CastCancel.advance (scope 8) [] (cast .sign 0 1 32)
def middle : TEnv := [(1, .scalar (.int 32))] ++ scope 8
def capture (args : List Value) (uses : List Nat) : Op :=
  .mk "unknown.capture" [] [] [] [] [.mk [.mk args [ret uses]]] []

#guard facts.length == 1
#guard (Rules.CastCancel.advance middle facts (mkOp "unknown.write" [v 0 (.int 8)] [])).isEmpty
#guard (Rules.CastCancel.advance middle facts (mkOp "unknown.write" [v 1 (.int 32)] [])).isEmpty
#guard (Rules.CastCancel.advance middle facts (mkOp "unknown.write" [v 4 (.int 32)] [])).length == 1
#guard (Rules.CastCancel.advance (scope 8) [] (cast .sign 0 0 32)).isEmpty
#guard (Rules.CastCancel.eliminate middle facts (cast .trunc 1 2 8) [capture [] [2]]).isSome
#guard (Rules.CastCancel.eliminate middle facts (cast .trunc 1 2 8)
  [capture [v 0 (.int 8)] [2]]).isNone
#guard (Rules.CastCancel.eliminate middle facts (cast .trunc 1 2 8)
  [capture [v 2 (.int 8)] [2]]).isNone
#guard (Rules.CastCancel.eliminate middle facts (cast .trunc 1 2 8)
  [mkOp "unknown.write" [v 0 (.int 8)] [], ret [2]]).isNone
#guard (Rules.CastCancel.eliminate ((2, .scalar (.int 8)) :: middle) facts
  (cast .trunc 1 2 8) [ret [2]]).isNone
#guard !(Rules.CastCancel.drop? (scope 8) (cast .sign 0 1 32) [capture [] [1]])
#guard !(Rules.CastCancel.drop? middle (cast .sign 0 1 32) [ret []])

-- Modifiers and unknown syntax cannot seed origin facts or be cancelled/deleted.
def flagged (key : String) : Op := cast .sign 0 1 32 [(key, .unit)]
#guard (Rules.CastCancel.advance (scope 8) [] (flagged "nonNeg")).isEmpty
#guard (Rules.CastCancel.advance (scope 8) [] (flagged "isExact")).isEmpty
#guard (Rules.CastCancel.advance (scope 8) [] (flagged "disjoint")).isEmpty
#guard (Rules.CastCancel.advance (scope 8) [] (flagged "unknown.meaning")).isEmpty
#guard !(Rules.CastCancel.drop? (scope 8) (flagged "isExact") [ret []])
#guard (Rules.CastCancel.eliminate middle facts (cast .trunc 1 2 8 [("isExact", .unit)]) [ret [2]]).isNone
#guard (Rules.CastCancel.eliminate middle facts
  (cast .trunc 1 2 8 [("unknown.meaning", .unit)]) [ret [2]]).isNone
#guard (Rules.CastCancel.eliminate middle facts
  (cast .trunc 1 2 8 [("overflowFlags", .enum "arith" "overflow" "nsw")]) [ret [2]]).isNone
#guard (Rules.CastCancel.advance (scope 8) []
  (cast .sign 0 1 32 [("fastmath", .enum "arith" "fastmath" "nnan")])).isEmpty

def run (ops : List Op) (value : Val) :=
  readInts (evalOps Sem.Ops.opSem ctx (fun _ _ => .stuck "no calls") [(0, .scalar value)] ops)
#guard run (chain .sign 8 32) .poison == some [[none]]
#guard run ((normalized .sign 8 32).getD []) .poison == some [[none]]
#guard run (chain .zero 8 32) (.some (.bits 8 255)) == some [[some (-1)]]
#guard run ((normalized .zero 8 32).getD []) (.some (.bits 8 255)) == some [[some (-1)]]
#guard run (chain .sign 8 32 [1, 2]) (.some (.bits 8 128)) == some [[some (-128)], [some (-128)]]
#guard run ((normalized .sign 8 32 [1, 2]).getD []) (.some (.bits 8 128)) ==
  some [[some (-128)], [some (-128)]]

-- The runtime-domain premise is necessary: a mistyped i13 payload is preserved by the alias
-- while the original cast chain narrows it. The public theorem requires actual i8 typing.
#guard run (chain .zero 8 32) (.some (.bits 13 256)) == some [[some 0]]
#guard run ((normalized .zero 8 32).getD []) (.some (.bits 13 256)) == some [[some 256]]

example : IntegerCast.Kind.trunc.apply 0 (IntegerCast.Kind.sign.apply 1 .poison) = .poison := rfl

end MerlinVerify.Tests.CastCancel
