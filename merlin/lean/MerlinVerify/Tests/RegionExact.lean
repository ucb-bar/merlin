/- All-region traversal retains headers and scopes, including raw malformed/failing regions. -/
import MerlinVerify.Check.RegionExact
import MerlinVerify.Rules.IntegerOrder
import MerlinVerify.Tests.Semantics

namespace MerlinVerify.Tests.RegionExact

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops MerlinVerify.Tests.Semantics

def order := Rules.IntegerOrder.operations
theorem order_exact : Check.RegionExact.Exact order :=
  fun ctx call env ops => Rules.IntegerOrder.operations_equal ctx call ops env

def add : Op := mkOp "arith.addi" [v 9 i32] [7, 3]
def sortedAdd : Op := mkOp "arith.addi" [v 9 i32] [3, 7]
def block (ops : List Op) : Block := .mk [v 7 i32] ops
def box (ops : List Op) : Op :=
  mkOp "test.unknown" [v 99 i32] [3] [("retain", .str "attribute")]
    [("retain", .str "property")] [.mk [block ops]]
def body := [add, ret [9]]
def sortedBody := [sortedAdd, ret [9]]
def normalize := Check.RegionExact.recursive order
def operandsAt : Nat → List Op → List (List ValueId)
  | 0, ops => ops.map Op.operands
  | depth + 1, ops =>
    operandsAt depth (((ops.headD (ret [])).regions.headD (.mk [])).blocks.headD (.mk [] [])).ops

#guard operandsAt 0 (normalize 0 body) == [[3, 7], [9]]
example : normalize 0 [box body] = [box body] := rfl
#guard operandsAt 1 (normalize 1 [box body]) == [[3, 7], [9]]
example : normalize 1 [box [box body]] = [box [box body]] := rfl
#guard operandsAt 2 (normalize 2 [box [box body]]) == [[3, 7], [9]]
#guard operandsAt 2 (normalize (RegionDepth.ops [box [box body]]) [box [box body]]) == [[3, 7], [9]]
example : normalize 8 [box [box body]] = normalize 2 [box [box body]] := rfl
example : (Check.RegionExact.operation order (box body)).blind = (box body).blind := rfl
#guard (Check.RegionExact.operation order (box body)).operands == [3]
example : (Check.RegionExact.operation order (box body)).results = [v 99 i32] := rfl
#guard (Check.RegionExact.operation order (box body)).successors == (box body).successors
example : Check.RegionExact.region order (.mk []) = .mk [] := rfl
#guard (Check.RegionExact.region order (.mk [block body, block body])).blocks.map
  (fun b => b.ops.map Op.operands) == [[[3, 7], [9]], [[3, 7], [9]]]

def enclosing : Env := [(3, i8v 4), (7, i8v 99)]
def captured : Env := [(3, .scalar (.some (.bits 32 4)))]
def scalar (n : Int) : RVal := .scalar (.some (.bits 32 (BitVec.ofInt 32 n)))
def run (r : Region) (args : List RVal) := readInts (evalRegion opSem ctx (fun _ _ => .ub) captured r args)

-- Captures are evaluated in the unchanged outer environment; actual argument binding shadows it.
#guard run (.mk [block body]) [scalar 5] == some [[some 9]]
#guard run (Check.RegionExact.region order (.mk [block body])) [scalar 5] == some [[some 9]]
#guard run (.mk [block body]) [.scalar .poison] == some [[none]]
#guard run (Check.RegionExact.region order (.mk [block body])) [.scalar .poison] == some [[none]]
#guard isStuck (evalRegion opSem ctx (fun _ _ => .ub) captured (.mk [block body]) [])
#guard isStuck (evalRegion opSem ctx (fun _ _ => .ub) captured
  (Check.RegionExact.region order (.mk [block body])) [])
#guard isStuck (evalRegion opSem ctx (fun _ _ => .ub) []
  (Check.RegionExact.region order (.mk [block body])) [scalar 5])
#guard isStuck (evalRegion opSem ctx (fun _ _ => .ub) captured
  (Check.RegionExact.region order (.mk [block body, block body])) [scalar 5])
#guard isStuck (evalRegion opSem ctx (fun _ _ => .ub) captured
  (Check.RegionExact.region order (.mk [])) [])
#guard isStuck (evalOps opSem ctx (fun _ _ => .ub) captured (normalize 4 [ret [], add]))
#guard operandsAt 0 (normalize 3 [mkOp "arith.addf" [v 9 (.float .f32)] [7, 3]]) == [[7, 3]]
#guard operandsAt 0 (normalize 3 [mkOp "arith.addi" [v 9 i32] [7, 3]
  [("overflowFlags", .enum "arith" "overflow" "nsw")]]) == [[7, 3]]
#guard operandsAt 0 (normalize 2 [mkOp "linalg.yield" [] [7, 3]]) == [[7, 3]]

-- Exactness includes the full outcome, not only successful or initialized runs.
example (fuel : Nat) (ctx : Ctx) (call : CallSem) (env : Env) (ops : List Op) :
    evalOps opSem ctx call env (normalize fuel ops) = evalOps opSem ctx call env ops :=
  Check.RegionExact.recursive_equal order_exact fuel ctx call env ops

example : Check.RegionExact.Exact (order ∘ order) :=
  Check.RegionExact.compose order_exact order_exact

end MerlinVerify.Tests.RegionExact
