/- Return-boundary refinement is deliberately weaker than an exact raw-tensor alias. -/
import MerlinVerify.Check.IdentityReturn
import MerlinVerify.Tests.Semantics

namespace MerlinVerify.Tests.IdentityReturn

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops MerlinVerify.Tests.Semantics

def Γ (shape : List Nat) (elt : Ty := .int 13) : TEnv :=
  [(0, .tensor shape elt), (1, .tensor shape elt), (2, .scalar (.int 32))]
def ρ : Rho := [(0, 20), (1, 21), (2, 22)]
def generic (shape : List Nat) (elt : Ty := .int 13) (yieldId : Nat := 10)
    (result : Nat := 3) (old : Nat := 11) : Op :=
  mkOp "linalg.generic" [v result (tensorTy shape elt)] [0, 1]
    [("indexing_maps", .array [amap shape.length (List.range shape.length),
        amap shape.length (List.range shape.length)]),
     ("iterator_types", .array (List.replicate shape.length (iter "parallel"))),
     ("operandSegmentSizes", .denseArray (.int 32) [1, 1])] []
    [.mk [.mk [v 10 elt, v old elt] [mkOp "linalg.yield" [] [yieldId]]]]
def source (shape : List Nat) (elt : Ty := .int 13) :=
  [generic shape elt, ret [3, 2, 0, 3, 1]]
def target := [ret [20, 22, 20, 20, 21]]
def accepts (shape : List Nat) (elt : Ty := .int 13) :=
  IdentityReturn.check 64 (Γ shape elt) ρ (source shape elt) target

#guard [[], [0], [1], [2, 0, 3], [2, 3], [1, 2, 1]].all fun sh => accepts sh
#guard [Ty.int 1, .int 13, .int 65, .float .f16, .float .bf16, .float .f32, .float .f64, .index].all
  fun elt => accepts [2] elt
#guard IdentityReturn.check 32 (Γ [2] .index) ρ (source [2] .index) target
#guard IdentityReturn.check 64 (Γ [2]) ρ [generic [2], ret [3]] [ret [20]]

def withAttrs (s : Op) (attrs : List (String × Attr)) : Op :=
  match s with
  | .mk n r o _ p b succ => .mk n r o attrs p b succ
def withRegions (s : Op) (regions : List Region) : Op :=
  match s with
  | .mk n r o a p _ succ => .mk n r o a p regions succ
def attributes := (generic [2]).attrs

-- Different destinations, output uses, layouts and current binding types cannot masquerade as a copy.
#guard !IdentityReturn.check 64 (Γ [2]) ρ [generic [2] (.int 13) 11, ret [3]] [ret [20]]
#guard !IdentityReturn.check 64 (Γ [2]) ρ [generic [2], ret [3]] [ret [21]]
#guard !IdentityReturn.check 64 (Γ [2]) ρ [generic [2], ret [3, 77]] [ret [20, 22]]
#guard !IdentityReturn.check 64 (Γ [2]) [] (source [2]) target
#guard !IdentityReturn.check 64 (Γ [2]) ρ [generic [2] (.int 13) 10 0, ret [0]] [ret [20]]
#guard !IdentityReturn.check 64 (Γ [2]) ρ [generic [2] (.int 13) 10 3 10, ret [3]] [ret [20]]
#guard !IdentityReturn.check 64 ((0, .scalar (.int 13)) :: Γ [2]) ρ (source [2]) target
#guard !IdentityReturn.check 64 ((1, .tensor [3] (.int 13)) :: Γ [2]) ρ (source [2]) target
#guard !IdentityReturn.check 64 ((1, .tensor [2] (.int 8)) :: Γ [2]) ρ (source [2]) target
#guard !IdentityReturn.check 64 (Γ [2]) ρ [generic [2], ret [2]] [ret [22]]

def reduction := withAttrs (generic [2])
  [("iterator_types", .array [iter "reduction"]), attributes[0]!, attributes[2]!]
def wrongMap := withAttrs (generic [2])
  [("indexing_maps", .array [amap 1 [], amap 1 [0]]), attributes[1]!, attributes[2]!]
#guard !IdentityReturn.check 64 (Γ [2]) ρ [reduction, ret [3]] [ret [20]]
#guard !IdentityReturn.check 64 (Γ [2]) ρ [wrongMap, ret [3]] [ret [20]]
#guard !IdentityReturn.check 64 (Γ [2]) ρ
  [withAttrs (generic [2]) (("unknown", .unit) :: attributes), ret [3]] [ret [20]]
#guard !IdentityReturn.check 64 (Γ [2]) ρ
  [generic [2], withAttrs (ret [3]) [("unknown", .unit)]] [ret [20]]
#guard !IdentityReturn.check 64 (Γ [2]) ρ
  [generic [2], ret [3]] [withAttrs (ret [20]) [("unknown", .unit)]]
#guard !IdentityReturn.check 64 (Γ [2]) ρ
  [withRegions (generic [2]) [.mk [.mk [v 10 (.int 13), v 11 (.int 13)]
    [mkOp "linalg.yield" [] [10] [("unknown", .unit)]]]], ret [3]] [ret [20]]
#guard !IdentityReturn.check 64 (Γ [2]) ρ
  [withRegions (generic [2]) [.mk [.mk [v 10 (.int 13), v 11 (.int 13)]
    [mkOp "func.return" [] [10]]]], ret [3]] [ret [20]]

-- No prefix or arbitrary continuation may observe a representation difference.
def empty := mkOp "tensor.empty" [v 4 (tensorTy [2] (.int 13))] []
#guard !IdentityReturn.check 64 (Γ [2]) ρ (empty :: source [2]) target
#guard !IdentityReturn.check 64 (Γ [2]) ρ (source [2]) (empty :: target)

def bits (n : Nat) : Val := .some (.bits 13 (BitVec.ofNat 13 n))
def noncanonical : Tensor :=
  { shape := [2], elt := .int 13,
    get := fun pt => if Idx.InBounds [2] pt then bits (linearIndex [2] pt + 1) else bits 77 }
def sourceEnv : Env := [(0, .tensor noncanonical), (1, .tensor (Tensor.full [2] (.int 13) .poison)),
  (2, .scalar (.some (.bits 32 37)))]
def targetEnv : Env := [(20, .tensor noncanonical), (21, .tensor (Tensor.full [2] (.int 13) .poison)),
  (22, .scalar (.some (.bits 32 37)))]
def run (env : Env) (ops : List Op) := evalOps opSem ctx (fun _ _ => .ub) env ops
def outside : Outcome (List RVal) → Option Val
  | .ok (.tensor t :: _) => some (t.get [2])
  | _ => none

#guard readInts (run sourceEnv (source [2])) == readInts (run targetEnv target)
#guard outside (run sourceEnv (source [2])) == some .poison
#guard outside (run targetEnv target) == some (bits 77)
#guard readInts (run [(0, .tensor (Tensor.full [2] (.int 13) .poison)),
  (1, .tensor noncanonical), (2, .scalar (.some (.bits 32 37)))] (source [2])) ==
  some [[none, none], [some 37], [none, none], [none, none], [some 1, some 2]]

def inspect (id : Nat) := mkOp "func.call" [v 5 (.int 13)] [id]
  [("callee", .symbolRef "inspect" [])]
def observer : CallSem := fun _ args =>
  match args with
  | [.tensor t] => .ok [.scalar (t.get [2])]
  | _ => .ub
def sourceCall := [generic [2], inspect 3, ret [5]]
def targetCall := [inspect 20, ret [5]]
#guard !IdentityReturn.check 64 (Γ [2]) ρ sourceCall targetCall
#guard readInts (evalOps opSem ctx observer sourceEnv sourceCall) == some [[none]]
#guard readInts (evalOps opSem ctx observer targetEnv targetCall) == some [[some 77]]

end MerlinVerify.Tests.IdentityReturn
