/- Actual predicate precedence, exact stuck/poison behavior, and guarded canonical headers. -/
import MerlinVerify.Rules.IntegerCompareOrder
import MerlinVerify.Tests.Semantics

namespace MerlinVerify.Tests.IntegerCompareOrder

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops
open MerlinVerify.Rules.IntegerCompareOrder
open MerlinVerify.Tests.Semantics

def cmp (p : Predicate) (x y : Nat) : Op :=
  .mk "arith.cmpi" [⟨2,.int 1⟩] [x,y] [] [("predicate",.int p.code (.int 64))] [] []
def predicate (s : Op) := attrInt? s.blind "predicate"
def ret : Op := .mk "func.return" [] [2] [] [] [] []

#guard predicates.all fun p => (operation (cmp p 1 0)).operands == [0,1]
#guard predicates.all fun p => predicate (operation (cmp p 1 0)) == some p.flip.code
#guard predicates.all fun p => predicate (operation (cmp p 0 1)) == some p.code
#guard predicates.all fun p => (operation (cmp p 0 1)).props == []
#guard predicates.all fun p => (operation (cmp p 1 0)).results.map (·.id) == [2]
#guard predicates.all fun p => predicate (operation (operation (cmp p 1 0))) == some p.flip.code
#guard predicates.all fun p => (operation (operation (cmp p 1 0))).operands == [0,1]

def decorated : Op := .mk "arith.cmpi" [⟨2,.int 1⟩] [1,0]
  [("predicate",.int 1 (.int 64)),("prov.experiment",.str "retained-original")]
  [("predicate",.int 2 (.int 64))] [] []
-- Actual property lookup wins over a conflicting discardable attribute.
#guard predicate decorated == some 2
#guard predicate (operation decorated) == some 4
#guard (operation decorated).attrs == [("predicate",.int 4 (.int 64))]
#guard (operation decorated).props == []

def badProperty : Op := .mk "arith.cmpi" [⟨2,.int 1⟩] [1,0]
  [("predicate",.int 2 (.int 64))] [("predicate",.opaque "unparsed")] [] []
#guard (candidate badProperty).isNone
#guard (operation badProperty).props == badProperty.props

def withAttr (key : String) (value : Attr) : Op :=
  .mk "arith.cmpi" [⟨2,.int 1⟩] [1,0] [(key,value)] [("predicate",.int 2 (.int 64))] [] []
#guard ["nonNeg","isExact","disjoint","unreviewed.meaning"].all fun key =>
  (operation (withAttr key .unit)).operands == [1,0]
#guard (operation (withAttr "fastmath" (.enum "arith" "fastmath" "nnan"))).operands == [1,0]
#guard (operation (withAttr "fastmath" (.str "none"))).operands == [1,0]
#guard (operation (withAttr "overflowFlags" (.enum "arith" "overflow" "nsw"))).operands == [1,0]
-- Explicit recognized default fastmath is semantically integer-only and may canonicalize.
#guard (operation (withAttr "fastmath" (.enum "arith" "fastmath" "none"))).operands == [0,1]

def invalid (results : List Value) (operands : List Nat) (regions : List Region := [])
    (successors : List Nat := []) : Op :=
  .mk "arith.cmpi" results operands [] [("predicate",.int 2 (.int 64))] regions successors
#guard (operation (invalid [] [1,0])).operands == [1,0]
#guard (operation (invalid [⟨2,.int 8⟩] [1,0])).operands == [1,0]
#guard (operation (invalid [⟨2,tensorTy [3] (.int 1)⟩] [1,0])).operands == [1,0]
#guard (operation (invalid [⟨2,.int 1⟩] [1])).operands == [1]
#guard (operation (invalid [⟨2,.int 1⟩] [1,0] [.mk []])).operands == [1,0]
#guard (operation (invalid [⟨2,.int 1⟩] [1,0] [] [0])).operands == [1,0]
#guard (candidate (.mk "arith.cmpi" [⟨2,.int 1⟩] [1,0]
  [] [("predicate",.int 10 (.int 64))] [] [])).isNone
#guard (candidate (.mk "arith.cmpi" [⟨2,.int 1⟩] [1,0]
  [] [("predicate",.int (-1) (.int 64))] [] [])).isNone
#guard (operation (.mk "arith.cmpf" [⟨2,.int 1⟩] [1,0]
  [] [("predicate",.int 2 (.int 64))] [] [])).operands == [1,0]

def signature (result : Outcome (List RVal)) : String × Option (List (List (Option Int))) :=
  match result with
  | .ub => ("ub",none)
  | .stuck reason => (reason,none)
  | .ok values => ("ok",readInts (.ok values))
def run (p : Predicate) (env : Env) (normalized : Bool) :=
  signature (evalOps opSem ctx (fun _ _ => .stuck "no calls") env
    [if normalized then operation (cmp p 1 0) else cmp p 1 0,ret])
def bits (w n : Nat) : RVal := .scalar (.some (.bits w (BitVec.ofNat w n)))
def poison : RVal := .scalar .poison
def flt : RVal := .scalar (.some (.flt .f32 0))
def tensor (size : Nat) : RVal := .tensor (Tensor.canon [size] (.int 4)
  (fun i => .some (.bits 4 (BitVec.ofNat 4 ((i.headD 0) + 3)))))

-- Exhaustive initialized small-width pairs, including mathematical zero-width values.
#guard predicates.all fun p => [0,1,4].all fun w =>
  (List.range (2^w)).all fun x => (List.range (2^w)).all fun y =>
    run p [(0,bits w x),(1,bits w y)] false == run p [(0,bits w x),(1,bits w y)] true
#guard predicates.all fun p =>
  [poison,flt,bits 4 3,bits 8 3,tensor 2,tensor 3].all fun x =>
    [poison,flt,bits 4 3,bits 8 3,tensor 2,tensor 3].all fun y =>
      run p [(0,x),(1,y)] false == run p [(0,x),(1,y)] true
#guard predicates.all fun p => [[],[(0,bits 4 3)],[(1,bits 4 3)]].all fun env =>
  run p env false == run p env true
-- Current bindings, rather than older shadowed payloads, determine the comparison.
#guard predicates.all fun p =>
  run p [(0,bits 4 7),(0,poison),(1,bits 4 3)] false ==
    run p [(0,bits 4 7),(0,poison),(1,bits 4 3)] true

end MerlinVerify.Tests.IntegerCompareOrder
