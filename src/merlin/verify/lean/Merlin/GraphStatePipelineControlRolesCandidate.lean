/- Finite occurrence binding of selected bounded controls. Retagging changes
only an explicitly named opaque identity; every actual operand/result, pure
instruction, nested loop, terminator and continuation is retained exactly. -/
import Merlin.GraphStateScalarContract

namespace Merlin.GraphState.Pipeline.ControlRolesCandidate
set_option autoImplicit false
open Scalar

structure Mark where
  occurrence : Nat
  original : Nat
  selected : Nat
  deriving DecidableEq, BEq, Repr

def identity (marks : List Mark) (occurrence original : Nat) : Option Nat :=
  match marks.find? (fun mark => mark.occurrence == occurrence) with
  | none => some original
  | some mark => if mark.original = original then some mark.selected else none

def applyFrom (marks : List Mark) : Program → Nat → Option (Program × Nat)
  | .ret slots, next => some (.ret slots,next)
  | .pure op arguments later, next => do
      let result ← applyFrom marks later next
      return (.pure op arguments result.1,result.2)
  | .opaque original arguments outputs later, next => do
      let selected ← identity marks next original
      let result ← applyFrom marks later (next+1)
      return (.opaque selected arguments outputs result.1,result.2)
  | .loop lower upper stride initial carries body later, next => do
      let inside ← applyFrom marks body next
      let result ← applyFrom marks later inside.2
      return (.loop lower upper stride initial carries inside.1 result.1,result.2)

def apply (marks : List Mark) (program : Program) : Option Program := do
  let result ← applyFrom marks program 0
  if marks.all (fun mark => decide (mark.occurrence < result.2)) then
    return result.1
  else none

end Merlin.GraphState.Pipeline.ControlRolesCandidate
