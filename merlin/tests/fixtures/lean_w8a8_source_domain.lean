/- Guard the checker on poison, malformed runtime observations and shape/type mismatches.
This total test model is deliberately simple, not an IEEE interpretation. -/
import MerlinVerify.Check.PolicyW8A8SourceDomain

open MerlinVerify MerlinVerify.Check MerlinVerify.Rules

def base : FloatModel where
  op _ _ _ := 0
  cmp _ _ _ _ := false
  convert _ _ _ _ := 0
  toInt _ _ _ _ := none
  ofInt _ _ _ _ := 0

def input (shape : List Nat) (elt : Ty) (value : Val) : RVal :=
  .tensor ⟨shape, elt, fun _ => value⟩

def scale : RVal := input [] (.float .f32) (.some (.flt .f32 0x3f800000#32))
def zero : RVal := input [] (.int 64) (.some (.bits 64 0))
def check (shape : List Nat) (args : List RVal) : Bool := PolicyW8A8SourceDomain.check 64 base shape args

#guard check [2] [input [2] (.float .f32) (.some (.flt .f32 0)), scale, zero]
#guard !check [2] [input [2] (.float .f32) .poison, scale, zero]
#guard !check [2] [input [2] (.float .f32) (.some (.bits 32 0)), scale, zero]
#guard !check [2] [input [2] (.int 32) (.some (.flt .f32 0)), scale, zero]
#guard !check [2] [input [1] (.float .f32) (.some (.flt .f32 0)), scale, zero]
#guard !check [2] [input [2] (.float .f32) (.some (.flt .f32 0)), input [] (.float .f32) .poison, zero]
#guard !check [2] [input [2] (.float .f32) (.some (.flt .f32 0)), scale, input [] (.int 64) .poison]
#guard !check [2] [input [2] (.float .f32) (.some (.flt .f32 0)), scale, input [] (.int 32) (.some (.bits 32 0))]
#guard !check [2] [input [2] (.float .f32) (.some (.flt .f32 0)), scale]
#guard check [0] [input [0] (.float .f32) .poison, scale, zero]

def main : IO Unit := IO.println "runtime domain guards passed"
