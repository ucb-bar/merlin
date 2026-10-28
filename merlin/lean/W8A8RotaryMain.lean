/- Isolated entry for the per-argument numerical-bound candidate. -/
import MerlinVerify.Driver.W8A8Rotary

def main (args : List String) : IO UInt32 := do
  let [before, after, contracts, caps, width, scope] := args | return 2
  MerlinVerify.Driver.W8A8Rotary.run before after contracts caps width scope
