/- Separate development driver for the named mathematical trig interpretation. -/
import MerlinVerify.Driver.W8A8Trig

def main (args : List String) : IO UInt32 := do
  let [before, after, contracts, caps, width, scope] := args | return 2
  MerlinVerify.Driver.W8A8Trig.run before after contracts caps width scope
