/- Separate entry for the explicit-model numerical candidate. -/
import MerlinVerify.Driver.W8A8IEEE

def main (args : List String) : IO UInt32 := do
  let [before, after, contracts, caps, width, scope] := args | return 2
  MerlinVerify.Driver.W8A8IEEE.run before after contracts caps width scope
