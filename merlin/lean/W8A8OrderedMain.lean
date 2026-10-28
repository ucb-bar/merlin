/- Separate entry point for checked ordered floating-contraction bounds. -/
import MerlinVerify.Driver.W8A8Ordered

def main (args : List String) : IO UInt32 := do
  let [before, after, contracts, caps, width, scope] := args | return 2
  MerlinVerify.Driver.W8A8Ordered.run before after contracts caps width scope
