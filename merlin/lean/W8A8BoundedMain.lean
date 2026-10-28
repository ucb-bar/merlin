/- Standalone bounded-policy entry point. Keep shared Main independent while its other
verification routes are under concurrent development. This entry never approves a draft. -/
import MerlinVerify.Driver.W8A8Bounded

def main (args : List String) : IO UInt32 := do
  let before :: after :: contracts :: width :: scope :: suffix := args | return 2
  MerlinVerify.Driver.W8A8Bounded.run before after contracts width scope suffix
