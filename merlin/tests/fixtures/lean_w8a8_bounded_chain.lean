/- Test the actual bounded-domain reporting adapter without changing the shared Main. -/
import MerlinVerify.Driver.W8A8Bounded

def main (args : List String) : IO UInt32 := do
  let before :: after :: contracts :: width :: scope :: suffix := args | return 2
  MerlinVerify.Driver.W8A8Bounded.run before after contracts width scope suffix
