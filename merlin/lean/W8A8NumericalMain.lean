/- Separate candidate entry; shared Main and the historical qualified entry are unchanged. -/
import MerlinVerify.Driver.W8A8Numerical

def main (args : List String) : IO UInt32 := do
  let before :: after :: contracts :: width :: scope :: suffix := args | return 2
  MerlinVerify.Driver.W8A8Numerical.run before after contracts width scope suffix
