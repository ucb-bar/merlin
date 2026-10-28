/- Private entry for complete W8A8 draft-reference correspondence. -/
import MerlinVerify.Driver.W8A8Correspondence

def main (args : List String) : IO UInt32 := do
  let before :: after :: width :: scope :: suffix := args | return 2
  MerlinVerify.Driver.W8A8Correspondence.run before after width scope suffix
