import MerlinVerify.Sem.Ops
import MerlinVerify.Check.Types

/-- `merlin-lean`: checker driver. Subcommands are added by their owners (design §8.3). -/
def main (args : List String) : IO UInt32 := do
  IO.eprintln s!"merlin-lean: no subcommand implemented for {args}"
  return 2
