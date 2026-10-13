import MerlinVerify.Sem.Ops
import MerlinVerify.Check.Types

/-- `merlin-lean-audit`: guardrail audit (design §8.6). Implemented by the guardrails owner. -/
def main (_args : List String) : IO UInt32 := do
  IO.eprintln "merlin-lean-audit: not implemented"
  return 2
