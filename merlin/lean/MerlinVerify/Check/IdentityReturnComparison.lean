/- Adapt the bounded return-only refinement proof to the shared typed comparison interface. -/
import MerlinVerify.Check.IdentityReturn
import MerlinVerify.Check.TypedComparison

namespace MerlinVerify.Check.IdentityReturn

def comparison : TypedComparison where
  run := check
  sound _ _ _ _ _ ctx callS callT envS envT hr he hc hi _ :=
    RefinesC.of_refines (check_sound hr ctx callS callT envS envT he hc hi)

end MerlinVerify.Check.IdentityReturn
