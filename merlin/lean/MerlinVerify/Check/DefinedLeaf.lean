/- Assemble independently proved scalar and tensor leaf families once for all body checkers. -/
import MerlinVerify.Check.DefinedFloating
import MerlinVerify.Check.DefinedCarrier
import MerlinVerify.Check.DefinedConversions
import MerlinVerify.Check.DefinedBitcast
import MerlinVerify.Check.DefinedFromElements

namespace MerlinVerify.Check.Definedness

def sharedLeafBook (ib : Nat) : Book Sem.Ops.opSem ib :=
  (((controlLeafBook ib).orElse (floatingBook ib)).orElse (Carrier.book ib)).orElse
    (((DefinedConversions.book ib).orElse (DefinedBitcast.book ib)).orElse (FromElements.book ib))

end MerlinVerify.Check.Definedness
