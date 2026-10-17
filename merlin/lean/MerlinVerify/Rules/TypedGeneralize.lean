/- Rank-dependent named-operation expansions on the same checked typed domain. -/
import MerlinVerify.Check.TypedComposition
import MerlinVerify.Rules.FillGeneralize
import MerlinVerify.Rules.MapGeneralize

namespace MerlinVerify.Rules.TypedGeneralize

def expansion : Check.TypedExpansion :=
  MapGeneralize.expansion.orElse FillGeneralize.expansion

end MerlinVerify.Rules.TypedGeneralize
