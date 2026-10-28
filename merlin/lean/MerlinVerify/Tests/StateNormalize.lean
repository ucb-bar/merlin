/- Both existing state interfaces start empty; captured annotations create no producer facts. -/
import MerlinVerify.Tests.ConstantDead
import MerlinVerify.Tests.FromElementsIndex

namespace MerlinVerify.Tests.StateNormalize

open MerlinVerify MerlinVerify.Check MerlinVerify.Tests.Semantics

#guard Check.FromElementsSelect.family.initial.elements.isEmpty
#guard Check.FromElementsSelect.family.initial.constants.isEmpty
#guard (TailNormalize.stateFamily Check.ConstantAlias.family).initial.isEmpty

#guard ((Check.ConstantAlias.base 64).run 0 [(1, .scalar i32)] []
  [ConstantAlias.b, ret [2, 1]] [.scalar i32, .scalar i32]).map
  (List.map Op.operands) == some [[], [2, 1]]
#guard ((Check.FromElementsSelect.base 64).run 0
  [(3, .tensor [2] i32), (0, .scalar i32), (1, .scalar i32)] []
  [FromElementsSelect.index 1, FromElementsSelect.load, ret [6]] [.scalar i32]).map
  (List.map Op.name) == some ["arith.constant", "tensor.extract", "func.return"]

end MerlinVerify.Tests.StateNormalize
