/- Integer min/max commutation covers every width and shape; attributes and operation names stay fixed. -/
import MerlinVerify.Tests.Commutative
import MerlinVerify.Rules.IntegerOrder

namespace MerlinVerify.Tests.IntegerMinMax

open MerlinVerify MerlinVerify.Check MerlinVerify.Tests.Commutative
open MerlinVerify.Sem.Ops

def names := ["arith.minui","arith.maxui","arith.minsi","arith.maxsi"]

#guard names.all fun n => moduleCheck (moduleOf n false (.int 1)) (moduleOf n true (.int 1))
#guard names.all fun n => moduleCheck (moduleOf n false (.int 13)) (moduleOf n true (.int 13))
#guard names.all fun n => moduleCheck (moduleOf n false .index) (moduleOf n true .index)
#guard names.all fun n => moduleCheck (moduleOf n false (.tensor [some 2,some 3] (.int 13) none))
  (moduleOf n true (.tensor [some 2,some 3] (.int 13) none))
#guard names.all fun n => moduleCheck (moduleOf n false (.tensor [some 0] (.int 13) none))
  (moduleOf n true (.tensor [some 0] (.int 13) none))
#guard !moduleCheck (moduleOf "arith.minui" false) (moduleOf "arith.maxui" true)
#guard !moduleCheck (moduleOf "arith.minui" false) (moduleOf "arith.minsi" true)
#guard !moduleCheck (moduleOf "arith.maxsi" false) (moduleOf "arith.maxui" true)
#guard !moduleCheck (moduleOf "arith.minsi" false)
  (moduleOf "arith.minsi" true (.int 32) [("unknown.meaning",.unit)])
#guard names.all fun n => !moduleCheck (moduleOf n false (.int 32) [("isExact",.unit)])
  (moduleOf n true (.int 32) [("isExact",.unit)])
#guard names.all fun n => !moduleCheck (moduleOf n false (.int 32) [("overflowFlags",.enum "arith" "overflow" "nsw")])
  (moduleOf n true (.int 32) [("overflowFlags",.enum "arith" "overflow" "nsw")])

-- All four selectors, every 4-bit input pair, signed extrema and ties, without execution shortcuts.
#guard (List.range 16).all fun a => (List.range 16).all fun b =>
  let x := BitVec.ofNat 4 a
  let y := BitVec.ofNat 4 b
  (if x.ult y then x else y) == (if y.ult x then y else x) &&
  (if x.ult y then y else x) == (if y.ult x then x else y) &&
  (if x.slt y then x else y) == (if y.slt x then y else x) &&
  (if x.slt y then y else x) == (if y.slt x then x else y)

end MerlinVerify.Tests.IntegerMinMax
