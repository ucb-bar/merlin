/- Structural AST height supplies a completeness budget; no soundness claim relies on its size. -/
import MerlinVerify.Core.AST

namespace MerlinVerify.Check.RegionDepth

mutual
  def op : Op → Nat
    | .mk _ _ _ _ _ rs _ => 1 + regions rs

  def regions : List Region → Nat
    | [] => 0
    | .mk bs :: rest => max (blocks bs) (regions rest)

  def blocks : List Block → Nat
    | [] => 0
    | .mk _ ss :: rest => max (ops ss) (blocks rest)

  def ops : List Op → Nat
    | [] => 0
    | head :: rest => max (op head) (ops rest)
end

end MerlinVerify.Check.RegionDepth
