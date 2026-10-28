/- Connect the draft source recognizer's actual reduction extent to exact i32 accumulation.
This is an arithmetic side condition, independent of policy approval or source definedness. -/
import MerlinVerify.Rules.W8A8Reference
import MerlinVerify.Rules.W8A8Accumulation
import MerlinVerify.Rules.W8A8Matmul

namespace MerlinVerify.Rules.W8A8ReferenceBounds

open MerlinVerify

theorem view_bound {args : List Value} {seen : List Op} {source : Op} {view : W8A8Reference.View}
    (accepted : W8A8Reference.view? args seen source = some view) :
    (view.k : Int) * 16384 < 2147483648 := by
  unfold W8A8Reference.view? at accepted
  repeat' first
    | simp only [bind, Option.bind, pure, Pure.pure] at accepted
    | split at accepted
    | contradiction
  all_goals simp_all only [Option.some.injEq]
  all_goals cases accepted
  all_goals simp only [Bool.or_eq_true, decide_eq_true_eq, not_or] at *
  all_goals omega

theorem every_prefix_exact {args : List Value} {seen : List Op} {source : Op} {view : W8A8Reference.View}
    (accepted : W8A8Reference.view? args seen source = some view)
    (pairs : List W8A8Accumulation.Pair) (count : pairs.length = view.k) (length : Nat) :
    (W8A8Accumulation.accumulate (pairs.take length)).toInt = W8A8Accumulation.dot (pairs.take length) := by
  apply W8A8Accumulation.every_prefix_exact
  rw [count]
  exact view_bound accepted

/-- The mapped cell trace has exactly the source-derived reduction length, including empty
reductions. The full signed-byte range is allowed; no symmetric-quantization hint is used. -/
theorem mapped_prefix_exact {args : List Value} {seen : List Op} {source : Op} {view : W8A8Reference.View}
    (accepted : W8A8Reference.view? args seen source = some view)
    (left right : Idx → BitVec 8) (cell : Idx) (bounds : Idx.InBounds [view.m, view.n] cell)
    (length : Nat) :
    let pairs := W8A8IntegerTensor.pairs
      (W8A8IntegerTensor.coordinate [view.m, view.n] W8A8Matmul.outMap)
      (fun point => left (W8A8IntegerTensor.coordinate [view.m, view.k] W8A8Matmul.leftMap point))
      (fun point => right (W8A8IntegerTensor.coordinate [view.k, view.n] W8A8Matmul.rightMap point))
      (allPoints [view.m, view.n, view.k]) cell
    (W8A8Accumulation.accumulate (pairs.take length)).toInt = W8A8Accumulation.dot (pairs.take length) := by
  apply every_prefix_exact accepted
  exact W8A8Matmul.mapped_pair_count view.m view.n view.k _ _ cell bounds

#print axioms view_bound
#print axioms every_prefix_exact
#print axioms mapped_prefix_exact

end MerlinVerify.Rules.W8A8ReferenceBounds
