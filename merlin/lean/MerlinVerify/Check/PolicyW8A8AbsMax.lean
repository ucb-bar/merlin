/- Abs/max stage checker. Group dimensions must be supplied by the source policy contract,
not inferred from the candidate's proposed axis. No complete C5 receipt is issued here. -/
import MerlinVerify.Rules.W8A8AbsMax
import MerlinVerify.Check.PolicyW8A8Geometry

namespace MerlinVerify.Check.PolicyW8A8AbsMax

open MerlinVerify MerlinVerify.Sem.Ops.Linalg Rules W8A8TensorOperands W8A8Scalar W8A8FloatTensor
open PolicyW8A8Geometry

def iterators (rank : Nat) (kept : List Nat) : List IterKind :=
  (List.range rank).map (fun d => if kept.contains d then .parallel else .reduction)

def Correct (shape kept : List Nat) (maps : List AffineMap) (iters : List IterKind)
    (actual : Region) : Prop :=
  ∀ ctx call env get initial,
    genericSem ctx maps iters [(floatInput shape get).value]
      [.tensor (W8A8TensorTrace.pack (parameterShape shape kept) (.float .f32) floatValue initial)]
      (fun c => evalRegion Sem.Ops.opSem c call env actual) =
      .ok [.tensor (Tensor.canon (parameterShape shape kept) (.float .f32) (fun cell => floatValue
        (W8A8TensorTrace.accumulate
          (coordinate (parameterShape shape kept) (parameterMap shape.length kept))
          (W8A8AbsMax.step ctx.M get) initial (allPoints shape) cell)))]

def check (shape kept : List Nat) (maps : List AffineMap) (iters : List IterKind)
    (actual : Region) : Bool :=
  checkDimensions shape kept &&
  decide (maps = [identityMap shape.length, parameterMap shape.length kept]) &&
  decide (iters = iterators shape.length kept) &&
  (structuredRanges? maps iters.length
    [.tensor shape (.float .f32), .tensor (parameterShape shape kept) (.float .f32)] == some shape) &&
  PolicyW8A8Scalar.checkAbsMax actual

theorem check_sound {shape kept : List Nat} {maps : List AffineMap} {iters : List IterKind}
    {actual : Region} (h : check shape kept maps iters actual = true) :
    Correct shape kept maps iters actual := by
  have hh := h
  simp only [check, Bool.and_eq_true] at hh
  have hm := of_decide_eq_true hh.1.1.1.2
  have geometry := structuredRanges?_spec (eq_of_beq hh.1.2)
  intro ctx call env get initial
  subst maps
  exact W8A8AbsMax.generic_sound geometry hh.2 call env get initial

#print axioms check_sound

end MerlinVerify.Check.PolicyW8A8AbsMax
