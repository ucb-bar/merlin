/- Additive evaluator locality lemmas for frontend policy simulation. -/
import MerlinVerify.Sem.Ops

namespace MerlinVerify.Rules.FrontendBoolContext
open MerlinVerify

def pin (ctx : Ctx) (body : Ctx → RegionSem) : Ctx → RegionSem :=
  fun inner => body { ctx with loopIndex := inner.loopIndex }

/-- Structured regions inherit the caller's model, external environment and index width;
only their loop index changes. This follows from the frozen definitions, not an assumption. -/
theorem opSem_pin (ctx : Ctx) (call : CallSem) (op : Op) (args : List RVal)
    (regions : List (Ctx → RegionSem)) :
    Sem.Ops.opSem ctx call op args (regions.map (pin ctx)) =
      Sem.Ops.opSem ctx call op args regions := by
  simp only [Sem.Ops.opSem]
  split
  · rfl
  · simp only [Sem.Ops.dispatch]
    split <;> try rfl
    · split
      · cases hs : Sem.Ops.Linalg.splitInsOuts op.blind args <;>
          cases hm : attrAffineMaps? op.blind "indexing_maps" <;>
          cases hi : Sem.Ops.Linalg.iterKinds? op.blind
        all_goals simp only [Sem.Ops.Linalg.sem]
        all_goals cases regions with
        | nil => rfl
        | cons body rest =>
          cases rest with
          | nil => simp only [List.map_cons, List.map_nil]; repeat' first | rfl | (split <;> simp_all <;> subst_vars)
          | cons next rest => simp only [List.map_cons]; split <;> simp_all
      · cases regions with
        | nil => rfl
        | cons body rest =>
          cases rest with
          | nil =>
            rfl
          | cons next rest => rfl
    · simp [Sem.Ops.Dequant.sem]

/-- Pointwise agreement on the contexts the evaluator can actually supply is sufficient. -/
theorem opSem_regions (ctx : Ctx) (call : CallSem) (op : Op) (args : List RVal)
    (left right : List (Ctx → RegionSem))
    (h : left.map (pin ctx) = right.map (pin ctx)) :
    Sem.Ops.opSem ctx call op args left = Sem.Ops.opSem ctx call op args right := by
  rw [← opSem_pin ctx call op args left, h, opSem_pin]

end MerlinVerify.Rules.FrontendBoolContext
