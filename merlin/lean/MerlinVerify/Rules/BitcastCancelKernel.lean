/- Raw-bit identities for the scalar bitcast domains already modelled by Arith.sem.
No numerical float operation, FloatModel law, or float-to-float bitcast is involved. -/
import MerlinVerify.Rules.ScalarBitcast

namespace MerlinVerify.Rules.BitcastCancel

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops
open ScalarBitcast

/-- A total mathematical kernel. Executable uses separately check actual compatible types;
unsupported cases here are never used as evidence of successful execution. -/
def reinterpret (dst : Ty) : Val → Val
  | .poison => .poison
  | .some (.bits _ bits) => match dst with
    | .int width => .some (.bits width (bits.setWidth width))
    | .float format => .some (.flt format (bits.setWidth format.width))
    | _ => .poison
  | .some (.flt _ bits) => match dst with
    | .int width => .some (.bits width (bits.setWidth width))
    | _ => .poison

/-- Reinterpreting a well-typed scalar as its same integer type preserves its exact payload. -/
theorem integer_identity {ib width : Nat} {value : Val}
    (ht : Val.HasTy ib (.int width) value) : reinterpret (.int width) value = value := by
  rcases hasTy_intWidth (t := .int width) rfl ht with rfl | ⟨bits, rfl⟩
  · rfl
  · simp [reinterpret]

/-- Both supported cross-kind directions are involutions, including poison and every NaN bit.
The returned float format is the original format, not merely another format of equal width. -/
theorem roundtrip {ib : Nat} {src mid : Ty} {value : Val}
    (hc : Compatible src mid) (ht : Val.HasTy ib src value) :
    reinterpret src (reinterpret mid value) = value := by
  cases src <;> cases mid <;> simp only [Compatible] at hc
  · next width width' =>
    obtain ⟨rfl, _⟩ := hc
    rw [integer_identity ht, integer_identity ht]
  · next width format =>
    subst width
    rcases hasTy_intWidth (t := .int format.width) rfl ht with rfl | ⟨bits, rfl⟩
    · rfl
    · simp [reinterpret]
  · next format width =>
    subst width
    rcases hasTy_float ht with rfl | ⟨bits, rfl⟩
    · rfl
    · simp [reinterpret]

/-- The actual dispatcher returns precisely the raw-bit kernel on the checked runtime domain. -/
theorem meaning_exact {s : Op} {src dst : Ty} (hn : s.name = "arith.bitcast")
    (hm : unmodelledAttr? s.blind = none) (hr : Arith.resultElt s.blind = some dst)
    (hc : Compatible src dst) (ctx : Ctx) (call : CallSem) (regions : List (Ctx → RegionSem))
    {value : Val} (ht : Val.HasTy ctx.indexBits src value) :
    opSem ctx call s [.scalar value] regions = .ok [.scalar (reinterpret dst value)] := by
  have hd : dialectOf s.name = "arith" := by rw [hn]; cbv
  rw [opSem_modelled hm]
  simp only [dispatch, Op.blind_name, hd]
  cases src <;> cases dst <;> simp only [Compatible] at hc
  · next width width' =>
    obtain ⟨rfl, _⟩ := hc
    rcases hasTy_intWidth (t := .int width) rfl ht with rfl | ⟨bits, rfl⟩
    all_goals
      simp only [Arith.sem, Op.blind_name, hn, hr, Arith.unOp, Arith.pureN]
      rw [show [.scalar _] = ([_] : List Val).map RVal.scalar from rfl, liftEager_scalars]
      simp [reinterpret, Arith.bv]
  · next width format =>
    subst width
    rcases hasTy_intWidth (t := .int format.width) rfl ht with rfl | ⟨bits, rfl⟩
    all_goals
      simp only [Arith.sem, Op.blind_name, hn, hr, Arith.unOp, Arith.pureN]
      rw [show [.scalar _] = ([_] : List Val).map RVal.scalar from rfl, liftEager_scalars]
      simp [reinterpret]
  · next format width =>
    subst width
    rcases hasTy_float ht with rfl | ⟨bits, rfl⟩
    all_goals
      simp only [Arith.sem, Op.blind_name, hn, hr, Arith.unOp, Arith.pureN]
      rw [show [.scalar _] = ([_] : List Val).map RVal.scalar from rfl, liftEager_scalars]
      simp [reinterpret, Arith.bv]

end MerlinVerify.Rules.BitcastCancel
