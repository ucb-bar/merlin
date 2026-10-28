/- Decode and establish numerical facts for actual initialized scalar results. -/
import MerlinVerify.Check.FloatFacts
import MerlinVerify.Check.DefinedRules

namespace MerlinVerify.Check.FloatFacts.Scalar

open MerlinVerify Rules Definedness MerlinVerify.Sem.Ops

def fact (ty : Ty) (numeric : FiniteFacts.Class) : Refined.Fact Info :=
  ⟨⟨.scalar ty, true⟩, fromClass numeric⟩

def finite (info : Info) : Bool := FiniteFacts.Implication.classes info.numeric .finite

theorem finite_value {ib : Nat} {info : Info} {value : RVal}
    (checked : finite info = true)
    (holds : (⟨⟨.scalar (.float .f32), true⟩, info⟩ : Refined.Fact Info).Holds model ib value) :
    ∃ bits, value = .scalar (.some (.flt .f32 bits)) ∧ W8A8FiniteLaws.Finite bits := by
  obtain ⟨bits, rfl⟩ := initialized_float holds.1
  have numerical := FiniteFacts.Implication.classes_sound checked holds.2.1
  refine ⟨bits, rfl, ?_⟩
  simpa [FiniteFacts.Holds, FiniteFacts.pointCheck, PolicyW8A8FiniteDomain.pointCheck, Dequant.float32] using numerical

theorem normal_value {ib : Nat} {info : Info} {value : RVal}
    (checked : info.numeric = .normal)
    (holds : (⟨⟨.scalar (.float .f32), true⟩, info⟩ : Refined.Fact Info).Holds model ib value) :
    ∃ bits, value = .scalar (.some (.flt .f32 bits)) ∧ W8A8FiniteLaws.scaleCheck bits = true := by
  obtain ⟨bits, rfl⟩ := initialized_float holds.1
  refine ⟨bits, rfl, ?_⟩
  simpa [checked, FiniteFacts.model, FiniteFacts.Holds, FiniteFacts.pointCheck, Dequant.float32] using holds.2.1

theorem float_result {ib : Nat} {numeric : FiniteFacts.Class} {bits : FBits .f32}
    (checked : FiniteFacts.pointCheck numeric (.some (.flt .f32 bits)) = true) :
    (fact (.float .f32) numeric).Holds model ib (.scalar (.some (.flt .f32 bits))) := by
  refine ⟨⟨by simp [fact, RVal.HasTy, Val.HasTy, Scalar.conforms], by simp [RVal.Defined]⟩, ?_, trivial⟩
  cases numeric with
  | any => trivial
  | finite => exact checked
  | normal => exact checked

theorem finite_result {ib : Nat} {bits : FBits .f32} (finite : W8A8FiniteLaws.Finite bits) :
    (fact (.float .f32) .finite).Holds model ib (.scalar (.some (.flt .f32 bits))) :=
  float_result (by simpa [FiniteFacts.pointCheck, PolicyW8A8FiniteDomain.pointCheck, Dequant.float32] using finite)

theorem normal_result {ib : Nat} {bits : FBits .f32} (normal : W8A8FiniteLaws.scaleCheck bits = true) :
    (fact (.float .f32) .normal).Holds model ib (.scalar (.some (.flt .f32 bits))) :=
  float_result (by simpa [FiniteFacts.pointCheck, Dequant.float32] using normal)

theorem byte_result {ib : Nat} (byte : BitVec 8) :
    (fact (.int 8) .any).Holds model ib (.scalar (.some (.bits 8 byte))) := by
  exact ⟨⟨by simp [fact, RVal.HasTy, Val.HasTy, Scalar.conforms], by simp [RVal.Defined]⟩, trivial, trivial⟩

#print axioms finite_value
#print axioms normal_value
#print axioms float_result
#print axioms finite_result
#print axioms normal_result
#print axioms byte_result

end MerlinVerify.Check.FloatFacts.Scalar
