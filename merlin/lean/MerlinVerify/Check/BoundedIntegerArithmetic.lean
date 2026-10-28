/- Initialized integer operations retain only producer-proved bounds. The arithmetic
width comes from the actual integer operands/results, independently of the context index
width. Existing singleton/addition proofs and the full no-wrap multiplication rule are reused. -/
import MerlinVerify.Check.BoundedFloatFacts
import MerlinVerify.Check.IndexArithmetic
import MerlinVerify.Check.IndexRangeMul
import MerlinVerify.Check.IndexRangeEmpty

namespace MerlinVerify.Check.BoundedFloatFacts.IntegerArithmetic

open MerlinVerify Definedness Sem.Ops
open Definedness.IndexArithmetic (Kind kind? kind?_name)

def asIndex (range : Option IndexRanges.Interval) : IndexRanges.Fact := ⟨⟨.scalar .index, true⟩, range⟩

theorem asIndex_holds {width : Nat} (x : BitVec width) (range : Option IndexRanges.Interval)
    (holds : IndexRanges.model.holds range (.scalar (.some (.bits width x)))) :
    (asIndex range).Holds IndexRanges.model width (.scalar (.some (.bits width x))) :=
  ⟨⟨by simp [asIndex, RVal.HasTy, Val.HasTy, Scalar.conforms], by simp [RVal.Defined]⟩, holds⟩

def evidence (width : Nat) (kind : Kind) (left right : Option IndexRanges.Interval) : Option IndexRanges.Interval :=
  if IndexRangeEmpty.check left || IndexRangeEmpty.check right then some ⟨0, 0⟩ else
    (Definedness.IndexArithmetic.evidence width kind (asIndex left) (asIndex right)).orElse fun _ =>
      if kind = .mul then IndexRangeMul.evidence width left right else none

theorem evidence_sound {width : Nat} (kind : Kind) (left right : Option IndexRanges.Interval) (x y : BitVec width)
    (hx : IndexRanges.model.holds left (.scalar (.some (.bits width x))))
    (hy : IndexRanges.model.holds right (.scalar (.some (.bits width y)))) :
    IndexRanges.model.holds (evidence width kind left right) (.scalar (.some (.bits width (kind.bits x y)))) := by
  unfold evidence
  split
  · next empty =>
    simp only [Bool.or_eq_true] at empty
    rcases empty with leftEmpty | rightEmpty
    · exact (IndexRangeEmpty.not_holds leftEmpty _ hx).elim
    · exact (IndexRangeEmpty.not_holds rightEmpty _ hy).elim
  · cases selected : Definedness.IndexArithmetic.evidence width kind (asIndex left) (asIndex right) with
    | some range =>
      simp only [Option.orElse_some]
      simpa only [selected] using Definedness.IndexArithmetic.evidence_sound kind (asIndex left) (asIndex right)
        x y (asIndex_holds x left hx) (asIndex_holds y right hy)
    | none =>
      simp only [Option.orElse_none]
      split
      · next multiplied =>
        subst kind
        exact IndexRangeMul.evidence_sound left right x y hx hy
      · trivial

def rule (_ : Refined.FEnv Info) (op : Op) (facts : List (Refined.Fact Info)) : Option (List (Refined.Fact Info)) :=
  match facts, op.results, kind? op.name with
  | [⟨⟨.scalar (.int width), true⟩, left⟩, ⟨⟨.scalar (.int other), true⟩, right⟩], [result], some kind =>
    if unmodelledAttr? op.blind = none ∧ dialectOf op.name = "arith" ∧ other = width ∧ result.ty = .int width ∧
        op.regions = [] ∧ overflowFlags op = [] ∧ Arith.fastmathKey op = "0" then
      some [⟨⟨.scalar (.int width), true⟩, ⟨FloatFacts.model.empty, evidence width kind left.indices right.indices⟩⟩]
    else none
  | _, _, _ => none

theorem rule_sound (ib : Nat) : ∀ Γ op facts results, rule Γ op facts = some results → ∀ ctx call env args,
    ctx.indexBits = ib → Refined.EnvHolds model ib Γ env → Refined.FactsHold model ib args facts → ∃ values,
      opSem ctx call op args (evalRegions opSem call env op.regions) = .ok values ∧ Refined.FactsHold model ib values results := by
  intro Γ op facts results checked ctx call env args _ _ inputs
  unfold rule at checked
  split at checked
  · next width left other right result kind resultFound selected =>
    split at checked
    · next conditions =>
      obtain ⟨modelled, dialect, same, typed, regions, flags, _⟩ := conditions
      subst other
      cases checked
      obtain _ | ⟨head, rest⟩ := inputs
      obtain _ | ⟨next, tail⟩ := rest
      cases tail
      obtain ⟨x, rfl⟩ := initialized_int head.1
      obtain ⟨y, rfl⟩ := initialized_int next.1
      have rangeX : IndexRanges.model.holds left.indices (.scalar (.some (.bits width x))) := by
        cases range : left.indices with
        | none => trivial
        | some _ => simpa [IndexRanges.model, Uniform, range] using head.2.2
      have rangeY : IndexRanges.model.holds right.indices (.scalar (.some (.bits width y))) := by
        cases range : right.indices with
        | none => trivial
        | some _ => simpa [IndexRanges.model, Uniform, range] using next.2.2
      have resultHolds : (⟨⟨.scalar (.int width), true⟩, evidence width kind left.indices right.indices⟩ : IndexRanges.Fact).Holds
          IndexRanges.model ib (.scalar (.some (.bits width (kind.bits x y)))) :=
        ⟨⟨by simp [RVal.HasTy, Val.HasTy, Scalar.conforms], by simp [RVal.Defined]⟩,
          evidence_sound kind left.indices right.indices x y rangeX rangeY⟩
      refine ⟨[.scalar (.some (.bits width (kind.bits x y)))], ?_,
        .cons ((fromScalarRanges ib).sound _ _ resultHolds) .nil⟩
      have named := kind?_name selected
      have output : Arith.resultElt op.blind = some (.int width) := by
        simp [Arith.resultElt, Op.blind_results, resultFound, typed]
      have noFlags : overflowFlags op.blind = [] := by simpa [overflowFlags, attrEnum?] using flags
      rw [regions, evalRegions, opSem_modelled modelled]
      simp only [dispatch, Op.blind_name, dialect]
      cases kind <;> simp only [Kind.name] at named
      all_goals
        simp only [Arith.sem, Op.blind_name, named, Arith.binOp, Arith.pureN, output]
        rw [show [.scalar (.some (.bits width x)), .scalar (.some (.bits width y))] =
          ([.some (.bits width x), .some (.bits width y)] : List Val).map RVal.scalar from rfl,
          liftEager_scalars]
        simp [Arith.intBin, Arith.arithWithFlags, noFlags, Arith.bv, Kind.bits]
    · cases checked
  · cases checked

def book (ib : Nat) : Refined.Book model opSem ib where
  rule := rule
  sound := rule_sound ib

#print axioms asIndex_holds
#print axioms evidence_sound
#print axioms rule_sound
#print axioms book

end MerlinVerify.Check.BoundedFloatFacts.IntegerArithmetic
