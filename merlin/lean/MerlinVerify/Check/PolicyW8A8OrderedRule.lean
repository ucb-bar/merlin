/- Checked numerical facts for actual f32 matmul, batch_matmul and generic contractions.
Every bound is derived from current operand facts, shapes, maps, region and reduction size. -/
import MerlinVerify.Check.PolicyW8A8OrderedOp
import MerlinVerify.Check.PolicyW8A8IEEEQualified

namespace MerlinVerify.Check.PolicyW8A8OrderedRule

open MerlinVerify Rules Definedness Sem.Ops Sem.Ops.Linalg PolicyW8A8RotaryFacts
open PolicyW8A8OrderedRefined

local notation "sourceSem" => W8A8SourceSemantics.opSem

def allowed (ctx : Ctx) (_ : CallSem) : Prop := PolicyW8A8IEEEQualified.ModelContext ctx

def rule (_ : Refined.FEnv Info) (op : Op) (facts : List (Refined.Fact Info)) :
    Option (List (Refined.Fact Info)) :=
  if unmodelledAttr? op.blind = none ∧ dialectOf op.name = "linalg" ∧ op.successors = [] then
    match splitInsOuts op facts, PolicyW8A8OrderedOp.inspect op, op.regions with
    | some ([⟨⟨.tensor leftShape (.float .f32), true⟩, left⟩,
        ⟨⟨.tensor rightShape (.float .f32), true⟩, right⟩],
        [⟨⟨.tensor outputShape (.float .f32), true⟩, initial⟩]), some form, [body] => do
      let layout ← W8A8OrderedLayout.inspect leftShape rightShape outputShape form.maps form.iterators
      let leftCap ← upper left
      let rightCap ← upper right
      if initial.prior.prior.floating.literal = some (0 : FBits .f32) ∧
          PolicyW8A8OrderedBody.check body = true ∧
          W8A8OrderedDot.admissible leftCap rightCap layout.reductionSize = true ∧
          op.results.mapM (fun result => vtyOf 0 result.ty) = some [.tensor outputShape (.float .f32)] then
        some [outputFact outputShape (W8A8OrderedDot.cap leftCap rightCap layout.reductionSize)]
      else none
    | _, _, _ => none
  else none

theorem rule_sound (ib : Nat) : ∀ Γ op facts outputs,
    rule Γ op facts = some outputs → ∀ ctx call env args,
    ctx.indexBits = ib → allowed ctx call → Refined.EnvHolds model ib Γ env →
    Refined.FactsHold model ib args facts → ∃ values,
      sourceSem ctx call op args (evalRegions sourceSem call env op.regions) = .ok values ∧
      Refined.FactsHold model ib values outputs := by
  intro Γ op facts outputs checked ctx call env args _width permitted _valid holds
  unfold rule at checked
  split at checked
  · next header =>
    obtain ⟨modelled, dialect, _successors⟩ := header
    split at checked
    · next leftShape leftInfo rightShape rightInfo outputShape initialInfo form body splitFacts selected regions =>
      cases layoutFound : W8A8OrderedLayout.inspect leftShape rightShape outputShape form.maps form.iterators with
      | none => simp [layoutFound] at checked
      | some layout =>
        cases leftBound : upper leftInfo with
        | none => simp [layoutFound, leftBound] at checked
        | some leftCap =>
          cases rightBound : upper rightInfo with
          | none => simp [layoutFound, leftBound, rightBound] at checked
          | some rightCap =>
            simp only [layoutFound, leftBound, rightBound, bind, Option.bind] at checked
            split at checked
            · next conditions =>
              obtain ⟨zero, bodyChecked, safe, resultTypes⟩ := conditions
              cases checked
              obtain ⟨geometry, leftShapeEq, rightShapeEq, outputShapeEq, actualMaps, actualIterators⟩ :=
                W8A8OrderedLayout.inspect_sound layoutFound
              obtain ⟨xs, ys, splitArgs, inputFacts, outputFacts⟩ := Refined.splitInsOuts_refined splitFacts holds
              cases inputFacts with
              | cons leftHolds tail => cases tail with
                | cons rightHolds tail =>
                  cases tail
                  cases outputFacts with
                  | cons initialHolds tail =>
                    cases tail
                    rename_i left right initial
                    cases initial with
                    | scalar _ => cases initialHolds.1.1
                    | tensor initial =>
                      obtain ⟨fallback, interpretation⟩ := permitted
                      obtain ⟨values, executed, resultFacts⟩ := generic_refined geometry bodyChecked call env
                        fallback interpretation leftBound rightBound zero safe
                        (by simpa only [inputFact, leftShapeEq] using leftHolds)
                        (by simpa only [inputFact, rightShapeEq] using rightHolds)
                        (by simpa only [inputFact, outputShapeEq] using initialHolds)
                      cases resultFacts with
                      | cons resultHolds tail =>
                        cases tail
                        rename_i result
                        have arity : op.results.length = 1 := by
                          have length := optionMapM_length _ _ _ resultTypes
                          simpa only [List.length_cons, List.length_nil, Nat.zero_add] using length.symm
                        refine ⟨[result], ?_, .cons ?_ .nil⟩
                        · exact PolicyW8A8OrderedOp.execute modelled dialect selected regions splitArgs arity
                            (by simpa only [actualMaps, actualIterators] using executed)
                        · simpa only [outputShapeEq] using resultHolds
            · cases checked
    · cases checked
  · cases checked

def book (ib : Nat) : Contextual.Book model sourceSem ib allowed where
  rule := rule
  sound := rule_sound ib

#print axioms rule_sound
#print axioms book

end MerlinVerify.Check.PolicyW8A8OrderedRule
