/- Checked weakening of inferred intervals. A requested postcondition is not evidence:
all inferred producer facts must imply it before acceptance. -/
import MerlinVerify.Rules.FrontendRangeSelect

namespace MerlinVerify.Rules.FrontendRangeFacts
open MerlinVerify MerlinVerify.Check
open FrontendTensorRanges
open Definedness.IndexRanges (Interval)

def contains (inferred requested : Option Interval) : Bool :=
  match requested, inferred with
  | none, _ => true
  | some b, some a => decide (b.lower ≤ a.lower ∧ a.upper ≤ b.upper)
  | _, _ => false

theorem contains_sound {inferred requested : Option Interval} {v : RVal}
    (h : contains inferred requested = true) (hv : model.holds inferred v) : model.holds requested v := by
  cases requested with
  | none => trivial
  | some b =>
    cases inferred with
    | none => cases h
    | some a =>
      simp only [contains, decide_eq_true_eq] at h
      exact FrontendRangeSelect.holds_mono h.1 h.2 hv

def fact (inferred requested : Fact) : Bool :=
  decide (inferred.base = requested.base) && contains inferred.info requested.info

theorem fact_sound {ib : Nat} {inferred requested : Fact} {v : RVal}
    (h : fact inferred requested = true) (hv : Definedness.Refined.Fact.Holds model ib v inferred) :
    Definedness.Refined.Fact.Holds model ib v requested := by
  simp only [fact, Bool.and_eq_true, decide_eq_true_eq] at h
  exact ⟨h.1 ▸ hv.1, contains_sound h.2 hv.2⟩

def fits : List Fact → List Fact → Bool
  | [], [] => true
  | f :: fs, g :: gs => fact f g && fits fs gs
  | _, _ => false

theorem fits_sound {ib : Nat} {vs : List RVal} {fs requested : List Fact}
    (h : fits fs requested = true) (hv : Definedness.Refined.FactsHold model ib vs fs) :
    Definedness.Refined.FactsHold model ib vs requested := by
  induction hv generalizing requested with
  | nil => cases requested <;> first | exact .nil | cases h
  | cons hx hrest ih =>
    cases requested with
    | nil => cases h
    | cons g gs =>
      simp only [fits, Bool.and_eq_true] at h
      exact .cons (fact_sound h.1 hx) (ih h.2)
end MerlinVerify.Rules.FrontendRangeFacts
