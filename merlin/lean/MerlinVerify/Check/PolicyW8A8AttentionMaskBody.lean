/- Actual scalar mask-body recognition. Exact op keys, predicates, operand order
and types are checked; copying provenance cannot authorize a different mask. -/
import MerlinVerify.Rules.W8A8AttentionMaskBody

namespace MerlinVerify.Check.PolicyW8A8AttentionMaskBody

open MerlinVerify Rules W8A8AttentionMaskBody

inductive Kind where
  | equal | lessEqual | conjunction | inversion | copy | select
  deriving DecidableEq, Repr

def region : Kind → Region
  | .equal => comparisonRegion .equal
  | .lessEqual => comparisonRegion .lessEqual
  | .conjunction => conjunctionRegion
  | .inversion => inversionRegion
  | .copy => copyRegion
  | .select => selectRegion

def Correct (kind : Kind) (actual : Region) : Prop :=
  ∀ (ctx : Ctx) (call : CallSem) (env : Env) (args : List RVal),
    evalRegion W8A8SourceSemantics.opSem ctx call env actual args =
      evalRegion W8A8SourceSemantics.opSem ctx call env (region kind) args

def check (kind : Kind) (actual : Region) : Bool := alphaRegion [] (region kind) actual

theorem check_sound {kind : Kind} {actual : Region} (checked : check kind actual = true) : Correct kind actual := by
  intro ctx call env args
  have same := ParametricAlpha.alphaRegion_sound W8A8SourceAlpha.laws call call []
    (region kind) actual ctx env env checked
    (by intro x y missing; simp [Rho.find, List.lookup] at missing) (fun _ _ => rfl)
  exact (congrFun same args).symm

end MerlinVerify.Check.PolicyW8A8AttentionMaskBody
