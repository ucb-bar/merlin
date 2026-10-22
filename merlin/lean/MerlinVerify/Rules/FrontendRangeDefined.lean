/- Assembled inactive bounds-aware checker. Unknown element bounds are never
input facts. They must be derived by the current SSA producer. -/
import MerlinVerify.Rules.FrontendRangeTensor
import MerlinVerify.Rules.FrontendPrefixDefined

namespace MerlinVerify.Rules.FrontendRangeDefined
open MerlinVerify MerlinVerify.Check Sem.Ops
open FrontendTensorRanges FrontendRangeFrame

def base (ib : Nat) (ranges : List Nat) : BookAt ib ranges :=
  let scalar := ((FrontendRangeScalars.constantBook ib).orElse
    (FrontendRangeScalars.castBook ib)).orElse (FrontendRangeArithmetic.book ib)
  let tensor := (FrontendRangeTensor.splatBook ib).orElse (FrontendRangeTensor.reshapeBook ib)
  let existing := (scalarBook (Definedness.Bounds.leafBook ib)).orElse
    (Definedness.Refined.Book.lift model (FrontendPrefixDefined.book ib))
  orElse (FrontendRangeScalars.indexBook ib ranges) (lift (scalar.orElse (tensor.orElse existing)) ranges)

def nested (ib : Nat) : Nat → (ranges : List Nat) → BookAt ib ranges
  | 0, ranges => base ib ranges
  | fuel + 1, ranges => orElse (lift ((FrontendRangeGeneric.emptyBook ib).orElse
      (FrontendRangeGeneric.book ib (nested ib fuel))) ranges) (base ib ranges)

def frameBook (ib : Nat) (ranges : List Nat) : BookAt ib ranges where
  rule Γ op fs := (nested ib (RegionDepth.op op) ranges).rule Γ op fs
  sound Γ op fs rs h ctx call env args hib hf hΓ ha :=
    (nested ib (RegionDepth.op op) ranges).sound Γ op fs rs h ctx call env args hib hf hΓ ha

def book (ib : Nat) : Definedness.Refined.Book model opSem ib := root (frameBook ib [])

def candidateBook (limit ib : Nat) : Definedness.Refined.Book model (FrontendMaskRuntime.sem limit) ib where
  rule Γ op fs := if FrontendRuntimeBridge.delegates op && FrontendRuntimeBridge.regions op.regions then
    (book ib).rule Γ op fs else none
  sound Γ op fs rs h ctx call env args hib hΓ ha := by
    split at h
    · next hc =>
      simp only [Bool.and_eq_true] at hc
      rw [FrontendRuntimeBridge.regions_sound limit call env op.regions hc.2,
        FrontendRuntimeBridge.delegates_sound hc.1]
      exact (book ib).sound Γ op fs rs h ctx call env args hib hΓ ha
    · cases h

/-- Initialization is required only at declared inputs; every internal interval is derived.
The interpretation is exactly the same inactive runtime used by bool repair and checkpoint proofs. -/
theorem check_sound {ib limit : Nat} {m : Module} {fn : String} {args results : List VTy}
    (h : Definedness.Refined.check (candidateBook limit ib) m fn args results = true) :
    FuncDefined (FrontendMaskRuntime.sem limit) (fun ctx => ctx.indexBits = ib)
      (fun vs => HasTys ib vs args ∧ ResultsDefined vs) m fn :=
  Definedness.Refined.check_sound (candidateBook limit ib) h

#print axioms check_sound
end MerlinVerify.Rules.FrontendRangeDefined
