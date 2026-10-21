/- Executable prefix definedness with explicit iteration-frame proofs.
The frozen checker and its facts remain unchanged. -/
import MerlinVerify.Rules.FrontendDefinedGeneric
import MerlinVerify.Rules.FrontendDefinedCompare
import MerlinVerify.Rules.FrontendDefinedSlice

namespace MerlinVerify.Rules.FrontendPrefixDefined
open MerlinVerify MerlinVerify.Check MerlinVerify.Check.Definedness MerlinVerify.Sem.Ops
open FrontendDefinedFrame

def base (ib depth : Nat) : BookAt opSem ib depth :=
  FrontendDefinedFrame.orElse (FrontendDefinedIndex.book ib depth)
    (lift (((FrontendDefinedIndex.castBook ib).orElse (FrontendDefinedCompare.comparisonBook ib)).orElse
      ((FrontendDefinedSlice.book ib).orElse (FrontendBoolDefined.book ib))) depth)

def nested (ib : Nat) : Nat → (depth : Nat) → BookAt opSem ib depth
  | 0, depth => base ib depth
  | fuel + 1, depth => FrontendDefinedFrame.orElse
      (lift (FrontendDefinedGeneric.book ib (nested ib fuel)) depth) (base ib depth)

def frameBook (ib depth : Nat) : BookAt opSem ib depth where
  rule Γ op fs := (nested ib (RegionDepth.op op) depth).rule Γ op fs
  sound Γ op fs rs h ctx call env args hib hd hΓ ha :=
    (nested ib (RegionDepth.op op) depth).sound Γ op fs rs h ctx call env args hib hd hΓ ha

def book (ib : Nat) : Book opSem ib := root (frameBook ib 0)

def candidateBook (limit ib : Nat) : Book (FrontendMaskRuntime.sem limit) ib :=
  FrontendRuntimeBridge.liftBook limit ib (book ib)

theorem check_sound {ib : Nat} {m : Module} {fn : String} {args results : List VTy}
    (h : Definedness.check (candidateBook 1024 ib) m fn args results = true) :
    FuncDefined (FrontendMaskRuntime.sem 1024) (fun ctx => ctx.indexBits = ib)
      (fun vs => HasTys ib vs args ∧ ResultsDefined vs) m fn := by
  exact Definedness.check_sound (candidateBook 1024 ib) h

#print axioms check_sound
end MerlinVerify.Rules.FrontendPrefixDefined
