/- Total scalar integer primitives for exact pointwise loop composition. Typed poison is retained;
no initializedness, arithmetic identity, float law or index-width assumption is introduced. -/
import MerlinVerify.Check.RulesArith

namespace MerlinVerify.Rules.ParallelFusionScalar

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops

inductive Kind where
  | add | mul | band | bor | bxor
  deriving DecidableEq

def Kind.name : Kind → String
  | .add => "arith.addi" | .mul => "arith.muli" | .band => "arith.andi"
  | .bor => "arith.ori" | .bxor => "arith.xori"

def Kind.bits (kind : Kind) {w : Nat} (x y : BitVec w) : BitVec w :=
  match kind with
  | .add => x + y | .mul => x * y | .band => x &&& y | .bor => x ||| y | .bxor => x ^^^ y

/-- This total value function is used only after proving the actual input widths below.
Malformed widths do not become evidence: the semantic theorem requires both runtime types. -/
def Kind.value (kind : Kind) (w : Nat) (left right : Val) : Val :=
  match left, right with
  | .some (.bits _ x), .some (.bits _ y) =>
    .some (.bits w (kind.bits (BitVec.ofNat w x.toNat) (BitVec.ofNat w y.toNat)))
  | _, _ => .poison

theorem value_typed (ib : Nat) (kind : Kind) (w : Nat) (left right : Val) :
    Val.HasTy ib (.int w) (kind.value w left right) := by
  unfold Kind.value
  split <;> simp [Val.HasTy, Scalar.conforms]

def kind? (name : String) : Option Kind :=
  if name = "arith.addi" then some .add else
  if name = "arith.muli" then some .mul else
  if name = "arith.andi" then some .band else
  if name = "arith.ori" then some .bor else
  if name = "arith.xori" then some .bxor else none

theorem kind?_name {name : String} {kind : Kind} (h : kind? name = some kind) :
    name = kind.name := by
  unfold kind? at h
  split at h
  · cases h; assumption
  · split at h
    · cases h; assumption
    · split at h
      · cases h; assumption
      · split at h
        · cases h; assumption
        · split at h
          · cases h; assumption
          · cases h

/-- Every supported primitive succeeds on its actual typed scalar inputs, including poison.
This is stronger than a source-success refinement and is suitable for exact loop fusion. -/
theorem meaning {ib w : Nat} {s : Op} {r : Value} {left right : Val}
    (kind : Kind) (hd : dialectOf s.name = "arith") (hn : s.name = kind.name)
    (hm : unmodelledAttr? s.blind = none) (hr : s.results = [r]) (ht : r.ty = .int w)
    (hf : overflowFlags s = [])
    (hl : Val.HasTy ib (.int w) left) (hrr : Val.HasTy ib (.int w) right)
    (ctx : Ctx) (call : CallSem) :
    opSem ctx call s [.scalar left, .scalar right] [] =
      .ok [.scalar (kind.value w left right)] := by
  rw [opSem_modelled hm]
  simp only [dispatch, Op.blind_name, hd]
  have hf' : overflowFlags s.blind = [] := by simpa [overflowFlags, attrEnum?] using hf
  rcases hasTy_intWidth (ib := ib) (t := .int w) rfl hl with rfl | ⟨a, rfl⟩ <;>
    rcases hasTy_intWidth (ib := ib) (t := .int w) rfl hrr with rfl | ⟨b, rfl⟩
  all_goals
    cases kind <;>
      simp_all [Kind.name, Kind.value, Kind.bits, Arith.sem, Arith.binOp, Arith.pureN,
        Arith.resultElt, Op.blind_results, liftEager, Arith.intBin, Arith.arithWithFlags, Arith.bv]

end MerlinVerify.Rules.ParallelFusionScalar
