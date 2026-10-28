/-
Executable recognition of the integer region emitted by Merlin's contraction_int8 pass.
This diagnostic component deliberately makes no whole-pass policy claim.
-/
import MerlinVerify.Check.Alpha
import MerlinVerify.Check.AlphaFast
import MerlinVerify.Rules.W8A8IntegerBody

namespace MerlinVerify.Check.PolicyW8A8Body

open MerlinVerify Rules.W8A8IntegerBody

/-- The semantic obligation, stated independently of the executable matcher. -/
def Correct (actual : Region) : Prop :=
  ∀ (ctx : Ctx) (call : CallSem) (env : Env)
    (a b : Option (BitVec 8)) (acc : Option (BitVec 32)),
    evalRegion Sem.Ops.opSem ctx call env actual
      [.scalar (value 8 a), .scalar (value 8 b), .scalar (value 32 acc)] =
      .ok [.scalar (result a b acc)]

/-- Check the actual operations, types, attributes, order, and SSA wiring; hints grant no facts. -/
def check (actual : Region) : Bool := alphaRegion [] region actual

/-- Executable acceptance establishes execution of signed i8 multiplication and i32 accumulation. -/
theorem check_sound {actual : Region} (h : check actual = true) : Correct actual := by
  intro ctx call env a b acc
  have he := alphaRegion_sound call call [] region actual ctx env env h
    (by intro x y hxy; simp [Rho.find, List.lookup] at hxy) (fun _ _ => rfl)
  rw [← he]
  exact region_sound ctx call env a b acc

/-- Initialized inputs yield the exact modular MAC result; no claim of absence of wraparound yet. -/
theorem check_initialized_sound {actual : Region} (h : check actual = true)
    (ctx : Ctx) (call : CallSem) (env : Env) (a b : BitVec 8) (acc : BitVec 32) :
    evalRegion Sem.Ops.opSem ctx call env actual
      [.scalar (.some (.bits 8 a)), .scalar (.some (.bits 8 b)),
        .scalar (.some (.bits 32 acc))] =
      .ok [.scalar (.some (.bits 32 (Rules.W8A8Accumulation.step acc (a, b))))] :=
  check_sound h ctx call env (some a) (some b) (some acc)

/-- Execute the checked region once per pair. Tensor-coordinate selection is a separate obligation. -/
def run (actual : Region) (ctx : Ctx) (call : CallSem) (env : Env) :
    List Rules.W8A8Accumulation.Pair → BitVec 32 → Outcome (BitVec 32)
  | [], acc => .ok acc
  | (a, b) :: xs, acc => do
    let [.scalar (.some (.bits 32 next))] ← evalRegion Sem.Ops.opSem ctx call env actual
      [.scalar (.some (.bits 8 a)), .scalar (.some (.bits 8 b)),
        .scalar (.some (.bits 32 acc))]
      | .stuck "W8A8 integer body did not return one initialized i32"
    run actual ctx call env xs next

/-- Accepted actual-region execution agrees with the proved modular fold for any number of steps. -/
theorem run_sound {actual : Region} (h : check actual = true)
    (ctx : Ctx) (call : CallSem) (env : Env)
    (xs : List Rules.W8A8Accumulation.Pair) (acc : BitVec 32) :
    run actual ctx call env xs acc = .ok (Rules.W8A8Accumulation.accumulate xs acc) := by
  induction xs generalizing acc with
  | nil => rfl
  | cons p ps ih =>
    obtain ⟨a, b⟩ := p
    simpa [run, check_initialized_sound h, Rules.W8A8Accumulation.accumulate,
      List.foldl_cons] using ih (Rules.W8A8Accumulation.step acc (a, b))

/-- Under an explicit length bound and zero initialization, actual-region execution is exact over ℤ. -/
theorem run_exact {actual : Region} (h : check actual = true)
    (ctx : Ctx) (call : CallSem) (env : Env)
    (xs : List Rules.W8A8Accumulation.Pair)
    (hk : (xs.length : Int) * 16384 < 2147483648) :
    (run actual ctx call env xs 0).bind (fun x => .ok x.toInt) =
      .ok (Rules.W8A8Accumulation.dot xs) := by
  rw [run_sound h, Outcome.bind_ok, Rules.W8A8Accumulation.exact_of_length xs hk]

#print axioms check_sound
#print axioms check_initialized_sound
#print axioms run_sound
#print axioms run_exact

end MerlinVerify.Check.PolicyW8A8Body
