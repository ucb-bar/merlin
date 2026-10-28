/-
G0-FROZEN. The result of running a program fragment.

* `ub`     — immediate undefined behaviour (e.g. `arith.divsi` by zero). Anything refines it.
* `stuck`  — the semantics does not model this input (unsupported op, type error, missing value).
            Nothing refines `stuck` and `stuck` refines nothing, so an unmodelled construct can
            never be part of a verified claim (design §1.2 rule 3).
* `ok a`   — a defined result, which may still contain `poison` values.
-/

namespace MerlinVerify

inductive Outcome (α : Type) where
  | ub
  | stuck (reason : String)
  | ok (value : α)
  deriving Repr, Inhabited

namespace Outcome

def bind {α β : Type} : Outcome α → (α → Outcome β) → Outcome β
  | .ub, _ => .ub
  | .stuck r, _ => .stuck r
  | .ok a, f => f a

instance : Monad Outcome where
  pure := .ok
  bind := Outcome.bind

@[simp] theorem bind_ok {α β : Type} (a : α) (f : α → Outcome β) : (Outcome.ok a).bind f = f a := rfl
@[simp] theorem bind_ub {α β : Type} (f : α → Outcome β) : (Outcome.ub : Outcome α).bind f = .ub := rfl
@[simp] theorem bind_stuck {α β : Type} (r : String) (f : α → Outcome β) :
    (Outcome.stuck r : Outcome α).bind f = .stuck r := rfl

@[simp] theorem pure_eq {α : Type} (a : α) : (pure a : Outcome α) = .ok a := rfl
@[simp] theorem bindOp_eq {α β : Type} (x : Outcome α) (f : α → Outcome β) : x >>= f = x.bind f := rfl

def isOk {α : Type} : Outcome α → Bool
  | .ok _ => true
  | _ => false

/-- Lift an `Option`, treating `none` as `stuck reason`. -/
def ofOption {α : Type} (reason : String) : Option α → Outcome α
  | some a => .ok a
  | none => .stuck reason

/-- Run `f` over a list, stopping at the first non-`ok` outcome. -/
def mapM {α β : Type} (f : α → Outcome β) : List α → Outcome (List β)
  | [] => .ok []
  | a :: as => (f a).bind fun b => (mapM f as).bind fun bs => .ok (b :: bs)

end Outcome

end MerlinVerify
