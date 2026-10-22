/- Local runtime-binding contract. A tagged static empty denotes an explicitly supplied value,
not the frozen poison-producing tensor.empty. No claim about archive provenance is assumed.
The main claim equates complete executions under this interpretation and an ordered ABI. -/
import MerlinVerify.Check.ParametricAlpha
import MerlinVerify.Check.Typing
import MerlinVerify.Sem.Defined
namespace MerlinVerify.Rules.QinnerBinding
open MerlinVerify MerlinVerify.Check
structure Site where
  key : String
  result : Value

def site? : Op → Option Site
  | .mk "tensor.empty" [r] [] attrs [] [] [] =>
    match attrs.lookup "prov.quant_inner" with
    | some (.str key) => if key.isEmpty then none else some ⟨key,r⟩
    | _ => none
  | _ => none

theorem site_shape {op : Op} {site : Site} (found : site? op = some site) :
    op.operands = [] ∧ op.results = [site.result] ∧ op.isTerminator = false := by
  fun_cases site? op <;> simp_all [site?, Op.name, Op.operands, Op.results, Op.isTerminator, terminatorNames]
  all_goals cases found; rfl

def sem (binding : String → RVal) : OpSem := fun ctx call op args regions =>
  match site? op with
  | some site => .ok [binding site.key]
  | none => Sem.Ops.opSem ctx call op args regions
structure Slot where
  key : String
  arg : Value

def Supplies (slots : List Slot) (binding : String → RVal) (env : Env) : Prop :=
  ∀ s ∈ slots, env.get? s.arg.id = some (binding s.key)
def protects (slots : List Slot) (ids : List Nat) : Bool :=
  ids.all fun id => slots.all fun s => id != s.arg.id

theorem supplies_extend {slots : List Slot} {binding : String → RVal} {env : Env}
    (supplied : Supplies slots binding env) {ids : List Nat} (values : List RVal)
    (guarded : protects slots ids = true) :
    Supplies slots binding (ids.zip values ++ env) := by
  intro s hs
  have hn : ∀ id ∈ ids, id ≠ s.arg.id := by
    intro id hi
    exact (by simpa using (List.all_eq_true.mp
      (List.all_eq_true.mp guarded id hi) s hs))
  have lookup : ∀ (xs : List Nat) (vs : List RVal),
      (∀ id ∈ xs, id ≠ s.arg.id) →
      (Env.get? (xs.zip vs ++ env) s.arg.id) = env.get? s.arg.id := by
    intro xs
    induction xs with
    | nil => intros; rfl
    | cons x xs ih =>
      intro vs h
      cases vs with
      | nil => rfl
      | cons v vs =>
        have hx : (s.arg.id == x) = false := by simp [Ne.symm (h x (by simp))]
        simpa [Env.get?, List.lookup, hx] using ih vs (fun id hi => h id (by simp [hi]))
  rw [lookup ids values hn]
  exact supplied s hs

theorem corr_bind_left {rho : Rho} {envS envT : Env} (corr : Corr rho envS envT)
    (x y : Nat) (v : RVal) (bound : envT.get? y = some v) :
    Corr ((x,y)::rho) ((x,v)::envS) envT := by
  intro a b mapped
  by_cases h : a = x
  · subst a
    simp [Rho.find, List.lookup] at mapped
    subst b
    simpa [Env.get?, List.lookup] using bound.symm
  · simpa [Rho.find, List.lookup, show (a == x) = false by simpa using h, Env.get?] using corr a b
      (by simpa [Rho.find, List.lookup, show (a == x) = false by simpa using h] using mapped)

def BindingTyped (ib : Nat) (slots : List Slot) (binding : String → RVal) : Prop :=
  ∀ s ∈ slots, ∃ ty, vtyOf ib s.arg.ty = some ty ∧ RVal.HasTy ib (binding s.key) ty
def BindingClean (slots : List Slot) (binding : String → RVal) : Prop :=
  ∀ s ∈ slots, (binding s.key).Defined
#print axioms supplies_extend
#print axioms corr_bind_left
end MerlinVerify.Rules.QinnerBinding
