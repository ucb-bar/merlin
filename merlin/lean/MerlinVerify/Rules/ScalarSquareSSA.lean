/- Runtime SSA facts for scalar-square composition. These are source-side typing and
constant-producer obligations, not numerical identities or source/target equivalence. -/
import MerlinVerify.Check.ScalarSquare
import MerlinVerify.Check.Typing

namespace MerlinVerify.Rules.ScalarSquareSSA

open MerlinVerify MerlinVerify.Check MerlinVerify.Check.ScalarSquare MerlinVerify.Sem.Ops

def HasType (ib : Nat) (ty : Ty) (value : RVal) : Prop :=
  ∃ vt, vtyOf ib ty = some vt ∧ RVal.HasTy ib value vt

def Arguments (ib : Nat) (args : List Value) (values : List RVal) : Prop :=
  List.Forall₂ (fun arg value => HasType ib arg.ty value) args values

/-- Facts refer to the newest lexical producer, not to a globally unique spelling. -/
def Good (ib : Nat) (id : Nat) (binding : Binding) (value : RVal) : Prop :=
  HasType ib binding.value.ty value ∧
  ∀ k producer, binding.producer = some producer →
    literalTwo k producer ⟨id, .float k⟩ = true →
    value = .scalar (.some (.flt k (ScalarSquareContract.two k)))

def Invariant (ib : Nat) (scope : Scope) (env : Env) : Prop :=
  ∀ id binding, scope.lookup id = some binding →
    ∃ value, env.get? id = some value ∧ Good ib id binding value

def ResultFacts (ib : Nat) (op : Op) (values : List RVal) : Prop :=
  List.Forall₂ (fun result value => Good ib result.id ⟨result, some op⟩ value)
    op.results values

theorem map_facts {α : Type} {f : α → Nat × Binding} {ib : Nat} {xs : List α}
    {vs : List RVal} (h : List.Forall₂ (fun x v => Good ib (f x).1 (f x).2 v) xs vs) :
    List.Forall₂ (fun b v => Good ib b.1 b.2 v) (xs.map f) vs := by
  induction h with
  | nil => exact .nil
  | cons head _ ih => exact .cons head ih

theorem empty (ib : Nat) : Invariant ib [] [] := by
  intro id binding found
  simp [List.lookup] at found

theorem extend {ib : Nat} {scope : Scope} {env : Env} (valid : Invariant ib scope env)
    {bindings : Scope} {values : List RVal}
    (facts : List.Forall₂ (fun b value => Good ib b.1 b.2 value) bindings values) :
    Invariant ib (bindings ++ scope) ((bindings.map Prod.fst).zip values ++ env) := by
  induction facts with
  | nil => simpa using valid
  | @cons b value bs values head tail ih =>
    intro id binding found
    simp only [List.cons_append, List.lookup] at found
    simp only [List.map_cons, List.zip_cons_cons, List.cons_append, Env.get?, List.lookup]
    by_cases same : id = b.1
    · subst id
      simp only [beq_self_eq_true, ↓reduceIte, Option.some.injEq] at found
      subst binding
      exact ⟨value, by simp, head⟩
    · have different : (id == b.1) = false := by simp [same]
      simp only [different, Bool.false_eq_true, ↓reduceIte] at found ⊢
      exact ih id binding found

theorem arguments_bind {ib : Nat} {scope : Scope} {env bound : Env}
    {args : List Value} {values : List RVal} (valid : Invariant ib scope env)
    (typed : Arguments ib args values) (boundEq : env.bindAll args values = .ok bound) :
    Invariant ib (bindArgs scope args) bound := by
  have facts : List.Forall₂ (fun b value => Good ib b.1 b.2 value)
      (args.map fun v => (v.id, (⟨v, none⟩ : Binding))) values := by
    clear boundEq
    induction typed with
    | nil => exact .nil
    | cons h _ ih =>
      exact .cons ⟨h, by intro k producer impossible; cases impossible⟩ ih
  have length := List.Forall₂.length_eq typed
  simp only [Env.bindAll, length, ↓reduceIte, Outcome.ok.injEq] at boundEq
  subst bound
  simpa [bindArgs, List.map_map, Function.comp_def] using extend valid facts

theorem results_bind {ib : Nat} {scope : Scope} {env bound : Env}
    {op : Op} {values : List RVal} (valid : Invariant ib scope env)
    (facts : ResultFacts ib op values) (boundEq : env.bindAll op.results values = .ok bound) :
    Invariant ib (bindOp scope op) bound := by
  have mapped : List.Forall₂ (fun b value => Good ib b.1 b.2 value)
      (op.results.map fun v => (v.id, (⟨v, some op⟩ : Binding))) values := by
    exact map_facts facts
  have length := List.Forall₂.length_eq facts
  simp only [Env.bindAll, length, ↓reduceIte, Outcome.ok.injEq] at boundEq
  subst bound
  simpa [bindOp, List.map_map, Function.comp_def] using extend valid mapped

theorem float_cases {ib : Nat} {k : FKind} {v : RVal} (typed : HasType ib (.float k) v) :
    v = .scalar .poison ∨ ∃ bits : FBits k, v = .scalar (.some (.flt k bits)) := by
  obtain ⟨vt, typeEq, has⟩ := typed
  simp [vtyOf, Ty.bitWidth?] at typeEq
  subst vt
  cases v with
  | tensor _ => cases has
  | scalar v =>
    cases v with
    | poison => exact Or.inl rfl
    | some scalar =>
      cases scalar with
      | bits _ _ => simp [RVal.HasTy, Val.HasTy, Scalar.conforms] at has
      | flt other bits =>
        simp [RVal.HasTy, Val.HasTy, Scalar.conforms] at has
        subst other
        exact Or.inr ⟨bits, rfl⟩

theorem selected_bindings {scope : Scope} {path : List Nat} {op : Op} {site : Site}
    (selected : site? scope path op = some site) :
    ∃ left right, scope.lookup site.base = some left ∧ left.value.ty = .float site.kind ∧
      scope.lookup site.exponent = some right ∧ right.producer = some site.producer := by
  unfold site? at selected
  split at selected <;> try simp_all
  split at selected <;> try simp_all
  all_goals simp only [bind, Option.bind] at selected
  all_goals repeat' (split at selected <;> try simp_all)
  all_goals try simp_all [Option.pure_def]
  all_goals obtain ⟨types, _, _, rfl⟩ := selected
  all_goals aesop

theorem site_values {ib : Nat} {scope : Scope} {env : Env} {path : List Nat}
    {op : Op} {site : Site} (selected : site? scope path op = some site)
    (valid : Invariant ib scope env) :
    ∃ base, env.get? site.base = some base ∧ HasType ib (.float site.kind) base ∧
      env.get? site.exponent = some (.scalar (.some (.flt site.kind
        (ScalarSquareContract.two site.kind)))) := by
  obtain ⟨left, right, leftEq, typeEq, rightEq, producer⟩ := selected_bindings selected
  obtain ⟨base, baseEq, baseGood⟩ := valid _ _ leftEq
  obtain ⟨exponent, exponentEq, exponentGood⟩ := valid _ _ rightEq
  have exactTwo := exponentGood.2 site.kind site.producer producer site.literal
  subst exponent
  exact ⟨base, baseEq, typeEq ▸ baseGood.1, exponentEq⟩

theorem get_two (env : Env) (x y : Nat) (a b : RVal)
    (hx : env.get? x = some a) (hy : env.get? y = some b) :
    env.getAll [x,y] = .ok [a,b] := by
  simp [Env.getAll, Outcome.mapM, hx, hy, Outcome.ofOption, Outcome.bind]

theorem arithmetic_call_irrelevant (ctx : Ctx) (left right : CallSem)
    (op : Op) (args : List RVal) (rs : List (Ctx → RegionSem))
    (dialect : dialectOf op.name = "arith" ∨ dialectOf op.name = "math") :
    opSem ctx left op args rs = opSem ctx right op args rs := by
  unfold opSem
  split
  · rfl
  · have blindName : op.blind.name = op.name := rfl
    rcases dialect with hd | hd <;> simp only [dispatch, blindName, hd] <;> rfl

theorem site_step {ib : Nat} {scope : Scope} {env : Env} {path : List Nat}
    {op : Op} {site : Site} (selected : site? scope path op = some site)
    (valid : Invariant ib scope env) (ctx : Ctx) (left right : CallSem)
    (law : ScalarSquareContract.Law ctx.M site.kind site.attrs) :
    (env.getAll site.source.operands).bind (fun args => opSem ctx left site.source args []) =
    (env.getAll site.target.operands).bind (fun args => opSem ctx right site.target args []) := by
  obtain ⟨base, baseEq, typed, exponentEq⟩ := site_values selected valid
  have sourceArgs := get_two env site.base site.exponent base _ baseEq exponentEq
  have targetArgs := get_two env site.base site.base base base baseEq baseEq
  change env.getAll [site.base, site.exponent] = _ at sourceArgs
  change env.getAll [site.base, site.base] = _ at targetArgs
  change (env.getAll [site.base, site.exponent]).bind _ =
    (env.getAll [site.base, site.base]).bind _
  rw [sourceArgs, targetArgs]
  simp only [Outcome.bind_ok]
  rw [← arithmetic_call_irrelevant ctx left right site.target _ _ (Or.inl (by cbv))]
  rcases float_cases typed with poison | ⟨bits, bitsEq⟩
  · subst base
    exact ScalarSquareContract.poison_sound ctx left site.kind site.result site.base site.exponent
      site.attrs site.checks.1 site.checks.2.2.2.1 site.checks.2.2.2.2
  · subst base
    exact site.scalar_sound ctx left law bits

#print axioms results_bind
#print axioms site_values
#print axioms site_step

end MerlinVerify.Rules.ScalarSquareSSA
