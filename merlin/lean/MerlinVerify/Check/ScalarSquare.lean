/- Complete source-derived boundary validation, including every nested block and region.
The checker never accepts a numerical-law assertion from a certificate. The contract law
remains an explicit hypothesis of the local semantic theorem. -/
import MerlinVerify.Rules.ScalarSquareContract
import MerlinVerify.Check.ExactSyntax

namespace MerlinVerify.Check.ScalarSquare

open MerlinVerify MerlinVerify.Sem.Ops
open Rules

structure Binding where
  value : Value
  producer : Option Op

abbrev Scope := List (Nat × Binding)

def bindArgs (scope : Scope) (args : List Value) : Scope :=
  args.map (fun v => (v.id, ⟨v, none⟩)) ++ scope

def bindOp (scope : Scope) (op : Op) : Scope :=
  op.results.map (fun v => (v.id, ⟨v, some op⟩)) ++ scope

def literalTwo (kind : FKind) (op : Op) (value : Value) : Bool :=
  op.name == "arith.constant" && op.operands.isEmpty && op.regions.isEmpty &&
    op.successors.isEmpty && op.props.isEmpty &&
    (match op.results with | [r] => r.id == value.id && decide (r.ty = value.ty) | _ => false) && decide (value.ty = .float kind) &&
    decide (op.attr? "value" = some (.float (ScalarSquareContract.two kind).toNat (.float kind))) &&
    decide (unmodelledAttr? op.blind = none)

def Checks (result : Value) (kind : FKind) (attrs : List (String × Attr)) : Prop :=
  result.ty = .float kind ∧ ScalarSquareContract.admitted kind = true ∧ ScalarSquareContract.flags attrs = true ∧
  unmodelledAttr? (ScalarSquareContract.primitive "math.powf" attrs kind).blind = none ∧
  unmodelledAttr? (ScalarSquareContract.primitive "arith.mulf" attrs kind).blind = none

instance (r : Value) (k : FKind) (a : List (String × Attr)) : Decidable (Checks r k a) := by
  unfold Checks
  infer_instance

structure Site where
  path : List Nat
  result : Value
  kind : FKind
  base : Nat
  exponent : Nat
  attrs : List (String × Attr)
  producer : Op
  checks : Checks result kind attrs
  literal : literalTwo kind producer ⟨exponent, .float kind⟩ = true

def Site.source (s : Site) : Op :=
  .mk "math.powf" [s.result] [s.base, s.exponent] s.attrs [] [] []

def Site.target (s : Site) : Op := ScalarSquareContract.multiply s.source s.base

/-- Uses lexical source bindings; block arguments shadow captured producer facts, and no
fact from a region/block is exported into its parent or siblings. -/
def site? (scope : Scope) (path : List Nat) (op : Op) : Option Site := do
  let .mk "math.powf" [result] [base, exponent] attrs [] [] [] := op | none
  let .float kind := result.ty | none
  let left ← scope.lookup base
  let right ← scope.lookup exponent
  if left.value.ty != .float kind || right.value.ty != .float kind then none
  let producer ← right.producer
  if checked : Checks result kind attrs then
    if literal : literalTwo kind producer ⟨exponent, .float kind⟩ = true then
      return ⟨path, result, kind, base, exponent, attrs, producer, checked, literal⟩
    else none
  else none

theorem site_source {scope : Scope} {path : List Nat} {op : Op} {site : Site}
    (selected : site? scope path op = some site) : site.source = op := by
  unfold site? at selected
  split at selected <;> try simp_all
  split at selected <;> try simp_all
  all_goals simp only [bind, Option.bind] at selected
  all_goals repeat' (split at selected <;> try simp_all)
  all_goals simp_all [Option.pure_def, Site.source]
  all_goals obtain ⟨_, _, _, rfl⟩ := selected
  all_goals exact ⟨rfl, ⟨rfl, rfl⟩, rfl⟩

/-- Numerical obligation for every recorded site, on every typed scalar bit pattern. No
numerical law has been assumed by normalization or by complete-output comparison. -/
theorem Site.scalar_sound (site : Site) (ctx : Ctx) (call : CallSem)
    (numerical : ScalarSquareContract.Law ctx.M site.kind site.attrs) (x : FBits site.kind) :
    opSem ctx call site.source
      [.scalar (.some (.flt site.kind x)), .scalar (.some (.flt site.kind (ScalarSquareContract.two site.kind)))] [] =
    opSem ctx call site.target
      [.scalar (.some (.flt site.kind x)), .scalar (.some (.flt site.kind x))] [] :=
  ScalarSquareContract.scalar_sound ctx call site.kind site.result site.base site.exponent site.attrs
    site.checks.1 site.checks.2.2.2.1 site.checks.2.2.2.2 numerical x

structure Body where
  ops : List Op
  sites : List Site

structure Regions where
  regions : List Region
  sites : List Site

structure Blocks where
  blocks : List Block
  sites : List Site

mutual
  def operations (scope : Scope) (path : List Nat) (index : Nat) : List Op → Body
    | [] => ⟨[], []⟩
    | (.mk name results operands attrs props rs successors) :: rest =>
      let op := Op.mk name results operands attrs props rs successors
      let here := path ++ [index]
      let nested := regions scope here 0 rs
      let picked := site? scope here op
      let current := match picked with
        | some site => site.target
        | none => .mk op.name op.results op.operands op.attrs op.props nested.regions op.successors
      let tail := operations (bindOp scope op) path (index + 1) rest
      ⟨current :: tail.ops, picked.toList ++ nested.sites ++ tail.sites⟩
    termination_by ops => sizeOf ops
    decreasing_by all_goals simp_wf; all_goals omega
  def regions (scope : Scope) (path : List Nat) (index : Nat) : List Region → Regions
    | [] => ⟨[], []⟩
    | .mk bs :: rest =>
      let here := blocks scope (path ++ [index]) 0 bs
      let tail := regions scope path (index + 1) rest
      ⟨.mk here.blocks :: tail.regions, here.sites ++ tail.sites⟩
    termination_by rs => sizeOf rs
    decreasing_by all_goals simp_wf; all_goals omega
  def blocks (scope : Scope) (path : List Nat) (index : Nat) : List Block → Blocks
    | [] => ⟨[], []⟩
    | .mk args ops :: rest =>
      let here := operations (bindArgs scope args) (path ++ [index]) 0 ops
      let tail := blocks scope path (index + 1) rest
      ⟨.mk args here.ops :: tail.blocks, here.sites ++ tail.sites⟩
    termination_by bs => sizeOf bs
    decreasing_by all_goals simp_wf; all_goals omega
end

def reference (before : Module) : Module :=
  { before with ops := (operations [] [] 0 before.ops).ops }

def sites (before : Module) : List Site := (operations [] [] 0 before.ops).sites

/-- Independent source census: identify scalar literal-two powers without consulting target
operations or the flag eligibility test used to construct a Site. Unsupported flags therefore
cannot silently turn a required numerical obligation into an identity claim. -/
def candidate (scope : Scope) (op : Op) : Bool := Id.run do
  let .mk "math.powf" [result] [_, exponent] _ _ _ _ := op | return false
  let .float kind := result.ty | return false
  if !ScalarSquareContract.admitted kind then return false
  let some binding := scope.lookup exponent | return false
  let some producer := binding.producer | return false
  return literalTwo kind producer ⟨exponent, .float kind⟩

mutual
  def candidateOps (scope : Scope) (path : List Nat) (index : Nat) : List Op → List (List Nat)
    | [] => []
    | (.mk name results operands attrs props rs successors) :: rest =>
      let op := Op.mk name results operands attrs props rs successors
      let here := path ++ [index]
      (if candidate scope op then [here] else []) ++ candidateRegions scope here 0 rs ++
        candidateOps (bindOp scope op) path (index + 1) rest
    termination_by ops => sizeOf ops
    decreasing_by all_goals simp_wf; all_goals omega
  def candidateRegions (scope : Scope) (path : List Nat) (index : Nat) : List Region → List (List Nat)
    | [] => []
    | .mk bs :: rest => candidateBlocks scope (path ++ [index]) 0 bs ++
        candidateRegions scope path (index + 1) rest
    termination_by rs => sizeOf rs
    decreasing_by all_goals simp_wf; all_goals omega
  def candidateBlocks (scope : Scope) (path : List Nat) (index : Nat) : List Block → List (List Nat)
    | [] => []
    | .mk args ops :: rest => candidateOps (bindArgs scope args) (path ++ [index]) 0 ops ++
        candidateBlocks scope path (index + 1) rest
    termination_by bs => sizeOf bs
    decreasing_by all_goals simp_wf; all_goals omega
end

def candidates (before : Module) : List (List Nat) := candidateOps [] [] 0 before.ops

/-- Resolve a recorded position against the actual module and reconstruct its lexical scope.
This is checked independently of the reference builder, on both complete input modules. -/
def locate (scope : Scope) (ops : List Op) : List Nat → Option (Scope × Op)
  | [] => none
  | index :: rest => do
    let op ← ops[index]?
    let lexical :=  (ops.take index).foldl bindOp scope
    match rest with
    | [] => return (lexical, op)
    | ri :: bi :: tail => do
      let region ← op.regions[ri]?
      let block ← region.blocks[bi]?
      locate (bindArgs lexical block.args) block.ops tail
    | _ => none

def pairCheck (before after : Module) (site : Site) : Bool :=
  match locate [] before.ops site.path, locate [] after.ops site.path with
  | some (scope, source), some (_, target) =>
    (site? scope site.path source).isSome &&
      ExactSyntax.operation source site.source && ExactSyntax.operation target site.target
  | _, _ => false

theorem pairCheck_sound {before after : Module} {site : Site}
    (checked : pairCheck before after site = true) :
    ∃ scope targetScope selected,
      locate [] before.ops site.path = some (scope, site.source) ∧
      locate [] after.ops site.path = some (targetScope, site.target) ∧
      site? scope site.path site.source = some selected := by
  unfold pairCheck at checked
  split at checked
  · next scope source targetScope target hs ht =>
    simp only [Bool.and_eq_true] at checked
    have sourceEq := ExactSyntax.operation_sound source site.source checked.1.2
    have targetEq := ExactSyntax.operation_sound target site.target checked.2
    subst source
    subst target
    cases found : site? scope site.path site.source with
    | none => simp [found] at checked
    | some selected => exact ⟨scope, targetScope, selected, hs, ht, found⟩
  · cases checked

def check (before after : Module) : Bool :=
  decide ((sites before).map (·.path) = candidates before) &&
    ExactSyntax.modules (reference before) after && (sites before).all (pairCheck before after)

theorem coverage {before after : Module} (accepted : check before after = true) :
    (sites before).map (·.path) = candidates before := by
  simp only [check, Bool.and_eq_true, decide_eq_true_eq] at accepted
  exact accepted.1.1

theorem accepted_pairs {before after : Module} (accepted : check before after = true) :
    ∀ site ∈ sites before, ∃ scope targetScope selected,
      locate [] before.ops site.path = some (scope, site.source) ∧
      locate [] after.ops site.path = some (targetScope, site.target) ∧
      site? scope site.path site.source = some selected := by
  simp only [check, Bool.and_eq_true] at accepted
  intro site member
  exact pairCheck_sound (List.all_eq_true.1 accepted.2 site member)

/-- Complete outputs, not sampled functions or a bag of local sites, must match the reference.
This equality retains all attributes, SSA bindings, operations, block arguments and regions. -/
theorem check_sound {before after : Module} (accepted : check before after = true) :
    after = reference before ∧
    ∀ site ∈ sites before, ∀ ctx call, ScalarSquareContract.Law ctx.M site.kind site.attrs →
      ∀ x : FBits site.kind,
        opSem ctx call site.source
          [.scalar (.some (.flt site.kind x)), .scalar (.some (.flt site.kind (ScalarSquareContract.two site.kind)))] [] =
        opSem ctx call site.target
          [.scalar (.some (.flt site.kind x)), .scalar (.some (.flt site.kind x))] [] := by
  simp only [check, Bool.and_eq_true] at accepted
  exact ⟨(ExactSyntax.modules_sound accepted.1.2).symm,
    fun site _ ctx call numerical x => site.scalar_sound ctx call numerical x⟩

/-- No surrounding primitive is silently declared supported. Complete-output equality to
the numerical reference holds under every operation interpretation, including errors. -/
theorem complete_output_sound {before after : Module} (accepted : check before after = true)
    (sem : OpSem) (ctx : Ctx) (fn : String) (args : List RVal) :
    denote sem ctx after fn args = denote sem ctx (reference before) fn args := by
  rw [(check_sound accepted).1]

#print axioms check_sound
#print axioms complete_output_sound
#print axioms Site.scalar_sound
#print axioms coverage
#print axioms pairCheck_sound
#print axioms accepted_pairs

end MerlinVerify.Check.ScalarSquare
