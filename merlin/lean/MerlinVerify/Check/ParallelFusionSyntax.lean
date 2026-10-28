/- Recompute the bounded fusion candidate from the actual operations. This initial family has
one tensor input per loop and one unflagged primitive using that scalar argument twice. -/
import MerlinVerify.Check.ParallelFusionRuntime
import MerlinVerify.Check.StateNormalize

namespace MerlinVerify.Check.ParallelFusionSyntax

open MerlinVerify MerlinVerify.Sem.Ops MerlinVerify.Sem.Ops.Linalg
open ParallelFusionBody ParallelFusionRuntime MerlinVerify.Rules.ParallelFusionScalar

private def valueDecEq (a b : Value) : Decidable (a = b) :=
  decidable_of_iff (a.id = b.id ∧ a.ty = b.ty) (by cases a; cases b; simp)
local instance : DecidableEq Value := valueDecEq

structure Body where
  input : Value
  old : Value
  primitive : Primitive
  terminator : Op

def Body.region (b : Body) : Region :=
  .mk [.mk [b.input, b.old] [b.primitive.operation, b.terminator]]

def body? : Region → Option Body
  | .mk [.mk [input, old] [op, term]] => do
    let [result] := op.results | none
    let kind ← kind? op.name
    pure ⟨input, old, ⟨kind, op, result⟩, term⟩
  | _ => none

theorem body?_region {source : Region} {b : Body} (h : body? source = some b) :
    b.region = source := by
  unfold body? at h
  split at h
  · next input old op term =>
    split at h
    · next result hr =>
      cases hk : kind? op.name with
      | none => simp [hk] at h
      | some kind => simp only [hk, bind, Option.bind, Option.pure_def, Option.some.injEq] at h
                     subst b; rfl
    · cases h
  · cases h

def Body.Valid (b : Body) (w : Nat) : Prop :=
  b.primitive.Valid w ∧ b.input.ty = .int w ∧ b.old.ty = .int w ∧
  b.primitive.operation.operands = [b.input.id, b.input.id] ∧
  b.primitive.operation.successors = [] ∧ Arith.fastmathKey b.primitive.operation = "0" ∧
  hasUnit b.primitive.operation "nonNeg" = false ∧ hasUnit b.primitive.operation "isExact" = false ∧
  hasUnit b.primitive.operation "disjoint" = false ∧
  b.terminator.name = "linalg.yield" ∧ b.terminator.operands = [b.primitive.result.id] ∧
  b.terminator.results = [] ∧ b.terminator.regions = [] ∧ b.terminator.successors = [] ∧
  b.terminator.attrs = [] ∧ b.terminator.props = []

instance (b : Body) (w : Nat) : Decidable (b.Valid w) := by
  unfold Body.Valid Primitive.Valid
  infer_instance

theorem Body.valid_core {b : Body} {w : Nat} (h : b.Valid w) :
    b.primitive.Valid w ∧ b.primitive.operation.operands = [b.input.id, b.input.id] ∧
      b.terminator.name = "linalg.yield" ∧ b.terminator.operands = [b.primitive.result.id] := by
  obtain ⟨hp, _, _, ho, _, _, _, _, _, hn, ht, _⟩ := h
  exact ⟨hp, ho, hn, ht⟩

structure Generic where
  operation : Op
  input : ValueId
  destination : ValueId
  result : Value
  shape : List Nat
  width : Nat
  body : Body

def generic? (ib : Nat) (op : Op) : Option Generic := do
  let [input, destination] := op.operands | none
  let [result] := op.results | none
  let [source] := op.regions | none
  let body ← body? source
  let some (.tensor shape (.int width)) := vtyOf ib result.ty | none
  pure ⟨op, input, destination, result, shape, width, body⟩

theorem generic?_spec {ib : Nat} {op : Op} {g : Generic} (h : generic? ib op = some g) :
    g.operation = op ∧ op.operands = [g.input, g.destination] ∧ op.results = [g.result] ∧
      op.regions = [g.body.region] ∧ vtyOf ib g.result.ty = some (.tensor g.shape (.int g.width)) := by
  unfold generic? at h
  split at h
  · next input destination ho =>
    split at h
    · next result hr =>
      split at h
      · next source hregions =>
        cases hb : body? source with
        | none => simp [hb] at h
        | some body =>
          simp only [hb, bind, Option.bind] at h
          split at h
          · next shape width ht =>
            simp only [Option.pure_def, Option.some.injEq] at h
            subst g
            exact ⟨rfl, ho, hr, by simpa [body?_region hb] using hregions, ht⟩
          · cases h
      · cases h
    · cases h
  · cases h

def Generic.Valid (g : Generic) : Prop :=
  Header g.operation g.shape g.result ∧ g.operation.successors = [] ∧
  g.body.Valid g.width ∧ 0 < g.width

instance (g : Generic) : Decidable g.Valid := by
  unfold Generic.Valid Header
  infer_instance

def localIds (g : Generic) : List ValueId :=
  [g.body.input.id, g.body.old.id, g.body.primitive.result.id]

def fusedRegion (p c : Generic) : Region :=
  .mk [.mk [p.body.input, p.body.old]
    [p.body.primitive.operation,
     withOperands c.body.primitive.operation [p.body.primitive.result.id, p.body.primitive.result.id],
     c.body.terminator]]

def empty (p : Generic) : Op := .mk "tensor.empty" [p.result] [] [] [] [] []

def fused (p c : Generic) : Op :=
  SplatNested.withRegions (withOperands c.operation [p.input, p.result.id]) [fusedRegion p c]

/-- A disappeared producer may not escape through any later operation or nested capture. The
alpha check can admit a later fresh rebinding, but cannot reuse stale producer facts. -/
def Legal (Γ : TEnv) (p c : Generic) (tail : List Op) : Prop :=
  p.Valid ∧ c.Valid ∧ c.shape = p.shape ∧ c.width = p.width ∧ c.input = p.result.id ∧
  p.input ≠ p.result.id ∧ c.destination ≠ p.result.id ∧
  Γ.lookup p.input = some (.tensor p.shape (.int p.width)) ∧
  Γ.lookup p.destination = some (.tensor p.shape (.int p.width)) ∧
  Γ.lookup c.destination = some (.tensor p.shape (.int p.width)) ∧
  freshIds (SplatNormalize.diagonal Γ)
    ([p.result.id, c.result.id] ++ localIds p ++ localIds c) = true ∧
  alphaOps ((c.result.id, c.result.id) :: SplatNormalize.diagonal Γ) tail tail = true

instance (Γ : TEnv) (p c : Generic) (tail : List Op) : Decidable (Legal Γ p c tail) := by
  unfold Legal
  infer_instance

def attempt (ib : Nat) (Γ : TEnv) (first second : Op) (tail : List Op) : Option (List Op) := do
  let producer ← generic? ib first
  let consumer ← generic? ib second
  if Legal Γ producer consumer tail then some (empty producer :: fused producer consumer :: tail)
  else none

end MerlinVerify.Check.ParallelFusionSyntax
