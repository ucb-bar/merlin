/- Propose lexical names for internal SSA bindings, then validate the entire proposal with
the proved alpha checker. Entry bindings and actual source/target snapshots remain unchanged. -/
import MerlinVerify.Check.TypedNormalizer
import MerlinVerify.Check.SplatNormalize
import MerlinVerify.Check.RegionDepth

namespace MerlinVerify.Check.AlphaNormalize

open MerlinVerify MerlinVerify.Sem.Ops

def freshValues (next : Nat) (values : List Value) : List Value :=
  values.zipIdx |>.map fun (value, offset) => { value with id := next + offset }

def extend (ρ : Rho) (source target : List Value) : Rho :=
  source.map Value.id |>.zip (target.map Value.id) |>.append ρ

/-- Region-local arguments are visible only in their own block. The allocation counter is
global to the candidate; bindings from a sibling block never enter the correspondence. -/
def inRegions (body : Rho → Nat → List Op → List Op × Nat)
    (ρ : Rho) (next : Nat) (regions : List Region) : List Region × Nat :=
  let (reversed, next) := regions.foldl (fun (done, next) region =>
    let (blocks, next) := region.blocks.foldl (fun (done, next) block =>
      let args := freshValues next block.args
      let (ops, next) := body (extend ρ block.args args) (next + args.length) block.ops
      (Block.mk args ops :: done, next)) (([] : List Block), next)
    (Region.mk blocks.reverse :: done, next)) (([] : List Region), next)
  (reversed.reverse, next)

/-- Results reserve their lexical names before regions are visited, but are bound only for
the following operations. A result is never made visible inside its own operation's regions. -/
def flat (descend : Rho → Nat → List Region → List Region × Nat)
    (ρ : Rho) (next : Nat) (ops : List Op) : List Op × Nat :=
  let (reversed, _, next) := ops.foldl (fun (done, known, next) op =>
    let results := freshValues next op.results
    let (regions, next) := descend known (next + results.length) op.regions
    let operands := op.operands.map fun id => (known.find id).getD id
    let renamed := Op.mk op.name results operands op.attrs op.props regions op.successors
    (renamed :: done, extend known op.results results, next)) (([] : List Op), ρ, next)
  (reversed.reverse, next)

/-- This is an untrusted candidate generator. Insufficient fuel or an unbound use can only
make alpha validation refuse. The actual AST height supplies the normal traversal budget. -/
def operations : Nat → Rho → Nat → List Op → List Op × Nat
  | 0, ρ, next, ops => flat (fun _ next regions => (regions, next)) ρ next ops
  | fuel + 1, ρ, next, ops => flat (inRegions (operations fuel)) ρ next ops

def firstFresh (Γ : TEnv) : Nat :=
  Γ.foldl (fun next binding => max next (binding.1 + 1)) 0

def candidate (Γ : TEnv) (ops : List Op) : List Op :=
  (operations (RegionDepth.ops ops) (SplatNormalize.diagonal Γ) (firstFresh Γ) ops).1

/-- Every accepted renaming preserves complete outcomes on arbitrary environments. No typing
or successful-execution premise is needed by alpha equivalence. -/
def normalize (Γ : TEnv) (ops : List Op) : Option (List Op) :=
  let renamed := candidate Γ ops
  if alphaOps (SplatNormalize.diagonal Γ) ops renamed then some renamed else none

theorem normalize_sound {Γ : TEnv} {ops renamed : List Op}
    (h : normalize Γ ops = some renamed) (ctx : Ctx) (call : CallSem) (env : Env) :
    evalOps opSem ctx call env renamed = evalOps opSem ctx call env ops := by
  unfold normalize at h
  dsimp only at h
  split at h
  · cases h
    exact (alphaOps_sound call call _ ops _ ctx env env (by assumption)
      (SplatNormalize.diagonal_refl Γ env) (fun _ _ => rfl)).symm
  · cases h

/-- Place this after all alias/scope changes and before numeric-ID operand ordering. Keeping
entry IDs does not promise a unique form for arbitrary permutations of entry identifiers. -/
def normalizer : TypedNormalizer where
  run _ _ Γ ops _ _ := normalize Γ ops
  sound _ _ _ _ _ _ _ ctx call env _ hn _ _ _ := normalize_sound hn ctx call env

end MerlinVerify.Check.AlphaNormalize
