/- Exact normalization for adjacent single-input, identity-map, fully overwriting parallel loops.
Each body initially contains one unflagged fixed-integer primitive reading its input twice.
Captured scalar reads, distinct operands, reductions, producer escapes and other layouts abstain. -/
import MerlinVerify.Check.ParallelFusionProof
import MerlinVerify.Check.RegionDepth
import MerlinVerify.Check.TypedNormalizer

namespace MerlinVerify.Check.ParallelFusion

open MerlinVerify MerlinVerify.Sem.Ops
open ParallelFusionSyntax ParallelFusionProof

theorem attempt_sound {ib : Nat} {Γ : TEnv} {first second : Op} {tail normalized : List Op}
    (h : attempt ib Γ first second tail = some normalized)
    {ctx : Ctx} {call : CallSem} {env : Env} (he : EnvTyped ib Γ env) :
    evalOps opSem ctx call env normalized = evalOps opSem ctx call env (first :: second :: tail) := by
  unfold attempt at h
  cases hp : generic? ib first with
  | none => simp [hp] at h
  | some p =>
    simp only [hp, bind, Option.bind] at h
    cases hc : generic? ib second with
    | none => simp [hc] at h
    | some c =>
      simp only [hc, bind, Option.bind] at h
      split at h
      · next hl =>
        cases h
        obtain ⟨hop, hpO, _, hpR, hpT⟩ := generic?_spec hp
        obtain ⟨hoc, hcO, _, hcR, _⟩ := generic?_spec hc
        have hpO' : p.operation.operands = [p.input, p.destination] := by simpa [hop] using hpO
        have hcO' : c.operation.operands = [c.input, c.destination] := by simpa [hoc] using hcO
        have hpR' : p.operation.regions = [p.body.region] := by simpa [hop] using hpR
        have hcR' : c.operation.regions = [c.body.region] := by simpa [hoc] using hcR
        simpa only [hop, hoc] using pair_sound hl hpO' hcO' hpR' hcR' hpT he
      · cases h

def finish (ib : Nat) (Γ : TEnv) (op : Op) : List Op → List Op
  | [] => [op]
  | second :: tail => (attempt ib Γ op second tail).getD (op :: second :: tail)

theorem finish_sound (ib : Nat) (Γ : TEnv) (op : Op) (tail : List Op)
    (ctx : Ctx) (call : CallSem) (env : Env) (he : EnvTyped ib Γ env) :
    evalOps opSem ctx call env (finish ib Γ op tail) = evalOps opSem ctx call env (op :: tail) := by
  cases tail with
  | nil => rfl
  | cons second tail =>
    change evalOps opSem ctx call env ((attempt ib Γ op second tail).getD (op :: second :: tail)) = _
    cases h : attempt ib Γ op second tail with
    | none => rw [Option.getD_none]
    | some normalized => exact attempt_sound h he

/-- The shared continuation-sensitive walker advances types through successful original steps.
No producer facts are guessed, and no target syntax is used as semantic evidence. -/
def family : StateNormalize.Family Unit (fun _ _ => True) where
  initial := ()
  initial_valid _ := trivial
  advance _ _ _ _ := ()
  advance_sound _ _ _ _ _ _ _ _ _ _ _ _ := trivial
  finish ib Γ _ _ op tail := finish ib Γ op tail
  finish_sound ib Γ _ _ op tail ctx call env _ he _ _ := finish_sound ib Γ op tail ctx call env he

def base (ib : Nat) : SplatNormalizer.Normalizer ib := StateNormalize.base family ib []

def normalizer : TypedNormalizer where
  run ib d Γ ops hints yields :=
    (BodyNormalize.recursive (base ib) (max (RegionDepth.ops ops) (RegionDepth.ops hints))).run
      d Γ [] ops yields
  sound ib d Γ ops hints normalized yields ctx call env hc hn he hi hd :=
    (BodyNormalize.recursive (base ib) (max (RegionDepth.ops ops) (RegionDepth.ops hints))).sound
      d Γ [] ops normalized yields ctx call env hc hn he (by intro f hf; cases hf) hi hd

end MerlinVerify.Check.ParallelFusion
