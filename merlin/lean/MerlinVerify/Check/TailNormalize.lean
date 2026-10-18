/- Exact typed local normalization with continuation-sensitive edits. Families justify alias
substitution, total deletion, or a changed prefix; one induction owns source facts and typing. -/
import MerlinVerify.Check.BindingFacts
import MerlinVerify.Check.StateNormalize

namespace MerlinVerify.Check.TailNormalize

open MerlinVerify MerlinVerify.Sem.Ops

structure Family (F : Type) (model : BindingFacts.Model F) where
  advance : Nat → TEnv → List F → Op → List F
  advance_sound : ∀ ib Γ facts s ctx call env env',
    EnvTyped ib Γ env → BindingFacts.Valid model facts env → ctx.indexBits = ib →
    stepOp opSem ctx call env s = .ok env' →
    BindingFacts.Valid model (advance ib Γ facts s) env'
  finish : Nat → TEnv → List F → List Op → Op → List Op → List Op
  finish_sound : ∀ ib Γ facts hints s tail ctx call env,
    s.isTerminator = false → EnvTyped ib Γ env → BindingFacts.Valid model facts env →
    ctx.indexBits = ib →
    evalOps opSem ctx call env (finish ib Γ facts hints s tail) =
      evalOps opSem ctx call env (s :: tail)

variable {F : Type} {model : BindingFacts.Model F}

/-- The list-fact interface keeps its original API while sharing the general state induction. -/
def stateFamily (E : Family F model) : StateNormalize.Family (List F) (BindingFacts.Valid model) where
  initial := []
  initial_valid _ f hf := by cases hf
  advance := E.advance
  advance_sound := E.advance_sound
  finish := E.finish
  finish_sound := E.finish_sound

def normalize (E : Family F model) (ib : Nat) (Γ : TEnv) (facts : List F) (hints : List Op) :
    List Op → Option (List Op) := StateNormalize.normalize (stateFamily E) ib Γ facts hints

/-- Every fact is justified by successful original execution. Continuation edits must be exact
for the actual environment, so unknown source failures cannot silently become total. -/
theorem normalize_sound (E : Family F model) {ib : Nat} (R : Rulebook opSem ib) (d : Nat) :
    ∀ Γ facts hints ss ns yields ctx call env,
      checkOps R d Γ ss yields = true → normalize E ib Γ facts hints ss = some ns →
      EnvTyped ib Γ env → BindingFacts.Valid model facts env →
      ctx.indexBits = ib → d ≤ ctx.loopIndex.length →
      evalOps opSem ctx call env ns = evalOps opSem ctx call env ss := by
  intro Γ facts hints ss ns yields ctx call env hc hn he hf hi hd
  exact StateNormalize.normalize_sound (stateFamily E) R d Γ facts hints ss ns yields
    ctx call env hc hn he hf hi hd

/-- Start every block without invented facts. BodyNormalize supplies the separate structured
region traversal and actual local typing; family facts are not inferred from captures. -/
def base (E : Family F model) (ib : Nat) (hints : List Op) : SplatNormalizer.Normalizer ib :=
  StateNormalize.base (stateFamily E) ib hints

end MerlinVerify.Check.TailNormalize
