/- Exact deletion of a statically decoded, unused tensor.empty after scalar aliases.
Poison is a valid produced value; deleting a failing or still-used producer is forbidden. -/
import MerlinVerify.Check.StateNormalize
import MerlinVerify.Check.TypedDead

namespace MerlinVerify.Check.TensorEmptyDead

open MerlinVerify MerlinVerify.Sem.Ops

/-- Deliberately restricted to a default header and an actual static tensor result. -/
def total (s : Op) : Bool :=
  decide (s.name = "tensor.empty" ∧ s.operands = [] ∧ s.attrs = [] ∧ s.props = [] ∧
    s.regions = [] ∧ s.successors = [] ∧ s.results.length = 1) &&
    (Sem.Ops.Tensor.resultTensorTy s).isSome

/-- A decoded empty executes successfully in every environment, including zero extents.
No claim about initialized tensor contents is needed or made. -/
theorem total_exec {s : Op} (h : total s = true) (ctx : Ctx) (call : CallSem) (env : Env) :
    ∃ r value, s.results = [r] ∧
      stepOp opSem ctx call env s = .ok ((r.id, value) :: env) := by
  simp only [total, Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨hn, ho, ha, hp, hreg, hsucc, hlen⟩ := h.1
  obtain ⟨r, hr⟩ := List.length_eq_one_iff.mp hlen
  obtain ⟨⟨shape, elt⟩, hty⟩ := Option.isSome_iff_exists.mp h.2
  refine ⟨r, .tensor (Tensor.full shape elt .poison), hr, ?_⟩
  have hrty : r.ty.staticTensor? = some (shape, elt) := by
    simpa [Sem.Ops.Tensor.resultTensorTy, hr] using hty
  have hdialect : dialectOf "tensor.empty" = "tensor" := by cbv
  cases s with
  | mk name results operands attrs props regions successors =>
    simp only [Op.name, Op.operands, Op.attrs, Op.props, Op.regions, Op.successors,
      Op.results] at hn ho ha hp hreg hsucc hr
    subst name results operands attrs props regions successors
    simp [stepOp, Env.getAll, opSem, Op.blind, unmodelledAttr?, dispatch,
      Sem.Ops.Tensor.sem, Sem.Ops.Tensor.resultTensorTy, hrty, Env.bindAll, Outcome.mapM,
      Op.name, Op.results, Op.operands, Op.attrs, Op.props, Op.successors, hdialect]

def drop? (Γ : TEnv) (s : Op) (tail : List Op) : Bool :=
  total s && TypedDead.unused Γ s tail

theorem drop_sound {Γ : TEnv} {s : Op} {tail : List Op} (h : drop? Γ s tail = true)
    (ctx : Ctx) (call : CallSem) (env : Env) :
    ∃ env', stepOp opSem ctx call env s = .ok env' ∧
      evalOps opSem ctx call env' tail = evalOps opSem ctx call env tail := by
  simp only [drop?, Bool.and_eq_true] at h
  exact TypedDead.drop_sound h.2 (total_exec h.1 ctx call env)

def finish (Γ : TEnv) (s : Op) (tail : List Op) : List Op :=
  if drop? Γ s tail then tail else s :: tail

theorem finish_sound {Γ : TEnv} {s : Op} {tail : List Op} {ctx : Ctx} {call : CallSem}
    {env : Env} (ht : s.isTerminator = false) :
    evalOps opSem ctx call env (finish Γ s tail) = evalOps opSem ctx call env (s :: tail) := by
  unfold finish
  split
  · next hdrop =>
    obtain ⟨env', hs, heq⟩ := drop_sound hdrop ctx call env
    rw [evalOps_cons_step _ _ _ _ _ _ ht, hs, Outcome.bind_ok]
    exact heq.symm
  · rfl

/-- No facts are inferred at entry or transported across regions. The existing typed local
walker supplies the source environment and checks freshness across the entire continuation. -/
def family : StateNormalize.Family Unit (fun _ _ => True) where
  initial := ()
  initial_valid _ := True.intro
  advance _ _ _ _ := ()
  advance_sound _ _ _ _ _ _ _ _ _ _ _ _ := True.intro
  finish _ Γ _ _ s tail := finish Γ s tail
  finish_sound _ _ _ _ _ _ _ _ _ ht _ _ _ := finish_sound ht

def normalize (ib : Nat) (Γ : TEnv) : List Op → Option (List Op) :=
  StateNormalize.normalize family ib Γ () []

theorem normalize_sound {ib : Nat} (R : Rulebook opSem ib) (d : Nat) :
    ∀ Γ ss ns yields ctx call env,
      checkOps R d Γ ss yields = true → normalize ib Γ ss = some ns →
      EnvTyped ib Γ env → ctx.indexBits = ib → d ≤ ctx.loopIndex.length →
      evalOps opSem ctx call env ns = evalOps opSem ctx call env ss := by
  intro Γ ss ns yields ctx call env hc hn he hi hd
  exact StateNormalize.normalize_sound family R d Γ () [] ss ns yields ctx call env
    hc hn he True.intro hi hd

def base (ib : Nat) : SplatNormalizer.Normalizer ib := StateNormalize.base family ib []

end MerlinVerify.Check.TensorEmptyDead
