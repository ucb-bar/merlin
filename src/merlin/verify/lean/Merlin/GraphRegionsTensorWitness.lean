/- Concrete initialized inputs for source-execution instantiations. These are
inhabitation witnesses, not an additional native implementation contract. -/
import Merlin.GraphRegionsSourceMatcher
import Merlin.GraphRegionsSourceSyntax

namespace Merlin.GraphRegionsSource
open MerlinVerify

def filled (type : TensorType) (value : Nat) : Tensor :=
  Tensor.full type.shape (.int type.width) (.some (.bits type.width (BitVec.ofNat type.width value)))

theorem filled_initialized (type : TensorType) (value : Nat) : type.initialized (filled type value) := by
  refine ⟨rfl, rfl, ?_⟩
  intro point bounds
  refine ⟨BitVec.ofNat type.width value, ?_⟩
  simp [filled, Tensor.full, Tensor.canon, bounds]

theorem initialized_inputs (arguments : TensorTypes) (values : List Nat)
    (arity : values.length = arguments.length) :
    InputsDomain arguments ((arguments.zip values).map fun (pair, value) => filled pair.2 value) := by
  induction arguments generalizing values with
  | nil => simp_all [InputsDomain]
  | cons pair arguments ih =>
    cases values with
    | nil => simp at arity
    | cons value values =>
      exact List.Forall₂.cons (filled_initialized _ _) (ih values (by simpa using arity))

/-- The structural trace is only an inequality discriminator, never acceptance.
No injectivity or serialization equivalence premise is used. -/
theorem checkSource_refuse_changed {before after : Module} {program : Program} {graph : GraphRegions.Graph}
    (different : sourceTrace after ≠ sourceTrace program.module) :
    checkSource before after program graph = false := by
  apply Bool.eq_false_iff.mpr
  intro accepted
  exact different (congrArg sourceTrace (checkSource_iff.mp accepted).1.1.2)

end Merlin.GraphRegionsSource
