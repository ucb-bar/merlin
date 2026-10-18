/- Reusable exact SSA alias substitution. The syntax walk is only a proposal: alpha checking
validates all captures, region arguments, rebinding, and escaping uses before it is accepted. -/
import MerlinVerify.Check.SplatNormalize
import MerlinVerify.Rules.Cse

namespace MerlinVerify.Check.TypedAlias

open MerlinVerify MerlinVerify.Sem.Ops

mutual
  def substituteOps (oldId newId : ValueId) : List Op → List Op
    | [] => []
    | s :: ss => substituteOp oldId newId s :: substituteOps oldId newId ss
  def substituteOp (oldId newId : ValueId) : Op → Op
    | .mk name results operands attrs props regions successors =>
      .mk name results (operands.map fun x => if x == oldId then newId else x) attrs props
        (substituteRegions oldId newId regions) successors
  def substituteRegions (oldId newId : ValueId) : List Region → List Region
    | [] => []
    | .mk blocks :: rest => .mk (substituteBlocks oldId newId blocks) :: substituteRegions oldId newId rest
  def substituteBlocks (oldId newId : ValueId) : List Block → List Block
    | [] => []
    | .mk args ops :: rest => .mk args (substituteOps oldId newId ops) :: substituteBlocks oldId newId rest
end

/-- The caller proves the producer returns an actual existing payload value. Alpha agreement
then checks the complete continuation against the proposed alias, including malformed SSA. -/
theorem continuation {Γ : TEnv} {oldId newId : ValueId} {tail renamed : List Op}
    (ha : alphaOps ((oldId, newId) :: SplatNormalize.diagonal Γ) tail renamed = true)
    {ctx : Ctx} {call : CallSem} {env : Env} {value : RVal}
    (hv : env.get? newId = some value) :
    evalOps opSem ctx call env renamed = evalOps opSem ctx call ((oldId, value) :: env) tail :=
  (alphaOps_sound call call _ tail renamed ctx _ _ ha
    (Rules.Cse.Corr.bindAlias (SplatNormalize.diagonal_refl Γ env) hv) (fun _ _ => rfl)).symm

def rename? (Γ : TEnv) (oldId newId : ValueId) (tail : List Op) : Option (List Op) :=
  let renamed := substituteOps oldId newId tail
  if alphaOps ((oldId, newId) :: SplatNormalize.diagonal Γ) tail renamed then some renamed else none

theorem rename_sound {Γ : TEnv} {oldId newId : ValueId} {tail renamed : List Op}
    (hr : rename? Γ oldId newId tail = some renamed)
    {ctx : Ctx} {call : CallSem} {env : Env} {value : RVal}
    (hv : env.get? newId = some value) :
    evalOps opSem ctx call env renamed = evalOps opSem ctx call ((oldId, value) :: env) tail := by
  unfold rename? at hr
  dsimp only at hr
  split at hr
  · cases hr
    exact continuation (by assumption) hv
  · cases hr

end MerlinVerify.Check.TypedAlias
