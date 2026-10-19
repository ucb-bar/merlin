/- Lift an unconditional exact operation-list transform through every region. No type, shape,
fact, initializedness or successful-execution premise is introduced by the traversal. -/
import MerlinVerify.Check.TypedNormalizer
import MerlinVerify.Check.RegionDepth

namespace MerlinVerify.Check.RegionExact

open MerlinVerify MerlinVerify.Sem.Ops

/-- The local transformation must preserve the full outcome, including stuck reasons and UB. -/
def Exact (F : List Op → List Op) : Prop :=
  ∀ ctx call env ops, evalOps opSem ctx call env (F ops) = evalOps opSem ctx call env ops

/-- Multiple local exact families can share one traversal. Place this stage after aliases;
unconditional semantic equality alone does not make a chosen syntactic order alias stable. -/
theorem compose {F G : List Op → List Op} (hF : Exact F) (hG : Exact G) :
    Exact (G ∘ F) := by
  intro ctx call env ops
  exact (hG ctx call env _).trans (hF ctx call env ops)

def region (F : List Op → List Op) : Region → Region
  | .mk blocks => .mk (blocks.map fun | .mk args ops => .mk args (F ops))

theorem region_equal {F : List Op → List Op} (hF : Exact F)
    (ctx : Ctx) (call : CallSem) (env : Env) (r : Region) :
    evalRegion opSem ctx call env (region F r) = evalRegion opSem ctx call env r := by
  cases r with
  | mk blocks =>
    cases blocks with
    | nil => rfl
    | cons b rest =>
      cases rest with
      | cons _ _ => rfl
      | nil =>
        cases b with
        | mk args ops =>
          funext vals
          simp only [region, List.map_cons, List.map_nil, evalRegion, evalBlocks, evalBlock]
          congr 1
          funext bound
          exact hF ctx call bound ops

theorem regions_equal {F : List Op → List Op} (hF : Exact F)
    (call : CallSem) (env : Env) (rs : List Region) :
    evalRegions opSem call env (rs.map (region F)) = evalRegions opSem call env rs := by
  induction rs with
  | nil => rfl
  | cons r rs ih =>
    simp only [List.map_cons, evalRegions, ih]
    congr 1
    funext ctx
    exact region_equal hF ctx call env r

/-- Retain the actual header, bindings and operands. Only region bodies are transformed. -/
def operation (F : List Op → List Op) : Op → Op
  | .mk n results operands attrs props rs successors =>
    .mk n results operands attrs props (rs.map (region F)) successors

@[simp] theorem operation_terminator (F : List Op → List Op) (op : Op) :
    (operation F op).isTerminator = op.isTerminator := by cases op; rfl

@[simp] theorem operation_operands (F : List Op → List Op) (op : Op) :
    (operation F op).operands = op.operands := by cases op; rfl

@[simp] theorem operation_blind (F : List Op → List Op) (op : Op) :
    (operation F op).blind = op.blind := by cases op; rfl

theorem operation_step {F : List Op → List Op} (hF : Exact F)
    (ctx : Ctx) (call : CallSem) (env : Env) (op : Op) :
    stepOp opSem ctx call env (operation F op) = stepOp opSem ctx call env op := by
  cases op with
  | mk n results operands attrs props rs successors =>
    simp only [stepOp, operation, Op.operands, Op.results, Op.regions, regions_equal hF]
    congr 1

def lift (F : List Op → List Op) (ops : List Op) : List Op := ops.map (operation F)

theorem lift_equal {F : List Op → List Op} (hF : Exact F) : Exact (lift F) := by
  intro ctx call env ops
  induction ops generalizing env with
  | nil => rfl
  | cons op ops ih =>
    simp only [lift, List.map_cons]
    by_cases ht : op.isTerminator = true
    · rw [evalOps_cons_term _ _ _ _ _ _ (by simpa using ht),
        evalOps_cons_term _ _ _ _ _ _ ht]
      simp only [List.isEmpty_map, operation_operands]
    · have ht' : op.isTerminator = false := by simpa using ht
      rw [evalOps_cons_step _ _ _ _ _ _ (by simpa using ht'),
        evalOps_cons_step _ _ _ _ _ _ ht', operation_step hF]
      cases stepOp opSem ctx call env op with
      | ub => rfl
      | stuck _ => rfl
      | ok next => exact ih next

/-- Every budget is sound. A structural-height budget visits all original regions when F is
region preserving (e.g. operand ordering). Arbitrary exact F need not have that coverage property,
and no idempotence or stability under later aliases is claimed. -/
def recursive (F : List Op → List Op) : Nat → List Op → List Op
  | 0, ops => F ops
  | fuel + 1, ops => F (lift (recursive F fuel) ops)

theorem recursive_equal {F : List Op → List Op} (hF : Exact F) (fuel : Nat) :
    Exact (recursive F fuel) := by
  induction fuel with
  | zero => exact hF
  | succ fuel ih =>
    intro ctx call env ops
    exact (hF ctx call env _).trans (lift_equal ih ctx call env ops)

/-- The adapter uses the actual source AST height. Hints and typing evidence are unnecessary for
this unconditional family; the surrounding checker still independently checks both actual sides. -/
def normalizer (F : List Op → List Op) (hF : Exact F) : TypedNormalizer where
  run _ _ _ ops _ _ := some (recursive F (RegionDepth.ops ops) ops)
  sound _ _ _ ops _ normalized _ ctx call env _ hn _ _ _ := by
    cases hn
    exact recursive_equal hF (RegionDepth.ops ops) ctx call env ops

end MerlinVerify.Check.RegionExact
