/- Erasing reviewed semantics-free parent annotations for the source softmax segment.
Numerical attributes, properties, operand identities and nested bodies are preserved. -/
import MerlinVerify.Rules.W8A8SourceSemantics
import MerlinVerify.Check.ProgramLemmas

namespace MerlinVerify.Rules.W8A8AttentionAnnotations

open MerlinVerify Sem.Ops

def semantic (attrs : List (String × Attr)) : List (String × Attr) :=
  attrs.filter fun entry => !Spec.Annotations.semanticsFree entry.1

def clean (op : Op) : Op :=
  .mk op.name op.results op.operands (semantic op.attrs) (semantic op.props) op.regions op.successors

/-- This normalization is proved only for the seven parent operation families used here. -/
def Allowed (op : Op) : Prop :=
  op.name = "arith.constant" ∨ op.name = "tensor.splat" ∨ op.name = "tensor.empty" ∨
  op.name = "tensor.collapse_shape" ∨ op.name = "tensor.expand_shape" ∨
  op.name = "linalg.generic" ∨ op.name = "linalg.reduce"

instance (op : Op) : Decidable (Allowed op) := inferInstanceAs (Decidable (_ ∨ _ ∨ _ ∨ _ ∨ _ ∨ _ ∨ _))

@[simp] theorem clean_name (op : Op) : (clean op).name = op.name := rfl
@[simp] theorem clean_results (op : Op) : (clean op).results = op.results := rfl
@[simp] theorem clean_operands (op : Op) : (clean op).operands = op.operands := rfl
@[simp] theorem clean_regions (op : Op) : (clean op).regions = op.regions := rfl
@[simp] theorem clean_successors (op : Op) : (clean op).successors = op.successors := rfl

theorem lookup_semantic (attrs : List (String × Attr)) {key : String}
    (meaningful : Spec.Annotations.semanticsFree key = false) :
    (semantic attrs).lookup key = attrs.lookup key := by
  induction attrs with
  | nil => rfl
  | cons entry rest induction =>
    obtain ⟨name, value⟩ := entry
    by_cases free : Spec.Annotations.semanticsFree name = true
    · have different : key ≠ name := by
        intro same
        rw [same, free] at meaningful
        cases meaningful
      have unequal : (key == name) = false := by simpa using different
      simp [semantic, free, List.lookup, unequal, ← induction]
    · have kept : Spec.Annotations.semanticsFree name = false := by simpa using free
      by_cases same : key = name
      · subst name
        simp [semantic, kept, List.lookup]
      · have unequal : (key == name) = false := by simpa using same
        simp [semantic, kept, List.lookup, unequal, ← induction]

theorem attr_clean (op : Op) {key : String} (meaningful : Spec.Annotations.semanticsFree key = false) :
    (clean op).attr? key = op.attr? key := by
  simp only [Op.attr?, clean, Op.props, Op.attrs, lookup_semantic _ meaningful]

private theorem unmodelled_list (name : String) (attrs : List (String × Attr)) :
    (semantic attrs).findSome? (fun (key, _) =>
      if (interpretedAttrs name).contains key || Spec.Annotations.semanticsFree key then none else some key) =
    attrs.findSome? (fun (key, _) =>
      if (interpretedAttrs name).contains key || Spec.Annotations.semanticsFree key then none else some key) := by
  induction attrs with
  | nil => rfl
  | cons entry rest induction =>
    obtain ⟨key, value⟩ := entry
    simp [semantic] at induction
    by_cases free : Spec.Annotations.semanticsFree key = true
    · simp [semantic, free, induction]
    · have kept : Spec.Annotations.semanticsFree key = false := by simpa using free
      simp [semantic, kept, induction, List.findSome?]

theorem unmodelled_clean (op : Op) : unmodelledAttr? (clean op) = unmodelledAttr? op := by
  unfold unmodelledAttr?
  change (semantic op.props ++ semantic op.attrs).findSome? _ = _
  simp only [semantic, clean_name]
  rw [← List.filter_append]
  exact unmodelled_list op.name (op.props ++ op.attrs)

theorem clean_blind (op : Op) : (clean op).blind = clean op.blind := by cases op; rfl

private theorem callee_clean (op : Op) : calleeOf (clean op) = calleeOf op := by
  unfold calleeOf
  rw [attr_clean op (by cbv : Spec.Annotations.semanticsFree "callee" = false)]

private theorem dispatch_clean (ctx : Ctx) (call : CallSem) (op : Op)
    (args : List RVal) (regions : List (Ctx → RegionSem)) (allowed : Allowed op) :
    dispatch ctx call (clean op) args regions = dispatch ctx call op args regions := by
  have value := attr_clean op (by cbv : Spec.Annotations.semanticsFree "value" = false)
  have reassociation := attr_clean op (by cbv : Spec.Annotations.semanticsFree "reassociation" = false)
  have segments := attr_clean op (by cbv : Spec.Annotations.semanticsFree "operandSegmentSizes" = false)
  have maps := attr_clean op (by cbv : Spec.Annotations.semanticsFree "indexing_maps" = false)
  have iterators := attr_clean op (by cbv : Spec.Annotations.semanticsFree "iterator_types" = false)
  have dimensions := attr_clean op (by cbv : Spec.Annotations.semanticsFree "dimensions" = false)
  rcases allowed with named | named | named | named | named | named | named
  · have dialect : dialectOf "arith.constant" = "arith" := by cbv
    simp [dispatch, dialect, named, Arith.sem, value]
  · have dialect : dialectOf "tensor.splat" = "tensor" := by cbv
    simp [dispatch, dialect, named, Tensor.sem, Tensor.resultTensorTy]
  · have dialect : dialectOf "tensor.empty" = "tensor" := by cbv
    simp [dispatch, dialect, named, Tensor.sem, Tensor.resultTensorTy]
  · have dialect : dialectOf "tensor.collapse_shape" = "tensor" := by cbv
    simp [dispatch, dialect, named, Tensor.sem, Tensor.resultTensorTy, Tensor.reassociation?, reassociation]
  · have dialect : dialectOf "tensor.expand_shape" = "tensor" := by cbv
    simp [dispatch, dialect, named, Tensor.sem, Tensor.resultTensorTy, Tensor.reassociation?, reassociation]
  · have dialect : dialectOf "linalg.generic" = "linalg" := by cbv
    simp [dispatch, dialect, named, Linalg.sem, Linalg.splitInsOuts, attrDenseArray?, attrAffineMaps?,
      Linalg.iterKinds?, segments, maps, iterators]
    rfl
  · have dialect : dialectOf "linalg.reduce" = "linalg" := by cbv
    simp [dispatch, dialect, named, LinalgNamed.sem, LinalgNamed.natArray?, Linalg.splitInsOuts,
      attrDenseArray?, segments, dimensions, LinalgNamed.run]

theorem source_clean (ctx : Ctx) (call : CallSem) (op : Op)
    (args : List RVal) (regions : List (Ctx → RegionSem)) (allowed : Allowed op) :
    W8A8SourceSemantics.opSem ctx call (clean op) args regions =
      W8A8SourceSemantics.opSem ctx call op args regions := by
  have ordinary : W8A8SourceSemantics.isSource op.name = false := by
    rcases allowed with named | named | named | named | named | named | named <;>
      simp [W8A8SourceSemantics.isSource, W8A8SourceContract.name, named]
  rw [W8A8SourceSemantics.unchanged (by simpa using ordinary), W8A8SourceSemantics.unchanged ordinary]
  simp only [Sem.Ops.opSem, clean_blind, unmodelled_clean, callee_clean]
  split
  · rfl
  · exact dispatch_clean _ _ op.blind args regions (by simpa [Allowed] using allowed)

/-- Parent erasure preserves the complete evaluation, including failure and poison behavior. -/
theorem eval_clean (ctx : Ctx) (call : CallSem) (ops : List Op)
    (allowed : ∀ op ∈ ops, Allowed op ∧ op.isTerminator = false) :
    ∀ (env : Env) (rest : List Op),
      evalOps W8A8SourceSemantics.opSem ctx call env (ops.map clean ++ rest) =
      evalOps W8A8SourceSemantics.opSem ctx call env (ops ++ rest) := by
  induction ops with
  | nil => intro env rest; rfl
  | cons op ops induction =>
    intro env rest
    have head := allowed op (by simp)
    have tail : ∀ child ∈ ops, Allowed child ∧ child.isTerminator = false :=
      fun child member => allowed child (by simp [member])
    have nonterm : (clean op).isTerminator = false := by simpa [Op.isTerminator] using head.2
    simp only [List.map_cons, List.cons_append]
    rw [evalOps_cons_nonterm _ _ _ _ _ _ nonterm, evalOps_cons_nonterm _ _ _ _ _ _ head.2]
    simp only [clean_operands, clean_results, clean_regions]
    congr 1
    funext args
    rw [source_clean _ _ _ _ _ head.1]
    congr 1
    funext results
    congr 1
    funext next
    exact induction tail next rest

end MerlinVerify.Rules.W8A8AttentionAnnotations
