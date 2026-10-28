/- Source-dispatch facts for the canonical attention segment. These retain static
shape/reassociation checks and evaluate the actual scalar regions in their environment. -/
import MerlinVerify.Rules.W8A8AttentionTemplate
import MerlinVerify.Rules.W8A8AttentionSum
import MerlinVerify.Rules.W8A8AttentionMapTensor

namespace MerlinVerify.Rules.W8A8AttentionTemplateOps

open MerlinVerify Check Sem.Ops Sem.Ops.Linalg W8A8AttentionTemplate

theorem static_tensor (shape : List Nat) : (tensorTy shape).staticTensor? = some (shape, .float .f32) := by
  have dimensions : (shape.map some).mapM id = some shape := by
    induction shape with
    | nil => rfl
    | cons head tail induction => simp [induction]
  simp [tensorTy, Ty.staticTensor?, dimensions]

private theorem dialect_constant : dialectOf "arith.constant" = "arith" := by cbv
private theorem dialect_splat : dialectOf "tensor.splat" = "tensor" := by cbv
private theorem dialect_empty : dialectOf "tensor.empty" = "tensor" := by cbv
private theorem dialect_collapse : dialectOf "tensor.collapse_shape" = "tensor" := by cbv
private theorem dialect_expand : dialectOf "tensor.expand_shape" = "tensor" := by cbv
private theorem dialect_generic : dialectOf "linalg.generic" = "linalg" := by cbv
private theorem dialect_reduce : dialectOf "linalg.reduce" = "linalg" := by cbv

theorem constant_execute (ctx : Ctx) (call : CallSem) (id bits : Nat) :
    W8A8SourceSemantics.opSem ctx call (constant id bits) [] [] =
      .ok [.scalar (W8A8OrderedBody.value (BitVec.ofNat 32 bits))] := by
  simp [
      W8A8SourceSemantics.opSem, W8A8SourceSemantics.isSource, W8A8SourceContract.name,
      Sem.Ops.opSem, unmodelledAttr?, interpretedAttrs, dispatch, constant, Op.blind, Op.name,
      Op.results, Op.props, Op.attrs, Op.successors, Op.attr?, dialect_constant, Arith.sem,
      scalarOfAttr, W8A8OrderedBody.value, FKind.width]

theorem splat_execute (ctx : Ctx) (call : CallSem) (shape : List Nat) (input result : Nat) (element : Val) :
    W8A8SourceSemantics.opSem ctx call (splat shape input result) [.scalar element] [] =
      .ok [.tensor (Tensor.full shape (.float .f32) element)] := by
  simp [W8A8SourceSemantics.opSem, W8A8SourceSemantics.isSource, W8A8SourceContract.name,
    Sem.Ops.opSem, unmodelledAttr?, interpretedAttrs, dispatch, splat, Op.blind,
    Op.name, Op.results, Op.props, Op.attrs, Op.successors, value, Tensor.sem,
    dialect_splat, Tensor.resultTensorTy, static_tensor]

theorem empty_execute (ctx : Ctx) (call : CallSem) (shape : List Nat) (result : Nat) :
    W8A8SourceSemantics.opSem ctx call (empty shape result) [] [] =
      .ok [.tensor (Tensor.full shape (.float .f32) .poison)] := by
  simp [W8A8SourceSemantics.opSem, W8A8SourceSemantics.isSource, W8A8SourceContract.name,
    Sem.Ops.opSem, unmodelledAttr?, interpretedAttrs, dispatch, empty, Op.blind,
    Op.name, Op.results, Op.props, Op.attrs, Op.successors, value, Tensor.sem,
    dialect_empty, Tensor.resultTensorTy, static_tensor]

theorem reassociation_decode (indices : List Nat) :
    (indices.map (fun (d : Nat) => Attr.int (Int.ofNat d) (.int 64))).mapM (fun
      | .int i _ => if i ≥ 0 then some i.toNat else none
      | _ => none) = some indices := by
  induction indices with
  | nil => rfl
  | cons head tail induction =>
    simp only [List.map_cons, List.mapM_cons, ge_iff_le]
    rw [induction]
    rfl

theorem collapse_execute (ctx : Ctx) (call : CallSem) (shape : List Nat) (input result : Nat)
    (tensor : Tensor) (sameShape : tensor.shape = shape) (sameType : tensor.elt = .float .f32)
    (valid : Tensor.reassocValid [List.range shape.length] shape [shapeSize shape] = true) :
    W8A8SourceSemantics.opSem ctx call (collapse shape input result) [.tensor tensor] [] =
      .ok [.tensor (ReshapeIndex.reshape tensor [shapeSize shape] (.float .f32))] := by
  have decoded : Tensor.reassociation? (collapse shape input result).blind = some [List.range shape.length] := by
    simp only [Tensor.reassociation?, Op.blind, collapse, Op.attr?, Op.props, Op.attrs,
      List.lookup, beq_self_eq_true, List.mapM_cons, List.mapM_nil]
    change ((List.range shape.length).map (fun (d : Nat) => Attr.int (Int.ofNat d) (.int 64))).mapM
      (fun | .int i _ => if i ≥ 0 then some i.toNat else none | _ => none) >>= (fun xs => some [xs]) = _
    exact congrArg (fun xs : Option (List Nat) => xs.bind (fun ys => some [ys]))
      (reassociation_decode (List.range shape.length))
  have ordinary : W8A8SourceSemantics.isSource (collapse shape input result).name = false := by cbv
  have modelled : unmodelledAttr? (collapse shape input result).blind = none := by
    simp [collapse, Op.blind, unmodelledAttr?, interpretedAttrs, Op.name, Op.props, Op.attrs]
  rw [W8A8SourceSemantics.unchanged ordinary, tensor_sem_of modelled dialect_collapse]
  simp only [Tensor.sem, Op.blind_name, show (collapse shape input result).name = "tensor.collapse_shape" from rfl,
    decoded, resultTensorTy_blind]
  simp [collapse, Tensor.resultTensorTy, Op.results, value, static_tensor, sameShape, sameType, valid, ReshapeIndex.reshape]

theorem expand_execute (ctx : Ctx) (call : CallSem) (shape : List Nat) (input result : Nat)
    (tensor : Tensor) (sameType : tensor.elt = .float .f32)
    (valid : Tensor.reassocValid [List.range shape.length] shape tensor.shape = true) :
    W8A8SourceSemantics.opSem ctx call (expand shape input result) [.tensor tensor] [] =
      .ok [.tensor (ReshapeIndex.reshape tensor shape (.float .f32))] := by
  have decoded : Tensor.reassociation? (expand shape input result).blind = some [List.range shape.length] := by
    simp only [Tensor.reassociation?, Op.blind, expand, Op.attr?, Op.props, Op.attrs,
      List.lookup, beq_self_eq_true, List.mapM_cons, List.mapM_nil]
    change ((List.range shape.length).map (fun (d : Nat) => Attr.int (Int.ofNat d) (.int 64))).mapM
      (fun | .int i _ => if i ≥ 0 then some i.toNat else none | _ => none) >>= (fun xs => some [xs]) = _
    exact congrArg (fun xs : Option (List Nat) => xs.bind (fun ys => some [ys]))
      (reassociation_decode (List.range shape.length))
  have ordinary : W8A8SourceSemantics.isSource (expand shape input result).name = false := by cbv
  have modelled : unmodelledAttr? (expand shape input result).blind = none := by
    simp [expand, Op.blind, unmodelledAttr?, interpretedAttrs, Op.name, Op.props, Op.attrs]
  rw [W8A8SourceSemantics.unchanged ordinary, tensor_sem_of modelled dialect_expand]
  simp only [Tensor.sem, Op.blind_name, show (expand shape input result).name = "tensor.expand_shape" from rfl,
    decoded, resultTensorTy_blind]
  simp [expand, Tensor.resultTensorTy, Op.results, value, static_tensor, sameType, valid, ReshapeIndex.reshape]

private theorem maps_decode (maps : List AffineMap) :
    (maps.map Attr.affineMap).mapM (fun | .affineMap m => some m | _ => none) = some maps := by
  induction maps with
  | nil => rfl
  | cons head tail induction => simp [induction]

private theorem iterators_decode (rank : Nat) :
    (List.replicate rank (Attr.enum "linalg" "iterator_type" "parallel")).mapM (fun
      | .enum "linalg" "iterator_type" "parallel" => some IterKind.parallel
      | .enum "linalg" "iterator_type" "reduction" => some IterKind.reduction
      | _ => none) = some (List.replicate rank IterKind.parallel) := by
  induction rank with
  | zero => rfl
  | succ rank induction => simp [List.replicate_succ, induction]

theorem generic_execute (ctx : Ctx) (call : CallSem) (env : Env)
    (shape : List Nat) (ids : List Nat) (destination result : Nat) (maps : List AffineMap) (body : Region)
    (inputs : List RVal) (initial output : Tensor) (arity : inputs.length = ids.length)
    (executed : genericSem ctx maps (List.replicate shape.length .parallel) inputs [.tensor initial]
      (fun c => evalRegion W8A8SourceSemantics.opSem c call env body) = .ok [.tensor output]) :
    W8A8SourceSemantics.opSem ctx call (generic shape ids destination result maps body)
      (inputs ++ [.tensor initial]) (evalRegions W8A8SourceSemantics.opSem call env [body]) =
      .ok [.tensor output] := by
  simp only [evalRegions]
  have split : splitInsOuts (generic shape ids destination result maps body).blind
      (inputs ++ [.tensor initial]) = some (inputs, [.tensor initial]) := by
    simp [
      generic, Op.blind, splitInsOuts, attrDenseArray?, Op.attr?, Op.props, Op.attrs, List.lookup,
      ← arity]
  have ordinary : W8A8SourceSemantics.isSource (generic shape ids destination result maps body).name = false := by cbv
  have modelled : unmodelledAttr? (generic shape ids destination result maps body).blind = none := by
    simp [generic, Op.blind, unmodelledAttr?, interpretedAttrs, Op.name, Op.props, Op.attrs]
  rw [W8A8SourceSemantics.unchanged ordinary,
    Check.linalg_sem_of modelled dialect_generic (Or.inl rfl)]
  have selectedMaps : attrAffineMaps? (generic shape ids destination result maps body) "indexing_maps" = some maps := by
    change (maps.map Attr.affineMap).mapM (fun | .affineMap m => some m | _ => none) = some maps
    exact maps_decode maps
  have selectedIters : iterKinds? (generic shape ids destination result maps body) =
      some (List.replicate shape.length .parallel) := by
    simp only [generic, iterKinds?, Op.attr?, Op.props, Op.attrs, List.lookup]
    change (List.replicate shape.length (Attr.enum "linalg" "iterator_type" "parallel")).mapM _ = _
    exact iterators_decode shape.length
  simp only [Linalg.sem, Op.blind_name,
    show (generic shape ids destination result maps body).name = "linalg.generic" from rfl,
    split, attrAffineMaps?_blind, selectedMaps, iterKinds?_blind, selectedIters]
  rw [executed]
  rfl

end MerlinVerify.Rules.W8A8AttentionTemplateOps
