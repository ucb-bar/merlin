/- Provisional source-driven structural checker for lower_torchao_affine_quant.
Its whole-program theorem compares the target to the explicitly materialized local contract.
It does NOT yet relate captured opaque-call execution to TorchAO's reference. In particular,
reduction lifting, zero-point elision and actual capture/version admission remain obligations.
No production verifier may treat this result as approved source-to-target verification. -/
import MerlinVerify.Rules.TorchAOAffineScalar
import MerlinVerify.Rules.FrontendBoolProgram
import MerlinVerify.Rules.LocalProgramCorrespondence

namespace MerlinVerify.Rules.TorchAOAffineProgram
open MerlinVerify MerlinVerify.Check
open FrontendBoolProgram (defsOps sourceTypes freshStart)
open DequantScalar (op floatOp yield)
open TorchAOAffineScalar (constant)

abbrev Types := List (ValueId × Ty)
def chooseName := "torchao_choose_qparams_affine_default"
def quantizeName := "torchao_quantize_affine_default"

def named (o : Op) (name : String) : Bool := o.name == "func.call" &&
  o.attr? "callee" == some (.symbolRef name [])

def provenance (o : Op) (tag : String) : List (String × Attr) :=
  let kept := o.attrs.filter fun (k, _) => k.startsWith "prov."
  let previous := match kept.lookup "prov.transforms" with
    | some (Attr.str s) => s.splitOn "," | _ => []
  (kept.filter (fun (k, _) => k != "prov.transforms") ++
    [("prov.transforms", Attr.str (String.intercalate "," (previous.filter (· != "") ++ [tag])))]).mergeSort
      (fun a b => a.1 ≤ b.1)

def tensorTy (shape : List Nat) (elt : Ty) : Ty := .tensor (shape.map some) elt none

def empty (id : Nat) (ty : Ty) : Op := .mk "tensor.empty" [{id, ty}] [] [] [] [] []

def identity (rank : Nat) : AffineMap :=
  {numDims := rank, numSyms := 0, results := (List.range rank).map AffineExpr.dim}

def projection (rank kept : Nat) : AffineMap :=
  {numDims := rank, numSyms := 0, results := (List.range kept).map AffineExpr.dim}

def generic (id : Nat) (ty : Ty) (ins : List Nat) (out : Nat)
    (maps : List AffineMap) (iters : List String) (args : List Value) (body : List Op)
    (attrs : List (String × Attr)) : Op :=
  .mk "linalg.generic" [{id, ty}] (ins ++ [out])
    ([("indexing_maps", .array (maps.map Attr.affineMap)),
      ("iterator_types", .array (iters.map fun s => .enum "linalg" "iterator_type" s)),
      ("operandSegmentSizes", .denseArray (.int 32) [Int.ofNat ins.length, 1])] ++ attrs)
    [] [.mk [.mk args body]] []

def chooseExpansion (source : Nat) (result : Value) (shape : List Nat) (s : Nat) (o : Op) : List Op :=
  let scaleTy := result.ty
  let kept := shape.length - 1
  let fillArgs : List Value := [{id := s+3, ty := .float .f32}, {id := s+4, ty := .float .f32}]
  let fill := Op.mk "linalg.fill" [{id := s+2, ty := scaleTy}] [s+1, s]
    [("operandSegmentSizes", .denseArray (.int 32) [1, 1])]
    [] [.mk [.mk fillArgs [yield (s+3)]]] []
  let reduction := generic (s+9) scaleTy [source] (s+2)
    [identity shape.length, projection shape.length kept]
    (List.replicate kept "parallel" ++ ["reduction"])
    [{id := s+5, ty := .float .f32}, {id := s+6, ty := .float .f32}]
    [floatOp "math.absf" (s+7) [s+5], floatOp "arith.maximumf" (s+8) [s+6, s+7], yield (s+8)]
    (provenance o "torchao_choose_qparams_affine")
  let scale := generic result.id scaleTy [s+9] (s+10) [identity kept, identity kept]
    (List.replicate kept "parallel")
    [{id := s+11, ty := .float .f32}, {id := s+12, ty := .float .f32}]
    [constant (s+13) 0x42fe0000, constant (s+14) 0x3727c5ac,
     floatOp "arith.divf" (s+15) [s+11, s+13],
     floatOp "arith.maximumf" (s+16) [s+15, s+14], yield (s+16)]
    (provenance o "torchao_choose_qparams_affine")
  [empty s scaleTy, constant (s+1) 0, fill, reduction, empty (s+10) scaleTy, scale]

def shifted (offset : Nat) : Op → Op
  | .mk n rs args attrs props regions successors =>
    .mk n (rs.map fun v => {v with id := v.id + offset}) (args.map (· + offset)) attrs props regions successors

def quantizeExpansion (source scale : Nat) (result : Value) (shape : List Nat) (s : Nat) (o : Op) : List Op :=
  [empty s result.ty,
   generic result.id result.ty [source, scale] s
    [identity shape.length, projection shape.length (shape.length-1), identity shape.length]
    (List.replicate shape.length "parallel")
    [{id := s+1, ty := .float .f32}, {id := s+2, ty := .float .f32}, {id := s+3, ty := .int 8}]
    (TorchAOAffineScalar.body.map (shifted (s+1)))
    (provenance o "torchao_quantize_affine")]

/-- Static, unencoded f32 per-token layout. Other ranks/layouts/numeric symbol suffixes are
explicitly unsupported by this initial checker, even when production handles them. -/
def shape? (types : Types) (id : Nat) : Option (List Nat) := do
  let some ty := types.lookup id | none
  let some (shape, elt) := ty.staticTensor? | none
  if ty != tensorTy shape (.float .f32) || elt != .float .f32 || shape.isEmpty || shape.any (· == 0) then none
  else some shape

def metadataOK (o : Op) : Bool :=
  o.regions.isEmpty && o.successors.isEmpty && o.props.isEmpty &&
    o.attrs.all (fun (k, _) => k == "callee" || k.startsWith "prov.")

def expansion (types : Types) (o : Op) : StateT Nat Option (List Op) := do
  if !(metadataOK o) then failure
  let [result] := o.results | failure
  if named o chooseName then
    let [source] := o.operands | failure
    let some shape := shape? types source | failure
    if result.ty != tensorTy (shape.take (shape.length-1)) (.float .f32) then failure
    let start ← get
    modify (· + 17)
    pure (chooseExpansion source result shape start o)
  else if named o quantizeName then
    let [source, scale] := o.operands | failure
    let some shape := shape? types source | failure
    if types.lookup scale != some (tensorTy (shape.take (shape.length-1)) (.float .f32)) ||
      result.ty != tensorTy shape (.int 8) then failure
    let start ← get
    modify (· + 13)
    pure (quantizeExpansion source scale result shape start o)
  else failure

def removableDecl (o : Op) : Bool :=
  o.name == "func.func" && (o.attr? "sym_name" == some (.str chooseName) ||
    o.attr? "sym_name" == some (.str quantizeName)) &&
    o.regions.all (fun r => r.blocks.isEmpty)

def rewrite (types : Types) : Nat → List Op → StateT Nat Option (List Op)
  | 0, _ => failure
  | fuel+1, os => do
    let groups ← os.mapM fun o => do
      if removableDecl o then return []
      if named o chooseName || named o quantizeName then return (← expansion types o)
      -- Refuse an unsupported numeric overload instead of silently ignoring an eligible call.
      if o.name == "func.call" then
        match o.attr? "callee" with
        | some (.symbolRef n _) =>
          if n.startsWith chooseName || n.startsWith quantizeName then failure
        | _ => pure ()
      let rs ← o.regions.mapM fun r => do
        let bs ← r.blocks.mapM fun b => do
          let children ← rewrite types fuel b.ops
          pure (.mk b.args children : Block)
        pure (.mk bs : Region)
      match o with
      | .mk n res args attrs props _ successors => return [.mk n res args attrs props rs successors]
    pure groups.flatten

def materialize? (src : Module) : Option Module := do
  if src.attrs.lookup "prov.quantization" != some (.str "int8_dyn_act_int8_weight") then none else
  let defs ← defsOps 64 src.ops
  if !decide (defs.map (·.id)).Nodup then none else
  let start := freshStart defs
  let (ops, finish) ← (rewrite (sourceTypes defs) 64 src.ops).run start
  if finish == start then none else pure {src with ops}

/-- Strict whole-AST accounting plus the existing complete-program refinement checker.
This is materialized-contract correspondence, not a source opaque-call execution proof. -/
def check (src tgt : Module) : Bool :=
  match materialize? src with
  | none => false
  | some intended => LocalProgramCorrespondence.check intended tgt

theorem check_sound {src tgt intended : Module} (hi : materialize? src = some intended)
    (h : check src tgt = true) :
    alphaOps [] intended.ops tgt.ops = true ∧
      ∀ (ctx : Ctx) (name : String) (args : List RVal),
        denote Sem.Ops.opSem ctx intended name args = denote Sem.Ops.opSem ctx tgt name args := by
  simp only [check, hi] at h
  have hc := h
  simp only [LocalProgramCorrespondence.check, Bool.and_eq_true] at hc
  exact ⟨hc.1.1.1, LocalProgramCorrespondence.check_sound h⟩

#print axioms check_sound
end MerlinVerify.Rules.TorchAOAffineProgram
