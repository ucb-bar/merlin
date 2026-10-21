/- Concrete SmolVLA mask-selection cut, transcribed from the authoritative MLIR export.
The checker compares decoded programs to this AST; this file contributes data, not proof facts.
SSA identifiers are renamed to stable proof-local names. Input/output shapes remain captured.
-/
import MerlinVerify.Rules.FrontendMaskCountProgram
import MerlinVerify.Rules.FrontendMaskAlias

namespace MerlinVerify.Rules.FrontendMaskTemplate
open MerlinVerify

def maskCollapse0 : Op :=
  (.mk "tensor.collapse_shape" [{id := 60, ty := (.tensor [(some 1024)] (.int 1) none)}] [50] [("prov._pattern_hint", (.str "view")), ("prov.aten", (.str "aten.view.default")), ("prov.family", (.str "layout")), ("prov.fqn", (.str "model.vlm_with_expert.vlm.model.vision_model.embeddings")), ("prov.module", (.str "model")), ("prov.op", (.str "view")), ("prov.orig_dtype", (.str "bool")), ("prov.region_id", (.str "view_2")), ("reassociation", (.array [(.array [(.int 0 (.int 64)), (.int 1 (.int 64)), (.int 2 (.int 64))])]))] [] [] [])

def maskExpand0 : Op :=
  (.mk "tensor.expand_shape" [{id := 61, ty := (.tensor [(some 1), (some 1024)] (.int 1) none)}] [60] [("prov._pattern_hint", (.str "view")), ("prov.aten", (.str "aten.view.default")), ("prov.family", (.str "layout")), ("prov.fqn", (.str "model.vlm_with_expert.vlm.model.vision_model.embeddings")), ("prov.module", (.str "model")), ("prov.op", (.str "view")), ("prov.orig_dtype", (.str "bool")), ("prov.region_id", (.str "view_2")), ("reassociation", (.array [(.array [(.int 0 (.int 64)), (.int 1 (.int 64))])])), ("static_output_shape", (.denseArray (.int 64) [1, 1024]))] [] [] [])

def valuesCollapse : Op :=
  (.mk "tensor.collapse_shape" [{id := 40, ty := (.tensor [(some 1024)] (.int 64) none)}] [51] [("prov._pattern_hint", (.str "mask_gather")), ("prov.aten", (.str "aten.index.Tensor")), ("prov.family", (.str "gather_scatter")), ("prov.fqn", (.str "model.vlm_with_expert.vlm.model.vision_model.embeddings")), ("prov.module", (.str "model")), ("prov.op", (.str "mask_gather")), ("prov.orig_dtype", (.str "int64")), ("prov.region_id", (.str "mask_gather_0")), ("reassociation", (.array [(.array [(.int 0 (.int 64)), (.int 1 (.int 64))])]))] [] [] [])

def maskFlatten0 : Op :=
  (.mk "tensor.collapse_shape" [{id := 0, ty := (.tensor [(some 1024)] (.int 1) none)}] [61] [("prov._pattern_hint", (.str "mask_gather")), ("prov.aten", (.str "aten.index.Tensor")), ("prov.family", (.str "gather_scatter")), ("prov.fqn", (.str "model.vlm_with_expert.vlm.model.vision_model.embeddings")), ("prov.module", (.str "model")), ("prov.op", (.str "mask_gather")), ("prov.orig_dtype", (.str "int64")), ("prov.region_id", (.str "mask_gather_0")), ("reassociation", (.array [(.array [(.int 0 (.int 64)), (.int 1 (.int 64))])]))] [] [] [])

def countEmpty : Op :=
  (.mk "tensor.empty" [{id := 1, ty := (.tensor [(some 1024)] (.int 64) none)}] [] [] [] [] [])

def countMap : Op :=
  (.mk "linalg.generic" [{id := 2, ty := (.tensor [(some 1024)] (.int 64) none)}] [0, 1] [("indexing_maps", (.array [(.affineMap {numDims := 1, numSyms := 0, results := [(.dim 0)]}), (.affineMap {numDims := 1, numSyms := 0, results := [(.dim 0)]})])), ("iterator_types", (.array [(.enum "linalg" "iterator_type" "parallel")])), ("operandSegmentSizes", (.denseArray (.int 32) [1, 1])), ("prov._pattern_hint", (.str "mask_gather")), ("prov.aten", (.str "aten.index.Tensor")), ("prov.family", (.str "gather_scatter")), ("prov.fqn", (.str "model.vlm_with_expert.vlm.model.vision_model.embeddings")), ("prov.module", (.str "model")), ("prov.op", (.str "mask_gather")), ("prov.orig_dtype", (.str "int64")), ("prov.region_id", (.str "mask_gather_0"))] [] [(.mk [(.mk [{id := 300, ty := (.int 1)}, {id := 301, ty := (.int 64)}] [(.mk "arith.extui" [{id := 302, ty := (.int 64)}] [300] [] [] [] []), (.mk "linalg.yield" [] [302] [] [] [] [])])])] [])

def countZero : Op :=
  (.mk "arith.constant" [{id := 3, ty := (.int 64)}] [] [("prov._pattern_hint", (.str "mask_gather")), ("prov.aten", (.str "aten.index.Tensor")), ("prov.family", (.str "gather_scatter")), ("prov.fqn", (.str "model.vlm_with_expert.vlm.model.vision_model.embeddings")), ("prov.module", (.str "model")), ("prov.op", (.str "mask_gather")), ("prov.orig_dtype", (.str "int64")), ("prov.region_id", (.str "mask_gather_0")), ("value", (.int 0 (.int 64)))] [] [] [])

def countSplat : Op :=
  (.mk "tensor.splat" [{id := 4, ty := (.tensor [] (.int 64) none)}] [3] [("prov._pattern_hint", (.str "mask_gather")), ("prov.aten", (.str "aten.index.Tensor")), ("prov.family", (.str "gather_scatter")), ("prov.fqn", (.str "model.vlm_with_expert.vlm.model.vision_model.embeddings")), ("prov.module", (.str "model")), ("prov.op", (.str "mask_gather")), ("prov.orig_dtype", (.str "int64")), ("prov.region_id", (.str "mask_gather_0"))] [] [] [])

def countReduce : Op :=
  (.mk "linalg.reduce" [{id := 5, ty := (.tensor [] (.int 64) none)}] [2, 4] [("dimensions", (.denseArray (.int 64) [0]))] [] [(.mk [(.mk [{id := 303, ty := (.int 64)}, {id := 304, ty := (.int 64)}] [(.mk "arith.addi" [{id := 305, ty := (.int 64)}] [303, 304] [("overflowFlags", (.enum "arith" "overflow" "none"))] [] [] []), (.mk "linalg.yield" [] [305] [] [] [] [])])])] [])

def countExtract : Op :=
  (.mk "tensor.extract" [{id := 6, ty := (.int 64)}] [5] [] [] [] [])

def countCast : Op :=
  (.mk "arith.index_cast" [{id := 7, ty := .index}] [6] [("prov._pattern_hint", (.str "mask_gather")), ("prov.aten", (.str "aten.index.Tensor")), ("prov.family", (.str "gather_scatter")), ("prov.fqn", (.str "model.vlm_with_expert.vlm.model.vision_model.embeddings")), ("prov.module", (.str "model")), ("prov.op", (.str "mask_gather")), ("prov.orig_dtype", (.str "int64")), ("prov.region_id", (.str "mask_gather_0"))] [] [] [])

def gatherLower : Op :=
  (.mk "arith.constant" [{id := 9, ty := .index}] [] [("prov._pattern_hint", (.str "mask_gather")), ("prov.aten", (.str "aten.index.Tensor")), ("prov.family", (.str "gather_scatter")), ("prov.fqn", (.str "model.vlm_with_expert.vlm.model.vision_model.embeddings")), ("prov.module", (.str "model")), ("prov.op", (.str "mask_gather")), ("prov.orig_dtype", (.str "int64")), ("prov.region_id", (.str "mask_gather_0")), ("value", (.int 0 .index))] [] [] [])

def gatherStep : Op :=
  (.mk "arith.constant" [{id := 10, ty := .index}] [] [("prov._pattern_hint", (.str "mask_gather")), ("prov.aten", (.str "aten.index.Tensor")), ("prov.family", (.str "gather_scatter")), ("prov.fqn", (.str "model.vlm_with_expert.vlm.model.vision_model.embeddings")), ("prov.module", (.str "model")), ("prov.op", (.str "mask_gather")), ("prov.orig_dtype", (.str "int64")), ("prov.region_id", (.str "mask_gather_0")), ("value", (.int 1 .index))] [] [] [])

def gatherUpper : Op :=
  (.mk "arith.constant" [{id := 11, ty := .index}] [] [("prov._pattern_hint", (.str "mask_gather")), ("prov.aten", (.str "aten.index.Tensor")), ("prov.family", (.str "gather_scatter")), ("prov.fqn", (.str "model.vlm_with_expert.vlm.model.vision_model.embeddings")), ("prov.module", (.str "model")), ("prov.op", (.str "mask_gather")), ("prov.orig_dtype", (.str "int64")), ("prov.region_id", (.str "mask_gather_0")), ("value", (.int 1024 .index))] [] [] [])

def compactEmpty : Op :=
  (.mk "tensor.empty" [{id := 12, ty := (.tensor [none] (.int 64) none)}] [7] [] [] [] [])

def gatherLoop : Op :=
  (.mk "scf.for" [{id := 13, ty := (.tensor [none] (.int 64) none)}, {id := 14, ty := .index}] [9, 11, 10, 12, 9] [] [] [(.mk [(.mk [{id := 100, ty := .index}, {id := 101, ty := (.tensor [none] (.int 64) none)}, {id := 102, ty := .index}] [(.mk "tensor.extract" [{id := 103, ty := (.int 1)}] [0, 100] [] [] [] []), (.mk "scf.if" [{id := 104, ty := (.tensor [none] (.int 64) none)}, {id := 105, ty := .index}] [103] [] [] [(.mk [(.mk [] [(.mk "tensor.extract" [{id := 106, ty := (.int 64)}] [40, 100] [] [] [] []), (.mk "tensor.insert" [{id := 107, ty := (.tensor [none] (.int 64) none)}] [106, 101, 102] [] [] [] []), (.mk "arith.addi" [{id := 108, ty := .index}] [102, 10] [("overflowFlags", (.enum "arith" "overflow" "none"))] [] [] []), (.mk "scf.yield" [] [107, 108] [] [] [] [])])]), (.mk [(.mk [] [(.mk "scf.yield" [] [101, 102] [] [] [] [])])])] []), (.mk "scf.yield" [] [104, 105] [] [] [] [])])])] [])

def maskCollapse1 : Op :=
  (.mk "tensor.collapse_shape" [{id := 62, ty := (.tensor [(some 1024)] (.int 1) none)}] [50] [("prov._pattern_hint", (.str "view")), ("prov.aten", (.str "aten.view.default")), ("prov.family", (.str "layout")), ("prov.fqn", (.str "model.vlm_with_expert.vlm.model.vision_model.embeddings")), ("prov.module", (.str "model")), ("prov.op", (.str "view")), ("prov.orig_dtype", (.str "bool")), ("prov.region_id", (.str "view_3")), ("reassociation", (.array [(.array [(.int 0 (.int 64)), (.int 1 (.int 64)), (.int 2 (.int 64))])]))] [] [] [])

def maskExpand1 : Op :=
  (.mk "tensor.expand_shape" [{id := 63, ty := (.tensor [(some 1), (some 1024)] (.int 1) none)}] [62] [("prov._pattern_hint", (.str "view")), ("prov.aten", (.str "aten.view.default")), ("prov.family", (.str "layout")), ("prov.fqn", (.str "model.vlm_with_expert.vlm.model.vision_model.embeddings")), ("prov.module", (.str "model")), ("prov.op", (.str "view")), ("prov.orig_dtype", (.str "bool")), ("prov.region_id", (.str "view_3")), ("reassociation", (.array [(.array [(.int 0 (.int 64)), (.int 1 (.int 64))])])), ("static_output_shape", (.denseArray (.int 64) [1, 1024]))] [] [] [])

def seedCollapse : Op :=
  (.mk "tensor.collapse_shape" [{id := 41, ty := (.tensor [(some 1024)] (.int 64) none)}] [52] [("prov._pattern_hint", (.str "index_put")), ("prov.aten", (.str "aten.index_put.default")), ("prov.family", (.str "gather_scatter")), ("prov.fqn", (.str "model.vlm_with_expert.vlm.model.vision_model.embeddings")), ("prov.module", (.str "model")), ("prov.op", (.str "index_put")), ("prov.orig_dtype", (.str "int64")), ("prov.region_id", (.str "mask_scatter_0")), ("reassociation", (.array [(.array [(.int 0 (.int 64)), (.int 1 (.int 64))])]))] [] [] [])

def maskFlatten1 : Op :=
  (.mk "tensor.collapse_shape" [{id := 42, ty := (.tensor [(some 1024)] (.int 1) none)}] [63] [("prov._pattern_hint", (.str "index_put")), ("prov.aten", (.str "aten.index_put.default")), ("prov.family", (.str "gather_scatter")), ("prov.fqn", (.str "model.vlm_with_expert.vlm.model.vision_model.embeddings")), ("prov.module", (.str "model")), ("prov.op", (.str "index_put")), ("prov.orig_dtype", (.str "int64")), ("prov.region_id", (.str "mask_scatter_0")), ("reassociation", (.array [(.array [(.int 0 (.int 64)), (.int 1 (.int 64))])]))] [] [] [])

def scatterLower : Op :=
  (.mk "arith.constant" [{id := 15, ty := .index}] [] [("prov._pattern_hint", (.str "index_put")), ("prov.aten", (.str "aten.index_put.default")), ("prov.family", (.str "gather_scatter")), ("prov.fqn", (.str "model.vlm_with_expert.vlm.model.vision_model.embeddings")), ("prov.module", (.str "model")), ("prov.op", (.str "index_put")), ("prov.orig_dtype", (.str "int64")), ("prov.region_id", (.str "mask_scatter_0")), ("value", (.int 0 .index))] [] [] [])

def scatterStep : Op :=
  (.mk "arith.constant" [{id := 16, ty := .index}] [] [("prov._pattern_hint", (.str "index_put")), ("prov.aten", (.str "aten.index_put.default")), ("prov.family", (.str "gather_scatter")), ("prov.fqn", (.str "model.vlm_with_expert.vlm.model.vision_model.embeddings")), ("prov.module", (.str "model")), ("prov.op", (.str "index_put")), ("prov.orig_dtype", (.str "int64")), ("prov.region_id", (.str "mask_scatter_0")), ("value", (.int 1 .index))] [] [] [])

def scatterUpper : Op :=
  (.mk "arith.constant" [{id := 17, ty := .index}] [] [("prov._pattern_hint", (.str "index_put")), ("prov.aten", (.str "aten.index_put.default")), ("prov.family", (.str "gather_scatter")), ("prov.fqn", (.str "model.vlm_with_expert.vlm.model.vision_model.embeddings")), ("prov.module", (.str "model")), ("prov.op", (.str "index_put")), ("prov.orig_dtype", (.str "int64")), ("prov.region_id", (.str "mask_scatter_0")), ("value", (.int 1024 .index))] [] [] [])

def scatterLoop : Op :=
  (.mk "scf.for" [{id := 18, ty := (.tensor [(some 1024)] (.int 64) none)}, {id := 19, ty := .index}] [15, 17, 16, 41, 15] [] [] [(.mk [(.mk [{id := 200, ty := .index}, {id := 201, ty := (.tensor [(some 1024)] (.int 64) none)}, {id := 202, ty := .index}] [(.mk "tensor.extract" [{id := 203, ty := (.int 1)}] [42, 200] [] [] [] []), (.mk "scf.if" [{id := 204, ty := (.tensor [(some 1024)] (.int 64) none)}, {id := 205, ty := .index}] [203] [] [] [(.mk [(.mk [] [(.mk "tensor.extract" [{id := 206, ty := (.int 64)}] [13, 202] [] [] [] []), (.mk "tensor.insert" [{id := 207, ty := (.tensor [(some 1024)] (.int 64) none)}] [206, 201, 200] [] [] [] []), (.mk "arith.addi" [{id := 208, ty := .index}] [202, 16] [("overflowFlags", (.enum "arith" "overflow" "none"))] [] [] []), (.mk "scf.yield" [] [207, 208] [] [] [] [])])]), (.mk [(.mk [] [(.mk "scf.yield" [] [201, 202] [] [] [] [])])])] []), (.mk "scf.yield" [] [204, 205] [] [] [] [])])])] [])

def resultExpand : Op :=
  (.mk "tensor.expand_shape" [{id := 21, ty := (.tensor [(some 1), (some 1024)] (.int 64) none)}] [18] [("prov._pattern_hint", (.str "index_put")), ("prov.aten", (.str "aten.index_put.default")), ("prov.family", (.str "gather_scatter")), ("prov.fqn", (.str "model.vlm_with_expert.vlm.model.vision_model.embeddings")), ("prov.module", (.str "model")), ("prov.op", (.str "index_put")), ("prov.orig_dtype", (.str "int64")), ("prov.region_id", (.str "mask_scatter_0")), ("reassociation", (.array [(.array [(.int 0 (.int 64)), (.int 1 (.int 64))])])), ("static_output_shape", (.denseArray (.int 64) [1, 1024]))] [] [] [])

def returned : Op :=
  (.mk "func.return" [] [21, 19] [] [] [] [])

def metadata : FrontendMaskCountProgram.Metadata :=
  {zero := countZero.attrs, splat := countSplat.attrs, cast := countCast.attrs}

def afterCount : List Op :=
  [gatherLower, gatherStep, gatherUpper, compactEmpty, gatherLoop, maskCollapse1, maskExpand1,
   seedCollapse, maskFlatten1, scatterLower, scatterStep, scatterUpper, scatterLoop, resultExpand, returned]

def operations : List Op :=
  [maskCollapse0, maskExpand0, valuesCollapse, maskFlatten0] ++
  FrontendMaskCountProgram.countOps 1024 countMap.attrs countMap.props metadata ++ afterCount

def body : Region := .mk [.mk [
  {id := 50, ty := .tensor [some 1, some 32, some 32] (.int 1) none},
  {id := 51, ty := .tensor [some 1, some 1024] (.int 64) none},
  {id := 52, ty := .tensor [some 1, some 1024] (.int 64) none}] operations]

def subject : Module := {ops := [.mk "func.func" [] [] [("function_type", (.type (.function [(.tensor [(some 1), (some 32), (some 32)] (.int 1) none), (.tensor [(some 1), (some 1024)] (.int 64) none), (.tensor [(some 1), (some 1024)] (.int 64) none)] [(.tensor [(some 1), (some 1024)] (.int 64) none), .index]))), ("sym_name", (.str "subject"))] [] [.mk [.mk [{id := 50, ty := (.tensor [(some 1), (some 32), (some 32)] (.int 1) none)}, {id := 51, ty := (.tensor [(some 1), (some 1024)] (.int 64) none)}, {id := 52, ty := (.tensor [(some 1), (some 1024)] (.int 64) none)}] operations]] []]}

end MerlinVerify.Rules.FrontendMaskTemplate
