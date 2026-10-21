"builtin.module"() ({
  "func.func"() <{sym_name = "subject", function_type = (tensor<1x32x32xi1>, tensor<1x1024xi64>, tensor<1x1024xi64>) -> (tensor<1x1024xi64>, index)}> ({
  ^bb0(%0: tensor<1x32x32xi1>, %1: tensor<1x1024xi64>, %2: tensor<1x1024xi64>):
    %3 = "tensor.collapse_shape"(%0) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bool", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.embeddings"} : (tensor<1x32x32xi1>) -> tensor<1024xi1>
    %4 = "tensor.expand_shape"(%3) <{reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 1, 1024>}> {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bool", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.embeddings"} : (tensor<1024xi1>) -> tensor<1x1024xi1>
    %5 = "tensor.collapse_shape"(%1) <{reassociation = [[0 : i64, 1 : i64]]}> {prov.region_id = "mask_gather_0", prov.family = "gather_scatter", prov._pattern_hint = "mask_gather", prov.op = "mask_gather", prov.aten = "aten.index.Tensor", prov.orig_dtype = "int64", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.embeddings"} : (tensor<1x1024xi64>) -> tensor<1024xi64>
    %6 = "tensor.collapse_shape"(%4) <{reassociation = [[0 : i64, 1 : i64]]}> {prov.region_id = "mask_gather_0", prov.family = "gather_scatter", prov._pattern_hint = "mask_gather", prov.op = "mask_gather", prov.aten = "aten.index.Tensor", prov.orig_dtype = "int64", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.embeddings"} : (tensor<1x1024xi1>) -> tensor<1024xi1>
    %7 = "tensor.empty"() : () -> tensor<1024xi64>
    %8 = "linalg.generic"(%6, %7) <{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%9: i1, %10: i64):
      %11 = "arith.extui"(%9) : (i1) -> i64
      "linalg.yield"(%11) : (i64) -> ()
    }) {prov.region_id = "mask_gather_0", prov.family = "gather_scatter", prov._pattern_hint = "mask_gather", prov.op = "mask_gather", prov.aten = "aten.index.Tensor", prov.orig_dtype = "int64", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.embeddings"} : (tensor<1024xi1>, tensor<1024xi64>) -> tensor<1024xi64>
    %12 = "arith.constant"() <{value = 0 : i64}> {prov.region_id = "mask_gather_0", prov.family = "gather_scatter", prov._pattern_hint = "mask_gather", prov.op = "mask_gather", prov.aten = "aten.index.Tensor", prov.orig_dtype = "int64", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.embeddings"} : () -> i64
    %13 = "tensor.splat"(%12) {prov.region_id = "mask_gather_0", prov.family = "gather_scatter", prov._pattern_hint = "mask_gather", prov.op = "mask_gather", prov.aten = "aten.index.Tensor", prov.orig_dtype = "int64", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.embeddings"} : (i64) -> tensor<i64>
    %14 = "linalg.reduce"(%8, %13) <{dimensions = array<i64: 0>}> ({
    ^bb0(%15: i64, %16: i64):
      %17 = "arith.addi"(%15, %16) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
      "linalg.yield"(%17) : (i64) -> ()
    }) : (tensor<1024xi64>, tensor<i64>) -> tensor<i64>
    %18 = "tensor.extract"(%14) : (tensor<i64>) -> i64
    %19 = "arith.index_cast"(%18) {prov.region_id = "mask_gather_0", prov.family = "gather_scatter", prov._pattern_hint = "mask_gather", prov.op = "mask_gather", prov.aten = "aten.index.Tensor", prov.orig_dtype = "int64", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.embeddings"} : (i64) -> index
    %20 = "arith.constant"() <{value = 0 : index}> {prov.region_id = "mask_gather_0", prov.family = "gather_scatter", prov._pattern_hint = "mask_gather", prov.op = "mask_gather", prov.aten = "aten.index.Tensor", prov.orig_dtype = "int64", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.embeddings"} : () -> index
    %21 = "arith.constant"() <{value = 1 : index}> {prov.region_id = "mask_gather_0", prov.family = "gather_scatter", prov._pattern_hint = "mask_gather", prov.op = "mask_gather", prov.aten = "aten.index.Tensor", prov.orig_dtype = "int64", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.embeddings"} : () -> index
    %22 = "arith.constant"() <{value = 1024 : index}> {prov.region_id = "mask_gather_0", prov.family = "gather_scatter", prov._pattern_hint = "mask_gather", prov.op = "mask_gather", prov.aten = "aten.index.Tensor", prov.orig_dtype = "int64", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.embeddings"} : () -> index
    %23 = "tensor.empty"(%19) : (index) -> tensor<?xi64>
    %24, %25 = "scf.for"(%20, %22, %21, %23, %20) ({
    ^bb0(%26: index, %27: tensor<?xi64>, %28: index):
      %29 = "tensor.extract"(%6, %26) : (tensor<1024xi1>, index) -> i1
      %30, %31 = "scf.if"(%29) ({
        %32 = "tensor.extract"(%5, %26) : (tensor<1024xi64>, index) -> i64
        %33 = "tensor.insert"(%32, %27, %28) : (i64, tensor<?xi64>, index) -> tensor<?xi64>
        %34 = "arith.addi"(%28, %21) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        "scf.yield"(%33, %34) : (tensor<?xi64>, index) -> ()
      }, {
        "scf.yield"(%27, %28) : (tensor<?xi64>, index) -> ()
      }) : (i1) -> (tensor<?xi64>, index)
      "scf.yield"(%30, %31) : (tensor<?xi64>, index) -> ()
    }) : (index, index, index, tensor<?xi64>, index) -> (tensor<?xi64>, index)
    %35 = "tensor.collapse_shape"(%0) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bool", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.embeddings"} : (tensor<1x32x32xi1>) -> tensor<1024xi1>
    %36 = "tensor.expand_shape"(%35) <{reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 1, 1024>}> {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bool", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.embeddings"} : (tensor<1024xi1>) -> tensor<1x1024xi1>
    %37 = "tensor.collapse_shape"(%2) <{reassociation = [[0 : i64, 1 : i64]]}> {prov.region_id = "mask_scatter_0", prov.family = "gather_scatter", prov._pattern_hint = "index_put", prov.op = "index_put", prov.aten = "aten.index_put.default", prov.orig_dtype = "int64", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.embeddings"} : (tensor<1x1024xi64>) -> tensor<1024xi64>
    %38 = "tensor.collapse_shape"(%36) <{reassociation = [[0 : i64, 1 : i64]]}> {prov.region_id = "mask_scatter_0", prov.family = "gather_scatter", prov._pattern_hint = "index_put", prov.op = "index_put", prov.aten = "aten.index_put.default", prov.orig_dtype = "int64", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.embeddings"} : (tensor<1x1024xi1>) -> tensor<1024xi1>
    %39 = "arith.constant"() <{value = 0 : index}> {prov.region_id = "mask_scatter_0", prov.family = "gather_scatter", prov._pattern_hint = "index_put", prov.op = "index_put", prov.aten = "aten.index_put.default", prov.orig_dtype = "int64", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.embeddings"} : () -> index
    %40 = "arith.constant"() <{value = 1 : index}> {prov.region_id = "mask_scatter_0", prov.family = "gather_scatter", prov._pattern_hint = "index_put", prov.op = "index_put", prov.aten = "aten.index_put.default", prov.orig_dtype = "int64", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.embeddings"} : () -> index
    %41 = "arith.constant"() <{value = 1024 : index}> {prov.region_id = "mask_scatter_0", prov.family = "gather_scatter", prov._pattern_hint = "index_put", prov.op = "index_put", prov.aten = "aten.index_put.default", prov.orig_dtype = "int64", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.embeddings"} : () -> index
    %42, %43 = "scf.for"(%39, %41, %40, %37, %39) ({
    ^bb0(%44: index, %45: tensor<1024xi64>, %46: index):
      %47 = "tensor.extract"(%38, %44) : (tensor<1024xi1>, index) -> i1
      %48, %49 = "scf.if"(%47) ({
        %50 = "tensor.extract"(%24, %46) : (tensor<?xi64>, index) -> i64
        %51 = "tensor.insert"(%50, %45, %44) : (i64, tensor<1024xi64>, index) -> tensor<1024xi64>
        %52 = "arith.addi"(%46, %40) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        "scf.yield"(%51, %52) : (tensor<1024xi64>, index) -> ()
      }, {
        "scf.yield"(%45, %46) : (tensor<1024xi64>, index) -> ()
      }) : (i1) -> (tensor<1024xi64>, index)
      "scf.yield"(%48, %49) : (tensor<1024xi64>, index) -> ()
    }) : (index, index, index, tensor<1024xi64>, index) -> (tensor<1024xi64>, index)
    %53 = "tensor.expand_shape"(%42) <{reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 1, 1024>}> {prov.region_id = "mask_scatter_0", prov.family = "gather_scatter", prov._pattern_hint = "index_put", prov.op = "index_put", prov.aten = "aten.index_put.default", prov.orig_dtype = "int64", prov.module = "model", prov.fqn = "model.vlm_with_expert.vlm.model.vision_model.embeddings"} : (tensor<1024xi64>) -> tensor<1x1024xi64>
    "func.return"(%53, %43) : (tensor<1x1024xi64>, index) -> ()
  }) : () -> ()
}) : () -> ()