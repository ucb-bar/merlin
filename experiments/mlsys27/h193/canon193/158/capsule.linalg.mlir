"builtin.module"() ({
  "func.func"() <{sym_name = "forward", function_type = (tensor<2x3xf32>, tensor<2x2xi64>) -> tensor<2x3xf32>}> ({
  ^bb0(%0: tensor<2x3xf32>, %1: tensor<2x2xi64>):
    %2 = "tensor.empty"() {prov._pattern_hint = "scatter", prov.op = "scatter", prov.source_node_ids = ["g:prepared:root:n2"], prov.origin_node_ids = ["g:original:root:n2", "g:quantized:root:n2"], prov.trace_role = "lowering", prov.aten = "aten.scatter.value", prov.orig_dtype = "float32"} : () -> tensor<2x3xf32>
    %3 = "linalg.generic"(%2) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 0, 1>}> ({
    ^bb1(%4: f32):
      %5 = "linalg.index"() <{dim = 0 : i64}> {prov.source_node_ids = ["g:prepared:root:n2"], prov.origin_node_ids = ["g:original:root:n2", "g:quantized:root:n2"], prov.trace_role = "lowering"} : () -> index
      %6 = "linalg.index"() <{dim = 1 : i64}> {prov.source_node_ids = ["g:prepared:root:n2"], prov.origin_node_ids = ["g:original:root:n2", "g:quantized:root:n2"], prov.trace_role = "lowering"} : () -> index
      %7 = "arith.constant"() <{value = 0 : index}> {prov.source_node_ids = ["g:prepared:root:n2"], prov.origin_node_ids = ["g:original:root:n2", "g:quantized:root:n2"], prov.trace_role = "lowering"} : () -> index
      %8 = "arith.constant"() <{value = 1 : index}> {prov.source_node_ids = ["g:prepared:root:n2"], prov.origin_node_ids = ["g:original:root:n2", "g:quantized:root:n2"], prov.trace_role = "lowering"} : () -> index
      %9 = "arith.constant"() <{value = 4 : index}> {prov.source_node_ids = ["g:prepared:root:n2"], prov.origin_node_ids = ["g:original:root:n2", "g:quantized:root:n2"], prov.trace_role = "lowering"} : () -> index
      %10 = "tensor.extract"(%0, %5, %6) {prov.source_node_ids = ["g:prepared:root:n2"], prov.origin_node_ids = ["g:original:root:n2", "g:quantized:root:n2"], prov.trace_role = "lowering"} : (tensor<2x3xf32>, index, index) -> f32
      %11, %12 = "scf.for"(%7, %9, %8, %10, %7) ({
      ^bb2(%13: index, %14: f32, %15: index):
        %16 = "arith.constant"() <{value = 2 : index}> {prov.source_node_ids = ["g:prepared:root:n2"], prov.origin_node_ids = ["g:original:root:n2", "g:quantized:root:n2"], prov.trace_role = "lowering"} : () -> index
        %17 = "arith.remui"(%13, %16) {prov.source_node_ids = ["g:prepared:root:n2"], prov.origin_node_ids = ["g:original:root:n2", "g:quantized:root:n2"], prov.trace_role = "lowering"} : (index, index) -> index
        %18 = "arith.divui"(%13, %16) {prov.source_node_ids = ["g:prepared:root:n2"], prov.origin_node_ids = ["g:original:root:n2", "g:quantized:root:n2"], prov.trace_role = "lowering"} : (index, index) -> index
        %19 = "arith.constant"() <{value = 2 : index}> {prov.source_node_ids = ["g:prepared:root:n2"], prov.origin_node_ids = ["g:original:root:n2", "g:quantized:root:n2"], prov.trace_role = "lowering"} : () -> index
        %20 = "arith.remui"(%18, %19) {prov.source_node_ids = ["g:prepared:root:n2"], prov.origin_node_ids = ["g:original:root:n2", "g:quantized:root:n2"], prov.trace_role = "lowering"} : (index, index) -> index
        %21 = "tensor.extract"(%1, %20, %17) {prov.source_node_ids = ["g:prepared:root:n2"], prov.origin_node_ids = ["g:original:root:n2", "g:quantized:root:n2"], prov.trace_role = "lowering"} : (tensor<2x2xi64>, index, index) -> i64
        %22 = "arith.index_cast"(%21) {prov.source_node_ids = ["g:prepared:root:n2"], prov.origin_node_ids = ["g:original:root:n2", "g:quantized:root:n2"], prov.trace_role = "lowering"} : (i64) -> index
        %23 = "arith.constant"() <{value = 3 : index}> {prov.source_node_ids = ["g:prepared:root:n2"], prov.origin_node_ids = ["g:original:root:n2", "g:quantized:root:n2"], prov.trace_role = "lowering"} : () -> index
        %24 = "arith.cmpi"(%22, %7) <{predicate = 5 : i64}> {prov.source_node_ids = ["g:prepared:root:n2"], prov.origin_node_ids = ["g:original:root:n2", "g:quantized:root:n2"], prov.trace_role = "lowering"} : (index, index) -> i1
        %25 = "arith.cmpi"(%22, %23) <{predicate = 2 : i64}> {prov.source_node_ids = ["g:prepared:root:n2"], prov.origin_node_ids = ["g:original:root:n2", "g:quantized:root:n2"], prov.trace_role = "lowering"} : (index, index) -> i1
        %26 = "arith.andi"(%24, %25) {prov.source_node_ids = ["g:prepared:root:n2"], prov.origin_node_ids = ["g:original:root:n2", "g:quantized:root:n2"], prov.trace_role = "lowering"} : (i1, i1) -> i1
        "cf.assert"(%26) <{msg = "scatter index out of bounds"}> {prov.source_node_ids = ["g:prepared:root:n2"], prov.origin_node_ids = ["g:original:root:n2", "g:quantized:root:n2"], prov.trace_role = "lowering"} : (i1) -> ()
        %27 = "arith.cmpi"(%20, %5) <{predicate = 0 : i64}> {prov.source_node_ids = ["g:prepared:root:n2"], prov.origin_node_ids = ["g:original:root:n2", "g:quantized:root:n2"], prov.trace_role = "lowering"} : (index, index) -> i1
        %28 = "arith.cmpi"(%22, %6) <{predicate = 0 : i64}> {prov.source_node_ids = ["g:prepared:root:n2"], prov.origin_node_ids = ["g:original:root:n2", "g:quantized:root:n2"], prov.trace_role = "lowering"} : (index, index) -> i1
        %29 = "arith.andi"(%27, %28) {prov.source_node_ids = ["g:prepared:root:n2"], prov.origin_node_ids = ["g:original:root:n2", "g:quantized:root:n2"], prov.trace_role = "lowering"} : (i1, i1) -> i1
        %30, %31 = "scf.if"(%29) ({
          %32 = "arith.constant"() <{value = 2.000000e+00 : f32}> {prov.source_node_ids = ["g:prepared:root:n2"], prov.origin_node_ids = ["g:original:root:n2", "g:quantized:root:n2"], prov.trace_role = "lowering"} : () -> f32
          %33 = "arith.addi"(%15, %8) <{overflowFlags = #arith.overflow<none>}> {prov.source_node_ids = ["g:prepared:root:n2"], prov.origin_node_ids = ["g:original:root:n2", "g:quantized:root:n2"], prov.trace_role = "lowering"} : (index, index) -> index
          "scf.yield"(%32, %33) {prov.source_node_ids = ["g:prepared:root:n2"], prov.origin_node_ids = ["g:original:root:n2", "g:quantized:root:n2"], prov.trace_role = "lowering"} : (f32, index) -> ()
        }, {
          "scf.yield"(%14, %15) {prov.source_node_ids = ["g:prepared:root:n2"], prov.origin_node_ids = ["g:original:root:n2", "g:quantized:root:n2"], prov.trace_role = "lowering"} : (f32, index) -> ()
        }) {prov.source_node_ids = ["g:prepared:root:n2"], prov.origin_node_ids = ["g:original:root:n2", "g:quantized:root:n2"], prov.trace_role = "lowering"} : (i1) -> (f32, index)
        "scf.yield"(%30, %31) {prov.source_node_ids = ["g:prepared:root:n2"], prov.origin_node_ids = ["g:original:root:n2", "g:quantized:root:n2"], prov.trace_role = "lowering"} : (f32, index) -> ()
      }) {prov.source_node_ids = ["g:prepared:root:n2"], prov.origin_node_ids = ["g:original:root:n2", "g:quantized:root:n2"], prov.trace_role = "lowering"} : (index, index, index, f32, index) -> (f32, index)
      "linalg.yield"(%11) {prov.source_node_ids = ["g:prepared:root:n2"], prov.origin_node_ids = ["g:original:root:n2", "g:quantized:root:n2"], prov.trace_role = "lowering"} : (f32) -> ()
    }) {prov._pattern_hint = "scatter", prov.op = "scatter", prov.source_node_ids = ["g:prepared:root:n2"], prov.origin_node_ids = ["g:original:root:n2", "g:quantized:root:n2"], prov.trace_role = "lowering", prov.aten = "aten.scatter.value", prov.orig_dtype = "float32"} : (tensor<2x3xf32>) -> tensor<2x3xf32>
    "func.return"(%3) : (tensor<2x3xf32>) -> ()
  }) : () -> ()
}) {m2m.result_layout = "{\22authority\22: \22same_conversion_exported_program\22, \22inputs\22: [{\22dtype\22: \22float32\22, \22requires_grad\22: false, \22shape\22: [2, 3], \22storage_offset\22: 0, \22stride\22: [3, 1]}, {\22dtype\22: \22int64\22, \22requires_grad\22: false, \22shape\22: [2, 2], \22storage_offset\22: 0, \22stride\22: [2, 1]}], \22results\22: [{\22alias_inputs\22: [], \22alias_results\22: [], \22dtype\22: \22float32\22, \22requires_grad\22: false, \22role\22: \22user_output\22, \22shape\22: [2, 3], \22storage_offset\22: 0, \22stride\22: [3, 1]}], \22schema_version\22: 1}", prov.weights_file = "/scratch/loren/backend-verify/out-c1/captures/case-fa62f4b169c26481987f30eda8aeafba5e57450867fd7f932ed809da72286d62/capture-work/weights.safetensors", prov.level = "linalg-on-tensors"} : () -> ()