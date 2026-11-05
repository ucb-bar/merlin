"builtin.module"() ({
  "func.func"() <{sym_name = "forward", function_type = (tensor<16x16xi8>) -> tensor<16x16xi32>}> ({
  ^bb0(%0: tensor<16x16xi8>):
    %1 = "tensor.empty"() : () -> tensor<16x16xi32>
    %2 = "linalg.generic"(%0, %1) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb1(%3: i8, %4: i32):
      %5 = "arith.extsi"(%3) : (i8) -> i32
      %6 = "arith.constant"() <{value = 2 : i32}> : () -> i32
      %7 = "arith.muli"(%5, %6) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "linalg.yield"(%7) : (i32) -> ()
    }) {prov.region_id = "widened_0", prov.family = "elementwise", prov.op = "mul_Scalar", prov.aten = "aten.mul.Scalar", prov.orig_dtype = "int32"} : (tensor<16x16xi8>, tensor<16x16xi32>) -> tensor<16x16xi32>
    "func.return"(%2) : (tensor<16x16xi32>) -> ()
  }) : () -> ()
}) {m2m.result_layout = "{\22authority\22: \22same_conversion_exported_program\22, \22inputs\22: [{\22dtype\22: \22int8\22, \22requires_grad\22: false, \22shape\22: [16, 16], \22storage_offset\22: 0, \22stride\22: [16, 1]}], \22results\22: [{\22alias_inputs\22: [], \22alias_results\22: [], \22dtype\22: \22int32\22, \22requires_grad\22: false, \22role\22: \22user_output\22, \22shape\22: [16, 16], \22storage_offset\22: 0, \22stride\22: [16, 1]}], \22schema_version\22: 1}", prov.level = "linalg-on-tensors"} : () -> ()
