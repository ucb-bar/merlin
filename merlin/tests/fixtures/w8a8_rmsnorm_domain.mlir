// Diagnostic slice of the retained TinyLlama development capture, not a full-model claim.
// Original authoritative export SHA-256:
// 651a1568cb65ee40c4fff5eb00bd9067a1f505954b8e37754cbfbb100808a908
// Keep the 21 original producer operations, region bodies, constants and attributes unchanged.
// Embedding %v979 and gain %v1 become explicit inputs; return inverse, normalization and gain product.
"builtin.module"() ({
^bb0:
  "func.func"() ({
    ^bb0(%v979: tensor<1x8x2048xf32>, %v1: tensor<2048xf32>):
      %v1205 = "tensor.empty"() : () -> (tensor<1x8x2048xf32>)
      %v1206 = "linalg.generic"(%v979, %v1205) ({
        ^bb0(%v1207: f32, %v1208: f32):
          %v1209 = "arith.constant"() {value = 0x40000000 : f32} : () -> (f32)
          %v1210 = "arith.mulf"(%v1207, %v1207) {fastmath = #arith.fastmath<none>} : (f32, f32) -> (f32)
          "linalg.yield"(%v1210) : (f32) -> ()
        }) {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>, prov._pattern_hint = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.family = "elementwise", prov.fqn = "lm.model.layers.0.input_layernorm", prov.module = "lm", prov.op = "pow", prov.orig_dtype = "float32", prov.region_id = "pow_0"} : (tensor<1x8x2048xf32>, tensor<1x8x2048xf32>) -> (tensor<1x8x2048xf32>)
      %v1211 = "arith.constant"() {prov._pattern_hint = "reduce_mean", prov.aten = "aten.mean.dim", prov.family = "reduce", prov.fqn = "lm.model.layers.0.input_layernorm", prov.module = "lm", prov.op = "reduce_mean", prov.orig_dtype = "float32", prov.region_id = "reduce_0", value = 0x00000000 : f32} : () -> (f32)
      %v1212 = "tensor.splat"(%v1211) {prov._pattern_hint = "reduce_mean", prov.aten = "aten.mean.dim", prov.family = "reduce", prov.fqn = "lm.model.layers.0.input_layernorm", prov.module = "lm", prov.op = "reduce_mean", prov.orig_dtype = "float32", prov.region_id = "reduce_0"} : (f32) -> (tensor<1x8xf32>)
      %v1213 = "linalg.reduce"(%v1206, %v1212) ({
        ^bb0(%v1214: f32, %v1215: f32):
          %v1216 = "arith.addf"(%v1214, %v1215) {fastmath = #arith.fastmath<none>} : (f32, f32) -> (f32)
          "linalg.yield"(%v1216) : (f32) -> ()
        }) {dimensions = array<i64: 2>} : (tensor<1x8x2048xf32>, tensor<1x8xf32>) -> (tensor<1x8xf32>)
      %v1217 = "arith.constant"() {prov._pattern_hint = "reduce_mean", prov.aten = "aten.mean.dim", prov.family = "reduce", prov.fqn = "lm.model.layers.0.input_layernorm", prov.module = "lm", prov.op = "reduce_mean", prov.orig_dtype = "float32", prov.region_id = "reduce_0", value = 0x45000000 : f32} : () -> (f32)
      %v1218 = "tensor.splat"(%v1217) {prov._pattern_hint = "reduce_mean", prov.aten = "aten.mean.dim", prov.family = "reduce", prov.fqn = "lm.model.layers.0.input_layernorm", prov.module = "lm", prov.op = "reduce_mean", prov.orig_dtype = "float32", prov.region_id = "reduce_0"} : (f32) -> (tensor<1x8xf32>)
      %v1219 = "tensor.empty"() : () -> (tensor<1x8xf32>)
      %v1220 = "linalg.generic"(%v1213, %v1218, %v1219) ({
        ^bb0(%v1221: f32, %v1222: f32, %v1223: f32):
          %v1224 = "arith.divf"(%v1221, %v1222) {fastmath = #arith.fastmath<none>} : (f32, f32) -> (f32)
          "linalg.yield"(%v1224) : (f32) -> ()
        }) {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>, prov._pattern_hint = "reduce_mean", prov.aten = "aten.mean.dim", prov.family = "reduce", prov.fqn = "lm.model.layers.0.input_layernorm", prov.module = "lm", prov.op = "reduce_mean", prov.orig_dtype = "float32", prov.region_id = "reduce_0"} : (tensor<1x8xf32>, tensor<1x8xf32>, tensor<1x8xf32>) -> (tensor<1x8xf32>)
      %v1225 = "tensor.collapse_shape"(%v1220) {prov._pattern_hint = "reduce_mean", prov.aten = "aten.mean.dim", prov.family = "reduce", prov.fqn = "lm.model.layers.0.input_layernorm", prov.module = "lm", prov.op = "reduce_mean", prov.orig_dtype = "float32", prov.region_id = "reduce_0", reassociation = [[0 : i64, 1 : i64]]} : (tensor<1x8xf32>) -> (tensor<8xf32>)
      %v1226 = "tensor.expand_shape"(%v1225) {prov._pattern_hint = "reduce_mean", prov.aten = "aten.mean.dim", prov.family = "reduce", prov.fqn = "lm.model.layers.0.input_layernorm", prov.module = "lm", prov.op = "reduce_mean", prov.orig_dtype = "float32", prov.region_id = "reduce_0", reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 1, 8, 1>} : (tensor<8xf32>) -> (tensor<1x8x1xf32>)
      %v1227 = "arith.constant"() {prov._pattern_hint = "add", prov.aten = "aten.add.Tensor", prov.family = "elementwise", prov.fqn = "lm.model.layers.0.input_layernorm", prov.module = "lm", prov.op = "add", prov.orig_dtype = "float32", prov.region_id = "add_3", value = 0x3727c5ac : f32} : () -> (f32)
      %v1228 = "tensor.splat"(%v1227) {prov._pattern_hint = "add", prov.aten = "aten.add.Tensor", prov.family = "elementwise", prov.fqn = "lm.model.layers.0.input_layernorm", prov.module = "lm", prov.op = "add", prov.orig_dtype = "float32", prov.region_id = "add_3"} : (f32) -> (tensor<1x8x1xf32>)
      %v1229 = "tensor.empty"() : () -> (tensor<1x8x1xf32>)
      %v1230 = "linalg.generic"(%v1226, %v1228, %v1229) ({
        ^bb0(%v1231: f32, %v1232: f32, %v1233: f32):
          %v1234 = "arith.addf"(%v1231, %v1232) {fastmath = #arith.fastmath<none>} : (f32, f32) -> (f32)
          "linalg.yield"(%v1234) : (f32) -> ()
        }) {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>, prov._pattern_hint = "add", prov.aten = "aten.add.Tensor", prov.family = "elementwise", prov.fqn = "lm.model.layers.0.input_layernorm", prov.module = "lm", prov.op = "add", prov.orig_dtype = "float32", prov.region_id = "add_3"} : (tensor<1x8x1xf32>, tensor<1x8x1xf32>, tensor<1x8x1xf32>) -> (tensor<1x8x1xf32>)
      %v1235 = "tensor.empty"() : () -> (tensor<1x8x1xf32>)
      %v1236 = "linalg.generic"(%v1230, %v1235) ({
        ^bb0(%v1237: f32, %v1238: f32):
          %v1239 = "math.rsqrt"(%v1237) {fastmath = #arith.fastmath<none>} : (f32) -> (f32)
          "linalg.yield"(%v1239) : (f32) -> ()
        }) {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>, prov._pattern_hint = "rsqrt", prov.aten = "aten.rsqrt.default", prov.family = "elementwise", prov.fqn = "lm.model.layers.0.input_layernorm", prov.module = "lm", prov.op = "rsqrt", prov.orig_dtype = "float32", prov.region_id = "rsqrt_0"} : (tensor<1x8x1xf32>, tensor<1x8x1xf32>) -> (tensor<1x8x1xf32>)
      %v1240 = "tensor.empty"() : () -> (tensor<1x8x2048xf32>)
      %v1241 = "linalg.generic"(%v979, %v1236, %v1240) ({
        ^bb0(%v1242: f32, %v1243: f32, %v1244: f32):
          %v1245 = "arith.mulf"(%v1242, %v1243) {fastmath = #arith.fastmath<none>} : (f32, f32) -> (f32)
          "linalg.yield"(%v1245) : (f32) -> ()
        }) {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>, prov._pattern_hint = "mul", prov.aten = "aten.mul.Tensor", prov.family = "elementwise", prov.fqn = "lm.model.layers.0.input_layernorm", prov.module = "lm", prov.op = "mul", prov.orig_dtype = "float32", prov.region_id = "mul_2"} : (tensor<1x8x2048xf32>, tensor<1x8x1xf32>, tensor<1x8x2048xf32>) -> (tensor<1x8x2048xf32>)
      %v1246 = "tensor.empty"() : () -> (tensor<1x8x2048xf32>)
      %v1247 = "linalg.generic"(%v1, %v1241, %v1246) ({
        ^bb0(%v1248: f32, %v1249: f32, %v1250: f32):
          %v1251 = "arith.mulf"(%v1248, %v1249) {fastmath = #arith.fastmath<none>} : (f32, f32) -> (f32)
          "linalg.yield"(%v1251) : (f32) -> ()
        }) {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>, prov._pattern_hint = "mul", prov.aten = "aten.mul.Tensor", prov.family = "elementwise", prov.fqn = "lm.model.layers.0.input_layernorm", prov.module = "lm", prov.op = "mul", prov.orig_dtype = "float32", prov.region_id = "mul_3"} : (tensor<2048xf32>, tensor<1x8x2048xf32>, tensor<1x8x2048xf32>) -> (tensor<1x8x2048xf32>)
      "func.return"(%v1236, %v1241, %v1247) : (tensor<1x8x1xf32>, tensor<1x8x2048xf32>, tensor<1x8x2048xf32>) -> ()
    }) {function_type = (tensor<1x8x2048xf32>, tensor<2048xf32>) -> (tensor<1x8x1xf32>, tensor<1x8x2048xf32>, tensor<1x8x2048xf32>), sym_name = "forward"} : () -> ()
}) : () -> ()

