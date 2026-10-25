"builtin.module"() ({
^bb0:
  "func.func"() ({
    ^bb0(%v201: tensor<32xf32>):
      %v985 = "tensor.empty"() : () -> (tensor<8xi64>)
      %v986 = "linalg.generic"(%v985) ({
        ^bb0(%v987: i64):
          %v988 = "linalg.index"() {dim = 0 : i64} : () -> (index)
          %v989 = "arith.index_cast"(%v988) : (index) -> (i64)
          %v990 = "arith.constant"() {value = 1 : i64} : () -> (i64)
          %v991 = "arith.muli"(%v989, %v990) {overflowFlags = #arith.overflow<none>} : (i64, i64) -> (i64)
          %v992 = "arith.constant"() {value = 0 : i64} : () -> (i64)
          %v993 = "arith.addi"(%v992, %v991) {overflowFlags = #arith.overflow<none>} : (i64, i64) -> (i64)
          "linalg.yield"(%v993) : (i64) -> ()
        }) {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 0, 1>, prov._pattern_hint = "arange", prov.aten = "aten.arange.start_step", prov.family = "iota", prov.fqn = "lm.model", prov.module = "lm", prov.op = "arange", prov.orig_dtype = "int64", prov.region_id = "iota_0"} : (tensor<8xi64>) -> (tensor<8xi64>)
      %v994 = "arith.constant"() {prov._pattern_hint = "add", prov.aten = "aten.add.Tensor", prov.family = "elementwise", prov.fqn = "lm.model", prov.module = "lm", prov.op = "add", prov.orig_dtype = "int64", prov.region_id = "add_0", value = 0 : i64} : () -> (i64)
      %v995 = "tensor.splat"(%v994) {prov._pattern_hint = "add", prov.aten = "aten.add.Tensor", prov.family = "elementwise", prov.fqn = "lm.model", prov.module = "lm", prov.op = "add", prov.orig_dtype = "int64", prov.region_id = "add_0"} : (i64) -> (tensor<8xi64>)
      %v996 = "tensor.empty"() : () -> (tensor<8xi64>)
      %v997 = "linalg.generic"(%v986, %v995, %v996) ({
        ^bb0(%v998: i64, %v999: i64, %v1000: i64):
          %v1001 = "arith.addi"(%v998, %v999) {overflowFlags = #arith.overflow<none>} : (i64, i64) -> (i64)
          "linalg.yield"(%v1001) : (i64) -> ()
        }) {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>, prov._pattern_hint = "add", prov.aten = "aten.add.Tensor", prov.family = "elementwise", prov.fqn = "lm.model", prov.module = "lm", prov.op = "add", prov.orig_dtype = "int64", prov.region_id = "add_0"} : (tensor<8xi64>, tensor<8xi64>, tensor<8xi64>) -> (tensor<8xi64>)
      %v1002 = "tensor.expand_shape"(%v997) {prov._pattern_hint = "unsqueeze", prov.aten = "aten.unsqueeze.default", prov.family = "layout", prov.fqn = "lm.model", prov.module = "lm", prov.op = "unsqueeze", prov.orig_dtype = "int64", prov.region_id = "unsqueeze_0", reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 1, 8>} : (tensor<8xi64>) -> (tensor<1x8xi64>)
      %v1144 = "tensor.expand_shape"(%v201) {prov._pattern_hint = "unsqueeze", prov.aten = "aten.unsqueeze.default", prov.family = "layout", prov.fqn = "lm.model.rotary_emb", prov.module = "lm", prov.op = "unsqueeze", prov.orig_dtype = "float32", prov.region_id = "unsqueeze_10", reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 1, 32>} : (tensor<32xf32>) -> (tensor<1x32xf32>)
      %v1145 = "tensor.collapse_shape"(%v1144) {prov._pattern_hint = "unsqueeze", prov.aten = "aten.unsqueeze.default", prov.family = "layout", prov.fqn = "lm.model.rotary_emb", prov.module = "lm", prov.op = "unsqueeze", prov.orig_dtype = "float32", prov.region_id = "unsqueeze_11", reassociation = [[0 : i64, 1 : i64]]} : (tensor<1x32xf32>) -> (tensor<32xf32>)
      %v1146 = "tensor.expand_shape"(%v1145) {prov._pattern_hint = "unsqueeze", prov.aten = "aten.unsqueeze.default", prov.family = "layout", prov.fqn = "lm.model.rotary_emb", prov.module = "lm", prov.op = "unsqueeze", prov.orig_dtype = "float32", prov.region_id = "unsqueeze_11", reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 1, 32, 1>} : (tensor<32xf32>) -> (tensor<1x32x1xf32>)
      %v1147 = "tensor.empty"() : () -> (tensor<1x32x1xf32>)
      %v1148 = "linalg.generic"(%v1146, %v1147) ({
        ^bb0(%v1149: f32, %v1150: f32):
          "linalg.yield"(%v1149) : (f32) -> ()
        }) {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>, prov._pattern_hint = "expand", prov.aten = "aten.expand.default", prov.family = "layout", prov.fqn = "lm.model.rotary_emb", prov.module = "lm", prov.op = "expand", prov.orig_dtype = "float32", prov.region_id = "expand_1"} : (tensor<1x32x1xf32>, tensor<1x32x1xf32>) -> (tensor<1x32x1xf32>)
      %v1151 = "tensor.collapse_shape"(%v1002) {prov._pattern_hint = "unsqueeze", prov.aten = "aten.unsqueeze.default", prov.family = "layout", prov.fqn = "lm.model.rotary_emb", prov.module = "lm", prov.op = "unsqueeze", prov.orig_dtype = "int64", prov.region_id = "unsqueeze_12", reassociation = [[0 : i64, 1 : i64]]} : (tensor<1x8xi64>) -> (tensor<8xi64>)
      %v1152 = "tensor.expand_shape"(%v1151) {prov._pattern_hint = "unsqueeze", prov.aten = "aten.unsqueeze.default", prov.family = "layout", prov.fqn = "lm.model.rotary_emb", prov.module = "lm", prov.op = "unsqueeze", prov.orig_dtype = "int64", prov.region_id = "unsqueeze_12", reassociation = [[0 : i64, 1 : i64, 2 : i64]], static_output_shape = array<i64: 1, 1, 8>} : (tensor<8xi64>) -> (tensor<1x1x8xi64>)
      %v1153 = "tensor.empty"() : () -> (tensor<1x1x8xf32>)
      %v1154 = "linalg.generic"(%v1152, %v1153) ({
        ^bb0(%v1155: i64, %v1156: f32):
          %v1157 = "arith.sitofp"(%v1155) : (i64) -> (f32)
          "linalg.yield"(%v1157) : (f32) -> ()
        }) {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>, prov._pattern_hint = "dtype_cast", prov.aten = "aten._to_copy.default", prov.family = "cast", prov.fqn = "lm.model.rotary_emb", prov.module = "lm", prov.op = "dtype_cast", prov.orig_dtype = "float32", prov.region_id = "dtype_cast_0"} : (tensor<1x1x8xi64>, tensor<1x1x8xf32>) -> (tensor<1x1x8xf32>)
      %v1158 = "tensor.empty"() : () -> (tensor<1x32x1xf32>)
      %v1159 = "linalg.generic"(%v1148, %v1158) ({
        ^bb0(%v1160: f32, %v1161: f32):
          "linalg.yield"(%v1160) : (f32) -> ()
        }) {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>, prov._pattern_hint = "expand", prov.aten = "aten.expand.default", prov.family = "layout", prov.fqn = "lm.model.rotary_emb", prov.module = "lm", prov.op = "expand", prov.orig_dtype = "float32", prov.region_id = "expand_2"} : (tensor<1x32x1xf32>, tensor<1x32x1xf32>) -> (tensor<1x32x1xf32>)
      %v1162 = "tensor.empty"() : () -> (tensor<1x1x8xf32>)
      %v1163 = "linalg.generic"(%v1154, %v1162) ({
        ^bb0(%v1164: f32, %v1165: f32):
          "linalg.yield"(%v1164) : (f32) -> ()
        }) {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>, prov._pattern_hint = "expand", prov.aten = "aten.expand.default", prov.family = "layout", prov.fqn = "lm.model.rotary_emb", prov.module = "lm", prov.op = "expand", prov.orig_dtype = "float32", prov.region_id = "expand_3"} : (tensor<1x1x8xf32>, tensor<1x1x8xf32>) -> (tensor<1x1x8xf32>)
      %v1166 = "arith.constant"() {prov._pattern_hint = "batch_matmul", prov.aten = "aten.bmm.default", prov.family = "contraction", prov.fqn = "lm.model.rotary_emb", prov.module = "lm", prov.op = "batch_matmul", prov.orig_dtype = "float32", prov.region_id = "matmul_0", value = 0x00000000 : f32} : () -> (f32)
      %v1167 = "tensor.splat"(%v1166) {prov._pattern_hint = "batch_matmul", prov.aten = "aten.bmm.default", prov.family = "contraction", prov.fqn = "lm.model.rotary_emb", prov.module = "lm", prov.op = "batch_matmul", prov.orig_dtype = "float32", prov.region_id = "matmul_0"} : (f32) -> (tensor<1x32x8xf32>)
      %v1168 = "linalg.generic"(%v1159, %v1163, %v1167) ({
        ^bb0(%v1169: f32, %v1170: f32, %v1171: f32):
          %v1172 = "arith.mulf"(%v1169, %v1170) {fastmath = #arith.fastmath<none>} : (f32, f32) -> (f32)
          %v1173 = "arith.addf"(%v1171, %v1172) {fastmath = #arith.fastmath<none>} : (f32, f32) -> (f32)
          "linalg.yield"(%v1173) : (f32) -> ()
        }) {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>, prov._pattern_hint = "batch_matmul", prov.aten = "aten.bmm.default", prov.family = "contraction", prov.fqn = "lm.model.rotary_emb", prov.module = "lm", prov.op = "batch_matmul", prov.orig_dtype = "float32", prov.region_id = "matmul_0"} : (tensor<1x32x1xf32>, tensor<1x1x8xf32>, tensor<1x32x8xf32>) -> (tensor<1x32x8xf32>)
      %v1174 = "tensor.empty"() : () -> (tensor<1x8x32xf32>)
      %v1175 = "linalg.transpose"(%v1168, %v1174) ({
        ^bb0(%v1176: f32, %v1177: f32):
          "linalg.yield"(%v1176) : (f32) -> ()
        }) {permutation = array<i64: 0, 2, 1>} : (tensor<1x32x8xf32>, tensor<1x8x32xf32>) -> (tensor<1x8x32xf32>)
      %v1178 = "tensor.concat"(%v1175, %v1175) {dim = 2 : i64, prov._pattern_hint = "cat", prov.aten = "aten.cat.default", prov.family = "concat", prov.fqn = "lm.model.rotary_emb", prov.module = "lm", prov.op = "cat", prov.orig_dtype = "float32", prov.region_id = "cat_1"} : (tensor<1x8x32xf32>, tensor<1x8x32xf32>) -> (tensor<1x8x64xf32>)
      %v1179 = "tensor.empty"() : () -> (tensor<1x8x64xf32>)
      %v1180 = "linalg.generic"(%v1178, %v1179) ({
        ^bb0(%v1181: f32, %v1182: f32):
          %v1183 = "math.cos"(%v1181) {fastmath = #arith.fastmath<none>} : (f32) -> (f32)
          "linalg.yield"(%v1183) : (f32) -> ()
        }) {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>, prov._pattern_hint = "cos", prov.aten = "aten.cos.default", prov.family = "elementwise", prov.fqn = "lm.model.rotary_emb", prov.module = "lm", prov.op = "cos", prov.orig_dtype = "float32", prov.region_id = "cos_0"} : (tensor<1x8x64xf32>, tensor<1x8x64xf32>) -> (tensor<1x8x64xf32>)
      %v1184 = "arith.constant"() {prov._pattern_hint = "mul", prov.aten = "aten.mul.Tensor", prov.family = "elementwise", prov.fqn = "lm.model.rotary_emb", prov.module = "lm", prov.op = "mul", prov.orig_dtype = "float32", prov.region_id = "mul_0", value = 0x3f800000 : f32} : () -> (f32)
      %v1185 = "tensor.splat"(%v1184) {prov._pattern_hint = "mul", prov.aten = "aten.mul.Tensor", prov.family = "elementwise", prov.fqn = "lm.model.rotary_emb", prov.module = "lm", prov.op = "mul", prov.orig_dtype = "float32", prov.region_id = "mul_0"} : (f32) -> (tensor<1x8x64xf32>)
      %v1186 = "tensor.empty"() : () -> (tensor<1x8x64xf32>)
      %v1187 = "linalg.generic"(%v1180, %v1185, %v1186) ({
        ^bb0(%v1188: f32, %v1189: f32, %v1190: f32):
          %v1191 = "arith.mulf"(%v1188, %v1189) {fastmath = #arith.fastmath<none>} : (f32, f32) -> (f32)
          "linalg.yield"(%v1191) : (f32) -> ()
        }) {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>, prov._pattern_hint = "mul", prov.aten = "aten.mul.Tensor", prov.family = "elementwise", prov.fqn = "lm.model.rotary_emb", prov.module = "lm", prov.op = "mul", prov.orig_dtype = "float32", prov.region_id = "mul_0"} : (tensor<1x8x64xf32>, tensor<1x8x64xf32>, tensor<1x8x64xf32>) -> (tensor<1x8x64xf32>)
      %v1192 = "tensor.empty"() : () -> (tensor<1x8x64xf32>)
      %v1193 = "linalg.generic"(%v1178, %v1192) ({
        ^bb0(%v1194: f32, %v1195: f32):
          %v1196 = "math.sin"(%v1194) {fastmath = #arith.fastmath<none>} : (f32) -> (f32)
          "linalg.yield"(%v1196) : (f32) -> ()
        }) {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>, prov._pattern_hint = "sin", prov.aten = "aten.sin.default", prov.family = "elementwise", prov.fqn = "lm.model.rotary_emb", prov.module = "lm", prov.op = "sin", prov.orig_dtype = "float32", prov.region_id = "sin_0"} : (tensor<1x8x64xf32>, tensor<1x8x64xf32>) -> (tensor<1x8x64xf32>)
      %v1197 = "arith.constant"() {prov._pattern_hint = "mul", prov.aten = "aten.mul.Tensor", prov.family = "elementwise", prov.fqn = "lm.model.rotary_emb", prov.module = "lm", prov.op = "mul", prov.orig_dtype = "float32", prov.region_id = "mul_1", value = 0x3f800000 : f32} : () -> (f32)
      %v1198 = "tensor.splat"(%v1197) {prov._pattern_hint = "mul", prov.aten = "aten.mul.Tensor", prov.family = "elementwise", prov.fqn = "lm.model.rotary_emb", prov.module = "lm", prov.op = "mul", prov.orig_dtype = "float32", prov.region_id = "mul_1"} : (f32) -> (tensor<1x8x64xf32>)
      %v1199 = "tensor.empty"() : () -> (tensor<1x8x64xf32>)
      %v1200 = "linalg.generic"(%v1193, %v1198, %v1199) ({
        ^bb0(%v1201: f32, %v1202: f32, %v1203: f32):
          %v1204 = "arith.mulf"(%v1201, %v1202) {fastmath = #arith.fastmath<none>} : (f32, f32) -> (f32)
          "linalg.yield"(%v1204) : (f32) -> ()
        }) {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>, prov._pattern_hint = "mul", prov.aten = "aten.mul.Tensor", prov.family = "elementwise", prov.fqn = "lm.model.rotary_emb", prov.module = "lm", prov.op = "mul", prov.orig_dtype = "float32", prov.region_id = "mul_1"} : (tensor<1x8x64xf32>, tensor<1x8x64xf32>, tensor<1x8x64xf32>) -> (tensor<1x8x64xf32>)
      "func.return"(%v1178, %v1187, %v1200) : (tensor<1x8x64xf32>, tensor<1x8x64xf32>, tensor<1x8x64xf32>) -> ()
    }) {function_type = (tensor<32xf32>) -> (tensor<1x8x64xf32>, tensor<1x8x64xf32>, tensor<1x8x64xf32>), sym_name = "forward"} : () -> ()
}) : () -> ()
