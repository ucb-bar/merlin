module {
  func.func @forward(%image: tensor<1x1x3x4xi8>, %weight: tensor<2x9xi8>, %state: tensor<2x12xi32>) -> (tensor<2x12xi32>, tensor<1x3x3x1x3x4xi8>, tensor<2x12xi32>) {
    %zero = arith.constant 0 : i32
    %padding = arith.constant 0 : i8
    %ep = tensor.empty() : tensor<1x1x5x6xi8>
    %filled = linalg.fill ins(%padding : i8) outs(%ep : tensor<1x1x5x6xi8>) -> tensor<1x1x5x6xi8>
    %padded = "tensor.insert_slice"(%image, %filled) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 1, 3, 4>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (tensor<1x1x3x4xi8>, tensor<1x1x5x6xi8>) -> tensor<1x1x5x6xi8>
    %ec = tensor.empty() : tensor<1x3x3x1x3x4xi8>
    %columns = linalg.generic {indexing_maps = [affine_map<(c, kh, kw, n, h, w) -> (n, c, h + kh, w + kw)>, affine_map<(c, kh, kw, n, h, w) -> (c, kh, kw, n, h, w)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%padded : tensor<1x1x5x6xi8>) outs(%ec : tensor<1x3x3x1x3x4xi8>) {
    ^bb_columns(%value: i8, %unused: i8):
      linalg.yield %value : i8
    } -> tensor<1x3x3x1x3x4xi8>
    %flat = tensor.collapse_shape %columns [[0, 1, 2, 3, 4, 5]] : tensor<1x3x3x1x3x4xi8> into tensor<108xi8>
    %matrix = tensor.expand_shape %flat [[0, 1]] output_shape [9, 12] : tensor<108xi8> into tensor<9x12xi8>
    %ea = tensor.empty() : tensor<2x12xi32>
    %initial = linalg.fill ins(%zero : i32) outs(%ea : tensor<2x12xi32>) -> tensor<2x12xi32>
    %wide = linalg.generic {indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%weight, %matrix : tensor<2x9xi8>, tensor<9x12xi8>) outs(%initial : tensor<2x12xi32>) {
    ^bb_dot(%a: i8, %b: i8, %acc: i32):
      %ae = arith.extsi %a : i8 to i32
      %be = arith.extsi %b : i8 to i32
      %product = arith.muli %ae, %be : i32
      %sum = arith.addi %acc, %product : i32
      linalg.yield %sum : i32
    } -> tensor<2x12xi32>
    %es = tensor.empty() : tensor<2x12xi32>
    %next = linalg.generic {indexing_maps = [affine_map<(m, n) -> (m, n)>, affine_map<(m, n) -> (m, n)>, affine_map<(m, n) -> (m, n)>], iterator_types = ["parallel", "parallel"]} ins(%wide, %state : tensor<2x12xi32>, tensor<2x12xi32>) outs(%es : tensor<2x12xi32>) {
    ^bb_state(%value: i32, %prior: i32, %unused: i32):
      %updated = arith.addi %value, %prior : i32
      linalg.yield %updated : i32
    } -> tensor<2x12xi32>
    return %next, %columns, %wide : tensor<2x12xi32>, tensor<1x3x3x1x3x4xi8>, tensor<2x12xi32>
  }
}
