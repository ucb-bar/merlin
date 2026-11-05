module {
  func.func @forward(%x: tensor<2x4xi8>, %w1: tensor<4x6xi8>, %w2: tensor<8x5xi8>, %state: tensor<2x6xi32>) -> (tensor<2x6xi32>, tensor<4x8xi8>, tensor<4x5xi32>, tensor<4x8xi8>, tensor<2x6xi32>) {
    %zero32 = arith.constant 0 : i32
    %zero8 = arith.constant 0 : i8
    %e1 = tensor.empty() : tensor<2x6xi32>
    %init1 = linalg.fill ins(%zero32 : i32) outs(%e1 : tensor<2x6xi32>) -> tensor<2x6xi32>
    %wide = linalg.generic {indexing_maps=[affine_map<(m,n,k)->(m,k)>,affine_map<(m,n,k)->(k,n)>,affine_map<(m,n,k)->(m,n)>],iterator_types=["parallel","parallel","reduction"]} ins(%x,%w1:tensor<2x4xi8>,tensor<4x6xi8>) outs(%init1:tensor<2x6xi32>) {
    ^bb0(%a:i8,%b:i8,%acc:i32):
      %ae = arith.extsi %a : i8 to i32
      %be = arith.extsi %b : i8 to i32
      %product = arith.muli %ae,%be : i32
      %sum = arith.addi %acc,%product : i32
      linalg.yield %sum : i32
    } -> tensor<2x6xi32>
    %es = tensor.empty() : tensor<2x6xi32>
    %next_state = linalg.generic {indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>],iterator_types=["parallel","parallel"]} ins(%wide,%state:tensor<2x6xi32>,tensor<2x6xi32>) outs(%es:tensor<2x6xi32>) {
    ^bb1(%v:i32,%s:i32,%unused:i32):
      %sum_state = arith.addi %v,%s : i32
      linalg.yield %sum_state : i32
    } -> tensor<2x6xi32>
    %eq = tensor.empty() : tensor<2x6xi8>
    %pad = tensor.splat %zero8 : tensor<4x8xi8>
    %narrow = linalg.generic {indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>],iterator_types=["parallel","parallel"]} ins(%next_state:tensor<2x6xi32>) outs(%eq:tensor<2x6xi8>) {
    ^bb2(%value:i32,%unused_byte:i8):
      %lo = arith.constant -128 : i32
      %hi = arith.constant 127 : i32
      %below = arith.cmpi slt,%value,%lo : i32
      %lower = arith.select %below,%lo,%value : i32
      %above = arith.cmpi sgt,%lower,%hi : i32
      %bounded = arith.select %above,%hi,%lower : i32
      %byte = arith.trunci %bounded : i32 to i8
      linalg.yield %byte : i8
    } -> tensor<2x6xi8>
    %padded = "tensor.insert_slice"(%narrow,%pad) <{static_offsets=array<i64:1,1>,static_sizes=array<i64:2,6>,static_strides=array<i64:1,1>,operandSegmentSizes=array<i32:1,1,0,0,0>}> : (tensor<2x6xi8>,tensor<4x8xi8>) -> tensor<4x8xi8>
    %e2 = tensor.empty() : tensor<4x5xi32>
    %init2 = linalg.fill ins(%zero32 : i32) outs(%e2 : tensor<4x5xi32>) -> tensor<4x5xi32>
    %next = linalg.generic {indexing_maps=[affine_map<(m,n,k)->(m,k)>,affine_map<(m,n,k)->(k,n)>,affine_map<(m,n,k)->(m,n)>],iterator_types=["parallel","parallel","reduction"]} ins(%padded,%w2:tensor<4x8xi8>,tensor<8x5xi8>) outs(%init2:tensor<4x5xi32>) {
    ^bb3(%left:i8,%right:i8,%old:i32):
      %le = arith.extsi %left : i8 to i32
      %re = arith.extsi %right : i8 to i32
      %mul = arith.muli %le,%re : i32
      %add = arith.addi %old,%mul : i32
      linalg.yield %add : i32
    } -> tensor<4x5xi32>
    return %next_state,%padded,%next,%pad,%wide : tensor<2x6xi32>,tensor<4x8xi8>,tensor<4x5xi32>,tensor<4x8xi8>,tensor<2x6xi32>
  }
}
