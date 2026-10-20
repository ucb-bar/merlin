module {
  func.func @forward(%x: tensor<2x4xi8>, %w1: tensor<4x12xi8>, %w2: tensor<12x12xi8>, %state: tensor<2x12xi32>) -> (tensor<2x12xi32>, tensor<2x4x3xi8>, tensor<2x12xi32>) {
    %z = arith.constant 0 : i32
    %lo = arith.constant -128 : i32
    %hi = arith.constant 127 : i32
    %e1 = tensor.empty() : tensor<2x12xi32>
    %zero1 = linalg.fill ins(%z : i32) outs(%e1 : tensor<2x12xi32>) -> tensor<2x12xi32>
    %wide = linalg.generic {indexing_maps=[affine_map<(m,n,k)->(m,k)>,affine_map<(m,n,k)->(k,n)>,affine_map<(m,n,k)->(m,n)>],iterator_types=["parallel","parallel","reduction"]} ins(%x,%w1:tensor<2x4xi8>,tensor<4x12xi8>) outs(%zero1:tensor<2x12xi32>) {
      ^bb_wide(%a_wide:i8,%b_wide:i8,%c_wide:i32):
        %ae_wide=arith.extsi %a_wide:i8 to i32
        %be_wide=arith.extsi %b_wide:i8 to i32
        %mul_wide=arith.muli %ae_wide,%be_wide:i32
        %acc_wide=arith.addi %c_wide,%mul_wide:i32
        linalg.yield %acc_wide:i32
    } -> tensor<2x12xi32>
    %ep = tensor.empty() : tensor<2x12xi8>
    %narrow = linalg.generic {indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>], iterator_types = ["parallel", "parallel"]} ins(%wide, %state : tensor<2x12xi32>, tensor<2x12xi32>) outs(%ep : tensor<2x12xi8>) {
    ^bb0(%a: i32, %s: i32, %unused: i8):
      %sum = arith.addi %a, %s : i32
      %below = arith.cmpi slt, %sum, %lo : i32
      %lower = arith.select %below, %lo, %sum : i32
      %above = arith.cmpi sgt, %lower, %hi : i32
      %bounded = arith.select %above, %hi, %lower : i32
      %byte = arith.trunci %bounded : i32 to i8
      linalg.yield %byte : i8
    } -> tensor<2x12xi8>
    %split = tensor.expand_shape %narrow [[0], [1, 2]] output_shape [2, 3, 4] : tensor<2x12xi8> into tensor<2x3x4xi8>
    %et = tensor.empty() : tensor<2x4x3xi8>
    %packed = linalg.transpose ins(%split : tensor<2x3x4xi8>) outs(%et : tensor<2x4x3xi8>) permutation = [0, 2, 1]
    %matrix = tensor.collapse_shape %packed [[0], [1, 2]] : tensor<2x4x3xi8> into tensor<2x12xi8>
    %e2 = tensor.empty() : tensor<2x12xi32>
    %zero2 = linalg.fill ins(%z : i32) outs(%e2 : tensor<2x12xi32>) -> tensor<2x12xi32>
    %next = linalg.generic {indexing_maps=[affine_map<(m,n,k)->(m,k)>,affine_map<(m,n,k)->(k,n)>,affine_map<(m,n,k)->(m,n)>],iterator_types=["parallel","parallel","reduction"]} ins(%matrix,%w2:tensor<2x12xi8>,tensor<12x12xi8>) outs(%zero2:tensor<2x12xi32>) {
      ^bb_next(%a_next:i8,%b_next:i8,%c_next:i32):
        %ae_next=arith.extsi %a_next:i8 to i32
        %be_next=arith.extsi %b_next:i8 to i32
        %mul_next=arith.muli %ae_next,%be_next:i32
        %acc_next=arith.addi %c_next,%mul_next:i32
        linalg.yield %acc_next:i32
    } -> tensor<2x12xi32>
    return %next, %packed, %wide : tensor<2x12xi32>, tensor<2x4x3xi8>, tensor<2x12xi32>
  }
}
