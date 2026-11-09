// Public synthetic integer source; each K=5 MAC prefix has |sum| <= 81920.
module {
  func.func @forward(%x: tensor<2x5xi8>, %bias0: tensor<2x5xi8>, %bias1: tensor<2x5xi8>, %w0: tensor<5x5xi8>, %w1: tensor<5x3xi8>, %state: tensor<2x5xi32>) -> (tensor<2x5xi32>, tensor<2x5xi32>, tensor<2x3xi32>, tensor<2x5xi8>, tensor<2x5xi8>, tensor<2x5xi8>, tensor<5x5xi8>, tensor<5x3xi8>, tensor<2x5xi32>) {
    %z = arith.constant 0 : i32
    %pe = tensor.empty() : tensor<2x5xi8>
    %e0 = tensor.empty() : tensor<2x5xi8>
    %e1 = tensor.empty() : tensor<2x5xi8>
    %c0e = tensor.empty() : tensor<2x5xi32>
    %c1e = tensor.empty() : tensor<2x3xi32>
    %se = tensor.empty() : tensor<2x5xi32>
    %c0z = linalg.fill ins(%z : i32) outs(%c0e : tensor<2x5xi32>) -> tensor<2x5xi32>
    %c1z = linalg.fill ins(%z : i32) outs(%c1e : tensor<2x3xi32>) -> tensor<2x3xi32>
    %p = linalg.generic {indexing_maps=[affine_map<(m,k)->(m,k)>,affine_map<(m,k)->(m,k)>,affine_map<(m,k)->(m,k)>],iterator_types=["parallel","parallel"]} ins(%x,%state:tensor<2x5xi8>,tensor<2x5xi32>) outs(%pe:tensor<2x5xi8>) {
      ^bbp(%px:i8,%ps:i32,%pi:i8):
        %narrow = arith.trunci %ps : i32 to i8
        %pv = arith.addi %px,%narrow : i8
        linalg.yield %pv : i8
    } -> tensor<2x5xi8>
    %a = linalg.generic {indexing_maps=[affine_map<(m,k)->(m,k)>,affine_map<(m,k)->(m,k)>,affine_map<(m,k)->(m,k)>],iterator_types=["parallel","parallel"]} ins(%p,%bias0:tensor<2x5xi8>,tensor<2x5xi8>) outs(%e0:tensor<2x5xi8>) {
      ^bba(%ap:i8,%ab:i8,%ai:i8):
        %av = arith.addi %ap,%ab : i8
        linalg.yield %av : i8
    } -> tensor<2x5xi8>
    %b = linalg.generic {indexing_maps=[affine_map<(m,k)->(m,k)>,affine_map<(m,k)->(m,k)>,affine_map<(m,k)->(m,k)>],iterator_types=["parallel","parallel"]} ins(%p,%bias1:tensor<2x5xi8>,tensor<2x5xi8>) outs(%e1:tensor<2x5xi8>) {
      ^bbb(%bp:i8,%bb:i8,%bi:i8):
        %bv = arith.muli %bp,%bb : i8
        linalg.yield %bv : i8
    } -> tensor<2x5xi8>
    %c0 = linalg.generic {indexing_maps=[affine_map<(m,n,k)->(m,k)>,affine_map<(m,n,k)->(k,n)>,affine_map<(m,n,k)->(m,n)>],iterator_types=["parallel","parallel","reduction"]} ins(%a,%w0:tensor<2x5xi8>,tensor<5x5xi8>) outs(%c0z:tensor<2x5xi32>) {
      ^bbc0(%a0:i8,%b0:i8,%acc0:i32):
        %a0x = arith.extsi %a0 : i8 to i32
        %b0x = arith.extsi %b0 : i8 to i32
        %v0 = arith.muli %a0x,%b0x : i32
        %r0 = arith.addi %acc0,%v0 : i32
        linalg.yield %r0 : i32
    } -> tensor<2x5xi32>
    %c1 = linalg.generic {indexing_maps=[affine_map<(m,n,k)->(m,k)>,affine_map<(m,n,k)->(k,n)>,affine_map<(m,n,k)->(m,n)>],iterator_types=["parallel","parallel","reduction"]} ins(%b,%w1:tensor<2x5xi8>,tensor<5x3xi8>) outs(%c1z:tensor<2x3xi32>) {
      ^bbc1(%a1:i8,%b1:i8,%acc1:i32):
        %a1x = arith.extsi %a1 : i8 to i32
        %b1x = arith.extsi %b1 : i8 to i32
        %v1 = arith.muli %a1x,%b1x : i32
        %r1 = arith.addi %acc1,%v1 : i32
        linalg.yield %r1 : i32
    } -> tensor<2x3xi32>
    %next = linalg.generic {indexing_maps=[affine_map<(m,k)->(m,k)>,affine_map<(m,k)->(m,k)>,affine_map<(m,k)->(m,k)>],iterator_types=["parallel","parallel"]} ins(%c0,%state:tensor<2x5xi32>,tensor<2x5xi32>) outs(%se:tensor<2x5xi32>) {
      ^bbs(%sc:i32,%ss:i32,%si:i32):
        %sv = arith.addi %sc,%ss : i32
        linalg.yield %sv : i32
    } -> tensor<2x5xi32>
    return %next,%c0,%c1,%x,%bias0,%bias1,%w0,%w1,%state : tensor<2x5xi32>,tensor<2x5xi32>,tensor<2x3xi32>,tensor<2x5xi8>,tensor<2x5xi8>,tensor<2x5xi8>,tensor<5x5xi8>,tensor<5x3xi8>,tensor<2x5xi32>
  }
}
