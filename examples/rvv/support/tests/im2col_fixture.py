"""Closed signed-integer capture grammar; adapted from Merlin im2col panel tests."""


def _module(*, c=2, kh=3, kw=3, n=1, oh=4, ow=32, f=8, sh=1, sw=1, dh=1, dw=1):
    hin, win = (oh - 1) * sh + (kh - 1) * dh + 1, (ow - 1) * sw + (kw - 1) * dw + 1
    k, m = c * kh * kw, n * oh * ow
    return f"""
module {{
  func.func @forward(%in: tensor<{n}x{c}x{hin}x{win}xi8>, %w: tensor<{f}x{k}xi8>)
      -> tensor<{f}x{m}xi32> {{
    %c0 = arith.constant 0 : i32
    %e = tensor.empty() : tensor<{c}x{kh}x{kw}x{n}x{oh}x{ow}xi8>
    %g = linalg.generic {{
        indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5)
                           -> (d3, d0, d4 * {sh} + d1 * {dh}, d5 * {sw} + d2 * {dw})>,
                         affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>],
        iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]}}
      ins(%in : tensor<{n}x{c}x{hin}x{win}xi8>)
      outs(%e : tensor<{c}x{kh}x{kw}x{n}x{oh}x{ow}xi8>) {{
    ^bb0(%a: i8, %o: i8):
      linalg.yield %a : i8
    }} -> tensor<{c}x{kh}x{kw}x{n}x{oh}x{ow}xi8>
    %flat = tensor.collapse_shape %g [[0, 1, 2, 3, 4, 5]]
      : tensor<{c}x{kh}x{kw}x{n}x{oh}x{ow}xi8> into tensor<{k * m}xi8>
    %col = tensor.expand_shape %flat [[0, 1]] output_shape [{k}, {m}]
      : tensor<{k * m}xi8> into tensor<{k}x{m}xi8>
    %ae = tensor.empty() : tensor<{f}x{m}xi32>
    %acc = linalg.fill ins(%c0 : i32) outs(%ae : tensor<{f}x{m}xi32>) -> tensor<{f}x{m}xi32>
    %mm = linalg.generic {{
        indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                         affine_map<(d0, d1, d2) -> (d2, d1)>,
                         affine_map<(d0, d1, d2) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel", "reduction"]}}
      ins(%w, %col : tensor<{f}x{k}xi8>, tensor<{k}x{m}xi8>)
      outs(%acc : tensor<{f}x{m}xi32>) {{
    ^bb0(%x: i8, %y: i8, %o: i32):
      %xe = arith.extsi %x : i8 to i32
      %ye = arith.extsi %y : i8 to i32
      %p = arith.muli %xe, %ye : i32
      %s = arith.addi %p, %o : i32
      linalg.yield %s : i32
    }} -> tensor<{f}x{m}xi32>
    func.return %mm : tensor<{f}x{m}xi32>
  }}
}}
"""
