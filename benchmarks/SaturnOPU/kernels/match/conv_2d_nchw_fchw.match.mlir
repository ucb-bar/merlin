// Matches the named `linalg.conv_2d_nchw_fchw` op as it appears in dronet's
// phase-3 preprocessing IR. Dronet fuses the bias broadcast directly into
// the conv's `outs` operand (so the conv accumulates onto a pre-broadcast
// bias tensor), but our wrapper signature treats `outs` simply as the
// destination tensor — the kernel reads the bias-prefilled value from
// `binding2[...]` and accumulates on top.
//
// Stride/dilation are baked into the kernel (stride=2, dilation=1) — see
// the C source for follow-on notes about parameterizing them.

^bb0(%in: tensor<?x?x?x?xf32>, %weight: tensor<?x?x?x?xf32>, %bias_or_init: tensor<?x?x?x?xf32>):
  %conv = linalg.conv_2d_nchw_fchw
      {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
      ins(%in, %weight : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
      outs(%bias_or_init : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
