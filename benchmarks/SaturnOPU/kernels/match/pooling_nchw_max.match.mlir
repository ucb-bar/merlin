// Matches `linalg.pooling_nchw_max` with stride=2, dilation=1 (dronet's
// only max-pool layer right after the first conv). The `outs` operand is
// a `linalg.fill`-initialized tensor (filled with the -inf identity for
// max), so by replacing the pool we also subsume that init dispatch.

^bb0(%in: tensor<?x?x?x?xf32>, %window: tensor<?x?xf32>, %init: tensor<?x?x?x?xf32>):
  %p = linalg.pooling_nchw_max
      {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
      ins(%in, %window : tensor<?x?x?x?xf32>, tensor<?x?xf32>)
      outs(%init : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
