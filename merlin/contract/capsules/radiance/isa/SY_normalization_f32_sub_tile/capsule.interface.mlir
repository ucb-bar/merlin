module attributes {merlin_iface.version = "0.1", merlin_iface.target = "radiance", merlin_iface.abi_version = "0.1"} {
  %X = merlin_iface.tensor {name = "X", role = "input"} : tensor<2x4xf32>
  %G = merlin_iface.tensor {name = "G", role = "weight"} : tensor<1x4xf32>
  %Y0 = merlin_iface.rmsnorm %X, %G {name = "Y0", eps = 1.525878906e-05 : f64, output_dtype = "f32"} : (tensor<2x4xf32>, tensor<1x4xf32>) -> tensor<2x4xf32>
}
