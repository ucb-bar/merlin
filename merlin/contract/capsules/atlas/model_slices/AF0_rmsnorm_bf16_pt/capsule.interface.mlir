module attributes {merlin_iface.version = "0.1", merlin_iface.target = "atlas", merlin_iface.abi_version = "0.1"} {
  %X = merlin_iface.tensor {name = "X", role = "input"} : tensor<16x16xbf16>
  %G = merlin_iface.tensor {name = "G", role = "weight"} : tensor<1x16xbf16>
  %Y0 = merlin_iface.rmsnorm %X, %G {name = "Y0", eps = 1.000000000e-05 : f64, output_dtype = "f32"} : (tensor<16x16xbf16>, tensor<1x16xbf16>) -> tensor<16x16xf32>
}
