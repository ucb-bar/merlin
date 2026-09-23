module attributes {merlin_iface.version = "0.1", merlin_iface.target = "radiance", merlin_iface.abi_version = "0.1"} {
  %X = merlin_iface.tensor {name = "X", role = "input"} : tensor<2x4xbf16>
  %Y0 = merlin_iface.movement %X {name = "Y0", semantic = "mvin_mvout", output_dtype = "f32"} : (tensor<2x4xbf16>) -> tensor<2x4xf32>
}
