module attributes {merlin_iface.version = "0.1", merlin_iface.target = "atlas", merlin_iface.abi_version = "0.1"} {
  %X = merlin_iface.tensor {name = "X", role = "input"} : tensor<4x8xf8E4M3FN>
  %Y0 = merlin_iface.movement %X {name = "Y0", semantic = "mvin_mvout", output_dtype = "bf16"} : (tensor<4x8xf8E4M3FN>) -> tensor<4x8xbf16>
}
