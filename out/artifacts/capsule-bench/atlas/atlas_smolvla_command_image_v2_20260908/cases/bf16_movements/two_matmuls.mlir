module attributes {merlin_iface.version = "0.1", merlin_iface.target = "atlas", merlin_iface.abi_version = "0.1"} {
  %X0 = merlin_iface.tensor {name = "X0", role = "input"} : tensor<4x8xbf16>
  %X1 = merlin_iface.tensor {name = "X1", role = "input"} : tensor<3x7xbf16>
  %Y0 = merlin_iface.movement %X0 {name = "Y0"} : (tensor<4x8xbf16>) -> tensor<4x8xbf16>
  %Y1 = merlin_iface.movement %X1 {name = "Y1"} : (tensor<3x7xbf16>) -> tensor<3x7xbf16>
}
