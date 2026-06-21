// K_move_16x128 (wide movement)
module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %X = merlin_iface.tensor {name = "X", role = "input"} : tensor<16x128xi8>
  %Y0 = merlin_iface.movement %X {name = "Y0"} : (tensor<16x128xi8>) -> tensor<16x128xi8>
}
