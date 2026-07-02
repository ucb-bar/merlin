// K_move_64x64 (multi-tile movement)
module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %X = merlin_iface.tensor {name = "X", role = "input"} : tensor<64x64xi8>
  %Y0 = merlin_iface.movement %X {name = "Y0"} : (tensor<64x64xi8>) -> tensor<64x64xi8>
}
