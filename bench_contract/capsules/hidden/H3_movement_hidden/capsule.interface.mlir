module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %Xh = merlin_iface.tensor {name = "Xh", role = "input"} : tensor<16x16xi8>
  %Y0 = merlin_iface.movement %Xh {name = "Y0"} : (tensor<16x16xi8>) -> tensor<16x16xi8>
}
