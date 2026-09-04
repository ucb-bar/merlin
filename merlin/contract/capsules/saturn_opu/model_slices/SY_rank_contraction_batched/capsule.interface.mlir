module attributes {merlin_iface.version = "0.1", merlin_iface.target = "saturn_opu_mxv256d128", merlin_iface.abi_version = "0.1"} {
  %A0 = merlin_iface.tensor {name = "A0", role = "input"} : tensor<2x16x32xi8>
  %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<2x32x16xi8>
  %Y0 = merlin_iface.matmul_batched %A0, %W {name = "Y0", batch = 2 : i64, output_dtype = "i32"} : (tensor<2x16x32xi8>, tensor<2x32x16xi8>) -> tensor<32x16xi32>
}
