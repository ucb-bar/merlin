module attributes {merlin_iface.version = "0.1", merlin_iface.target = "atlas", merlin_iface.abi_version = "0.1"} {
  %A0 = merlin_iface.tensor {name = "A0", role = "input"} : tensor<15x50x64xf8E4M3FN>
  %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<15x64x163xf8E4M3FN>
  %Y0 = merlin_iface.matmul_batched %A0, %W {name = "Y0", batch = 15 : i64, output_dtype = "bf16"} : (tensor<15x50x64xf8E4M3FN>, tensor<15x64x163xf8E4M3FN>) -> tensor<15x50x163xbf16>
}
