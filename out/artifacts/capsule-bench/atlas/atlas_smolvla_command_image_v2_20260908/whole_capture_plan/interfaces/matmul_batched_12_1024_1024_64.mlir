module attributes {merlin_iface.version = "0.1", merlin_iface.target = "atlas", merlin_iface.abi_version = "0.1"} {
  %A0 = merlin_iface.tensor {name = "A0", role = "input"} : tensor<12x1024x1024xf8E4M3FN>
  %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<12x1024x64xf8E4M3FN>
  %Y0 = merlin_iface.matmul_batched %A0, %W {name = "Y0", batch = 12 : i64, output_dtype = "bf16"} : (tensor<12x1024x1024xf8E4M3FN>, tensor<12x1024x64xf8E4M3FN>) -> tensor<12x1024x64xbf16>
}
