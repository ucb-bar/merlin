module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %IFM = merlin_iface.tensor {name = "IFM", role = "input"} : tensor<1x14x14x1024xi8>
  %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<1024x2048xi8>
  %B = merlin_iface.tensor {name = "B", role = "bias"} : tensor<2048xi32>
  %W_res = merlin_iface.resident_pack %W {layout = "packed_conv_rhs"} : (tensor<1024x2048xi8>) -> !merlin_iface.resident
  %Y0 = merlin_iface.conv2d %IFM, %W_res {kernel = [1, 1, 1024, 2048], stride = [2, 2], padding = [0, 0, 0, 0], dilation = [1, 1], name = "Y0", epilogue = ["bias_add"], output_dtype = "i32", bias = "B", layout = "nhwc"} : (tensor<1x14x14x1024xi8>, !merlin_iface.resident) -> tensor<49x2048xi32>
  merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()
}
