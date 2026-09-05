module attributes {merlin_iface.version = "0.1", merlin_iface.target = "atlas", merlin_iface.abi_version = "0.1"} {
  %IFM = merlin_iface.tensor {name = "IFM", role = "input"} : tensor<1x8x8x128xf8E4M3FN>
  %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<1152x32xf8E4M3FN>
  %W_res = merlin_iface.resident_pack %W {layout = "packed_conv_rhs"} : (tensor<1152x32xf8E4M3FN>) -> !merlin_iface.resident
  %Y0 = merlin_iface.conv2d %IFM, %W_res {kernel = [3, 3, 128, 32], stride = [1, 1], padding = [0, 0, 0, 0], dilation = [1, 1], name = "Y0", epilogue = [], output_dtype = "bf16", layout = "nhwc"} : (tensor<1x8x8x128xf8E4M3FN>, !merlin_iface.resident) -> tensor<36x32xbf16>
  merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()
}
