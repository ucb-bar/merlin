module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %IFM = merlin_iface.tensor {name = "IFM", role = "input"} : tensor<1x5x7x3xi8>
  %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<27x16xi8>
  %W_res = merlin_iface.resident_pack %W {layout = "packed_conv_rhs"} : (tensor<27x16xi8>) -> !merlin_iface.resident
  %Y0 = merlin_iface.conv2d %IFM, %W_res {kernel = [3, 3, 3, 16], stride = [2, 2], padding = [1, 1, 1, 1], dilation = [1, 1], name = "Y0", epilogue = [], output_dtype = "i8", layout = "nhwc"} : (tensor<1x5x7x3xi8>, !merlin_iface.resident) -> tensor<12x16xi8>
  merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()
}
