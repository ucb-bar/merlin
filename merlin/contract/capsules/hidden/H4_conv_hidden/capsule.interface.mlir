module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %IFMh = merlin_iface.tensor {name = "IFMh", role = "input"} : tensor<1x8x8x4xi8>
  %Wc = merlin_iface.tensor {name = "Wc", role = "weight"} : tensor<36x8xi8>
  %Wc_res = merlin_iface.resident_pack %Wc {layout = "packed_conv_rhs"} : (tensor<36x8xi8>) -> !merlin_iface.resident
  %Y0 = merlin_iface.conv2d %IFMh, %Wc_res {kernel = [3, 3, 4, 8], stride = [1, 1], padding = [0, 0, 0, 0], dilation = [1, 1], name = "Y0", epilogue = [], output_dtype = "i32", layout = "nhwc"} : (tensor<1x8x8x4xi8>, !merlin_iface.resident) -> tensor<36x8xi32>
  merlin_iface.evict %Wc_res : (!merlin_iface.resident) -> ()
}
