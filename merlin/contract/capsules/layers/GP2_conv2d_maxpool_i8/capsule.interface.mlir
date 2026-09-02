// GP2: 3x3 int8 conv over 8x8x4 -> 6x6x16, with a 2x2/2 max-pool fused onto the store.
module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %IFM = merlin_iface.tensor {name = "IFM", role = "input"} : tensor<1x8x8x4xi8>
  %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<36x16xi8>
  %W_res = merlin_iface.resident_pack %W {layout = "packed_conv_rhs"} : (tensor<36x16xi8>) -> !merlin_iface.resident
  %Y0 = merlin_iface.conv2d %IFM, %W_res {kernel = [3, 3, 4, 16], stride = [1, 1], padding = [0, 0, 0, 0], dilation = [1, 1], name = "Y0", epilogue = ["maxpool"], output_dtype = "i8", layout = "nhwc", pool_in_dims = [6, 6], pool_size = [2, 2], pool_stride = [2, 2], pool_padding = [0, 0, 0, 0]} : (tensor<1x8x8x4xi8>, !merlin_iface.resident) -> tensor<9x16xi8>
  merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()
}
