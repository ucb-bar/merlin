# Buddy Gemmini lowering coverage (Sparsh)

This table is a quick view of what we’ve stress-tested and what Buddy lowers into.

| Test | Input dialect/op | Layout | Proof of Gemmini match | Proof of Gemmini command expansion | Notes |
|---|---|---|---|---|---|
| matmul | linalg.matmul | (varies) | `gemmini.tile_matmul` | `gemmini.intr.loop_ws_config*` + `gemmini.intr.loop_ws` | matmul lowered end-to-end |
| batch_matmul | linalg.batch_matmul | (varies) | `gemmini.tile_*` | `gemmini.intr.*` | batched path works |
| conv (NHWC/HWCF) | linalg.conv_2d_nhwc_hwcf | NHWC x HWCF | `gemmini.tile_conv` | `gemmini.intr.loop_conv_ws_config*` + `gemmini.intr.loop_conv_ws` | conv lowered to WS loop |
| conv (NCHW/FCHW) | linalg.conv_2d_nchw_fchw | NCHW x FCHW | `gemmini.tile_conv` | `gemmini.intr.loop_conv_ws_config*` + `gemmini.intr.loop_conv_ws` | alternate layout works |
| mini CNN block | 2x conv + copy | NCHW/FCHW | 2x `gemmini.tile_conv` | `gemmini.intr.loop_conv_ws*` appears | multi-layer block lowers |
