# B4_conv2d_relu_i8

Conv2d (1x8x8x4 NHWC, 3x3, Ci4->Co8) compiler-lowered to im2col + weight-stationary matmul. epilogue=['relu']. Uplifted from bareMetalC/conv_with_pool.c (input to the compiler, not a copied kernel).
