# B3_conv2d_im2col_i8

Conv2d (1x8x8x4 NHWC, 3x3, Ci4->Co8) compiler-lowered to im2col + weight-stationary matmul. epilogue=[]. Uplifted from bareMetalC/conv.c (input to the compiler, not a copied kernel).
