"""Expert-kernel ceiling drivers: measure the performance bar our RVV codegen is ranked against.

Bare-metal C drivers (XNNPACK / OpenBLAS / ours GEMM, bmm, int8, activation, dwconv) plus the
Python harnesses that build and run them on spike / K1 to measure the *expert* GEMM ceiling —
the cross-framework, bit-exact-verified cycle counts the compiler's generated kernels are compared
to (see ``run_expert_gemm`` for the single-point S4.2 bar and ``multishape_compare`` for the matrix).
"""
