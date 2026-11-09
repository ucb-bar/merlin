# M3_host_island_seam_gemmini

M3: the A->H->A seam as the subject of a capsule. An int8 GEMM on the mesh, a LayerNorm on the host lane (this target declares no normalization capability, so the island is host work because the hardware cannot do it), and the value back to the mesh for a second int8 GEMM. The four whole models all CONTAIN this shape and all classify as `routing`, so the seam was covered only incidentally; one region more on either side of this one would retitle it `routing` too.

kind=model label=public op=model modes={}
