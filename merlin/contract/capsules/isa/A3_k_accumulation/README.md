# A3_k_accumulation

A3: K=32 (> tile dim) forces K-accumulation (accumulate-onto PRELOAD across K tiles).

kind=isa label=public op=matmul modes={'k_accumulate': True}
