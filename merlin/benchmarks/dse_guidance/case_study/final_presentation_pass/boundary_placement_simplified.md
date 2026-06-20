# Boundary-placement (simplified, categorical)

Where each key abstraction *could* be implemented across the HW/SW stack — a **search space**, not a score. Cells: strong / possible / weak / blocked / unavail / — (n/a).

| abstraction | compiler transform | runtime / HAL object | command ISA | accelerator ISA | microcode / controller | fixed datapath |
|---|---|---|---|---|---|---|
| resident_weight_object | possible | strong | strong | weak | possible | — |
| skinny_gemm_or_gemv_engine | possible | — | possible | strong | possible | strong |
| partial_sum_object | — | weak | possible | strong | strong | strong |
| fused_requant_epilogue | possible | — | possible | possible | possible | possible |
| loop_carried_state_handle | — | possible | strong | possible | strong | possible |
| bounded_loop_command | — | possible | strong | possible | strong | possible |
| packed_lowbit_tensor | possible | blocked | blocked | blocked | blocked | blocked |
| scale_object | possible | blocked | blocked | blocked | blocked | blocked |
| dma_engine | weak | strong | strong | possible | possible | possible |
| async_queue | — | strong | strong | weak | possible | — |
