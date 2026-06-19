# Predicate audit (all)

> Every necessity predicate, its numeric inputs, thresholds, and whether it rests on configured/reference K or on capture-erased structure. The per-workload K / weight-bytes / MAC-fraction scalars live HERE — the necessity rollup carries only the class-invariant rule, so it never presents one workload's K as a corpus constant.

**50 suspicious cells** (necessity resting on configured K, or a non-discriminating predicate).

| abstraction | workload | class | inputs | thresholds | uses_K | erased | neg_control | discriminating | suspicious |
|---|---|---|---|---|---|---|---|---|---|
| resident_weight_object | bitvla | necessary | K=7, weight_bytes=5767168, avoidable_reload=34603008 | K>1 & weight_bytes>1e6 & avoidable_reload>weight_bytes | yes | yes | no | no | necessity rests on configured/reference K (not IR-recovered); class never varies across corpus (not discriminating) |
| resident_weight_object | groot_n1d7 | necessary | K=4, weight_bytes=2200436736, avoidable_reload=6601310208 | K>1 & weight_bytes>1e6 & avoidable_reload>weight_bytes | yes | yes | no | no | necessity rests on configured/reference K (not IR-recovered); class never varies across corpus (not discriminating) |
| resident_weight_object | molmoact | necessary | K=8, weight_bytes=3787456512, avoidable_reload=26512195584 | K>1 & weight_bytes>1e6 & avoidable_reload>weight_bytes | yes | yes | no | no | necessity rests on configured/reference K (not IR-recovered); class never varies across corpus (not discriminating) |
| resident_weight_object | openvla | necessary | K=7, weight_bytes=3145728, avoidable_reload=18874368 | K>1 & weight_bytes>1e6 & avoidable_reload>weight_bytes | yes | yes | no | no | necessity rests on configured/reference K (not IR-recovered); class never varies across corpus (not discriminating) |
| resident_weight_object | pi05 | necessary | K=10, weight_bytes=1719926784, avoidable_reload=15479341056 | K>1 & weight_bytes>1e6 & avoidable_reload>weight_bytes | yes | yes | no | no | necessity rests on configured/reference K (not IR-recovered); class never varies across corpus (not discriminating) |
| resident_weight_object | rdt | necessary | K=5, weight_bytes=393216000, avoidable_reload=1572864000 | K>1 & weight_bytes>1e6 & avoidable_reload>weight_bytes | yes | yes | no | no | necessity rests on configured/reference K (not IR-recovered); class never varies across corpus (not discriminating) |
| resident_weight_object | rdt2 | necessary | K=5, weight_bytes=309739520, avoidable_reload=1238958080 | K>1 & weight_bytes>1e6 & avoidable_reload>weight_bytes | yes | yes | no | no | necessity rests on configured/reference K (not IR-recovered); class never varies across corpus (not discriminating) |
| resident_weight_object | smolvla | necessary | K=10, weight_bytes=206126080, avoidable_reload=1855134720 | K>1 & weight_bytes>1e6 & avoidable_reload>weight_bytes | yes | yes | no | no | necessity rests on configured/reference K (not IR-recovered); class never varies across corpus (not discriminating) |
| resident_weight_object | tiny_llama | necessary | K=7, weight_bytes=614465536, avoidable_reload=3686793216 | K>1 & weight_bytes>1e6 & avoidable_reload>weight_bytes | yes | yes | no | no | necessity rests on configured/reference K (not IR-recovered); class never varies across corpus (not discriminating) |
| resident_weight_object | xr0 | necessary | K=5, weight_bytes=169086976, avoidable_reload=676347904 | K>1 & weight_bytes>1e6 & avoidable_reload>weight_bytes | yes | yes | no | no | necessity rests on configured/reference K (not IR-recovered); class never varies across corpus (not discriminating) |
| resident_packed_weight_object | bitvla | blocked | low-bit/packed weights+scales (dequantized in the f32 capture) | n/a | no | yes | no | no | — |
| resident_packed_weight_object | groot_n1d7 | blocked | low-bit/packed weights+scales (dequantized in the f32 capture) | n/a | no | yes | no | no | — |
| resident_packed_weight_object | molmoact | blocked | low-bit/packed weights+scales (dequantized in the f32 capture) | n/a | no | yes | no | no | — |
| resident_packed_weight_object | openvla | blocked | low-bit/packed weights+scales (dequantized in the f32 capture) | n/a | no | yes | no | no | — |
| resident_packed_weight_object | pi05 | blocked | low-bit/packed weights+scales (dequantized in the f32 capture) | n/a | no | yes | no | no | — |
| resident_packed_weight_object | rdt | blocked | low-bit/packed weights+scales (dequantized in the f32 capture) | n/a | no | yes | no | no | — |
| resident_packed_weight_object | rdt2 | blocked | low-bit/packed weights+scales (dequantized in the f32 capture) | n/a | no | yes | no | no | — |
| resident_packed_weight_object | smolvla | blocked | low-bit/packed weights+scales (dequantized in the f32 capture) | n/a | no | yes | no | no | — |
| resident_packed_weight_object | tiny_llama | blocked | low-bit/packed weights+scales (dequantized in the f32 capture) | n/a | no | yes | no | no | — |
| resident_packed_weight_object | xr0 | blocked | low-bit/packed weights+scales (dequantized in the f32 capture) | n/a | no | yes | no | no | — |
| packed_lowbit_tensor | bitvla | blocked | low-bit/packed weights+scales (dequantized in the f32 capture) | n/a | no | yes | no | no | — |
| packed_lowbit_tensor | groot_n1d7 | blocked | low-bit/packed weights+scales (dequantized in the f32 capture) | n/a | no | yes | no | no | — |
| packed_lowbit_tensor | molmoact | blocked | low-bit/packed weights+scales (dequantized in the f32 capture) | n/a | no | yes | no | no | — |
| packed_lowbit_tensor | openvla | blocked | low-bit/packed weights+scales (dequantized in the f32 capture) | n/a | no | yes | no | no | — |
| packed_lowbit_tensor | pi05 | blocked | low-bit/packed weights+scales (dequantized in the f32 capture) | n/a | no | yes | no | no | — |
| packed_lowbit_tensor | rdt | blocked | low-bit/packed weights+scales (dequantized in the f32 capture) | n/a | no | yes | no | no | — |
| packed_lowbit_tensor | rdt2 | blocked | low-bit/packed weights+scales (dequantized in the f32 capture) | n/a | no | yes | no | no | — |
| packed_lowbit_tensor | smolvla | blocked | low-bit/packed weights+scales (dequantized in the f32 capture) | n/a | no | yes | no | no | — |
| packed_lowbit_tensor | tiny_llama | blocked | low-bit/packed weights+scales (dequantized in the f32 capture) | n/a | no | yes | no | no | — |
| packed_lowbit_tensor | xr0 | blocked | low-bit/packed weights+scales (dequantized in the f32 capture) | n/a | no | yes | no | no | — |
| scale_object | bitvla | blocked | low-bit/packed weights+scales (dequantized in the f32 capture) | n/a | no | yes | no | no | — |
| scale_object | groot_n1d7 | blocked | low-bit/packed weights+scales (dequantized in the f32 capture) | n/a | no | yes | no | no | — |
| scale_object | molmoact | blocked | low-bit/packed weights+scales (dequantized in the f32 capture) | n/a | no | yes | no | no | — |
| scale_object | openvla | blocked | low-bit/packed weights+scales (dequantized in the f32 capture) | n/a | no | yes | no | no | — |
| scale_object | pi05 | blocked | low-bit/packed weights+scales (dequantized in the f32 capture) | n/a | no | yes | no | no | — |
| scale_object | rdt | blocked | low-bit/packed weights+scales (dequantized in the f32 capture) | n/a | no | yes | no | no | — |
| scale_object | rdt2 | blocked | low-bit/packed weights+scales (dequantized in the f32 capture) | n/a | no | yes | no | no | — |
| scale_object | smolvla | blocked | low-bit/packed weights+scales (dequantized in the f32 capture) | n/a | no | yes | no | no | — |
| scale_object | tiny_llama | blocked | low-bit/packed weights+scales (dequantized in the f32 capture) | n/a | no | yes | no | no | — |
| scale_object | xr0 | blocked | low-bit/packed weights+scales (dequantized in the f32 capture) | n/a | no | yes | no | no | — |
| prefix_kv_object | bitvla | blocked | attention/KV state (lowered in the flat capture) | n/a | no | yes | no | yes | — |
| prefix_kv_object | groot_n1d7 | not_applicable | attention/KV state (lowered in the flat capture) | n/a | no | yes | no | yes | — |
| prefix_kv_object | molmoact | blocked | attention/KV state (lowered in the flat capture) | n/a | no | yes | no | yes | — |
| prefix_kv_object | openvla | blocked | attention/KV state (lowered in the flat capture) | n/a | no | yes | no | yes | — |
| prefix_kv_object | pi05 | not_applicable | attention/KV state (lowered in the flat capture) | n/a | no | yes | no | yes | — |
| prefix_kv_object | rdt | not_applicable | attention/KV state (lowered in the flat capture) | n/a | no | yes | no | yes | — |
| prefix_kv_object | rdt2 | not_applicable | attention/KV state (lowered in the flat capture) | n/a | no | yes | no | yes | — |
| prefix_kv_object | smolvla | not_applicable | attention/KV state (lowered in the flat capture) | n/a | no | yes | no | yes | — |
| prefix_kv_object | tiny_llama | blocked | attention/KV state (lowered in the flat capture) | n/a | no | yes | no | yes | — |
| prefix_kv_object | xr0 | not_applicable | attention/KV state (lowered in the flat capture) | n/a | no | yes | no | yes | — |
| loop_carried_state_handle | bitvla | useful | K=7 | K>1 -> useful | yes | yes | no | no | class never varies across corpus (not discriminating) |
| loop_carried_state_handle | groot_n1d7 | useful | K=4 | K>1 -> useful | yes | yes | no | no | class never varies across corpus (not discriminating) |
| loop_carried_state_handle | molmoact | useful | K=8 | K>1 -> useful | yes | yes | no | no | class never varies across corpus (not discriminating) |
| loop_carried_state_handle | openvla | useful | K=7 | K>1 -> useful | yes | yes | no | no | class never varies across corpus (not discriminating) |
| loop_carried_state_handle | pi05 | useful | K=10 | K>1 -> useful | yes | yes | no | no | class never varies across corpus (not discriminating) |
| loop_carried_state_handle | rdt | useful | K=5 | K>1 -> useful | yes | yes | no | no | class never varies across corpus (not discriminating) |
| loop_carried_state_handle | rdt2 | useful | K=5 | K>1 -> useful | yes | yes | no | no | class never varies across corpus (not discriminating) |
| loop_carried_state_handle | smolvla | useful | K=10 | K>1 -> useful | yes | yes | no | no | class never varies across corpus (not discriminating) |
| loop_carried_state_handle | tiny_llama | useful | K=7 | K>1 -> useful | yes | yes | no | no | class never varies across corpus (not discriminating) |
| loop_carried_state_handle | xr0 | useful | K=5 | K>1 -> useful | yes | yes | no | no | class never varies across corpus (not discriminating) |
| bounded_loop_command | bitvla | useful | K=7 | K>1 -> useful | yes | yes | no | no | class never varies across corpus (not discriminating) |
| bounded_loop_command | groot_n1d7 | useful | K=4 | K>1 -> useful | yes | yes | no | no | class never varies across corpus (not discriminating) |
| bounded_loop_command | molmoact | useful | K=8 | K>1 -> useful | yes | yes | no | no | class never varies across corpus (not discriminating) |
| bounded_loop_command | openvla | useful | K=7 | K>1 -> useful | yes | yes | no | no | class never varies across corpus (not discriminating) |
| bounded_loop_command | pi05 | useful | K=10 | K>1 -> useful | yes | yes | no | no | class never varies across corpus (not discriminating) |
| bounded_loop_command | rdt | useful | K=5 | K>1 -> useful | yes | yes | no | no | class never varies across corpus (not discriminating) |
| bounded_loop_command | rdt2 | useful | K=5 | K>1 -> useful | yes | yes | no | no | class never varies across corpus (not discriminating) |
| bounded_loop_command | smolvla | useful | K=10 | K>1 -> useful | yes | yes | no | no | class never varies across corpus (not discriminating) |
| bounded_loop_command | tiny_llama | useful | K=7 | K>1 -> useful | yes | yes | no | no | class never varies across corpus (not discriminating) |
| bounded_loop_command | xr0 | useful | K=5 | K>1 -> useful | yes | yes | no | no | class never varies across corpus (not discriminating) |
| region_level_dispatch | bitvla | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| region_level_dispatch | groot_n1d7 | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| region_level_dispatch | molmoact | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| region_level_dispatch | openvla | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| region_level_dispatch | pi05 | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| region_level_dispatch | rdt | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| region_level_dispatch | rdt2 | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| region_level_dispatch | smolvla | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| region_level_dispatch | tiny_llama | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| region_level_dispatch | xr0 | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| persistent_command_buffer | bitvla | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| persistent_command_buffer | groot_n1d7 | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| persistent_command_buffer | molmoact | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| persistent_command_buffer | openvla | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| persistent_command_buffer | pi05 | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| persistent_command_buffer | rdt | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| persistent_command_buffer | rdt2 | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| persistent_command_buffer | smolvla | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| persistent_command_buffer | tiny_llama | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| persistent_command_buffer | xr0 | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| event_token | bitvla | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| event_token | groot_n1d7 | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| event_token | molmoact | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| event_token | openvla | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| event_token | pi05 | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| event_token | rdt | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| event_token | rdt2 | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| event_token | smolvla | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| event_token | tiny_llama | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| event_token | xr0 | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| async_queue | bitvla | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| async_queue | groot_n1d7 | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| async_queue | molmoact | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| async_queue | openvla | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| async_queue | pi05 | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| async_queue | rdt | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| async_queue | rdt2 | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| async_queue | smolvla | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| async_queue | tiny_llama | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| async_queue | xr0 | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| producer_consumer_queue | bitvla | possible | not gated by a discriminating signal | n/a | no | no | no | yes | — |
| producer_consumer_queue | groot_n1d7 | possible | not gated by a discriminating signal | n/a | no | no | no | yes | — |
| producer_consumer_queue | molmoact | possible | not gated by a discriminating signal | n/a | no | no | no | yes | — |
| producer_consumer_queue | openvla | possible | not gated by a discriminating signal | n/a | no | no | no | yes | — |
| producer_consumer_queue | pi05 | possible | not gated by a discriminating signal | n/a | no | no | no | yes | — |
| producer_consumer_queue | rdt | possible | not gated by a discriminating signal | n/a | no | no | no | yes | — |
| producer_consumer_queue | rdt2 | possible | not gated by a discriminating signal | n/a | no | no | no | yes | — |
| producer_consumer_queue | smolvla | possible | not gated by a discriminating signal | n/a | no | no | no | yes | — |
| producer_consumer_queue | tiny_llama | not_applicable | not gated by a discriminating signal | n/a | no | no | no | yes | — |
| producer_consumer_queue | xr0 | not_applicable | not gated by a discriminating signal | n/a | no | no | no | yes | — |
| double_buffered_action_chunk | bitvla | possible | not gated by a discriminating signal | n/a | no | no | no | yes | — |
| double_buffered_action_chunk | groot_n1d7 | possible | not gated by a discriminating signal | n/a | no | no | no | yes | — |
| double_buffered_action_chunk | molmoact | possible | not gated by a discriminating signal | n/a | no | no | no | yes | — |
| double_buffered_action_chunk | openvla | possible | not gated by a discriminating signal | n/a | no | no | no | yes | — |
| double_buffered_action_chunk | pi05 | possible | not gated by a discriminating signal | n/a | no | no | no | yes | — |
| double_buffered_action_chunk | rdt | possible | not gated by a discriminating signal | n/a | no | no | no | yes | — |
| double_buffered_action_chunk | rdt2 | possible | not gated by a discriminating signal | n/a | no | no | no | yes | — |
| double_buffered_action_chunk | smolvla | possible | not gated by a discriminating signal | n/a | no | no | no | yes | — |
| double_buffered_action_chunk | tiny_llama | not_applicable | not gated by a discriminating signal | n/a | no | no | no | yes | — |
| double_buffered_action_chunk | xr0 | not_applicable | not gated by a discriminating signal | n/a | no | no | no | yes | — |
| matrix_engine | bitvla | possible | dense_mac=0 | >0.5 necessary; >0.1 useful | no | no | yes | yes | — |
| matrix_engine | groot_n1d7 | possible | dense_mac=0 | >0.5 necessary; >0.1 useful | no | no | yes | yes | — |
| matrix_engine | molmoact | possible | dense_mac=0 | >0.5 necessary; >0.1 useful | no | no | yes | yes | — |
| matrix_engine | openvla | possible | dense_mac=0 | >0.5 necessary; >0.1 useful | no | no | yes | yes | — |
| matrix_engine | pi05 | possible | dense_mac=0.0 | >0.5 necessary; >0.1 useful | no | no | yes | yes | — |
| matrix_engine | rdt | necessary | dense_mac=0.8711 | >0.5 necessary; >0.1 useful | no | no | yes | yes | — |
| matrix_engine | rdt2 | possible | dense_mac=0 | >0.5 necessary; >0.1 useful | no | no | yes | yes | — |
| matrix_engine | smolvla | necessary | dense_mac=0.5243 | >0.5 necessary; >0.1 useful | no | no | yes | yes | — |
| matrix_engine | tiny_llama | possible | dense_mac=0 | >0.5 necessary; >0.1 useful | no | no | yes | yes | — |
| matrix_engine | xr0 | possible | dense_mac=0 | >0.5 necessary; >0.1 useful | no | no | yes | yes | — |
| skinny_gemm_or_gemv_engine | bitvla | necessary | gemv_mac=1.0 (true_gemv=0.0432, skinny_gemm=0.9568) | >0.5 necessary; >0.1 useful | no | no | no | yes | — |
| skinny_gemm_or_gemv_engine | groot_n1d7 | necessary | gemv_mac=1.0 (true_gemv=0.0041, skinny_gemm=0.9959) | >0.5 necessary; >0.1 useful | no | no | no | yes | — |
| skinny_gemm_or_gemv_engine | molmoact | necessary | gemv_mac=1.0 (true_gemv=0.1142, skinny_gemm=0.8858) | >0.5 necessary; >0.1 useful | no | no | no | yes | — |
| skinny_gemm_or_gemv_engine | openvla | necessary | gemv_mac=1.0 (true_gemv=0.0558, skinny_gemm=0.9442) | >0.5 necessary; >0.1 useful | no | no | no | yes | — |
| skinny_gemm_or_gemv_engine | pi05 | necessary | gemv_mac=0.6699 (true_gemv=0.0001, skinny_gemm=0.6698) | >0.5 necessary; >0.1 useful | no | no | no | yes | — |
| skinny_gemm_or_gemv_engine | rdt | useful | gemv_mac=0.1289 (true_gemv=0.0002, skinny_gemm=0.1287) | >0.5 necessary; >0.1 useful | no | no | no | yes | — |
| skinny_gemm_or_gemv_engine | rdt2 | necessary | gemv_mac=0.9977 (true_gemv=0.0436, skinny_gemm=0.9541) | >0.5 necessary; >0.1 useful | no | no | no | yes | — |
| skinny_gemm_or_gemv_engine | smolvla | useful | gemv_mac=0.2019 (true_gemv=0.0, skinny_gemm=0.2019) | >0.5 necessary; >0.1 useful | no | no | no | yes | — |
| skinny_gemm_or_gemv_engine | tiny_llama | necessary | gemv_mac=1.0 (true_gemv=0.2372, skinny_gemm=0.7628) | >0.5 necessary; >0.1 useful | no | no | no | yes | — |
| skinny_gemm_or_gemv_engine | xr0 | necessary | gemv_mac=0.9991 (true_gemv=0.0078, skinny_gemm=0.9913) | >0.5 necessary; >0.1 useful | no | no | no | yes | — |
| epilogue_requant_unit | bitvla | useful | epi_scale_act=False, epi_bias=True | scale+act -> necessary; bias -> useful | no | no | yes | yes | — |
| epilogue_requant_unit | groot_n1d7 | necessary | epi_scale_act=True, epi_bias=True | scale+act -> necessary; bias -> useful | no | no | yes | yes | — |
| epilogue_requant_unit | molmoact | useful | epi_scale_act=False, epi_bias=True | scale+act -> necessary; bias -> useful | no | no | yes | yes | — |
| epilogue_requant_unit | openvla | necessary | epi_scale_act=True, epi_bias=True | scale+act -> necessary; bias -> useful | no | no | yes | yes | — |
| epilogue_requant_unit | pi05 | necessary | epi_scale_act=True, epi_bias=True | scale+act -> necessary; bias -> useful | no | no | yes | yes | — |
| epilogue_requant_unit | rdt | necessary | epi_scale_act=True, epi_bias=True | scale+act -> necessary; bias -> useful | no | no | yes | yes | — |
| epilogue_requant_unit | rdt2 | necessary | epi_scale_act=True, epi_bias=True | scale+act -> necessary; bias -> useful | no | no | yes | yes | — |
| epilogue_requant_unit | smolvla | necessary | epi_scale_act=True, epi_bias=True | scale+act -> necessary; bias -> useful | no | no | yes | yes | — |
| epilogue_requant_unit | tiny_llama | possible | epi_scale_act=False, epi_bias=False | scale+act -> necessary; bias -> useful | no | no | yes | yes | — |
| epilogue_requant_unit | xr0 | necessary | epi_scale_act=True, epi_bias=True | scale+act -> necessary; bias -> useful | no | no | yes | yes | — |
| dma_engine | bitvla | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| dma_engine | groot_n1d7 | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| dma_engine | molmoact | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| dma_engine | openvla | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| dma_engine | pi05 | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| dma_engine | rdt | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| dma_engine | rdt2 | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| dma_engine | smolvla | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| dma_engine | tiny_llama | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| dma_engine | xr0 | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| multi_stream_dma_descriptor | bitvla | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| multi_stream_dma_descriptor | groot_n1d7 | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| multi_stream_dma_descriptor | molmoact | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| multi_stream_dma_descriptor | openvla | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| multi_stream_dma_descriptor | pi05 | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| multi_stream_dma_descriptor | rdt | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| multi_stream_dma_descriptor | rdt2 | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| multi_stream_dma_descriptor | smolvla | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| multi_stream_dma_descriptor | tiny_llama | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| multi_stream_dma_descriptor | xr0 | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| prefetch_descriptor | bitvla | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| prefetch_descriptor | groot_n1d7 | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| prefetch_descriptor | molmoact | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| prefetch_descriptor | openvla | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| prefetch_descriptor | pi05 | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| prefetch_descriptor | rdt | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| prefetch_descriptor | rdt2 | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| prefetch_descriptor | smolvla | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| prefetch_descriptor | tiny_llama | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| prefetch_descriptor | xr0 | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| partial_sum_object | bitvla | useful | k_shard=True, mn_shard=True | k_shard & not mn_shard -> necessary | no | no | no | no | class never varies across corpus (not discriminating) |
| partial_sum_object | groot_n1d7 | useful | k_shard=True, mn_shard=True | k_shard & not mn_shard -> necessary | no | no | no | no | class never varies across corpus (not discriminating) |
| partial_sum_object | molmoact | useful | k_shard=True, mn_shard=True | k_shard & not mn_shard -> necessary | no | no | no | no | class never varies across corpus (not discriminating) |
| partial_sum_object | openvla | useful | k_shard=True, mn_shard=True | k_shard & not mn_shard -> necessary | no | no | no | no | class never varies across corpus (not discriminating) |
| partial_sum_object | pi05 | useful | k_shard=True, mn_shard=True | k_shard & not mn_shard -> necessary | no | no | no | no | class never varies across corpus (not discriminating) |
| partial_sum_object | rdt | useful | k_shard=True, mn_shard=True | k_shard & not mn_shard -> necessary | no | no | no | no | class never varies across corpus (not discriminating) |
| partial_sum_object | rdt2 | useful | k_shard=True, mn_shard=True | k_shard & not mn_shard -> necessary | no | no | no | no | class never varies across corpus (not discriminating) |
| partial_sum_object | smolvla | useful | k_shard=True, mn_shard=True | k_shard & not mn_shard -> necessary | no | no | no | no | class never varies across corpus (not discriminating) |
| partial_sum_object | tiny_llama | useful | k_shard=True, mn_shard=True | k_shard & not mn_shard -> necessary | no | no | no | no | class never varies across corpus (not discriminating) |
| partial_sum_object | xr0 | useful | k_shard=True, mn_shard=True | k_shard & not mn_shard -> necessary | no | no | no | no | class never varies across corpus (not discriminating) |
| accumulator_merge | bitvla | useful | k_shard=True, mn_shard=True | k_shard & not mn_shard -> necessary | no | no | no | no | class never varies across corpus (not discriminating) |
| accumulator_merge | groot_n1d7 | useful | k_shard=True, mn_shard=True | k_shard & not mn_shard -> necessary | no | no | no | no | class never varies across corpus (not discriminating) |
| accumulator_merge | molmoact | useful | k_shard=True, mn_shard=True | k_shard & not mn_shard -> necessary | no | no | no | no | class never varies across corpus (not discriminating) |
| accumulator_merge | openvla | useful | k_shard=True, mn_shard=True | k_shard & not mn_shard -> necessary | no | no | no | no | class never varies across corpus (not discriminating) |
| accumulator_merge | pi05 | useful | k_shard=True, mn_shard=True | k_shard & not mn_shard -> necessary | no | no | no | no | class never varies across corpus (not discriminating) |
| accumulator_merge | rdt | useful | k_shard=True, mn_shard=True | k_shard & not mn_shard -> necessary | no | no | no | no | class never varies across corpus (not discriminating) |
| accumulator_merge | rdt2 | useful | k_shard=True, mn_shard=True | k_shard & not mn_shard -> necessary | no | no | no | no | class never varies across corpus (not discriminating) |
| accumulator_merge | smolvla | useful | k_shard=True, mn_shard=True | k_shard & not mn_shard -> necessary | no | no | no | no | class never varies across corpus (not discriminating) |
| accumulator_merge | tiny_llama | useful | k_shard=True, mn_shard=True | k_shard & not mn_shard -> necessary | no | no | no | no | class never varies across corpus (not discriminating) |
| accumulator_merge | xr0 | useful | k_shard=True, mn_shard=True | k_shard & not mn_shard -> necessary | no | no | no | no | class never varies across corpus (not discriminating) |
| accumulator_commit | bitvla | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| accumulator_commit | groot_n1d7 | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| accumulator_commit | molmoact | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| accumulator_commit | openvla | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| accumulator_commit | pi05 | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| accumulator_commit | rdt | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| accumulator_commit | rdt2 | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| accumulator_commit | smolvla | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| accumulator_commit | tiny_llama | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| accumulator_commit | xr0 | possible | not gated by a discriminating signal | n/a | no | no | no | no | — |
| fused_dequant_matmul | bitvla | blocked | low-bit/packed weights+scales (dequantized in the f32 capture) | n/a | no | yes | no | no | — |
| fused_dequant_matmul | groot_n1d7 | blocked | low-bit/packed weights+scales (dequantized in the f32 capture) | n/a | no | yes | no | no | — |
| fused_dequant_matmul | molmoact | blocked | low-bit/packed weights+scales (dequantized in the f32 capture) | n/a | no | yes | no | no | — |
| fused_dequant_matmul | openvla | blocked | low-bit/packed weights+scales (dequantized in the f32 capture) | n/a | no | yes | no | no | — |
| fused_dequant_matmul | pi05 | blocked | low-bit/packed weights+scales (dequantized in the f32 capture) | n/a | no | yes | no | no | — |
| fused_dequant_matmul | rdt | blocked | low-bit/packed weights+scales (dequantized in the f32 capture) | n/a | no | yes | no | no | — |
| fused_dequant_matmul | rdt2 | blocked | low-bit/packed weights+scales (dequantized in the f32 capture) | n/a | no | yes | no | no | — |
| fused_dequant_matmul | smolvla | blocked | low-bit/packed weights+scales (dequantized in the f32 capture) | n/a | no | yes | no | no | — |
| fused_dequant_matmul | tiny_llama | blocked | low-bit/packed weights+scales (dequantized in the f32 capture) | n/a | no | yes | no | no | — |
| fused_dequant_matmul | xr0 | blocked | low-bit/packed weights+scales (dequantized in the f32 capture) | n/a | no | yes | no | no | — |
| fused_requant_epilogue | bitvla | useful | epi_scale_act=False, epi_bias=True | scale+act -> necessary; bias -> useful | no | no | yes | yes | — |
| fused_requant_epilogue | groot_n1d7 | necessary | epi_scale_act=True, epi_bias=True | scale+act -> necessary; bias -> useful | no | no | yes | yes | — |
| fused_requant_epilogue | molmoact | useful | epi_scale_act=False, epi_bias=True | scale+act -> necessary; bias -> useful | no | no | yes | yes | — |
| fused_requant_epilogue | openvla | necessary | epi_scale_act=True, epi_bias=True | scale+act -> necessary; bias -> useful | no | no | yes | yes | — |
| fused_requant_epilogue | pi05 | necessary | epi_scale_act=True, epi_bias=True | scale+act -> necessary; bias -> useful | no | no | yes | yes | — |
| fused_requant_epilogue | rdt | necessary | epi_scale_act=True, epi_bias=True | scale+act -> necessary; bias -> useful | no | no | yes | yes | — |
| fused_requant_epilogue | rdt2 | necessary | epi_scale_act=True, epi_bias=True | scale+act -> necessary; bias -> useful | no | no | yes | yes | — |
| fused_requant_epilogue | smolvla | necessary | epi_scale_act=True, epi_bias=True | scale+act -> necessary; bias -> useful | no | no | yes | yes | — |
| fused_requant_epilogue | tiny_llama | possible | epi_scale_act=False, epi_bias=False | scale+act -> necessary; bias -> useful | no | no | yes | yes | — |
| fused_requant_epilogue | xr0 | necessary | epi_scale_act=True, epi_bias=True | scale+act -> necessary; bias -> useful | no | no | yes | yes | — |
| native_lowbit_matmul | bitvla | blocked | low-bit/packed weights+scales (dequantized in the f32 capture) | n/a | no | yes | no | no | — |
| native_lowbit_matmul | groot_n1d7 | blocked | low-bit/packed weights+scales (dequantized in the f32 capture) | n/a | no | yes | no | no | — |
| native_lowbit_matmul | molmoact | blocked | low-bit/packed weights+scales (dequantized in the f32 capture) | n/a | no | yes | no | no | — |
| native_lowbit_matmul | openvla | blocked | low-bit/packed weights+scales (dequantized in the f32 capture) | n/a | no | yes | no | no | — |
| native_lowbit_matmul | pi05 | blocked | low-bit/packed weights+scales (dequantized in the f32 capture) | n/a | no | yes | no | no | — |
| native_lowbit_matmul | rdt | blocked | low-bit/packed weights+scales (dequantized in the f32 capture) | n/a | no | yes | no | no | — |
| native_lowbit_matmul | rdt2 | blocked | low-bit/packed weights+scales (dequantized in the f32 capture) | n/a | no | yes | no | no | — |
| native_lowbit_matmul | smolvla | blocked | low-bit/packed weights+scales (dequantized in the f32 capture) | n/a | no | yes | no | no | — |
| native_lowbit_matmul | tiny_llama | blocked | low-bit/packed weights+scales (dequantized in the f32 capture) | n/a | no | yes | no | no | — |
| native_lowbit_matmul | xr0 | blocked | low-bit/packed weights+scales (dequantized in the f32 capture) | n/a | no | yes | no | no | — |
| kv_cache_object | bitvla | blocked | attention/KV state (lowered in the flat capture) | n/a | no | yes | no | yes | — |
| kv_cache_object | groot_n1d7 | not_applicable | attention/KV state (lowered in the flat capture) | n/a | no | yes | no | yes | — |
| kv_cache_object | molmoact | blocked | attention/KV state (lowered in the flat capture) | n/a | no | yes | no | yes | — |
| kv_cache_object | openvla | blocked | attention/KV state (lowered in the flat capture) | n/a | no | yes | no | yes | — |
| kv_cache_object | pi05 | not_applicable | attention/KV state (lowered in the flat capture) | n/a | no | yes | no | yes | — |
| kv_cache_object | rdt | not_applicable | attention/KV state (lowered in the flat capture) | n/a | no | yes | no | yes | — |
| kv_cache_object | rdt2 | not_applicable | attention/KV state (lowered in the flat capture) | n/a | no | yes | no | yes | — |
| kv_cache_object | smolvla | not_applicable | attention/KV state (lowered in the flat capture) | n/a | no | yes | no | yes | — |
| kv_cache_object | tiny_llama | blocked | attention/KV state (lowered in the flat capture) | n/a | no | yes | no | yes | — |
| kv_cache_object | xr0 | not_applicable | attention/KV state (lowered in the flat capture) | n/a | no | yes | no | yes | — |
| decode_loop_controller | bitvla | possible | not gated by a discriminating signal | n/a | no | no | no | yes | — |
| decode_loop_controller | groot_n1d7 | not_applicable | not gated by a discriminating signal | n/a | no | no | no | yes | — |
| decode_loop_controller | molmoact | possible | not gated by a discriminating signal | n/a | no | no | no | yes | — |
| decode_loop_controller | openvla | possible | not gated by a discriminating signal | n/a | no | no | no | yes | — |
| decode_loop_controller | pi05 | not_applicable | not gated by a discriminating signal | n/a | no | no | no | yes | — |
| decode_loop_controller | rdt | not_applicable | not gated by a discriminating signal | n/a | no | no | no | yes | — |
| decode_loop_controller | rdt2 | not_applicable | not gated by a discriminating signal | n/a | no | no | no | yes | — |
| decode_loop_controller | smolvla | not_applicable | not gated by a discriminating signal | n/a | no | no | no | yes | — |
| decode_loop_controller | tiny_llama | possible | not gated by a discriminating signal | n/a | no | no | no | yes | — |
| decode_loop_controller | xr0 | not_applicable | not gated by a discriminating signal | n/a | no | no | no | yes | — |
