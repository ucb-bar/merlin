# DSE search-space knobs (consolidated P5-P10)

> The structural search-space dimensions the workload-contract analysis discovered, consolidated as the bridge a future DSE engine consumes (alongside the per-workload abstraction-axis `dse_search_space_template.yaml`). **Structural only — no speedup, no chosen design.**

| knob group | phase | enabled | knobs | gated by |
|---|---|---|---|---|
| compute_primitive_shape | P5 | True | tile_8x8, tile_8x16, tile_16x16, tile_16x32, tile_32x32, gemv_lane_64, gemv_lane_128, gemv_lane_256 | structural tile/lane coverage of the real operator geometry (no perf) |
| intra_op_sharding | P7 | True | shard_axis in {M,N,K}, shard_count in [2, 4, 8] | 1797 reduction-free M/N shards available; K-sharding needs partial-sum reduction |
| inter_op_parallelism | P7 | False | num_engines | avg inter-op parallelism 1.291x (low -> limited; favors intra-op sharding) |
| processing_unit_set | P7/P8 | True | dma_engine, epilogue_requant_unit, loop_controller, matrix_engine, scalar_control_unit, vector_gemv_engine | distinct operator families (dense GEMM + skinny/GEMV) + epilogue + DMA |
| pipeline_overlap | P8 | True | async_queue, bounded_loop_command, double_buffered_action_chunk, event_token, loop_carried_state_handle, prefix_state_object, producer_consumer_queue, resident_weight_object | candidate overlaps gated on recovered structure (backbone compute / control loop); per-phase timing needed to schedule |
| memory_residency | P9 | True | resident_weight_object, weight_dtype in ['bf16', 'fp8', 'int8', 'int4', 'fp6'], prefetch_depth | weight-dominated memory pressure; bandwidth needs a design YAML |
| dma_streams | P9 | True | multi_stream_dma_descriptor, double_buffered_activation_tile, prefetch_weight_once | 3 byte-carrying streams/region (weight/activation/output) |
| epilogue_fusion | P10 | True | epilogue_op_set subset of ['bias', 'activation', 'scale'], accumulator_dtype, requant_in_epilogue | directly-fused epilogue slot proven (addmm bias); low-bit/scale gated by a low-bit capture + accuracy |
| hw_sw_boundary_placement | P12 | True | resident_weight_object@{compiler/runtime/command/isa/microcode/datapath}, region_level_dispatch@{compiler/runtime/command/isa/microcode/datapath}, event_token@{compiler/runtime/command/isa/microcode/datapath}, async_queue@{compiler/runtime/command/isa/microcode/datapath}, dma_engine@{compiler/runtime/command/isa/microcode/datapath}, partial_sum_object@{compiler/runtime/command/isa/microcode/datapath} | boundary placement is a search-space axis (Merlin does not choose); see boundary_candidate_contracts.yaml + boundary_dse_knobs.yaml |

Each knob group is evidence-labeled and grounded in the per-phase artifacts (P5 geometry/coverage, P7 sharding/hierarchy, P8 pipeline, P9 memory/DMA, P10 epilogue). The measurements each knob needs before a *quantitative* DSE decision (per-unit throughput, bandwidth, accuracy for low-bit, per-phase timing) are named in those artifacts. **No speedup is claimed.**
