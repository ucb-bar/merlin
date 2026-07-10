# Capture-fidelity matrix (all)

> The likely central result: which structural features the flat capture preserves vs erased. `strong`=recovered from IR; `assumed (config K)`=loop count is a reference value, not captured; `erased`=lowered/dequantized away; `measured (host)`=real host measurement; `not_claimed`=needs a target design. Findings that depend on `assumed`/`erased` rows are capture-limited.

| feature | bitvla | groot_n1d7 | molmoact | openvla | pi05 | rdt | rdt2 | smolvla | tiny_llama | xr0 |
|---|---|---|---|---|---|---|---|---|---|---|
| op_shapes_MNK | strong | strong | strong | strong | strong | strong | strong | strong | strong | strong |
| region_roles | strong | strong | strong | strong | strong | strong | strong | strong | strong | strong |
| dtype_information | strong | strong | strong | strong | strong | strong | strong | strong | strong | strong |
| attention_bmm_qkT_attnV | n/a (attention-free here) | recovered (32 ops) | n/a (attention-free here) | n/a (attention-free here) | recovered (232 ops) | recovered (25 ops) | n/a (attention-free here) | recovered (84 ops) | n/a (attention-free here) | recovered (14 ops) |
| softmax | recovered | recovered | recovered | recovered | recovered | n/a | recovered | recovered | recovered | n/a |
| normalization | n/a (norm as elementwise primitives) | recovered | n/a (norm as elementwise primitives) | n/a (norm as elementwise primitives) | recovered | n/a (norm as elementwise primitives) | n/a (norm as elementwise primitives) | recovered | n/a (norm as elementwise primitives) | n/a (norm as elementwise primitives) |
| K_or_decode_loop | recovered (K=7, IR scf.for) | recovered (K=4, IR scf.for) | recovered (K=8, IR scf.for) | recovered (K=7, IR scf.for) | recovered (K=10, IR scf.for) | recovered (K=5, IR scf.for) | recovered (K=5, IR scf.for) | recovered (K=10, IR scf.for) | recovered (K=7, IR scf.for) | recovered (K=5, IR scf.for) |
| kv_cache_state | recovered (79872 B, IR iter_arg) | n/a (prefix-KV invariant, closed-over) | recovered (262144 B, IR iter_arg) | recovered (221184 B, IR iter_arg) | n/a (prefix-KV invariant, closed-over) | n/a (prefix-KV invariant, closed-over) | n/a (prefix-KV invariant, closed-over) | n/a (prefix-KV invariant, closed-over) | recovered (61440 B, IR iter_arg) | n/a (prefix-KV invariant, closed-over) |
| loop_carried_state | recovered (5 iter_args: counter,kv_cache,token_buffer) | recovered (2 iter_args: counter,latent) | recovered (5 iter_args: counter,kv_cache,token_buffer) | recovered (5 iter_args: counter,kv_cache,token_buffer) | recovered (2 iter_args: counter,latent) | recovered (3 iter_args: counter,latent) | recovered (2 iter_args: counter,latent) | recovered (2 iter_args: counter,latent) | recovered (5 iter_args: counter,kv_cache,token_buffer) | recovered (2 iter_args: counter,latent) |
| packed_lowbit_layout | erased | erased | erased | erased | erased | erased | erased | erased | erased | erased |
| scale_metadata | erased | erased | erased | erased | erased | erased | erased | erased | erased | erased |
| host_dispatch_count | measured (host) | measured (host) | measured (host) | measured (host) | measured (host) | measured (host) | measured (host) | measured (host) | measured (host) | measured (host) |
| target_latency_cycles | not_claimed | not_claimed | not_claimed | not_claimed | not_claimed | not_claimed | not_claimed | not_claimed | not_claimed | not_claimed |

**Per-workload DSE risk:**

- **bitvla** (autoregressive_vla, severity low): lost async_backbone_head_overlap, action_chunk_horizon, replan_deadline; hides axes decode_kv_cache_path, autonomous_K_loop, command_batching, resident_prefix_kv, backbone_head_partition, async_chunk_overlap
- **groot_n1d7** (diffusion, severity low): lost action_chunk_horizon, replan_deadline; hides axes resident_action_head_weights, autonomous_K_loop, command_batching, async_chunk_overlap, backbone_head_partition
- **molmoact** (autoregressive_vla, severity low): lost async_backbone_head_overlap, action_chunk_horizon, replan_deadline; hides axes decode_kv_cache_path, autonomous_K_loop, command_batching, resident_prefix_kv, backbone_head_partition, async_chunk_overlap
- **openvla** (autoregressive_vla, severity low): lost async_backbone_head_overlap, action_chunk_horizon, replan_deadline; hides axes decode_kv_cache_path, autonomous_K_loop, command_batching, resident_prefix_kv, backbone_head_partition, async_chunk_overlap
- **pi05** (flow_matching, severity low): lost async_backbone_head_overlap, action_chunk_horizon, replan_deadline; hides axes resident_action_head_weights, autonomous_K_loop, command_batching, resident_prefix_kv, backbone_head_partition, async_chunk_overlap
- **rdt** (diffusion, severity low): lost action_chunk_horizon, replan_deadline; hides axes resident_action_head_weights, autonomous_K_loop, command_batching, async_chunk_overlap, backbone_head_partition
- **rdt2** (diffusion, severity low): lost action_chunk_horizon, replan_deadline; hides axes resident_action_head_weights, autonomous_K_loop, command_batching, async_chunk_overlap, backbone_head_partition
- **smolvla** (flow_matching, severity low): lost async_backbone_head_overlap, action_chunk_horizon, replan_deadline; hides axes resident_action_head_weights, autonomous_K_loop, command_batching, resident_prefix_kv, backbone_head_partition, async_chunk_overlap
- **tiny_llama** (llm, severity low): lost async_backbone_head_overlap; hides axes decode_kv_cache_path, autonomous_K_loop, command_batching, resident_prefix_kv, backbone_head_partition, async_chunk_overlap
- **xr0** (diffusion, severity low): lost async_backbone_head_overlap; hides axes resident_action_head_weights, autonomous_K_loop, command_batching, resident_prefix_kv, backbone_head_partition, async_chunk_overlap
