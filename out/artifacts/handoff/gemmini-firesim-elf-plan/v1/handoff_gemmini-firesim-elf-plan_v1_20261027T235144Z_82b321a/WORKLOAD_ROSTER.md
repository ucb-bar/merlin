# Frozen chunk roster at 82b321a89

These are the published fusion=2/prepack/bias-preload/CSE-off tasks. Re-extract after graph changes.

| Chunk | Status | Owned contraction workloads | Full boundary scope |
| --- | --- | --- | --- |
| resnet-classifier | ready_for_tuning | 1 | Complete source region; see restored ResNet catalog |
| resnet-stage1-block1 | ready_for_tuning | 3 | Complete source region; see restored ResNet catalog |
| resnet-stage1-block2 | ready_for_tuning | 1 | Complete source region; see restored ResNet catalog |
| resnet-stage1-block3 | ready_for_tuning | 0 | Complete source region; see restored ResNet catalog |
| resnet-stage2-block1 | ready_for_tuning | 4 | Complete source region; see restored ResNet catalog |
| resnet-stage2-block2 | ready_for_tuning | 1 | Complete source region; see restored ResNet catalog |
| resnet-stage2-block3 | ready_for_tuning | 0 | Complete source region; see restored ResNet catalog |
| resnet-stage2-block4 | ready_for_tuning | 0 | Complete source region; see restored ResNet catalog |
| resnet-stage3-block1 | ready_for_tuning | 4 | Complete source region; see restored ResNet catalog |
| resnet-stage3-block2 | ready_for_tuning | 1 | Complete source region; see restored ResNet catalog |
| resnet-stage3-block3 | ready_for_tuning | 0 | Complete source region; see restored ResNet catalog |
| resnet-stage3-block4 | ready_for_tuning | 0 | Complete source region; see restored ResNet catalog |
| resnet-stage3-block5 | ready_for_tuning | 0 | Complete source region; see restored ResNet catalog |
| resnet-stage3-block6 | ready_for_tuning | 0 | Complete source region; see restored ResNet catalog |
| resnet-stage4-block1 | ready_for_tuning | 4 | Complete source region; see restored ResNet catalog |
| resnet-stage4-block2 | ready_for_tuning | 1 | Complete source region; see restored ResNet catalog |
| resnet-stage4-block3 | ready_for_tuning | 0 | Complete source region; see restored ResNet catalog |
| resnet-stem | ready_for_tuning | 1 | Complete source region; see restored ResNet catalog |
| tinyllama-k-projection-m1 | ready_for_tuning | 1 | Full K=2048 and N=256 K projection with frozen INT8 weights and grouped scale reconstruction; quantized source live-in |
| tinyllama-k-projection-m16 | ready_for_tuning | 1 | Full K=2048 and N=256 K projection with frozen INT8 weights and grouped scale reconstruction; quantized source live-in |
| tinyllama-mlp-5728-m1 | ready_for_tuning | 2 | Complete RMSNorm, frozen quantize, gate/up, SiLU/product, down, scaling and residual; every K/N retained |
| tinyllama-mlp-5728-m16 | ready_for_tuning | 2 | Complete RMSNorm, frozen quantize, gate/up, SiLU/product, down, scaling and residual; every K/N retained |
| tinyllama-mlp-6096-m1 | ready_for_tuning | 2 | Complete RMSNorm, frozen quantize, gate/up, SiLU/product, down, scaling and residual; every K/N retained |
| tinyllama-mlp-6096-m16 | ready_for_tuning | 2 | Complete RMSNorm, frozen quantize, gate/up, SiLU/product, down, scaling and residual; every K/N retained |
| tinyllama-q-projection-m1 | ready_for_tuning | 1 | Full K=2048 and N=2048 Q projection with frozen INT8 weights and grouped scale reconstruction; quantized source live-in |
| tinyllama-q-projection-m16 | ready_for_tuning | 1 | Full K=2048 and N=2048 Q projection with frozen INT8 weights and grouped scale reconstruction; quantized source live-in |
| tinyllama-v-projection-m1 | ready_for_tuning | 0 | Full K=2048 and N=256 V projection with frozen INT8 weights and grouped scale reconstruction; quantized source live-in |
| tinyllama-v-projection-m16 | ready_for_tuning | 0 | Full K=2048 and N=256 V projection with frozen INT8 weights and grouped scale reconstruction; quantized source live-in |
| smolvla-action-input-projection | ready_for_tuning | 1 | Full 50x32 signed INT8 action projection to 720 features, INT32 bias and frozen scaling; quantized live-in captured from source |
| smolvla-action-projection-update | ready_for_tuning | 1 | Full 50x720->50x32 frozen quantized action projection, INT32 bias and Euler/timestep update; normalized source input is an explicit captured boundary |
| smolvla-action-time-embedding | ready_for_tuning | 2 | Full 50-row time/action embedding, concatenation, two linear projections, INT32 bias and SiLU; time sin/cos live-in captured after CPU work |
| smolvla-action-update | blocked_connected_host_check | 0 | Full final norm, all 50x32 projected actions, INT32 bias and Euler/timestep update |
| smolvla-expert-cross-attention-core | ready_for_tuning | 3 | All 15 query heads, 50 queries and full 113-key context; selected QK/PV integer planes, source softmax, output projection and residual; live-ins after source RoPE |
| smolvla-expert-cross-qkv | ready_for_tuning | 2 | Full query and prefix K/V projections and source norms; position/RoPE host work follows outside this chunk |
| smolvla-expert-mlp | ready_for_tuning | 2 | One full 50-row expert MLP, norm, gate/up, SiLU/product, down and residual |
| smolvla-expert-self-attention-core | ready_for_tuning | 2 | All 15 query heads, 50 queries and full 163-key context; selected QK/PV integer planes, source softmax, output projection and residual; live-ins after source RoPE |
| smolvla-expert-self-qkv | ready_for_tuning | 1 | Full RMSNorm and Q/K/V projections, all heads/features; position/RoPE host work follows outside this chunk |
