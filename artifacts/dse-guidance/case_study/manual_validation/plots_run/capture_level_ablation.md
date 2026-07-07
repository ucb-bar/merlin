# Capture-level ablation (all)

> REAL multi-level recaptures (m2m flags). **flat** = the committed capture (attention recovered by re-parsing generics, P18-A); **high_level** (`level="high-level"`) emits attention/softmax as NAMED `linalg_ext` ops; **quant_qdq** (`preserve_qdq` int8) emits `quant_ext.dequantize` (packed-lowbit + scales). **Loop-preserving is torch.export-blocked** (no Level — the named frontier a TorchDynamo/`torch.cond` frontend would unlock).

## Op vocabulary per level

| workload | level | available | matmul | generic | linalg_ext.softmax | quant_ext.dequantize | scf.for |
|---|---|---|---|---|---|---|---|
| bitvla | flat | True | 15 | 377 | 0 | 0 | 0 |
| bitvla | high_level | True | 15 | 371 | 2 | 0 | 0 |
| bitvla | quant_qdq | True | 15 | 377 | 0 | 1 | 0 |
| openvla | flat | True | 26 | 190 | 0 | 0 | 0 |
| openvla | high_level | True | 26 | 142 | 4 | 0 | 0 |
| openvla | quant_qdq | True | 26 | 190 | 0 | 26 | 0 |
| pi05 | flat | True | 777 | 4602 | 0 | 0 | 0 |
| pi05 | high_level | True | 777 | 2769 | 116 | 0 | 0 |
| pi05 | quant_qdq | True | 782 | 4156 | 0 | 782 | 0 |
| rdt | flat | True | 20 | 184 | 0 | 0 | 0 |
| rdt | high_level | True | 20 | 172 | 4 | 0 | 0 |
| rdt | quant_qdq | True | 20 | 184 | 0 | 20 | 0 |

## Axis-unlock matrix (which DSE axes become decidable at which level)

| workload | attention_primitive | softmax_reduction_unit | packed_lowbit_weights | scale_metadata | bounded_loop_command | kv_cache_object |
|---|---|---|---|---|---|---|
| bitvla | named (linalg_ext) @high_level | named @high_level | decidable @quant_qdq | decidable @quant_qdq | blocked (torch.export unrolls loops) | blocked (torch.export unrolls loops) |
| openvla | named (linalg_ext) @high_level | named @high_level | decidable @quant_qdq | decidable @quant_qdq | blocked (torch.export unrolls loops) | blocked (torch.export unrolls loops) |
| pi05 | named (linalg_ext) @high_level | named @high_level | decidable @quant_qdq | decidable @quant_qdq | blocked (torch.export unrolls loops) | blocked (torch.export unrolls loops) |
| rdt | named (linalg_ext) @high_level | named @high_level | decidable @quant_qdq | decidable @quant_qdq | blocked (torch.export unrolls loops) | blocked (torch.export unrolls loops) |

**Loop-preserving capture is the blocked frontier**: `scf.for` is absent at every level (torch.export unrolls Python loops), so `bounded_loop_command` / `kv_cache_object` stay blocked — the one axis a richer (TorchDynamo / torch.cond) frontend would unlock.

