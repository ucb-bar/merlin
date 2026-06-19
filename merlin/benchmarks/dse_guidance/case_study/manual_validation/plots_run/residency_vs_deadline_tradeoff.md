# Residency vs deadline tradeoff (all)

> At configured K and a 100 ms replan deadline: the required **weight bandwidth** if weights are reloaded every step (non-resident) vs kept resident (loaded once). The ratio is exactly K — residency removes a K× bandwidth requirement at the cost of the resident capacity shown (by dtype). Structural requirement only.

| workload | K | non-resident B/s | resident B/s | ratio | resident bf16 | resident int8 |
|---|---|---|---|---|---|---|
| bitvla | 7 | 385.0MB/s | 55.0MB/s | 7x | 2.8MB | 1.4MB |
| groot_n1d7 | 4 | 82.0GB/s | 20.5GB/s | 4x | 1.0GB | 524.6MB |
| molmoact | 8 | 282.2GB/s | 35.3GB/s | 8x | 1.8GB | 903.0MB |
| openvla | 7 | 210.0MB/s | 30.0MB/s | 7x | 1.5MB | 768.0KB |
| pi05 | 10 | 160.2GB/s | 16.0GB/s | 10x | 820.1MB | 410.1MB |
| rdt | 5 | 18.3GB/s | 3.7GB/s | 5x | 187.5MB | 93.8MB |
| rdt2 | 5 | 14.4GB/s | 2.9GB/s | 5x | 147.7MB | 73.8MB |
| smolvla | 10 | 19.2GB/s | 1.9GB/s | 10x | 196.6MB | 98.3MB |
| tiny_llama | 7 | 40.1GB/s | 5.7GB/s | 7x | 293.0MB | 146.5MB |
| xr0 | 5 | 7.9GB/s | 1.6GB/s | 5x | 80.6MB | 40.3MB |

Residency is therefore a search axis (capacity to keep weights on-chip vs the K× reload bandwidth it avoids). Absolute bytes are small/random-init; the K× ratio and the capacity-by-dtype ordering are the robust signal.

