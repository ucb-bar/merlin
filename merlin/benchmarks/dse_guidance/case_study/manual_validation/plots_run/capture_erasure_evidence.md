# Capture-erasure IR evidence (all)

> The 'erased' claims, demonstrated from the recapture IR rather than asserted. Loops are absent (torch.export unrolls them — only a gather/scatter artifact shows scf.for); attention lives in `linalg.generic` (now re-parsed); no packed low-bit integer tensor types survive (the capture is dequantized f32).

| workload | scf loops | linalg.matmul | linalg.generic | low-bit int types | loops preserved |
|---|---|---|---|---|---|
| bitvla | 0 | 15 | 377 | False | False |
| groot_n1d7 | 0 | 18 | 167 | False | False |
| molmoact | 0 | 17 | 188 | False | False |
| openvla | 0 | 26 | 190 | False | False |
| pi05 | 0 | 777 | 4602 | False | False |
| rdt | 0 | 20 | 184 | False | False |
| rdt2 | 0 | 23 | 193 | False | False |
| small_llama | 0 | 15 | 134 | False | False |
| smolvla | 2 | 106 | 852 | False | True |
| tiny_llama | 0 | 15 | 112 | False | False |
| xr0 | 0 | 19 | 144 | False | False |

**1/11 captures retain any scf loop** (and that one is a bool-mask gather artifact, not a model loop). No capture retains packed low-bit types. This is the concrete basis for the capture-fidelity 'erased' rows.

