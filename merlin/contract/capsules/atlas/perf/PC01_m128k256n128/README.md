# PC01_m128k256n128

PC01_m128k256n128: k_chain over A0[128, 256]:bf16, W[256, 128]:bf16, W2[128, 128]:bf16, authored from two chained contractions where the second reads the first's output: the inter-layer transition, at two reduction depths so the transition cost can be separated from the stages' own (tile=32; K=256, M=128, N=128).

kind=model_slice label=dev op=k_chain modes={}
