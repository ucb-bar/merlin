# DeepSeek-R1-Distill-Qwen-1.5B capture workload

The study needs this model and model2MLIR had no workload for it, so this is the one authored here.
It is kept in the experiment tree as the reproducible source; **model2MLIR is where it must be
installed to run**:

```sh
cp -r loader.py capture.toml $MERLIN_M2M_DIR/workloads/deepseek_r1_qwen/
scripts/kvc_capture.sh deepseek_r1_qwen fp32
```

Weights come from the HF cache (`HF_HOME=$HF_HOME`); the checkpoint is
~3.4 GB and was not present, unlike TinyLlama and SmolVLA.

## Why it is not a copy of the Llama loader

It is a Qwen2 decoder, and the op inventory sees the difference: the Q/K/V projections carry a
**bias** where Llama's do not, and GQA runs 12 query heads to 2 KV heads. Both show up in the
captured signature — `tensor<1536xf32>`, `tensor<256xf32>`, `tensor<256xf32>` sitting beside the
projection weights.

## Verified

Captured at full depth with pretrained weights: **197 `linalg.matmul` = 28 layers x 7 + lm_head**,
2055 weighable ops, 24.73 GFLOP, **0 opaque ops**. Config recovered from the capture: vocab 151936,
hidden 1536, KV width 256, FFN 8960.

⚠️ `M2M_QWEN_LAYERS` truncates AND switches to random init, exactly like `M2M_LLAMA_LAYERS` — it is a
smoke path, never a smaller version of the real model. Leave it unset.
