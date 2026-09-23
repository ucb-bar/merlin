# R3_attention_qk_fp16

R3_attention_qk_fp16: attention_qk over Q[16, 32]:fp16, K[16, 32]:fp16, authored from attention Q @ K^T scores, fp16 -> fp32 (TinyLlama/Gemma-2 attn tile).

kind=model_slice label=public op=attention_qk modes={}
