# Task accuracy

The full ResNet50, TinyLlama, and SmolVLA hosts are bit-exact against the Atlas arithmetic: every launch matches a reference that follows the MXU's FP8 E4M3 operands, power-of-two tensor scales, exact products, and BF16 rounding after every multiply-add in ascending K. That is the correctness claim for the compiler, runtime, and host code. It is not a claim that the outputs match the original models in PyTorch.

Because the arithmetic is coarser than PyTorch's, task outputs drift. On one input each:

| Model | PyTorch | Atlas |
| --- | --- | --- |
| TinyLlama, 128-token prompt, 840-launch projection-only host (v1) | Tokens [278, 3711, 5652] | Tokens [278, 14263, 658] |
| SmolVLA policy | Reference actions | 20.4% action relative L2 error |
| ResNet50, church image | Class 497 | Class 562 |

The TinyLlama Atlas tokens come from the earlier 840-launch host, which ran only the projection panels on Atlas; its recorded captured-reply replay of ELF SHA-256 `43761da497abf0442c15b60731b984c1f009d18c33731ec25d0c276743f81402` printed [278, 14263, 658]. The current 1,038-launch kit also runs attention and RMSNorm on Atlas; its recorded replay of default ELF `3b75125b5f552eeb93cdab158bd09349170e7b23894dc8f3a717040a732eb081` printed [278, 3711, 5652] ([TinyLlama attention build](tiny-attention-build.md)), the same tokens as PyTorch on this prompt. Equal tokens do not establish equal logits.

ResNet50 matches PyTorch's top-1 on the three other images checked. An independent CPU emulation of the Atlas arithmetic reproduces all three results, so the gap comes from the number format rather than the compiler or host code. BF16 accumulation over long K dominates for TinyLlama; FP8 weight precision dominates for ResNet50; SmolVLA has both.

## Opt-in precision modes

The MXU always rounds to BF16 after each multiply-add, so higher precision has to come from software. [`tools/precision_modes`](../tools/precision_modes/) emulates three recipes that use only existing hardware interfaces. It is an offline CPU emulation of the [declared arithmetic](../tools/precision_modes/arithmetic.py), not an Atlas run: no compiler, runtime, program, or ELF changes, and nothing else imports it. With no recipe flag it reproduces the frozen default bit for bit.

- Finer scales (`--finer-scales`, or `activation` or `weight` for one axis): one power-of-two scale per activation row and per weight output column instead of per tensor, applied through prepacking and a host output factor. No extra stores or launches.
- K split (`--k-split 128`): run each contraction as shorter K chunks, pop each raw BF16 partial, add the partials in FP64 on the host, and round to BF16 once. Cost: extra partial stores, host merge work, and more launches or a multi-partial program.
- Two-term weights (`--two-term`): a second FP8 weight encodes the residual of the FP32 weight with its own scale; both matmuls run on the same activations and merge in FP64. Cost: twice the MXU work and weight bytes at the selected sites.

`--sites` limits K split and two-term weights to comma-separated site globs or an `@file` list, such as the ResNet50 [half-work subset](../tools/precision_modes/resnet_half_work.txt). Run the tool from the repository root with PyTorch, the model's packages (safetensors and Transformers for TinyLlama, LeRobot for SmolVLA, and torchvision for ResNet50), local checkpoints, and the PR #26 run artifacts:

```sh
c++ -O3 -shared -fPIC tools/precision_modes/ordered_mac.cpp -o build/ordered_mac.so
export ATLAS_PRECISION_MAC_LIBRARY=$PWD/build/ordered_mac.so PYTHONPATH=tools OMP_NUM_THREADS=1
python -m precision_modes tiny --capture "$TINY_CAPTURE" --checkpoint "$TINY_CHECKPOINT" --k-split 128 --finer-scales
python -m precision_modes smol --checkpoint "$SMOL_CHECKPOINT" --backbone "$SMOL_BACKBONE" --observation "$SMOL_OBSERVATION" --k-split 128
python -m precision_modes resnet --artifact-dir "$RESNET_PROVIDER" --checkpoint "$RESNET_CHECKPOINT" --input "$RESNET_INPUT" --two-term
```

Each command prints a JSON report against native PyTorch; `python -m precision_modes <model> --help` lists the inputs. For the default recipe, the reports check TinyLlama's capture prompt and the SmolVLA and ResNet50 `--reference` artifacts byte for byte: three TinyLlama logit vectors, 300 SmolVLA actions, and 54 ResNet50 contraction outputs with 1,000 logits. [`test_precision_modes.py`](../test/test_precision_modes.py) checks the arithmetic against exact references.

## Multi-input results

These are offline emulation results on frozen inputs, not device runs or general benchmarks. TinyLlama, eight 128-token prompts with eight generated tokens each, against native FP32:

| Recipe | Continuations matching PyTorch | Forced top-1, of 64 | Pooled prefill RMSE |
| --- | ---: | ---: | ---: |
| Default | 0/8 | 43 | 0.423 |
| Finer scales | 2/8 | 52 | 0.428 |
| K128 plus finer scales | 6/8 | 62 | 0.208 |

SmolVLA, five TRAIN observations from episode 0 with the same explicit noise. K128 has the lowest mean, though K512 is better on the original observation:

| Recipe | Mean action relative L2 | Better than default |
| --- | ---: | ---: |
| Default | 14.2% | n/a |
| K128 | 7.3% | 5/5 |
| K512 | 8.7% | 4/5 |
| Activation-row scales | 12.6% | 3/5 |
| K512 plus activation-row scales | 9.5% | 4/5 |

ResNet50, 100 Imagenette images, where native PyTorch has 88% label top-1. The half-work subset uses two terms at 35 of 54 contractions, adding 49.7% MXU work and 38.2% padded weight bytes; it was chosen on four development images before this run:

| Recipe | Top-1 agreement with PyTorch | Label top-1 | Mean logit relative L2 |
| --- | ---: | ---: | ---: |
| Default | 94% | 86% | 0.301 |
| Finer scales | 95% | 86% | 0.299 |
| Two terms, half-work subset | 98% | 88% | 0.213 |
| Two terms everywhere | 97% | 89% | 0.209 |
