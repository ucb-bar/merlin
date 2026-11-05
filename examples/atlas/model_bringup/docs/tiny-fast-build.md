# TinyLlama program check and fast host build

Both tools take an exported full-model attention-v2 asset tree, as admitted by [the attention host build](tiny-attention-build.md), and the unmodified npu_model checkout at 47b8599d8ffbe5ddd96059ffe2b5be2427a410f5 as `--model-root`. Neither executes an Atlas instruction or reruns numerical qualification. Use [model bringup setup](../README.md#build) and the variables in [Inputs and environments](bringing-up-a-model.md#inputs-and-environments), keeping `ATLAS_ROOT` and `ATLAS_MLIR_BIN` on the canonical compiler; export `ATLAS="$VIEW"`, `ATLAS_SOURCE_ROOT="$VIEW"`, and `ATLAS_MLIR_ROOT="$VIEW"`, and use `cd "$VIEW"`; inspect each separate backend receipt. The recorded program census is 210 for the check and 850 for each fast variant, with at least `stream-verified` required for every program.

`tools/tiny_full/check_programs.py` regenerates every final device program source from its exported recipe (the traffic-reuse projection recipe with its epilogue and any q/k RoPE post-pass, relocated to the launch arena, the attention-v2 generator, or the standalone norm staging, whose initial arena must also match), requires it to equal the exported source byte for byte, checks matrix register pairs and banks, and assembles each unique program with the model's assembler to require the exported binary. `--contract-sha256` binds the export. It prints launch, unique program, role, and epilogue counts.

```sh
PYTHONPATH="$MODEL47:$ATLAS/tools" "$MODEL_PYTHON" -B "$KERNELS/lib/atlas_backend.py" run --backend atlas-mlir \
  --atlas-mlir-bin "$ATLAS_MLIR_BIN" --min-tier stream-verified --expect-min-programs 210 \
  --receipt "$MODEL_OUTPUT_ROOT/tiny-program-check-backend.json" -- \
  tools/tiny_full/check_programs.py --assets "$TINY_ASSETS" \
  --contract-sha256 "$TINY_CONTRACT_SHA256" --model-root "$MODEL47"
```

`tools/tiny_full/build_fast.py` builds the default-off fast transport host, compiled with `ATLAS_FAST_MODE=1`. Fast projection programs read FP8 weights in place from the linked `tiny_full_weights_N` symbols instead of staging them in the arena. `--default` names a SIMS output of `build_attention.py` for the same assets; its compile manifest and images are reused, with the fast programs, their panel table, and `fast-images.S` added. Programs are generated with a placeholder weight base, the host is linked, and generation repeats against the linked ELF until every weight base is stable. Each default projection program is regenerated and checked against the export first.

```sh
PYTHONPATH="$MODEL47:$ATLAS/tools" "$MODEL_PYTHON" -B "$KERNELS/lib/atlas_backend.py" run --backend atlas-mlir \
  --atlas-mlir-bin "$ATLAS_MLIR_BIN" --min-tier stream-verified --expect-min-programs 850 \
  --receipt "$MODEL_OUTPUT_ROOT/tiny-fast-backend.json" -- \
  tools/tiny_full/build_fast.py --assets "$TINY_ASSETS" --model-root "$MODEL47" \
  --default "$TINY_DEFAULT" --platform-root "$PLATFORM_ROOT" \
  --toolchain-bin "$RISCV_BIN" --host-level 0 --out "$TINY_FAST"
```

The ELF is `OUT/build/atlas-tinyllama.elf`, and the generated programs and tables are in `OUT/generated`. Only SIMS is built. The host built without `ATLAS_FAST_MODE` is the default host, so a default build is unchanged by the fast sources. A fast ELF is a build artifact; its execution evidence comes from a captured-reply replay of that exact ELF.

Use `--host-level 0` for no host optimizations, `--host-level 1` for immutable attention template admission, or `--host-level 2` for admission plus coalesced tiled output copies (the default). Each mode binds its own projection programs; keep its generated directory with its ELF.
