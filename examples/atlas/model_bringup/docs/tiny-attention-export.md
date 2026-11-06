# TinyLlama attention export

The exporter dispatches on `atlas_instruction_execution`. Captures with an absent or false flag and an absent or empty `attention_calls` ledger use the projection-only exporter. Attention mode requires a true flag and a complete, nonempty ledger in step/layer order; contradictory flags and ledgers are rejected before model configuration or asset creation.

Use the model, interpreter, and compiler variables from [model bringup setup](../README.md#build) and [Inputs and environments](bringing-up-a-model.md#inputs-and-environments), and set `$EXPECT_PROGRAMS` to the reviewed distinct-program inventory for this export profile. Keep its backend receipt separate from program checks and fast builds. Select the platform physical DRAM size explicitly for attention export. `--dram-base` defaults to `0x80000000`; `--dram-bytes` has no default. Both values accept decimal or hexadecimal numbers and must be aligned to 64 bytes. The range must contain the fixed `0x90000000` DMA arena and every launch's full arena extent. The host build also validates its ELF allocations against the selected range.

```sh
PYTHONPATH="$MODEL47:$ATLAS/tools" "$MODEL_PYTHON" "$KERNELS/lib/atlas_backend.py" run --backend atlas-mlir \
  --atlas-mlir-bin "$ATLAS_MLIR_BIN" --min-tier stream-verified --expect-min-programs "$EXPECT_PROGRAMS" \
  --receipt "$MODEL_OUTPUT_ROOT/tiny-export-backend.json" -- \
  tools/tiny_full/export_capture.py \
  --capture "$TINY_CAPTURE" --capture-sha256 "$TINY_CAPTURE_SHA256" \
  --model-root "$MODEL47" --out "$TINY_ASSETS" \
  --dram-base 0x80000000 --dram-bytes 0x80000000 --qualify-launches
```

The example describes a platform with 2 GiB of DRAM beginning at `0x80000000`; replace the range with the platform's actual physical memory. Export checks arena placement; the host builder checks the complete image placement.

`--capture-source-archive /path/to/archive.json` permits hash-verified captured source copies when originals changed. The FP8 codec is imported from the exact path returned by archive validation. Source and model bookend checks remain active. `--qualify-launches` uses the tracked `third_party/atlas-kernels/lib/atlas_execution.py` executor with the explicit `--model-root`, RTL numerics, complete final-arena comparisons, and requested DMA checks.

Qualification writes `qualification.json` in the original RTL gate format, binding the exact `replay.json`, every launch's source, binary, input, output, and final arena, the selected model source and data hashes, and the execution helper hashes. Its SHA-256 is listed in `SHA256SUMS`. For a complete full-model export, pass this file directly to `tools/tiny_full/build_attention.py --qualification /path/to/fresh-bundle/qualification.json --qualification-sha256 QUALIFICATION_SHA256` with the other documented build arguments. The builder requires the complete 22-layer, three-step scope; a qualified block fixture remains a block fixture.

After moving source trees, repeat `--relocate OLD=NEW` to rebind captured absolute source prefixes to their physical locations. The longest matching prefix wins. The exporter preserves the original capture bytes and SHA-256, verifies each original source hash at its selected physical path, and records the explicit mappings and substitutions. An exact source archive can also supply a captured source whose original path is absent. Relocation and archive fallback never accept a different source hash.
