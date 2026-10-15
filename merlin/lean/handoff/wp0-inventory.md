# WP-0 inventory handoff (stopped early: token budget)

> **Historical (2026-10-15).** A work-package handoff from before the code landed; its status
> lines are out of date. The current state is the code under `merlin/lean/` and the design doc.

## Done and committed
- Nothing beyond this note. No code was written: `stages.py`, `inventory.py`, the recorder call
  sites, `test_lean_stages.py` and `paper_denominator.json` do not exist yet. No capture was run.

## Environment state
- Merlin venv: `<worktree>/.venv` (CPython 3.14.7, `-e .[dev,verify,targetgen] -e packages/merlin-experiments`).
- model2MLIR: `/home/jeremy/Projects/model2MLIR` @ `905d2c9`, **no `.venv` created yet**. Its `uv.lock`
  pins torch 2.10.0, torchao 0.17.0, torchvision 0.25.0, transformers 4.57.6, xdsl 0.65.0.
  smolvla needs its own venv (`lerobot[smolvla]==0.5.1`, Python 3.12, `workloads/smolvla/capture.toml`).
- No `clang`, `riscv64-unknown-elf-gcc` or spike on PATH. `mlir-opt`/`mlir-translate`/`llc` (LLVM 23.0.0git)
  are in `/home/jeremy/Projects/llvm-project/build/bin`. Lean's toolchain ships a real clang 23.1.0
  (`~/.elan/toolchains/leanprover--lean4---v4.35.0/bin/clang`) that emits riscv64 IR; its riscv64
  data layout is `e-m:e-p:64:64-i64:64-i128:128-n32:64-S128`. It could serve as `MERLIN_CLANG` for
  the data-layout observation only (record this as a deviation from the paper toolchain).
- Host package `out/artifacts/targets/host/gemmini_rocket_scalar_int8_v0` is not minted on this
  machine; mint with `build_tools/scripts/mint_scalar_host_package.py examples/gemmini/target/scalar-host-recipe.yaml`.

## Findings from code reading (verify, then use for the inventory)
1. Paper build = `private_full_models.py` → `compile_saved_model(run="none", device=plan_before_build(...,
   granularity=BY_GROUP))` with the scalar int8 host package. So: `int8_compute=True`, `features=None`
   (scalar packages forbid `compiler_features`; `compile_saved_model` passes none), `backend="scalar"`
   → `lower_model_file(textual=False, vectorize=False)`: **U16 textual route is off**.
2. `spike_model.build` runs `gcc -print-sysroot` (`_mlir_runtime_compiler`) BEFORE lowering, so the real
   build cannot reach `.ll` without the cross gcc. A driver must replicate its steps: `prepare_build_bundle`
   → `zephyr_model.prepare_for_lowering` → (entry projection: off) → `lower_model_file(...)`.
3. With `features=None`, `prepare_for_lowering` fires only: `_prepare_model_mlir` [torchao affine,
   scalar squares, overrank collapse, `_propagate_quant_inner`, `qinner.lift`, `apply_quant(default)`,
   `lower_quant_ext`, `lower_bf16_matmul_f32acc`, `fix_bool_sitofp`, `fix_bool_fptosi`,
   `scalarize_splat_inputs`, `require_initialized` (check), write via `to_text`], then the device
   group rewrite + `apply_post_offload_transform` (no-op without a callback), `_strip_provenance`
   (no-op), `apply_prepared_model_transform(None)`. Hoist / quant_round / reduce_vec / per-op blocks /
   panels / im2col / requant fuse / destination reuse are OFF on this path.
4. `lower_model`: `passes_xdsl._PREPROCESS` (prune, quant_ext, c_interface), optional
   layer-norm-chunked-sums (env/`--pass`, default off), generic print → `lower/model.mlir` (this is
   the D1 hand-off, `recipe.prepared_mlir`). Then **always** `ciface_inline.inline_wide_ciface_implementations`
   on LLVM IR (an unconditional Merlin LLVM-IR transform, missing from design §2 D3); `static_arena`
   default off.
5. The "mlir-opt" of D1/D2 is actually `torch_mlir`'s bundled MLIR PassManager run by
   `run_lowering.py` in the model2MLIR venv (`compiler_python()`), not the standalone mlir-opt. D9's
   exporter must match that MLIR revision, or the lead must decide which parser is authoritative.
6. BY_GROUP routing (`device_offload.rewrite_groups_to_device`) replaces group members directly with
   `call @<stem>_N` + a private declaration; no `outline.py` / `$kernel_i` function exists on this
   path, so C7a/C2 as written in §5.2/5.7 do not match. It rewrites the prepared file in place, prints
   with custom `to_text`, then text-patches declarations (`patch_declaration_arg_attrs`).
   It needs `device_dtype_triples(device)` (capability manifest or RTL facts) and `form_groups`.
7. Paper captures come from Phase 0 recipe captures (`sealed_m2m`, static/dynamic W8A8 recipe via the
   capture worker `--recipe`), not model2MLIR's plain `int8` (= weight-only). The exact recipe is
   operator-private; a public stand-in is `--dtype int8 --scheme int8_dyn_act_int8_weight`.
8. model2MLIR `workloads/capture.py` applies `capture.toml [env]`; tiny_llama's sets the forbidden
   `M2M_LLAMA_LAYERS=2`. Use `capture.py --_inner tiny_llama ...` with your own env instead.
   ResNet needs `M2M_RESNET_INPUT_NPZ` or `M2M_RESNET_RANDOM=1` (synthetic inputs, not paper-ready).

## Next steps
1. Create model2MLIR venv (`uv sync` + `torch-mlir` dev wheel); smolvla venv per its README.
2. Implement `src/merlin/verify/lean/stages.py` (env `MERLIN_LEAN_STAGES`; `record_boundary` +
   a chain/step helper printing generic xDSL via `xdsl_dialects._common.text(generic=True)`).
3. Add one-line step calls at the sites in findings 3–6; byte-neutrality test on a small fixture.
4. Capture `examples/workloads/coverage_mlp`, then the paper programs; run the driver; build inventory.

## Open questions for the lead
- Which MLIR parses the stages for Lean (torch_mlir wheel vs llvm-project build)?
- Which quantization recipe is the paper capture (public stand-in acceptable?)
- Device route without the OOT: may a stub capability manifest/selector be used for the inventory only?
