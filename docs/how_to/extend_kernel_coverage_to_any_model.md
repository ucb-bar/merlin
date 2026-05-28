# Extend Kernel Coverage To Any Model

When you want **every** linalg op in a model to be replaced by one of your
own kernels — no IREE codegen fallback — the workflow is three steps:

1. **Discover** the ops your model contains.
2. **Author** a manifest entry per op family (most are one-liners thanks to
   named-op matching).
3. **Verify** coverage with `--kernels-strict-coverage`, which fails the
   compile if anything slipped through.

The same workflow scales to any (model, hw target) pair: discovery walks
whatever IR the preprocessing phase produces, named-op matching is shape-
agnostic so you don't rewrite the manifest for every input variant, and the
target list in each manifest entry gates per-hw object compilation.

## 1. Discovery — bootstrap the manifest

```bash
conda run -n merlin-dev uv run python -m tools.kernels.discover \
    models/dronet/dronet.mlir \
    --target spacemit_x60 --hw RVV \
    --output benchmarks/SaturnOPU/kernels \
    --target-key llvm-cpu-spacemit-x60 \
    --iree-compile-arg='--iree-opt-data-tiling=false'
```

The tool runs `./merlin compile --compile-to=preprocessing` (so you see
the same IR `iree-compile` will run after the rewrite phase), parses the
linalg ops, and emits one stub per unique `(op_name, signature)` it sees:

```
Discoveries:
    9x  linalg.conv_2d_nchw_fchw  (2 ins) -> tensor<1x...xf32>
    2x  linalg.matmul             (2 ins) -> tensor<1x1xf32>
    1x  linalg.pooling_nchw_max   (2 ins) -> tensor<1x32x55x55xf32>
    4x  linalg.generic#parallel_parallel_parallel_parallel  (2 ins) -> ...
    ...
```

Output:

```
<output-dir>/
├── manifest.json                                 — appended (existing entries preserved)
├── abi/discovered_<op>_workgroup.c               — TODO stubs you fill in
└── match/discovered_<op>.match.mlir              — only for non-named ops (linalg.generic)
```

Named ops (`linalg.matmul`, `linalg.conv_2d_*`, `linalg.pooling_*`,
`linalg.broadcast`, `linalg.transpose`) get `match.kind: "named_op"`
automatically — no `match.mlir` file needed. `linalg.generic` ops require
a body-level match scaffold; the tool emits a placeholder.

Re-running discovery against the same `--output` directory **never
overwrites existing entries** (skips `name` collisions) — so once you've
filled in a kernel, running discovery on a new model only adds the new
op patterns.

## 2. Authoring with named-op matching

For a linalg named op the manifest entry is now four short blocks:

```json
{
  "name":        "saturnopu_conv_2d_nchw_fchw_f32",
  "source":      "abi/conv_2d_nchw_fchw_workgroup.c",
  "source_lang": "c",
  "entry_symbol": "conv_2d_nchw_fchw_workgroup",
  "signature": {
    "operands": [
      {"role": "in",  "tensor": "tensor<?x?x?x?xf32>"},
      {"role": "in",  "tensor": "tensor<?x?x?x?xf32>"},
      {"role": "in",  "tensor": "tensor<?x?x?x?xf32>"},
      {"role": "out", "tensor": "tensor<?x?x?x?xf32>"}
    ],
    "constants": [
      {"name": "N", "type": "i32", "from": {"input": 0, "dim": 0}, "aliases": [{"input": 2, "dim": 0}]},
      ...
    ],
    "output_dims": ["N", "F", "H_out", "W_out"]
  },
  "match": {
    "kind": "named_op",
    "op_name": "linalg.conv_2d_nchw_fchw",
    "outs_from_input": 2,
    "op_attrs": "{dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}"
  },
  "targets": ["llvm-cpu-spacemit-x60"]
}
```

Key fields, all optional but used here:

- **`outs_from_input: N`** — the matched op's `outs(...)` is bound to the
  Nth input operand instead of being synthesized via a `linalg.fill`
  scaffold. Use this for ops where the framework folds something else into
  the accumulator (dronet feeds a broadcasted bias into conv's `outs`).
- **`op_attrs: "..."`** — verbatim attribute string for ops with required
  attrs. `cast_compatible_dag_from_root` matches attrs by exact equality,
  so conv/pool need their `dilations/strides` declared.
- **`signature.constants`** — runtime dim values passed as i32 push
  constants. `aliases` lists additional `(input, dim)` positions that
  equal this constant (e.g. matmul's K = lhs[1] = rhs[0]).
- **`signature.output_dims`** — explicit output-dim → constant map. Needed
  whenever the output's dynamic dims aren't trivially `input[0].dim(k)`.

For `linalg.generic` ops (custom bodies — BN, ReLU, residuals, requant,
etc.), use `match.kind: "linalg_dag"` with a hand-written `.match.mlir`
that includes the body's iterator types + arith ops.

## 3. Strict coverage verification

```bash
./merlin compile models/dronet/dronet.mlir \
  --target spacemit_x60 --hw RVV \
  --kernels-dir benchmarks/SaturnOPU/kernels \
  --kernels-strict-coverage \
  --output-dir build/dronet_strict/ \
  --iree-compile-arg='--iree-opt-data-tiling=false'
```

After the compile (success or failure), the audit walks
`<out>/phases/*.5.dispatch-creation.mlir` and counts `flow.dispatch.workgroups`
blocks (one per dispatch that fell through to IREE codegen). If any survived,
the build fails with a structured breakdown:

```
❌ --kernels-strict-coverage: dispatches survived past kernel rewrite
   (these went through IREE codegen, not your kernels):
        15x  linalg.generic
         3x  linalg.fill
   Inspect build/dronet_strict/phases/dronet.5.dispatch-creation.mlir and
   add matching manifest entries (or run `python -m tools.kernels.discover`
   to auto-generate stubs).
```

When coverage is 100%:

```
  ✅ kernels-strict-coverage: 0 unmatched dispatches (100% kernel coverage)
```

## Multi-target story

A single manifest extends across hardware targets just by adding entries to
the `targets` list:

```json
"targets": ["llvm-cpu-spacemit-x60", "llvm-cpu-aarch64", "llvm-cpu-x86_64"]
```

`kernels/core/precompile.py` builds one `.o` per target via clang
(`_CPU_TARGET_FLAGS` defines the per-target arch flags). At compile time,
the right object is selected for the host triple.

To add a brand-new HW target:

1. Add an entry to `kernels/core/precompile.py:_CPU_TARGET_FLAGS` —
   `(target_triple, [arch flags])`.
2. Add an entry to `kernels/core/spec_gen.py:_HAL_TARGET_ATTR` — the
   `#hal.executable.target<...>` template.
3. List the new key in your manifest's `targets`.

## Putting it together

The end-to-end flow for any new model on any HW:

```bash
# 1. Discover ops + emit stubs.
python -m tools.kernels.discover models/foo.mlir \
    --target T --hw HW --output kernels/

# 2. Fill in the C bodies — `kernels/abi/*.c` files have `// TODO`
# markers. Named ops only need C; generic ops also need match.mlir.

# 3. Verify completeness on every compile.
./merlin compile models/foo.mlir --target T --hw HW \
    --kernels-dir kernels/ --kernels-strict-coverage \
    --output-dir build/foo/
```

If step 3 fails, step 1 tells you exactly which patterns are missing.

## Where to look in the source

- `kernels/core/manifest.py` — schema (named_op, outs_from_input, op_attrs,
  signature.constants/aliases/output_dims).
- `kernels/core/spec_gen.py` — `_named_op_match_body` synthesizes the
  match scaffold; `_inner_module_for_c` emits the IREE wrapper.
- `kernels/core/discover.py` — preprocessing-phase IR walker + stub
  generator.
- `tools/compile/cli.pykernels-strict-coverage` — phase-5 audit pass.
- `benchmarks/SaturnOPU/kernels/manifest.json` — worked examples mixing
  named-op (matmul, conv, pool) and linalg-dag (add, linear, bias_add).
- `benchmarks/SaturnOPU/kernels/phase_dumps/dronet_partial/` — what a
  partial-coverage dronet compile looks like in IR.

## Selecting which kernels run, and finding the minimum set

### Manifest `select` — explicit opt-in per compile

The default is "every kernel listed in `manifest.json` is enabled". For a
more conservative workflow, add a top-level `select` array:

```json
{
  "schema_version": 1,
  "select": ["saturnopu_matmul_f32", "saturnopu_pooling_nchw_max_f32"],
  "kernels": [ ... ]
}
```

Only the listed names get pre-compiled and wired into the auto-spec for
this compile. The rest stay in the catalog but inert. Lets you keep a
50-kernel library and enable 5 for today's run without editing the
catalog. Compile output prints `🧬 select: 5 of 50 kernels enabled`.

### `--minimum-cover` — find the smallest covering set

```bash
python -m tools.kernels.discover models/<model>.mlir \
    --target T --hw HW --output kernels/ --minimum-cover
```

Greedy set-cover on the discovery output. Each row is one author-unit
kernel (dynamic-shape matcher), with cumulative compute % covered as
you add kernels in priority order. Example for dronet:

```
   #   cov%   cum_disp  shapes  signature
   1  35.7%     10/43        4  linalg.conv_2d_nchw_fchw
   2  53.7%     16/43        3  linalg.generic#mulf#parallel_parallel_parallel_parallel
   3  65.7%     23/43        5  linalg.generic#addf#parallel_parallel_parallel_parallel
   4  77.6%     29/43        4  linalg.generic#relu#parallel_parallel_parallel_parallel
   5  86.6%     32/43        3  linalg.generic#subf#parallel_parallel_parallel_parallel
   6  93.1%     34/43        2  linalg.fill
   7  99.6%     35/43        1  linalg.pooling_nchw_max
   8  100.0%    36/43        1  linalg.generic#relu#parallel_parallel
  ──→ 8 kernels = 100% coverage of dronet's compute
```

Implementing those 8 kernels covers 100% of dronet's compute regardless
of shape — each row is one author unit that handles all observed shape
variants. Pair with `select` to enable exactly that subset for the
compile.
