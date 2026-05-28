# Kernel Matching Methodology

How the kernel-embedding pipeline decides which IR ops in a model get
replaced by user-supplied C kernels. This is the architectural counterpart
to the user-facing how-tos under `docs/how_to/embed_custom_kernel_via_manifest.md`
and `docs/how_to/extend_kernel_coverage_to_any_model.md`.

The methodology answers four questions:

1. **What is a "match"?** — what counts as the same op for substitution.
2. **Where in the IREE pipeline does it happen?** — and what's the trade-off.
3. **How does a shape-agnostic kernel get its concrete shape values?** — push-constant
   propagation.
4. **How do we infer matchers automatically from a model?** — discovery
   inverts the matching problem.

## 1. The fundamental problem

A user model is MLIR. A kernel is a C function compiled to a `.o`. We need
to make those two things meet in a way that:

- The user only writes the kernel + a description of what it computes.
- The compiler decides where in the IR to substitute calls to that kernel.
- The same kernel covers every concrete shape variant of "what it computes".
- Coverage is verifiable — the compiler can tell you whether every op in
  the model went through a user kernel or fell back to IREE codegen.

The substitution unit is one **dispatch** in IREE's vocabulary — a chunk
of computation IREE would otherwise codegen for the target. We replace
the dispatch's body with `func.call @<our_kernel>` plus the wiring needed
to satisfy IREE's binding/workgroup conventions.

---

## 2. The four-step pipeline

```
┌──────────┐     ┌────────────┐     ┌──────────────┐     ┌─────────────┐
│ DECLARE  │ ──▶ │ SYNTHESIZE │ ──▶ │ MATCH        │ ──▶ │ REWRITE     │
│ manifest │     │ transform  │     │ over user IR │     │ flow.dispatch│
│ entry    │     │ spec       │     │              │     │ → executable │
└──────────┘     └────────────┘     └──────────────┘     └─────────────┘
   user            spec_gen.py        IREE preprocess     IREE preprocess
                                      pass interpreter    cast_and_call
```

### Step 1 — Declare

The user writes a manifest entry that names: the op pattern to match,
the kernel that replaces it, the shape values the kernel needs at
runtime, and which HW targets to build it for.

```json
{
  "name":         "saturnopu_matmul_f32",
  "source":       "abi/matmul_f32_workgroup.c",
  "entry_symbol": "matmul_f32_workgroup",
  "signature": {
    "operands": [...],
    "constants": [{"name": "M", "type": "i32", "from": {"input": 0, "dim": 0}}, ...],
    "output_dims": ["M", "N"]
  },
  "match": { "kind": "...", ... },
  "targets": ["llvm-cpu-spacemit-x60"]
}
```

The manifest is a **declarative contract**. No imperative code, no
compiler-internal references. This is the only thing the user touches
to add a kernel.

### Step 2 — Synthesize

`kernels/core/spec_gen.py` reads the manifest and emits a single
`transform_spec.mlir` containing five blocks per kernel:

| Block | Role |
|---|---|
| `hal.executable.source` | Carries the precompiled `.o` reference + the binding-subspan + call shim that IREE codegen would otherwise have produced |
| `util.func @call_<name>` | The dispatch wrapper — what the matched IR op gets substituted with |
| `transform.named_sequence @match_<name>` | The IR pattern to look for (next section) |
| `transform.named_sequence @cast_and_call_<name>` | The rewrite action |
| `@__transform_main` | Top-level driver: `foreach_match` across every `util.func` in the module |

The synthesis is **deterministic** — same manifest in, same spec out.
The user never edits the spec; they edit the manifest.

### Step 3 — Match

iree-compile loads the spec via `--iree-preprocessing-transform-spec-filename`
and runs the transform interpreter. For each `util.func` in the user's
program, the driver applies every (`match_*`, `cast_and_call_*`) pair.

The matcher is `transform.iree.match.cast_compatible_dag_from_root` —
it walks **upward** from a candidate root op through producer SSA edges,
checking that each op in the matched DAG corresponds to a like-shaped op
in the user's IR. "Like-shaped" is up to the matcher's annotations:

- Default: op name + result types + operand types must match exactly,
  plus all attributes.
- `{"match.operation_name_only"}` on an op: ignore everything but the op
  name (used on `tensor.empty` and `linalg.fill` so shape arguments don't
  have to match).

When a match succeeds, the matcher returns two SSA value packs: the
**inputs** of the matched DAG and the **output** value of its root.
Those become the arguments to the rewriter.

### Step 4 — Rewrite

`transform.util.cast_and_call %func(%ins) -> %out after %root`:

- Inserts a call to `%func` (our `util.func @call_<name>`) right after
  the matched root op.
- Binds the matched `%ins` to the call's parameters, inserting
  `tensor.cast` ops as needed when the wrapper's signature uses dynamic
  shapes and the payload IR has static ones (controlled by the
  `transform.type_conversion.tensor.cast_shape_dynamic_dims` modifier).
- Replaces uses of `%out` with the call's result.
- The matched ops become dead code; `transform.apply_dce` removes them.

After this step, the linalg op is gone. In its place is a
`util.call @call_<name>` that dispatches into our kernel. Subsequent
IREE phases (flow → stream → hal → vm) treat that call as opaque
foreign code; iree-lld links the precompiled `.o` into the final vmfb.

---

## 3. Two match kinds, two trade-offs

The methodology offers two ways to express "what op pattern to match",
each with different generality vs. structural-control trade-offs.

### 3.1 `linalg_dag` — full body match

The user authors a `match.mlir` body that the matcher matches verbatim:

```mlir
^bb0(%lhs: tensor<?xf32>, %rhs: tensor<?xf32>):
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %lhs, %c0 : tensor<?xf32>
  %empty = tensor.empty(%dim) {"match.operation_name_only"} : tensor<?xf32>
  %add = linalg.generic
      {indexing_maps = [...], iterator_types = ["parallel"]}
      ins(%lhs, %rhs : tensor<?xf32>, tensor<?xf32>)
      outs(%empty : tensor<?xf32>) {
    ^bb_inner(%a: f32, %b: f32, %_out: f32):
      %s = arith.addf %a, %b : f32
      linalg.yield %s : f32
  } -> tensor<?xf32>
```

Used when the op is `linalg.generic` (the body's arith chain is the
discriminator). The match captures op name + iterator types + indexing
maps + body op chain.

**Coverage rule:** any payload `linalg.generic` whose iterator types,
indexing maps, AND body op sequence match this body — at any concrete
tensor shape — is rewritten.

### 3.2 `named_op` — match by op name

```json
"match": {
  "kind": "named_op",
  "op_name": "linalg.matmul",
  "outs_from_input": -1,
  "op_attrs": ""
}
```

`spec_gen.py` synthesizes the match scaffold automatically: a
canonical `tensor.empty + linalg.fill + <named op>` chain for ops with
accumulator `outs(...)`, or just `tensor.empty + <named op>` for the
non-accumulator case. The user supplies only the op name.

**Coverage rule:** any payload op with the named identity — regardless
of shape, accumulator-init, or surrounding fill structure — is rewritten.

Two knobs handle the messy cases:

- `outs_from_input: N` — when N ≥ 0, input #N is bound to the matched
  op's `outs(...)` slot instead of synthesizing a fill scaffold. Used
  when the framework folds something else into the accumulator (dronet's
  conv has a broadcasted bias fed into `outs`).
- `op_attrs: "{...}"` — verbatim attribute string spliced into the
  matched op. Required for ops with structural attributes
  (`linalg.conv_2d_nchw_fchw` needs `dilations` + `strides`); without
  these the matcher rejects because attribute equality is enforced.

### Comparison

| | `linalg_dag` | `named_op` |
|---|---|---|
| User writes | match body MLIR (~15-30 lines) | one JSON field |
| Best for | `linalg.generic` (body-driven discrimination) | named linalg ops (matmul, conv, pool, broadcast, transpose, fill) |
| Generalizes over | shape variants of the same body | shape + attribute variants of the same op name |
| Fragility | Sensitive to canonicalizer changes (constant hoisting, etc.) | Robust — only the op name + declared attrs matter |

The two are interchangeable in the manifest; one kernel per entry.

---

## 4. Why dynamic shapes are the heart of the methodology

The kernel author writes ONE kernel for, say, `linalg.matmul`. The
model has matmuls at shapes `(8, 64, 16)`, `(128, 128, 128)`, `(1, 6272, 1)`.
We want one kernel to cover all three.

Mechanism: every operand in the manifest's `signature` is declared with
**dynamic** dims (`tensor<?x?xf32>` instead of `tensor<8x64xf32>`). The
synthesized matcher uses dynamic shapes too. The IREE matcher's
`cast_compatible_dag_from_root` is shape-permissive: a dynamic-shape
match accepts any concrete-shape payload.

When the rewrite fires, `transform.type_conversion.tensor.cast_shape_dynamic_dims`
inserts `tensor.cast` ops at the wrapper boundary so the static-shape
payload feeds the dynamic-shape wrapper, and the dynamic-shape result
gets cast back to whatever the downstream IR expected:

```mlir
%2 = flow.tensor.reshape %0 : tensor<8xf32> -> tensor<?xf32>{%c8}
%3 = flow.tensor.reshape %1 : tensor<8xf32> -> tensor<?xf32>{%c8}
%4 = util.call @call_saturnopu_add_f32(%2, %3)
     : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
%5 = flow.tensor.reshape %4 : tensor<?xf32>{%c8} -> tensor<8xf32>
```

This is what makes "8 kernels for all of dronet" feasible — each kernel
covers a whole family of shapes.

---

## 5. How concrete shape values reach the kernel

The kernel itself needs concrete dim values at runtime (M, K, N for
matmul; H_in, W_in, KH, KW, H_out, W_out for conv). Push constants are
the transport.

### The chain

```
manifest.signature.constants
        │
        ▼
spec_gen.py emits hal.pipeline.layout<constants = N, ...>
        │
        ▼
spec_gen.py emits util.func @call_<name>:
   %M = tensor.dim %in0, %c0 : tensor<?x?xf32>     ← derive from input
   %M_i32 = arith.index_cast %M : index to i32
   ...
   flow.dispatch ...[<workload>](%M_i32, %K_i32, %N_i32, %in0, %in1)
        │                      └──── push constants leading the dispatch
        ▼
hal.executable.source's inner builtin.module:
   func.func @<export_name>() {
     %M_i32 = hal.interface.constant.load layout(...) ordinal(0) : i32
     %M = arith.index_cast %M_i32 : i32 to index
     ...
     func.call @<entry_symbol>(<bindings>, %M, %K, %N, %tid)
   }
        │
        ▼
   C kernel: void <entry>(... size_t M, size_t K, size_t N, size_t tid)
```

### Aliases

When the same logical dim appears in multiple inputs (matmul's
K = `lhs.dim(1)` = `rhs.dim(1)` for transposed-B form), the manifest
declares it once with `aliases`:

```json
{"name": "K", "type": "i32",
 "from":    {"input": 0, "dim": 1},
 "aliases": [{"input": 1, "dim": 1}]}
```

`spec_gen.py` builds a lookup table `(input_idx, dim_idx) → constant_name`
that includes both the source and the aliases. This is how the wrapper
can annotate `tensor<?x?xf32>{%K, ...}` for input 1 even though K's
"primary" source is input 0.

### `output_dims`

When the output's dynamic dims aren't trivially `input[0].dim(k)` (matmul
output dim 1 = N comes from input 1, not input 0), the manifest declares
the mapping:

```json
"output_dims": ["M", "N"]
```

`spec_gen.py` uses this to:
- Annotate the output tensor type with the right constants:
  `tensor<?x?xf32>{%M, %N}`.
- Compute the workload as the product of output dims:
  `%workload = arith.muli %M, %N : index`.

### Workload = total output elements

The synthesized executable uses `count(%workload) -> (%workload, 1, 1)`.
This means **one workgroup per output element**. The kernel sees
`tid = workgroup.id[0]` and decodes it into per-axis indices internally
(`m = tid / N; n = tid % N` for matmul). One workgroup-per-element
is suboptimal at scale (you'd want a tile per workgroup), but it makes
the C kernel signature uniform and is what `count(workload)` produces
without further plumbing.

---

## 6. The rewrite mechanically

```mlir
// BEFORE rewrite (preprocessing-phase IR)
func.func @main(%lhs: tensor<8xf32>, %rhs: tensor<8xf32>) -> tensor<8xf32> {
  %empty = tensor.empty() : tensor<8xf32>
  %sum = linalg.generic
      {iterator_types = ["parallel"], indexing_maps = [...]}
      ins(%lhs, %rhs : tensor<8xf32>, tensor<8xf32>)
      outs(%empty : tensor<8xf32>) {
    ^bb_inner(%a: f32, %b: f32, %_: f32):
      %s = arith.addf %a, %b : f32
      linalg.yield %s : f32
  } -> tensor<8xf32>
  return %sum : tensor<8xf32>
}

// AFTER rewrite — the linalg.generic is gone
func.func @main(%lhs: tensor<8xf32>, %rhs: tensor<8xf32>) -> tensor<8xf32> {
  %0 = flow.tensor.reshape %lhs : tensor<8xf32> -> tensor<?xf32>{%c8}
  %1 = flow.tensor.reshape %rhs : tensor<8xf32> -> tensor<?xf32>{%c8}
  %2 = util.call @call_saturnopu_add_f32(%0, %1)
       : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %3 = flow.tensor.reshape %2 : tensor<?xf32>{%c8} -> tensor<8xf32>
  return %3 : tensor<8xf32>
}
```

The `util.call` is a normal MLIR call. It survives flow → stream → hal
lowering. By the HAL phase, the `call_saturnopu_add_f32` body has been
inlined into the dispatch site as a `flow.dispatch @kb_*::@*(...)` call
into the `hal.executable.source` we declared in the spec; the executable
carries the `.o`; iree-lld links it.

---

## 7. The inverse problem: discovery

The user shouldn't have to know what ops their model contains. Discovery
walks the model IR and emits a manifest stub for every op pattern it sees:

```python
# Pseudo-code for kernels/core/discover.py

def discover(model_mlir, target, hw):
    # 1. Compile to preprocessing phase WITHOUT kernels — get canonical IR
    ir = run_iree_compile(model_mlir, target, hw, compile_to="preprocessing")

    # 2. Walk every linalg op
    discoveries = []
    for op in find_named_linalg_ops(ir):       # matmul, conv, pool, broadcast, fill
        discoveries.append((op.name, op.signature, occurrence_count))
    for op in find_linalg_generic(ir):
        body_label = classify_body(op.body)    # rsqrt | addf | mulf | relu | ...
        discoveries.append((op.iterator_types, body_label, op.signature))

    # 3. Group by (op_name, body_label) — dedupe shape variants
    by_kernel = group_by(lambda d: (d.op_name, d.body_label))

    # 4. For each unique (op_name, body_label):
    for key, occurrences in by_kernel.items():
        if recognized(key):
            emit_manifest_entry_with_complete_C_body(key, occurrences)
        else:
            emit_stub_entry(key, occurrences)   # parked under stubs/

    # 5. Print impact-ranked list (occurrences × output element count)
    print_minimum_cover(by_kernel)
```

The methodology in one sentence: **discover** computes the **inverse** of
match — given an observed dispatch population, what kernel signatures
would cover it? Set-cover analysis (`--minimum-cover`) tells the user
the smallest implementing set.

### Recognized body classes (auto-emit)

`kernels/core/discover.py:_BODY_RECOGNIZERS` is a regex table that
matches body op chains and emits a complete C kernel + match.mlir for
each. Currently 13 patterns: `rsqrt`, `sqrt`, `exp`, `log`, `absf`,
`addf`, `subf`, `mulf`, `divf`, `maxf`, `minf`, `negf`, `identity`,
plus `relu` (C body emits, match.mlir is a stub due to the constant-
hoist quirk discussed in `kernel_embedding_walkthrough.md`).

### Coverage prioritization

`--minimum-cover` ranks the discovered (op_name, body_label) signatures
by `occurrences × output_elements` (a proxy for compute coverage) and
prints them as a greedy-cover ladder:

```
   #   cov%   cum_disp  shapes  signature
   1  41.4%     10/53        9  linalg.generic#unknown#parallel*6     ← im2col conv
   2  62.3%     22/53        5  linalg.matmul
   3  72.9%     28/53        3  linalg.generic#mulf#parallel*4        ← BN scale
   ...
   9 100.0%    48/53        1  linalg.generic#relu#parallel_parallel
  ──→ 9 kernels = 100% coverage of dronet's compute
```

The user knows: implementing the top 3 covers ~75% of compute. The full
9 covers everything. The ranking is the methodology's "prioritize
implementation effort" output.

---

## 8. Phase choice — where matching runs in the IREE pipeline

The `--iree-preprocessing-transform-spec-filename=` flag drops the spec
into IREE's **preprocessing** phase (phase 3 in the dump numbering). This
is one of three architecturally-feasible insertion points:

| Phase | What the IR looks like | Match scope | Trade-off |
|---|---|---|---|
| **Preprocessing (3)** | Each linalg op separate. `linalg.matmul` is `linalg.matmul`; `linalg.generic` bodies are unfused single-arith chains. | Per-op. One kernel = one op family. | Stable across IREE versions. Many small dispatches per chain. **CURRENT DEFAULT.** |
| **Flow (6)** | Inside each `flow.dispatch.workgroups` body, IREE has fused multiple ops (BN+ReLU as one 5-op body, conv+bias as one 2-op body). | Per-fused-dispatch. One kernel = one fused chain. | Bigger kernels do more work per workgroup; less DRAM bandwidth. Brittle: depends on IREE fusion heuristics. |
| **HAL (11)** | Each `hal.executable.variant` is a fully-lowered `func.func` (memref + arith + vector). | Per-executable. Replace whole `func.func` body. | Most invasive; closest to how `iree_uk_mmt4d` is wired. Bypasses IREE codegen entirely for replaced ops. |

`kernels/core/discover.py --auto-fuse` walks phase 6 and reports the
fused dispatch inventory (without auto-emitting matchers — that's the
known follow-on). The methodology supports all three phases in
principle; the implementation today exercises preprocessing.

---

## 9. End-to-end summary

The kernel-matching methodology is:

1. **Declarative manifest.** User describes the kernel + the op pattern;
   no compiler-internal references.
2. **Deterministic spec synthesis.** `spec_gen.py` emits a transform-
   dialect spec from the manifest; the spec is human-readable and
   inspectable.
3. **IREE-native pattern matching.** The spec uses standard IREE
   transform-dialect ops (`cast_compatible_dag_from_root`,
   `cast_and_call`, `foreach_match`); we add no new compiler passes.
4. **Dynamic-shape generalization.** One matcher covers every concrete
   shape variant of an op family; static-payload-to-dynamic-wrapper
   bridging is automatic via `cast_shape_dynamic_dims`.
5. **Push-constant shape propagation.** Manifest declares which dims
   the kernel needs; spec_gen wires `tensor.dim → arith.index_cast →
   flow.dispatch operand → hal.interface.constant.load → kernel arg`.
6. **Inverse problem solved by discovery.** Walks the model's
   preprocessing-phase IR, classifies linalg ops, emits manifest stubs
   ranked by impact.
7. **Verifiable coverage.** `--kernels-strict-coverage` audits phase 5
   post-compile and fails if any dispatch went through IREE codegen
   instead of a user kernel.
8. **Opt-in selection.** Manifest's `select` field controls which
   subset of the catalog enters this compile; default = all enabled.

The contribution is the **methodology**, not the kernels. Adding a new
kernel for a new HW target is a manifest entry + a `.c` file; the rest
is generated.

---

## Cross references

- `kernels/core/manifest.py` — schema enforcement
- `kernels/core/spec_gen.py` — spec synthesis (steps 2 above)
- `kernels/core/discover.py` — inverse problem (step 7)
- `tools/compile/cli.py` — wiring into iree-compile
- `docs/how_to/kernel_embedding_walkthrough.md` — concrete MLIR
  snippets at every phase
- `docs/how_to/extend_kernel_coverage_to_any_model.md` — user recipe
- `docs/dev_blog/2026-04-29-kernel-embedding-status-and-demo-guide.md` —
  what's actually wired vs documented-only
