# Triton → an accelerator that is not a GPU

Write an ordinary Triton kernel. Compile it for a weight-stationary systolic array. Certify the result
on that array's own RTL. Nothing in this example is GPU-adjacent, and nothing in the compiler learns
the accelerator's name.

Everything up to and including a numerically checked command buffer needs **only a fresh clone and its
venv** — no toolchain, no simulator, no board. The accelerator itself is tracked in the repo, so the
interesting half of this example costs seconds.

```bash
cd examples/triton
./run.sh preflight          # what this machine can run, and what would fix the rest
./run.sh walk --pause       # every stage between kernel and hardware, explained  (seconds)
./run.sh compile            # the one command you would actually run              (seconds)
./run.sh route              # why an elementwise kernel takes a DIFFERENT route   (seconds)
./run.sh compare            # the same kernels against two accelerators           (seconds)
./run.sh converge           # proof no Triton-specific compiler grew beside ours  (seconds)
./run.sh certify            # the same command buffer on real RTL                 (~3 min, needs Verilator)
```

`--dry-run` prints every command without running it, so this example is readable as documentation on a
machine with nothing installed.

## Set up from a fresh clone

```bash
git clone https://github.com/ucb-bar/merlin.git && cd merlin
uv sync --all-extras                    # creates .venv; plain `python` is not on PATH here
uv pip install -e '.[triton]'           # the frontend is an optional extra
./examples/triton/run.sh preflight
```

That is the whole setup for stages `walk` through `converge`. `certify` additionally needs a Gemmini
Verilator build in a chipyard checkout; `preflight` names the variable if it is missing. The version pin
is exact and checked up front — `merlin.triton` drives Triton's *compiler-internal* frontend
(`ASTSource`/`make_ir`), which is not a stable API, so a drifted Triton would either crash deep inside
the frontend or silently emit different TTIR.

## The kernel

Stock Triton. No Merlin import, no decorator of ours, no target hints — this file would run on a GPU:

```python
@triton.jit
def repeated_rhs_matmul(a0_ptr, a1_ptr, w_ptr, c0_ptr, c1_ptr,
                        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    offs_m = tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    w  = tl.load(w_ptr  + offs_k[:, None] * BN + offs_n[None, :])
    a0 = tl.load(a0_ptr + offs_m[:, None] * BK + offs_k[None, :])
    a1 = tl.load(a1_ptr + offs_m[:, None] * BK + offs_k[None, :])
    out = offs_m[:, None] * BN + offs_n[None, :]
    tl.store(c0_ptr + out, tl.dot(a0, w, out_dtype=tl.int32))
    tl.store(c1_ptr + out, tl.dot(a1, w, out_dtype=tl.int32))
```

**Two activations against one shared weight**, deliberately. A weight-stationary array earns its keep by
keeping the weight resident, so a lone matmul would never exercise the thing that makes this hardware
worth compiling for. The shared `w_ptr` is what makes residency *inferrable*.

## What you have to declare, and why none of it is guessed

A Triton kernel is not self-describing: its parameters are untyped, its pointers carry no shape, its
grid lives at the call site, and which buffers it writes is visible only inside the body.

```bash
./run.sh compile     # runs exactly this:
merlin-compile-kernel examples/triton/matmul_simple.py:repeated_rhs_matmul \
    --target-package out/artifacts/targets/gemmini/hand_v0 \
    --arg 'a0_ptr=*i8:16x32:read'  --arg 'a1_ptr=*i8:16x32:read' \
    --arg 'w_ptr=*i8:32x16:read' \
    --arg 'c0_ptr=*i32:16x16:write' --arg 'c1_ptr=*i32:16x16:write' \
    --constexpr BM=16 --constexpr BN=16 --constexpr BK=32 --grid 1 --emit all
```

| flag | states |
| --- | --- |
| `--arg NAME=*DTYPE:SHAPE:EFFECT` | a pointer: element type, **static** shape, `read`/`write`/`readwrite` |
| `--arg NAME=DTYPE` | a scalar parameter |
| `--constexpr NAME=VALUE` | a `tl.constexpr` |
| `--assume NAME=VALUE` | the compile-time value of a *runtime* scalar |
| `--grid X[,Y[,Z]]` | the SPMD launch grid |

A wrong guess here is a **miscompile, not an error**, which is why each one is asked for:

- **Effects are declared, then cross-checked** against what the kernel actually does. A kernel that
  mutates a buffer the caller believes is read-only is a miscompile, so a disagreement in *either*
  direction is rejected.
- **A pointer with no static shape is refused outright.** There is deliberately no dynamic-shape
  fallback, because a fallback here silently papers over the missing fact.
- **`--assume` is the one that surprises people.** In `vector_add` the extent arrives as a runtime
  scalar, so nothing in the kernel says `n_elements` equals the declared shape — and without that the
  compiler cannot prove the mask keeps the launch inside the tensor. It refuses rather than assuming.

## Connecting it to an accelerator

This is the part worth understanding, and it is one argument: `--target-package <dir>`.

The accelerator arrives as a **directory**, loaded at run time. Core does not import it, has no table of
supported targets, and never learns its name:

```
out/artifacts/targets/gemmini/hand_v0/     # tracked — present in a fresh clone
├── manifest.yaml                   what this package is, and how honest its status claim is
├── contracts/
│   ├── target_contract.yaml        capacities and features it declares
│   └── dialect_plan.yaml           the ops it declares + what it can materialize
├── dialect.py                      those ops as a real xDSL dialect
├── lowering.yaml                   neutral interface op -> its op -> its command opcode
└── inputs/rtl_facts.yaml           the RTL-derived facts that CONSTRAIN the above
```

`lowering.yaml` is the entire connection, and it is data rather than code:

```yaml
interface_to_target:               target_to_opcode:
  interface.resident_pack:  gemmini.pack        gemmini.pack:    RES_PACK
  interface.matmul:         gemmini.matmul      gemmini.matmul:  MATMUL_RESIDENT
  interface.commit:         gemmini.commit      gemmini.commit:  COMMIT
  interface.resident_evict: gemmini.release     gemmini.release: EVICT
```

To compile the same kernel for different hardware, point `--package` somewhere else. That is the test of
whether this design is real, so the example takes the flag everywhere:

```bash
./run.sh walk --package out/artifacts/targets/radiance/hand_v0    # a SIMT machine instead
```

Same kernel, same declarations, and the identical descent ends in a **different dialect**:

```
gemmini    interface.resident_pack -> gemmini.pack    radiance   interface.resident_pack -> radiance.stage
           interface.matmul        -> gemmini.matmul             interface.matmul        -> radiance.matmul
           interface.commit        -> gemmini.commit             interface.elementwise   -> radiance.elementwise
           interface.resident_evict-> gemmini.release            interface.commit        -> radiance.commit
                                                                 interface.resident_evict-> radiance.release
```

Writing such a package is [`docs/guides/adding_a_target.md`](../../docs/guides/adding_a_target.md).

## The route is chosen by the payload, not the target

`./run.sh route` compiles a **vector add** against the same accelerator and gets a different answer:

```
route: llvm — target 'gemmini' cannot materialize elementwise through the staged pipeline
  (declared coverage: ['commit', 'matmul', 'resident_evict', 'resident_pack'];
   interface-buildable: ['matmul']) — compiling as generic computation
```

This is the design working. "Accelerator ⇒ accelerator pipeline" is the wrong model: a vector add has no
matmul, so it compiles as generic computation *even on an accelerator*, and a matmul takes the staged
descent *even on a CPU-like target that covers it*. The decision reads the intersection of the payload
with what the package's own dialect plan claims — never the target's name. An unreadable plan **fails
closed**, because routing cannot distinguish "accelerates nothing" from "the plan is somewhere else".

`./run.sh compare` is the clearest single demonstration in this example — two kernels × two accelerators,
one compiler:

|  | matmul | vector add |
| --- | --- | --- |
| **gemmini** | `staged` — materializes matmul through its own dialect | `llvm` — cannot materialize elementwise |
| **radiance** | `staged` — materializes matmul through its own dialect | `staged` — materializes elementwise |

The vector add is accelerated on one and not the other. Nothing about that was keyed on a name: radiance
declares `interface.elementwise` in its lowering table and gemmini does not, so **its own declaration
decided its own route.** Add that line to a package and the route changes with no compiler edit.

## What `./run.sh walk` shows you

The pipeline is a sequence of *named decisions*, not numbered passes:

| stage | decides |
| --- | --- |
| `input` | what to compute — the frontend's linalg, unchanged |
| `contract` | what must be **true** for the accelerator path to be legal, plus the proof |
| `schedule` | placement, layout, liveness, dispatch grouping |
| `interface` | target-**neutral** accelerator vocabulary: pack / matmul / commit / evict |
| `target` | this package's own dialect, via its lowering table |
| `runtime` | device acquisition, command-buffer construction, submit, wait |

Read the middle two as an argument. The contract *proves* the weight is immutable and fits in resident
storage before anything relies on it:

```mlir
"contract.assume"(%2) <{kind = "immutable", ...}>
"contract.fact"(%2)   <{role = #contract<memory_role reusable_weight>, reuse_count = 2 : i64}>
%4 = "contract.prove"(%2) <{requirement = "rhs_immutable"}>
%5 = "contract.prove"(%2) <{requirement = "capacity_fit"}>
```

and the command buffer that falls out **infers residency** rather than being told:

```
0  RES_PACK         src=W dst=W_res        <- packed once
1  MATMUL_RESIDENT  lhs=A0 rhs=W_res dst=acc0
2  COMMIT           src=acc0 dst=Y0
3  MATMUL_RESIDENT  lhs=A1 rhs=W_res dst=acc1   <- reused, not re-packed
4  COMMIT           src=acc1 dst=Y1
5  EVICT            handle=W_res
```

Give each matmul its own weight and that inference correctly disappears. Nothing in the kernel requested
any of it.

## Is it right?

Three levels, and the example keeps them apart on purpose, because conflating them is how a functional
model gets reported as a hardware result.

| level | what it proves | needs |
| --- | --- | --- |
| **L0** `./run.sh walk` | the command buffer computes `A @ W`, checked against numpy | nothing |
| **L1** `./run.sh certify` | it runs on a functional simulator (`derived_from_rtl: false`) | spike-gemmini |
| **L2** `./run.sh certify` | **the real RTL computes it bit for bit** (`derived_from_rtl: true`) | Verilator build |

L0 compares twice — the command-buffer interpreter against the reference outputs, *and* every `COMMIT`
against a matmul computed in numpy — because one comparison would only prove Merlin agrees with itself.

The L1/L2 distinction is **asserted, not documented**: the L1 test requires `derived_from_rtl` to be
`False` and L2 requires it to be `True`, so a functional-model number cannot be passed off as silicon.

`--verify` on the CLI is **MLIR structural verification** of each stage module. It is not a numerical
check — use `walk` or `certify` for that.

## The guard that correctness cannot give you

`./run.sh converge` compares the Triton descent against the *same workload hand-written in linalg*,
stage by stage. If a second, Triton-specific lowering stack were quietly growing beside the main one,
every numerical test above would still pass. Only identity catches it.

The measured result is stronger than "equivalent": the printed modules are **byte-identical, SSA
numbering included, with no canonicalization step**. That only holds because the bridge emits
`linalg.quantized_matmul` with zero points — Merlin's own idiom — rather than an equivalent spelling of
its own.

## What this does not do yet

Stated plainly, because the gap is architectural rather than a missing feature.

**Your schedule is currently discarded.** The bridge normalizes the grid away so the parallelism
decision stays unmade — which is what lets a systolic array sequentialize a kernel that a SIMT target
maps to warps, from identical input. But normalizing and *destroying* got collapsed into one step:

- `--num-warps` / `--num-stages` are recorded as provenance and never interpreted. That part is right:
  they are CUDA scheduling knobs, and treating them as portable target semantics would be a lie.
- `BLOCK_*` / `GROUP_*` currently reach **nothing** downstream. That part is not right. `BM=16` is a
  statement about the *iteration space*, which is portable; only its binding to warps is GPU-specific.
- A multi-program `tl.dot` (a `k`-loop accumulating over tiles) is **refused**, because a tiled
  accumulation has nowhere to be expressed — the `schedule` stage has no vocabulary for decomposition
  and no channel to receive one.

Those are three symptoms of one missing abstraction, not three independent limits. The intended fix — a
target-independent *schedule proposal* with `derive` / `seed` / `honor` policies, where `honor` fails
closed naming the derived fact that refused a tile size — is written up in
[`docs/design/triton_frontend.md`](../../docs/design/triton_frontend.md). Until it lands, treat this
frontend as excellent at *semantics* portability and silent on *schedule* portability.

## See also

- [`docs/guides/triton_kernels.md`](../../docs/guides/triton_kernels.md) — the author-facing reference
- [`docs/design/triton_frontend.md`](../../docs/design/triton_frontend.md) — why linalg-on-tensors is
  the convergence point, the invariants, and the designs that were tried and rejected
- [`docs/guides/adding_a_target.md`](../../docs/guides/adding_a_target.md) — write your own package
- [`../README.md`](../README.md) — the whole-model examples, for a tapeout you cannot attach to
