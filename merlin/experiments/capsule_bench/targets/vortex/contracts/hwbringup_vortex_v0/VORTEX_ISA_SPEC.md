# Vortex hardware bring-up reference

You are bringing up a compiler backend for **Vortex**, a programmable RISC-V SIMT GPGPU. You have:

- `rtl/` — the Vortex SystemVerilog RTL, and a **HW-dialect MLIR import** of the elaborated design.
  The hardware is the ground truth; this document is a reading of it.
- this spec sheet — the architecture/ISA facts a code generator needs.

You do **not** get Vortex's software stack (intrinsics headers, spawn runtime, LLVM fork, PoCL,
chipStar, bundled kernels). Everything needed to emit correct code is below.

---

## 1. Machine model

A Vortex core executes **warps** of **threads** in lockstep. Threads within a warp share a program
counter; per-thread participation is controlled by a **thread mask**. Multiple cores are grouped into
sockets and clusters.

Work is distributed by hardware. The **KMU (Kernel Management Unit)** launches every
`(block, thread)` coordinate of the grid at the kernel's entry PC — there is no software spawn loop to
write. Each launched coordinate reads its own identity from CSRs (§4).

### Frozen geometry for this benchmark

Every capsule is compiled and graded against one machine:

| | |
|---|---|
| clusters | 2 |
| cores per cluster | 2 (**4 cores total**) |
| socket size | 2 (⇒ 1 socket per cluster) |
| warps per core | 8 |
| threads per warp | 8 |
| **total threads** | **256** |
| L2 cache | enabled |
| XLEN | 64 |
| FPU | `STD` (pure-RTL IEEE, fp32 and fp64) |

Note `NUM_CORES` is **per cluster**, and `NUM_SOCKETS = NUM_CORES / SOCKET_SIZE`.

This is a genuinely multi-core machine, so work may be spread across cores that do **not** share an L1
data cache. The L2 is the coherence point; see §5 for what that means for ordering.

Do not assume these numbers at compile time where a CSR can tell you at run time (§4 exposes both the
per-launch geometry and the machine geometry).

---

## 2. Base ISA

Standard **RV64IMAFD** (`-march=rv64imafd -mabi=lp64d`), plus the `Zicond` extension and the CUSTOM0
extension in §3. Integer and IEEE-754 float behave exactly as the RISC-V spec says. `FLEN` is 64, so
both single- and double-precision are hardware operations; the capsule numerics in this benchmark are
fp32 and int, but a `double` in your lowering will not trap.

Pointers and `long` are 64-bit. The kernel-argument block (§6) holds 64-bit device addresses, and
`XLEN`-sized CSR reads return 64-bit values even where the field is narrower.

The CUSTOM0 instructions below are ordinary R-type encodings, so a **stock, unmodified LLVM** can
assemble them via inline `.insn` — no compiler fork is required or permitted.

---

## 3. The CUSTOM0 SIMT extension

Opcode **`0x0B`** (RISC-V *custom-0*), R-type. Assembler form:

```
.insn r 0x0B, <funct3>, <funct7>, <rd>, <rs1>, <rs2>
```

**`funct7` selects between two families**: `0` = warp/thread control, `1` = cooperative-thread
primitives. `funct3` then selects the operation within the family. Both dimensions matter — `funct3 = 0`
means `tmc` at `funct7 = 0` but `vote_all` at `funct7 = 1`.

### 3.1 `funct7 = 0` — warp and thread control

| funct3 | name | rd | rs1 | rs2 | effect |
|---|---|---|---|---|---|
| 0 | **tmc** | `x0` | thread mask | `x0` | Set the current warp's thread mask. A zero mask retires the warp. |
| 1 | **wspawn** | `x0` | warp count | PC | Activate `rs1` warps on this core, each beginning at address `rs2`. |
| 2 | **split** | token | predicate | `x0` | Begin divergence on a per-thread predicate; pushes an IPDOM entry and returns a token in `rd`. With `rs2 = x1`, splits on the *inverted* predicate. |
| 3 | **join** | `x0` | token | `x0` | End the divergence region identified by the token from a matching `split`. |
| 4 | **barrier** | `x0` | barrier id | warp count | Block until `rs2` warps have arrived at barrier `rs1`. |
| 5 | **pred** | `x0` | condition | thread mask | Set the thread mask to `rs2` for threads where `rs1` holds; with `rd = x1`, uses the inverted condition. |
| 6 | — | | | | Not observed in use. Treat as reserved; do not emit. |
| 7 | **wsync** | `x0` | `x0` | `x0` | Synchronise the warp pipeline (drains outstanding operations). Used before reading a cycle counter, and as a pipeline fence. |

### 3.2 `funct7 = 1` — cooperative-thread primitives

Warp-level collectives. Each reads the operand across the **active lanes of the current warp** and
returns a value in `rd`, so they need no memory traffic and no barrier. Two of these matter for the
shapes in this benchmark: a reduction across a warp is a `shfl` tree rather than a trip through memory,
and a loop whose exit condition differs per lane can be driven by `vote_any` instead of a mask dance.

| funct3 | name | rd | rs1 | rs2 | effect |
|---|---|---|---|---|---|
| 0 | **vote_all** | result | predicate | `x0` | 1 if every active lane's predicate is non-zero, else 0. |
| 1 | **vote_any** | result | predicate | `x0` | 1 if any active lane's predicate is non-zero, else 0. |
| 2 | **vote_uni** | result | predicate | `x0` | 1 if the predicate is uniform across active lanes. |
| 3 | **vote_ballot** | mask | predicate | `x0` | A bitmask of the lanes whose predicate is non-zero. |
| 4 | **shfl_up** | value | value | b/c control | Shift values up the lanes of the warp. |
| 5 | **shfl_down** | value | value | b/c control | Shift values down the lanes. |
| 6 | **shfl_bfly** | value | value | b/c control | Butterfly exchange — the pattern a log-depth warp reduction wants. |
| 7 | **shfl_idx** | value | value | b/c control | Read another lane's value by index (arbitrary permutation/broadcast). |

Provenance for §3.1-3.2: the encodings are the ones the hardware's own kernel runtime emits, so they are
what the decoder accepts. The `rs2` "b/c control" packing for the `shfl` family is not restated here;
derive it from the RTL if you use those ops.

### 3.3 Reconvergence contract — READ THIS

Divergence is **not** automatic. A conditional branch whose predicate differs across threads in a
warp is **incorrect** unless it is bracketed by `split` / `join`:

- Every `split` must be matched by exactly one `join` consuming its token.
- The IPDOM stack is **LIFO**: regions must nest, never interleave.
- Uniform (warp-invariant) control flow needs no split/join — an ordinary branch is correct there.
- Fully predicated, branch-free code is also correct and needs neither.

A backend that emits a bare branch for divergent control flow will produce plausible results on
uniform data and **silently wrong** results the moment threads in a warp disagree.

---

## 4. CSR map

All are read with a standard `csrr`.

### Per-launch CTA identity and geometry (set by the KMU)

| CSR | name | CSR | name |
|---|---|---|---|
| `0xCD0` | cta_id | `0xCD9`–`0xCDB` | block_dim x / y / z |
| `0xCD1` | cta_rank | `0xCDC`–`0xCDE` | grid_dim x / y / z |
| `0xCD2` | cta_size | `0xCDF` | local(shared)-memory base address |
| `0xCD3`–`0xCD5` | thread_id x / y / z | `0xCE0` | cluster_size |
| `0xCD6`–`0xCD8` | block_id x / y / z | `0xCE1` | cta entry PC |

`thread_id.*` varies per thread within a warp; `block_id.*`, `block_dim.*`, `grid_dim.*` are uniform
across the CTA.

### Machine identity and geometry

| CSR | name | CSR | name |
|---|---|---|---|
| `0xCC0` | thread_id (within warp) | `0xFC0` | num_threads (per warp) |
| `0xCC1` | warp_id | `0xFC1` | num_warps (per core) |
| `0xCC2` | core_id | `0xFC2` | num_cores |
| `0x340` | mscratch (kernel-argument pointer) | `0xFC4` | num_barriers |

---

## 5. Memory model

- **Global memory** — the address space the host allocates buffers in. Device addresses are passed to
  the kernel through the argument block (§6). Cached; coherent within a core.
- **Shared / local memory** — a per-CTA scratchpad based at the address in CSR `0xCDF`. Visible to all
  threads of the CTA; not coherent across CTAs.
- **Ordering** — reads and writes are *not* ordered across threads without synchronisation. A
  cross-warp producer/consumer pair requires a `barrier` (funct3 4) between the write and the read.
  Within a single thread, program order holds.
- **Coherence across cores** — the machine has 4 cores with private, write-back L1 data caches; the
  **L2 is the only shared coherence point**. Two consequences: a `barrier` synchronises warps *within*
  a core and does not by itself make a write visible to another core, and results must reach the L2
  before the host reads them back. Do not assume a value written by one core is observable by another
  without going through that point.

---

## 6. Kernel entry convention

The runner-owned harness supplies startup, the linker script, and the KMU entry stub. Your compiler
emits exactly **one** device symbol:

```c
void merlin_kernel_body(const merlin_vx_kernel_arg_t* arg);
```

(declared in the harness header `merlin_vortex_abi.h`). It is called once per `(block, thread)`
coordinate the KMU launches. `arg` points to a device-memory block carrying one 64-bit device address
per `forward()` operand.

**Operand order is the capsule's.** The harness allocates and binds buffers in the order the capsule's
`inputs[]` list declares them (outputs included — they carry `role: output`), so `arg->args[i]` is the
address of the *i*-th declared operand. Your module's `merlin.arg_table` annotation must agree with
that order; it is what you are graded against, but it is not what the harness reads.

**Declare your launch grid.** Your emitted module must also carry a `merlin.grid = <N>` attribute (an
integer, e.g. `merlin.grid = 64 : i64`) stating the launch grid your mapping was compiled for — the
number of block coordinates the KMU should launch for a capsule. This is a **required** annotation: a
module with no `merlin.grid` is rejected (the runner will not guess a default, because a silent
"one coordinate per output element" would force every backend into the same mapping and defeat the
benchmark). How you map the iteration space onto blocks, warps and threads is yours (§7) — `merlin.grid`
just states the block count that mapping assumes.

Identity is **not** passed in — read it from the CSRs in §4.

---

## 7. Deliberately not specified

These are compiler decisions, not hardware facts, and nothing here constrains them:

- how an iteration space maps onto blocks, warps and threads;
- tile sizes, loop order, unrolling, or layout/packing;
- whether to use shared memory, and for what;
- where to place barriers beyond the ordering rule in §5;
- whether to handle divergence by split/join or by predication.
