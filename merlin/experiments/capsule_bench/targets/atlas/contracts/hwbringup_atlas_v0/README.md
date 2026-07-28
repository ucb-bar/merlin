# Atlas NPU — architecture & ISA overview

Distilled from the Atlas NPU design specification (`npu_spec/`) and the instruction green card. Atlas
(a.k.a. the "Penguin TPU") is a self-hosted matrix accelerator tile attached to a RISC-V system through
the Chipyard SoC framework. This overview is the factual companion to the RTL (`rtl/`) and the ISA
reference (`isa_include/`); where they disagree, the RTL is ground truth.

## 1. Self-hosted execution model

Atlas runs its own program. It fetches 32-bit fixed-width instructions from IMEM behind an architectural
program counter (`pc`) — IFU -> IDU -> execution units (Scalar / Matrix / DMA). This is unlike a RoCC
co-processor: there is no host CPU issuing custom instructions. Control flow (`beq/bne/blt/.../jal/jalr`)
takes effect **after 2 delay slots** (the ISA encodes static delays so the pipeline needs no dynamic
dependency checking — `DELAY imm` explicitly holds decode issue).

## 2. Register and memory state

- `x0..x31` — 32 scalar registers, RV32I ABI convention (`x0` = zero).
- `m0..m63` — 64 flat **tensor registers**, each `32 rows x 32 bytes = 1024 bytes`.
- `e0..e31` — 32 **scale registers**, one `FP8_E8M0` block-scale exponent each.
- Per-MXU local state: 2 FP8 weight slots + 2 `32x32 BF16` accumulation buffers per MXU.

Memory map (frozen architectural parameters):

| Region | Base | Size |
|---|---|---|
| `IMEM`  | `0x0002_0000` | 128 KiB |
| `VMEM`  | `0x2000_0000` | 1 MiB |
| `DRAM`  | `0x8000_0000` | 16 GiB |

8 DMA channels (32-byte aligned) move data between DRAM and VMEM; `dma.load/store/config/wait.ch<N>`.

## 3. The MXU (matrix unit) — the differentiator

Two MXUs (`MXU_COUNT = 2`), each a **32x32** fabric with 2 weight slots and 2 accumulation buffers:

- `mxu0` — a **systolic array** (`mxu/sa/`, `MXU0_MATMUL_LATENCY_CYCLES = 96`).
- `mxu1` — a **reduction / inner-product tree** fabric (`mxu/ipt/`, `MXU1_MATMUL_LATENCY_CYCLES = 35`).

Numerics: **FP8-E4M3** weights and activations feed the array; the array **accumulates in BF16** into the
local accumulation buffer; block scaling uses the `E8M0` scale registers (`vpack.bf16.fp8` /
`vunpack.fp8.bf16` quantize/dequantize against `e[]`). The matmul dataflow is:

```
DMA.LOAD  -> VLOAD (m[])            ; stage activations / weights into tensor regs
VTRPOSE.XLU                          ; transpose the weight tile (XLU)
VMATPUSH.W.MXU{0,1}                  ; push weights into an MXU weight slot
VMATPUSH.ACC.{FP8,BF16}.MXU{0,1}     ; seed the accumulator (bias)
VMATMUL[.ACC].MXU{0,1}  vd, vs1, ws  ; acc = (acc +) m[vs1] @ mxu.w[ws]
VMATPOP.{FP8,BF16}.MXU{0,1}          ; read the accumulator back to a tensor reg
VSTORE -> DMA.STORE                  ; spill the result tile to DRAM
```

The VPU (`vadd/vmul/vexp/vtanh/vredsum/...` `.bf16`) and XLU handle elementwise / reduction / activation
work in BF16 around the matmul.

## 4. Deriving the rest

Everything above that the RTL grounds (the exact opcode encodings, DIM=32, the memory map, the functional
model) is **derived** by mlc from the Atlas RTL (the arc model `libatlas_model.so` + `atlas_hw.mlir`,
`arc_available=True`) — it is not re-declared in the experiment descriptor. Use the green card and the
worked kernels as your entry point, then generalize from the RTL + ISA to the full op set.

## 5. What you deliver

An assembled Atlas **`kernel.S`** (endpoint kind `external_backend`) — assembled by the Atlas assembler
and executed on the mlc arc cosimulation model / Verilator. There is no host-side `.insn` intrinsic
stream: Atlas fetches and runs its own program.
