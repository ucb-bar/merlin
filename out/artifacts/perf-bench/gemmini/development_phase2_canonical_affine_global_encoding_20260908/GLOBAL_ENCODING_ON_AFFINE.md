# Affine-im2col plus global encoding/layout checkpoint

This is the integrated Phase-2 compiler artifact to evaluate. It is an isolated child of
`development_phase2_canonical_multimodel_affine_im2col_20260908`; only `compiler/` is candidate
compiler source. The source compiler hash is
`8f7dae852ed8e8f80e8a67207582e8842af098d3d131f6bdba1da3680315adb6` (42 files) and this
candidate is `624c96b345a4ed3112e8b8d5f91f05082c1709a33e7f6068f84d50a7f1d00ffc` (43 files).
The compiler-only review patch is [`global_encoding_on_affine.patch`](global_encoding_on_affine.patch).

## Integrated mechanisms

- The inherited affine im2col interior/border specialization is unchanged. AST comparison proves
  `_universal_valid_x_span` and `Emitter.emit_im2col_row` are identical to the qualified parent.
- Native scalar `roundeven`/`fptosi` is unchanged.
- The default-off dynamic weight bridge is unchanged and recompiles to its canonical target and
  command buffer under its explicit numeric contract.
- Native `LOOP_CONV_WS` remains capability-selected and is now safe across warm/reentrant calls.
- A new target-neutral global encoding solver selects operation alternatives and physical
  encodings across producer/consumer chains. It prices structural materialization bytes and
  accounts for layout transforms, fanout, lifetimes, peak live bytes, and optional capacity. It
  contains no target, model, layer, or fixed-shape rule.

The sole source overlap is `llvm_emit.py`. The integration keeps the affine im2col definitions and
adds only generic zero-initialized internal storage plus the native convolution bias address.

## Exact reduced proof

The two-layer chain retains its first i8 NHWC result as a compiler-owned allocation and feeds it
directly to the second native convolution. It emits two native descriptors, zero im2col tasks, and
one required device-dependency fence. Its caller ABI remains exactly `IFM, W0, W1, Y1`; `Y0` does
not leak into the ABI.

The qualification invokes the kernel once as untimed warmup and once in the measured window. All
96 outputs are exact, and the measured window captures only 102 Spike proxy cycles. This is a
warm-path semantic witness, not FPGA performance evidence. It also caught and fixed a real defect:
`no_bias=true` had allowed persistent accumulator rows to contaminate the second invocation. Each
native operation now uses an internal read-only `zeroinitializer` bias. Capacity-forced split
reductions fail closed because their partial-sum state cannot satisfy this repeatability contract.

The target-neutral solver has independent proofs for a logical-NCHW/physical-NHWC chain (two
internal transforms eliminated), conflicting fanout (one explicit adapter), and a one-byte arena
(machine-readable capacity refusal).

## Full-model position

The sequential default compile gate passes for ResNet-50, TinyLLaMA, LSTMNetViT, and SmolVLA with
302 MiB peak RSS. Every target is byte-identical to the affine parent. The opt-in TinyLLaMA dynamic
bridge also remains byte-identical.

Because the integrated ResNet target is byte-identical to the affine parent, its exact local Spike
evidence transfers without another expensive full-model run: 506,265,226 cycles versus
751,827,143, a 32.662% reduction / 1.485x speedup, with 1,000/1,000 exact logits. This remains a
Spike comparative proxy, not an FPGA cycle claim.

Global encoding does not add another ResNet executable delta yet. All 53 convolutions retain the
affine row-streamed fallback because the frozen program exposes i32 NCHW boundaries with ordered
f32 epilogues. The new ledger quantifies the opportunity but refuses to change semantics:

- current convolution output boundaries: 44,455,936 logical bytes;
- narrow encoded NHWC if proven legal: 11,113,984 bytes;
- conditional reduction: 33,341,952 bytes;
- peak live: 3,211,264 logical bytes versus 802,816 encoded bytes.

The next lever is therefore precise: form/prove the target-neutral narrow quantized epilogue, then
let this solver legalize NHWC/HWIO through the resulting producer/consumer region. Do not tune
native shapes before that proof. A FireSim run is warranted for the already-material affine target
or after global encoding makes an actual full-model structural change; it is not warranted solely
for the new refusal metadata.

## Verify and use

```sh
./verify.py

PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/compiler:/scratch/agustin/projects/oscar-merlin/merlin/python" \
/scratch/agustin/projects/oscar-merlin/.venv/bin/python -m pytest -q tests
```

Expected: verifier PASS and 43 tests PASS. The authoritative machine receipt is
`validation/global_encoding/receipt.json`. Re-run the cheap strict-warm witness with:

```sh
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/scratch/agustin/projects/oscar-merlin/merlin/python \
/scratch/agustin/projects/oscar-merlin/.venv/bin/python \
  validation/run_encoding_chain_spike.py
```

Compile a model directly:

```sh
MERLIN_PYTHON=/scratch/agustin/projects/oscar-merlin/.venv/bin/python \
./run-gemmini-opt --source-convolution --convert-iface-to-gemmini \
  --emit-command-buffer=/absolute/output/command_buffer.json \
  --emit-target-artifact -o /absolute/output/target.mlir /absolute/input.mlir
```

No shared source, full-model Spike, L3, FireSim, queue state, or git history was mutated during
this integration. If hardware is run later, use the queue and exactly `firesim kill -> firesim
infrasetup -> firesim runworkload -> firesim kill`, with untimed warmup and measured compute-only
cycles.
