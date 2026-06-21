# B3: requant reconciliation (Gemmini scale vs merlin shift) — finding

**Outcome: requant (C2/C3) is NOT bit-exact certifiable on Gemmini against merlin's reference.
This is a genuine hardware/semantics divergence, documented here rather than papered over.**

## merlin semantics

`Tensor.requant(shift)` = **integer round-half-up arithmetic shift**:
`(x + (1 << (shift-1))) >> shift`, applied post-matmul to the i32 accumulator, output kept i32
(for `output_dtype: i32`). Deterministic integer arithmetic.

## Gemmini semantics (measured)

Empirically (spike-gemmini, `acc_scale = 1/16`, shift 4, single 16×16 tile):

| | first 8 elements |
|---|---|
| merlin reference (requant, shift 4) | `[6, 6, 2, 2, 6, 6, 2, 2]` |
| Gemmini (full-i32 mvout, acc_scale=1/16) | `[96, 96, 32, 32, 96, 96, 32, 32]` |

→ **256/256 mismatch.** Gemmini returned the *raw, unscaled* matmul. So:

1. **`acc_scale` is ignored on the full_C (i32) accumulator readout.** The scale only applies on
   the **i8-downscale** readout path. Our C0–C5 path uses full-i32 readout (to preserve the i32
   accumulator), which bypasses the scale entirely.
2. The i8-downscale path applies the scale as a **float multiply with round-to-nearest(-even)**,
   not merlin's **integer round-half-up shift** — different rounding mode — and downconverts to
   **i8** (clamped), not i32. Even routing through it would not bit-match merlin's i32 requant.
3. `config_ex`'s `sys_shift` is a **pre-accumulation** shift on the PE output, not a post-matmul
   requant — also not equivalent.

## Conclusion

There is no Gemmini configuration that reproduces merlin's `requant(shift)` bit-exactly:
full-i32 readout ignores scale; i8 readout uses a different rounding mode + dtype. To certify a
requant rung one of the following must change (each is a deliberate decision, not a silent fix):

- **(A)** redefine merlin's `requant` to match Gemmini's float `acc_scale` rounding (round-to-
  nearest-even) + i8 output, and certify on the i8 path; or
- **(B)** keep merlin's integer requant and accept that Gemmini cannot bit-match it (requant
  stays a host/runtime-side op, not delegated to Gemmini); or
- **(C)** restrict requant to power-of-two scales where Gemmini's rounding happens to coincide
  with round-half-up (needs per-value verification; not assumed here).

Until one is chosen, **the certifiable Gemmini rungs are the integer-exact ones: C0 (matmul),
C1 (relu — `max(0,x)` is identical both sides), C4/C4e (tiled/padded matmul), C5 (reuse).**
`gemmini_codegen` therefore lists only `relu` in `SUPPORTED_EPILOGUE` and rejects requant with a
pointer to this document. This is the honest scope; C2/C3 are open by design, not by oversight.

---

## Update (decision A, applied additively): Gemmini float acc_scale is now SUPPORTED

merlin now supports **both** requant formats; the integer `requant` is unchanged.

- `Tensor.requant_acc_scale(scale)` = `clamp_i8(round_near_even(x * scale))` in float32 —
  bit-identical to `gemmini_params.h` `ACC_SCALE`/`ROUND_NEAR_EVEN`. Epilogue stage `acc_scale`
  (+ `output_dtype: i8`, commit attr `acc_scale: <float>`). The codegen emits the **i8 readout**
  (accumulator address without the `full_C` bit) with `config_st` carrying the float scale bits +
  activation. **RTL-certified** three-way bit-exact on Gemmini Verilator (Q0/Q1/Q2/Q1t), including
  a **non-power-of-two scale (Q2 = 0.013)** that exercises the float multiply.
- `Tensor.requant(shift)` (round-half-up integer shift) stays as merlin's **host/runtime-side**
  requant — not delegated to Gemmini (it would not bit-match the float scale).

### "Support all Gemmini requant formats" — getgo vs generated

Gemmini's requant/readout space is **bounded and config-determined**, not open-ended. It factors
into orthogonal knobs, all of which are **structural facts of the build** (readable from
`gemmini_params.h` + the Chisel config), not things to invent:

| knob | values (this build) | grounded in |
|---|---|---|
| scale fn | float `acc_scale` (round-near-even); rounding-right-shift; identity | `acc_scale_t` type, `acc_scale_args` |
| activation | NONE, RELU, RELU6, LAYERNORM, IGELU, SOFTMAX | `has_nonlinear_activations`, ISA act enum |
| readout width | full-i32 (no scale) / i8 (scale+clamp) | `ACC_READ_{FULL,SMALL}_WIDTH` |
| bias | accumulator preload (D) | ISA |

**Recommended split:**
- **Supported from the getgo (deterministic primitive library):** the *linear, bit-exact*
  primitives that mirror the `gemmini_params.h` macros 1:1 — float acc_scale (round-near-even),
  rounding-right-shift, identity, RELU/RELU6, i8 clamp, full-i32 vs i8 readout, bias. These are
  real hardware-defined functions; merlin implements + certifies them once, additively. (Done:
  `requant_acc_scale` + the i8 readout path.)
- **Generated / extracted per target:** *which* of these a given build supports is a **fact**
  (`ACC_SCALE_T_IS_FLOAT`, `has_nonlinear_activations`, the readout-width defines, the act enum).
  The target spec declares the supported set; the codegen maps a merlin epilogue → the config
  encoding (acc_scale bits, act field, readout bit). The agentic slot *selects/maps*; it does not
  invent semantics. This is the thesis in microcosm: the requant format must be **grounded in RTL
  facts and certified**, never hand-assumed.
- **Deferred tier (nonlinear):** LAYERNORM/IGELU/SOFTMAX use polynomial/float approximations in
  HW — not the linear bit-exact slice. Certify separately (match the HW approximation exactly, or
  with a documented tolerance). Out of the getgo linear set.
