# HW/accelerator-agnostic "best plan" starter prompt (data-driven)

Derived from the abc4 struggle analysis: the single biggest, **arm-independent** source of wasted rounds
was **instruction ordering / config + the conv-im2col lowering** — *not* anything target-specific. Every
arm tripped on some subset (merlin+CIRCT failed ALL capsules in round 1 on config-ordering; merlin missed
conv-im2col; these are exactly what the CIRCT checks encode). Front-loading this generic checklist should
cut rounds for every approach and on any systolic/tiled accelerator — it is phrased target-agnostically.

Intended use: prepend to the task for all arms (it gives no target-specific answer — only a *method*), or
as a "plan" the agent commits to before writing code.

---

## Before you emit any instructions, write down this plan and follow it

You are lowering a tensor interface to an accelerator's command stream. Regardless of the specific
accelerator, a correct backend almost always must satisfy these target-agnostic invariants. Verify each
against the ISA spec / RTL you were given, then enforce it in your lowering:

1. **Configure before use.** Emit every configuration instruction (execution mode, load/store params,
   scale/activation) **before** the first instruction that depends on it. "Use-before-config" is the most
   common silent failure — the RTL rejects or mis-computes.
2. **Decode-clean.** Only emit instruction encodings the ISA actually defines. If your encoder can produce
   an unknown/garbage opcode form, the hardware will reject it. Round-trip-decode your own output and
   assert zero UNKNOWN instructions.
3. **Respect the datapath shape.** Tile every operand to the hardware's native dimension (the systolic
   array DIM / vector width). Never feed an operand whose rank/shape the compute unit can't accept.
4. **Match the op to its instruction pattern — including degenerate ops:**
   - *matmul-family* (matmul, k-accum, linear, attention): load stationary operand → stream moving operand
     into the accumulator → read out with scale/activation.
   - *movement* (mvin/mvout, copy, transpose): a load→store with **no compute** — do not emit a matmul for
     it (a frequent bug: forcing a compute step into a pure data-movement op).
   - *convolution*: lower to a **2D im2col matrix**, then treat as matmul — do not hand 4D tensors to a 2D
     compute unit.
   - *padding / edge*: handle boundary regions explicitly; don't assume full tiles.
5. **Capacity-bound everything.** Keep scratchpad/accumulator residency within the hardware's stated
   capacities; spill/retile if a tile would overflow.
6. **Self-check structurally first, then numerically.** A static check of (1)–(5) over your emitted trace
   catches the dominant failure mode in milliseconds; only run the (slow) RTL sim once the structure is
   clean, to certify numerics.

If a target-specific spec contradicts any item above, follow the spec — these are defaults, not overrides.

---

### Why this is expected to help (abc4 evidence)
- merlin+CIRCT's entire round-1 failure (20/20 capsules) was item **1** (config-ordering) + item **2**
  (4 UNKNOWN instructions) — both in the checklist.
- merlin's only struggle was item **4/convolution** (im2col) — in the checklist.
- These are precisely the checks CIRCT encodes; the prompt surfaces that structural knowledge *up front*
  and *target-agnostically*, so even the no-tool baseline benefits. Validate with an N>1 A/B: same arms,
  with vs without this starter plan, compare rounds-to-correct.
