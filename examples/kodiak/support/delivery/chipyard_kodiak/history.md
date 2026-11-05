## Why every multi-hart image failed last time, and what changed

Your table was the shape of the diagnosis: **every `h1` PASSED and every `h2`/`h3` FAILED**,
deterministically. Your trap logs are the proof, and they are worth more than the table — thank you
for them. All of this is our bug.

`mcause: 2` is an illegal instruction, and `mtval` is the instruction that took it. Both of yours
decode to the same thing:

```
mtval 0xcd827057   (deepjscc h2, mepc 0x8001d794)
mtval 0xcd817057   (whisper  h2, mepc 0x80079cf8)
   opcode[6:0]   = 0x57   OP-V  — a vector instruction
   funct3[14:12] = 111    OPCFG — the vsetvli family
   bits[31:30]   = 11            vsetivli
```

`vsetivli` is the instruction that *begins* every RVV block. Two different models, two different PCs,
the identical instruction — that is not an unsupported opcode or a bad encoding, it is a vector unit
that was **off** when the code reached its first vector instruction.

Here is why it was off. Our generated config did not set `CONFIG_RISCV_ISA_EXT_V`, so Zephyr never put
`mstatus.VS` into a thread's initial mstatus and never saved or restored vector state across a context
switch. The single-worker image survives that because it enables VS by hand and then never
context-switches again. A multi-hart image calls `merlin_omp_init`, which creates and joins a probe
thread per hart and then creates the pool — so the **master** switches out and comes back with
VS = Off, and traps on its own share of the first parallel region. Your `mepc` values land exactly
there.

What then made it unreadable was ours too: the fault was reported, and the run never **ended**. HTIF
kept polling to no exit, so from outside a trap and a hang looked the same. Zephyr's default handler
halts the faulting CPU and stops; it never terminates in the console protocol. Every image here now
prints `FAIL fatal reason= hart= mcause= mepc= mtval= vs= build_hash=` followed by `DONE`, so a
crashed run is a finished, attributable, gradeable run instead of a log that just stops.

The reason we had those two settings was a stale note in our own board descriptor, which described the
Zephyr your **`kodiak` branch** pins (submodule `5a06eb0d`). We build against the `dev` pin
(`852bb170`), two commits later, which contains *"riscv: decouple V/F save-restore + add
RISCV_V_KERNEL_ONLY"* — there, `v.c` compiles under `CONFIG_RISCV_ISA_EXT_V` independently of
`FPU_SHARING`, exactly as its CMakeLists comment says. We had simply never re-checked.

What these images use instead is **your own working sample's configuration** —
`samples/q8_gemm_minmax/prj.conf` on the `kodiak` branch, the one with a checked-in `ref-out`:
`CONFIG_RISCV_ISA_EXT_V=y`, `CONFIG_RISCV_VECTOR_MAX_LEN=512`, `CONFIG_FPU_SHARING=n`.
Belt and braces, the OpenMP shim now re-arms `mstatus.VS` at every parallel-region entry including the
master's, so the image is correct even where Zephyr is not managing it.

Your probe run settled the vector length independently, so it is no longer taken on trust from a
config file: `vlenb 64` is **VLEN = 512 bits**, and your `vlmax_e8 64` / `vlmax_e32 16` are exactly
what 512 predicts. Everything here is built for that.

**We could not reproduce the trap in any simulator**, and that is worth saying plainly rather than
implying the fix is tested: neither spike nor the vector RTL we run on FireSim enforces `mstatus.VS`, so
both ran the broken image perfectly. Your two `mtval` values are the only direct evidence that exists, which
is why they mattered so much. Every image here now reports the state itself — look for
`METRIC hart<N>_mstatus_vs`. A `2` or `3` means Zephyr is managing vector state; a `0` would mean we
are back where we started.

### The other failure mode in your logs, which we have NOT explained

Three runs — `deepjscc h3 scalar`, `spectformer h2`, `whisper h3 scalar` — produced **no Zephyr banner
at all** before the log ended. That is not the vector-state bug: those images had not reached any of
our code yet, so the fix above does not address them and we are not going to pretend otherwise.

The explanation we find most likely is that the capture ended before the upload did. `pyuartsi` sends
the loadable sections over the serial link, and at the **57600** baud in your `run_experiments.py` that
is minutes to tens of minutes before the first instruction executes — the estimate for each image is in
the table above. If your harness timeout is shorter than that, the log ends exactly the way yours did:
ELF loaded, hart kicked, silence, EOF.

Two things would confirm or kill that theory in one round:

1. **Raise the baud.** 921600 is 16× faster and `pyuartsi` only transmits `SHT_PROGBITS`, not the
   zero-fill, so nothing else has to change.
2. **Run `vlen_probe.elf` and the `_debug` twin of one of the three.** The probe is 10 KB and uploads
   instantly; if it banners and the big image does not, the difference is transport, not the image.

If the banner is still missing with a generous timeout at a high baud, tell us — that is a different
bug and we would want the log.

### Two corrections to the upload times we sent last time

Both ours. We quoted them from `MemSiz` at 921600 baud. Your loader transmits only `SHT_PROGBITS`
sections — so the byte count was too *high* — and you run it at 57600 — so the rate was 16x too
*optimistic*. The table below fixes both. We think this, not the compiled code, is why `whisper_tiny`
came back FAIL on every hart count: at that baud its upload alone outlasts any reasonable timeout.

