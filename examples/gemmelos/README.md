# gemmelos bearly25: a chip nothing knows about

The Kodiak example targets a board Zephyr has files for. This one targets a tapeout with **no Zephyr port,
no debug host, and no board files anywhere** — its SDK is a bare-metal bring-up repo with zero Zephyr in
it. Everything the RTOS needs to print a character has to be *derived from that chip's own headers* at
build time.

That is the whole lesson of this example, and the failure it prevents is specific: an image that prints
nothing is indistinguishable from a hang, so guessing a UART address does not cost you a wrong number, it
costs you a round trip with no information in it.

```bash
export GEMMELOS_SDK=/path/to/gemmelos-bringup   # see "the SDK" below
./run.sh preflight
./run.sh probe                   # no SDK needed
./run.sh facts                   # what is read out of the SDK, and nothing more
./run.sh build                   # add --full for the shipped matrix
./run.sh package
./run.sh grade                   # with no arguments: grades the real failure shipped here
```

## 1. `facts` — the derivation that replaces a table of addresses

```
uart_base   0x10020000     sys_clk_hz  50000000     mtime_hz  50000
pll_base    0x140000       clksel_base 0x130000
```

Those come from parsing the chip's headers, and the *same* values feed both the Kconfig (the baud divisor,
`CONFIG_SYS_CLOCK_HW_CYCLES_PER_SEC`) and the device-tree overlay — one derivation, two consumers, so the
config and the device tree cannot disagree about which UART this is. `merlin/python/merlin/runtime/sdk_facts.py`
does the reading; nothing about this chip is a literal in Merlin.

Note `sys_clk_hz` is the **reset** clock: 50 MHz, while the vendor's own demos run the part at 500 MHz.
There is a second board descriptor that programs the PLL first (`BOARD=gemmelos_bearly25_zephyr_500mhz`),
replaying the vendor's own ordering — park every clock domain slow, program the PLL, switch the domains
over, *then* re-derive the UART divisor for the clock that just changed by 10×. Get that order wrong and
the console emits line noise, which reads as a corrupt program. So the 50 MHz set is the one to run first
and the raised-clock set ships beside it.

## 2. `build`, `package` — and the twin that makes a gate possible

A UART console cannot be simulated: spike has no such peripheral. So the gate builds an **HTIF twin** from
the same IR with the same `build_hash`, runs *that*, and ships its console as `expected_console.txt` with
the twin relationship stated in the package. The packager refuses to ship the pair if the hashes disagree.

The 500 MHz variant cannot even be twinned that way, because it differs by the PLL bring-up. Its evidence
is instead an instruction-sequence comparison against the gated 50 MHz sibling: same program, different
addresses. That comparison tolerates exactly one thing — an `auipc` the linker added or dropped because the
layout moved — counts them, and says so per binary. Anything else is recorded as a mismatch.

## 3. Read a log that came back broken

`./run.sh grade` with no arguments grades `returned/whisper_h1_debug_500mhz_fault.txt`, a **real** console
log from the chip. It needs only python and numpy. Here is what those lines are worth.

```
PRPORBOE E htid 1
...
*** Booting Zephyr OS build 852bb170cc56 ***
 mcause: 5, Load access fault
  mtval: 0
     a0: 0000000083b3ca00
     a4: 0000000000000000
     a5: fffffffff0f0f0f0
   mepc: 000000008004d570
FAIL fatal reason=0(cpu_exception) hart=0 thread= mcause=5 mepc=0x8004d570 mtval=0x0 vs=2 fs=1 build_hash=9f0afbc703640bcb
```

**Every line here is load-bearing:**

- `build_hash` names the exact binary, so the fault is attributable months later.
- The banner names the exact Zephyr commit (`852bb170cc56`), checkable against your own checkout.
- `mepc` lands inside `z_check_stack_sentinel`, whose two instructions are
  `ld a4, 0x130(a0)` (`_current->stack_info.start`) then `lw a3, 0(a4)`. With `a4 = 0` and `mtval = 0`,
  the current thread's stack bookkeeping is **zero**.
- `a0` is `z_main_thread`, and `sp` is inside `z_main_stack` — so it really is the main thread.
- `thread=` is **empty** although the build has `CONFIG_THREAD_NAME=y` and Zephyr names main `"main"`.
  Both the name (offset 0x110) and `stack_info.start` (0x130) read zero, so the struct was wiped *after*
  setup.
- `a5 = 0xfffffffff0f0f0f0` is the stack sentinel constant, confirming which check fired.

**The cause was one wrong field in a board descriptor.** It said VLEN 128; the chip's own probe reports
`vlenb 32`, i.e. VLEN 256. Zephyr sizes its per-thread vector context as a fixed
`vreg[32][CONFIG_RISCV_VECTOR_MAX_LEN/8]` — from *our* number — but `z_riscv_vstate_save` fills it with a
length read from the hardware (`vsetvli x0, e8, m8`, then four `vse8.v`), with no clamp and no comparison.
So every context switch wrote 1024 bytes into a 512-byte array. `sizeof(k_thread)` was 0x400 and
`z_idle_threads` ended *exactly* at `z_main_thread`, so the 512-byte overrun landed on its first half. A
thread that has never executed a vector instruction has a zeroed register file — hence zeros — and the
first timer tick after `z_smp_init()` dereferenced one.

Three things worth taking from it:

1. **No simulator can catch this class of bug.** spike is handed the VLEN we declared, so configured and
   actual agree there by construction. The check that catches it is the chip's own probe, which is why the
   packager now refuses to build when a returned probe log contradicts the descriptor
   (`make_delivery.py --probe-console <log>`).
2. **The plain image had the same corruption, silently.** `STACK_SENTINEL` is a debug-only feature. It is
   the only reason this was a diagnosable fault instead of another unexplained hang — which is what the
   earlier round of plain images looked like.
3. **The garbled `PROBE` lines at the top were also a bug of ours.** `crt.S` runs `main` on every hart by
   design (per-hart `mstatus.VS` and `vlenb` are the point), but nothing serialised the shared console, so
   a two-hart chip returned interleaved characters — and a garbled `vlenb` line is what hid the wrong
   width in the first place. The harts now take turns.

Both are fixed, and the corrected package states the asymmetry plainly: a unit **wider** than the declared
width overruns the save area, a unit **narrower** than the `zvl` minimum invalidates the code, and the
README used to say a different width was harmless.

## The SDK

`GEMMELOS_SDK` points at a third-party bring-up repo that is not ours to redistribute. It supplies only
facts — one UART base address, two clock rates, a PLL register map — and `./run.sh facts` prints exactly
what is read out of it, so if you cannot get the checkout you can see precisely what would have to be
supplied another way. `preflight`, `probe` and `grade` all run without it.
