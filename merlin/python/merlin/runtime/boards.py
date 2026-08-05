"""Board facts as DATA, so targeting a new board is a descriptor rather than a code change.

The generated Zephyr app used to assume its board: an HTIF console, a ``&ram0`` label at
``0x80000000``, and a 256-bit vector-state save area. Those are true of the chipyard boards it was
written against and are not properties of "a RISC-V board" — which matters now that we build for a
tapeout whose facts come from *its own* repo, and for boards nobody here can test on.

Each field is a fact someone can check against the board's device tree / defconfig, and every one of
them has a failure mode if it is wrong rather than a performance cost:

* ``console`` — the wrong driver options mean **no output at all**, which is indistinguishable from a
  hang. The HTIF options we set are also the fix for a real one: unbuffered HTIF emits one character
  per host round-trip, which on a ~20 MHz core looks like the model never finishes.
* ``dram_bytes`` — the region the image is linked for. Larger than the chip has = a boot that dies
  before ``main``; smaller than the model needs = an allocation failure mid-inference.
* ``vlen`` — sizes the per-thread vector save area AND (via ``march_with_vlen``) what the compiler
  assumes. A mismatch is the documented K1 trap: fixed-width vectors land at a different LMUL.
* ``harts`` — how many the SoC actually has. Zephyr's SMP boot hangs waiting for harts that do not
  exist, with no fault printed.
* ``fpu_sharing`` — ``y`` mis-routes V-illegal-instruction traps into the FP path, which retries
  forever: a silent hang. Kept ``False`` unless a board is known to need otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass

#: Console driver families we know how to configure.
CONSOLE_HTIF = "htif"
CONSOLE_UART = "uart"

#: How an image for this board is produced. Not every RISC-V target runs an RTOS: `baremetal` targets
#: are built by `runtime.backends.spike_model` (crt.S + our own linker script + an absolute memory map),
#: which is the closer match for a Baremetal-IDE-style SDK than porting a Zephyr board would be.
FLOW_ZEPHYR = "zephyr"
FLOW_BAREMETAL = "baremetal"


@dataclass(frozen=True)
class Board:
    """Everything the generated app needs to know about a target."""

    name: str                     # this descriptor's identity (appears in filenames, manifests)
    dram_bytes: int               # usable DRAM at `dram_base` (the REAL chip's, not the DTS default)
    harts: int                    # harts the SoC has
    vlen: int | None = None       # hardware vector length in bits; None = unknown, assume the V minimum
    console: str = CONSOLE_HTIF
    dram_base: int = 0x80000000
    ram_label: str = "ram0"       # DT label the `&<label> { reg = ... }` overlay targets
    fpu_sharing: bool = False
    #: Set CONFIG_RISCV_ISA_EXT_V in the Zephyr config? Not "does the board have vectors" — our
    #: model.o always carries `v` from its own -march. This is only about whether ZEPHYR's kernel is
    #: compiled with V, and on some trees it cannot be: without RISCV_V_KERNEL_ONLY (absent from the
    #: Zephyr the Kodiak branch pins) setting it puts `v` in the GLOBAL march, and SDK 0.17.0 has no
    #: rv64imafdcv/lp64d libgcc multilib -- the link falls back to a 32-bit one and dies with
    #: "ELFCLASS32 incompatible with ELFCLASS64". Turning it off is safe because `mstatus.VS` is
    #: enabled at boot by reset.S under CONFIG_FPU (which these boards set), not under
    #: RISCV_ISA_EXT_V; what is lost is Zephyr saving vector state across a context switch, which
    #: this image does not need (one pinned cooperative worker per hart, no preemption mid-kernel).
    zephyr_vector_ext: bool = True
    #: Kernel tick rate to force, or None to accept the board's own. This is about OUR image, not
    #: about the board: it runs a single-shot inference on one pinned COOP worker per hart, with no
    #: preemption and no timeouts to resolve, so it needs almost no ticks. Where a board pairs a slow
    #: timer with a high tick rate the default is pathological -- Kodiak declares
    #: SYS_CLOCK_HW_CYCLES_PER_SEC=40000 with SYS_CLOCK_TICKS_PER_SEC=10000, i.e. a tick every 4
    #: cycles, and every tick saves/restores 32 vector registers under the FPU_SHARING=y that board
    #: also requires. The result is an image that spends essentially all of its time in the timer ISR.
    tick_hz: int | None = None
    #: The Zephyr board to build against, when it differs from `name`. Some chips have no Zephyr port
    #: of their own: gemmelos-bringup is a Baremetal-IDE fork with zero Zephyr in it, but its SoCs are
    #: Chipyard-based, so the generic `chipyard_riscv64` board describes them (DRAM at 0x80000000,
    #: CLINT at 0x02000000, HTIF console over the TSI/FESVR link they already load through). Keeping
    #: the names separate lets the package say WHICH CHIP it is for while the build says which port it
    #: used -- so the README can be honest that it is a generic port, not a bespoke one.
    zephyr_board: str | None = None
    #: For `console == CONSOLE_UART`: the key that selects this chip's platform directory inside its
    #: SDK checkout, from whose headers the UART/PLL/clock-selector facts are DERIVED at build time
    #: (`runtime.sdk_facts`). It is a lookup key into the target's own tree, not a fact about the
    #: chip -- the facts themselves are never written down here, because a literal MMIO address in
    #: shared code is silently wrong for the next tapeout. None for boards whose console needs no
    #: bring-up (a host-assisted HTIF link is alive before the core starts).
    sdk_chip: str | None = None
    #: DT label of the console UART node, for the `chosen`/`&label` overlay. A label is a property of
    #: the board's device tree, not of the chip -- unlike the address, which is derived.
    uart_label: str = "uart0"
    #: PLL target for a UART console, or None to stay on the chip's reset clock. Also the clock a
    #: returned `METRIC cycles` should be divided by, which is why the image prints it.
    chip_freq_hz: int | None = None
    flow: str = FLOW_ZEPHYR
    #: bytes to reserve for code+stack before the weights blob in a baremetal layout
    code_reserve: int = 64 * 1024 * 1024
    notes: str = ""

    @property
    def build_board(self) -> str:
        """The Zephyr board identifier to pass to ``-DBOARD=`` (defaults to this descriptor's name)."""
        return self.zephyr_board or self.name

    @property
    def vector_max_len(self) -> int:
        """Bits to size the per-thread vector save area. 32 registers of this width per thread, so an
        over-large value is paid by every thread; the V minimum is 128."""
        return int(self.vlen or 128)


#: Boards we can target. Facts are derived from each board's own repo, not assumed:
#:
#: * ``chipyard_kodiak`` — `boards/chipyard/kodiak/` on the `kodiak` branch of ucb-bar/zephyr-chipyard-sw.
#:   Its DTS declares `&ram0` as 256 MB and `CONFIG_MP_MAX_NUM_CPUS=2`, but the silicon has **512 MB and
#:   3 working Saturn-vector cores**, so those are what we build for (merlin emits the `&ram0` overlay).
#:   Console is HTIF (`CONFIG_UART_HTIF=y`, the SiFive UART driver is off) and `zephyr,console = &htif`.
#:   Its defconfig sets `CONFIG_FPU_SHARING=y`, which we deliberately override — see the class doc.
#:   No VLEN is declared anywhere in its DT (`riscv,isa = "rv64gc"`, V comes only from Kconfig), so it
#:   is left None until someone measures `vlenb` on the chip.
#: * ``spike_riscv64`` / ``chipyard_riscv64`` — the substrates we already validate on, described here so
#:   the same code path serves them.
BOARDS: dict[str, Board] = {
    "spike_riscv64": Board(
        name="spike_riscv64", dram_bytes=1 << 31, harts=8, console=CONSOLE_HTIF,
        notes="functional simulator; DRAM is whatever `-m` says, VLEN comes from the ISA string"),
    "chipyard_riscv64": Board(
        name="chipyard_riscv64", dram_bytes=256 * 1024 * 1024, harts=4, vlen=256,
        console=CONSOLE_HTIF, notes="Verilator/FireSim multi-Saturn SoC"),
    # The FPGA TARGET is its own board, not the DTS default. `chipyard_riscv64` keeps ram0's 256 MB
    # because that is what a Verilator/spike run of that config has; the FireSim dual-Saturn target has
    # WithExtMemSize = 16 GB at 0x80000000 (see zephyr_model.DRAM_END, and the whole-model TinyLlama
    # run whose 482 MB of weights could not otherwise have been linked). Conflating the two is what
    # made whisper_tiny -- 464 MB of region -- look like it did not fit on a 16 GB target.
    "firesim_dual_saturn": Board(
        name="firesim_dual_saturn", zephyr_board="chipyard_riscv64",
        dram_bytes=16 * 1024 * 1024 * 1024, harts=2, vlen=256, console=CONSOLE_HTIF,
        notes="FireSim alveo_u250_firesim_dual_saturn_v256d128: 2 Shuttle tiles each with a Saturn "
              "unit at vLen=256/dLen=128, 25 MHz fpga_frequency (~24.9 MHz effective measured), "
              "16 GB target DRAM."),
    "chipyard_kodiak": Board(
        name="chipyard_kodiak", dram_bytes=512 * 1024 * 1024, harts=3, vlen=None,
        console=CONSOLE_HTIF,
        # FPU_SHARING=y here is REQUIRED, not preferred, and it is why the board's own defconfig sets
        # it. The Zephyr this board pins calls `z_riscv_vstate_save`/`_restore` from isr.S whenever
        # RISCV_ISA_EXT_V=y and ISA_EXT_V_LAZY=n, and those symbols live in fpu.c/fpu.S, which are only
        # compiled under FPU_SHARING. With our usual FPU_SHARING=n the image does not LINK
        # ("undefined reference to z_riscv_vstate_save"). The newer Zephyr on the `dev` branch
        # restructured that guard and added RISCV_V_KERNEL_ONLY, which is why merlin's default config
        # works there and not here. Trade-off accepted knowingly: y gives eager vector save/restore
        # (what we want for correctness across any preemption) but is the setting that mis-routed
        # V-illegal-instruction traps on a FireSim Saturn tile. This image runs ONE pinned cooperative
        # worker per hart, so it should never preempt mid-kernel.
        fpu_sharing=True, zephyr_vector_ext=False, tick_hz=100,
        notes="Kodiak tapeout. DTS says ram0=256MB and MP_MAX_NUM_CPUS=2; the chip has 512MB and 3 "
              "working cores. Its Zephyr lacks RISCV_V_KERNEL_ONLY and requires FPU_SHARING=y "
              "alongside V with eager switching."),
    # gemmelos: NOT a Zephyr target. github.com/Rakanic/gemmelos-bringup is a fork of
    # ucb-bar/Baremetal-IDE -- a bare-metal CMake SDK with no RTOS (grep -ri zephyr returns nothing) --
    # covering two chips selected by -DCHIP=. Facts below come from its platform/<chip>/*.ld and
    # chip_config.h. Its default linker script declares 256 MB of DRAM; the silicon has 1 GB, which is
    # what we build for. VLEN=128 is stated by its own OPE kernel ("VLEN=128/LMUL=4 gives VLMAX=16");
    # their spike runs use 256, so the two are not interchangeable. Console is UART0 @115200 8N2 for
    # PLATFORM=CHIP builds and HTIF for PLATFORM=SIMS; our bare-metal harness speaks HTIF, which is what
    # the uart_tsi/FESVR link carries.
    "gemmelos_bearly25": Board(
        name="gemmelos_bearly25", dram_bytes=1024 * 1024 * 1024, harts=2, vlen=128,
        console=CONSOLE_UART, sdk_chip="bearly25", chip_freq_hz=500_000_000,
        flow=FLOW_BAREMETAL,
        notes="Bearly ML 25 via Baremetal-IDE (-DCHIP=bearly25). 2 harts, hart 1 idles in wfi until "
              "dispatched; -march=rv64gcv_zfh -mabi=lp64d -mcmodel=medany; entry _start resumed at "
              "0x80000000. NOT Zephyr. Console is the chip's own UART0 (facts derived from its SDK "
              "headers): PLATFORM=CHIP builds have no host to service HTIF, so an HTIF image hangs in "
              "its first print. PLL raised to 500 MHz, the frequency their own demos run at."),
    # The SAME chip through Zephyr rather than bare metal, which is the only way to get RVV MULTICORE
    # there: on bare metal hart 1 waits in wfi for their own thread-lib to dispatch it, while Zephyr
    # SMP + merlin's OpenMP shim drives both harts the way every other multicore image here does.
    # Facts from platform/bearly25/chip_config.h: SYS_CLK_FREQ 50 MHz, MTIME_FREQ 50 kHz (hence the
    # tick override -- a 50 kHz timebase against a default 10 kHz tick rate is the Kodiak pathology
    # again), CLINT at 0x02000000, DRAM at 0x80000000. TWO harts, confirmed by the chip's owner; their
    # tree is ambiguous about it (thread-lib/hthread.h says 2, four other lib copies say 4) and
    # guessing high would be a silent SMP-boot hang, so this is a fact worth having been told.
    "gemmelos_bearly25_zephyr": Board(
        name="gemmelos_bearly25_zephyr", zephyr_board="chipyard_riscv64",
        dram_bytes=1024 * 1024 * 1024, harts=2, vlen=128,
        console=CONSOLE_UART, sdk_chip="bearly25",
        flow=FLOW_ZEPHYR, tick_hz=100,
        notes="Bearly ML 25 via the GENERIC chipyard Zephyr board (their SDK has no Zephyr port). "
              "50 MHz core, 50 kHz CLINT timebase, CLINT 0x02000000, DRAM 0x80000000 (1 GB real, "
              "256 MB in their .ld). 2 harts (confirmed by the chip's owner). Console is the chip's "
              "own UART: that board's defconfig selects UART_HTIF, which needs a host servicing "
              "tohost and hangs on silicon, and the generic chipyard DT already describes this SoC's "
              "UART at the address the SDK headers give. No PLL programming on this path -- the RTOS "
              "brings its console up before our code runs, so the chip stays on its 50 MHz reset "
              "clock and the baud divisor matches it; cycle counts are unaffected."),
    "gemmelos_dsp25": Board(
        name="gemmelos_dsp25", dram_bytes=1024 * 1024 * 1024, harts=2, vlen=128,
        console=CONSOLE_UART, sdk_chip="dsp25", chip_freq_hz=500_000_000,
        flow=FLOW_BAREMETAL,
        notes="DSP 25 via Baremetal-IDE (-DCHIP=dsp25). Same ABI/entry as bearly25; NOT Zephyr. "
              "Console is UART0, derived from its own platform/dsp25 headers -- where the clock "
              "selector sits at RCC_BASE+0x30000 rather than at RCC_BASE, which is why that address "
              "is derived from the SDK's RCC_CLOCK_SELECTOR define and not assumed."),
}


def board(name: str, **overrides) -> Board:
    """The descriptor for ``name``, with any field overridden.

    An unknown board is NOT an error: it falls back to conservative defaults (the V-minimum vector
    width, the 256 MB stock region, HTIF) so a new board can be tried before anyone writes it down —
    but the caller can override every fact, which is how a delivery states the DRAM and core count it
    was actually built for.
    """
    base = BOARDS.get(name)
    if base is None:
        base = Board(name=name, dram_bytes=256 * 1024 * 1024, harts=2,
                     notes="not in BOARDS — conservative defaults; state the real facts explicitly")
    if not overrides:
        return base
    from dataclasses import replace
    return replace(base, **overrides)
