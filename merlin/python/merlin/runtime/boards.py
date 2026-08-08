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
  assumes. Over-declaring costs memory and a different LMUL (the documented K1 trap). UNDER-declaring
  corrupts kernel memory: the save area is a fixed ``vreg[32][vlen/8]`` but Zephyr fills it with a
  hardware-derived length, so a too-small ``vlen`` overruns the thread struct on every context switch.
  See ``backends.zephyr_model._vector_max_len_bits``.
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

#: How the operator loads an image, which decides how many bytes cross the serial link.
#:
#: * `uart_tsi` (the C fesvr tool, used by gemmelos) walks PT_LOAD and writes **MemSiz**, zero-filling
#:   the part past `filesz`. An image whose `.bss`/arena claims the rest of DRAM therefore pays for
#:   hundreds of megabytes of zeros before it starts.
#: * `pyuartsi` (the Python loader on the Kodiak branch) walks the SECTION table and writes only
#:   `SHT_PROGBITS` sections with `sh_addr > 0`. `SHT_NOBITS` is skipped entirely, so it sends far less
#:   than MemSiz -- roughly `filesz`.
#:
#: Estimating both with one formula is how the shipped README came to quote "4 min" for an image that
#: takes an hour on the baud its own loader line specifies.
LOADER_UART_TSI = "uart_tsi"
LOADER_PYUARTSI = "pyuartsi"


@dataclass(frozen=True)
class Board:
    """Everything the generated app needs to know about a target."""

    name: str                     # this descriptor's identity (appears in filenames, manifests)
    dram_bytes: int               # usable DRAM at `dram_base` (the REAL chip's, not the DTS default)
    harts: int                    # harts the SoC has
    #: How many of those harts can execute VECTOR code, when that differs from `harts`. A
    #: heterogeneous SoC is normal -- a chip may bring up three cores and attach a vector unit to only
    #: two of them -- and the difference is invisible in every place you would look for it: the device
    #: tree lists identical `cpu@N` nodes, and `arch_num_cpus()` counts all of them. Fanning an RVV
    #: model out over a hart with no vector unit does not fail cleanly: that worker takes an illegal
    #: instruction, never reaches the barrier its peers are waiting on, and the image hangs until
    #: whoever is running it gives up on a timeout. Measured on a 3-core tapeout where 2 cores have V:
    #: the 1-hart images passed and every 3-hart image timed out.
    #: None means "all of them"; with only a count, the vector-capable harts are taken to be
    #: 0..vector_harts-1 -- see `vector_hart_ids` when that is not true.
    vector_harts: int | None = None
    #: WHICH harts are vector-capable, when they are not the first `vector_harts` of them. A count
    #: alone silently assumes 0..N-1, and on a chip whose vector units sit on (say) harts 0 and 2 that
    #: assumption deadlocks exactly like building too many harts does -- a worker lands on a scalar
    #: hart, traps, and never reaches the barrier. Nothing readable states the mapping (the device tree
    #: lists identical cpu@N nodes), so it is a fact someone has to tell us. None = the count's default.
    vector_hart_ids: tuple[int, ...] | None = None
    vlen: int | None = None       # hardware vector length in bits; None = unknown, assume the V minimum
    console: str = CONSOLE_HTIF
    dram_base: int = 0x80000000
    ram_label: str = "ram0"       # DT label the `&<label> { reg = ... }` overlay targets
    fpu_sharing: bool = False
    #: Set CONFIG_RISCV_ISA_EXT_V in the Zephyr config? Not "does the board have vectors" — our
    #: model.o always carries `v` from its own -march. This is only about whether ZEPHYR's kernel is
    #: compiled with V, and on a tree WITHOUT RISCV_V_KERNEL_ONLY it cannot be: setting it puts `v` in
    #: the GLOBAL march, and SDK 0.17.0 has no rv64imafdcv/lp64d libgcc multilib -- the link falls back
    #: to a 32-bit one and dies with "ELFCLASS32 incompatible with ELFCLASS64". `_prj_conf` therefore
    #: gates on the TREE's capability as well as this flag.
    #:
    #: Turning it off is NOT free, and the cost is not what the earlier note here claimed. reset.S does
    #: enable `mstatus.VS`, but only for the BOOT context: a Zephyr thread's initial mstatus comes from
    #: MSTATUS_DEF_RESTORE, which carries VS only under RISCV_ISA_EXT_V. So with this off, every thread
    #: starts with VS = Off, and any context switch puts it back to Off -- on silicon that enforces VS
    #: (Kodiak does; spike and Saturn do not) the next vector instruction traps. That is the Kodiak
    #: multi-hart hang: the single-worker image survives because it never switches again after poking
    #: VS by hand, and the multi-hart image dies because creating the OpenMP pool switches the master
    #: out and back. Leave this ON wherever the tree allows it.
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
    #: How this board's operator gets the image onto the chip. This decides HOW MANY BYTES cross the
    #: wire, which is not a detail: the two loaders in use here disagree by a factor of ten on the same
    #: ELF, and both were reported as "FAIL" when the real answer was "the upload had not finished".
    #: See `upload_bytes` for what each one actually sends.
    loader: str = LOADER_UART_TSI
    #: Baud of the LOADER link (not of the runtime console, which can differ). Bytes/second is derived
    #: from it rather than pinned, because a constant here silently survives a change of loader command.
    loader_baud: int = 921_600
    #: bytes to reserve for code+stack before the weights blob in a baremetal layout
    code_reserve: int = 64 * 1024 * 1024
    notes: str = ""

    @property
    def loader_bytes_per_s(self) -> float:
        """Payload throughput of the loader link. 8N1 framing is 10 bits on the wire per byte, which
        matches the 92 KB/s measured at 921600 baud, so derive it instead of carrying a constant."""
        return self.loader_baud / 10.0

    @property
    def build_board(self) -> str:
        """The Zephyr board identifier to pass to ``-DBOARD=`` (defaults to this descriptor's name)."""
        return self.zephyr_board or self.name

    @property
    def n_vector_harts(self) -> int:
        """Harts that can execute vector code. Defaults to all of them."""
        if self.vector_hart_ids is not None:
            return len(self.vector_hart_ids)
        return int(self.vector_harts if self.vector_harts is not None else self.harts)

    def hart_ids_for(self, backend: str) -> tuple[int, ...]:
        """The harts an image for ``backend`` may run on.

        A vector image is restricted to the vector-capable harts; a scalar one may use every hart,
        which is the only way to reach a core that has no vector unit.
        """
        if backend != "rvv":
            return tuple(range(self.harts))
        if self.vector_hart_ids is not None:
            return tuple(self.vector_hart_ids)
        return tuple(range(self.n_vector_harts))

    @property
    def vector_max_len(self) -> int:
        """Bits to size the per-thread vector save area. 32 registers of this width per thread.

        The two directions are NOT symmetric. Over-large is paid in RAM by every thread. Too small is a
        buffer overrun on every context switch, because the code that fills the area takes its length
        from the hardware and never compares it to the area it was given -- so the consumer
        (`zephyr_model._vector_max_len_bits`) floors this at the Zephyr tree's own default rather than
        emitting it as-is. The V minimum is 128.
        """
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
    # VLEN=512 is CONFIRMED by the chip's owner (2026-08-05), which retires the earlier inference: the
    # DTS says `riscv,isa = "rv64gc"` with no `v` at all, and the samples' CONFIG_RISCV_VECTOR_MAX_LEN
    # only sizes Zephyr's save area, so nothing in the repo stated the real width. Building for 128 on
    # a 512-bit unit is the documented K1 trap (a fixed-width kernel lands at a lower LMUL and leaves
    # three quarters of the datapath idle), so the fact matters even though our vector code is scalable.
    "chipyard_kodiak": Board(
        name="chipyard_kodiak", dram_bytes=512 * 1024 * 1024, harts=3, vector_harts=2, vlen=512,
        console=CONSOLE_HTIF,
        # These two flags used to read `fpu_sharing=True, zephyr_vector_ext=False`, on the reasoning
        # that this board's Zephyr lacks RISCV_V_KERNEL_ONLY and routes isr.S's
        # `z_riscv_vstate_save`/`_restore` through fpu.c/fpu.S, which only compile under FPU_SHARING --
        # so V without FPU_SHARING would not link. That was true of the submodule the `kodiak` BRANCH
        # pins (5a06eb0d) and false of the tree we actually build: ZEPHYR_BASE resolves to the `dev`
        # pin (852bb170), two commits later, which includes "riscv: decouple V/F save-restore + add
        # RISCV_V_KERNEL_ONLY (Saturn fork)". There, arch/riscv/core/CMakeLists.txt compiles v.c under
        # CONFIG_RISCV_ISA_EXT_V *independently of* CONFIG_FPU_SHARING, and says so in a comment.
        #
        # Carrying the stale pair cost a delivery round. Without RISCV_ISA_EXT_V no thread's mstatus
        # carries VS, so the OpenMP master loses vector state the moment pool creation switches it out;
        # with FPU_SHARING=y the resulting V illegal-instruction trap is mis-routed into the FP retry
        # path and the image hangs with no fault printed. Single-hart passed, every multi-hart image
        # failed. The chip's own known-working RVV+SMP sample (origin/kodiak:samples/q8_gemm_minmax,
        # which ships a ref-out) uses RISCV_ISA_EXT_V=y, RISCV_VECTOR_MAX_LEN=512, FPU_SHARING=n --
        # the settings below, and the only place in that repo the real VLEN is written down.
        fpu_sharing=False, zephyr_vector_ext=True, tick_hz=100,
        # The loader line this board's own scripts/run_experiments.py uses. pyuartsi sends PROGBITS
        # sections, not MemSiz, and 57600 is 16x slower than the 921600 our old fixed 92 KB/s assumed --
        # two errors in opposite directions that between them made every upload estimate we shipped
        # wrong, including the one that had whisper looking like a hang rather than an unfinished load.
        loader=LOADER_PYUARTSI, loader_baud=57_600,
        notes="Kodiak tapeout. DTS says ram0=256MB and MP_MAX_NUM_CPUS=2; the chip has 512MB and 3 "
              "working cores, 2 of them vector. Vector config matches the chip's own q8_gemm_minmax "
              "sample: Zephyr-managed V state, no FPU_SHARING."),
    # gemmelos: NOT a Zephyr target. github.com/Rakanic/gemmelos-bringup is a fork of
    # ucb-bar/Baremetal-IDE -- a bare-metal CMake SDK with no RTOS (grep -ri zephyr returns nothing) --
    # covering two chips selected by -DCHIP=. Facts below come from its platform/<chip>/*.ld and
    # chip_config.h. Its default linker script declares 256 MB of DRAM; the silicon has 1 GB, which is
    # what we build for.
    #
    # VLEN=256, and this was WRONG here (128) until the chip itself answered. Its own OPE kernel says
    # "VLEN=128/LMUL=4 gives VLMAX=16" while their spike runs use 256, so the tree was ambiguous and 128
    # was the cautious-looking pick. `vlen_probe.elf` on the silicon reports `vlenb 32`, `vlmax_e8 32`,
    # `vlmax_e32 8` -- three mutually consistent readings of the hardware CSR, i.e. VLEN=256. A comment
    # in a vendor kernel is not a fact about the part; a `csrr vlenb` on the part is. Under-declaring it
    # is not merely slow: see `zephyr_model._vector_max_len_bits` for the kernel-memory corruption it
    # caused, and note that no simulator gate can catch it, because spike is given the VLEN we declare.
    # Console is UART0 @115200 8N2 for
    # PLATFORM=CHIP builds and HTIF for PLATFORM=SIMS; our bare-metal harness speaks HTIF, which is what
    # the uart_tsi/FESVR link carries.
    "gemmelos_bearly25": Board(
        name="gemmelos_bearly25", dram_bytes=1024 * 1024 * 1024, harts=2, vlen=256,
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
        dram_bytes=1024 * 1024 * 1024, harts=2, vlen=256,
        console=CONSOLE_UART, sdk_chip="bearly25",
        flow=FLOW_ZEPHYR, tick_hz=100,
        notes="Bearly ML 25 via the GENERIC chipyard Zephyr board (their SDK has no Zephyr port). "
              "50 MHz core, 50 kHz CLINT timebase, CLINT 0x02000000, DRAM 0x80000000 (1 GB real, "
              "256 MB in their .ld). 2 harts (confirmed by the chip's owner). Console is the chip's "
              "own UART: that board's defconfig selects UART_HTIF, which needs a host servicing "
              "tohost and hangs on silicon, and the generic chipyard DT already describes this SoC's "
              "UART at the address the SDK headers give. No PLL programming on this path -- the chip "
              "stays on its 50 MHz reset clock and the baud divisor matches it; cycle counts are "
              "unaffected. See gemmelos_bearly25_zephyr_500mhz for the variant that raises it."),
    # The same board, with the chip's PLL raised to the frequency the vendor's own demos run at.
    #
    # Split into a second descriptor rather than made the default, deliberately. The upside is large --
    # 10x, on the one thing the chip's owner actually wants to do, and their own whisper demo decodes in
    # about fifteen seconds at this clock while ours had no reason to be at a tenth of it. The downside
    # is that programming a PLL wrong does not fail quietly into "no output": it garbles the console,
    # which reads as a corrupt program. So the 50 MHz set stays the one to run first, and this ships
    # beside it. If both come back, we learn the clock AND the correctness in one round trip; if only
    # the slow one does, we have still moved forward.
    #
    # The sequence itself is derived, not written: runtime.sdk_facts parses the PLL base, register
    # offsets, clock-selector map and reset clock out of the chip's own headers, and
    # runtime/c/merlin_socinit_zephyr.c replays their bmark-lib `init_test` order against it.
    "gemmelos_bearly25_zephyr_500mhz": Board(
        name="gemmelos_bearly25_zephyr_500mhz", zephyr_board="chipyard_riscv64",
        dram_bytes=1024 * 1024 * 1024, harts=2, vlen=256,
        console=CONSOLE_UART, sdk_chip="bearly25", chip_freq_hz=500_000_000,
        flow=FLOW_ZEPHYR, tick_hz=100,
        notes="Bearly ML 25 through the generic chipyard Zephyr port, with the PLL raised to 500 MHz "
              "before the console driver's divisor is applied (SYS_INIT at "
              "CONFIG_SERIAL_INIT_PRIORITY+1, so the driver cannot overwrite it). Identical "
              "computation to gemmelos_bearly25_zephyr; only the clock differs."),
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
