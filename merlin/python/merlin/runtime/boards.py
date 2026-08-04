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


@dataclass(frozen=True)
class Board:
    """Everything the generated app needs to know about a target."""

    name: str                     # the Zephyr board identifier passed to `-DBOARD=`
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
    notes: str = ""

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
        fpu_sharing=True, zephyr_vector_ext=False,
        notes="Kodiak tapeout. DTS says ram0=256MB and MP_MAX_NUM_CPUS=2; the chip has 512MB and 3 "
              "working cores. Its Zephyr lacks RISCV_V_KERNEL_ONLY and requires FPU_SHARING=y "
              "alongside V with eager switching."),
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
