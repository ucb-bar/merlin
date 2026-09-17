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

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

    name: str  # this descriptor's identity (appears in filenames, manifests)
    dram_bytes: int  # usable DRAM at `dram_base` (the REAL chip's, not the DTS default)
    harts: int  # harts the SoC has
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
    vlen: int | None = None  # hardware vector length in bits; None = unknown, assume the V minimum
    console: str = CONSOLE_HTIF
    dram_base: int = 0x80000000  # derived-ok: per-board dataclass default; each board declares its own
    ram_label: str = "ram0"  # DT label the `&<label> { reg = ... }` overlay targets
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
    #: The merlin target whose RTL this board elaborates, when one is registered. Per-target environment
    #: names derive from it (``common.paths.target_env_name``) -- the Verilator binary override is
    #: MERLIN_<TARGET>_VERILATOR -- so the board, not shared code, says whose variable applies.
    target: str | None = None
    #: The chipyard harness config that elaborates THIS board's SoC, i.e. its RTL simulator is
    #: ``simulator-chipyard.harness-<rtl_sim_config>``. None = no elaborated simulator is declared.
    rtl_sim_config: str | None = None
    #: For a FireSim board: the hardware config (bitstream) its measurements were taken on, so a result
    #: quoted from FireSim names its hardware from the registry rather than from a literal where it is quoted.
    bitstream: str | None = None
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


#: Where the board registry lives: ``merlin/contract/boards.yaml`` (bundled into the wheel with the rest of
#: the contract tree). The boards are DATA, so targeting a new board is an entry there, not an edit here --
#: and the per-board reasoning (why each fact is what it is, and what it cost when it was wrong) sits
#: beside the entry it explains.
BOARDS_FILE: tuple[str, ...] = ("contract", "boards.yaml")
_SCHEMA_VERSION = 1

#: The closed vocabularies a registry entry may use, by field. An unknown value is refused at load: a
#: console or loader nobody wrote a driver for would otherwise surface as a silent hang on the board.
_ENUMS: dict[str, tuple[str, ...]] = {
    "console": (CONSOLE_HTIF, CONSOLE_UART),
    "flow": (FLOW_ZEPHYR, FLOW_BAREMETAL),
    "loader": (LOADER_UART_TSI, LOADER_PYUARTSI),
}
#: Fields written as byte sizes, which the registry may spell "<n> KiB|MiB|GiB" for legibility.
_SIZE_FIELDS = frozenset({"dram_bytes", "code_reserve"})
_SIZE_UNITS = {"KiB": 1 << 10, "MiB": 1 << 20, "GiB": 1 << 30}


class BoardRegistryError(ValueError):
    """The board registry is malformed. Raised when it is loaded -- never papered over with a default,
    because a board fact that silently fell back is exactly the wrong-DRAM / wrong-hart-count image that
    hangs on the chip with nothing printed."""


def _byte_size(value: Any, where: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        number, _, unit = value.strip().partition(" ")
        unit = unit.strip()
        if number.isdigit() and unit in _SIZE_UNITS:
            return int(number) * _SIZE_UNITS[unit]
    raise BoardRegistryError(f"{where}: {value!r} is not a byte size (an integer, or '<n> {'|'.join(_SIZE_UNITS)}')")


def _coerce(key: str, value: Any, ftype: str, where: str) -> Any:
    """Check one registry value against the ``Board`` field it fills.

    The field's declared type is read from the dataclass itself (``int``, ``str | None``,
    ``tuple[int, ...] | None``), so a field added to ``Board`` is loadable with no edit here.
    """
    alternatives = [t.strip() for t in ftype.split("|")]
    if value is None:
        if "None" in alternatives:
            return None
        raise BoardRegistryError(f"{where}: may not be null")
    base = alternatives[0]
    if key in _SIZE_FIELDS:
        return _byte_size(value, where)
    if base == "bool":
        if not isinstance(value, bool):
            raise BoardRegistryError(f"{where}: {value!r} is not a boolean")
        return value
    if base == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise BoardRegistryError(f"{where}: {value!r} is not an integer")
        return value
    if base == "str":
        if not isinstance(value, str):
            raise BoardRegistryError(f"{where}: {value!r} is not a string")
        allowed = _ENUMS.get(key)
        if allowed is not None and value not in allowed:
            raise BoardRegistryError(f"{where}: {value!r} is not one of {list(allowed)}")
        return value
    if base.startswith("tuple"):
        if not isinstance(value, list) or not all(isinstance(v, int) and not isinstance(v, bool) for v in value):
            raise BoardRegistryError(f"{where}: {value!r} is not a list of integers")
        return tuple(value)
    raise BoardRegistryError(f"{where}: Board field type {ftype!r} has no registry spelling")


def load_boards(path: str | Path | None = None) -> dict[str, Board]:
    """Read the board registry into ``{name: Board}``.

    Fails closed: a missing file, an unknown field, a value of the wrong type or outside its vocabulary,
    or a missing required fact (``dram_bytes``, ``harts``) raises :class:`BoardRegistryError` naming the
    board and the field. ``path`` defaults to :data:`BOARDS_FILE` resolved through
    ``common.paths.data_path`` (the checkout's tree, else the copy bundled in the wheel).
    """
    import yaml

    from ..common.paths import data_path

    p = Path(path) if path is not None else data_path(*BOARDS_FILE)
    if not p.is_file():
        raise BoardRegistryError(f"no board registry at {p}; boards are declared in merlin/{'/'.join(BOARDS_FILE)}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict) or not isinstance(raw.get("boards"), dict):
        raise BoardRegistryError(f"{p}: expected a mapping with a `boards:` mapping of name -> facts")
    if raw.get("schema_version") != _SCHEMA_VERSION:
        raise BoardRegistryError(
            f"{p}: schema_version {raw.get('schema_version')!r}, this loader reads {_SCHEMA_VERSION}"
        )
    fields = {f.name: f for f in dataclasses.fields(Board)}
    out: dict[str, Board] = {}
    for name, entry in raw["boards"].items():
        where = f"{p}: board {name!r}"
        if not isinstance(name, str) or not isinstance(entry, dict):
            raise BoardRegistryError(f"{where}: an entry is `<name>: {{field: value, ...}}`")
        unknown = sorted(set(entry) - set(fields))
        if unknown:
            raise BoardRegistryError(f"{where}: unknown field(s) {unknown}; a Board has {sorted(fields)}")
        if entry.get("name", name) != name:
            raise BoardRegistryError(f"{where}: `name: {entry['name']}` disagrees with its key")
        kwargs = {
            key: _coerce(key, value, str(fields[key].type), f"{where}, field {key!r}")
            for key, value in entry.items()
            if key != "name"
        }
        required = [
            f.name
            for f in fields.values()
            if f.name != "name" and f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
        ]
        missing = [key for key in required if key not in kwargs]
        if missing:
            raise BoardRegistryError(f"{where}: missing required fact(s) {missing}")
        out[name] = Board(name=name, **kwargs)
    return out


#: Boards we can target, as declared in the registry file (see :data:`BOARDS_FILE`).
BOARDS: dict[str, Board] = load_boards()


def board(name: str, **overrides) -> Board:
    """The descriptor for ``name``, with any field overridden.

    An unknown board is NOT an error: it falls back to conservative defaults (the V-minimum vector
    width, the 256 MB stock region, HTIF) so a new board can be tried before anyone writes it down —
    but the caller can override every fact, which is how a delivery states the DRAM and core count it
    was actually built for.
    """
    base = BOARDS.get(name)
    if base is None:
        base = Board(
            name=name,
            dram_bytes=256 * 1024 * 1024,
            harts=2,
            notes="not in BOARDS — conservative defaults; state the real facts explicitly",
        )
    if not overrides:
        return base
    from dataclasses import replace

    return replace(base, **overrides)
