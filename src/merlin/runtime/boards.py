"""Target-owned board facts consumed by generic runtime builders.

Merlin ships the schema and loader, not a board registry. An OOT catalog must
state hardware and port facts explicitly; guessing DRAM, harts, console, memory
labels or upload protocol can create images that hang without a diagnostic.
"""

from __future__ import annotations

import dataclasses
import os
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
#: * `uart_tsi` walks PT_LOAD and writes **MemSiz**, zero-filling
#:   the part past `filesz`. An image whose `.bss`/arena claims the rest of DRAM therefore pays for
#:   hundreds of megabytes of zeros before it starts.
#: * `pyuartsi` walks the SECTION table and writes only
#:   `SHT_PROGBITS` sections with `sh_addr > 0`. `SHT_NOBITS` is skipped entirely, so it sends far less
#:   than MemSiz -- roughly `filesz`.
#:
#: A loader choice must match the target's actual transport; the byte volume can
#: differ substantially even for the same ELF.
LOADER_UART_TSI = "uart_tsi"
LOADER_PYUARTSI = "pyuartsi"


@dataclass(frozen=True)
class Board:
    """Everything the generated app needs to know about a target."""

    name: str  # this descriptor's identity (appears in filenames, manifests)
    dram_bytes: int  # usable physical DRAM at dram_base
    harts: int  # harts the SoC has
    dram_base: int  # physical DRAM origin
    console: str  # runtime console family
    flow: str  # Zephyr or bare metal
    loader: str | None = None  # operator's ELF upload protocol; None until an upload path is qualified
    loader_baud: int | None = None  # baud of that link; None when no upload path is qualified
    #: How many of those harts can execute VECTOR code, when that differs from `harts`. A
    #: heterogeneous SoC may attach a vector unit to only some harts. Fanning an
    #: RVV model out over a scalar hart can trap before a worker barrier completes.
    #: None means "all of them"; with only a count, the vector-capable harts are taken to be
    #: 0..vector_harts-1 -- see `vector_hart_ids` when that is not true.
    vector_harts: int | None = None
    #: WHICH harts are vector-capable, when they are not the first `vector_harts` of them. A count
    #: alone assumes 0..N-1. None = the count's default.
    vector_hart_ids: tuple[int, ...] | None = None
    vlen: int | None = None  # hardware vector length in bits; None = unknown, assume the V minimum
    ram_label: str | None = None  # required for Zephyr; DT memory label to override
    fpu_sharing: bool | None = None
    #: Whether the selected Zephyr port can save vector state for threads. The
    #: runtime also checks whether the selected Zephyr tree defines the symbols.
    zephyr_vector_ext: bool | None = None
    #: Kernel tick rate to force, or None to accept the selected port's own.
    tick_hz: int | None = None
    #: The Zephyr port to build against, when it differs from this descriptor's name.
    zephyr_board: str | None = None
    #: Maximum CPU nodes declared by this Zephyr port's device tree. If the SoC
    #: has fewer harts, generate a disabling overlay; unknown means no overlay.
    dt_cpu_nodes: int | None = None
    #: For `console == CONSOLE_UART`: the key that selects this chip's platform directory inside its
    #: SDK checkout, from whose headers the UART/PLL/clock facts are derived.
    sdk_chip: str | None = None
    #: DT label of the console UART node, for the `chosen`/`&label` overlay. A label is a property of
    #: the board's device tree, not of the chip -- unlike the address, which is derived.
    uart_label: str | None = None
    #: PLL target for a UART console, or None to stay on the chip's reset clock. Also the clock a
    #: returned `METRIC cycles` should be divided by, which is why the image prints it.
    chip_freq_hz: int | None = None
    #: Minimum code/static-I/O/stack region before weights; large generated content raises this floor.
    code_reserve: int | None = None
    #: The merlin target whose RTL this board elaborates, when one is registered. Per-target environment
    #: names derive from it (``common.paths.target_env_name``) -- the Verilator binary override is
    #: MERLIN_<TARGET>_VERILATOR -- so the board, not shared code, says whose variable applies.
    target: str | None = None
    #: The selected simulator harness config, if an RTL simulator is declared.
    rtl_sim_config: str | None = None
    #: SHA256 of the elaborated DTS selected for this board. A caller-supplied DTS
    #: must match this board-owned identity before its CPU ISA can authorize code.
    host_dts_sha256: str | None = None
    #: FPGA bitstream identity for measurements, when applicable.
    bitstream: str | None = None
    #: The selected Zephyr port's unmodified DT RAM-region size. A generated
    #: overlay is needed only when the image needs more than this region.
    zephyr_default_ram_bytes: int | None = None
    #: Maximum linked code + weights + arena region supported by this port's
    #: model-object relocation mode. Larger models require an external layout.
    zephyr_link_limit_bytes: int | None = None
    #: When the port supports separate weights, size of its low code/arena
    #: region. Weights begin immediately after it within this board's DRAM.
    zephyr_external_ram_bytes: int | None = None
    #: Physical DRAM reserved after an external weights blob, if applicable.
    zephyr_external_tail_reserve_bytes: int | None = None
    #: Set only when this board descriptor represents the Spike simulator;
    #: callers use this instead of inspecting a target-specific board name.
    simulator: str | None = None
    notes: str = ""

    @property
    def loader_bytes_per_s(self) -> float:
        """Payload throughput of an 8N1 upload link (10 wire bits per byte)."""
        if self.loader is None or self.loader_baud is None:
            raise BoardRegistryError(f"{self.name}: no qualified upload loader and baud")
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


#: Board facts belong to the selected target, not to Merlin's installed core. An example
#: catalog lives in ``examples/board-catalog.yaml``; deployments can use an OOT catalog.
BOARD_CATALOG_ENV = "MERLIN_BOARD_CATALOG"
_SCHEMA_VERSION = 1

#: The closed vocabularies a registry entry may use, by field. An unknown value is refused at load: a
#: console or loader nobody wrote a driver for would otherwise surface as a silent hang on the board.
_ENUMS: dict[str, tuple[str, ...]] = {
    "console": (CONSOLE_HTIF, CONSOLE_UART),
    "flow": (FLOW_ZEPHYR, FLOW_BAREMETAL),
    "loader": (LOADER_UART_TSI, LOADER_PYUARTSI),
    "simulator": ("spike",),
}
#: Fields written as byte sizes, which the registry may spell "<n> KiB|MiB|GiB" for legibility.
_SIZE_FIELDS = frozenset(
    {
        "dram_bytes",
        "code_reserve",
        "zephyr_default_ram_bytes",
        "zephyr_link_limit_bytes",
        "zephyr_external_ram_bytes",
        "zephyr_external_tail_reserve_bytes",
    }
)
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
    or a missing required hardware or port fact raises :class:`BoardRegistryError` naming the
    board and the field. The caller passes ``path`` or selects an OOT catalog with
    ``MERLIN_BOARD_CATALOG``. Merlin never guesses a board or loads example target
    facts implicitly.
    """
    import yaml

    selected = path if path is not None else os.environ.get(BOARD_CATALOG_ENV)
    if not selected:
        raise BoardRegistryError(f"no board catalog selected; set {BOARD_CATALOG_ENV} to a target-owned YAML file")
    p = Path(selected).expanduser().resolve()
    if not p.is_file():
        raise BoardRegistryError(f"no board catalog at {p}; check {BOARD_CATALOG_ENV}")
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
        conditional = []
        if kwargs["flow"] == FLOW_ZEPHYR:
            conditional.extend(
                (
                    "ram_label",
                    "fpu_sharing",
                    "zephyr_vector_ext",
                    "zephyr_default_ram_bytes",
                    "zephyr_link_limit_bytes",
                )
            )
            if kwargs["console"] == CONSOLE_UART:
                conditional.append("uart_label")
        if kwargs["flow"] == FLOW_BAREMETAL:
            conditional.append("code_reserve")
        if kwargs["flow"] == FLOW_ZEPHYR:
            conditional.extend(("loader", "loader_baud"))
        missing = [key for key in conditional if kwargs.get(key) is None or kwargs.get(key) == ""]
        if missing:
            raise BoardRegistryError(f"{where}: missing required fact(s) for {kwargs['flow']}: {missing}")
        if (
            kwargs["flow"] == FLOW_ZEPHYR
            and kwargs.get("vector_harts") is None
            and kwargs.get("vector_hart_ids") is None
        ):
            raise BoardRegistryError(f"{where}: declare vector_harts or vector_hart_ids for a Zephyr board")
        if kwargs.get("host_dts_sha256") is not None and (
            len(kwargs["host_dts_sha256"]) != 64
            or any(letter not in "0123456789abcdef" for letter in kwargs["host_dts_sha256"])
        ):
            raise BoardRegistryError(f"{where}: host_dts_sha256 must be a lowercase SHA256 hex digest")
        for key in ("dram_bytes", "harts"):
            if kwargs[key] <= 0:
                raise BoardRegistryError(f"{where}: {key} must be positive")
        if (kwargs.get("loader") is None) != (kwargs.get("loader_baud") is None):
            raise BoardRegistryError(f"{where}: loader and loader_baud must be declared together")
        if kwargs.get("loader_baud") is not None and kwargs["loader_baud"] <= 0:
            raise BoardRegistryError(f"{where}: loader_baud must be positive")
        if kwargs["dram_base"] < 0:
            raise BoardRegistryError(f"{where}: dram_base must be nonnegative")
        if kwargs.get("code_reserve") is not None and kwargs["code_reserve"] <= 0:
            raise BoardRegistryError(f"{where}: code_reserve must be positive")
        if kwargs["flow"] == FLOW_ZEPHYR:
            default = kwargs["zephyr_default_ram_bytes"]
            limit = kwargs["zephyr_link_limit_bytes"]
            external = kwargs.get("zephyr_external_ram_bytes")
            tail = kwargs.get("zephyr_external_tail_reserve_bytes")
            if default <= 0 or default > kwargs["dram_bytes"]:
                raise BoardRegistryError(f"{where}: zephyr_default_ram_bytes must fit physical DRAM")
            if limit <= 0:
                raise BoardRegistryError(f"{where}: zephyr_link_limit_bytes must be positive")
            if (external is None) != (tail is None):
                raise BoardRegistryError(f"{where}: external RAM and tail reserve must be declared together")
            if external is not None and (external <= 0 or tail < 0 or external + tail >= kwargs["dram_bytes"]):
                raise BoardRegistryError(f"{where}: external layout leaves no room for weights in DRAM")
            if external is not None and external > limit:
                raise BoardRegistryError(f"{where}: external RAM region exceeds the linked-region limit")
        out[name] = Board(name=name, **kwargs)
    return out


#: The selected catalog, empty when no target owner selected one. Importing the generic
#: runtime must remain possible without any target installation.
BOARDS: dict[str, Board] = load_boards() if os.environ.get(BOARD_CATALOG_ENV) else {}


def board(name: str, **overrides) -> Board:
    """The selected catalog's descriptor for ``name``, with any field overridden.

    Unknown boards fail closed: invented DRAM, hart, console or vector facts can
    produce a silently wrong image. Add the board to a target-owned catalog first.
    """
    base = BOARDS.get(name)
    if base is None:
        raise BoardRegistryError(
            f"board {name!r} is not in the selected catalog"
            + (f" ({os.environ[BOARD_CATALOG_ENV]})" if os.environ.get(BOARD_CATALOG_ENV) else "")
            + f"; set {BOARD_CATALOG_ENV} and declare its hardware facts"
        )
    if not overrides:
        return base
    from dataclasses import replace

    return replace(base, **overrides)
