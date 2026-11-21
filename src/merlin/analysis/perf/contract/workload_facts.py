"""Machine geometry, address windows, numeric formats and scalar register facts."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from merlin.analysis.perf.contract.workload_programs import KernelOps

from dataclasses import dataclass, field
from typing import Any, Sequence


class WorkloadError(RuntimeError):
    """A required machine fact or faithfully encodable workload is unavailable.

    Refuse rather than guessing geometry, format, or control-flow behavior: a
    guessed program can run and return a plausible but invalid measurement.
    """


@dataclass(frozen=True)
class TileGeometry:
    """The compute array's own row/column count, which is equally the byte geometry of one tensor
    register (``rows x cols`` bytes). Discovered from the RTL, never written down: a guessed tile edge
    strides every operand load wrong and the kernel still runs."""

    rows: int
    cols: int
    source: str

    @property
    def register_bytes(self) -> int:
        return self.rows * self.cols


def tile_geometry(target: str) -> TileGeometry:
    """The target's compute-array geometry from mlc's RTL discovery. Raises when no array is discovered
    (fail closed -- there is no defensible default tile edge)."""
    from merlin.analysis.perf.contract.workload_facts import TileGeometry, WorkloadError
    from merlin.target.rtl import facts as RTLF

    arrays = (RTLF.load_facts(target).get("facts") or {}).get("arrays") or []
    mesh = next((a for a in arrays if a.get("rows") and a.get("cols")), None)
    if mesh is None:
        raise WorkloadError(
            f"{target!r}: RTL discovery reports no compute array, so the tile edge is not derivable; "
            f"refusing to assume one"
        )
    return TileGeometry(int(mesh["rows"]), int(mesh["cols"]), str(mesh.get("source") or "rtl_facts.arrays"))


def datapath_formats(target: str, *, accum_dtype: str | None = None) -> tuple[str, str]:
    """``(operand_dtype, accum_dtype)`` for the target's CONTRACTION unit, read from the compute-unit
    declaration in its capability manifest.

    The manifest describes each unit's datapath as a list of ``{in, weight, acc}`` combinations, so the
    pair is READ rather than inferred from the format names. A unit that declares several accumulate
    combinations (a wider PE variant alongside the native one) is resolved by the manifest's own
    declaration order unless the caller pins ``accum_dtype`` -- which it must when it selects a readout
    instruction for a specific accumulate format, since the two have to agree.

    Raises when no unit declares an accumulate combination: a reference built on a guessed accumulator
    grades a correct device as broken, so there is no defensible fallback."""
    from merlin.analysis.perf.contract.workload_facts import WorkloadError, _accumulate_from_rtl_datapaths
    from merlin.target.capabilities import manifest as CM

    man = CM.manifest_for(target)
    combos: list[tuple[str, str]] = []
    for unit in man.get("compute_units") or []:
        for combo in unit.get("accumulate") or []:
            src, acc = combo.get("in"), combo.get("acc")
            if src and acc:
                combos.append((str(src), str(acc)))
    if not combos:
        combos = _accumulate_from_rtl_datapaths(target)
    if not combos:
        raise WorkloadError(
            f"{target!r}: no compute unit in the capability manifest declares an accumulate datapath, "
            f"and its RTL facts evidence no operand/accumulator datapath pair either; "
            f"the operand and accumulate formats are not derivable and neither may be assumed"
        )
    if accum_dtype is not None:
        for src, acc in combos:
            if acc == accum_dtype:
                return src, acc
        raise WorkloadError(f"{target!r}: no declared datapath accumulates in {accum_dtype!r}; declared: {combos}")
    return combos[0]


def _accumulate_from_rtl_datapaths(target: str) -> list[tuple[str, str]]:
    """The (operand, accumulate) pair a target's OWN RTL evidences, when its manifest declares none.

    This is a DERIVATION, not a fallback default. A manifest may describe a unit's datapath as
    ``dtypes: [int8]`` -- the operand format only -- while the elaborated RTL carries both halves with
    the memory each was read from (gemmini: ``input i8`` from "scratchpad smem UInt<8>",
    ``accumulator i32`` from "AccumulatorMem SInt<32>"). Reading the accumulator out of the RTL is
    stronger evidence than a manifest line, not weaker, so it is consulted before refusing.

    What stays refused is a target whose RTL evidences no such pair: nothing is assumed, because a
    reference built on a guessed accumulator grades a correct device as broken.
    """
    try:
        from merlin.target.rtl import facts as _facts

        body = _facts.body_if_present(target)
    except Exception:  # noqa: BLE001 - no facts bundle is an absence of evidence, not an error here
        return []
    by_name = {
        str(d.get("name")): str(d.get("dtype") or "") for d in body.get("datapaths") or [] if isinstance(d, dict)
    }
    operand, accumulate = by_name.get("input"), by_name.get("accumulator")
    return [(operand, accumulate)] if operand and accumulate else []


def dram_window_bytes(target: str) -> int | None:
    """The size of the FINITE DRAM window the target's program runner models, or ``None`` when the runner
    does not publish one.

    This is the fact that turns a layer-scale run from a slow number into a WRONG one. The runner's
    memory is a power-of-two window and every address is reduced modulo it -- silently, with no flag in
    the returned result -- so a tensor larger than the window aliases onto itself and a pair of tensors
    whose reduced spans overlap corrupt each other. Read from the runner module rather than written down,
    and ``None`` (UNKNOWN, never 0) when it cannot be read, so :func:`alias_report` refuses to certify a
    footprint it could not check."""
    import importlib

    from merlin.target.rtl.extraction import mlc_discovery as _merlin_mlc_discovery_1
    from merlin.target.rtl.extraction import mlc_toolchain as mlc_bridge

    d = mlc_bridge.mlc_dir()
    if d is None:
        return None
    try:
        with _merlin_mlc_discovery_1._mlc_cwd():
            from mlc.discover import fingerprint

            backend_name = fingerprint.cosim_backend(mlc_bridge._arc_target(target))
            mod = importlib.import_module(f"mlc.backends.{backend_name}")
    except Exception:  # noqa: BLE001 — no registered program runner: honestly UNKNOWN
        return None
    # The runner publishes its window as a module-level constant; the name is the runner protocol's
    # (like ``run_program`` itself), not any one target's.
    win = getattr(mod, "_DRAM_WINDOW", None)
    if not isinstance(win, int) or win <= 0 or (win & (win - 1)):
        return None
    return int(win)


def _shipped_spec_docs(target: str):
    """The prose ISA references a target's descriptor ships (its green card and friends). These are the
    target's OWN declaration of its architectural parameters, which is where a fact absent from the RTL
    facts bundle can still be READ rather than assumed."""

    from merlin.common.paths import resolve_grant
    from merlin.target.isa.machine_model import _descriptor_for
    from merlin.target.registry.descriptor_loading import load_target_experiment

    desc = _descriptor_for(target)
    if desc is None:
        return []
    out = []
    for h in load_target_experiment(desc).isa_headers:
        if not str(h).endswith(".md"):
            continue
        p = resolve_grant(str(h))
        if p.is_file():
            out.append(p)
    return out


def _spec_table_int(text: str, *label_words: str) -> int | None:
    """The integer in the value cell of a two-column markdown table row whose label carries every word in
    ``label_words``. Structural: split the row into cells, the label into words, and read the first
    integer token of the value. No regex, and no assumption about the table's position in the document.
    ``None`` when no such row exists."""
    want = {w.upper() for w in label_words}
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 2:
            continue
        if not want <= set(cells[0].replace("`", " ").upper().split()):
            continue
        for tok in cells[1].replace("`", " ").split():
            cleaned = tok.replace("_", "").replace(",", "")
            try:
                return int(cleaned, 0)
            except ValueError:
                continue
    return None


def _spec_table_range(text: str, *label_words: str) -> tuple[int, int] | None:
    """The ``start ~ end`` pair in the value cell of a memory-map row whose label carries every word in
    ``label_words``. Same structural read as :func:`_spec_table_int`, for the rows that name a REGION
    rather than a count. ``None`` when the row is absent or carries fewer than two addresses."""
    want = {w.upper() for w in label_words}
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 2 or not want <= set(cells[0].replace("`", " ").upper().split()):
            continue
        found: list[int] = []
        for tok in cells[1].replace("`", " ").replace("~", " ").split():
            cleaned = tok.replace("_", "")
            if cleaned.lower().startswith("0x"):
                try:
                    found.append(int(cleaned, 16))
                except ValueError:
                    continue
        if len(found) >= 2:
            return found[0], found[1]
    return None


def instruction_memory_words(target: str, word_bytes: int) -> int | None:
    """How many instruction words the target's instruction memory holds, from the region row of the
    memory map in its shipped ISA reference. ``None`` (UNKNOWN, never a number) when the map does not
    name one -- the caller then reports the fit as unchecked rather than as passing.

    This is the fact that makes the loops necessary rather than tidy: a fully unrolled layer does not
    fit, and the failure is not a compile error but a program whose tail is never loaded."""
    from merlin.analysis.perf.contract.workload_facts import _shipped_spec_docs, _spec_table_range

    for doc in _shipped_spec_docs(target):
        rng = _spec_table_range(doc.read_text(encoding="utf-8", errors="replace"), "IMEM")
        if rng and rng[1] > rng[0]:
            return (rng[1] - rng[0]) // max(1, int(word_bytes))
    return None


def scalar_register_count(target: str, isa, ops: "KernelOps") -> tuple[int, str]:
    """How many scalar registers this generator may name, and where the bound came from.

    Two independent bounds, and the answer is the tighter of the two so neither can be exceeded:

    * what the ENCODING can express -- the width of the destination field of the target's own
      add-immediate, read from the derived field map. A kernel naming a register beyond it does not
      raise; it encodes a different register.
    * what the target DECLARES it has -- the scalar-register row of the architectural-parameters table
      in its shipped ISA reference. A file smaller than its field is legal and common.

    Raises when neither is readable: allocating from a guessed file size is the failure mode where a
    perfectly well-formed kernel addresses a register that is not there."""
    from merlin.analysis.perf.contract.workload_facts import (
        WorkloadError,
        _shipped_spec_docs,
        _spec_table_int,
    )

    bits = isa.fields_of(ops.add_imm).get("rd")
    bounds: list[tuple[int, str]] = []
    if bits:
        bounds.append(
            (1 << len(bits), f"the {len(bits)}-bit destination field of {ops.add_imm} in the derived encoding")
        )
    for doc in _shipped_spec_docs(target):
        declared = _spec_table_int(doc.read_text(encoding="utf-8", errors="replace"), "scalar", "registers")
        if declared:
            bounds.append((int(declared), f"the architectural-parameters table in {doc.name}"))
            break
    if not bounds:
        raise WorkloadError(
            f"{target!r}: the scalar register-file size is not derivable from either the encoding or the "
            f"shipped ISA reference; refusing to allocate registers against a guess"
        )
    bounds.sort()
    return bounds[0][0], " and ".join(src for _, src in bounds)


class _ScalarFile:
    """Role -> scalar-register index, allocated from a DERIVED bound.

    The indices are names for roles, but the BOUND is a hardware fact, and hardcoding the roles as
    literal indices silently assumes a file at least that large. Allocation counts up from the first
    writable register and refuses -- loudly, naming both numbers -- when the kernel needs more registers
    than the target has, instead of emitting a kernel that addresses one that is not there.

    Index ``zero_index`` is reserved: it is the register-file's fixed-zero register, the ABI convention
    every role here reads as a constant zero operand."""

    def __init__(self, target: str, isa, ops: "KernelOps", roles: Sequence[str], *, zero_index: int = 0):
        self.bound, self.provenance = scalar_register_count(target, isa, ops)
        self.zero = int(zero_index)
        need = len(roles) + 1
        if need > self.bound:
            raise WorkloadError(
                f"{target!r}: this kernel needs {need} scalar registers (one fixed zero plus "
                f"{len(roles)} roles) and the target has {self.bound} ({self.provenance}); refusing to "
                f"emit a kernel that names a register the machine does not have"
            )
        self._by_role = {r: self.zero + 1 + i for i, r in enumerate(roles)}

    def __getitem__(self, role: str) -> int:
        try:
            return self._by_role[role]
        except KeyError:
            raise WorkloadError(f"no scalar register allocated for role {role!r}") from None


@dataclass(frozen=True)
class MachineFacts:
    """Everything about the machine this generator needs, each field DERIVED and each carrying the
    derivation that produced it in :attr:`provenance`."""

    target: str
    isa: Any  # merlin.target.isa.machine_model.IsaModel
    tile: TileGeometry
    dram_base: int
    word_bytes: int  # bytes per machine word (VMEM addresses are word indices)
    operand_dtype: str
    accum_dtype: str
    dram_window: int | None
    imem_words: int | None = None
    provenance: dict[str, str] = field(default_factory=dict)

    @property
    def operand_bytes(self) -> int:
        from merlin.common import target_data as QF

        return max(1, int(QF.get(self.operand_dtype).element_bits or 8) // 8)

    @property
    def accum_bytes(self) -> int:
        from merlin.common import target_data as QF

        return max(1, int(QF.get(self.accum_dtype).element_bits or 8) // 8)

    @property
    def banks_per_tile(self) -> int:
        """How many tensor registers one accumulator tile occupies. A register holds ``rows x cols``
        BYTES, so a tile of a wider accumulate format spans that many registers -- 2 for a 2-byte format
        on a 1-byte-per-element register. Derived from the two, never the constant 2."""
        need = self.tile.rows * self.tile.cols * self.accum_bytes
        reg = self.tile.register_bytes
        if need % reg:
            raise WorkloadError(
                f"{self.target!r}: an accumulator tile is {need} B and a tensor register {reg} B; the "
                f"readout does not divide into whole registers, so the bank layout is not derivable"
            )
        return need // reg


def machine_facts(target: str, *, accum_dtype: str | None = None) -> MachineFacts:
    """Assemble the derived machine facts for ``target``. ``accum_dtype`` pins which declared accumulate
    datapath the workload uses, and must agree with the accumulator-readout instruction the caller
    selects. Raises :class:`WorkloadError` when the ISA model is empty (the target ships no ISA
    definition, so nothing can be encoded)."""
    from merlin.analysis.perf.contract.workload_facts import (
        MachineFacts,
        WorkloadError,
        datapath_formats,
        dram_window_bytes,
        instruction_memory_words,
        tile_geometry,
    )
    from merlin.target.isa.machine_model import isa_model_for_target

    isa = isa_model_for_target(target)
    if isa.is_empty():
        raise WorkloadError(f"{target!r} ships no ISA definition; no kernel can be encoded for it")
    from merlin.target.isa.machine_model import dram_base_for

    tile = tile_geometry(target)
    operand, accum = datapath_formats(target, accum_dtype=accum_dtype)
    return MachineFacts(
        target=target,
        isa=isa,
        tile=tile,
        dram_base=int(dram_base_for(target) or 0),
        word_bytes=max(1, isa.inst_width // 8),
        operand_dtype=operand,
        accum_dtype=accum,
        dram_window=dram_window_bytes(target),
        imem_words=instruction_memory_words(target, max(1, isa.inst_width // 8)),
        provenance={
            "isa": "merlin.target.isa.machine_model.isa_model_for_target (per-operand-bit probe of the "
            "target's own shipped ISA definition)",
            "tile": f"mlc RTL discovery ({tile.source})",
            "dram_base": "merlin.target.isa.machine_model.dram_base_for (the target's own memory map)",
            "word_bytes": "IsaModel.inst_width",
            "formats": "capability manifest compute-unit dtypes, ordered by quant-format element width",
            "dram_window": "the program runner's own published window size",
            "imem_words": "the instruction-memory region of the memory map in the shipped ISA reference",
        },
    )
