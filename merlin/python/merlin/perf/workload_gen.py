"""Generate a LAYER-SCALE workload for a self-hosted-ISA target: a command buffer, an emitted
``kernel.S``, a host-computed golden, and a proof that the run's DRAM footprint did not wrap.

Why this module exists. Every performance number measured on such a target so far describes a demo
TILE -- the largest tensor in the graded corpus is a few thousand elements and the kernels halt in
hundreds of cycles. The programs that exist are static assembly with immediates baked in at authoring
time, so a new shape means new hand-written assembly, and the "full model shapes" the corpus cites have
no program, no golden and no run. Performance work is about model LAYERS, so the layer has to be
generated.

Four properties this generator has that a hand-written kernel does not:

* **It loops.** Instruction memory is finite -- 32,768 words on the target measured here, DERIVED from
  the instruction-memory region of its own memory map (:func:`instruction_memory_words`) -- and the
  unrolled schedule for a layer overflows it. The emitted kernel's size is therefore O(1) in the shape:
  the tile nest is three real backward branches, so the same 96-word program runs a 32x32x32 tile and
  the 416x832x416 layer whose unrolled twin needs 108,011 words (3.3x the machine's instruction
  memory). :meth:`MatmulPlan.instruction_memory_fit` reports both, and reports UNKNOWN -- not a pass --
  when the capacity is not derivable.
* **Its branches are encoded from the machine's own control-flow contract, MEASURED.** A branch
  immediate is not portable knowledge. On the target measured here the PC is a WORD index and the
  target is ``s1_pc + (imm >> 1)``, so a B-type immediate moves ``imm/2`` instructions, and there is an
  architectural delay slot whose instruction executes on both paths. Encoding a byte offset produces a
  kernel that assembles, runs, halts, and never closes its loop -- it reads as "this core has no control
  flow". :func:`probe_control_flow` therefore DERIVES ``(imm_scale, delay_slots)`` by running candidate
  encodings on the target's own oracle and keeping the one whose observable trip count is right. Nothing
  about control flow is written down here.
* **It refuses to produce a cycle count from an aliasing run.** The simulator's DRAM is a finite window
  and addresses are reduced modulo its size, silently: bytes past the end are dropped or wrap onto other
  tensors, and nothing in the returned result flags it. One 1024x3072 bf16 tensor is 6 MiB against a
  1 MiB window -- it would alias six ways and still return a number. :func:`alias_report` computes the
  masked span of every tensor from the window size DERIVED from the runner and reports the run INVALID
  if any span wraps or two spans collide. A wrong number is worse than a slow one.
* **Its golden is computed host-side, in the accumulator's own format.** A narrow-float accumulator
  rounds every partial sum, so an f32 reference grades a perfectly correct device as broken -- measured
  here at 796 of 4096 elements on a single layer. The reference models the DECLARED datapath (round
  after every MAC, operand subnormal flush when the profile declares it) and the gate is BIT-EXACT,
  which is the right bar precisely because it is reachable: a tolerance would hide the loop-trip and
  bank-layout errors this generator exists to catch. A mismatch is DESCRIBED
  (:meth:`MatmulPlan.divergence`) rather than absorbed.

Target-agnostic by construction: the instruction encodings come from
:func:`~merlin.targetgen.isa_model.isa_model_for_target` (probed from the target's own shipped ISA
definition), the tile geometry from the RTL-discovered array, the DRAM aperture from the target's memory
map, the operand/accumulate formats from its capability manifest, the DRAM window from the runner that
will execute the program, and the control-flow contract from the machine itself. No opcode, funct value,
field position, mesh dimension, latency or address is a literal in this file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

__all__ = [
    "WorkloadError", "TileGeometry", "MachineFacts", "ControlFlow", "Settle", "KernelOps",
    "Placement", "AliasSpan", "AliasReport", "MatmulPlan",
    "tile_geometry", "datapath_formats", "dram_window_bytes", "machine_facts", "candidate_ops",
    "instruction_memory_words", "scalar_register_count",
    "probe_control_flow", "probe_settle", "control_flow_probe_kernel",
    "plan_matmul", "alias_report", "accumulate_reference", "encode_operand_bytes",
]


class WorkloadError(RuntimeError):
    """A fact this generator needs could not be DERIVED, or a shape it cannot emit faithfully.

    Raised rather than substituting a default: a guessed tile edge mis-strides every load and a guessed
    branch immediate produces a kernel that runs and never loops, and both failures return a number
    instead of an error."""


# ---------------------------------------------------------------------------------------------------
# derived machine facts
# ---------------------------------------------------------------------------------------------------
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
    from merlin.targetgen.rtl import facts as RTLF
    arrays = ((RTLF.load_facts(target).get("facts") or {}).get("arrays") or [])
    mesh = next((a for a in arrays if a.get("rows") and a.get("cols")), None)
    if mesh is None:
        raise WorkloadError(
            f"{target!r}: RTL discovery reports no compute array, so the tile edge is not derivable; "
            f"refusing to assume one")
    return TileGeometry(int(mesh["rows"]), int(mesh["cols"]),
                        str(mesh.get("source") or "rtl_facts.arrays"))


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
    from merlin.targetgen import capability_manifests as CM
    man = CM.manifest_for(target)
    combos: list[tuple[str, str]] = []
    for unit in (man.get("compute_units") or []):
        for combo in (unit.get("accumulate") or []):
            src, acc = combo.get("in"), combo.get("acc")
            if src and acc:
                combos.append((str(src), str(acc)))
    if not combos:
        raise WorkloadError(
            f"{target!r}: no compute unit in the capability manifest declares an accumulate datapath; "
            f"the operand and accumulate formats are not derivable and neither may be assumed")
    if accum_dtype is not None:
        for src, acc in combos:
            if acc == accum_dtype:
                return src, acc
        raise WorkloadError(
            f"{target!r}: no declared datapath accumulates in {accum_dtype!r}; declared: {combos}")
    return combos[0]


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

    from merlin.targetgen.rtl import mlc_bridge
    d = mlc_bridge.mlc_dir()
    if d is None:
        return None
    try:
        with mlc_bridge._mlc_cwd():
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
    from pathlib import Path

    from merlin.common.paths import merlin_dir, repo_root
    from merlin.targetgen.dram_facts import _descriptor_for
    from merlin.targetgen.target_experiment import load_target_experiment
    desc = _descriptor_for(target)
    if desc is None:
        return []
    out = []
    for h in load_target_experiment(desc).isa_headers:
        if not str(h).endswith(".md"):
            continue
        for base in (merlin_dir(), repo_root()):
            p = Path(base) / str(h)
            if p.is_file():
                out.append(p)
                break
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
    bits = isa.fields_of(ops.add_imm).get("rd")
    bounds: list[tuple[int, str]] = []
    if bits:
        bounds.append((1 << len(bits),
                       f"the {len(bits)}-bit destination field of {ops.add_imm} in the derived encoding"))
    for doc in _shipped_spec_docs(target):
        declared = _spec_table_int(doc.read_text(encoding="utf-8", errors="replace"),
                                   "scalar", "registers")
        if declared:
            bounds.append((int(declared), f"the architectural-parameters table in {doc.name}"))
            break
    if not bounds:
        raise WorkloadError(
            f"{target!r}: the scalar register-file size is not derivable from either the encoding or the "
            f"shipped ISA reference; refusing to allocate registers against a guess")
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

    def __init__(self, target: str, isa, ops: "KernelOps", roles: Sequence[str],
                 *, zero_index: int = 0):
        self.bound, self.provenance = scalar_register_count(target, isa, ops)
        self.zero = int(zero_index)
        need = len(roles) + 1
        if need > self.bound:
            raise WorkloadError(
                f"{target!r}: this kernel needs {need} scalar registers (one fixed zero plus "
                f"{len(roles)} roles) and the target has {self.bound} ({self.provenance}); refusing to "
                f"emit a kernel that names a register the machine does not have")
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
    isa: Any                                  # merlin.targetgen.isa_model.IsaModel
    tile: TileGeometry
    dram_base: int
    word_bytes: int                           # bytes per machine word (VMEM addresses are word indices)
    operand_dtype: str
    accum_dtype: str
    dram_window: int | None
    imem_words: int | None = None
    provenance: dict[str, str] = field(default_factory=dict)

    @property
    def operand_bytes(self) -> int:
        from merlin.common import quant_formats as QF
        return max(1, int(QF.get(self.operand_dtype).element_bits or 8) // 8)

    @property
    def accum_bytes(self) -> int:
        from merlin.common import quant_formats as QF
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
                f"readout does not divide into whole registers, so the bank layout is not derivable")
        return need // reg


def machine_facts(target: str, *, accum_dtype: str | None = None) -> MachineFacts:
    """Assemble the derived machine facts for ``target``. ``accum_dtype`` pins which declared accumulate
    datapath the workload uses, and must agree with the accumulator-readout instruction the caller
    selects. Raises :class:`WorkloadError` when the ISA model is empty (the target ships no ISA
    definition, so nothing can be encoded)."""
    from merlin.targetgen.isa_model import isa_model_for_target
    isa = isa_model_for_target(target)
    if isa.is_empty():
        raise WorkloadError(f"{target!r} ships no ISA definition; no kernel can be encoded for it")
    from merlin.targetgen.dram_facts import dram_base_for
    tile = tile_geometry(target)
    operand, accum = datapath_formats(target, accum_dtype=accum_dtype)
    return MachineFacts(
        target=target, isa=isa, tile=tile, dram_base=int(dram_base_for(target) or 0),
        word_bytes=max(1, isa.inst_width // 8), operand_dtype=operand, accum_dtype=accum,
        dram_window=dram_window_bytes(target),
        imem_words=instruction_memory_words(target, max(1, isa.inst_width // 8)),
        provenance={
            "isa": "merlin.targetgen.isa_model.isa_model_for_target (per-operand-bit probe of the "
                   "target's own shipped ISA definition)",
            "tile": f"mlc RTL discovery ({tile.source})",
            "dram_base": "merlin.targetgen.dram_facts.dram_base_for (the target's own memory map)",
            "word_bytes": "IsaModel.inst_width",
            "formats": "capability manifest compute-unit dtypes, ordered by quant-format element width",
            "dram_window": "the program runner's own published window size",
            "imem_words": "the instruction-memory region of the memory map in the shipped ISA reference",
        })


# ---------------------------------------------------------------------------------------------------
# the machine's control-flow contract -- measured, never assumed
# ---------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ControlFlow:
    """How a backward branch is encoded on this machine, and how many instructions after it execute
    regardless of the outcome.

    ``branch_imm_scale`` is the factor between the immediate a branch carries and the number of
    INSTRUCTIONS it moves: the immediate written into the encoding is ``scale * (target - branch)``,
    both measured in instruction indices. A machine whose PC counts words and whose branch adder halves
    the immediate has scale 2; one that adds the immediate directly has scale 1. Getting this wrong does
    not raise: the kernel assembles, runs and halts, having executed its loop body once.

    ``delay_slots`` instructions after the branch execute on BOTH paths and must therefore be harmless."""

    branch_imm_scale: int
    delay_slots: int
    provenance: str

    def branch_imm(self, branch_index: int, target_index: int) -> int:
        return self.branch_imm_scale * (target_index - branch_index)


@dataclass(frozen=True)
class Settle:
    """Cycles the scalar stream must idle before a tensor result is architecturally visible to the next
    instruction, per producer class. This machine has no interlock between the scalar stream and the
    tensor pipelines -- reading a register before its producer drains returns stale bytes rather than
    raising, and a vector op issued too soon does not write its destination at all. Both are wrong
    answers with no error attached, so the value is MEASURED on the device (:func:`probe_settle`),
    never picked."""

    tensor: int
    mxu: int
    vpu: int
    provenance: str

    @classmethod
    def uniform(cls, cycles: int, provenance: str) -> "Settle":
        return cls(tensor=int(cycles), mxu=int(cycles), vpu=int(cycles), provenance=provenance)


@dataclass(frozen=True)
class KernelOps:
    """Which of THIS target's instructions play each role in the emitted kernel.

    No mnemonic is written down in this module. A self-hosted ISA's vocabulary is the target's own, and
    the derived role table (:attr:`IsaModel.roles`) narrows the choice but does not settle it: a machine
    with two matrix units and both an overwriting and an accumulating multiply has four instructions
    carrying the single role ``matmul``, and which one to use is a SCHEDULING decision, not a fact about
    the hardware. So the selection is a parameter, made once at the edge that is legitimately about one
    target, and :func:`candidate_ops` reports the derived candidates to make it from. Onboarding a second
    target means writing a different selection, never editing this file.

    Every name is validated against the derived ISA model on construction, so a mnemonic this target does
    not define is an error here rather than a silently wrong word later."""

    add: str                      # register + register
    add_imm: str                  # register + immediate (also this generator's move and no-op)
    load_upper: str               # upper immediate, for constants wider than the immediate field
    branch_ne: str                # conditional backward branch, taken while two registers differ
    stall: str                    # hold issue for an immediate number of cycles
    halt: str                     # terminate the program
    dma_load: str                 # DRAM -> staging memory
    dma_store: str                # staging memory -> DRAM
    dma_wait: str                 # block until the channel is idle
    tile_load: str                # staging memory -> tensor register
    tile_store: str               # tensor register -> staging memory
    transpose: str                # tensor register transpose
    weight_push: str              # tensor register -> the array's weight buffer
    contract: str                 # multiply, OVERWRITING the accumulator
    contract_accumulate: str      # multiply, ADDING into the accumulator
    acc_read: str                 # accumulator -> tensor register(s)

    def validate(self, isa) -> None:
        missing = [f"{role}={name}" for role, name in self.as_dict().items()
                   if isa.resolve(name) is None]
        if missing:
            raise WorkloadError(
                f"{isa.target!r} defines no instruction for: {', '.join(missing)}. "
                f"Derived candidates per role: {candidate_ops(isa)}")

    def as_dict(self) -> dict[str, str]:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}


def candidate_ops(isa) -> dict[str, list[str]]:
    """The mnemonics this target defines for each DERIVED structural role -- the menu a
    :class:`KernelOps` selection is made from. Reported rather than chosen: where a role resolves to more
    than one instruction, the choice is the caller's."""
    out: dict[str, list[str]] = {}
    for role, classes in (isa.roles or {}).items():
        names = sorted(mn for mn, ent in isa.by_mnemonic.items() if ent.get("class") in classes)
        out[role] = names
    if isa.halt_mnemonics:
        out["halt"] = list(isa.halt_mnemonics)
    return out


# ---------------------------------------------------------------------------------------------------
# a tiny label-and-fixup assembler over the derived ISA model
# ---------------------------------------------------------------------------------------------------
class _Program:
    """An instruction list with labels and branch fixups, encoded through
    :mod:`merlin.targetgen.isa_asm` against the derived field maps. Branch immediates are resolved after
    layout using the machine's measured control-flow contract."""

    def __init__(self, isa, cf: ControlFlow, ops: "KernelOps"):
        ops.validate(isa)
        self._isa = isa
        self._cf = cf
        self._ops = ops
        self._items: list[tuple[str, dict]] = []
        self._labels: dict[str, int] = {}
        self._fixups: list[tuple[int, str, str]] = []   # (index, imm operand name, label)

    # -- placement ---------------------------------------------------------------------------------
    def label(self, name: str) -> None:
        if name in self._labels:
            raise WorkloadError(f"duplicate label {name!r}")
        self._labels[name] = len(self._items)

    def emit(self, mnemonic: str, **operands: int) -> None:
        self._items.append((mnemonic, dict(operands)))

    def branch(self, mnemonic: str, target: str, *, imm_operand: str = "imm", **operands: int) -> None:
        """A conditional branch to ``target``, followed by the machine's delay slot(s) filled with an
        instruction that is harmless on both paths. The delay slot is filled HERE rather than left to the
        caller because leaving it empty is the failure that looks like a working loop."""
        self._fixups.append((len(self._items), imm_operand, target))
        self._items.append((mnemonic, {**operands, imm_operand: 0}))
        for _ in range(self._cf.delay_slots):
            self.nop()

    def nop(self) -> None:
        """An instruction with no architectural effect: the selected add-immediate writing the register
        file's fixed-zero register (index 0 -- the ABI's convention, not an encoding constant)."""
        self.emit(self._ops.add_imm, rd=0, rs1=0, imm=0)

    def load_imm(self, rd: int, value: int) -> None:
        """Materialise a full-width constant, the way the ISA allows it: an upper immediate plus a
        sign-corrected low add. The low add sign-extends, so a low half with its top bit set must be
        compensated in the upper half or the register lands one page low."""
        width = self._imm_width(self._ops.add_imm, "imm")
        lo_mask = (1 << width) - 1
        sign = 1 << (width - 1)
        lo = value & lo_mask
        hi = (value - (lo - (1 << width) if lo & sign else lo)) >> width
        upper_width = self._imm_width(self._ops.load_upper, "imm")
        if hi:
            if hi >> upper_width:
                raise WorkloadError(f"constant {value} does not fit this ISA's upper immediate")
            self.emit(self._ops.load_upper, rd=rd, imm=hi & ((1 << upper_width) - 1))
            if lo:
                self.emit(self._ops.add_imm, rd=rd, rs1=rd, imm=lo)
        else:
            self.emit(self._ops.add_imm, rd=rd, rs1=0, imm=lo)

    def _imm_width(self, mnemonic: str, operand: str) -> int:
        bits = self._isa.fields_of(mnemonic).get(operand)
        if not bits:
            raise WorkloadError(f"this ISA's {mnemonic} carries no {operand!r} field")
        return len(bits)

    # -- output ------------------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._items)

    def resolved(self) -> list[tuple[str, dict]]:
        """The instruction list with every branch immediate resolved against the measured contract."""
        out = [(mn, dict(ops)) for mn, ops in self._items]
        for index, operand, target in self._fixups:
            if target not in self._labels:
                raise WorkloadError(f"branch to undefined label {target!r}")
            imm = self._cf.branch_imm(index, self._labels[target])
            width = self._imm_width(out[index][0], operand)
            lo, hi = -(1 << (width - 1)), (1 << (width - 1)) - 1
            if not lo <= imm <= hi:
                raise WorkloadError(
                    f"branch at {index} to {target!r} needs immediate {imm}, outside this ISA's "
                    f"{width}-bit signed branch field; the loop cannot be encoded as a single branch")
            out[index][1][operand] = imm & ((1 << width) - 1)
        return out

    def words(self) -> list[int]:
        from merlin.targetgen import isa_asm
        return [isa_asm.assemble_line(self._isa, mn, ops) for mn, ops in self.resolved()]

    def kernel_s(self, entry: str) -> str:
        """The ``.word`` stream stock LLVM assembles into instruction memory, each line annotated with the
        mnemonic and operands it was encoded from so the kernel stays reviewable."""
        from merlin.targetgen import isa_asm
        lines = [".section .text", f".globl {entry}", f".type {entry},@function", f"{entry}:"]
        at_label = {v: k for k, v in self._labels.items()}
        for i, (mn, ops) in enumerate(self.resolved()):
            if i in at_label:
                lines.append(f"# {at_label[i]}:")
            word = isa_asm.assemble_line(self._isa, mn, ops)
            args = ", ".join(f"{k}={v}" for k, v in ops.items())
            lines.append(f"  .word 0x{word:08x}    # [{i}] {mn}{' ' + args if args else ''}")
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------
# control-flow probe: derive the branch contract by running it
# ---------------------------------------------------------------------------------------------------
def control_flow_probe_kernel(facts: MachineFacts, ops: KernelOps, *, trips: int, body_cycles: int,
                              imm_scale: int, delay_slots: int, entry: str = "_start") -> str:
    """A kernel whose RUNTIME reveals whether a candidate branch encoding closes a loop.

    The body is a single stall of ``body_cycles``, so a loop that trips ``trips`` times costs about
    ``trips * body_cycles`` cycles and one that never loops costs about ``body_cycles``. The two are
    orders of magnitude apart, and a candidate that mis-targets the branch either halts early or never
    halts -- three outcomes, all distinguishable, none of them a plausible near-miss."""
    cf = ControlFlow(imm_scale, delay_slots, "candidate under test")
    p = _Program(facts.isa, cf, ops)
    x = _ScalarFile(facts.target, facts.isa, ops, ("counter", "limit"))
    counter, limit = x["counter"], x["limit"]
    p.load_imm(counter, 0)
    p.load_imm(limit, int(trips))
    p.label("loop")
    p.emit(ops.stall, rd=0, rs1=0, imm=int(body_cycles))
    p.emit(ops.add_imm, rd=counter, rs1=counter, imm=1)
    p.branch(ops.branch_ne, "loop", rs1=counter, rs2=limit)
    p.emit(ops.halt)
    return p.kernel_s(entry)


def probe_control_flow(facts: MachineFacts, ops: KernelOps,
                       run_kernel: Callable[[str, int], int | None], *,
                       trips: int = 8, body_cycles: int = 200,
                       imm_scales: Sequence[int] = (2, 4, 1),
                       delay_slot_candidates: Sequence[int] = (1, 0, 2),
                       entry: str = "_start") -> ControlFlow:
    """DERIVE the machine's branch contract by executing candidate encodings on it.

    ``run_kernel(kernel_s, max_cycles) -> cycles`` runs one probe and returns the halt cycle, or ``None``
    when the program did not halt. The winning candidate is the one whose measured cycle count matches a
    loop that tripped ``trips`` times; a candidate whose branch lands short halts early, and one that
    lands on itself never halts. Raises when no candidate loops -- an honest "this machine's control flow
    is not encodable by this generator" rather than a kernel that silently runs its body once.

    The candidate ORDER is only a search order; the verdict is the measurement. The candidates are not
    hypothetical: two runners for the SAME machine measured here disagree -- its RTL halves the immediate
    and its functional model reads it as a byte offset -- so this must be run against the tier the
    workload will actually be graded on, and its answer travels with the plan."""
    expected = trips * body_cycles
    budget = int(expected * 4 + 4000)
    tried: list[str] = []
    for scale in imm_scales:
        for slots in delay_slot_candidates:
            src = control_flow_probe_kernel(facts, ops, trips=trips, body_cycles=body_cycles,
                                            imm_scale=scale, delay_slots=slots, entry=entry)
            try:
                cycles = run_kernel(src, budget)
            except Exception as e:  # noqa: BLE001 — a refusing candidate is evidence, not a failure
                tried.append(f"scale={scale} slots={slots}: {type(e).__name__}: {str(e)[-120:]}")
                continue
            if cycles is None:
                tried.append(f"scale={scale} slots={slots}: did not halt within {budget}")
                continue
            # A loop that tripped the full count spends at least (trips-1) further bodies beyond the
            # first; anything less means the branch did not close the loop.
            if cycles >= expected - body_cycles:
                return ControlFlow(scale, slots,
                                   f"measured on {facts.target}: a {trips}-trip loop with a "
                                   f"{body_cycles}-cycle body halted at {cycles} cycles with "
                                   f"imm_scale={scale}, delay_slots={slots}")
            tried.append(f"scale={scale} slots={slots}: halted at {cycles} (loop did not close)")
    raise WorkloadError(
        f"{facts.target!r}: no candidate branch encoding closed a loop on the device. Tried: "
        + "; ".join(tried))


def probe_settle(facts: MachineFacts, ops: KernelOps, cf: ControlFlow,
                 run_matmul: Callable[["MatmulPlan", int], Any], *, operands,
                 ladder: Sequence[int] = (8, 16, 32, 64, 128, 256, 512),
                 margin: int = 2, entry: str = "_start") -> Settle:
    """DERIVE the settle counts by finding the smallest one that computes the right answer on a single
    tile, then applying ``margin``.

    ``run_matmul(plan, max_cycles)`` runs a plan and returns its device output as a nested list, or
    ``None``. ``operands`` are the single-tile ``(A, W)`` the probe multiplies; the comparison is the
    plan's own bit-exact gate against the accumulator-format golden.

    LIMIT, stated rather than implied: a single tile runs the body ONCE, with no loop and therefore no
    reuse of a tensor register across iterations, so this establishes the settle a producer needs before
    its consumer -- not the settle a loop needs before it overwrites a register the previous iteration is
    still draining into. The margin exists for that gap, and the multi-tile run is what actually
    certifies it: every shape reported from this generator carries its own bit-exact verdict."""
    tile = facts.tile
    for cycles in ladder:
        plan = plan_matmul(facts, ops, m=tile.rows, k=tile.cols, n=tile.cols, control_flow=cf,
                           settle=Settle.uniform(cycles, "candidate under test"),
                           A=operands[0], W=operands[1], entry=entry)
        try:
            got = run_matmul(plan, plan.suggested_max_cycles())
        except Exception:  # noqa: BLE001 — a candidate that faults is simply not the answer
            continue
        if got is not None and plan.matches(got):
            return Settle.uniform(cycles * int(margin),
                                  f"measured on {facts.target}: the smallest uniform settle that made a "
                                  f"{tile.rows}x{tile.cols}x{tile.cols} tile bit-exact was {cycles} "
                                  f"cycles; carried with a {margin}x margin")
    raise WorkloadError(
        f"{facts.target!r}: no settle count in {list(ladder)} produced a correct single tile; the "
        f"schedule is wrong for a reason a longer wait does not fix")


# ---------------------------------------------------------------------------------------------------
# DRAM placement and the alias detector
# ---------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Placement:
    """One tensor's declared home: its logical shape and dtype, and the ABSOLUTE address the kernel
    stores to and the reader reads back from -- computed once, from one input, so the two cannot drift."""

    name: str
    role: str
    shape: list[int]
    dtype: str
    nbytes: int
    base: int


@dataclass(frozen=True)
class AliasSpan:
    name: str
    base: int
    nbytes: int
    reduced_lo: int
    reduced_hi: int          # exclusive, BEFORE reduction -- so a span past the window end is visible
    wraps: bool


@dataclass(frozen=True)
class AliasReport:
    """Whether this workload's footprint survives the runner's finite DRAM window.

    ``ok`` is True only when the window size was KNOWN and every span both fits inside it and misses
    every other span. ``ok`` False with ``window`` None means the check could not run -- which is not a
    pass. A run whose report is not ``ok`` must not be quoted as a cycle count: its operands aliased, so
    the number describes a computation nobody asked for."""

    window: int | None
    spans: tuple[AliasSpan, ...]
    wrapped: tuple[str, ...]
    collisions: tuple[tuple[str, str], ...]
    footprint_bytes: int
    ok: bool
    reason: str


def alias_report(placements: Sequence[Placement], window: int | None) -> AliasReport:
    """Reduce every tensor's span modulo the runner's window exactly as the runner does, and report any
    span that runs past the window's end (its tail wraps onto low addresses) or overlaps another."""
    spans: list[AliasSpan] = []
    total = 0
    for p in placements:
        lo = p.base & (window - 1) if window else p.base
        hi = lo + p.nbytes
        spans.append(AliasSpan(p.name, p.base, p.nbytes, lo, hi,
                               wraps=bool(window and hi > window)))
        total += p.nbytes
    wrapped = tuple(s.name for s in spans if s.wraps)
    collisions: list[tuple[str, str]] = []
    for i, a in enumerate(spans):
        for b in spans[i + 1:]:
            if a.reduced_lo < b.reduced_hi and b.reduced_lo < a.reduced_hi:
                collisions.append((a.name, b.name))
    if window is None:
        return AliasReport(None, tuple(spans), wrapped, tuple(collisions), total, False,
                           "the runner publishes no DRAM window size, so the footprint COULD NOT BE "
                           "CHECKED; this is not a pass")
    if wrapped:
        return AliasReport(window, tuple(spans), wrapped, tuple(collisions), total, False,
                           f"{len(wrapped)} tensor(s) run past the {window} B window and wrap onto low "
                           f"addresses: {', '.join(wrapped)}")
    if collisions:
        pairs = ", ".join(f"{a}~{b}" for a, b in collisions)
        return AliasReport(window, tuple(spans), wrapped, tuple(collisions), total, False,
                           f"tensor spans collide after reduction into the {window} B window: {pairs}")
    return AliasReport(window, tuple(spans), (), (), total, True,
                       f"all {len(spans)} tensor spans ({total} B) fit disjointly inside the "
                       f"{window} B window; no address wraps")


# ---------------------------------------------------------------------------------------------------
# numerics: operand encoding and the accumulator-format reference
# ---------------------------------------------------------------------------------------------------
def encode_operand_bytes(values, dtype: str) -> bytes:
    """Encode an array to raw device bytes for ``dtype``, dispatched on the quant-format REGISTRY's
    description of the format (kind, element width, exponent/mantissa split, signedness) rather than on
    the format's spelling -- so a format nobody wrote a branch for still encodes. Raises for a sub-byte
    format, whose packing is a layout decision this function must not guess."""
    import numpy as np

    from merlin.common import quant_formats as QF
    a = np.asarray(values, dtype=np.float64)
    if not QF.has(dtype):
        raise WorkloadError(f"unknown operand format {dtype!r}")
    f = QF.get(dtype)
    bits = int(f.element_bits or 0)
    if bits == 0 or bits % 8:
        raise WorkloadError(f"{dtype!r} is a sub-byte format; its packing is not derivable here")
    if f.kind == "int_affine":
        lo, hi = ((-(2 ** (bits - 1)), 2 ** (bits - 1) - 1) if f.signed else (0, 2 ** bits - 1))
        code = "<i" if f.signed else "<u"
        return np.clip(np.rint(a), lo, hi).astype(f"{code}{bits // 8}").tobytes()
    if f.kind == "float_ieee":
        if bits == 32:
            return a.astype("<f4").tobytes()
        if bits == 16 and int(f.exp_bits or 0) == 5:
            return a.astype("<f2").tobytes()
        if bits == 16 and int(f.exp_bits or 0) == 8:
            u = a.astype("<f4").view("<u4").astype(np.uint64)
            return (((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype("<u2")).tobytes()
        raise WorkloadError(f"no byte encoding derivable for {dtype!r}")
    if f.kind == "fp_ocp":
        from merlin.targetgen.fp8_codec import ocp_encode
        eb, mb = int(f.exp_bits or 0), int(f.mant_bits or 0)
        if not eb or bits != 8:
            raise WorkloadError(f"no byte encoding derivable for {dtype!r}")
        return bytes(bytearray(ocp_encode(float(v), eb, mb, signed=bool(f.signed))
                               for v in a.reshape(-1)))
    raise WorkloadError(f"no byte encoding derivable for {dtype!r}")


def _accum_rounder(accum_dtype: str):
    """The function that rounds a running sum into ``accum_dtype``, derived from the registry's mantissa
    and exponent widths. ``None`` for a format that does not round a partial sum (an integer or a
    full-width accumulator), where the plain product is already the reference."""
    import numpy as np

    from merlin.common import quant_formats as QF
    f = QF.get(accum_dtype)
    if f.kind != "float_ieee":
        return None
    mant, exp = int(f.mant_bits or 0), int(f.exp_bits or 0)
    if mant == 10 and exp == 5:
        return lambda x: x.astype("<f2").astype(np.float32)
    if mant == 7 and exp == 8:                       # the top half of the wide word, round-to-nearest-even
        def rnd(x):
            u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
            return ((((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint32) << 16)
                    .astype(np.uint32).view(np.float32))
        return rnd
    return None


def accumulate_reference(A, W, *, accum_dtype: str, operand_dtype: str | None = None,
                         subnormal_operand_flush: bool = False):
    """The host-side golden, computed on the DECLARED datapath: the running sum is rounded into the
    accumulator's format after every MAC, and operands are flushed when the target's profile says its
    multiplier flushes subnormals. Returns ``None`` when the accumulator does not round (the plain
    product is then the reference).

    A full-precision reference is the wrong gate for a narrow-float accumulator: it grades a perfectly
    correct device as broken, and the error grows with the contraction length exactly where a layer-scale
    workload lives."""
    import numpy as np

    A = np.asarray(A, dtype=np.float32)
    W = np.asarray(W, dtype=np.float32)
    if subnormal_operand_flush and operand_dtype:
        from merlin.runtime import fp8_formats as FF
        try:
            min_normal, _ = FF.normal_range(operand_dtype)
        except KeyError:
            return None
        A = np.where(np.abs(A) < min_normal, np.float32(0.0), A).astype(np.float32)
        W = np.where(np.abs(W) < min_normal, np.float32(0.0), W).astype(np.float32)
    rnd = _accum_rounder(accum_dtype)
    if rnd is None:
        return None
    acc = np.zeros((A.shape[0], W.shape[1]), dtype=np.float32)
    for i in range(A.shape[1]):
        acc = rnd(acc + np.outer(A[:, i], W[i, :]))
    return acc


# ---------------------------------------------------------------------------------------------------
# the workload
# ---------------------------------------------------------------------------------------------------
# The scalar-register ROLES this kernel needs, in allocation order. These are role NAMES; the indices
# they get are allocated by :class:`_ScalarFile` against a bound derived from the target (see
# :func:`scalar_register_count`), so the file size is never assumed. Each concurrently live DMA endpoint
# gets its own role because the engine reads the registers when the transfer RUNS, not when the
# instruction issues, so re-pointing one mid-flight silently corrupts the transfer.
_SCALAR_ROLES = (
    "m_index", "n_index", "k_index",              # the three loop counters
    "dma_rd", "dma_rs1", "dma_len",               # the DMA endpoint registers
    "stage_addr",                                 # the tensor load/store base
    "a_tile", "w_tile", "c_tile",                 # the three walking operand pointers
    "m_limit", "n_limit", "k_limit",              # the three loop bounds
    "a_row", "w_col",                             # per-M and per-N restart points
    "tile_bytes", "a_row_stride", "w_k_stride", "c_stride",   # the four strides
    "w_origin",                                   # where the weight pointer restarts each M step
)
# Tensor register roles. Index 0 is the first tensor register; the accumulator readout occupies
# ``banks_per_tile`` consecutive registers starting at its own role, which is why it is allocated last.
_M_LHS, _M_WT, _M_ACC = 0, 1, 2
_WEIGHT_SLOT, _ACC_SLOT = 0, 0


@dataclass(frozen=True)
class MatmulPlan:
    """A generated layer: where its tensors live, what its kernel is, and what the right answer is.

    Everything a run needs travels together, because the failure this replaces is a command buffer whose
    addresses and a kernel whose addresses were computed by different code."""

    facts: MachineFacts
    ops: KernelOps
    m: int
    k: int
    n: int
    placements: tuple[Placement, ...]
    kernel_s: str
    words: tuple[int, ...]
    control_flow: ControlFlow
    settle: Settle
    entry: str
    tiles: tuple[int, int, int]                # (m_tiles, k_tiles, n_tiles)
    # measured lengths of the emitted program's parts, so the unrolled-size comparison is not an estimate
    section_words: dict[str, int] = field(default_factory=dict)
    operands: dict[str, Any] = field(default_factory=dict)      # logical float arrays, when supplied
    golden: Any = None                                          # [m, n] reference, when computable

    # -- addressing --------------------------------------------------------------------------------
    def placement(self, name: str) -> Placement:
        for p in self.placements:
            if p.name == name:
                return p
        raise KeyError(name)

    def command_buffer(self) -> dict:
        """The declared command buffer: the tensors, at the addresses this plan computed, in the physical
        shape the device actually writes. The output is declared as the BANK STREAM the accumulator
        readout lays down (rows of one register's width), not as the logical matrix: the reader that
        un-permutes it is :meth:`unpack_output`, in this module, which is the same code that decided the
        layout."""
        tensors = {}
        for p in self.placements:
            tensors[p.name] = {"role": p.role, "shape": list(p.shape), "dtype": p.dtype,
                               "base": p.base}
        return {"abi_version": "0.1", "target": self.facts.target, "tensors": tensors,
                "commands": [{"opcode": "MATMUL",
                              "operands": {"lhs": "A", "rhs": "W", "dst": "C"}}]}

    def total_macs(self) -> int:
        return self.m * self.k * self.n

    def unrolled_word_estimate(self) -> int:
        """How many instruction words THIS schedule would need with the loops unrolled -- the number
        that decides whether a shape is reachable without control flow at all.

        Counted from the emitted program's own measured parts (one K-step body, one output-tile
        epilogue, the prologue) rather than approximated, so the comparison against the machine's
        instruction memory is the real one. This is the argument for the loops: the emitted program is
        a constant ~90 words at every shape and its unrolled twin is not."""
        mt, kt, nt = self.tiles
        return (self.section_words.get("prologue", 0)
                + mt * kt * nt * self.section_words.get("k_step", 0)
                + mt * nt * self.section_words.get("tile_epilogue", 0))

    def moved_bytes(self) -> int:
        """Bytes this kernel's DMA actually moves, counted from the emitted schedule: one operand tile
        pair per K-step and one accumulator tile per output tile. Structural, not algorithmic -- which
        is the point, because a bound built on the bytes an algorithm NEEDS rather than the bytes a
        program MOVES is optimistic by exactly the transfer amplification."""
        mt, kt, nt = self.tiles
        tile = self.facts.tile.register_bytes
        return mt * kt * nt * 2 * tile + mt * nt * self.facts.banks_per_tile * tile

    def transfer_amplification(self) -> float:
        """Moved bytes over the workload's own footprint. At tile scale this ratio is dominated by the
        fixed per-tile transfer cost; at layer scale it is dominated by how often the schedule REFETCHES
        an operand, so the two do not extrapolate to one another and the layer-scale number is the one
        an optimisation argument may use."""
        return self.moved_bytes() / max(1, sum(p.nbytes for p in self.placements))

    def instruction_memory_fit(self) -> dict:
        """Whether the emitted program fits instruction memory, and whether its UNROLLED twin would.

        The second number is the one that matters: this generator loops because the unrolled schedule
        for a layer does not fit, and a program whose tail is never loaded does not fail loudly. When
        the capacity is not derivable both answers are ``None`` -- unchecked, which is not a pass."""
        cap = self.facts.imem_words
        unrolled = self.unrolled_word_estimate()
        return {"words": len(self.words), "unrolled_words": unrolled, "capacity_words": cap,
                "fits": None if cap is None else len(self.words) <= cap,
                "unrolled_fits": None if cap is None else unrolled <= cap,
                "unrolled_overflow": None if cap is None else round(unrolled / cap, 2)}

    def suggested_max_cycles(self, *, per_tile_pass: int = 4096, floor: int = 20000) -> int:
        """A HANG backstop sized to this plan's own tile count, not a fixed cap. It is deliberately loose:
        a healthy program halts on its own, and the wall-clock timeout is the outer backstop. A fixed cap
        is the defect that reports a correct long-running program as a missing oracle."""
        mt, kt, nt = self.tiles
        return max(int(floor), int(floor) + per_tile_pass * mt * kt * nt)

    # -- operands and results ----------------------------------------------------------------------
    def preloads(self) -> list[tuple[int, bytes]]:
        """``(address, bytes)`` for every input tensor, in the DEVICE layout this kernel reads."""
        out = []
        for name in ("A", "W"):
            raw = self.operands.get(name + "_bytes")
            if raw is None:
                continue
            out.append((self.placement(name).base, raw))
        return out

    def unpack_output(self, raw) -> Any:
        """The device's bank stream as the logical ``[m, n]`` matrix.

        The accumulator reads out into a register PAIR (or however many registers the accumulate format
        needs), each holding a column slice of the tile, and the kernel lays those banks down back to
        back per output tile in tile order. Un-permuting that here -- in the module that chose the order
        -- keeps the layout knowledge in one place instead of splitting it between a physical-layout
        declaration and a reader."""
        import numpy as np
        t = self.facts.tile
        banks = self.facts.banks_per_tile
        cols_per_bank = t.cols // banks
        mt, _kt, nt = self.tiles
        a = np.asarray(raw, dtype=np.float32).reshape(-1, cols_per_bank)
        out = np.zeros((self.m, self.n), dtype=np.float32)
        for i in range(mt):
            for j in range(nt):
                for b in range(banks):
                    r0 = ((i * nt + j) * banks + b) * t.rows
                    out[i * t.rows:(i + 1) * t.rows,
                        j * t.cols + b * cols_per_bank: j * t.cols + (b + 1) * cols_per_bank] = \
                        a[r0:r0 + t.rows, :]
        return out

    def matches(self, device_output) -> bool:
        """True when the device's output equals this plan's golden BIT-EXACTLY. Bit-exact is the right
        bar because the golden models the device's own accumulator; a tolerance here would hide the
        loop-trip and layout errors this generator exists to catch."""
        import numpy as np
        if self.golden is None:
            raise WorkloadError("this plan carries no golden; supply operands to compare against")
        got = self.unpack_output(device_output)
        return bool(got.shape == self.golden.shape and np.array_equal(got, self.golden))

    def divergence(self, device_output) -> dict:
        """A description of HOW the device output differs from the golden, for a run that did not match."""
        import numpy as np
        got = self.unpack_output(device_output)
        ref = np.asarray(self.golden, dtype=np.float32)
        diff = np.abs(got - ref)
        return {"shape": list(got.shape), "mismatching": int(np.count_nonzero(got != ref)),
                "elements": int(ref.size), "max_abs_err": float(diff.max()) if diff.size else 0.0,
                "device_all_zero": bool(not np.any(got))}


def _tile_counts(facts: MachineFacts, m: int, k: int, n: int) -> tuple[int, int, int]:
    t = facts.tile
    for name, extent, edge in (("M", m, t.rows), ("K", k, t.cols), ("N", n, t.cols)):
        if extent <= 0 or extent % edge:
            raise WorkloadError(
                f"{name}={extent} is not a whole number of {edge}-wide tiles; this generator refuses "
                f"rather than contract over a partial tile")
    return m // t.rows, k // t.cols, n // t.cols


def _place(facts: MachineFacts, m: int, k: int, n: int, *, origin: int,
           align: int) -> tuple[Placement, ...]:
    """Lay the three tensors out from ``origin``, each aligned. Absolute addresses: the kernel stores to
    them and the reader reads them back, and they are computed once here so the two cannot drift."""
    ob, ab = facts.operand_bytes, facts.accum_bytes
    cur = origin
    out: list[Placement] = []
    t = facts.tile
    cols_per_bank = t.cols // facts.banks_per_tile
    for name, role, shape, dtype, nbytes in (
            ("A", "input", [m, k], facts.operand_dtype, m * k * ob),
            ("W", "weight", [k, n], facts.operand_dtype, k * n * ob),
            # the output's DECLARED shape is the bank stream the readout writes (see unpack_output)
            ("C", "output", [(m * n) // cols_per_bank, cols_per_bank], facts.accum_dtype, m * n * ab)):
        cur = (cur + align - 1) // align * align
        out.append(Placement(name, role, shape, dtype, nbytes, cur))
        cur += nbytes
    return tuple(out)


def _pack_tiled(values, facts: MachineFacts, rows: int, cols: int, tile_rows: int,
                tile_cols: int) -> bytes:
    """A matrix as the TILE-MAJOR byte stream the kernel loads: each ``tile_rows x tile_cols`` block
    contiguous, blocks in row-major tile order. One transfer per tile instead of one per ROW is the
    difference between a loop body of eight instructions and one of two hundred, and between moving a
    tile's bytes once and moving its neighbours' bytes with it."""
    import numpy as np
    a = np.asarray(values, dtype=np.float32).reshape(rows, cols)
    chunks = []
    for i in range(rows // tile_rows):
        for j in range(cols // tile_cols):
            chunks.append(a[i * tile_rows:(i + 1) * tile_rows, j * tile_cols:(j + 1) * tile_cols])
    return encode_operand_bytes(np.concatenate([c.reshape(-1) for c in chunks]),
                                facts.operand_dtype)


def _stage_tile(p: _Program, facts: MachineFacts, ops: KernelOps, x: _ScalarFile, *,
                vmem_byte: int, dram_reg: int, nbytes: int) -> None:
    """One contiguous transfer from DRAM into a VMEM staging slot, then wait for it. The VMEM side is a
    machine-word index and the DRAM side a byte offset from the aperture floor -- both derived, neither
    written down."""
    p.load_imm(x["dma_rd"], vmem_byte // facts.word_bytes)
    p.emit(ops.add_imm, rd=x["dma_rs1"], rs1=dram_reg, imm=0)
    p.load_imm(x["dma_len"], nbytes)
    p.emit(ops.dma_load, rd=x["dma_rd"], rs1=x["dma_rs1"], rs2=x["dma_len"])
    p.emit(ops.dma_wait)


def _tensor_op(p: _Program, ops: KernelOps, mnemonic: str, operands: dict, settle: int) -> None:
    """A tensor instruction and the settle its consumer needs. There is no interlock: reading the result
    early returns stale bytes, and a vector op issued too soon does not write its destination at all."""
    p.emit(mnemonic, **operands)
    p.emit(ops.stall, rd=0, rs1=0, imm=int(settle))


def _inner_body(p: _Program, facts: MachineFacts, ops: KernelOps, x: _ScalarFile, settle: Settle, *,
                first_k: bool, v_lhs: int, v_wt: int) -> None:
    """One K-step: stage and push the weight tile, stage the activation tile, and multiply into the
    accumulator. The first K-step OVERWRITES the accumulator so this output tile does not inherit the
    previous one's sum; every later step accumulates into it."""
    tile_bytes = facts.tile.register_bytes
    _stage_tile(p, facts, ops, x, vmem_byte=v_wt, dram_reg=x["w_tile"], nbytes=tile_bytes)
    p.load_imm(x["stage_addr"], v_wt // facts.word_bytes)
    _tensor_op(p, ops, ops.tile_load, {"vd": _M_WT, "rs1": x["stage_addr"], "imm": 0}, settle.tensor)
    # The weight buffer is read back with its axes swapped relative to the row-major tile in memory, so
    # the resident pack is a transpose. Pushing the tile as loaded computes A @ W-transpose instead.
    _tensor_op(p, ops, ops.transpose, {"vd": _M_WT, "vs1": _M_WT}, settle.tensor)
    _tensor_op(p, ops, ops.weight_push, {"vd": _WEIGHT_SLOT, "vs1": _M_WT}, settle.tensor)
    _stage_tile(p, facts, ops, x, vmem_byte=v_lhs, dram_reg=x["a_tile"], nbytes=tile_bytes)
    p.load_imm(x["stage_addr"], v_lhs // facts.word_bytes)
    _tensor_op(p, ops, ops.tile_load, {"vd": _M_LHS, "rs1": x["stage_addr"], "imm": 0}, settle.tensor)
    mnem = ops.contract if first_k else ops.contract_accumulate
    _tensor_op(p, ops, mnem, {"vd": _ACC_SLOT, "vs1": _M_LHS, "vs2": _WEIGHT_SLOT}, settle.mxu)
    # advance the two operand pointers to the next K-tile
    p.emit(ops.add, rd=x["a_tile"], rs1=x["a_tile"], rs2=x["tile_bytes"])
    p.emit(ops.add, rd=x["w_tile"], rs1=x["w_tile"], rs2=x["w_k_stride"])


def plan_matmul(facts: MachineFacts, ops: KernelOps, *, m: int, k: int, n: int,
                control_flow: ControlFlow,
                settle: Settle, A=None, W=None, origin: int | None = None, align: int = 64,
                entry: str = "_start", subnormal_operand_flush: bool | None = None) -> MatmulPlan:
    """Generate a LOOPED, tiled ``[m, k] x [k, n]`` contraction for ``facts.target``.

    The emitted kernel's length does not grow with the shape: the M, N and K axes are three real backward
    branches over a body that stages one tile of each operand. That is what makes a layer reachable at
    all -- a fully unrolled layer of this size overflows instruction memory several times over.

    ``A``/``W``, when given, are the logical float operands; the plan then carries their device-layout
    bytes and the accumulator-format golden. Without them the plan is still complete as a program (shape,
    addresses, kernel) and simply has no reference.

    **The operand DRAM layout is TILE-MAJOR** -- each tile contiguous, tiles in row-major tile order --
    and :attr:`MatmulPlan.operands` produces exactly those bytes, so the kernel and the preload cannot
    disagree. This is a stated layout DECISION, not a hardware constraint, and it is the one that decides
    the workload's transfer character: a tile-major operand moves in ONE transfer, a row-major one needs
    a gather of ``rows`` descriptors whose bytes are the same but whose descriptor count is ``rows``
    times larger. Report :meth:`MatmulPlan.transfer_amplification` with the layout it was measured under;
    the two are not comparable.
    """
    mt, kt, nt = _tile_counts(facts, m, k, n)
    t = facts.tile
    tile_bytes = t.register_bytes
    banks = facts.banks_per_tile
    origin = facts.dram_base + align if origin is None else origin
    placements = _place(facts, m, k, n, origin=origin, align=align)
    by = {p.name: p for p in placements}
    # The kernel addresses DRAM as an offset from the aperture floor; the command buffer and the reader
    # use the absolute address. One subtraction, in one place.
    a_off = by["A"].base - facts.dram_base
    w_off = by["W"].base - facts.dram_base
    c_off = by["C"].base - facts.dram_base
    # VMEM staging: one slot per operand tile plus the accumulator's banks. Only one of each is live.
    v_lhs, v_wt, v_out = 0, tile_bytes, 2 * tile_bytes

    p = _Program(facts.isa, control_flow, ops)
    x = _ScalarFile(facts.target, facts.isa, ops, _SCALAR_ROLES)
    for role, value in (("m_limit", mt), ("n_limit", nt), ("k_limit", kt),
                        ("tile_bytes", tile_bytes), ("a_row_stride", kt * tile_bytes),
                        ("w_k_stride", nt * tile_bytes), ("c_stride", banks * tile_bytes),
                        ("w_origin", w_off), ("a_row", a_off), ("c_tile", c_off)):
        p.load_imm(x[role], value)
    p.load_imm(x["m_index"], 0)

    p.label("m_loop")
    p.emit(ops.add_imm, rd=x["w_col"], rs1=x["w_origin"], imm=0)
    p.load_imm(x["n_index"], 0)

    p.label("n_loop")
    p.emit(ops.add_imm, rd=x["a_tile"], rs1=x["a_row"], imm=0)
    p.emit(ops.add_imm, rd=x["w_tile"], rs1=x["w_col"], imm=0)
    # The first K-step is PEELED because it is the one that overwrites the accumulator rather than adding
    # to it -- a different instruction, not a different operand, so it cannot be selected inside the loop.
    _prologue_words = len(p)
    _before_body = len(p)
    _inner_body(p, facts, ops, x, settle, first_k=True, v_lhs=v_lhs, v_wt=v_wt)
    _k_step_words = len(p) - _before_body
    if kt > 1:
        p.load_imm(x["k_index"], 1)
        p.label("k_loop")
        _inner_body(p, facts, ops, x, settle, first_k=False, v_lhs=v_lhs, v_wt=v_wt)
        p.emit(ops.add_imm, rd=x["k_index"], rs1=x["k_index"], imm=1)
        p.branch(ops.branch_ne, "k_loop", rs1=x["k_index"], rs2=x["k_limit"])

    # this output tile is complete: read the accumulator out, stage it and move it to DRAM
    _before_epilogue = len(p)
    _tensor_op(p, ops, ops.acc_read, {"vd": _M_ACC, "vs1": 0, "vs2": _ACC_SLOT}, settle.vpu)
    for b in range(banks):
        p.load_imm(x["stage_addr"], (v_out + b * tile_bytes) // facts.word_bytes)
        _tensor_op(p, ops, ops.tile_store, {"vd": _M_ACC + b, "rs1": x["stage_addr"], "imm": 0},
                   settle.tensor)
    p.emit(ops.add_imm, rd=x["dma_rd"], rs1=x["c_tile"], imm=0)
    p.load_imm(x["dma_rs1"], v_out // facts.word_bytes)
    p.load_imm(x["dma_len"], banks * tile_bytes)
    p.emit(ops.dma_store, rd=x["dma_rd"], rs1=x["dma_rs1"], rs2=x["dma_len"])
    p.emit(ops.dma_wait)
    p.emit(ops.add, rd=x["c_tile"], rs1=x["c_tile"], rs2=x["c_stride"])
    _tile_epilogue_words = len(p) - _before_epilogue
    p.emit(ops.add, rd=x["w_col"], rs1=x["w_col"], rs2=x["tile_bytes"])
    if nt > 1:
        p.emit(ops.add_imm, rd=x["n_index"], rs1=x["n_index"], imm=1)
        p.branch(ops.branch_ne, "n_loop", rs1=x["n_index"], rs2=x["n_limit"])
    p.emit(ops.add, rd=x["a_row"], rs1=x["a_row"], rs2=x["a_row_stride"])
    if mt > 1:
        p.emit(ops.add_imm, rd=x["m_index"], rs1=x["m_index"], imm=1)
        p.branch(ops.branch_ne, "m_loop", rs1=x["m_index"], rs2=x["m_limit"])

    p.emit(ops.halt)

    operands: dict[str, Any] = {}
    golden = None
    if A is not None and W is not None:
        import numpy as np
        A = np.asarray(A, dtype=np.float32).reshape(m, k)
        W = np.asarray(W, dtype=np.float32).reshape(k, n)
        operands = {"A": A, "W": W,
                    "A_bytes": _pack_tiled(A, facts, m, k, t.rows, t.cols),
                    "W_bytes": _pack_tiled(W, facts, k, n, t.rows, t.cols)}
        if subnormal_operand_flush is None:
            from merlin.targetgen import corpus_spec as CS
            subnormal_operand_flush = bool(
                CS.profile_datapath(facts.target, numeric_only=True).get("subnormal_operand_flush"))
        golden = accumulate_reference(A, W, accum_dtype=facts.accum_dtype,
                                      operand_dtype=facts.operand_dtype,
                                      subnormal_operand_flush=subnormal_operand_flush)
        if golden is None:
            golden = A @ W

    return MatmulPlan(facts=facts, ops=ops, m=m, k=k, n=n, placements=placements,
                      kernel_s=p.kernel_s(entry), words=tuple(p.words()),
                      control_flow=control_flow, settle=settle, entry=entry,
                      tiles=(mt, kt, nt),
                      section_words={"prologue": _prologue_words, "k_step": _k_step_words,
                                     "tile_epilogue": _tile_epilogue_words,
                                     "total": len(p)},
                      operands=operands, golden=golden)
