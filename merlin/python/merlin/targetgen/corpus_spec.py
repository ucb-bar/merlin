"""Target-agnostic capsule-corpus builder.

ONE builder turns an abstract capsule *entry* (op + shapes-in-tiles + epilogue — the target-agnostic test
definition, declared per target in ``capsules/<target>/corpus_profile.yaml``) plus a *binding* DERIVED from
the target's descriptor (dtypes, tile dim, compare policy, instruction classes, oracle tiers) into a
concrete capsule dict + interface MLIR. It replaces the two forked generators (``generate_corpus.py``
gemmini/integer + ``generate_atlas_corpus.py`` atlas/float): the LOGIC here is shared and carries no target
name or dtype literal in its control flow; the per-target DATA (numeric datapath, which capsules, dtypes)
lives in the descriptor + the profile.

Golden VALUES are not computed here (the integer engine lives in :mod:`capsule_golden`; the float engine
needs the external ``specir`` refmodel, available only at generation time) — the driver
``merlin/contract/capsules/generate_corpus.py`` computes them and writes ``golden.yaml``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from merlin.targetgen.capsule_dram import dtype_bits

# canonical dtype token -> (capsule.yaml spelling, MLIR spelling, is_integer). Keyed on the dtype, never
# on a target — a target selects its dtypes via its compute-unit contract.
#
# The WIDTH is deliberately absent. It used to be a fourth literal column, and it was a second copy of a
# fact the format registry already owns: it read 1 byte/element for mxfp6 and mxfp4, which are 6- and
# 4-bit formats, so the two authorities disagreed about how large an MX tensor is. Nothing was
# mis-addressed only because the one consumer read it on the integer path, where the literal happened to
# be right. `dtype_info` derives the width from the registry instead, so there is one source and a new
# format is a registry entry rather than an edit here.
_DTYPE = {
    "int8": ("i8", "i8", True),
    "i8": ("i8", "i8", True),
    "int32": ("i32", "i32", True),
    "i32": ("i32", "i32", True),
    "fp8_e4m3": ("fp8_e4m3", "f8E4M3FN", False),
    "fp8_e5m2": ("fp8_e5m2", "f8E5M2", False),
    # MX microscaling operand widths (block-scaled). ``mxfp*`` are the manifest tokens; ``fp*_e*m*`` the
    # canonical layout spellings — same code<->value map, so both resolve to the OCP MX MLIR type names.
    "mxfp8": ("mxfp8", "f8E4M3FN", False),
    "mxfp6": ("mxfp6", "f6E3M2FN", False),
    "mxfp4": ("mxfp4", "f4E2M1FN", False),
    "fp6_e3m2": ("fp6_e3m2", "f6E3M2FN", False),
    "fp4_e2m1": ("fp4_e2m1", "f4E2M1FN", False),
    "e8m0": ("e8m0", "f8E8M0FNU", False),
    "fp16": ("fp16", "f16", False),
    "f16": ("fp16", "f16", False),
    "bf16": ("bf16", "bf16", False),
    "f32": ("f32", "f32", False),
    "fp32": ("f32", "f32", False),      # alias: a contract may spell single precision "fp32"
}


def dtype_info(token: str) -> tuple[str, str, int | None, bool]:
    """(capsule spelling, MLIR spelling, byte width, is_integer) for a canonical dtype token.

    The width is DERIVED from ``merlin.common.quant_formats`` rather than declared here, so this module
    and the capsule DRAM address map cannot disagree about how large a tensor is. It is ``None`` for a
    sub-byte format (mxfp4, mxfp6), where bytes-per-element has no integer answer and rounding it up
    would over-stride a packed tensor -- size those whole-tensor via ``capsule_dram.tensor_nbytes``.
    """
    if token not in _DTYPE:
        raise KeyError(f"unknown dtype token {token!r} (extend corpus_spec._DTYPE)")
    capsule_spelling, mlir_spelling, integer = _DTYPE[token]
    bits = dtype_bits(capsule_spelling)
    return (capsule_spelling, mlir_spelling, bits // 8 if bits % 8 == 0 else None, integer)


@dataclass(frozen=True)
class CorpusBinding:
    """Per-target axes DERIVED from the descriptor (nothing hand-set per target)."""
    target: str
    tile_dim: int
    operand_dtype: str          # canonical token, e.g. "int8" / "fp8_e4m3"
    accum_dtype: str            # canonical token, e.g. "i32" / "bf16"
    integer: bool               # numeric regime (drives compare policy + golden engine)
    tiers: list[str]            # required_oracle_tiers for this target
    compare: str                # "exact_int" | "tolerance_float"
    atol: float | None = None
    rtol: float | None = None
    scaling: str | None = None  # compute-unit SCALE_KIND (block_e8m0 -> the MX numeric regime) or None
    requant_output_dtype: str | None = None   # narrow output an acc_scale epilogue requants to (e.g. i8)
    # Does the compute unit admit SUBNORMAL operands, or does it see zero where the operand's exponent
    # field is zero? A measured property of the datapath, declared per target in the profile's
    # ``datapath`` block; the float golden engine decodes operands the same way the hardware does.
    subnormal_operand_flush: bool = False
    # Oracle tiers that CANNOT corroborate a result for this target, each with the reason. A tier whose
    # model disagrees with the RTL about the machine itself (not about precision) grades a correct kernel
    # as wrong every time; running it anyway turns the ladder into noise. Declared per target in the
    # profile's ``datapath`` block, carried into every capsule, and honoured by the runner as an honest
    # ``skipped``/not_applicable — never as a pass.
    inapplicable_tiers: dict[str, str] = field(default_factory=dict)
    # Corpus-wide defaults for the per-capsule generalization-intent block (``must_accelerate``,
    # ``generalization_axis``, ``eligible``). Authorial, not derivable — declared once in the profile's
    # ``datapath.semantic_defaults`` so a target states its posture in one place instead of repeating it
    # on every entry; a capsule's own ``semantic:`` overrides it.
    semantic_defaults: dict = field(default_factory=dict)
    # Elements per E8M0 block scale, DERIVED from the manifest's declared scale group; None when the
    # target declares no block scaling, or declares it without a group. Never assumed -- a different
    # microscaling profile uses a different run length, and a guessed one silently mis-shapes the
    # scale operands.
    scale_block: int | None = None
    # instruction-class source: a callable (op, output_dtype, epilogue, movement) -> [class,...]
    classes_for: Callable[..., list[str]] = field(default=lambda **_: [])

    def cap_dtype(self, token: str) -> str:
        return dtype_info(token)[0]

    def mlir_dtype(self, token: str) -> str:
        return dtype_info(token)[1]


# Compiler software-tiling default for a target with NO fixed hardware mesh (SIMT / vector): the matmul
# tile size is a software blocking choice, not a hardware fact. A hardware-mesh target derives its dim
# from facts/manifest and never reaches this. NOT an ISA/hardware constant.
_DEFAULT_SW_TILE = 16


# The OCP microscaling operand tokens: a value in one of these formats is a PAIR -- quantized elements
# plus a shared E8M0 exponent per fixed-length block of them -- so a capsule in one of these dtypes needs
# its scale streams declared as operands. One source of truth: the corpus generator routes an entry to the
# MX golden by the same predicate, so the dtypes that get an MX golden are exactly the ones that get
# scale operands.
BLOCK_SCALED_DTYPES = frozenset({"mxfp4", "mxfp6", "mxfp8"})


def regime_for_dtype(token: str) -> str:
    """The numeric regime a dtype token routes to: ``int`` | ``specir`` | ``mx`` | ``simt``.

    Routed purely by the token -- no target name anywhere in it. Lifted here from the generator so the
    SYNTHESIZER can ask the same question the WRITER will answer with: an op with no direct-MLIR builder
    can only be written by the PyTorch path, and that path is float-only, so a synthesizer that could not
    see the regime emitted entries nothing could write.
    """
    from merlin.runtime.fp8_formats import canonical_float

    if is_block_scaled(token):
        return "mx"
    try:
        canon = canonical_float(token)
    except KeyError:
        canon = None
    if canon in ("fp8_e4m3", "fp8_e5m2"):
        return "specir"
    if canon in ("fp16", "bf16", "f32"):
        return "simt"
    return "int"


def is_block_scaled(token: str | None) -> bool:
    """Whether an operand dtype token is a block-scaled (microscaling) format."""
    return str(token or "") in BLOCK_SCALED_DTYPES


def _scale_block_elems(contract: dict) -> int | None:
    """Elements per E8M0 block scale, DERIVED from the target's manifest, or None.

    A block-scaled compute unit (``scaling: block_e8m0``) carries one shared exponent per fixed-length run
    of K elements. The run length is the target's OWN declared scale ``group`` -- searched for in the
    manifest rather than assumed, because a different microscaling profile uses a different one and a
    guessed length silently mis-shapes every scale operand.

    Returns None when the target declares no block-scaled unit, or declares one without a group. The
    caller then declares NO scale operands and the capsule keeps its present shape: fail closed, never a
    baked constant.
    """
    units = contract.get("compute_units") or []
    if not any(str((u or {}).get("scaling") or "").startswith("block") for u in units):
        return None

    def _group(o) -> int | None:
        """The first declared ``group`` at any depth -- the manifest names its MX interface block
        per target, so the group is found by KEY, not by a path spelled here."""
        if isinstance(o, dict):
            g = o.get("group")
            if isinstance(g, int) and g > 0:
                return g
            for v in o.values():
                found = _group(v)
                if found is not None:
                    return found
        elif isinstance(o, list):
            for v in o:
                found = _group(v)
                if found is not None:
                    return found
        return None

    return _group(contract)


def _tile_dim(target: str, contract: dict) -> int:
    """Tile dim for sizing capsule shapes. When the target has a FIXED HARDWARE mesh, it is DERIVED
    (``capabilities.mesh.rows`` / ``.tile.rows`` from the manifest, else the CIRCT ``arrays[mesh].rows``
    fact) — so gemmini's 16 comes from its RTL facts, never a literal. A target with NO fixed hardware
    mesh (a SIMT / vector target such as radiance, whose matmul tiling is a SOFTWARE choice, not a
    hardware dimension) has nothing to derive; it uses ``_DEFAULT_SW_TILE`` — a compiler software-tiling
    default, NOT a per-target hardware fact. Both derivation sources are keyed on ``target``."""
    # A matrix extension driven as INSTRUCTIONS has two different tile notions at once, and they are
    # different NUMBERS: the spatial array's own edge (the manifest's mesh/tile rows — 16 for the unit
    # this was measured on) and the LOGICAL tile a kernel addresses, which is VLMAX for the element width
    # (32 on the same hardware at VLEN 256). Sizing capsules against the array edge on the instruction
    # surface would bracket the wrong boundary — the tile-edge cases would sit mid-tile and prove nothing.
    # So a unit that declares which hardware CONFIG it is elaborated as gets its edge from that config's
    # own Scala, and a declared-but-underivable geometry raises instead of falling back to a mesh row.
    unit, config = _declared_matrix_unit(contract), _declared_hardware_config(contract)
    if unit and config:
        from merlin.llvmlower import opu_shim
        return int(opu_shim.load_contract(unit).geometry(config)[0])
    caps = (contract.get("capabilities") or {})
    for geom in (caps.get("mesh") or {}, caps.get("tile") or {}):      # systolic mesh OR spatial tile
        if geom.get("rows"):
            return int(geom["rows"])
    try:
        from merlin.targetgen.rtl.facts import load_facts
        arrays = (load_facts(target).get("facts") or {}).get("arrays") or []
        m = next((a for a in arrays if a.get("name") == "mesh"), {})
        if m.get("rows"):
            return int(m["rows"])
    except Exception:  # noqa: BLE001 — no facts for this target -> software-tiling default (no hw mesh)
        pass
    return _DEFAULT_SW_TILE


def _inapplicable_tiers(datapath: dict, required) -> dict[str, str]:
    """``{tier: reason}`` for the oracle tiers this target declares CANNOT corroborate a result.

    Each entry must carry a non-empty reason — a tier switched off without one is indistinguishable from
    a tier switched off because it was failing. A tier that is both REQUIRED and inapplicable is a
    contradiction in the declaration and raises rather than resolving silently either way."""
    raw = datapath.get("inapplicable_oracle_tiers") or {}
    if not isinstance(raw, dict):
        raise ValueError("datapath.inapplicable_oracle_tiers must be a {tier: reason} mapping, "
                         f"got {type(raw).__name__}")
    out: dict[str, str] = {}
    for tier, reason in raw.items():
        if not str(reason or "").strip():
            raise ValueError(f"inapplicable oracle tier {tier!r} declares no reason; a tier is only "
                             f"skippable with the reason it cannot corroborate a result")
        if tier in set(required or ()):
            raise ValueError(f"oracle tier {tier!r} is declared BOTH required and inapplicable")
        out[str(tier)] = str(reason).strip()
    return out


def _accum_dtype(contract: dict, operand: str) -> str:
    """Accumulate dtype from the compute unit's declared ``accumulate`` matrix, else the widening default
    for an integer operand (i32) — a family property (``widening_integer_accumulate``), not a target one."""
    cu = (contract.get("compute_units") or [{}])[0]
    for acc in (cu.get("accumulate") or []):
        if acc.get("acc"):
            return str(acc["acc"])
    return "i32" if dtype_info(operand)[3] else "f32"


def _declared_hardware_config(contract: dict) -> str | None:
    """The elaborated hardware configuration a compute unit declares (a config class in the unit's own
    generator). Which config a target IS is a statement about the hardware being targeted, not a fact to
    derive; what it IMPLIES — tile edge, operand alignment — is derived from that config's declaration."""
    for cu in (contract.get("compute_units") or []):
        cfg = cu.get("hardware_config")
        if cfg:
            return str(cfg)
    return None


def _declared_matrix_unit(contract: dict) -> str | None:
    """The ``matrix_unit`` a compute unit declares, if any — the name of a block in ``matrix_units.yaml``.

    Declared per compute unit rather than per target because a target can carry both a matrix extension
    and something else, and it is the UNIT that has the encodings.
    """
    for cu in (contract.get("compute_units") or []):
        name = cu.get("matrix_unit")
        if name:
            return str(name)
    return None


def _classes_source(te, contract: dict) -> Callable[..., list[str]]:
    """The instruction-class deriver for a matmul-family op. Two regimes, chosen by what the target ships:
    a self-hosted-ISA target (an ``isa_definition.py`` is present) derives its classes from the taxonomy;
    a RoCC/command target derives the RoCC semantic classes from its ``encoding`` map. Never hardcoded."""
    from merlin.targetgen import isa_taxonomy as IT
    # A compute unit may declare that it IS a matrix extension whose encodings are derived by the
    # matrix-unit reader (``matrix_units.yaml``). That surface ships no isa_definition.py and no RoCC
    # ``encoding`` map, so both regimes below saw nothing and the class list came out empty — the state in
    # which a coverage expectation is satisfied by emitting no instructions at all. Checked FIRST and
    # allowed to raise: a declared unit whose classes cannot be derived must stop corpus generation, not
    # quietly produce an unfalsifiable corpus.
    unit = _declared_matrix_unit(contract)
    if unit:
        return IT.matrix_unit_classes_for(unit)
    try:
        tax = IT.derive_isa_taxonomy(te)
    except Exception:  # noqa: BLE001
        tax = {}
    if tax.get("by_class"):
        def _from_taxonomy(*, op="matmul", output_dtype=None, epilogue=(), movement=False):
            return IT.required_classes_for_op(tax, op=op, output_dtype=output_dtype,
                                              epilogue=tuple(epilogue), movement=movement)
        return _from_taxonomy
    # RoCC command target: the matmul-relevant semantic classes for a weight-stationary tile, with the
    # single CONFIG class expanded to its declared config subtypes, in RoCC issue order. `pool` = what
    # this target's encoding actually defines (so a target missing a class simply drops it); the order is
    # the RoCC weight-stationary matmul sequence, filtered by `pool`.
    enc = contract.get("encoding") or {}
    sem_vals = set((enc.get("semantic_class") or {}).values())
    sub_vals = set((enc.get("config_subtype") or {}).values())
    pool = (sem_vals - {"CONFIG"}) | sub_vals
    order = ["FLUSH", "CONFIG_EX", "CONFIG_LD", "MVIN", "CONFIG_ST", "PRELOAD", "COMPUTE_PRELOADED", "MVOUT"]
    classes = [c for c in order if c in pool]

    def _from_encoding(*, op="matmul", output_dtype=None, epilogue=(), movement=False):
        if movement:
            # A movement capsule is a DMA load-to-store round trip, not a contraction.  The RoCC
            # fallback previously ignored the operation and returned the full matrix sequence for every
            # capsule, unlike both of the role-aware regimes above.  Project the target's OWN declared
            # class labels through the same coarse-role function used to cross-check them against RTL,
            # then omit matrix-compute commands.  Config/load/store/barrier classes remain derived from
            # the encoding map; no target name, opcode, or per-target class list is introduced here.
            from merlin.targetgen.rtl.mlc_bridge import _coarse_of_hand_class
            return [c for c in classes if _coarse_of_hand_class(c) != "compute"]
        return list(classes)
    return _from_encoding


#: Keys of the profile's ``datapath`` block that are CORPUS-AUTHORING choices, not facts about how the
#: hardware computes. The block mixes the two, and only the second kind may be handed to a caller that did
#: not ask for it: ``required_oracle_tiers`` is which tiers the graded corpus demands (a tile certification
#: wants the tiers that actually resolve, which is the fallback), ``semantic_defaults`` is the corpus's
#: ``must_accelerate`` posture, and ``requant_output_dtype`` is a per-op epilogue handoff the caller passes
#: when its own chain needs one. ``inapplicable_tiers`` goes with ``required_oracle_tiers`` because the two
#: are a MATCHED PAIR -- :func:`_inapplicable_tiers` fails closed when a tier is both required and
#: inapplicable, so taking one without the other is incoherent by construction (atlas declares L2
#: inapplicable, and L2 is in the tiers that actually resolve). Held back as a DENYLIST, so a numeric
#: datapath fact added to the profile later reaches every consumer by default -- which is the failure this
#: whole change is about.
_DATAPATH_AUTHORING_KEYS = frozenset({"required_oracle_tiers", "inapplicable_oracle_tiers",
                                      "semantic_defaults", "requant_output_dtype"})


def profile_datapath(target: str, *, numeric_only: bool = False) -> dict:
    """The ``datapath`` block of ``target``'s capsule profile, or ``{}`` when it has no profile.

    ``numeric_only=True`` drops :data:`_DATAPATH_AUTHORING_KEYS`, leaving just the facts about how the
    device computes -- operand/accumulate formats, tolerances, block scaling, subnormal handling. That is
    what a caller wants when it is describing HARDWARE rather than generating a corpus. Measured on the
    three targets: gemmini's binding is bit-identical either way, and atlas/radiance gain only numeric
    fields, so nothing that already worked changes behaviour.

    This is where a target writes down the numeric facts its manifest does not carry -- tolerances, the
    requant handoff dtype, inapplicable oracle tiers, and ``subnormal_operand_flush``. The corpus
    generator has always passed the block into :func:`derive_binding`; every OTHER caller hand-built a
    dict holding just the two dtypes it happened to care about, so for those callers each of those facts
    silently took its dataclass default. Atlas declares ``subnormal_operand_flush: true`` and the mesh
    boundary read ``False`` for exactly that reason. One loader, so a fact declared once arrives
    everywhere.

    Reads only the TRACKED public profile, never the ``<target>.hidden.yaml`` holdout sidecar: callers
    here emit run reports, and a report that enumerated the holdouts would leak them.
    """
    from merlin.common.paths import merlin_dir
    from merlin.common.yaml import load_yaml
    prof = merlin_dir() / "contract" / "capsules" / "profiles" / f"{target}.yaml"
    if not prof.is_file():
        return {}
    block = dict((load_yaml(prof) or {}).get("datapath") or {})
    if numeric_only:
        for key in _DATAPATH_AUTHORING_KEYS:
            block.pop(key, None)
    return block


def derive_binding(te, datapath: dict) -> CorpusBinding:
    """Derive the per-target binding from the descriptor + the profile's ``datapath`` block (compare +
    tolerances + optional requant-output dtype — the numeric contract the manifest does not yet carry)."""
    from merlin.targetgen.target_experiment import load_capability_manifest
    from merlin.targetgen import capsule_runner as CR
    m = load_capability_manifest(te.target)
    c = m.contract
    cu = (c.get("compute_units") or [{}])[0]
    # The profile may pin the DEFAULT operand/accumulate dtypes (a target with several compute units — e.g.
    # radiance's simt_cluster + contained mx_pe — needs the profile to say which regime a capsule set drives);
    # both fall back to the primary compute unit's declared datapath, never a target literal.
    operand = datapath.get("operand_dtype") or (cu.get("dtypes") or ["int8"])[0]
    accum = datapath.get("accum_dtype") or _accum_dtype(c, operand)
    integer = dtype_info(accum)[3]
    scaling = datapath.get("scaling") or cu.get("scaling")
    tiers = datapath.get("required_oracle_tiers") \
        or sorted((CR.oracle_adapters(te.target, te.sim_via) or {}).keys())
    return CorpusBinding(
        target=te.target,
        tile_dim=_tile_dim(te.target, c),
        operand_dtype=operand,
        accum_dtype=accum,
        integer=integer,
        tiers=list(tiers),
        compare=datapath.get("compare", "exact_int" if integer else "tolerance_float"),
        atol=datapath.get("atol"),
        rtol=datapath.get("rtol"),
        scaling=(scaling if scaling and scaling != "none" else None),
        scale_block=_scale_block_elems(c),
        requant_output_dtype=datapath.get("requant_output_dtype"),
        subnormal_operand_flush=bool(datapath.get("subnormal_operand_flush", False)),
        inapplicable_tiers=_inapplicable_tiers(datapath, tiers),
        semantic_defaults=dict(datapath.get("semantic_defaults") or {}),
        classes_for=_classes_source(te, c),
    )


# ---------------------------------------------------------------------------------------------------
# capsule builders — one per op shape; each takes the abstract entry + binding and returns the capsule
# dict + interface MLIR. Shapes are given in TILE units in the profile and scaled by binding.tile_dim.
# ---------------------------------------------------------------------------------------------------

def _numeric_policy(binding: CorpusBinding, output_dtype: str, acc_scale: float | None) -> dict:
    np_: dict[str, Any] = {"compare": binding.compare, "dtype": binding.cap_dtype(output_dtype)}
    if not binding.integer:
        if binding.atol is not None:
            np_["atol"] = binding.atol
        if binding.rtol is not None:
            np_["rtol"] = binding.rtol
    if acc_scale is not None:
        np_["acc_scale"] = acc_scale
    return np_


def _resolve_output_dtype(binding: CorpusBinding, epilogue: list[str], requested: str | None = None) -> str:
    """Resolve the operation's authored output dtype.

    An explicit profile value wins: narrowing can be a property of a fused hardware readout, not only
    of an ``acc_scale`` stage. Otherwise the historical acc-scale/default-accumulator rule remains.
    """
    if requested is not None:
        binding.cap_dtype(str(requested))  # validate through the canonical dtype registry
        return str(requested)
    if "acc_scale" in epilogue and binding.requant_output_dtype:
        return binding.requant_output_dtype
    return binding.accum_dtype


def _default_modes(binding: CorpusBinding, output_dtype: str, epilogue: list[str]) -> dict:
    """The mode flags a capsule gets when its profile entry declares none: relu/acc_scale from the
    epilogue (+ the integer-narrow-output flag for an integer target). An entry MAY instead declare
    ``modes`` verbatim (e.g. ``{k_accumulate: true}``), which is used as-is."""
    modes = {"relu": "relu" in epilogue, "acc_scale": "acc_scale" in epilogue}
    if binding.integer:
        modes["i8"] = binding.cap_dtype(output_dtype) == "i8"
    return modes


def _pool_epilogue(entry: dict, *, rows: int, in_dims: tuple[int, int] | None,
                   op: str) -> tuple[dict[str, Any], int]:
    """``(pool-geometry attributes, the row count the op commits)`` for a ``maxpool`` epilogue.

    Pooling on this class of target is FUSED onto the store path -- the store config carries
    ``pool_size``/``pool_stride``/``pool_out_dim``/``porows``/``pocols``/``orows``/``ocols``/``upad``/
    ``lpad`` and the mvout writes the pooled window -- so a standalone pooling capsule is not the
    honest capsule for it (the eligibility oracle refuses one as a false fallback). The geometry
    therefore rides on the fused op's own attributes, in the ABI's own vocabulary.

    ``in_dims`` is the spatial extent the committed ROWS unflatten to. A conv2d DERIVES it (it is the
    conv's own ``Ho x Wo``); a contraction has no spatial extent of its own, so its entry must DECLARE
    ``pool_in_dims`` -- 25 rows is 5x5 or 25x1 and nothing else in the capsule says which. There is no
    default for the window either: a pool size or stride this generator picked would be a target fact
    invented in library code, and every engine downstream would then be grading a window nobody chose.

    Declaring a ``pool_*`` knob WITHOUT the stage also raises. A parameter that is read by nothing is
    the same silent-wrong-answer shape as a stage applied by nobody, just pointing the other way.
    """
    from merlin.runtime.tensor import pool_out_dims

    stages = [str(x) for x in (entry.get("epilogue") or [])]
    declared = sorted(k for k in entry if k.startswith("pool_"))
    if "maxpool" not in stages:
        if declared:
            raise ValueError(
                f"{entry.get('name', op)}: declares {declared} but no 'maxpool' epilogue stage; the "
                f"pool geometry would be carried into the capsule and read by nothing")
        return {}, rows
    if in_dims is None:
        want = entry.get("pool_in_dims")
        if not isinstance(want, (list, tuple)) or len(want) != 2:
            raise ValueError(
                f"{entry.get('name', op)}: a 'maxpool' epilogue on {op} needs pool_in_dims = [H, W] "
                f"(the extent the {rows} committed rows unflatten to); got {want!r}")
        in_dims = (int(want[0]), int(want[1]))
    elif entry.get("pool_in_dims") is not None:
        raise ValueError(
            f"{entry.get('name', op)}: pool_in_dims is DERIVED for {op} (it is the op's own output "
            f"extent {list(in_dims)}); declaring it invites the two to disagree")
    size, stride = entry.get("pool_size"), entry.get("pool_stride")
    for key, val in (("pool_size", size), ("pool_stride", stride)):
        if not isinstance(val, (list, tuple)) or len(val) != 2:
            raise ValueError(
                f"{entry.get('name', op)}: a 'maxpool' epilogue needs {key} = [h, w]; got {val!r}. "
                f"There is no default -- the window is a property of the workload, not of this builder")
    padding = list(entry.get("pool_padding", [0, 0, 0, 0]))
    if len(padding) != 4:
        raise ValueError(
            f"{entry.get('name', op)}: pool_padding must be [top, left, bottom, right]; got {padding!r}")
    attrs: dict[str, Any] = {
        "pool_in_dims": [int(in_dims[0]), int(in_dims[1])],
        "pool_size": [int(x) for x in size],
        "pool_stride": [int(x) for x in stride],
        "pool_padding": [int(x) for x in padding],
    }
    if entry.get("pool_pad_value") is not None:
        attrs["pool_pad_value"] = int(entry["pool_pad_value"])
    plane = in_dims[0] * in_dims[1]
    if plane <= 0 or rows % plane:
        raise ValueError(
            f"{entry.get('name', op)}: {rows} committed rows are not a whole number of "
            f"{in_dims[0]}x{in_dims[1]} planes, so pool_in_dims does not describe this operation")
    Ho, Wo = pool_out_dims(in_dims[0], in_dims[1], attrs["pool_size"],
                           attrs["pool_stride"], attrs["pool_padding"])
    if Ho < 1 or Wo < 1:
        raise ValueError(
            f"{entry.get('name', op)}: window {attrs['pool_size']} stride {attrs['pool_stride']} "
            f"padding {attrs['pool_padding']} leaves no output position over "
            f"{in_dims[0]}x{in_dims[1]}")
    return attrs, (rows // plane) * Ho * Wo


def _pool_mlir_attrs(pool_attrs: dict[str, Any]) -> str:
    """Render pool geometry as interface-grammar attribute text (``, pool_size = [2, 2]`` ...).

    Integer LISTS, unquoted: the grammar's attribute parser reads ``[2, 2]`` back as ints and ``["2",
    "2"]`` back as strings, and a geometry that arrived as strings would fail arity/typing checks far
    from here with a message about the window rather than about the spelling.
    """
    out = ""
    for k, v in pool_attrs.items():
        if isinstance(v, list):
            out += f", {k} = [{', '.join(str(int(x)) for x in v)}]"
        else:
            out += f", {k} = {int(v)} : i64"
    return out


def _matmul_inputs(lhs: str, weight: str, M: int, K: int, N: int, idt: str,
                   binding: CorpusBinding) -> list[dict]:
    """The declared leaf operands of a matmul capsule.

    On a BLOCK-SCALED datapath the operand format is (elements, per-block scales); declaring only the
    elements hands a backend half a number and asks it for the product. The scale streams are therefore
    declared operands in their own right, shaped by the target's own derived scale group -- one shared
    exponent per run of ``scale_block`` K elements, per lhs row and per weight column.

    Declared only when the target's manifest yields BOTH a block-scaled compute unit and a scale group,
    and the contraction depth divides into whole blocks. Otherwise the operand list is exactly what it
    has always been, so an unscaled target's capsules are unchanged."""
    inputs = [{"name": weight, "role": "weight", "shape": [K, N], "dtype": idt},
              {"name": lhs, "role": "input", "shape": [M, K], "dtype": idt}]
    blk = binding.scale_block if is_block_scaled(binding.operand_dtype) else None
    if blk and K % blk == 0:
        g = K // blk
        inputs += [
            {"name": f"{lhs}_scale", "role": "scale", "scale_of": lhs, "block": blk,
             "shape": [g, M], "dtype": "e8m0"},
            {"name": f"{weight}_scale", "role": "scale", "scale_of": weight, "block": blk,
             "shape": [g, N], "dtype": "e8m0"},
        ]
    return inputs


def build_matmul(entry: dict, binding: CorpusBinding) -> tuple[dict, str]:
    """A single weight-stationary matmul/linear capsule (op ∈ {matmul, linear})."""
    D = binding.tile_dim
    M = entry.get("M_tiles", 1) * D if "M" not in entry else entry["M"]
    K = entry.get("K_tiles", 1) * D if "K" not in entry else entry["K"]
    N = entry.get("N_tiles", 1) * D if "N" not in entry else entry["N"]
    lhs, weight, out = entry.get("lhs", "A0"), entry.get("weight", "W"), entry.get("out", "Y0")
    epilogue = list(entry.get("epilogue", []))
    acc_scale = entry.get("acc_scale")
    output_dtype = _resolve_output_dtype(binding, epilogue, entry.get("output_dtype"))
    op = entry.get("op", "matmul")
    odt = binding.cap_dtype(output_dtype)
    idt = binding.cap_dtype(binding.operand_dtype)
    attrs: dict[str, Any] = {"lhs": lhs, "weight": weight, "out": out,
                             "epilogue": epilogue, "output_dtype": odt}
    # A pooling epilogue rides on the SAME attributes the store path configures, and it changes the
    # committed extent -- the commit result type below is the POOLED row count, not M. A contraction
    # carries no spatial extent of its own, so the entry declares pool_in_dims (see _pool_epilogue).
    pool_attrs, commit_rows = _pool_epilogue(entry, rows=M, in_dims=None, op=op)
    attrs.update(pool_attrs)
    if acc_scale is not None:
        attrs["acc_scale"] = acc_scale
    if entry.get("semantic"):
        attrs["semantic"] = entry["semantic"]
    expected = {"instruction_classes": binding.classes_for(op=op, output_dtype=odt,
                                                           epilogue=epilogue, movement=False),
                "modes": dict(entry["modes"]) if "modes" in entry
                else _default_modes(binding, output_dtype, epilogue)}
    if entry.get("forbidden"):
        expected["forbidden_classes"] = entry["forbidden"]
    cap = {
        "name": entry["name"], "kind": entry["kind"],
        "source_role": entry["source_role"], "source_reference": entry["source_reference"],
        "label": entry.get("label", "public"), "interface_mlir": "capsule.interface.mlir",
        "inputs": _matmul_inputs(lhs, weight, M, K, N, idt, binding),
        "operation": {"op": op, "attributes": attrs},
        "numeric_policy": _numeric_policy(binding, output_dtype, acc_scale),
        "expected": expected, "required_oracle_tiers": list(binding.tiers),
        "vcs": "optional", "firesim": "optional",
    }
    from merlin.targetgen import model_slice_export as MSE
    mlir = MSE.emit_interface_mlir(
        lhs=lhs, weight=weight, out=out, M=M, K=K, N=N, epilogue=epilogue, output_dtype=odt,
        acc_scale=acc_scale, comment=entry.get("comment", ""),
        pool_attrs=pool_attrs or None, commit_rows=commit_rows,
        target=binding.target, operand_dtype=binding.mlir_dtype(binding.operand_dtype),
        acc_dtype=binding.mlir_dtype(binding.accum_dtype),
        scale_block=(binding.scale_block if is_block_scaled(binding.operand_dtype) else None))
    return cap, mlir


def _iface_prelude(target: str, comment: str) -> list[str]:
    head = ('module attributes {merlin_iface.version = "0.1", '
            f'merlin_iface.target = "{target}", merlin_iface.abi_version = "0.1"}} {{')
    return ([f"// {comment}", head] if comment else [head])


def build_resident_reuse(entry: dict, binding: CorpusBinding) -> tuple[dict, str]:
    """One resident weight reused across two matmuls (op == resident_reuse).

    AN EXTENT MAY BE DECLARED EITHER WAY, and reading only the ``_tiles`` spelling silently drops the
    other. ``expand_sweeps`` resolves an ``axes:`` entry to the BARE name (``K``, ``N``), so a sweep
    declaring ``K: ["4*tile", "8*tile"]`` lands in ``entry["K"]`` while this builder read
    ``entry["K_tiles"]``, defaulted to 1, and emitted the tile edge for BOTH points. Reproduced
    2026-09-04: PC00_k64 and PC01_k128 both came out ``W[16,16]`` -- byte-identical interfaces
    differing only in a name and a prose string. The family's own gate demands two separation
    regimes; it would have had one regime with two labels, and a paired differential over them would
    have measured the same program twice. Same failure shape as holdout sets that turned out to be
    renames -- see the memory `holdout-sets-were-renames`.
    """
    D = binding.tile_dim
    K = entry.get("K", entry.get("K_tiles", 1) * D)
    N = entry.get("N", entry.get("N_tiles", 1) * D)
    weight = entry.get("weight", "W")
    idt, adt = binding.cap_dtype(binding.operand_dtype), binding.cap_dtype(binding.accum_dtype)
    midt, madt = binding.mlir_dtype(binding.operand_dtype), binding.mlir_dtype(binding.accum_dtype)
    inputs = [{"name": weight, "role": "weight", "shape": [K, N], "dtype": idt}]
    matmuls, mlir_mm = [], []
    L = _iface_prelude(binding.target, entry.get("comment", ""))
    L.append(f'  %{weight} = merlin_iface.tensor {{name = "{weight}", role = "weight"}} : tensor<{K}x{N}x{midt}>')
    for idx, mm in enumerate(entry["matmuls"]):
        an, oname = mm["lhs"], mm["out"]
        M = mm.get("M_tiles", 1) * D
        epi = list(mm.get("epilogue", []))
        inputs.append({"name": an, "role": "input", "shape": [M, K], "dtype": idt})
        matmuls.append({"lhs": an, "out": oname, "epilogue": epi, "output_dtype": adt})
        L.append(f'  %{an} = merlin_iface.tensor {{name = "{an}", role = "input"}} : tensor<{M}x{K}x{midt}>')
    L.append(f'  %{weight}_res = merlin_iface.resident_pack %{weight} {{layout = "packed_rhs"}} '
             f': (tensor<{K}x{N}x{midt}>) -> !merlin_iface.resident')
    for idx, mm in enumerate(entry["matmuls"]):
        an, oname = mm["lhs"], mm["out"]
        M = mm.get("M_tiles", 1) * D
        epi = ", ".join(f'"{e}"' for e in mm.get("epilogue", []))
        L.append(f'  %acc{idx} = merlin_iface.matmul %{an}, %{weight}_res '
                 f': (tensor<{M}x{K}x{midt}>, !merlin_iface.resident) -> !merlin_iface.acc<{madt}>')
        L.append(f'  %{oname} = merlin_iface.commit %acc{idx} {{name = "{oname}", epilogue = [{epi}], '
                 f'output_dtype = "{adt}"}} : (!merlin_iface.acc<{madt}>) -> tensor<{M}x{N}x{adt}>')
    L.append(f'  merlin_iface.evict %{weight}_res : (!merlin_iface.resident) -> ()')
    L.append("}")
    attrs = {"weight": weight, "matmuls": matmuls, "semantic": entry.get("semantic", "resident_reuse")}
    cap = {
        "name": entry["name"], "kind": entry["kind"], "source_role": entry["source_role"],
        "source_reference": entry["source_reference"], "label": entry.get("label", "public"),
        "interface_mlir": "capsule.interface.mlir", "inputs": inputs,
        "operation": {"op": "resident_reuse", "attributes": attrs},
        "numeric_policy": _numeric_policy(binding, binding.accum_dtype, None),
        "expected": {"instruction_classes": binding.classes_for(op="matmul", output_dtype=adt),
                     "modes": {"resident_reuse": True}},
        "required_oracle_tiers": list(binding.tiers), "vcs": "optional", "firesim": "optional",
    }
    return cap, "\n".join(L) + "\n"


def build_movement(entry: dict, binding: CorpusBinding) -> tuple[dict, str]:
    """A load->store dequant movement capsule (op == movement): operand dtype in, accumulate dtype out."""
    D = binding.tile_dim
    M = entry.get("M", entry.get("M_tiles", 1) * D)
    N = entry.get("N", entry.get("N_tiles", 1) * D)
    src, out = entry.get("src", "X"), entry.get("out", "Y0")
    idt, odt = binding.cap_dtype(binding.operand_dtype), binding.cap_dtype(binding.accum_dtype)
    midt, modt = binding.mlir_dtype(binding.operand_dtype), binding.mlir_dtype(binding.accum_dtype)
    attrs = {"out": out, "src": src, "semantic": entry.get("semantic", "mvin_mvout"), "output_dtype": odt}
    expected = {"instruction_classes": binding.classes_for(op="movement", movement=True),
                "modes": {"movement": True}}
    if entry.get("forbidden"):
        expected["forbidden_classes"] = entry["forbidden"]
    cap = {
        "name": entry["name"], "kind": entry["kind"], "source_role": entry["source_role"],
        "source_reference": entry["source_reference"], "label": entry.get("label", "public"),
        "interface_mlir": "capsule.interface.mlir",
        "inputs": [{"name": src, "role": "input", "shape": [M, N], "dtype": idt}],
        "operation": {"op": "movement", "attributes": attrs},
        "numeric_policy": _numeric_policy(binding, binding.accum_dtype, None),
        "expected": expected, "required_oracle_tiers": list(binding.tiers),
        "vcs": "optional", "firesim": "optional",
    }
    L = _iface_prelude(binding.target, entry.get("comment", ""))
    L += [
         f'  %{src} = merlin_iface.tensor {{name = "{src}", role = "input"}} : tensor<{M}x{N}x{midt}>',
         f'  %{out} = merlin_iface.movement %{src} {{name = "{out}", semantic = "mvin_mvout", '
         f'output_dtype = "{odt}"}} : (tensor<{M}x{N}x{midt}>) -> tensor<{M}x{N}x{modt}>', "}"]
    return cap, "\n".join(L) + "\n"


def build_attention_qk(entry: dict, binding: CorpusBinding) -> tuple[dict, str]:
    """Q @ K^T attention scores (op == attention_qk): the device does the transpose internally.

    Reads both extent spellings, for the reason given on :func:`build_resident_reuse`: a sweep axis
    resolves to the bare name, and taking only the ``_tiles`` form collapses every point onto the
    tile edge without failing.
    """
    D = binding.tile_dim
    M = entry.get("M", entry.get("M_tiles", 1) * D)
    Kd = entry.get("K", entry.get("K_tiles", 1) * D)
    q, k, out = entry.get("q", "Q"), entry.get("k", "K"), entry.get("out", "Y0")
    idt, odt = binding.cap_dtype(binding.operand_dtype), binding.cap_dtype(binding.accum_dtype)
    midt, modt = binding.mlir_dtype(binding.operand_dtype), binding.mlir_dtype(binding.accum_dtype)
    attrs = {"q": q, "k": k, "out": out, "epilogue": [], "output_dtype": odt}
    cap = {
        "name": entry["name"], "kind": entry["kind"], "source_role": entry["source_role"],
        "source_reference": entry["source_reference"], "label": entry.get("label", "public"),
        "interface_mlir": "capsule.interface.mlir",
        "inputs": [{"name": q, "role": "input", "shape": [M, Kd], "dtype": idt},
                   {"name": k, "role": "input", "shape": [M, Kd], "dtype": idt}],
        "operation": {"op": "attention_qk", "attributes": attrs},
        "numeric_policy": _numeric_policy(binding, binding.accum_dtype, None),
        "expected": {"instruction_classes": binding.classes_for(op="matmul", output_dtype=odt),
                     "modes": entry.get("modes", {})},
        "required_oracle_tiers": list(binding.tiers), "vcs": "optional", "firesim": "optional",
    }
    L = _iface_prelude(binding.target, entry.get("comment", ""))
    L += [
         f'  %{q} = merlin_iface.tensor {{name = "{q}", role = "input"}} : tensor<{M}x{Kd}x{midt}>',
         f'  %{k} = merlin_iface.tensor {{name = "{k}", role = "input"}} : tensor<{M}x{Kd}x{midt}>',
         f'  %{out} = merlin_iface.attention_qk %{q}, %{k} {{name = "{out}", output_dtype = "{odt}"}} '
         f': (tensor<{M}x{Kd}x{midt}>, tensor<{M}x{Kd}x{midt}>) -> tensor<{M}x{M}x{modt}>', "}"]
    return cap, "\n".join(L) + "\n"


def build_rmsnorm(entry: dict, binding: CorpusBinding) -> tuple[dict, str]:
    """A row RMSNorm capsule (op == rmsnorm): X[M,K] * rsqrt(mean(X^2)+eps) * gamma[1,K] -> [M,K]. A SIMT
    (elementwise/reduction) op — its golden is ordinary IEEE float, computed by the driver."""
    D = binding.tile_dim
    M = entry.get("M", entry.get("M_tiles", 1) * D)
    K = entry.get("K", entry.get("K_tiles", 1) * D)
    x, gamma, out = entry.get("src", "X"), entry.get("gamma", "G"), entry.get("out", "Y0")
    eps = entry.get("eps", 1.0 / 65536.0)
    idt = binding.cap_dtype(binding.operand_dtype)
    odt = binding.cap_dtype(binding.accum_dtype)
    midt, modt = binding.mlir_dtype(binding.operand_dtype), binding.mlir_dtype(binding.accum_dtype)
    attrs = {"src": x, "gamma": gamma, "out": out, "eps": eps, "semantic": "rmsnorm", "output_dtype": odt}
    cap = {
        "name": entry["name"], "kind": entry["kind"], "source_role": entry["source_role"],
        "source_reference": entry["source_reference"], "label": entry.get("label", "public"),
        "interface_mlir": "capsule.interface.mlir",
        "inputs": [{"name": x, "role": "input", "shape": [M, K], "dtype": idt},
                   {"name": gamma, "role": "weight", "shape": [1, K], "dtype": idt}],
        "operation": {"op": "rmsnorm", "attributes": attrs},
        "numeric_policy": _numeric_policy(binding, binding.accum_dtype, None),
        "expected": {"instruction_classes": [], "modes": {"rmsnorm": True}},
        "required_oracle_tiers": list(binding.tiers), "vcs": "optional", "firesim": "optional",
    }
    L = _iface_prelude(binding.target, entry.get("comment", ""))
    L += [
         f'  %{x} = merlin_iface.tensor {{name = "{x}", role = "input"}} : tensor<{M}x{K}x{midt}>',
         f'  %{gamma} = merlin_iface.tensor {{name = "{gamma}", role = "weight"}} : tensor<1x{K}x{midt}>',
         f'  %{out} = merlin_iface.rmsnorm %{x}, %{gamma} {{name = "{out}", eps = {eps:.9e} : f64, '
         f'output_dtype = "{odt}"}} : (tensor<{M}x{K}x{midt}>, tensor<1x{K}x{midt}>) -> tensor<{M}x{K}x{modt}>',
         "}"]
    return cap, "\n".join(L) + "\n"


def _attention_mx_inputs(q: str, k: str, v: str, M: int, H: int, Skv: int, Dv: int, idt: str,
                         binding: CorpusBinding) -> list[dict]:
    """Declared leaf operands of a fused block-scaled attention capsule.

    Two MX stages contract over different axes, so they carry different scale streams: the QK stage over
    the head dim ``H`` (scaling q's rows and k's rows), the PV stage over the key count ``Skv`` (scaling
    v's columns). The PV stage's LHS is ``P`` -- the softmax output, an INTERMEDIATE the kernel produces
    and requantizes -- and the exponent it is requantized against was chosen when the golden was built.
    It is therefore declared too: a value the kernel cannot derive is an input, whatever produced it."""
    inputs = [{"name": q, "role": "input", "shape": [M, H], "dtype": idt},
              {"name": k, "role": "input", "shape": [Skv, H], "dtype": idt},
              {"name": v, "role": "input", "shape": [Skv, Dv], "dtype": idt}]
    blk = binding.scale_block if is_block_scaled(binding.operand_dtype) else None
    if blk and H % blk == 0 and Skv % blk == 0:
        hg, sg = H // blk, Skv // blk
        inputs += [
            {"name": f"{q}_scale", "role": "scale", "scale_of": q, "block": blk,
             "shape": [hg, M], "dtype": "e8m0"},
            {"name": f"{k}_scale", "role": "scale", "scale_of": k, "block": blk,
             "shape": [hg, Skv], "dtype": "e8m0"},
            {"name": f"{v}_scale", "role": "scale", "scale_of": v, "block": blk,
             "shape": [sg, Dv], "dtype": "e8m0"},
            {"name": "P_scale", "role": "scale", "scale_of": "P", "block": blk,
             "shape": [sg, M], "dtype": "e8m0"},
        ]
    return inputs


def build_attention_mx(entry: dict, binding: CorpusBinding) -> tuple[dict, str]:
    """A fused MX flash-attention capsule (op == attention_mx): O = softmax(Q @ K^T / sqrt(H) [+ soft-cap])
    @ V, with the two matmuls on the block-scaled MX PE (E8M0 per 32-element K group, bf16 accumulate) and
    a bf16 row-softmax between them. The interface is a fused merlin_iface program (attention_qk -> softmax
    -> attention_pv) in mxfp8; the golden is composed by the driver from mx_ref (QK & PV) + a numpy bf16
    softmax. Dims: M queries, H head dim (%32), Skv keys (%32), Dv value dim (%16)."""
    D = binding.tile_dim
    M = entry.get("M", entry.get("M_tiles", 1) * D)
    H = entry.get("H", entry.get("head_dim", 2 * D))
    Skv = entry.get("Skv", entry.get("keys", 2 * D))
    Dv = entry.get("Dv", D)
    q, k, v, out = entry.get("q", "Q"), entry.get("k", "K"), entry.get("v", "V"), entry.get("out", "Y0")
    idt = binding.cap_dtype(binding.operand_dtype)               # mxfp8
    odt = binding.cap_dtype(binding.accum_dtype)                 # bf16
    midt, modt = binding.mlir_dtype(binding.operand_dtype), binding.mlir_dtype(binding.accum_dtype)
    import math
    scale = float(entry.get("scale", 1.0 / math.sqrt(H)))
    softcap = entry.get("softcap")
    attrs: dict[str, Any] = {"q": q, "k": k, "v": v, "out": out, "scale": scale,
                             "block_scale": "e8m0", "output_dtype": odt}
    if softcap is not None:
        attrs["softcap"] = float(softcap)
    cap = {
        "name": entry["name"], "kind": entry["kind"], "source_role": entry["source_role"],
        "source_reference": entry["source_reference"], "label": entry.get("label", "public"),
        "interface_mlir": "capsule.interface.mlir",
        "inputs": _attention_mx_inputs(q, k, v, M, H, Skv, Dv, idt, binding),
        "operation": {"op": "attention_mx", "attributes": attrs},
        "numeric_policy": _numeric_policy(binding, binding.accum_dtype, None),
        # two matmuls (QK, PV) drive the MX PE; the softmax rides the SIMT/vector lanes (no matmul classes).
        "expected": {"instruction_classes": binding.classes_for(op="matmul", output_dtype=odt),
                     "modes": entry.get("modes", {})},
        "required_oracle_tiers": list(binding.tiers), "vcs": "optional", "firesim": "optional",
    }
    sc_attr = f", scale = {scale} : f32"
    cap_attr = f", softcap = {float(softcap)} : f32" if softcap is not None else ""
    L = _iface_prelude(binding.target, entry.get("comment", ""))
    L += [
        f'  %{q} = merlin_iface.tensor {{name = "{q}", role = "input"}} : tensor<{M}x{H}x{midt}>',
        f'  %{k} = merlin_iface.tensor {{name = "{k}", role = "input"}} : tensor<{Skv}x{H}x{midt}>',
        f'  %{v} = merlin_iface.tensor {{name = "{v}", role = "input"}} : tensor<{Skv}x{Dv}x{midt}>',
    ]
    _blk = binding.scale_block if is_block_scaled(binding.operand_dtype) else None
    if _blk and H % _blk == 0 and Skv % _blk == 0:
        _hg, _sg = H // _blk, Skv // _blk
        L += [
            f'  %{q}_scale = merlin_iface.tensor {{name = "{q}_scale", role = "scale", '
            f'scale_of = "{q}", block = {_blk} : i64}} : tensor<{_hg}x{M}xi8>',
            f'  %{k}_scale = merlin_iface.tensor {{name = "{k}_scale", role = "scale", '
            f'scale_of = "{k}", block = {_blk} : i64}} : tensor<{_hg}x{Skv}xi8>',
            f'  %{v}_scale = merlin_iface.tensor {{name = "{v}_scale", role = "scale", '
            f'scale_of = "{v}", block = {_blk} : i64}} : tensor<{_sg}x{Dv}xi8>',
            f'  %P_scale = merlin_iface.tensor {{name = "P_scale", role = "scale", '
            f'scale_of = "P", block = {_blk} : i64}} : tensor<{_sg}x{M}xi8>',
        ]
    L += [
        f'  %S = merlin_iface.attention_qk %{q}, %{k} {{name = "S", block_scale = "e8m0", '
        f'output_dtype = "{odt}"}} : (tensor<{M}x{H}x{midt}>, tensor<{Skv}x{H}x{midt}>) '
        f'-> tensor<{M}x{Skv}x{modt}>',
        f'  %P = merlin_iface.softmax %S {{name = "P", axis = 1 : i64{sc_attr}{cap_attr}}} '
        f': (tensor<{M}x{Skv}x{modt}>) -> tensor<{M}x{Skv}x{modt}>',
        f'  %{out} = merlin_iface.attention_pv %P, %{v} {{name = "{out}", block_scale = "e8m0", '
        f'output_dtype = "{odt}"}} : (tensor<{M}x{Skv}x{modt}>, tensor<{Skv}x{Dv}x{midt}>) '
        f'-> tensor<{M}x{Dv}x{modt}>', "}"]
    return cap, "\n".join(L) + "\n"


def build_rmsnorm_qkv(entry: dict, binding: CorpusBinding) -> tuple[dict, str]:
    """Fused pre-norm QKV projection (op == rmsnorm_qkv): H = rmsnorm(X, gamma); Y = H @ Wqkv. A SIMT
    op (rmsnorm on the vector lanes feeding a matmul); golden is ordinary IEEE float (driver-composed).
    Interface chains the two EXISTING merlin_iface ops (rmsnorm -> matmul)."""
    D = binding.tile_dim
    M = entry.get("M", entry.get("M_tiles", 1) * D)
    K = entry.get("K", entry.get("K_tiles", 1) * D)
    N = entry.get("N", entry.get("N_tiles", 1) * D)
    x, gamma, weight, out = (entry.get("src", "X"), entry.get("gamma", "G"),
                             entry.get("weight", "Wqkv"), entry.get("out", "Y0"))
    eps = entry.get("eps", 1.0 / 65536.0)
    idt, odt = binding.cap_dtype(binding.operand_dtype), binding.cap_dtype(binding.accum_dtype)
    midt, modt = binding.mlir_dtype(binding.operand_dtype), binding.mlir_dtype(binding.accum_dtype)
    attrs = {"src": x, "gamma": gamma, "weight": weight, "out": out, "eps": eps,
             "semantic": "rmsnorm_qkv", "output_dtype": odt}
    cap = {
        "name": entry["name"], "kind": entry["kind"], "source_role": entry["source_role"],
        "source_reference": entry["source_reference"], "label": entry.get("label", "public"),
        "interface_mlir": "capsule.interface.mlir",
        "inputs": [{"name": x, "role": "input", "shape": [M, K], "dtype": idt},
                   {"name": gamma, "role": "weight", "shape": [1, K], "dtype": idt},
                   {"name": weight, "role": "weight", "shape": [K, N], "dtype": idt}],
        "operation": {"op": "rmsnorm_qkv", "attributes": attrs},
        "numeric_policy": _numeric_policy(binding, binding.accum_dtype, None),
        "expected": {"instruction_classes": binding.classes_for(op="matmul", output_dtype=odt),
                     "modes": entry.get("modes", {"rmsnorm": True})},
        "required_oracle_tiers": list(binding.tiers), "vcs": "optional", "firesim": "optional",
    }
    L = _iface_prelude(binding.target, entry.get("comment", ""))
    L += [
        f'  %{x} = merlin_iface.tensor {{name = "{x}", role = "input"}} : tensor<{M}x{K}x{midt}>',
        f'  %{gamma} = merlin_iface.tensor {{name = "{gamma}", role = "weight"}} : tensor<1x{K}x{midt}>',
        f'  %{weight} = merlin_iface.tensor {{name = "{weight}", role = "weight"}} : tensor<{K}x{N}x{midt}>',
        f'  %H = merlin_iface.rmsnorm %{x}, %{gamma} {{name = "H", eps = {eps:.9e} : f64, '
        f'output_dtype = "{odt}"}} : (tensor<{M}x{K}x{midt}>, tensor<1x{K}x{midt}>) -> tensor<{M}x{K}x{modt}>',
        f'  %{weight}_res = merlin_iface.resident_pack %{weight} {{layout = "packed_rhs"}} '
        f': (tensor<{K}x{N}x{midt}>) -> !merlin_iface.resident',
        f'  %acc0 = merlin_iface.matmul %H, %{weight}_res '
        f': (tensor<{M}x{K}x{modt}>, !merlin_iface.resident) -> !merlin_iface.acc<{modt}>',
        f'  %{out} = merlin_iface.commit %acc0 {{name = "{out}", epilogue = [], output_dtype = "{odt}"}} '
        f': (!merlin_iface.acc<{modt}>) -> tensor<{M}x{N}x{modt}>',
        f'  merlin_iface.evict %{weight}_res : (!merlin_iface.resident) -> ()', "}"]
    return cap, "\n".join(L) + "\n"


def build_rope_qkv(entry: dict, binding: CorpusBinding) -> tuple[dict, str]:
    """Fused QKV projection + RoPE (op == rope_qkv): H = X @ Wqkv; Y = rope(H) (GPT-NeoX/Llama rotation,
    theta=10000, position = row). SIMT float; golden driver-composed. Interface chains matmul -> rope."""
    D = binding.tile_dim
    M = entry.get("M", entry.get("M_tiles", 1) * D)
    K = entry.get("K", entry.get("K_tiles", 1) * D)
    N = entry.get("N", entry.get("N_tiles", 1) * D)
    x, weight, out = entry.get("src", "X"), entry.get("weight", "Wqkv"), entry.get("out", "Y0")
    theta = float(entry.get("rope_theta", 10000.0))
    idt, odt = binding.cap_dtype(binding.operand_dtype), binding.cap_dtype(binding.accum_dtype)
    midt, modt = binding.mlir_dtype(binding.operand_dtype), binding.mlir_dtype(binding.accum_dtype)
    attrs = {"src": x, "weight": weight, "out": out, "rope_theta": theta,
             "semantic": "rope_qkv", "output_dtype": odt}
    cap = {
        "name": entry["name"], "kind": entry["kind"], "source_role": entry["source_role"],
        "source_reference": entry["source_reference"], "label": entry.get("label", "public"),
        "interface_mlir": "capsule.interface.mlir",
        "inputs": [{"name": x, "role": "input", "shape": [M, K], "dtype": idt},
                   {"name": weight, "role": "weight", "shape": [K, N], "dtype": idt}],
        "operation": {"op": "rope_qkv", "attributes": attrs},
        "numeric_policy": _numeric_policy(binding, binding.accum_dtype, None),
        "expected": {"instruction_classes": binding.classes_for(op="matmul", output_dtype=odt),
                     "modes": entry.get("modes", {"rope": True})},
        "required_oracle_tiers": list(binding.tiers), "vcs": "optional", "firesim": "optional",
    }
    L = _iface_prelude(binding.target, entry.get("comment", ""))
    L += [
        f'  %{x} = merlin_iface.tensor {{name = "{x}", role = "input"}} : tensor<{M}x{K}x{midt}>',
        f'  %{weight} = merlin_iface.tensor {{name = "{weight}", role = "weight"}} : tensor<{K}x{N}x{midt}>',
        f'  %{weight}_res = merlin_iface.resident_pack %{weight} {{layout = "packed_rhs"}} '
        f': (tensor<{K}x{N}x{midt}>) -> !merlin_iface.resident',
        f'  %acc0 = merlin_iface.matmul %{x}, %{weight}_res '
        f': (tensor<{M}x{K}x{midt}>, !merlin_iface.resident) -> !merlin_iface.acc<{modt}>',
        f'  %H = merlin_iface.commit %acc0 {{name = "H", epilogue = [], output_dtype = "{odt}"}} '
        f': (!merlin_iface.acc<{modt}>) -> tensor<{M}x{N}x{modt}>',
        f'  %{out} = merlin_iface.rope %H {{name = "{out}", theta = {theta} : f64, '
        f'output_dtype = "{odt}"}} : (tensor<{M}x{N}x{modt}>) -> tensor<{M}x{N}x{modt}>',
        f'  merlin_iface.evict %{weight}_res : (!merlin_iface.resident) -> ()', "}"]
    return cap, "\n".join(L) + "\n"


def _batched_mx_inputs(lhs: str, weight: str, B: int, M: int, H: int, N: int, idt: str,
                       binding: CorpusBinding) -> list[dict]:
    """Declared leaf operands of a BATCHED block-scaled matmul: one scale stream per batch.

    Same reasoning as :func:`_matmul_inputs` -- the elements alone are half the operand -- with the batch
    dimension leading, because each batch is an independent GEMM with its own block scales."""
    inputs = [{"name": weight, "role": "weight", "shape": [B, H, N], "dtype": idt},
              {"name": lhs, "role": "input", "shape": [B, M, H], "dtype": idt}]
    blk = binding.scale_block if is_block_scaled(binding.operand_dtype) else None
    if blk and H % blk == 0:
        g = H // blk
        inputs += [
            {"name": f"{lhs}_scale", "role": "scale", "scale_of": lhs, "block": blk,
             "shape": [B, g, M], "dtype": "e8m0"},
            {"name": f"{weight}_scale", "role": "scale", "scale_of": weight, "block": blk,
             "shape": [B, g, N], "dtype": "e8m0"},
        ]
    return inputs


def build_gemv_batched_mx(entry: dict, binding: CorpusBinding) -> tuple[dict, str]:
    """Batched MX matmul (op == gemv_batched): B independent MX GEMMs A_b[M,H] @ W_b[H,N] on the mx_pe,
    output stacked row-major to [B*M, N] bf16. mxfp8; golden from mx_ref. Interface presents the batched
    tensors + a batched matmul op (the device iterates the B tiles)."""
    D = binding.tile_dim
    B = int(entry.get("B", 2))
    M = entry.get("M", entry.get("M_tiles", 1) * D)
    H = entry.get("H", entry.get("K", 2 * D))
    N = entry.get("N", D)
    lhs, weight, out = entry.get("lhs", "A0"), entry.get("weight", "W"), entry.get("out", "Y0")
    idt, odt = binding.cap_dtype(binding.operand_dtype), binding.cap_dtype(binding.accum_dtype)
    midt, modt = binding.mlir_dtype(binding.operand_dtype), binding.mlir_dtype(binding.accum_dtype)
    # BLOCK SCALING IS A PROPERTY OF THE DATATYPE, NOT OF BATCHING. The operand list and the interface
    # already emit the per-batch scale streams only when the datatype is block-scaled; the attribute
    # naming their encoding was declared unconditionally, so on a plain-integer target this op claimed
    # an e8m0 scale encoding whose streams do not exist. Declared where the streams are.
    scaled = is_block_scaled(binding.operand_dtype)
    attrs = {"lhs": lhs, "weight": weight, "out": out, "batch": B,
             "semantic": "gemv_batched", "output_dtype": odt}
    if scaled:
        attrs["block_scale"] = "e8m0"
    cap = {
        "name": entry["name"], "kind": entry["kind"], "source_role": entry["source_role"],
        "source_reference": entry["source_reference"], "label": entry.get("label", "public"),
        "interface_mlir": "capsule.interface.mlir",
        "inputs": _batched_mx_inputs(lhs, weight, B, M, H, N, idt, binding),
        "operation": {"op": "gemv_batched", "attributes": attrs},
        "numeric_policy": _numeric_policy(binding, binding.accum_dtype, None),
        "expected": {"instruction_classes": binding.classes_for(op="matmul", output_dtype=odt),
                     "modes": entry.get("modes", {})},
        "required_oracle_tiers": list(binding.tiers), "vcs": "optional", "firesim": "optional",
    }
    L = _iface_prelude(binding.target, entry.get("comment", ""))
    L += [
        f'  %{lhs} = merlin_iface.tensor {{name = "{lhs}", role = "input"}} : tensor<{B}x{M}x{H}x{midt}>',
        f'  %{weight} = merlin_iface.tensor {{name = "{weight}", role = "weight"}} : tensor<{B}x{H}x{N}x{midt}>',
    ]
    _blk = binding.scale_block if is_block_scaled(binding.operand_dtype) else None
    if _blk and H % _blk == 0:
        # The batched form carries one scale stream per batch: the op declares `block_scale = "e8m0"`
        # already, and these are the streams that attribute refers to.
        _g = H // _blk
        L += [
            f'  %{lhs}_scale = merlin_iface.tensor {{name = "{lhs}_scale", role = "scale", '
            f'scale_of = "{lhs}", block = {_blk} : i64}} : tensor<{B}x{_g}x{M}xi8>',
            f'  %{weight}_scale = merlin_iface.tensor {{name = "{weight}_scale", role = "scale", '
            f'scale_of = "{weight}", block = {_blk} : i64}} : tensor<{B}x{_g}x{N}xi8>',
        ]
    scale_attr = 'block_scale = "e8m0", ' if scaled else ""
    L += [
        f'  %{out} = merlin_iface.matmul_batched %{lhs}, %{weight} {{name = "{out}", batch = {B} : i64, '
        f'{scale_attr}output_dtype = "{odt}"}} '
        f': (tensor<{B}x{M}x{H}x{midt}>, tensor<{B}x{H}x{N}x{midt}>) -> tensor<{B * M}x{N}x{modt}>', "}"]
    return cap, "\n".join(L) + "\n"


def build_conv2d(entry: dict, binding: CorpusBinding) -> tuple[dict, str]:
    """An im2col conv2d capsule (op == conv2d): NHWC IFM + pre-im2col'd weight [KH*KW*Ci, Cout] -> a resident
    matmul over conv windows, output [Ho*Wo, Cout]. Reuses the runtime's canonical conv geometry so the golden
    (capsule_golden conv2d branch) and the harness agree. Native operand dtype (e.g. gemmini int8)."""
    from merlin.runtime.commandbuffer import conv_out_dims
    ifm, weight, out = entry.get("ifm", "IFM"), entry.get("weight", "W"), entry.get("out", "Y0")
    ci = entry.get("ci", entry.get("Cin", 4))
    cout = entry.get("N", entry.get("Cout", binding.tile_dim))
    H, W = entry.get("Himg", 8), entry.get("Wimg", 8)
    kh, kw = entry.get("kh", 3), entry.get("kw", 3)
    stride = list(entry.get("stride", [1, 1]))
    padding = list(entry.get("padding", [0, 0, 0, 0]))
    dilation = list(entry.get("dilation", [1, 1]))
    Ho, Wo = conv_out_dims(H, W, kh, kw, stride, padding, dilation)
    Kdim = kh * kw * ci
    epilogue = list(entry.get("epilogue", []))
    output_dtype = _resolve_output_dtype(binding, epilogue, entry.get("output_dtype"))
    idt, odt = binding.cap_dtype(binding.operand_dtype), binding.cap_dtype(output_dtype)
    midt, modt = binding.mlir_dtype(binding.operand_dtype), binding.mlir_dtype(output_dtype)
    attrs = {"ifm": ifm, "weight": weight, "out": out, "ci": ci, "kh": kh, "kw": kw,
             "stride": stride, "padding": padding, "dilation": dilation, "layout": "nhwc",
             "epilogue": epilogue, "output_dtype": odt, "semantic": "conv2d_im2col"}
    # A pooling epilogue is fused onto the conv's store path (the fused conv loop takes the window and
    # stores the pooled result), so it rides on this op's own attributes. ``pool_in_dims`` is DERIVED
    # from the conv's output extent rather than declared: the golden sees only the flat [N*Ho*Wo, Co]
    # product and has to trust the declaration, so deriving it is what keeps the golden and the
    # simulator pooling the same image. The result extent below is the POOLED one.
    pool_attrs, out_rows = _pool_epilogue(entry, rows=Ho * Wo, in_dims=(Ho, Wo), op="conv2d")
    attrs.update(pool_attrs)
    cap = {
        "name": entry["name"], "kind": entry["kind"], "source_role": entry["source_role"],
        "source_reference": entry["source_reference"], "label": entry.get("label", "public"),
        "interface_mlir": "capsule.interface.mlir",
        "inputs": [{"name": weight, "role": "weight", "shape": [Kdim, cout], "dtype": idt},
                   {"name": ifm, "role": "input", "shape": [1, H, W, ci], "dtype": idt}],
        "operation": {"op": "conv2d", "attributes": attrs},
        "numeric_policy": _numeric_policy(binding, output_dtype, entry.get("acc_scale")),
        "expected": {"instruction_classes": binding.classes_for(op="conv2d", output_dtype=odt,
                                                                epilogue=epilogue, movement=False),
                     "modes": dict(entry["modes"]) if "modes" in entry
                     else {"conv2d": True, "k_accumulate": True}},
        "required_oracle_tiers": list(binding.tiers), "vcs": "optional", "firesim": "optional",
    }
    epi = ", ".join(f'"{e}"' for e in epilogue)
    L = _iface_prelude(binding.target, entry.get("comment", ""))
    L += [
        f'  %{ifm} = merlin_iface.tensor {{name = "{ifm}", role = "input"}} : tensor<1x{H}x{W}x{ci}x{midt}>',
        f'  %{weight} = merlin_iface.tensor {{name = "{weight}", role = "weight"}} : tensor<{Kdim}x{cout}x{midt}>',
        f'  %{weight}_res = merlin_iface.resident_pack %{weight} {{layout = "packed_conv_rhs"}} '
        f': (tensor<{Kdim}x{cout}x{midt}>) -> !merlin_iface.resident',
        f'  %{out} = merlin_iface.conv2d %{ifm}, %{weight}_res {{kernel = [{kh}, {kw}, {ci}, {cout}], '
        f'stride = [{stride[0]}, {stride[1]}], padding = [{padding[0]}, {padding[1]}, {padding[2]}, {padding[3]}], '
        f'dilation = [{dilation[0]}, {dilation[1]}], name = "{out}", epilogue = [{epi}], '
        f'output_dtype = "{odt}", layout = "nhwc"{_pool_mlir_attrs(pool_attrs)}}} '
        f': (tensor<1x{H}x{W}x{ci}x{midt}>, !merlin_iface.resident) -> tensor<{out_rows}x{cout}x{modt}>',
        f'  merlin_iface.evict %{weight}_res : (!merlin_iface.resident) -> ()', "}"]
    return cap, "\n".join(L) + "\n"


# op -> builder, for the driver to dispatch on the entry's declared op.
BUILDERS: dict[str, Callable[[dict, CorpusBinding], tuple[dict, str]]] = {
    "matmul": build_matmul, "linear": build_matmul,
    "resident_reuse": build_resident_reuse, "movement": build_movement,
    "attention_qk": build_attention_qk, "rmsnorm": build_rmsnorm, "conv2d": build_conv2d,
    "attention_mx": build_attention_mx, "rmsnorm_qkv": build_rmsnorm_qkv, "rope_qkv": build_rope_qkv,
    "gemv_batched": build_gemv_batched_mx,
}


#: Capsule kind -> the generalization axis it probes. A fact about what the capsule tests, not an
#: opinion about it, so it is derived rather than re-typed on every entry.
_AXIS_BY_KIND: dict[str, str] = {"isa": "seen", "layer": "composition",
                                 "model_slice": "composition", "model": "model"}


def _semantic_block(entry: dict, binding: CorpusBinding) -> dict:
    """The capsule's generalization-intent block: half derived, half authorial.

    ``semantic_family`` is DERIVED from the op — the same mapping the eligibility oracle and the
    leave-one-family-out splitter already fall back to, so a generated capsule and a graded capsule can
    never disagree about which family they belong to.

    ``must_accelerate`` / ``generalization_axis`` are NOT derivable: they are the author's claim about
    what this capsule is for, so they come from the profile entry (or the profile's
    ``datapath.semantic_defaults``). ⚠️ ``must_accelerate`` defaults to FALSE on purpose. It is the
    strongest assertion in the block — an eligible region that falls back is a hard failure — and
    defaulting it true would turn every generated capsule into a gate the moment this lands.

    Emitting this here is what stops a corpus regeneration from silently deleting the annotation:
    before, the 40 radiance blocks were hand-written and ``_write_capsule`` dumps the dict wholesale,
    so a regen would have zeroed the only working generalization annotation in the repo.
    """
    from merlin.targetgen import semantic_families as _sf

    defaults = dict(getattr(binding, "semantic_defaults", None) or {})
    # ⚠️ NOT ``entry["semantic"]`` -- a profile entry already uses that key for a free-form op-semantics
    # label ("quantized_linear", "matmul_mxu0") that flows into ``operation.attributes.semantic``.
    # Reusing it here would collide with 12 existing entries across gemmini and atlas and crash the
    # generator, so the generalization-intent override gets a key of its own.
    raw = entry.get("generalization")
    authored = dict(raw) if isinstance(raw, dict) else {}
    fam = authored.get("semantic_family") or _sf.from_op(entry.get("op", "matmul"))
    block: dict = {}
    if fam:
        block["semantic_family"] = fam
    # The axis follows the capsule KIND, which is a fact about what the capsule tests rather than an
    # opinion: an ``isa`` capsule exercises a primitive the corpus has SEEN, a ``model_slice`` exercises
    # a COMPOSITION of primitives, and a whole ``model`` is the end-to-end integration case. Deriving it
    # keeps 40 annotations from having to be re-typed and keeps a new capsule from being mislabelled.
    kind = entry.get("kind", "isa")
    axis = authored.get("generalization_axis") or defaults.get("generalization_axis") \
        or _AXIS_BY_KIND.get(kind, "seen")
    block["generalization_axis"] = axis
    # A whole-model capsule spans families by construction, is never held out (see
    # generalization_splits), and is expected to contain regions the target cannot run -- so it must
    # never carry must_accelerate, whatever the corpus default says.
    must_default = False if kind == "model" else bool(defaults.get("must_accelerate", False))
    block["must_accelerate"] = bool(authored.get("must_accelerate", must_default))
    block["eligible"] = authored.get("eligible", defaults.get("eligible", "auto"))
    # FAMILIES THIS CAPSULE FUSES, so that a fused-only family can be covered at all.
    #
    # A target may declare a family reachable ONLY in composition -- `elementwise_map` with
    # `composed_with: [contraction]` on the target measured here. A standalone elementwise capsule would
    # then be the WRONG capsule (the eligibility oracle refuses it as a false fallback), so the only
    # capsule that can ever evidence that cell is a contraction with an epilogue. Crediting a capsule for
    # exactly one family therefore left such a cell permanently uncoverable while the requirement kept
    # demanding it -- a gap no capsule could close, reported forever as corpus debt.
    #
    # Derived, not declared: each epilogue stage the capsule carries is resolved to its family through
    # the same op->family table everything else uses, and kept only when the manifest says that family
    # composes with this capsule's own. A stage whose family does not resolve is recorded rather than
    # dropped, because silently withholding coverage a capsule earned is the failure this replaces.
    fused, unresolved = _fused_families(entry, fam, binding)
    if fused:
        block["composed_families"] = sorted(fused)
    if unresolved:
        block["composed_families_unresolved"] = sorted(unresolved)
    return block


def _fused_families(entry: dict, family: str | None, binding: CorpusBinding) -> tuple[set, set]:
    """``(families this capsule's epilogue exercises, stage names that did not resolve)``.

    Empty for a capsule with no epilogue: a plain contraction must not claim the epilogue cell, or every
    capsule in the corpus would cover a family none of them exercises.
    """
    from merlin.targetgen import semantic_families as _sf

    stages = [str(s) for s in (entry.get("epilogue") or []) if s]
    if not stages or not family:
        return set(), set()
    try:
        from merlin.targetgen.eligibility import capability_map_for_target
        cap_map = capability_map_for_target(getattr(binding, "target", "") or "")
    except Exception:                                      # noqa: BLE001 — unresolvable manifest
        return set(), set()
    fused, unresolved = set(), set()
    for stage in stages:
        sfam = _sf.from_op(stage)
        if sfam is None:
            unresolved.add(stage)
            continue
        cap = cap_map.get(sfam)
        if cap is not None and family in (getattr(cap, "composed_with", ()) or ()):
            fused.add(sfam)
    return fused, unresolved


def build(entry: dict, binding: CorpusBinding) -> tuple[dict, str]:
    """Dispatch an abstract capsule entry to its op builder -> (capsule dict, interface MLIR)."""
    op = entry.get("op", "matmul")
    if op not in BUILDERS:
        raise ValueError(f"no corpus builder for op {op!r} (have {sorted(BUILDERS)})")
    cap, mlir = BUILDERS[op](entry, binding)
    # AN EPILOGUE THE BUILDER DID NOT CARRY IS A COVERAGE LIE, so it is refused here rather than in each
    # builder. Only `matmul` and `conv2d` read the entry's `epilogue:`; every other builder writes its
    # own (often empty) list, so a stage declared on, say, a `movement` entry vanished from the capsule
    # -- while `_semantic_block` still credited that stage's FAMILY in `composed_families` off the same
    # entry. The capsule would then be counted as evidence for a family whose arithmetic no engine ever
    # performed, which is the exact failure the pooling epilogue was implemented to close. One check
    # here covers every present and future builder; per-matmul epilogues (resident_reuse) use their own
    # key and are unaffected.
    declared_epilogue = [str(x) for x in (entry.get("epilogue") or [])]
    if declared_epilogue:
        carried = [str(x) for x in ((cap.get("operation") or {}).get("attributes") or {}).get(
            "epilogue", [])]
        dropped = [x for x in declared_epilogue if x not in carried]
        if dropped:
            raise ValueError(
                f"{entry.get('name', op)}: the {op!r} builder dropped epilogue stage(s) {dropped} "
                f"(carried: {carried}). The capsule would still be CREDITED for those stages' semantic "
                f"families, so it would count as evidence for arithmetic nothing computed")
    # Stamped once here rather than in each builder, so every capsule a target emits carries the same
    # declaration and a new builder cannot silently forget it.
    if binding.inapplicable_tiers:
        cap["inapplicable_oracle_tiers"] = dict(binding.inapplicable_tiers)
    sem = _semantic_block(entry, binding)
    if sem:
        cap["semantic"] = sem
    return cap, mlir
