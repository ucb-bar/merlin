"""Target-agnostic capsule-corpus builder.

ONE builder turns an abstract capsule *entry* (op + shapes-in-tiles + epilogue — the target-agnostic test
definition, declared per target in ``capsules/<target>/corpus_profile.yaml``) plus a *binding* DERIVED from
the target's descriptor (dtypes, tile dim, compare policy, instruction classes, oracle tiers) into a
concrete capsule dict + interface MLIR. It replaces the two forked per-target generators (one
integer, one float): the LOGIC here is shared and carries no target
name or dtype literal in its control flow; the per-target DATA (numeric datapath, which capsules, dtypes)
lives in the descriptor + the profile.

Golden VALUES are not computed here (the integer engine lives in :mod:`capsule_golden`; the float engine
needs the external ``specir`` refmodel, available only at generation time) — the driver
``experiments/corpora/generate_corpus.py`` computes them and writes ``golden.yaml``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from merlin.development.targetgen.corpus.corpus_activation import activation_matmul_program
from merlin.development.targetgen.corpus.corpus_pool import build as build_maxpool2d
from merlin.development.targetgen.corpus.corpus_sibling_feed import build as _build_sibling_feed
from merlin.development.targetgen.corpus.declarations import _semantic_block
from merlin.development.targetgen.grading.golden_inputs import dtype_bits

# ---- corpus_spec -------------------------------------------------------------------------------

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
    "fp32": ("f32", "f32", False),  # alias: a contract may spell single precision "fp32"
}


def dtype_info(token: str) -> tuple[str, str, int | None, bool]:
    """(capsule spelling, MLIR spelling, byte width, is_integer) for a canonical dtype token.

    The width is DERIVED from ``merlin.common.target_data`` rather than declared here, so this module
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
    tile_dim: int | None  # None for source-only components with no hardware geometry grant
    operand_dtype: str  # canonical token, e.g. "int8" / "fp8_e4m3"
    accum_dtype: str  # canonical token, e.g. "i32" / "bf16"
    integer: bool  # numeric regime (drives compare policy + golden engine)
    tiers: list[str]  # required_oracle_tiers for this target
    compare: str  # "exact_int" | "tolerance_float"
    atol: float | None = None
    rtol: float | None = None
    scaling: str | None = None  # compute-unit SCALE_KIND (block_e8m0 -> the MX numeric regime) or None
    requant_output_dtype: str | None = None  # narrow output an acc_scale epilogue requants to (e.g. i8)
    # The round-half-up arithmetic shift the ABI's INTEGER ``requant`` epilogue stage applies. Declared
    # per target in the profile's ``datapath`` block (a capsule entry may override it); ``None`` means the
    # target has not declared one, and a capsule asking for the stage then FAILS TO BUILD rather than
    # inheriting a default. That is the point: three engines (golden, reference, simulator) each carry
    # their own fallback of 4, so an undeclared shift makes them agree with each other by coincidence
    # while the backend under test sees no shift at all and cannot know what to emit.
    requant_shift: int | None = None
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
    # Explicit data-only readout selection. None preserves legacy backend discovery;
    # an empty mapping is an explicit absence and must not discover another provider.
    readout_inputs: dict | None = None

    def cap_dtype(self, token: str) -> str:
        return dtype_info(token)[0]

    def mlir_dtype(self, token: str) -> str:
        return dtype_info(token)[1]


_DEFAULT_SW_TILE = 16


BLOCK_SCALED_DTYPES = frozenset({"mxfp4", "mxfp6", "mxfp8"})


def is_block_scaled(token: str | None) -> bool:
    """Whether an operand dtype token is a block-scaled (microscaling) format."""
    return str(token or "") in BLOCK_SCALED_DTYPES


def _scale_block_elems(contract: dict, *, numerical_semantics: dict | None = None) -> int | None:
    """Elements per block scale from selected software semantics or a legacy manifest.

    A block-scaled compute unit (``scaling: block_e8m0``) carries one shared exponent per fixed-length run
    of K elements. The run length is the target's OWN declared scale ``group`` -- searched for in the
    manifest rather than assumed, because a different microscaling profile uses a different one and a
    guessed length silently mis-shapes every scale operand.

    Returns None when the target declares no block-scaled unit, or declares one without a group. The
    caller then declares NO scale operands and the capsule keeps its present shape: fail closed, never a
    baked constant.
    """
    if numerical_semantics is not None:
        from merlin.target.spec.software import validate_numerical_semantics

        selected = validate_numerical_semantics(numerical_semantics)
        group = selected.get("block_size")
        if group is not None:
            return int(group)

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


def _structural_mesh_rows(facts: dict) -> int | None:
    """Read source-bound grid geometry without treating it as a compute array."""
    body = facts.get("facts") or {}
    observations = [
        o
        for o in body.get("structural_observations") or []
        if isinstance(o, dict) and o.get("kind") == "mesh_tile_grid"
    ]
    if not observations:
        return None
    consistency = facts.get("source_consistency") or {}
    fir_sha = (facts.get("inputs") or {}).get("fir_sha256")
    production = consistency.get("production") or {}
    if (
        consistency.get("status") != "verified"
        or not isinstance(fir_sha, str)
        or len(fir_sha) != 64
        or production.get("firrtl_sha256") != fir_sha
        or (body.get("source") or {}).get("fir_sha256") != fir_sha
    ):
        raise ValueError("structural mesh lacks a matching verified FIRRTL source")
    if len(observations) != 1:
        raise ValueError("structural mesh geometry is ambiguous")
    observation = observations[0]
    rows, cols, instances = (observation.get(k) for k in ("rows", "cols", "instances"))
    if (
        observation.get("source") != "selected_firrtl"
        or observation.get("firrtl_sha256") != fir_sha
        or any(type(v) is not int or v < 1 for v in (rows, cols, instances))
        or rows * cols != instances
    ):
        raise ValueError("structural mesh geometry does not match its selected FIRRTL")
    return rows


def _tile_dim(target: str, contract: dict, *, operand: str | None = None, facts: dict | None = None) -> int:
    """Tile dim for sizing capsule shapes. When the target has a FIXED HARDWARE mesh, it is DERIVED
    (``capabilities.mesh.rows`` / ``.tile.rows`` from the manifest, else the CIRCT ``arrays[mesh].rows``
    fact or a source-bound structural mesh observation) — so a 16x16 mesh's 16 comes from its RTL facts,
    never a literal. A target with NO fixed hardware
    mesh (a SIMT / vector target, whose matmul tiling is a SOFTWARE choice, not a
    hardware dimension) has nothing to derive; it uses ``_DEFAULT_SW_TILE`` — a compiler software-tiling
    default, NOT a per-target hardware fact. Both derivation sources are keyed on ``target``."""
    # A matrix extension driven as INSTRUCTIONS has two different tile notions at once, and they are
    # different NUMBERS: the spatial array's own edge (the manifest's mesh/tile rows — 16 for the unit
    # this was measured on) and the LOGICAL tile a kernel addresses, which is VLMAX for the element width
    # (32 on the same hardware at VLEN 256). Sizing capsules against the array edge on the instruction
    # surface would bracket the wrong boundary — the tile-edge cases would sit mid-tile and prove nothing.
    # So a unit that declares which hardware CONFIG it is elaborated as gets its edge from that config's
    # own Scala, and a declared-but-underivable geometry raises instead of falling back to a mesh row.
    from merlin.common.target_data import get as quant_format

    units = contract.get("compute_units") or []
    if operand is not None:
        selected_format = quant_format(operand).name
        selected = [u for u in units if any(quant_format(dt).name == selected_format for dt in (u.get("dtypes") or []))]
        if units and not selected:
            raise ValueError(f"{target}: operand dtype {operand!r} is admitted by no compute unit")
    else:
        selected = units[:1]
    if selected and not any(u.get("kind") in {"systolic", "spatial"} for u in selected):
        return _DEFAULT_SW_TILE
    unit, config = _declared_matrix_unit(contract), _declared_hardware_config(contract)
    if unit and config and facts is None:
        from merlin.target.registry.plugins import load_declared

        return int(load_declared(target, "matrix_lowering").geometry(unit=unit, config=config)[0])
    caps = contract.get("capabilities") or {}
    for geom in (caps.get("mesh") or {}, caps.get("tile") or {}):  # systolic mesh OR spatial tile
        if geom.get("rows"):
            return int(geom["rows"])
    selected_facts = None
    try:
        from merlin.target.rtl.facts import load_facts

        selected_facts = load_facts(target) if facts is None else facts
        body = selected_facts.get("facts") or {}
        arrays = body.get("arrays") or []
        m = next((a for a in arrays if a.get("name") == "mesh"), {})
        if m.get("rows"):
            return int(m["rows"])
        # Spatial tile extractors publish a field bundle, not a systolic
        # arrays[] record. Keep the two fact shapes distinct.
        spatial_rows = (((body.get("fields") or {}).get("tile_dim") or {}).get("value") or {}).get("rows")
        if spatial_rows:
            return int(spatial_rows)
    except Exception:  # noqa: BLE001 — handled below according to the declared compute-unit kind
        pass
    if selected_facts is not None:
        structural_rows = _structural_mesh_rows(selected_facts)
        if structural_rows is not None:
            return structural_rows
    # A software tile is legitimate only when no unit selected by the operand
    # is fixed. A missing RTL array is missing evidence, not evidence for a
    # 16-wide array. This also covers single-unit
    # contracts whose geometry is intentionally absent from the authored contract.
    if any(u.get("kind") in {"systolic", "spatial"} for u in selected):
        raise ValueError(f"{target}: fixed compute unit has no derived tile geometry in its contract or RTL facts")
    return _DEFAULT_SW_TILE


def _inapplicable_tiers(datapath: dict, required) -> dict[str, str]:
    """``{tier: reason}`` for the oracle tiers this target declares CANNOT corroborate a result.

    Each entry must carry a non-empty reason — a tier switched off without one is indistinguishable from
    a tier switched off because it was failing. A tier that is both REQUIRED and inapplicable is a
    contradiction in the declaration and raises rather than resolving silently either way."""
    raw = datapath.get("inapplicable_oracle_tiers") or {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"datapath.inapplicable_oracle_tiers must be a {{tier: reason}} mapping, got {type(raw).__name__}"
        )
    out: dict[str, str] = {}
    for tier, reason in raw.items():
        if not str(reason or "").strip():
            raise ValueError(
                f"inapplicable oracle tier {tier!r} declares no reason; a tier is only "
                f"skippable with the reason it cannot corroborate a result"
            )
        if tier in set(required or ()):
            raise ValueError(f"oracle tier {tier!r} is declared BOTH required and inapplicable")
        out[str(tier)] = str(reason).strip()
    return out


def derived_accumulator(contract: dict, operand: str, facts: dict | None = None) -> str | None:
    """The accumulate dtype the target itself states for ``operand``, or ``None`` when nothing does.

    Two sources, in order: the primary compute unit's declared ``accumulate`` matrix, then the RTL
    facts' ``datapaths`` (the ``input``/``accumulator`` pair a CIRCT extraction reports). The facts pair
    is used only when its input element format IS the operand's, so an integer accumulator is never
    attached to a floating operand because the unit also carries one. ``None`` is an answer: a caller
    that must not guess (a captured group stated as a capsule) refuses on it.
    """
    from merlin.common.target_data import get as quant_format

    cu = (contract.get("compute_units") or [{}])[0]
    for acc in cu.get("accumulate") or []:
        if acc.get("acc"):
            return str(acc["acc"])
    body = facts if isinstance(facts, dict) else {}
    body = body.get("facts") if isinstance(body.get("facts"), dict) else body
    rows = {str(row.get("name")): row for row in (body.get("datapaths") or []) if isinstance(row, dict)}
    source, accumulator = rows.get("input") or {}, rows.get("accumulator") or {}
    if not source.get("dtype") or not accumulator.get("dtype"):
        return None
    try:
        same = quant_format(str(source["dtype"])).name == quant_format(operand).name
    except (KeyError, ValueError):
        same = False
    return str(accumulator["dtype"]) if same else None


def _accum_dtype(contract: dict, operand: str, facts: dict | None = None) -> str:
    """Accumulate dtype the target states (:func:`derived_accumulator`), else the widening default for an
    integer operand (i32) — a family property (``widening_integer_accumulate``), not a target one."""
    stated = derived_accumulator(contract, operand, facts)
    if stated is not None:
        return stated
    return "i32" if dtype_info(operand)[3] else "f32"


def entry_binding(entry: dict, binding: CorpusBinding) -> tuple[str, CorpusBinding]:
    """Route an entry to its numeric regime and return the per-entry binding (operand/accum overridden).

    ``int`` (integer mesh), ``specir`` (fp8), ``mx`` (microscaling block-scaled FP), ``simt`` (IEEE
    fp16/bf16/f32), routed purely by the entry's operand dtype token -- no target name. One definition,
    used by the Phase 0 writer and by every caller that states a single entry as a capsule, so a group
    asked of a package and the capsule that certifies it are built under the same binding.
    """
    import dataclasses

    tok = entry.get("operand_dtype") or binding.operand_dtype
    regime = regime_for_dtype(tok)
    acc = {"mx": "bf16", "simt": "f32"}.get(regime, binding.accum_dtype)
    eb = dataclasses.replace(
        binding,
        operand_dtype=tok,
        accum_dtype=acc,
        integer=(regime == "int"),
        compare=("exact_int" if regime == "int" else "tolerance_float"),
    )
    return regime, eb


def _declared_hardware_config(contract: dict) -> str | None:
    """The elaborated hardware configuration a compute unit declares (a config class in the unit's own
    generator). Which config a target IS is a statement about the hardware being targeted, not a fact to
    derive; what it IMPLIES — tile edge, operand alignment — is derived from that config's declaration."""
    for cu in contract.get("compute_units") or []:
        cfg = cu.get("hardware_config")
        if cfg:
            return str(cfg)
    return None


def _declared_matrix_unit(contract: dict) -> str | None:
    """The ``matrix_unit`` a compute unit declares, if any — the name of a block in ``matrix_units.yaml``.

    Declared per compute unit rather than per target because a target can carry both a matrix extension
    and something else, and it is the UNIT that has the encodings.
    """
    for cu in contract.get("compute_units") or []:
        name = cu.get("matrix_unit")
        if name:
            return str(name)
    return None


def _classes_source(te, contract: dict, *, taxonomy: dict | None = None) -> Callable[..., list[str]]:
    """The instruction-class deriver for a matmul-family op. Two regimes, chosen by what the target ships:
    a self-hosted-ISA target (an ``isa_definition.py`` is present) derives its classes from the taxonomy;
    a RoCC/command target derives the RoCC semantic classes from its ``encoding`` map. Never hardcoded."""
    # Explicit effect declarations avoid provider discovery and legacy issue schedules.
    enc = contract.get("encoding") or {}
    if "corpus_required_classes" in enc or "class_family" in enc:
        from merlin.development.targetgen.corpus.selection import source

        return source(enc)

    from merlin.development.targetgen.corpus.selection import selected_classes
    from merlin.target.isa import taxonomy as IT

    explicit_roles = selected_classes(contract)
    if explicit_roles is not None:
        return explicit_roles

    # A compute unit may declare that it IS a matrix extension whose encodings are derived by the
    # matrix-unit reader (``matrix_units.yaml``). That surface ships no isa_definition.py and no RoCC
    # ``encoding`` map, so both regimes below saw nothing and the class list came out empty — the state in
    # which a coverage expectation is satisfied by emitting no instructions at all. Checked FIRST and
    # allowed to raise: a declared unit whose classes cannot be derived must stop corpus generation, not
    # quietly produce an unfalsifiable corpus.
    unit = _declared_matrix_unit(contract)
    if unit and taxonomy is None:
        return IT.matrix_unit_classes_for(unit, support_target=te.target)
    if taxonomy is None:
        try:
            tax = IT.derive_isa_taxonomy(te)
        except Exception:  # noqa: BLE001
            tax = {}
    else:
        tax = taxonomy
    if tax.get("by_class"):

        def _from_taxonomy(*, op="matmul", output_dtype=None, epilogue=(), movement=False):
            return IT.required_classes_for_op(
                tax, op=op, output_dtype=output_dtype, epilogue=tuple(epilogue), movement=movement
            )

        return _from_taxonomy
    # A self-hosted ISA with missing model tooling is not a command-ISA target.
    # In particular, an unknown taxonomy must not be misdiagnosed as a missing
    # RoCC issue order or silently generate capsules with no required classes.
    if "self_hosted_isa" in (contract.get("features") or ()) or any(
        str(header).endswith("isa_definition.py") for header in (getattr(te, "isa_headers", ()) or ())
    ):
        IT.require_taxonomy(tax, f"target {getattr(te, 'target', '?')!r}", needs="Phase 0 instruction-class derivation")
    # Command target: its OWN contract declares the corpus obligation and issue order. A common
    # transport says nothing about whether this machine is weight-stationary, has configuration
    # instructions, or even names its work as a load/compute/store sequence. Do not infer a
    # device's requirements from a shared literal or silently remove undeclared classes.
    enc = contract.get("encoding") or {}
    sem_vals = set((enc.get("semantic_class") or {}).values())
    sub_vals = set((enc.get("config_subtype") or {}).values())
    pool = sem_vals | sub_vals
    sequence = enc.get("corpus_issue_order")
    if not isinstance(sequence, list) or not sequence or any(not isinstance(c, str) or not c for c in sequence):
        raise ValueError("command target must declare non-empty encoding.corpus_issue_order")
    if len(sequence) != len(set(sequence)):
        raise ValueError("encoding.corpus_issue_order contains duplicate instruction classes")
    missing = set(sequence) - pool
    if missing:
        raise ValueError(f"encoding.corpus_issue_order names undeclared instruction classes: {sorted(missing)}")
    classes = list(sequence)

    def _from_encoding(*, op="matmul", output_dtype=None, epilogue=(), movement=False):
        # ⚠️ NOT `list(classes)` unconditionally. This deriver was written for the matmul family and is
        # reached for EVERY op on a RoCC target, so it used to answer the weight-stationary systolic
        # sequence for a pure data copy: measured, six tracked capsules (two of them HIDDEN) required
        # PRELOAD + COMPUTE_PRELOADED of a `movement` capsule that performs no arithmetic at all. That is
        # the fabricated requirement `isa_taxonomy.required_role_slots` fails closed to avoid, in the
        # direction running the capsule cannot catch: a backend that correctly moves the tile and issues
        # no multiply is marked non-conformant. `build_movement` even passes `movement=True` for exactly
        # this reason -- the argument was accepted and discarded.
        #
        # Which classes belong to the contraction is READ from the shared class vocabulary
        # (`semantic_families._ISA_CLASS_FAMILY`, whose whole job is this mapping) rather than named
        # here. A class that table does not recognise -- CONFIG*/FLUSH plumbing, or a spelling it has
        # not learned -- is KEPT: dropping what we cannot classify is the silent-drop this repo's
        # parsing rule forbids, and over-requiring plumbing is harmless where over-requiring a multiply
        # is not.
        from merlin.target.capabilities import semantic_families as _sf

        prims = _sf.primitives_of(_sf.from_op(op) or "")
        if movement or (prims and "contraction" not in prims):
            return [c for c in classes if _sf.from_isa_class(c) != "contraction"]
        return list(classes)

    return _from_encoding


_DATAPATH_AUTHORING_KEYS = frozenset(
    {"required_oracle_tiers", "inapplicable_oracle_tiers", "semantic_defaults", "requant_output_dtype", "requant_shift"}
)


def profile_datapath(profile: dict, *, numeric_only: bool = False) -> dict:
    """Project an explicitly loaded author's profile without discovering or reading files.

    ``numeric_only`` drops corpus-generation controls, preserving the entire declared numerical
    regime (including subnormal handling and comparison tolerances). These are author-declared
    assumptions, not hardware-verified facts. Loading the public recipe and binding its provenance
    belongs to the experiment edge; private sidecars are never needed for this projection.
    """
    if not isinstance(profile, dict):
        raise TypeError("profile_datapath requires an explicitly loaded profile mapping, not a target name")
    block = dict(profile.get("datapath") or {})
    if numeric_only:
        for key in _DATAPATH_AUTHORING_KEYS:
            block.pop(key, None)
    return block


def derive_binding(
    te,
    datapath: dict,
    *,
    contract: dict | None = None,
    facts: dict | None = None,
    taxonomy: dict | None = None,
    defer_classes: bool = False,
) -> CorpusBinding:
    """Derive the per-target binding from the descriptor + the profile's ``datapath`` block (compare +
    tolerances + optional requant-output dtype — the numeric contract the manifest does not yet carry)."""
    from merlin.target.registry.experiment_descriptor import load_capability_manifest
    from merlin.target.simulators.oracle_policy import inferred_oracle_tiers

    c = load_capability_manifest(te.target).contract if contract is None else contract
    cu = (c.get("compute_units") or [{}])[0]
    # The profile may pin the DEFAULT operand/accumulate dtypes (a target with several compute units — e.g.
    # a SIMT cluster with a contained MX PE — needs the profile to say which regime a capsule set drives);
    # both fall back to the primary compute unit's declared datapath, never a target literal.
    declared_dtypes = [dt for unit in (c.get("compute_units") or []) for dt in (unit.get("dtypes") or [])]
    if not declared_dtypes:
        raise ValueError(f"{te.target}: no compute-unit operand dtypes declared; cannot derive a corpus binding")
    operand = datapath.get("operand_dtype") or (cu.get("dtypes") or [None])[0]
    if operand is None:
        raise ValueError(f"{te.target}: primary compute unit has no operand dtype; select an admitted datapath")
    from merlin.common.target_data import get as quant_format

    if quant_format(operand).name not in {quant_format(dt).name for dt in declared_dtypes}:
        raise ValueError(f"{te.target}: corpus operand dtype {operand!r} is not admitted by any compute unit")
    accum = datapath.get("accum_dtype") or _accum_dtype(c, operand, facts)
    integer = dtype_info(accum)[3]
    scaling = datapath.get("scaling") or cu.get("scaling")
    tiers = datapath.get("required_oracle_tiers") or sorted(inferred_oracle_tiers(te.target, te.sim_via))
    readout_inputs = None
    if (c.get("runner") or {}).get("backend") == "chipyard_rocc":
        from merlin.runtime.backends.rocc.chipyard_backend import readout_metadata

        readout_inputs = readout_metadata(c)
    return CorpusBinding(
        target=te.target,
        tile_dim=_tile_dim(te.target, c, operand=operand, facts=facts),
        operand_dtype=operand,
        accum_dtype=accum,
        integer=integer,
        tiers=list(tiers),
        compare=datapath.get("compare", "exact_int" if integer else "tolerance_float"),
        atol=datapath.get("atol"),
        rtol=datapath.get("rtol"),
        scaling=(scaling if scaling and scaling != "none" else None),
        scale_block=_scale_block_elems(c, numerical_semantics=datapath.get("numerical_semantics")),
        requant_output_dtype=datapath.get("requant_output_dtype"),
        requant_shift=(None if datapath.get("requant_shift") is None else int(datapath["requant_shift"])),
        subnormal_operand_flush=bool(datapath.get("subnormal_operand_flush", False)),
        inapplicable_tiers=_inapplicable_tiers(datapath, tiers),
        semantic_defaults=dict(datapath.get("semantic_defaults") or {}),
        classes_for=(lambda **kw: _classes_source(te, c, taxonomy=taxonomy)(**kw))
        if defer_classes
        else _classes_source(te, c, taxonomy=taxonomy),
        readout_inputs=readout_inputs,
    )


def regime_for_dtype(token: str) -> str:
    """The numeric regime a dtype token routes to: ``int`` | ``specir`` | ``mx`` | ``simt``.

    Routed purely by the token -- no target name anywhere in it. Lifted here from the generator so the
    SYNTHESIZER can ask the same question the WRITER will answer with: an op with no direct-MLIR builder
    can only be written by the PyTorch path, and that path is float-only, so a synthesizer that could not
    see the regime emitted entries nothing could write.
    """
    from merlin.common.fp8_formats import canonical_float

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


def shape_quantum(token: str, *, tile_dim: int | None, scale_block: int | None) -> dict:
    """The SHAPE GRANULARITY this dtype's datapath imposes: ``{"row": int, "reduction": int}``.

    A dtype is not only a value encoding; on a block-scaled datapath it is also a constraint on which
    extents can exist AT ALL. Two independent granularities, each derived, neither guessed:

    * ``reduction`` -- one E8M0 exponent covers a whole run of ``scale_block`` contraction elements, and
      that run length is the TARGET'S OWN declared scale group (``_scale_block_elems``). A K with a
      remainder emits a scale stream covering fewer elements than the operand codes do, and the shortfall
      is silent, so the golden refuses it. The reduction also walks the array edge, hence the lcm.
    * ``row`` -- sub-byte codes are stored two per byte (the packer nibble-packs fp4 AND the LUT-indexed
      fp6 alike), so the datapath addresses rows in PAIRS and its row tile is twice the array edge. At a
      smaller row extent the reference computes zero tiles and returns an all-zero result, which is a
      golden any broken kernel matches.

    Returns ``{"row": 1, "reduction": 1}`` for a dtype whose regime imposes none.
    A block-scaled dtype requires its declared numerical group; a missing group
    refuses rather than consulting an external model's default. Missing geometry
    supplies no additional tile constraint. The caller is `conformance.required_cells` (which alignment classes can exist) and
    `corpus_synth.extents_for` (what shape to give the ones that can).
    """
    free = {"row": 1, "reduction": 1}
    if regime_for_dtype(token) != "mx":
        return free
    if not scale_block or int(scale_block) <= 0:
        raise ValueError("block-scaled shape requirements need an explicit numerical block_size or manifest group")
    if not tile_dim or int(tile_dim) <= 0 or not scale_block or int(scale_block) <= 0:
        return free  # nothing derivable -> claim no constraint
    from merlin.common.fp8_formats import canonical_float, storage_bits

    try:
        bits = int(storage_bits(canonical_float(token)))
    except KeyError:
        return free
    edge = int(tile_dim)
    per_byte = 2 if bits < 8 else 1  # the packer's own nibble pairing
    return {"row": edge * per_byte, "reduction": _lcm(int(scale_block), edge)}


def _lcm(a: int, b: int) -> int:
    import math

    return abs(a * b) // math.gcd(a, b) if a and b else max(a, b, 1)


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


REQUANTIZING_STAGES: tuple[str, ...] = ("acc_scale", "requant")


BUILDER_EPILOGUE_STAGES: tuple[str, ...] = ("relu", "acc_scale", "bias_add", "maxpool", "requant")


def requant_shift_for(entry: dict, binding: CorpusBinding, epilogue: list[str], *, op: str) -> int | None:
    """The integer ``requant`` shift a capsule DECLARES, or ``None`` when it has no requant stage.

    FAIL CLOSED IN BOTH DIRECTIONS, for the same reason :func:`_pool_epilogue` does.

    * A ``requant`` stage with no declared shift raises. It would not have raised anywhere downstream:
      the golden engine, the reference and the simulator each carry their OWN fallback of 4, so the
      capsule's golden and the value the reference computes from the backend's command buffer would
      agree -- with each other, by coincidence -- while the backend under test was handed a stage whose
      one parameter nobody told it. The capsule passes, and it passes for the wrong reason.
    * A declared shift with no stage raises. A parameter read by nothing is the same silent-wrong-answer
      shape pointing the other way: the capsule looks like it requants and commits the raw accumulator.

    The value is the ENTRY's when it declares one, else the target's own ``datapath.requant_shift``. One
    declared number reaches the golden (via the capsule's operation attributes) and the reference (via
    the emitted interface, hence the backend's commit) -- never two defaults that happen to match.
    """
    declared = entry.get("requant_shift")
    staged = "requant" in epilogue
    if not staged:
        if declared is not None:
            raise ValueError(
                f"{op}: {entry.get('name', op)!r} declares requant_shift={declared!r} but its epilogue "
                f"{list(epilogue)} has no 'requant' stage. The shift would be read by nothing and the "
                f"capsule would commit the raw accumulator while looking like it requantized"
            )
        return None
    shift = declared if declared is not None else binding.requant_shift
    if shift is None:
        raise ValueError(
            f"{op}: {entry.get('name', op)!r} declares a 'requant' epilogue stage but neither the entry "
            f"nor target {binding.target!r}'s datapath declares requant_shift. The stage is NOT given a "
            f"default here: the golden, the reference and the simulator each fall back to their own "
            f"shift, so an undeclared one makes them agree with each other while the backend under test "
            f"is handed a stage with no parameter -- a pass earned by coincidence"
        )
    return int(shift)


def _readout_selectors(binding: CorpusBinding):
    """The target's declared readouts, or ``None`` when it describes none (UNKNOWN, never 'anything goes')."""
    from merlin.target.capabilities.readout_derivation import (
        ReadoutSupportError,
        epilogue_readouts,
        validated_readouts,
    )

    if binding.readout_inputs is not None:
        return validated_readouts(binding.readout_inputs.get("readouts")) or None

    try:
        return epilogue_readouts(binding.target) or None
    except ReadoutSupportError:
        raise  # selected support owes readout facts and cannot serve them: refuse, never "unknown"
    except Exception:  # noqa: BLE001 -- an unreadable declaration evidences nothing
        return None


def _readout_routes(binding: CorpusBinding):
    from merlin.target.capabilities.readout_derivation import epilogue_stage_routes, validated_stage_routes

    if binding.readout_inputs is not None:
        return validated_stage_routes(binding.readout_inputs.get("stage_routes"))
    return epilogue_stage_routes(binding.target)


def _readout_selector_for(
    binding: CorpusBinding, epilogue: list[str], *, available_operand_roles: frozenset[str] = frozenset()
) -> str | None:
    """The dtype a fused contraction must commit at, from readout and scoped stage routes.

    ``None`` when the target declares no readouts: the caller then falls back to the stage heuristics
    below, which is the pre-existing behaviour for a target that has described nothing, and the same
    UNKNOWN posture the grade-time check takes.

    RAISES when the target DOES declare readouts and no application path applies every stage. That capsule would
    declare a computation this hardware cannot perform, and writing it anyway is exactly how twelve
    such capsules reached a graded corpus and were then refused one at a time, hours into an agent run,
    by a check that had the declaration available the whole time. Generation is where this is cheap to
    find out.
    """
    from merlin.verification.references.epilogue_applicability import selectors_applying

    readouts = _readout_selectors(binding)
    if not readouts:
        return None
    routes = tuple(route for route in _readout_routes(binding) if route.operand_role in available_operand_roles)
    applying = selectors_applying(readouts, epilogue, routes=routes, composition="contraction")
    if applying:
        return applying[0]
    raise ValueError(
        f"{binding.target!r} declares no readout or contraction route that applies "
        f"the epilogue {sorted(set(epilogue))}: "
        + "; ".join(f"{r.selector!r} applies {sorted(r.applies)}" for r in readouts)
        + f". A capsule fusing {sorted(set(epilogue))} onto this target would declare a computation its "
        "hardware does not perform, so it is NOT written. Either the stage is not fusable here (drop it "
        "from the conformance spec's epilogue axis, or carry it as a host stage) or the backend's "
        "readout/route declarations under-declare what the hardware does."
    )


def _resolve_output_dtype(
    binding: CorpusBinding,
    epilogue: list[str],
    entry: dict | None = None,
    *,
    available_operand_roles: frozenset[str] = frozenset(),
) -> str:
    """Output dtype: the entry's own declaration if it makes one, else the accumulate dtype, unless a
    REQUANTIZING epilogue narrows the accumulator (the target declares that narrow dtype in its datapath
    as ``requant_output_dtype``, e.g. i8 for an integer mesh).

    BOTH requantizing stages narrow, and reading only one of them was a real gap. ``acc_scale`` (the
    float readout multiply) and ``requant`` (the integer round-half-up shift) are two spellings of the
    same handoff -- commit the wide accumulator back to the operand's narrow width -- and only
    ``acc_scale`` was consulted here. A capsule declaring the integer stage therefore committed an i32
    output: it declared the WIDE dtype for the one stage whose whole job is to narrow, so the property
    under test was absent from the capsule that was supposed to test it, and a backend that dropped the
    narrowing would have matched the declaration exactly.

    **An entry's declaration wins, and that ordering is the fix for a real defect.** This function used
    to read only the epilogue, so a profile entry that explicitly declared ``output_dtype: i8`` had the
    declaration silently discarded and got the accumulator width instead. Nothing failed at generation
    time -- a well-formed capsule came out, just not the one the entry asked for. It surfaced only
    downstream, when `2f06e353` first regenerated the three hand-authored gemmini pooling capsules from
    their profile entries and turned every one of them from ``i8`` to ``i32``: the backend refuses a
    native max-pool that is not an i8 store (pooling runs in the store DMA at the input width, not over
    the full-width accumulator container), so the capsules became uncompilable and 16 tests went red.

    Only the requantizing stages have a derivable answer here -- the target *declares* its requant
    output width.
    Every other reason a commit narrows is a property of the capsule, which is why the entry has to be
    able to say so, and why a declaration it makes must not be second-guessed. Unknown tokens raise
    (``dtype_info``) rather than falling back, so a typo is a generation failure and not another
    silently substituted default.
    """
    declared = (entry or {}).get("output_dtype")
    if (entry or {}).get("outcome") == "refuse":
        # A MUST-REFUSE capsule states, on purpose, a computation no readout performs: that is what it
        # tests (merlin.development.targetgen.grading.boundary). The readout check below, which keeps every other
        # capsule honest, is exactly the fact it relies on, so it is not applied; the committed width is
        # the entry's own, else the requantized width the stages would narrow to.
        if declared:
            dtype_info(str(declared))
            return str(declared)
        if binding.requant_output_dtype and any(st in epilogue for st in REQUANTIZING_STAGES):
            return binding.requant_output_dtype
        return binding.accum_dtype
    if declared:
        dtype_info(str(declared))  # raises on an unknown token -- fail closed, never fall back
        # AN AUTHOR MAY CHOOSE THE WIDTH; NOBODY MAY CHOOSE ONE THE HARDWARE CANNOT HONOUR. The
        # declaration still wins over the derivation below -- that ordering fixed a real defect where an
        # explicit i8 was silently replaced by the accumulator width -- but a declared width at which no
        # readout applies the declared stages is not a preference, it is an error, and it is only
        # discoverable here. MEASURED: two model-derived capsules carried `bias_add` at i32 from their
        # capture and reached a graded corpus, where the grade-time readout check refused them.
        if epilogue:
            _selectors = _readout_selectors(binding)
            if _selectors is not None:
                from merlin.verification.references.epilogue_applicability import selectors_applying

                routes = tuple(
                    route for route in _readout_routes(binding) if route.operand_role in available_operand_roles
                )
                _ok = selectors_applying(
                    _selectors,
                    epilogue,
                    routes=routes,
                    composition="contraction",
                )
                if str(declared) not in _ok:
                    raise ValueError(
                        f"{(entry or {}).get('name', '?')!r} declares output_dtype {str(declared)!r} with "
                        f"epilogue {sorted(set(epilogue))}, but on {binding.target!r} no readout applies "
                        f"those stages at that width (readouts applying them: {list(_ok) or 'none'}). The "
                        f"capsule would declare a computation this hardware does not perform."
                    )
        return str(declared)
    # DERIVED, not guessed: the target declares which readout applies which stages, and the selector IS
    # the committed dtype. The stage heuristics below predate that declaration and only ever covered the
    # stages someone had thought of -- `relu` was in none of them, so a capsule fusing an activation
    # committed the raw accumulator width and declared a program the hardware cannot run.
    if epilogue:
        selector = _readout_selector_for(binding, epilogue, available_operand_roles=available_operand_roles)
        if selector is not None:
            dtype_info(selector)  # same fail-closed check an entry's own declaration gets
            return selector
    if binding.requant_output_dtype and any(st in epilogue for st in REQUANTIZING_STAGES):
        return binding.requant_output_dtype
    if "maxpool" in epilogue:
        # A fused max-pool is not an accumulator-side stage: it runs in the STORE DMA, which reads the
        # target's own operand width (the RTL's `inputType`), not the full-width accumulator container.
        # So the committed dtype is the operand dtype, derived from the descriptor rather than named
        # here. This is what the three hand-authored pooling entries were saying with their explicit
        # `output_dtype: i8`, and deriving it means the SYNTHESIZED pooling capsule -- which declares no
        # dtype and cannot, since the axis that builds it knows nothing about a store path -- gets the
        # same answer instead of an accumulator-width commit the backend refuses to compile.
        return binding.operand_dtype
    return binding.accum_dtype


def _default_modes(binding: CorpusBinding, output_dtype: str, epilogue: list[str]) -> dict:
    """The mode flags a capsule gets when its profile entry declares none: relu/acc_scale from the
    epilogue (+ the integer-narrow-output flag for an integer target). An entry MAY instead declare
    ``modes`` verbatim (e.g. ``{k_accumulate: true}``), which is used as-is."""
    modes = {"relu": "relu" in epilogue, "acc_scale": "acc_scale" in epilogue}
    if binding.integer:
        modes["i8"] = binding.cap_dtype(output_dtype) == "i8"
    return modes


def _pool_epilogue(entry: dict, *, rows: int, in_dims: tuple[int, int] | None, op: str) -> tuple[dict[str, Any], int]:
    """``(pool-geometry attributes, the row count the op commits)`` for a ``maxpool`` epilogue.

    Pooling on this class of target is FUSED onto the store path -- the store config carries
    ``pool_size``/``pool_stride``/``pool_out_dim``/``porows``/``pocols``/``orows``/``ocols``/``upad``/
    ``lpad`` and the mvout writes the pooled window. Geometry therefore rides on the readout's
    attributes. A standalone source pool can use movement plus this readout when the capability
    oracle admits that composition; it does not require an independent pooling engine.

    ``in_dims`` is the spatial extent the committed ROWS unflatten to. A conv2d DERIVES it (it is the
    conv's own ``Ho x Wo``); a contraction has no spatial extent of its own, so its entry must DECLARE
    ``pool_in_dims`` -- 25 rows is 5x5 or 25x1 and nothing else in the capsule says which. There is no
    default for the window either: a pool size or stride this generator picked would be a target fact
    invented in library code, and every engine downstream would then be grading a window nobody chose.

    Declaring a ``pool_*`` knob WITHOUT the stage also raises. A parameter that is read by nothing is
    the same silent-wrong-answer shape as a stage applied by nobody, just pointing the other way.
    """
    from merlin.runtime.engine.command_buffer import pool_out_dims

    stages = [str(x) for x in (entry.get("epilogue") or [])]
    declared = sorted(k for k in entry if k.startswith("pool_"))
    if "maxpool" not in stages:
        if declared:
            raise ValueError(
                f"{entry.get('name', op)}: declares {declared} but no 'maxpool' epilogue stage; the "
                f"pool geometry would be carried into the capsule and read by nothing"
            )
        return {}, rows
    if in_dims is None:
        want = entry.get("pool_in_dims")
        if not isinstance(want, (list, tuple)) or len(want) != 2:
            raise ValueError(
                f"{entry.get('name', op)}: a 'maxpool' epilogue on {op} needs pool_in_dims = [H, W] "
                f"(the extent the {rows} committed rows unflatten to); got {want!r}"
            )
        in_dims = (int(want[0]), int(want[1]))
    elif entry.get("pool_in_dims") is not None:
        raise ValueError(
            f"{entry.get('name', op)}: pool_in_dims is DERIVED for {op} (it is the op's own output "
            f"extent {list(in_dims)}); declaring it invites the two to disagree"
        )
    size, stride = entry.get("pool_size"), entry.get("pool_stride")
    for key, val in (("pool_size", size), ("pool_stride", stride)):
        if not isinstance(val, (list, tuple)) or len(val) != 2:
            raise ValueError(
                f"{entry.get('name', op)}: a 'maxpool' epilogue needs {key} = [h, w]; got {val!r}. "
                f"There is no default -- the window is a property of the workload, not of this builder"
            )
    padding = list(entry.get("pool_padding", [0, 0, 0, 0]))
    if len(padding) != 4:
        raise ValueError(f"{entry.get('name', op)}: pool_padding must be [top, left, bottom, right]; got {padding!r}")
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
            f"{in_dims[0]}x{in_dims[1]} planes, so pool_in_dims does not describe this operation"
        )
    Ho, Wo = pool_out_dims(in_dims[0], in_dims[1], attrs["pool_size"], attrs["pool_stride"], attrs["pool_padding"])
    if Ho < 1 or Wo < 1:
        raise ValueError(
            f"{entry.get('name', op)}: window {attrs['pool_size']} stride {attrs['pool_stride']} "
            f"padding {attrs['pool_padding']} leaves no output position over "
            f"{in_dims[0]}x{in_dims[1]}"
        )
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


def _matmul_inputs(lhs: str, weight: str, M: int, K: int, N: int, idt: str, binding: CorpusBinding) -> list[dict]:
    """The declared leaf operands of a matmul capsule.

    On a BLOCK-SCALED datapath the operand format is (elements, per-block scales); declaring only the
    elements hands a backend half a number and asks it for the product. The scale streams are therefore
    declared operands in their own right, shaped by the target's own derived scale group -- one shared
    exponent per run of ``scale_block`` K elements, per lhs row and per weight column.

    Declared only when the target's manifest yields BOTH a block-scaled compute unit and a scale group,
    and the contraction depth divides into whole blocks. Otherwise the operand list is exactly what it
    has always been, so an unscaled target's capsules are unchanged."""
    inputs = [
        {"name": weight, "role": "weight", "shape": [K, N], "dtype": idt},
        {"name": lhs, "role": "input", "shape": [M, K], "dtype": idt},
    ]
    blk = binding.scale_block if is_block_scaled(binding.operand_dtype) else None
    if blk and K % blk == 0:
        g = K // blk
        inputs += [
            {"name": f"{lhs}_scale", "role": "scale", "scale_of": lhs, "block": blk, "shape": [g, M], "dtype": "e8m0"},
            {
                "name": f"{weight}_scale",
                "role": "scale",
                "scale_of": weight,
                "block": blk,
                "shape": [g, N],
                "dtype": "e8m0",
            },
        ]
    return inputs


_BIAS_STAGES = ("bias_add", "bias")


def _bias_operand(entry: dict, epilogue: list[str], N: int, binding: CorpusBinding, op: str) -> tuple[str | None, str]:
    """``(bias tensor name, its capsule dtype)`` for a capsule whose epilogue adds a bias, else
    ``(None, "")``.

    The dtype is the ACCUMULATOR's, not the operand's, because that is the domain the addition happens
    in: the bias lands on the accumulator before any requant, which is exactly what
    ``Tensor.add_bias`` does (it returns ``self.dtype``). Declaring it in the operand dtype would
    describe a different computation from the one the golden performs -- on an int8/i32 datapath, a
    visibly different one.

    ``fused_matmul_bias`` implies the stage: the op name IS the declaration that a bias is added, so a
    capsule need not repeat it in ``epilogue`` to get its operand.
    """
    fused_by_name = op == "fused_matmul_bias"
    if not fused_by_name and not any(s in _BIAS_STAGES for s in epilogue):
        return None, ""
    return entry.get("bias", "B"), binding.cap_dtype(binding.accum_dtype)
