"""Explicit corpus role selections bound to a neutral command-class roster.

These are authored corpus requirements, not inferred hardware capabilities or
instruction schedules. The ordinary numerical and execution gates remain separate.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from merlin.target.capabilities import semantic_families as sf
from merlin.target.isa.taxonomy import required_classes_from_roles, required_role_slots

# ---- corpus_instruction_roles ------------------------------------------------------------------

_ROLES = frozenset(
    {
        "memory",
        "weight_load",
        "matmul",
        "acc_readout",
        "acc_readout_scaled",
        "tensor_compute_unary",
        "tensor_compute_binary",
    }
)


def selected_classes(contract: dict):
    """Return an explicit role selector, or None when none was selected."""
    if "corpus_instruction_roles" not in contract:
        return None
    selection = contract["corpus_instruction_roles"]
    if (
        not isinstance(selection, dict)
        or set(selection)
        not in (
            {"version", "instruction_roster_sha256", "by_role"},
            {"version", "instruction_roster_sha256", "by_role", "epilogue_roles"},
            {"version", "instruction_roster_sha256", "by_role", "operation_roles"},
            {"version", "instruction_roster_sha256", "by_role", "epilogue_roles", "operation_roles"},
        )
        or type(selection["version"]) is not int
        or selection["version"] not in (1, 2)
    ):
        raise ValueError("invalid corpus_instruction_roles selection")
    if "operation_roles" in selection and selection["version"] != 2:
        raise ValueError("operation roles require corpus selection version 2")
    roster = (contract.get("rocc_operand_roles") or {}).get("instructions")
    if not isinstance(roster, list) or not roster:
        raise ValueError("corpus instruction roles require an explicit command roster")
    names = []
    for row in roster:
        if not isinstance(row, dict) or not isinstance(row.get("class"), str) or not row["class"]:
            raise ValueError("invalid command class in corpus role roster")
        names.append(row["class"])
    if len(names) != len(set(names)):
        raise ValueError("duplicate command class in corpus role roster")
    digest = hashlib.sha256(
        json.dumps(roster, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    if selection["instruction_roster_sha256"] != digest:
        raise ValueError("corpus instruction roles disagree with selected command roster")
    roles = selection["by_role"]
    if not isinstance(roles, dict) or not roles or set(roles) - _ROLES:
        raise ValueError("invalid corpus instruction role vocabulary")
    epilogues = selection.get("epilogue_roles", {})
    if not isinstance(epilogues, dict) or any(
        not isinstance(stage, str) or not stage or not isinstance(mapping, dict) or not mapping or set(mapping) - _ROLES
        for stage, mapping in epilogues.items()
    ):
        raise ValueError("invalid corpus epilogue role selection")
    operations = selection.get("operation_roles", {})
    from merlin.target.capabilities.semantic_families import from_op

    if not isinstance(operations, dict) or any(
        not isinstance(op, str)
        or not from_op(op)
        or not isinstance(mapping, dict)
        or not mapping
        or set(mapping) - _ROLES
        for op, mapping in operations.items()
    ):
        raise ValueError("invalid corpus operation role selection")
    class_groups = [
        *(classes for mapping in operations.values() for classes in mapping.values()),
        *roles.values(),
        *(classes for mapping in epilogues.values() for classes in mapping.values()),
    ]
    for classes in class_groups:
        if (
            not isinstance(classes, list)
            or not classes
            or any(not isinstance(name, str) or name not in names for name in classes)
            or len(classes) != len(set(classes))
        ):
            raise ValueError("corpus instruction role names an undeclared or duplicate class")
    # Snapshot caller-owned lists: a later mutation cannot change the selection.
    selected = {role: list(classes) for role, classes in roles.items()}
    selected_epilogues = {
        stage: {role: list(classes) for role, classes in mapping.items()} for stage, mapping in epilogues.items()
    }

    selected_operations = {
        op: {role: list(classes) for role, classes in mapping.items()} for op, mapping in operations.items()
    }

    def classes_for(**kwargs):
        from merlin.target.capabilities.semantic_families import from_op, primitives_of

        bound = {role: list(classes) for role, classes in selected.items()}
        for role, classes in selected_operations.get(kwargs.get("op", "matmul"), {}).items():
            if role in bound and bound[role] != classes:
                raise ValueError("conflicting corpus operation role selection")
            bound[role] = list(classes)
        # Fused stage requirements do not declare a general unary/binary unit.
        if "contraction" in primitives_of(from_op(kwargs.get("op", "matmul")) or ""):
            for stage in kwargs.get("epilogue", ()):
                for role, classes in selected_epilogues.get(stage, {}).items():
                    if role in bound and bound[role] != classes:
                        raise ValueError("conflicting corpus epilogue role selection")
                    bound[role] = list(classes)
        for slot in required_role_slots(**kwargs):
            if not any(role in bound for role in slot):
                raise ValueError(f"selected corpus roles lack required role slot {slot}")
        classes = required_classes_from_roles(bound, **kwargs)
        # Every explicitly selected operation class remains an obligation; the
        # generic role resolver chooses only the first class for each slot.
        for names in selected_operations.get(kwargs.get("op", "matmul"), {}).values():
            for name in names:
                if name not in classes:
                    classes.append(name)
        if not classes:
            raise ValueError("selected corpus roles yield no required instruction classes")
        return classes

    return classes_for


def classes_for_checked_source(binding, *, op: str, output_dtype: str):
    """Resolve roles after the caller proves typed source and eligibility.

    Capture's integer contraction spelling may have no role-vocabulary entry.
    Preserve any original provider choice first, then ask that same provider
    for the canonical contraction spelling. This does not establish eligibility.
    """
    for spelling in (op, "matmul") if op == "int_matmul" else (op,):
        try:
            classes = binding.classes_for(op=spelling, output_dtype=output_dtype)
        except Exception:  # noqa: BLE001 — underivable roles create no device demand
            continue
        if classes:
            return classes
    return ()


# ---- corpus_effect_classes ---------------------------------------------------------------------
# Project reviewed instruction effect classes into operation-presence obligations.
#
# Equivalent selectors may share a class. This projection assigns neither an issue
# order nor a preferred selector, and does not qualify arithmetic or execution.


def source(encoding: dict):
    """Return a pure operation-to-class projection of an explicit declaration."""
    classes, families = encoding.get("corpus_required_classes"), encoding.get("class_family")
    declarations = [encoding.get(key, {}) for key in ("semantic_class", "config_subtype")]
    if (
        type(classes) is not list
        or not classes
        or any(type(label) is not str or not label.isidentifier() for label in classes)
        or len(set(classes)) != len(classes)
    ):
        raise ValueError("encoding.corpus_required_classes must declare unique effect classes")
    if (
        type(families) is not dict
        or set(families) != set(classes)
        or any(type(family) is not str or family not in sf.PRIMITIVES for family in families.values())
    ):
        raise ValueError("encoding.class_family must bind every required class to a primitive family")
    if any(
        type(mapping) is not dict or any(type(label) is not str for label in mapping.values())
        for mapping in declarations
    ):
        raise ValueError("required effect classes need explicit instruction-class declarations")
    declared = {label for mapping in declarations for label in mapping.values()}
    if set(classes) - declared:
        raise ValueError("required effect class has no declared instruction selector")

    def classes_for(*, op="matmul", output_dtype=None, epilogue=(), movement=False):
        primitives = ("movement",) if movement else sf.primitives_of(sf.from_op(op) or "")
        required = [label for label in classes if families[label] in primitives]
        if not required:
            raise ValueError(f"no required effect class is declared for operation {op!r}")
        return required

    return classes_for


# ---- corpus_shapes -----------------------------------------------------------------------------
# Derive capsule extents and certification budgets from selected contracts.


class SynthesisError(ValueError):
    """A requirement that cannot be turned into a capsule, named so it can be fixed."""


def _tile_int(token, tile: int):
    """A tile-relative extent as an integer, or ``None`` when it cannot be resolved here."""
    if isinstance(token, int):
        return token
    text = str(token or "").strip()
    if not text:
        return None
    if text == "tile":
        return int(tile)
    if text.startswith("tile/"):
        try:
            divisor = int(text.partition("/")[2])
            return tile // divisor if divisor > 0 and tile % divisor == 0 else None
        except ValueError:
            return None
    mult, sep, rest = text.partition("*tile")
    if sep and not rest:
        try:
            return int(mult) * int(tile)
        except ValueError:
            return None
    base, sep, delta = text.partition("-")
    if sep and base.strip() == "tile":
        try:
            return int(tile) - int(delta)
        except ValueError:
            return None
    try:
        return int(text)
    except ValueError:
        return None


def cap_to_affordable(entry: dict, spec_doc: dict, *, extends: str = "") -> str | None:
    """Cap ``entry`` at the loop tier when its SIZE cannot be certified. Returns the reason, or None.

    One rule for every axis, applied to the shape rather than to the axis that produced it. The tier a
    capsule may demand is a fact about how long it takes to simulate, and until now it was a property
    of which code path emitted it -- so the accumulation-depth axis capped its deep capsules while the
    memory-regime axis, emitting shapes of the same order, demanded the cycle-accurate tier from all
    of them.

    Fails OPEN, deliberately: a target with no measured history, or an extent this function cannot
    resolve, leaves the tier untouched. Capping on a cost nobody measured would shrink the certified
    corpus on a guess, which is worse than an expensive capsule somebody can see in the bill.
    """
    # The ceiling comes from `conformance._cert_affordability`, which prices PER (target, engine) via
    # `cert_affordability` and returns the slowest engine's figure -- the one that binds, because the
    # cap has to be payable on whichever engine the run picks. It is read here, never recomputed: a
    # second inversion of the cost law is a second place for the extrapolation clamp to be forgotten,
    # which is how the producer's own inline `(budget - intercept) / per_element` came to bypass it.
    aff = spec_doc.get("cert_affordability") or {}
    ceiling = aff.get("max_elements")
    if not ceiling:
        return None
    tile = ((spec_doc.get("boundaries") or {}).get("tile_edge")) or 0
    if not tile:
        return None
    dims = {k: _tile_int(entry.get(k), tile) for k in ("M", "K", "N")}
    # WRITTEN OUTPUT, which is what the cost law was calibrated on -- see `_cert_affordability`. The
    # reduction depth is deliberately absent from this product: a capsule drains `M x N` elements
    # whatever `K` is, so a deep accumulation is cheap to certify and must not be capped as though it
    # were large. Pricing by operand size instead (the first attempt) capped exactly the capsules most
    # worth certifying.
    if entry.get("op") == "conv2d" and entry.get("Himg") is not None and entry.get("Wimg") is not None:
        from merlin.runtime.engine.command_buffer import conv_out_dims

        if dims["N"] is None:
            return None
        oh, ow = conv_out_dims(
            int(entry["Himg"]),
            int(entry["Wimg"]),
            int(entry["kh"]),
            int(entry["kw"]),
            entry["stride"],
            entry["padding"],
            entry["dilation"],
        )
        elements = oh * ow * dims["N"]
    else:
        if any(v is None for v in dims.values()):
            return None
        elements = dims["M"] * dims["N"]
    if elements <= int(ceiling):
        return None
    # Constructed tiers are authoritative. During deterministic Phase 0
    # derivation no oracle is constructed yet, but the selected recipe still
    # declares which tiers MAY be used. That declaration can size a candidate;
    # Phase 1 must establish actual availability before treating a pass as
    # verified. Never invent a cheaper tier on a target declaring only one.
    observed = list(spec_doc.get("oracle_tiers") or ())
    declared = list(spec_doc.get("oracle_tiers_declared") or ())
    source = "constructed" if observed else "declared"
    tiers = [str(t) for t in (observed or declared) if str(t) not in ("L0", "L1")]
    loop_tier = tiers[0] if len(tiers) > 1 else None
    if loop_tier is None:
        return (
            f"{elements} written output elements exceeds the {int(ceiling)} a "
            f"{aff.get('budget_s')}s certification budget affords, and this target declares no "
            f"tier cheaper than its cert tier ({tiers or 'none resolved'}) to fall back to, so the "
            f"capsule is emitted UNCAPPED and is expected to be expensive"
        )
    entry["max_oracle_tier"] = loop_tier
    if extends:
        entry["extends"] = extends
    # NAME THE ENGINE THE CEILING BELONGS TO. A certification second is a property of the simulator,
    # not of the capsule: the same ELF answered on two elaborated-RTL engines differs by more than an
    # order of magnitude, and this reason string is written into a TRACKED capsule where it long
    # outlives the run that produced it. Without the engine, a reader comparing two capsules capped
    # years apart cannot tell a corpus that got more expensive from a corpus priced on a faster
    # machine -- which is exactly what the mixed fit this ceiling used to come from was hiding.
    on = f" on {aff['engine']}" if aff.get("engine") else ""
    tier_note = " (declared in the selected recipe; execution must verify availability)" if source == "declared" else ""
    return (
        f"{elements} written output elements exceeds the {int(ceiling)} a {aff.get('budget_s')}s "
        f"certification budget affords{on} on this target, so it is graded at {loop_tier}{tier_note} and "
        f"rests on {extends!r}"
        if extends
        else f"{elements} written output elements exceeds the {int(ceiling)} a {aff.get('budget_s')}s "
        f"certification budget affords{on} on this target, so it is graded at {loop_tier}{tier_note}"
    )


def narrowest_admitted(admitted: set[str] | frozenset[str]) -> str:
    """The narrowest admitted dtype -- the fallback when no ``precision_preference`` is declared.

    ⚠️ THIS USED TO BE ``sorted(admitted)[0]``, i.e. ALPHABETICAL. On the microscaling target, whose
    whole purpose is mxfp4/mxfp6/mxfp8, the admitted set is {bf16, i8, mxfp4, mxfp6, mxfp8} and the
    alphabetical answer is `bf16` -- the WIDEST of the five, picked by spelling. The stated goal is to
    exercise the best quantization format the hardware supports rather than a wide one, so the
    alphabet was choosing the opposite of the intent, silently, on the three targets that declare no
    preference.

    A declared preference is still authorial and still wins: which formats a target's owner wants
    ranked, and in what order, is an intent nobody can derive. But "narrowest first" is not an
    invention -- storage width is a derived fact, from `quant_formats` via `capsule_dram.dtype_bits`,
    which already knows the sub-byte packed widths (mxfp4 -> 4 bits, mxfp6 -> 6). Ties break
    alphabetically so the choice stays deterministic and byte-stable regeneration is preserved.

    An unmeasurable token sorts LAST rather than raising: it should not be picked over a format whose
    width is known, and it should not stop a corpus from generating either.
    """
    from merlin.development.targetgen.grading.golden_inputs import dtype_bits

    if not admitted:
        return ""

    def _rank(token: str) -> tuple:
        try:
            return (0, int(dtype_bits(str(token))), str(token))
        except Exception:  # noqa: BLE001 -- unknown width sorts last, not raises
            return (1, 0, str(token))

    return sorted((str(d) for d in admitted), key=_rank)[0]


def quanta_by_dtype(spec_doc: dict) -> dict[str, dict]:
    """``{dtype: {"row": int, "reduction": int}}`` read off the spec's own cell rows.

    The granularity a dtype's datapath imposes is derived where the manifest is readable
    (`conformance.spec`) and travels on the cell; this module is pure and reads it back as data. A spec
    written before the field existed yields ``{}``, and every extent then keeps the tile-relative
    spelling it had -- the pre-existing behaviour, not a guessed granularity.
    """
    out: dict[str, dict] = {}
    for row in spec_doc.get("cells") or ():
        q = (row or {}).get("shape_quantum")
        dt = str((row or {}).get("dtype") or "")
        if dt and isinstance(q, dict):
            out[dt] = q
    return out


def _quantize(mult: int, quantum: int, tile: int) -> str:
    """Spell ``mult * tile`` rounded UP to a whole multiple of ``quantum``, still tile-relative.

    ``quantum`` is always a whole multiple of the tile edge where it constrains anything (a row quantum
    is the edge times the codes packed per storage byte; a reduction quantum is the lcm of the target's
    declared scale group and the edge), so the result stays spellable in tile units and the SAME entry
    still describes the same shape on a target with a different edge.
    """
    if not quantum or not tile or quantum <= tile:
        return "tile" if mult == 1 else f"{mult}*tile"
    step = -(-int(quantum) // int(tile))  # the quantum, in tile units (ceil for safety)
    m = -(-int(mult) // step) * step  # round the multiplier up to a whole step
    return "tile" if m == 1 else f"{m}*tile"


def extents_for(alignment: str, probes: list[dict], quantum: dict | None = None) -> dict[str, str]:
    """Tile-relative extents for an alignment, spelled so the entry stays geometry-free.

    ``aligned`` sits on the boundary; ``partial`` rags by one element.

    IT RAGS K AS WELL AS N, and the reason is measured. Ragging only the output column was enough for a
    contraction and reached NOTHING ELSE: a unary op's operand is ``M x K``, so `SY_elementwise_map_
    f32_partial` came out `tensor<16x32xf32>` -- exactly the aligned capsule under a partial name. Nine
    cells across elementwise_map, normalization, reduction and softmax were required, emitted, built,
    and classified `aligned` by the cover, so the class was uncoverable for every family that does not
    contract. K is the one axis every family's primary operand carries.

    Spelled ``tile`` / ``tile-1`` rather than resolved integers so the SAME entry describes the same
    shape on a target with a different edge; ``generate_corpus.resolve_extent`` parses the tokens
    structurally.
    """
    if not probes:
        raise SynthesisError(
            "no extent probe: this target's boundaries carry no tile edge, so an alignment-indexed "
            "requirement cannot be turned into a shape. Fix the boundary derivation rather than "
            "guessing an edge"
        )
    # The TILE-EDGE probe by name. `probes` also carries the block-scale-group boundary (edge 32 where
    # the tile edge is 16), so taking the first probe -- or the widest -- would spell every extent
    # against the wrong edge.
    tile = next((int(pr.get("edge") or 0) for pr in probes if str((pr or {}).get("boundary") or "") == "tile_edge"), 0)
    row_q = int((quantum or {}).get("row") or 1)
    red_q = int((quantum or {}).get("reduction") or 1)
    if alignment == "partial":
        return {"M": "tile", "K": "2*tile-1", "N": "tile-1"}
    if alignment == "sub_tile":
        # BARELY OCCUPIED, which `partial` (ragged by one) cannot express. Spelled tile-relative like
        # every other extent, so a target with a 64-wide edge gets 16/32/16 from the same entry. K
        # stays at tile//2 rather than dropping to tile//4 so the capsule still asks for a real
        # reduction: a single-pass contraction would exercise accumulation not at all, and every other
        # capsule already carries one.
        # THE SHAPE IS CHOSEN AGAINST THE STIMULUS, not only against the tile. The deterministic
        # operand generator is seeded by name and shape, and small square extents collide: measured,
        # `tile/4` square gave `SY_movement_i8_sub_tile`'s X only 3 distinct rows of an achievable 4,
        # so a row-addressing bug was invisible in it and the capsule could pass without the kernel
        # reading its operand correctly -- the exact degeneracy `test_no_capsule_operand_of_the_
        # graded_corpus_is_degenerate` exists to catch. Of the tile-relative spellings, this is the
        # one where A0, W and X are all non-degenerate while every extent stays well inside the
        # class (2 and 4 of a 16-wide edge, against the `tile/2` boundary).
        return {"M": "tile/8", "K": "tile/4", "N": "tile/4"}
    # THE DTYPE'S OWN GRANULARITY, not only the tile edge. A block-scaled datapath addresses rows in
    # nibble pairs and scales K one whole group at a time; at the tile edge alone its reference computes
    # zero tiles and returns an all-zero golden -- a golden any broken kernel matches, and the
    # `UnfalsifiablePolicy` refusal that made 4 of these capsules fail to generate. Rounding UP keeps
    # every non-quantized dtype at exactly the extents it had.
    return {"M": _quantize(1, row_q, tile), "K": _quantize(2, red_q, tile), "N": _quantize(1, row_q, tile)}


# ---- capsule_writer_selection ------------------------------------------------------------------
# Select a capsule writer without broadening its numeric regime.

_PYTORCH_REGIMES = ("simt",)


def select_writer(entry: dict, *, source_for_op) -> str | None:
    """Which writer can materialize this entry -- ``builder`` / ``pytorch`` / ``None`` for neither.

    ``available_ops`` is the union of the direct-MLIR BUILDERS and the PyTorch bodies, so an op can be
    "materializable" in the abstract and still have no writer AT THIS DTYPE: the PyTorch path is float
    only. A cell in that corner is a real capability gap and must be reported as an uncovered cell to
    argue about, never emitted as an entry nothing can write.
    """
    from merlin.development.targetgen.corpus.binding import regime_for_dtype
    from merlin.development.targetgen.corpus.declarations import native_storage_source_spec
    from merlin.development.targetgen.corpus.model_builders import BUILDERS

    op = entry.get("op")
    if native_storage_source_spec(entry) is not None:
        return "pytorch" if source_for_op(op) == "pytorch" else None
    if op == "dtype_cast":
        return None
    if op in BUILDERS:
        return "builder"
    try:
        regime = regime_for_dtype(str(entry.get("operand_dtype") or ""))
    except Exception:  # noqa: BLE001 -- an unknown regime is not a float one
        regime = None
    return "pytorch" if regime in _PYTORCH_REGIMES else None


def mark_source(entry: dict) -> None:
    """Say which writer can materialize this entry's op, when it is not the direct-MLIR one.

    `available_ops` is the union of the direct-MLIR BUILDERS and the PyTorch bodies, so an op can be
    materializable and still have no builder -- and an entry that does not say so goes down the
    direct-MLIR path and dies with "no corpus builder for op". Measured: four synthesized capsules
    failed exactly that way on the first target whose elementwise family resolved to `gelu`.
    """
    from merlin.development.targetgen.corpus.model_builders import BUILDERS

    op = entry.get("op")
    if op and op not in BUILDERS:
        entry["source"] = "pytorch"


# ---- sign_sensitivity --------------------------------------------------------------------------
# Which capsule entries test something that behaves differently below zero.
#
# One rule, read by the synthesizer (which gives such members a signed stimulus) and by the gate that
# holds the corpus to it, so the two cannot disagree about what needs one: an entry is sign-sensitive
# when it carries a readout stage, on the operation or on any of its per-command entries, or IS such
# a stage standing alone.


def is_sign_sensitive(entry: Mapping[str, Any]) -> bool:
    """``entry`` is a profile entry or an operation's ``{"op": ..., **attributes}``."""
    from merlin.runtime.engine.command_buffer import EPILOGUE_STAGE_SET

    if str(entry.get("op") or "") in EPILOGUE_STAGE_SET:
        return True
    commands = [entry, *(c for c in entry.get("matmuls") or () if isinstance(c, Mapping))]
    return any(str(stage) in EPILOGUE_STAGE_SET for command in commands for stage in command.get("epilogue") or ())


def has_signed_integer_operands(entry: Mapping[str, Any]) -> bool:
    """``entry`` feeds a signed, element-scaled integer format to the unit.

    Operand sign extension is itself sign-sensitive: a load that zero-extends where it should
    sign-extend reads ``-4`` as ``252``, and on the non-negative default stimulus the two agree on
    every element. Kept separate from :func:`is_sign_sensitive`, which is about readout stages and is
    the rule the corpus sign-coverage gate holds historical capsules to.
    """
    from merlin.common import target_data as quant_formats

    token = entry.get("operand_dtype")
    if not isinstance(token, str) or not token:
        return False
    try:
        fmt = quant_formats.get(token)
    except KeyError:
        return False
    return bool(fmt.signed) and not fmt.is_float and not fmt.is_block_scaled
