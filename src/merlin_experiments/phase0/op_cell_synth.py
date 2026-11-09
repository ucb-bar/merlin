"""Propose small form witnesses through the actual capsule builder.

Explicit inputs only. Generated entries are unqualified public experiment recipes;
normal Phase 0 generation owns materialization, answer isolation and run identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from merlin.targetgen import op_form_regimes as FR
from merlin.targetgen import opset_contract as OC
from merlin.targetgen import semantic_families as sf

from .sweeps import _resolve_flat_extents

PREFIX = "OC"

SOURCE_ROLE = "derived_sweep"

_AXIS_ORDER: tuple[str, ...] = (
    "occupancy",
    "aspect",
    "reduction_depth",
    "window_kernel",
    "window_stride",
    "window_dilation",
    "window_padding",
    "window_channels",
    "transfer",
    "accumulator_capacity",
    "operand_capacity",
)


@dataclass
class SynthReport:
    entries: list[dict] = field(default_factory=list)
    covered: list[str] = field(default_factory=list)
    unwritable: dict[str, str] = field(default_factory=dict)
    unplaceable: dict[str, str] = field(default_factory=dict)
    #: Cells whose entry the real builder refused, with its own message. A capability finding: the op
    #: exists, the shape places, and the builder cannot express that combination today.
    unbuildable: dict[str, str] = field(default_factory=dict)
    still_uncovered: list[str] = field(default_factory=list)


def _operand_shapes(family: str, shape: dict) -> list[dict]:
    """The operands a capsule of ``family`` declares, at ``shape``.

    Derived from the primitive rather than from the op name: a reduce-over-k carries ``[M, K]`` against
    ``[K, N]``, and everything else carries two same-shaped ``[M, N]`` tensors. That is the structure the
    family word already names, so a new op inherits the right operands without an entry here.
    """
    m, k, n = int(shape["m"]), int(shape["k"]), int(shape["n"])
    if "contraction" in sf.primitives_of(family):
        return [
            {"name": "A0", "role": "input", "shape": [m, k], "dtype": ""},
            {"name": "W", "role": "weight", "shape": [k, n], "dtype": ""},
        ]
    return [
        {"name": "X0", "role": "input", "shape": [m, n], "dtype": ""},
        {"name": "X1", "role": "weight", "shape": [m, n], "dtype": ""},
    ]


def _probe_capsule(op: str, family: str, shape: dict, *, dtype: str, epilogue=(), window=None) -> dict:
    """A capsule-shaped dict good enough for :func:`classify_capsule` to place. Never written to disk."""
    inputs = _operand_shapes(family, shape)
    for tensor in inputs:
        tensor["dtype"] = dtype
    attrs: dict = {"lhs": inputs[0]["name"], "weight": inputs[1]["name"], "out": "Y0", "dtype": dtype}
    if epilogue:
        attrs["epilogue"] = list(epilogue)
    if window:
        attrs.update(window)
    return {"inputs": inputs, "operation": {"op": op, "attributes": attrs}}


def _tile_token(value: int, tile: int) -> int | str:
    """``value`` spelled tile-relative where it divides, so the entry survives a different array edge.

    ``generate_corpus.resolve_extent`` parses ``[<mult>*]tile[+n|-n|/div]`` structurally. A value that
    is not expressible in that grammar is emitted as the integer it is rather than forced into a
    spelling that would mean something else on another target.
    """
    if not tile or value < 1:
        return int(value)
    if value == tile:
        return "tile"
    if value % tile == 0:
        return f"{value // tile}*tile"
    if value < tile and tile % value == 0:
        return f"tile/{tile // value}"
    whole = (value // tile) * tile
    delta = value - whole
    if whole and delta:
        head = "tile" if whole == tile else f"{whole // tile}*tile"
        return f"{head}+{delta}"
    return int(value)


def _entry_for(
    op: str,
    family: str,
    shape: dict,
    *,
    dtype: str,
    tile: int,
    axis: str,
    regime: str,
    epilogue=(),
    window=None,
) -> dict:
    """One profile entry, in the shape ``generate_corpus`` consumes."""
    contracting = "contraction" in sf.primitives_of(family)
    name = f"{PREFIX}_{op}_{axis}_{regime}".replace("-", "_")
    entry: dict = {
        "cat": "layers" if window else "isa",
        "kind": "layer" if window else "isa",
        "name": name,
        "op": op,
        "operand_dtype": dtype,
        "out": "Y0",
        "label": "public",
        "source_role": SOURCE_ROLE,
        "source_reference": (
            f"synthesized for op-coverage cell {op}/{axis}={regime}: a small candidate shape that enters "
            f"this regime on this target's own bounds (merlin.targetgen.op_form_regimes). Regime "
            f"boundaries are derived, not chosen -- see the entry's own extents against the array edge, "
            f"operand/accumulator store rows and transfer bound"
        ),
        "generalization": {"generalization_axis": "shape"},
    }
    if window:
        entry.update({"ifm": "IFM", "weight": "W"})
        entry.update(window)
        entry["N"] = _tile_token(int(shape["n"]), tile)
    elif contracting:
        entry.update({"lhs": "A0", "weight": "W"})
        entry["M"] = _tile_token(int(shape["m"]), tile)
        entry["K"] = _tile_token(int(shape["k"]), tile)
        entry["N"] = _tile_token(int(shape["n"]), tile)
    else:
        entry.update({"lhs": "X0", "rhs": "X1"})
        entry["M"] = _tile_token(int(shape["m"]), tile)
        entry["N"] = _tile_token(int(shape["n"]), tile)
    if epilogue:
        entry["epilogue"] = list(epilogue)
    source = None
    try:
        from merlin.targetgen.corpus_synth import source_for_op

        source = source_for_op(op)
    except Exception:  # noqa: BLE001 -- an unresolvable writer is decided by the caller, not guessed
        source = None
    if source:
        entry["source"] = source
    return entry


def _writer_for(entry: dict) -> str | None:
    from merlin.targetgen.corpus_synth import _writer_for as writer

    return writer(entry)


def complete_for_builder(entry: dict, binding) -> None:
    """Supply the operands a builder REQUIRES but a cell does not mention. In place.

    A cell says "this op, on this axis, in this regime". A builder additionally demands whatever its op
    structurally needs: a residual add needs its two operand scales and its rounding bound, a pooled
    epilogue needs the window it pools over, a resident-reuse capsule needs the matmuls that reuse the
    weight. Those are properties of the OP, not of any target, and each is supplied the way
    ``corpus_synth.declare_pool_window`` / ``declare_residual_add_params`` already supply theirs -- this
    reuses both rather than restating them.

    Nothing here guesses at correctness. Every completed entry is then handed to the real builder, and
    one that still will not build is dropped with the builder's own message (see :func:`_buildable`).
    """
    from merlin.targetgen.corpus_synth import declare_pool_window, declare_residual_add_params

    declare_pool_window(entry)
    declare_residual_add_params(entry)

    op = str(entry.get("op") or "")
    # A SEAM CARRIES AN ADD TOO, and the same arithmetic has to be declared for it. ``corpus_synth``'s
    # declarer keys on the standalone add's own op names, so a fused seam -- a contraction whose result
    # is added to a residual -- reaches the builder with no scales and no bound and is refused. Unity
    # scales and a zero bound for the same reason they are that for the standalone add: under any other
    # pair the capsule would test a multiplier this generator chose rather than the add.
    if op == "residual_seam":
        from merlin.targetgen.corpus_synth import SYNTH_RESIDUAL_BOUND_LSB, SYNTH_RESIDUAL_SCALE

        entry.setdefault("lhs_scale", SYNTH_RESIDUAL_SCALE)
        entry.setdefault("rhs_scale", SYNTH_RESIDUAL_SCALE)
        entry.setdefault("bound_lsb", SYNTH_RESIDUAL_BOUND_LSB)
    # RESIDENT REUSE IS DEFINED BY THE REUSE: one weight, more than one matmul against it. One matmul
    # would be a plain contraction under another name, so two is the minimum that exercises the thing
    # the op exists to test.
    if op == "resident_reuse" and "matmuls" not in entry:
        m = entry.get("M", "tile")
        entry["matmuls"] = [
            {"lhs": "A0", "out": "Y0", "M": m, "epilogue": list(entry.get("epilogue") or ())},
            {"lhs": "A1", "out": "Y1", "M": m, "epilogue": list(entry.get("epilogue") or ())},
        ]


def _falsifiable(cap: dict) -> None:
    """Raise if this capsule's golden admits an answer that is wrong but still passes.

    The integer path recomputes its golden from the capsule, so the check runs here exactly as it runs
    in the writer. A non-integer regime's golden comes from an external engine and cannot be recomputed;
    that case is left to the writer rather than guessed at.
    """
    from merlin.targetgen import capsule_golden as CG
    from merlin.targetgen import numeric_falsifiability as NF

    policy = cap.get("numeric_policy") or {}
    if str(policy.get("compare") or "") not in ("exact_int", "exact", "bounded_int"):
        return
    outputs = CG.golden({**cap, "__dir__": ""})
    NF.falsifiable_policy(policy, outputs or {}, name=str(cap.get("name") or "?"))


def _build_probe(entry: dict, binding) -> tuple[str | None, dict | None, int | None]:
    """``(refusal, produced_op)`` — ask the real builder to materialize this entry.

    THE BUILDER IS THE ARBITER, not a table here of what each op accepts. A generator that decided for
    itself which epilogue an op can carry would drift from the builders the moment one changed, and the
    drift would present as capsules that fail to generate -- one at a time, hours later. Asking the
    builder at synthesis time turns every such case into a recorded finding instead.

    ⚠️ THE OP THE BUILDER PRODUCES IS NOT ALWAYS THE OP THE ENTRY ASKED FOR, and the difference is not
    cosmetic: ``BUILDERS`` maps several names onto one builder, and that builder writes its own name
    into ``operation.op``. An entry naming ``add`` comes back as a ``residual_add`` capsule. Measured
    here: five entries minted under ``add`` landed on ``residual_add``'s coverage cells -- an op this
    run had been told to leave to another change -- and the coverage report showed those cells closed by
    capsules that were not supposed to exist. So the PRODUCED op is what the entry is credited and
    filtered as.
    """
    from merlin.targetgen import corpus_spec as CS

    try:
        # RESOLVE THE TILE TOKENS FIRST, exactly as the generator does before it calls a builder. The
        # entries stay tile-relative on purpose -- that is what lets one entry describe the same shape
        # on a target with a different array edge -- but a builder is handed integers. Probing with the
        # tokens made every builder that reads `entry["M"]` numerically refuse with `invalid literal for
        # int() with base 10: '5*tile'`, and 54 perfectly buildable cells were recorded as capability
        # gaps on the strength of the probe's own mistake.
        cap, _mlir = CS.build(_resolve_flat_extents(entry, binding), binding)
        # ...AND THE GOLDEN, AND WHETHER ANYTHING COULD FAIL IT. Building is not enough: the writer goes
        # on to compute the answer key and to check that some wrong answer would actually miss it, and
        # three entries that built cleanly died there -- one with `UnfalsifiablePolicy: the golden has
        # too little spread to grade`, which is the writer refusing a capsule that COULD NOT FAIL. A
        # capsule that cannot fail is worse than no capsule, so the same refusal is taken here, where it
        # becomes a recorded finding instead of a generation error hours later.
        _falsifiable(cap)
    except Exception as exc:  # noqa: BLE001 -- the refusal is the result, whatever its type
        return f"{type(exc).__name__}: {str(exc)[:240]}", None, None
    return None, cap, _output_elements(_mlir)


def _output_elements(mlir: str) -> int | None:
    """What the affordability gate will price this capsule on — read from its own interface MLIR.

    THE SAME METRIC, not a proxy. Pricing on ``M x N`` from the entry was right for a contraction and
    wrong for a convolution, whose output extents are DERIVED from its window and image and whose entry
    names no M at all: ``OC_conv2d_aspect_wide`` came out at 1,152 written elements against a 923
    ceiling and went to certification uncapped. Asking the built module is the only way the cap and the
    gate agree by construction.
    """
    try:
        from merlin.targetgen import cert_cost as CC

        return int(CC.capsule_output_elements(mlir))
    except Exception:  # noqa: BLE001 -- unpriceable here is left to the gate, never capped on a guess
        return None


def _default_shape(bounds: FR.FormBounds) -> dict:
    edge = int(bounds.tile_edge or 0)
    return {"m": edge, "k": edge, "n": edge}


def _candidate(op, family, axis, regime, bounds, tile, dtype, cov):
    """``(entry, probe_capsule)`` for one cell, or ``None`` when nothing places it."""
    if axis == OC.EPILOGUE_AXIS:
        shape = _default_shape(bounds)
        probe = _probe_capsule(op, family, shape, dtype=dtype, epilogue=(regime,))
        return _entry_for(
            op, family, shape, dtype=dtype, tile=tile, axis=axis, regime=regime, epilogue=(regime,)
        ), probe
    if axis in FR.WINDOW_AXES:
        window = FR.smallest_window_for(axis, regime, bounds)
        if window is None:
            return None
        shape = _default_shape(bounds)
        probe = _probe_capsule(op, family, shape, dtype=dtype, window=window)
        return _entry_for(op, family, shape, dtype=dtype, tile=tile, axis=axis, regime=regime, window=window), probe
    shape = FR.smallest_shape_for(axis, regime, bounds, family=str(family))
    if shape is None:
        return None
    probe = _probe_capsule(op, family, shape, dtype=dtype)
    return _entry_for(op, family, shape, dtype=dtype, tile=tile, axis=axis, regime=regime), probe


def synthesize_op_cells(
    target: str,
    *,
    report: dict,
    bounds: FR.FormBounds,
    binding,
    capability_map: dict,
    skip_ops=(),
    max_capsules: int,
    max_output_elements: int | None = None,
) -> dict:
    """Build, independently classify, then credit explicit uncovered obligations.

    The caller supplies the exact cohort report, binding and resource policy.
    Missing builders/layouts remain debt. A proposal is never credited, nor is
    an unvalidated entry emitted. Exceeding the synthesis budget refuses the run.
    """
    if binding is None or getattr(binding, "target", target) != target or bounds.target != target:
        raise ValueError("matching explicit target, bounds and builder binding required")
    if type(max_capsules) is not int or max_capsules <= 0:
        raise ValueError("max_capsules must be a positive explicit budget")
    if max_output_elements is not None and (type(max_output_elements) is not int or max_output_elements <= 0):
        raise ValueError("output certification ceiling must be positive")
    tile = bounds.tile_edge
    if not tile or getattr(binding, "tile_dim", None) != tile:
        raise ValueError("builder tile disagrees with derived form bounds")
    remaining = set(report["under"])
    if not remaining <= report["cells"].keys():
        raise ValueError("uncovered cells must belong to the explicit requirement")
    result = SynthReport()
    deferred = set(skip_ops)
    dtype = binding.cap_dtype(binding.operand_dtype)
    for key in sorted(remaining):
        if key not in remaining:
            continue
        cell = report["cells"][key]
        op, family, axis, regime = (cell[name] for name in ("op", "family", "axis", "regime"))
        if key != OC.OpCell(op, axis, regime).key():
            raise ValueError("cell key and declaration disagree")
        if op in deferred:
            result.unwritable[op] = "explicitly deferred"
            continue
        # Builders, not a synthetic geometry or optional framework body, must
        # materialize and independently evaluate every accepted proposal.
        from merlin.targetgen.corpus_spec import BUILDERS

        if op not in BUILDERS:
            result.unwritable[op] = "no direct capsule builder; source capture required"
            continue
        proposal = _candidate(op, family, axis, regime, bounds, tile, dtype, report)
        if proposal is None:
            result.unplaceable[key] = "no justified shape/layout witness"
            continue
        entry, _ = proposal
        complete_for_builder(entry, binding)
        refusal, cap, elements = _build_probe(entry, binding)
        if refusal:
            result.unbuildable[key] = refusal
            continue
        refusal = OC.capsule_admission(cap, capability_map, bounds)
        if refusal is not None:
            result.unbuildable[key] = refusal
            continue
        produced = cap["operation"]["op"]
        if produced in deferred:
            result.unbuildable[key] = f"builder produces deferred operation {produced}"
            continue
        landed = OC.capsule_cells(cap, bounds)
        intended = OC.OpCell(produced, axis, regime).key()
        if intended not in landed or intended not in remaining:
            result.unplaceable[key] = "actual built capsule does not close the intended outstanding cell"
            continue
        gained = landed & remaining
        if max_output_elements is not None:
            if elements is None:
                result.unbuildable[key] = "output extent unavailable for explicit certification budget"
                continue
            if elements > max_output_elements:
                entry["max_oracle_tier"] = "L2"
        if len(result.entries) == max_capsules:
            raise ValueError("op-cell synthesis exceeds explicit capsule budget")
        entry["name"] = f"{PREFIX}_{produced}_{axis}_{regime}"
        entry["source_reference"] += "; actual built cells: " + ", ".join(sorted(gained))
        result.entries.append(entry)
        result.covered.extend(sorted(gained))
        remaining -= gained
    return {
        "capsules": result.entries,
        "provenance": {
            "target": target,
            "generated_by": "merlin_experiments.phase0.op_cell_synth",
            "qualification": "builder_and_integer_falsifiability_only",
            "n_cells_required": len(report["cells"]),
            "n_cells_closed": len(result.covered),
            "entries_validated_against_builder": True,
            "bounds": bounds.to_dict(),
            "ops_deferred": sorted(deferred),
            "ops_not_minted_here": result.unwritable,
            "cells_the_builder_refused": result.unbuildable,
            "cells_with_no_placement": result.unplaceable,
            "still_uncovered": sorted(remaining),
        },
    }
