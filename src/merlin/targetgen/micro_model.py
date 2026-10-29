"""The smallest whole model that exercises everything a target declares — composed, not chosen.

A whole-model capsule is the only capsule that proves the compiler can take a real network END TO END:
partition it, lower the eligible regions, route the rest to the host lane, keep the boundary correct, and
get the right answer. The corpus has three of them and every one is far too large to run at the
cycle-accurate tier — one is 497 KB of interface MLIR against 6.6 MB of weights — so the end-to-end claim
rests entirely on a functional oracle whose own descriptor says ``derived_from_rtl: false``. The strongest
thing the corpus asserts is the one thing no hardware ever checked.

The fix is not "pick a smaller model". A model picked by hand covers whichever layers its author happened
to think of, and this repo already has that failure on record: a corpus whose only model capsule was a
LLaMA decoder, so its captured linalg carried no ``tanh``, no ``erf`` and no convolution, and nothing in
the scorecard said the evidence was one architecture wide.

So the inventory is DERIVED, from the same three sources the coverage requirement uses:

``accelerator layers``
    one layer per ``(family, dtype)`` cell whose emitted standalone operation is admitted at that
    dtype by both the capability manifest and the authored software spec. A fused-only capability is
    recorded as unexercised when this model has no matching fused statement; it never licenses a
    standalone operation merely because the two share a family.

``host layers``
    one layer per family a real capture CONTAINS whose emitted operation the target cannot run on the
    accelerator. These are not filler — they are what makes a host seam exist at all, and they are
    interleaved BETWEEN accelerator layers so the composition is ``A->H->A`` rather than a host prefix
    and a host suffix. On a matmul-only mesh the
    normalizations and activations of any real network have nowhere else to go, so this is the ordinary
    shape of real work rather than an artificial stress.

``op spelling``
    which concrete op stands for a family is taken from what the captures actually contain — the most
    frequent spelling for that family across every real capture — rather than from an author's taste. A
    family whose spelling cannot be mapped to a constructible layer is REPORTED, never quietly skipped.

``sizing``
    every extent is a small multiple of the target's OWN tile edge, so the result is minimal for that
    target rather than minimal for one geometry. The budget is a declared bound on the whole model's
    tensor footprint, because the cycle-accurate tier's cost is dominated by how much data the harness
    has to materialise — a model capsule on one target here was blocked outright by a 125 MB generated
    ``main.c`` of element-wise tensor initialisers.

What this module does NOT do is write the model. It states what the model must contain and why, so that
the authored network can be checked against a requirement instead of being trusted.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

#: A layer's side of the seam.
ACCELERATOR = "accelerator"
HOST = "host"

#: How many tile edges wide the model's working extents are. Small on purpose: this capsule exists to be
#: affordable at the cycle-accurate tier, and every extent is a multiple of the target's own edge so the
#: number means the same thing on a 16-wide mesh and a 64-wide one.
_DEFAULT_EXTENT_TILES = 2


@dataclass(frozen=True)
class LayerRequirement:
    """One layer the micro model must contain, and the evidence that it must."""

    family: str
    dtype: str | None
    side: str  # ACCELERATOR | HOST
    op: str | None = None  # the spelling real captures use for this family
    op_frequency: int = 0  # how many regions across all captures carried that spelling
    admitted_by: tuple[str, ...] = ()  # compute units declaring it (accelerator side)
    observed_in: tuple[str, ...] = ()  # captures containing the family (host side)
    emitted_op: str | None = None  # the concrete statement whose placement was checked
    why: str = ""

    def key(self) -> str:
        return "/".join(x for x in (self.family, self.dtype, self.side) if x)

    def to_dict(self) -> dict:
        out = {"family": self.family, "side": self.side, "why": self.why}
        if self.dtype:
            out["dtype"] = self.dtype
        if self.op:
            out["op"] = self.op
            out["op_frequency"] = self.op_frequency
        if self.admitted_by:
            out["admitted_by"] = list(self.admitted_by)
        if self.observed_in:
            out["observed_in"] = list(self.observed_in)
        if self.emitted_op:
            out["emitted_op"] = self.emitted_op
        return out


@dataclass
class MicroModelSpec:
    """What a target's minimal whole-model capsule must contain."""

    target: str = ""
    layers: list = field(default_factory=list)  # [LayerRequirement], in composition order
    extent: int | None = None  # the working extent, in elements
    tile_edge: int | None = None
    unmapped_families: dict = field(default_factory=dict)
    unexercised_capabilities: list = field(default_factory=list)  # declared cells no standalone statement proves
    notes: list = field(default_factory=list)

    def accelerator_layers(self) -> list:
        return [layer for layer in self.layers if layer.side == ACCELERATOR]

    def host_layers(self) -> list:
        return [layer for layer in self.layers if layer.side == HOST]

    def composition(self) -> str:
        """The composition shape this inventory produces, in the boundary axis's own vocabulary."""
        from merlin.targetgen import boundary as BD

        seq = [BD.ACCEL if layer.side == ACCELERATOR else BD.HOST for layer in self.layers]
        return BD.classify_sequence(seq)

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "tile_edge": self.tile_edge,
            "extent": self.extent,
            "composition": self.composition(),
            "n_accelerator_layers": len(self.accelerator_layers()),
            "n_host_layers": len(self.host_layers()),
            "layers": [layer.to_dict() for layer in self.layers],
            "unmapped_families": self.unmapped_families,
            "unexercised_capabilities": self.unexercised_capabilities,
            "notes": self.notes,
        }


def observed_spellings(captures: dict) -> dict:
    """``family -> Counter(op spelling)`` across every readable capture.

    Which concrete op stands for a family is a question about real networks, and the captures answer it.
    Choosing by taste is how a corpus ends up one architecture wide.
    """
    from merlin.targetgen import model_coverage as mc

    out: dict = {}
    for _label, path in sorted((captures or {}).items()):
        try:
            regions = mc.regions_from_module(mc.load_module(path))
        except Exception:  # noqa: BLE001 — unreadable capture
            continue
        for region in regions:
            fam = region.resolved_family()
            if fam is None:
                continue
            out.setdefault(fam, Counter())[region.op or "?"] += 1
    return out


def _observed_host_ops_by_family(captures: dict, admitted: dict) -> dict:
    """``family -> observed host operation rows``, over every captured dtype the target does not admit.

    Host layers are built from these, so the composed model's host islands are operations real
    captures hand the host rather than stock statements of the same family.
    """
    from collections import defaultdict

    from merlin.targetgen import conformance as CF
    from merlin.targetgen import host_lane_ops
    from merlin.targetgen import model_coverage as mc

    admitted_pairs = set()
    for fam, dtypes in (admitted or {}).items():
        for dtype in dtypes or ():
            try:
                admitted_pairs.add((str(fam), CF.capsule_dtype(str(dtype))))
            except Exception:  # noqa: BLE001 -- an unmappable spelling stays as declared
                admitted_pairs.add((str(fam), str(dtype)))
    pairs = set()
    for _label, path in sorted((captures or {}).items()):
        try:
            sources = list(mc.region_sources(mc.load_module(path)))
        except Exception:  # noqa: BLE001 -- unreadable capture
            continue
        for fam, dtype, _op, _frontend in sources:
            if fam and dtype:
                try:
                    pair = (str(fam), CF.capsule_dtype(str(dtype)))
                except Exception:  # noqa: BLE001
                    continue
                if pair not in admitted_pairs and not pair[1].startswith("unsupported_mlir:"):
                    pairs.add(pair)
    merged: dict = defaultdict(Counter)
    for key, rows in host_lane_ops.observed_host_ops(captures or {}, sorted(pairs)).items():
        family = key.partition("/")[0]
        for row in rows:
            merged[family][(row["op"], row.get("frontend_op"))] += int(row["n_regions"])
    return {
        family: [
            {"op": op, "frontend_op": frontend, "n_regions": n}
            for (op, frontend), n in sorted(counter.items(), key=lambda item: (-item[1], item[0][0]))
        ]
        for family, counter in merged.items()
    }


def interleave(accelerator: list, host: list) -> list:
    """Order the layers so the host work sits BETWEEN accelerator work.

    A host prefix and a host suffix would compose as ``H->A->H``, which the corpus already evidences. The
    shape it evidences NOWHERE is ``A->H->A`` — an accelerator region, a host island, another accelerator
    region — and that is precisely the placement decision a whole-model compiler gets wrong, because it is
    the one where keeping an intermediate resident and paying for a round trip actually differ.

    With no host layers at all the result is a run of accelerator layers, which is honest: a target that
    admits everything a real model contains has no seam to exercise, and manufacturing one would test a
    boundary that target does not have.
    """
    if not accelerator:
        return list(host)
    if not host:
        return list(accelerator)
    out = [accelerator[0]]
    rest = list(accelerator[1:])
    # Spread the host layers through the interior; never leading, never trailing, so the sequence opens
    # and closes on the accelerator and every host layer is a genuine island.
    per_gap = max(1, -(-len(host) // max(1, len(rest)))) if rest else len(host)
    hi = 0
    for acc in rest:
        for _ in range(per_gap):
            if hi < len(host):
                out.append(host[hi])
                hi += 1
        out.append(acc)
    if hi < len(host):
        # More host layers than gaps: put the remainder before the final accelerator layer rather than
        # after it, so the sequence still closes on the accelerator.
        tail = out.pop()
        out.extend(host[hi:])
        out.append(tail)
    return out


def _standalone_admission(
    family: str,
    dtype: str,
    op: str,
    capture_dtype: str | None,
    cap_map: dict,
    software_spec: dict | None,
) -> str | None:
    """Return why the emitted statement cannot serve as an accelerator layer, or ``None``.

    The capture worker casts the whole model for fp16/bf16 captures. Quantized formats start from an
    f32 graph and integerize contractions; the standalone body statements remain float operations.
    In particular, an int8 DMA capability cannot prove a floating-point transpose. The selected
    capture dtype is required: a different admitted dtype is a separate, unexercised cell.
    """
    from merlin.common import quant_formats as qf
    from merlin.targetgen.eligibility import RegionDescriptor, is_eligible
    from merlin.targetgen.software_spec import admit_operation

    if capture_dtype is None:
        return "selected model capture dtype is unknown"
    selected = "fp8_e4m3" if capture_dtype == "fp8" else capture_dtype
    if not qf.has(selected):
        return f"selected model capture dtype {capture_dtype!r} is unknown"
    selected = qf.get(selected).name
    if selected != qf.get(dtype).name:
        return f"this model captures {selected}, not the declared {dtype} capability cell"
    emitted_dtype = selected if op in {"matmul", "linear"} or selected in {"fp32", "fp16", "bf16"} else "fp32"
    region = RegionDescriptor(op=op, family=family, in_dtype=emitted_dtype, rank=2, layout="row_major_contiguous")
    hardware = is_eligible(region, cap_map)
    if not hardware.eligible:
        return hardware.reason
    if software_spec is None:
        return None
    numerics = software_spec["numerical_semantics"]
    signature = {
        "family": family,
        "operand_dtype": emitted_dtype,
        "accum_dtype": numerics["accumulator_dtype"],
        "readout_dtype": numerics["readout_dtype"],
        "rank": 2,
        "layout": "row_major_contiguous",
        "tails": "zero_pad_valid_window",
        "broadcasting": "none",
        "aliasing": "disjoint_inputs_outputs",
        "epilogues": [op] if family == "elementwise_map" else [],
        "composed_with": [],
        "scale_granularity": "tensor",
    }
    software = admit_operation(software_spec, op, signature, "accelerator")
    if software.get("constraints_status") != "matched":
        return f"authored software spec: {software['reason']}"
    return None


def spec(
    target: str,
    captures: dict,
    *,
    extent_tiles: int = _DEFAULT_EXTENT_TILES,
    software_spec: dict | None = None,
    capture_dtype: str | None = None,
) -> MicroModelSpec:
    """The derived inventory for ``target``'s minimal whole-model capsule.

    ``software_spec`` is the caller's selected snapshot, not a live target-path lookup. A frozen
    generation run passes both it and ``capture_dtype`` so placement uses the same capture inputs.
    """
    from merlin.targetgen import conformance as CF

    out = MicroModelSpec(target=target)
    bnd = CF.boundaries(target)
    out.tile_edge = bnd.tile_edge
    if bnd.tile_edge:
        out.extent = int(bnd.tile_edge) * int(extent_tiles)
    else:
        out.notes.append(
            "the target declares no tile edge, so extents cannot be sized against its own geometry; "
            "the model must state its extents explicitly and they are NOT minimal by derivation"
        )
    if not bnd.tile_edge_is_hardware_fact and bnd.tile_edge:
        out.notes.append(
            f"extents are sized against tile edge {bnd.tile_edge}, which is a SOFTWARE tiling default "
            f"for this target rather than a hardware boundary"
        )

    from merlin.targetgen.eligibility import capability_map_for_target

    admitted = CF.admitted(target)
    try:
        cap_map = capability_map_for_target(target)
    except Exception:  # noqa: BLE001 -- CF.admitted already reports an unresolvable contract as empty
        cap_map = {}
    units = CF.admitting_units(target)
    spellings = observed_spellings(captures)
    host_ops = _observed_host_ops_by_family(captures, admitted)

    # Which families a real capture contains, and how many regions carried each. The host side is drawn
    # from here rather than from imagination: a family no real model uses is not worth a layer.
    observed_counts: dict = {}
    observed_in: dict = {}
    for label, path in sorted((captures or {}).items()):
        try:
            hist = CF.observed(path, target)
        except Exception:  # noqa: BLE001
            continue
        for fam, n in hist.items():
            observed_counts[fam] = observed_counts.get(fam, 0) + int(n)
            observed_in.setdefault(fam, []).append(label)

    def _spelling(fam):
        c = spellings.get(fam)
        if not c:
            return None, 0
        op, n = c.most_common(1)[0]
        return (None, 0) if op == "?" else (op, int(n))

    accelerator: list = []
    host: list = []
    represented_on_host: set[str] = set()
    for fam in sorted(admitted):
        dtypes = tuple(admitted.get(fam) or ())
        if not dtypes:
            continue
        op, freq = _spelling(fam)
        emitted_op, _ = statement_for(fam)
        if op is None:
            out.unmapped_families[fam] = (
                "admitted by the hardware but no readable capture names an op for it, so the model "
                "cannot be composed from evidence here; state the layer explicitly or accept that this "
                "capability goes unexercised end to end"
            )
        blocked_reasons = []
        admitted_here = False
        for dt in dtypes:
            reason = _standalone_admission(fam, dt, emitted_op, capture_dtype, cap_map, software_spec)
            layer = LayerRequirement(
                family=fam,
                dtype=CF.capsule_dtype(dt),
                side=ACCELERATOR,
                op=op,
                op_frequency=freq,
                admitted_by=units.get((fam, CF.capsule_dtype(dt)), ()),
                emitted_op=emitted_op,
                why="the emitted operation is independently admitted at this dtype by hardware and software",
            )
            if reason is None:
                accelerator.append(layer)
                admitted_here = True
                continue
            out.unexercised_capabilities.append(
                {"family": fam, "dtype": dt, "emitted_op": emitted_op, "reason": reason}
            )
            blocked_reasons.append(reason)
        if not admitted_here and fam in observed_counts and fam not in represented_on_host:
            host.append(
                LayerRequirement(
                    family=fam,
                    dtype=None,
                    side=HOST,
                    op=op,
                    op_frequency=freq,
                    observed_in=tuple(sorted(observed_in.get(fam, ()))),
                    emitted_op=statement_for(fam, host_ops.get(fam))[0],
                    why="the emitted standalone operation cannot use this target's accelerator: "
                    + "; ".join(blocked_reasons),
                )
            )
            represented_on_host.add(fam)

    for fam in sorted(f for f in observed_counts if f not in admitted):
        op, freq = _spelling(fam)
        emitted_op, _ = statement_for(fam, host_ops.get(fam))
        host.append(
            LayerRequirement(
                family=fam,
                dtype=None,
                side=HOST,
                op=op,
                op_frequency=freq,
                observed_in=tuple(sorted(observed_in.get(fam, ()))),
                emitted_op=emitted_op,
                why=f"real captures contain {observed_counts[fam]} region(s) of this family and the target "
                f"declares no capability for it, so it MUST run on the host lane; placing it between "
                f"accelerator layers is what makes the seam exist",
            )
        )

    if len(accelerator) == 1 and host:
        # The same admitted contraction can close the round trip when no second standalone family can.
        # This does not pretend that a fused-only family supplied the second accelerator segment.
        from dataclasses import replace

        accelerator.append(replace(accelerator[0], why="repeated admitted operation closes the host seam"))
    out.layers = interleave(accelerator, host)
    if not host:
        out.notes.append(
            "every emitted operation in the captures has an accelerator placement, so the model has "
            "no host island and no seam to prove"
        )
    return out


class UnwritableLayer(ValueError):
    """A required layer no emitted statement expresses, named so the inventory can be fixed."""


#: The torch statement each op contributes to a composed model: an optional __init__ line and the
#: forward line. Keyed on the op names :func:`merlin.targetgen.corpus_synth.op_for_family` already
#: chooses from, so a composed model is built out of the SAME vocabulary the op capsules use rather
#: than a second one invented here.
#:
#: Every statement is SHAPE-PRESERVING. A layer that changed the extent would make the inventory's
#: order unwritable -- the composition axis is about what follows what, and a reduction that collapsed
#: a dimension would decide the rest of the model instead of exercising it.
_STATEMENT: dict[str, tuple[str | None, str]] = {
    "matmul": ("self.w{i} = nn.Parameter(_weight({i}, E, E))", "x = x @ self.w{i}"),
    "linear": ("self.w{i} = nn.Parameter(_weight({i}, E, E))", "x = torch.nn.functional.linear(x, self.w{i})"),
    "rmsnorm": (None, "x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)"),
    "layernorm": ("self.ln{i} = nn.LayerNorm(E)", "x = self.ln{i}(x)"),
    "softmax": (None, "x = torch.softmax(x, dim=-1)"),
    "silu": (None, "x = torch.nn.functional.silu(x)"),
    "gelu": (None, "x = torch.nn.functional.gelu(x)"),
    "add": (None, "x = x + 1.0"),
    "bias_add": ("self.b{i} = nn.Parameter(torch.zeros(E))", "x = x + self.b{i}"),
    "reduce_sum": (None, "x = x - x.sum(-1, keepdim=True) / E"),
    # Spelled as model2MLIR names captured host work (`prov.op`), so a host layer can reproduce an
    # operation real captures contain. Each is shape-preserving like every other statement here.
    "mul": (None, "x = x * 0.5"),
    "layer_norm": ("self.ln{i} = nn.LayerNorm(E)", "x = self.ln{i}(x)"),
    "reduce_mean": (None, "x = x + x.mean(-1, keepdim=True)"),
    "permute": (None, "x = x.permute(1, 0).contiguous().permute(1, 0)"),
    "movement": (None, "x = x.transpose(-1, -2).contiguous().transpose(-1, -2)"),
    # THE ATTENTION FAMILY IS `attention_full`. Two near misses are worth naming, because both look
    # right and neither works: `attention_qk` is classified as a CONTRACTION (`_op_family_map`) --
    # Q@K^T is exactly that -- so it would leave the attention layer unwritable while quietly adding a
    # second candidate to `contraction`; and `sdpa` is in the family but NOT in `available_ops()`,
    # which admits only ops with a direct-MLIR builder or a PyTorch body. That leaves `attention_full`
    # and `attention_mx`, and `attention_mx` is the one to avoid: its golden exists ONLY in the
    # block-scaled engine, and radiance's cells are fp16/bf16/f32 -- generation died with "no SIMT
    # golden for op 'attention_mx'" on six of them. Membership in this table IS the writability filter
    # (`available_ops() & set(_STATEMENT)`), so naming `attention_full` alone steers away from it.
    #
    # The composed input is a SQUARE `(E, E)` (see `get_model_and_inputs`), read here as `E` tokens of
    # width `E`, so attention returns `(E, E)` and the layer composes like every other statement. The
    # scale is SDPA's own `1/sqrt(E)` -- this capsule's golden is `host_torch_eager`, i.e. the emitted
    # source IS the reference, and it is graded at `atol 0.25 / rtol 0.02`, a band an unscaled E-term
    # dot product would leave immediately.
    # THE RESIDUAL IS LOAD-BEARING, not decoration. Attention is an averaging operator: softmax rows
    # sum to 1, so the output is a convex combination of V rows and its spread is far below its input's.
    # MEASURED on radiance's 28-layer inventory (6 attention layers): without the residual the composed
    # model ran std 1.017 -> 0.072 -> ... -> 0.0002, roughly 3.5x lost per layer, and generation then
    # failed the falsifiability gate ("the golden has too little spread to grade") -- a capsule whose
    # tolerance band cannot separate a right answer from a wrong one. Every real attention block is
    # residual for exactly this reason, so this is the faithful spelling as well as the gradeable one.
    "attention_full": (
        "self.qkv{i} = nn.Parameter(_weight({i}, 3 * E, E).reshape(3, E, E))",
        "x = x + torch.nn.functional.scaled_dot_product_attention("
        "x @ self.qkv{i}[0], x @ self.qkv{i}[1], x @ self.qkv{i}[2])",
    ),
}


def statement_for(family: str, observed: list | None = None) -> tuple[str, tuple[str | None, str]]:
    """``(op, (init_line, forward_line))`` for ``family``, or raise naming the family.

    With ``observed`` host operations (:func:`merlin.targetgen.host_lane_ops.observed_host_ops` rows),
    a HOST layer is one of those operations, so the composed model reproduces work the captures
    contain; none expressible raises. Otherwise the op is chosen by
    :func:`corpus_synth.op_for_family`, the same derivation the op capsules use.
    Raising rather than skipping is the point: a layer quietly dropped from a composed model changes
    the composition it was written to exercise, and the capsule would then test a different shape than
    the one its name claims.
    """
    from merlin.targetgen.corpus_synth import _op_family_map, available_ops, op_for_family

    if observed:
        from merlin.targetgen import host_lane_ops

        families = _op_family_map()
        writable = {op for op in available_ops() & set(_STATEMENT) if families.get(op) == family}
        chosen = host_lane_ops.choose(observed, writable)
        if chosen is None:
            raise UnwritableLayer(
                f"no emittable statement for the observed {family!r} host operations "
                f"{[row.get('op') for row in observed]}; add one to micro_model._STATEMENT"
            )
        return chosen["op"], _STATEMENT[chosen["op"]]
    op = op_for_family(family, admitted_ops=available_ops() & set(_STATEMENT))
    if op is None or op not in _STATEMENT:
        raise UnwritableLayer(
            f"no emittable statement for family {family!r}; add one to micro_model._STATEMENT or "
            f"establish that the inventory should not contain it -- do not drop the layer"
        )
    return op, _STATEMENT[op]


def emit_pytorch(spec) -> str:
    """A ``capsule.pytorch.py`` source for ``spec``: one composed model over its derived inventory.

    The layer ORDER is the spec's, which :func:`interleave` already arranged so host layers sit in the
    interior -- the composition a whole-model compiler actually gets wrong is the one where keeping an
    intermediate resident and paying for a round trip differ, and a host prefix or suffix does not
    exercise it.

    Everything dimensioned comes from the spec: the extent is the target's own tile edge times the
    declared multiple, so the same inventory emits a 32-wide model for a 16-wide array and a 128-wide
    one for a 64-wide array without being edited.

    Inputs and weights use exact integer operations and power-of-two scales:
    seeded random tensor generators are not byte-stable across PyTorch CPU
    capability paths, which makes sealed capture fail loader replay.
    """
    extent = int(getattr(spec, "extent", 0) or 0)
    if extent <= 0:
        raise UnwritableLayer(
            "the spec carries no extent, so there is no derived width to emit; a default here would be "
            "a geometry this repo does not have"
        )
    layers = list(getattr(spec, "layers", ()) or ())
    if not layers:
        raise UnwritableLayer("the spec carries no layers; there is no model to write")

    inits, fwd, notes = [], [], []
    for i, layer in enumerate(layers):
        if layer.emitted_op is not None:
            # The spec already chose (and, for a host layer, chose from observed host work).
            if layer.emitted_op not in _STATEMENT:
                raise UnwritableLayer(f"{layer.family}: classified statement {layer.emitted_op!r} has no source")
            op, (init_line, forward_line) = layer.emitted_op, _STATEMENT[layer.emitted_op]
        else:
            op, (init_line, forward_line) = statement_for(layer.family)
        if init_line:
            inits.append("        " + init_line.format(i=i))
        fwd.append(f"        # {layer.side}: {layer.family} (observed spelling {layer.op!r})")
        fwd.append("        " + forward_line.format(i=i))
        notes.append(f"#   {i}. {layer.side:11} {layer.family:16} -> {op}")

    body_init = "\n".join(inits) or "        pass"
    blocked = list(getattr(spec, "unexercised_capabilities", ()) or ())
    blocked_inventory = (
        "\nUnexercised accelerator capabilities (the emitted standalone statement cannot prove these):\n"
        + "\n".join(f"#   {row['family']}/{row['dtype']} via {row['emitted_op']}: {row['reason']}" for row in blocked)
        + "\n"
        if blocked
        else ""
    )
    return (
        '"""DERIVED micro model -- regenerate with merlin.targetgen.micro_model.emit_pytorch.\n'
        "\n"
        f"Composition: {spec.composition()}\n"
        "Layer inventory, in composition order:\n" + "\n".join(notes) + "\n" + blocked_inventory + "\n"
        "Accelerator labels require an admitted standalone statement at the emitted dtype; host labels\n"
        "mark statements the target cannot accelerate. Host layers sit between accelerator layers.\n"
        '"""\n'
        "import torch\n"
        "import torch.nn as nn\n"
        "\n"
        f"E = {extent}\n"
        "\n"
        "def _weight(index, rows, cols):\n"
        "    values = torch.arange(rows * cols, dtype=torch.int32).reshape(rows, cols)\n"
        "    return (((values * 37 + index * 101) % 251) - 125).to(torch.float32) / 1024.0\n"
        "\n"
        "\n"
        "class MicroModel(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        f"{body_init}\n"
        "\n"
        "    def forward(self, x):\n" + "\n".join(fwd) + "\n"
        "        return x\n"
        "\n"
        "\n"
        "def get_model_and_inputs():\n"
        "    model = MicroModel().eval()\n"
        "    values = torch.arange(E * E, dtype=torch.int32).reshape(E, E)\n"
        "    inputs = ((values % 251) - 125).to(torch.float32) / 64.0\n"
        "    return model, (inputs,)\n"
    )
