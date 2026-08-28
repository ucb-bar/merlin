"""Model-level accelerator coverage: of a REAL model's regions, how many can a target actually take?

Capsule pass-rate answers "does the corpus pass". It cannot answer "does a model compile": a corpus is a
closed set of hand-picked families, while a captured model is dominated by unnamed ``linalg.generic``
regions whose meaning lives in their bodies. A backend can pass every capsule and still lower almost
nothing of a real model -- and nothing in the capsule verdict would say so.

This module measures the second question, target-agnostically: walk a captured model's linalg, describe
each region structurally, and ask the target's OWN capability contract
(:mod:`merlin.targetgen.eligibility`) whether that region is acceleratable. Nothing here is target-specific
-- the target is a parameter and every capability fact comes from its contract.

Four counts per (model, target), and the last is the one that keeps the rest honest:

``routed``
    regions the contract says this target can accelerate.
``fallback``
    regions it cannot -- which a general compiler must hand to the scalar/vector path. Not a failure: a
    real compiler's job is to cover these, and a coverage number that hides them describes a kernel
    library rather than a compiler.
``unclassified``
    regions whose family could NOT be determined at all.

``unclassified`` is reported separately and never folded into either bucket. A region we cannot name is
evidence neither of coverage nor of a gap, and folding it into one is exactly how a coverage number
becomes a lie. Provenance tags are used when present but never trusted to be present -- real captures
leave a large fraction of regions untagged, and tags disagree with the IR often enough that they are a
hint, not an authority (structural evidence wins where both exist).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from merlin.targetgen import semantic_families as sf
from merlin.targetgen.eligibility import RegionDescriptor, capability_map_for_target, is_eligible

#: MLIR element-type spelling -> quant-format registry name. Types with no registry entry map to None so
#: the descriptor carries an honest "unknown dtype" rather than a guessed width (a wrong dtype silently
#: changes an eligibility verdict, so guessing here would fabricate coverage).
_ELEM_DTYPE: dict[str, str] = {
    "f32": "fp32", "f16": "fp16", "bf16": "bf16", "i8": "int8", "i4": "int4",
}

#: Weights-manifest dtype spelling (torch names) -> quant-format registry name. Unlisted spellings are
#: DROPPED rather than mapped to a nearby width: a wrong precision silently flips an eligibility verdict.
_MANIFEST_DTYPE: dict[str, str] = {
    "float32": "fp32", "float16": "fp16", "bfloat16": "bf16",
    "int8": "int8", "int4": "int4",
    "float8_e4m3fn": "fp8_e4m3", "float8_e5m2": "fp8_e5m2",
}

#: Storage width per format, used only to pick the quantized payload out of a parametrization that also
#: carries its fp32 scale. Narrowest wins, so the metadata never masks the quantization.
_FORMAT_BITS: dict[str, int] = {
    "int4": 4, "fp8_e4m3": 8, "fp8_e5m2": 8, "int8": 8, "fp16": 16, "bf16": 16, "fp32": 32,
}


@dataclass
class CoverageReport:
    """Per (model, target) region accounting, split so that each fraction has an explicit denominator.

    FAMILY coverage is the primary number and is dtype-agnostic: of the regions we could name, how many
    are in a family this target supports at all. PRECISION is reported separately, because a capture's
    element types do NOT express a model's quantization -- an ``int8`` capture carries int8 element types on
    a small minority of its regions and an ``fp8`` capture carries none at all, since the quantization lives
    in the sidecar weights manifest. Gating on the element type therefore measures the capture's
    annotations, not the hardware fit, and reads as a confident percentage while doing so.

    ``family_supported + family_unsupported + unclassified == n_regions``.
    """

    model: str
    target: str
    n_regions: int = 0
    family_supported: int = 0
    family_unsupported: int = 0
    unclassified: int = 0
    # precision, over the family_supported subset only
    precision_known: int = 0
    dtype_ok: int = 0
    dtype_blocked: int = 0
    by_family: Counter = field(default_factory=Counter)
    by_op: Counter = field(default_factory=Counter)
    unclassified_ops: Counter = field(default_factory=Counter)
    unsupported_families: Counter = field(default_factory=Counter)
    by_precision: Counter = field(default_factory=Counter)

    @property
    def family_fraction(self) -> float:
        """The PRIMARY metric: family-supported as a share of CLASSIFIED regions. Regions we could not name
        stay out of the denominator — a model full of unnameable regions must not read as well covered."""
        classified = self.family_supported + self.family_unsupported
        return (self.family_supported / classified) if classified else 0.0

    @property
    def classified_fraction(self) -> float:
        return ((self.n_regions - self.unclassified) / self.n_regions) if self.n_regions else 0.0

    @property
    def precision_fraction(self) -> float | None:
        """Of the family-supported regions whose precision the capture ACTUALLY expressed, how many are in
        a format the target accepts. ``None`` when no precision was expressed at all — the honest answer
        there is "unknown", not 0% and not 100%."""
        judged = self.dtype_ok + self.dtype_blocked
        return (self.dtype_ok / judged) if judged else None

    def to_dict(self) -> dict:
        return {
            "model": self.model, "target": self.target, "n_regions": self.n_regions,
            "family_supported": self.family_supported,
            "family_unsupported": self.family_unsupported,
            "unclassified": self.unclassified,
            "family_fraction_of_classified": round(self.family_fraction, 4),
            "classified_fraction": round(self.classified_fraction, 4),
            "precision_known": self.precision_known,
            "dtype_ok": self.dtype_ok, "dtype_blocked": self.dtype_blocked,
            "precision_fraction_of_judged": (None if self.precision_fraction is None
                                             else round(self.precision_fraction, 4)),
            "by_family": dict(self.by_family.most_common()),
            "by_op": dict(self.by_op.most_common()),
            "unclassified_ops": dict(self.unclassified_ops.most_common()),
            "unsupported_families": dict(self.unsupported_families.most_common()),
            "by_precision": dict(self.by_precision.most_common()),
        }


def _short_op(op_name: str) -> str:
    """``linalg.matmul`` -> ``matmul``. Structural split on the dialect separator, no pattern matching."""
    return op_name.rpartition(".")[2] or op_name


def _attr_str(op, key: str) -> str | None:
    """A string attribute's value, or None. Tolerates attributes that are not string-typed."""
    attr = op.attributes.get(key)
    data = getattr(attr, "data", None)
    return data if isinstance(data, str) else None


def _elem_dtype(op) -> str | None:
    """Registry dtype name for the region's first tensor operand element type, or None when the type is
    absent or has no registry entry. Never guesses."""
    for operand in getattr(op, "operands", ()):  # first ranked operand wins (the activation/lhs)
        elem = getattr(getattr(operand, "type", None), "element_type", None)
        if elem is None:
            continue
        return _ELEM_DTYPE.get(str(elem))
    return None


def _is_region_op(op) -> bool:
    """Structure-carrying ops only. A terminator or a pure-init op is not a unit of computation to route,
    and counting them inflates every denominator."""
    name = op.name
    if not name.startswith("linalg."):
        return False
    return _short_op(name) not in ("yield", "index", "init_tensor")


def weight_precisions(manifest_path: str | Path) -> dict[str, str]:
    """Owning-module fqn -> quant-format name, read from a capture's WEIGHTS manifest.

    This is where a capture's real precision lives. The IR does not carry it: an ``int8`` capture types only
    a minority of its regions as ``i8`` and an ``fp8`` capture carries no fp8 element type at all, while
    ``prov.orig_dtype`` is the pre-quantization torch dtype (``float32`` in every variant). The manifest
    entries are ``{weight: "a.b.c.weight", dtype: "int8"}``, and regions carry ``prov.fqn`` as the OWNING
    module (``a.b.c``), so the join is the weight name minus its trailing component.

    Only formats the registry knows are kept; anything else is dropped rather than mapped to a guess.
    """
    import json

    out: dict[str, str] = {}
    doc = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    entries = doc.values() if isinstance(doc, dict) else doc
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        weight, dtype = entry.get("weight"), entry.get("dtype")
        if not isinstance(weight, str) or not isinstance(dtype, str):
            continue
        owner, _, leaf = weight.rpartition(".")
        # A module owns several tensors -- the operand, plus a bias and often a scale -- and only the
        # OPERAND's precision decides whether a contraction can run on a mesh. Keying every tensor by its
        # owner let a module's fp32 bias overwrite its int8 weight (last write wins), which collapsed 67
        # int8 tensors to a single int8 module and made a quantized capture look like an fp32 one.
        if leaf != "weight":
            continue
        name = _MANIFEST_DTYPE.get(dtype)
        if name is None:
            continue
        if owner:
            out[owner] = name
    return out


def storage_precisions(manifest_path: str | Path) -> dict[str, str]:
    """Owning-module fqn -> the precision a weight is STORED at, which is not what the mesh sees.

    A quantized capture keeps its narrow tensor behind a torch parametrization
    (``<module>.parametrizations.weight.original0``) and materializes ``<module>.weight`` by dequantizing it
    at forward time. So a capture named ``_int8`` can hold 67 int8 tensors while every contraction still
    presents fp32 operands: the quantization is storage, and the compute graph dequantizes before the GEMM.

    Keeping the two apart is the difference between "this accelerator cannot run this model" and "this
    capture does not offer this accelerator anything to run" -- the second is a capture-pipeline gap, and
    reporting it as the first would blame the backend for it.
    """
    import json

    marker = ".parametrizations."
    out: dict[str, str] = {}
    doc = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    entries = doc.values() if isinstance(doc, dict) else doc
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        weight, dtype = entry.get("weight"), entry.get("dtype")
        if not isinstance(weight, str) or not isinstance(dtype, str) or marker not in weight:
            continue
        name = _MANIFEST_DTYPE.get(dtype)
        if name is None:
            continue
        module = weight.split(marker, 1)[0]
        # A parametrization holds the quantized PAYLOAD plus its scale/zero-point, so several tensors map to
        # one module. Keep the NARROWEST: the narrow one is the quantization, the wide ones are its
        # metadata. Last-write-wins let an fp32 scale overwrite an int8 payload and reported 1 int8 module
        # where the capture has 67 -- the same collision as keying compute precision by owner.
        prior = out.get(module)
        if prior is None or _FORMAT_BITS.get(name, 99) < _FORMAT_BITS.get(prior, 99):
            out[module] = name
    return out


def regions_from_module(module, *, precisions: dict[str, str] | None = None) -> tuple[RegionDescriptor, ...]:
    """Describe every computation-carrying linalg region in a parsed model module.

    Family resolution order: the op's own NAME first (structural — ``linalg.matmul`` is a contraction
    whatever a tag claims), then provenance tags as a fallback for the unnamed ``linalg.generic`` case.
    Unresolved stays None, so the caller counts it as unclassified instead of assuming a family.
    """
    out: list[RegionDescriptor] = []
    for op in module.walk():
        if not _is_region_op(op):
            continue
        short = _short_op(op.name)
        family = sf.from_op(short)
        if family is None:  # unnamed region: fall back to whatever provenance the capture stamped
            family = sf.from_prov(_attr_str(op, "prov.family"), _attr_str(op, "prov.op"))
        # Precision from the weights manifest when we have one, joined on the region's owning module.
        # Element type is the FALLBACK, not the authority: it under-reports quantization badly.
        precision = None
        if precisions:
            precision = precisions.get(_attr_str(op, "prov.fqn") or "")
        if precision is None:
            precision = _elem_dtype(op)
        out.append(RegionDescriptor(source=short, op=short, family=family, in_dtype=precision))
    return tuple(out)


def coverage_for(regions: tuple[RegionDescriptor, ...], target: str, *,
                 model: str = "") -> CoverageReport:
    """Ask ``target``'s capability contract about each region. Pure accounting — no lowering is attempted,
    so this is the CEILING a submission for this target could reach, not what any submission does reach."""
    cap_map = capability_map_for_target(target)
    rep = CoverageReport(model=model, target=target, n_regions=len(regions))
    for region in regions:
        rep.by_op[region.op or "?"] += 1
        family = region.resolved_family()
        if family is None:
            rep.unclassified += 1
            rep.unclassified_ops[region.op or "?"] += 1
            continue
        rep.by_family[family] += 1
        rep.by_precision[region.in_dtype or "<unexpressed>"] += 1
        if family not in cap_map:
            rep.family_unsupported += 1
            rep.unsupported_families[family] += 1
            continue
        rep.family_supported += 1
        # Precision is judged ONLY where the capture expressed one. Asking is_eligible with a None dtype
        # returns eligible (a None want is "not applicable"), so folding the unexpressed case into dtype_ok
        # would manufacture precision coverage out of missing metadata.
        if region.in_dtype is None:
            continue
        rep.precision_known += 1
        if is_eligible(region, cap_map).eligible:
            rep.dtype_ok += 1
        else:
            rep.dtype_blocked += 1
    return rep


def route_model(regions: tuple[RegionDescriptor, ...], target: str) -> dict:
    """Split an unseen model's regions across ``target``'s lanes via the ordinary router.

    This is the whole-model question the capsule verdict cannot answer: of everything the model actually
    contains, how much lands on the accelerator and how much the scalar/vector lane must carry. It reuses
    :func:`merlin.targetgen.routing.route_plan`, so a stress test and a real compilation agree on lane
    assignment by construction instead of by a second implementation that can drift.

    Counts, not verdicts: a region on the scalar/RVV lane is a correct outcome (a matmul-only mesh should
    not claim a norm), and the ratio between the lanes IS the result. Regions with no resolvable family are
    reported separately — they never became demands, so they are absent from the router's own totals and
    would otherwise vanish from the accounting.
    """
    from merlin.targetgen.routing import OpDemand, route_plan

    demands, unnamed = [], 0
    for region in regions:
        if region.resolved_family() is None:
            unnamed += 1
            continue
        demands.append(OpDemand(op=region.op or "", in_fmt=region.in_dtype or "",
                                weight_fmt=region.weight_dtype, site=region.source or ""))
    plan = route_plan(demands, target)
    return {
        "target": target,
        "n_regions": len(regions),
        "mesh": len(plan["mesh"]),
        "in_contract_fallback": len(plan["fallback"]),
        "scalar_rvv": len(plan["scalar_rvv"]),
        "unnamed": unnamed,
        "mesh_fraction_of_demands": (len(plan["mesh"]) / len(demands)) if demands else 0.0,
    }


def load_module(path: str | Path):
    """Parse a captured model's MLIR. Imported lazily so this module stays importable without xDSL."""
    from xdsl.context import Context
    from xdsl.parser import Parser
    from xdsl.universe import Universe

    # Register the FULL upstream dialect set. ``allow_unregistered`` alone is not enough: an unregistered
    # op can only be parsed in generic form, and a capture is written in custom assembly
    # (``tensor.empty() : tensor<...>``), so every real model fails to parse without this.
    ctx = Context(allow_unregistered=True)
    for name, factory in Universe.get_multiverse().all_dialects.items():
        ctx.register_dialect(name, factory)
    return Parser(ctx, Path(path).read_text(encoding="utf-8"), str(path)).parse_module()
