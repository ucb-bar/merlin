"""Preserve authored generalization intent and derive its semantic family."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

from merlin.common.record_formats import fast_safe_load
from merlin.development.targetgen.applications.native_storage import (
    native_storage_loader,
    native_storage_source_spec,
)
from merlin.development.targetgen.corpus import components as component_sources

# ---- capsule_intent ----------------------------------------------------------------------------

if TYPE_CHECKING:
    from merlin.development.targetgen.corpus.binding import CorpusBinding


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

    from merlin.development.targetgen.corpus.model_builders import _AXIS_BY_KIND, _fused_families
    from merlin.target.capabilities import semantic_families as _sf

    defaults = dict(getattr(binding, "semantic_defaults", None) or {})
    # ⚠️ NOT ``entry["semantic"]`` -- a profile entry already uses that key for a free-form op-semantics
    # label ("quantized_linear", "matmul_mxu0") that flows into ``operation.attributes.semantic``.
    # Reusing it here would collide with existing entries across target profiles and crash the
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
    axis = authored.get("generalization_axis") or defaults.get("generalization_axis") or _AXIS_BY_KIND.get(kind, "seen")
    block["generalization_axis"] = axis
    # A whole-model capsule spans families by construction, is never held out (see
    # generalization_splits), and is expected to contain regions the target cannot run -- so it must
    # never carry must_accelerate, whatever the corpus default says.
    #
    # ⚠️ NEITHER MAY A CAPSULE THAT FORBIDS THE MESH. `lanes.forbid: [on_mesh]` is the negative lane:
    # the capsule exists to prove the compiler does NOT put this work on the accelerator, and
    # `must_accelerate: true` demands the opposite. Emitting both produces a capsule that cannot pass
    # under any behaviour, and it is produced by exactly the combination that looks safest -- a
    # corpus-wide `semantic_defaults: {must_accelerate: true}` meeting a host-only capsule. Measured:
    # declaring that default on one profile gave `SY_host_only_elementwise_map` and
    # `SY_host_only_reduction` `forbid: [on_mesh]` alongside `must_accelerate: true`. The lane
    # declaration is the more specific statement and wins, structurally rather than by an exception
    # list -- the same treatment `kind == "model"` already gets above.
    lanes = entry.get("lanes") or {}
    forbids_mesh = "on_mesh" in {str(x) for x in (lanes.get("forbid") or ())}
    must_default = False if (kind == "model" or forbids_mesh) else bool(defaults.get("must_accelerate", False))
    # An AUTHORED `true` cannot override the negative lane either: the contradiction is the same
    # whichever half asserted it, and a capsule declaring both is an authoring error to report rather
    # than a preference to honour.
    authored_must = authored.get("must_accelerate", must_default)
    if forbids_mesh and bool(authored_must):
        raise ValueError(
            f"{entry.get('name', '<unnamed>')}: lanes.forbid names 'on_mesh' while must_accelerate is "
            f"true. The negative lane exists to prove this work does NOT reach the accelerator, so the "
            f"two demands cannot both hold and the capsule could not pass under any behaviour"
        )
    block["must_accelerate"] = bool(authored_must)
    # WHY the strongest assertion was withheld. The schema has a field for it, and for the interop and
    # host-island capsules it is the whole contract: they withhold ``must_accelerate`` on purpose,
    # because host-lane work is the behaviour under test rather than a fallback failure, and the demand
    # they DO make is `lanes.require`. That reason was hand-written into the capsule after generation
    # and therefore deleted by the next regeneration -- measured on three gemmini capsules at once --
    # which left a withheld assertion looking like an author who simply never made one. Same argument
    # as the block around it: emit it from the profile so a regen preserves it.
    reason = authored.get("not_asserted_reason", defaults.get("not_asserted_reason"))
    if reason:
        block["not_asserted_reason"] = reason
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
    # ANY OTHER AUTHORED KEY SURVIVES. The block above consumes the keys it knows and, until this,
    # silently dropped the rest -- so a capsule synthesized to discharge a NAMED obligation could not
    # say which one, and the coverage reader had to re-derive the answer from attributes. That works
    # only when the obligation is expressible in attributes: a `padUNKNOWN` convolution window is not,
    # because the member can only spell zero padding, so the obligation it was built for could never
    # be matched by it. Carrying the reference is what closes that, and it is general rather than a
    # special case for one axis.
    for key, value in authored.items():
        if key not in block and key not in (
            "must_accelerate",
            "eligible",
            "not_asserted_reason",
            "semantic_family",
            "generalization_axis",
        ):
            block[key] = value
    return block


# ---- capsule_privacy ---------------------------------------------------------------------------
# Reference artifact roles shared by capsule producers and candidate input views.
#
# This classifies protected files, never reads their tensor payloads and grants no
# capture, numerical or execution authority. Legacy capture schemas keep their
# original reference role even when they predate the explicit closure field.

REFERENCE_SCHEMA = "merlin.capsule_reference_closure.v1"
_CAPTURE_SCHEMA = "merlin.materialized_public_capture.v2"
_CAPTURE_REFERENCE = "golden.npy"
_CAPTURE_PREFIX = "source-capture-"


def capture_reference_closure(identities: dict, *, source_mode: str | None = None) -> dict:
    """Declare the exact expected-output artifact copied by the capture writer."""
    from merlin.development.targetgen.applications.independent_references import (
        reference_members as float_reference_members,
    )

    references = float_reference_members(identities)
    if source_mode is not None:
        from merlin.development.targetgen.applications.materialized_integer_capture import (
            MODE,
            reference_members,
        )

        if source_mode != MODE:
            raise ValueError("unsupported public capture reference mode")
        references = reference_members(identities)
    return {
        "schema": REFERENCE_SCHEMA,
        "members": [
            {
                "path": _CAPTURE_PREFIX + name,
                "role": "expected_output",
                "sha256": identities[name]["sha256"],
                "bytes": identities[name]["bytes"],
            }
            for name in references
        ],
    }


def _relative_member(value) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or "\x00" in value
        or Path(value).is_absolute()
        or any(part in {".", ".."} for part in value.split("/"))
        or any(not part for part in value.split("/"))
    ):
        raise ValueError("capsule reference member must be a safe relative file")
    return value


def capsule_answer_members(directory: Path, *, capsule: dict | None = None) -> dict[str, str]:
    """Return relative protected members and their roles from this exact capsule.

    Read only capsule metadata and file names, so callers can inspect a frozen
    copy independently of missing or changed original sources. Declared byte pins
    describe identity; this privacy classification does not verify those bytes.
    Every sandbox policy classifies each capsule of its live and frozen corpora,
    so metadata is scanned by libyaml under PyYAML's own safe constructor: the
    same documents, built several times faster (:func:`fast_safe_load`).
    """
    directory = Path(directory)
    if capsule is None:
        source = directory / "capsule.yaml"
        capsule = fast_safe_load(source.read_bytes()) if source.is_file() else {}
    if not isinstance(capsule, dict):
        raise ValueError("capsule privacy requires a mapping")
    members = {path.name: "golden" for path in directory.glob("golden.*") if path.is_file()}
    # Filename classification also protects the complete source copies when a
    # closure or schema was stripped. It never reads expected tensor values.
    members.update(
        {
            path.name: "golden"
            for path in directory.iterdir()
            if path.is_file()
            and path.name.startswith(
                ("source-capture-golden.", "source-capture-goldens.", "source-capture-integer-reference.")
            )
        }
    )
    coverage = directory / "expected_instruction_coverage.yaml"
    if coverage.is_file():
        members[coverage.name] = "golden"
    members.update(
        {
            path.name: "weight"
            for path in directory.iterdir()
            if path.is_file() and path.name.endswith((".safetensors", ".safetensors.manifest.json"))
        }
    )
    capture = capsule.get("materialized_capture") or {}
    if not isinstance(capture, dict):
        raise ValueError("capsule materialized capture is malformed")
    if capture.get("schema") == _CAPTURE_SCHEMA and capture.get("workload_role") == "performance_scale":
        members[_CAPTURE_PREFIX + _CAPTURE_REFERENCE] = "golden"
    closure = capsule.get("reference_closure")
    if closure is None:
        return members
    if (
        not isinstance(closure, dict)
        or set(closure) != {"schema", "members"}
        or closure.get("schema") != REFERENCE_SCHEMA
        or not isinstance(closure.get("members"), list)
        or not closure["members"]
    ):
        raise ValueError("capsule reference closure is malformed")
    seen = set()
    for row in closure["members"]:
        if not isinstance(row, dict) or set(row) != {"path", "role", "sha256", "bytes"}:
            raise ValueError("capsule reference artifact is malformed")
        path = _relative_member(row["path"])
        digest = row["sha256"]
        if (
            path in seen
            or row["role"] != "expected_output"
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or type(row["bytes"]) is not int
            or row["bytes"] < 0
        ):
            raise ValueError("capsule reference artifact role or identity is invalid")
        seen.add(path)
        members[path] = "golden"
    return members


# ---- capsule_results ---------------------------------------------------------------------------
# Typed complete result rosters for ordinary captured operation programs.


def _shape(value):
    if not isinstance(value, list):
        return []
    if not value:
        return [0]
    shape = _shape(value[0])
    if any(_shape(child) != shape for child in value):
        raise ValueError("captured result is not a rectangular tensor")
    return [len(value), *shape]


def publication(art, entry):
    """Require ABI, actual program signature and full golden for every result."""
    from merlin.compiler.ir.mlir_query import forward_signature

    abi = (art.meta or {}).get("output_abi")
    if not isinstance(abi, list) or len(abi) < 2:
        raise ValueError("multiple-result operation requires a complete typed output ABI")
    _, actual = forward_signature(art.linalg_mlir)
    if len(actual) != len(abi):
        raise ValueError("captured result ABI differs from the actual program signature")
    if not isinstance(art.golden, list) or len(art.golden) != len(abi):
        raise ValueError("captured operation omitted a complete result golden")
    names = entry.get("outs") or ["Y" + str(index) for index in range(len(abi))]
    if (
        not isinstance(names, list)
        or len(names) != len(abi)
        or any(not isinstance(name, str) or not name or not name.isidentifier() for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("captured operation requires one unique safe name per published result")
    for declared, typed, golden in zip(abi, actual, art.golden, strict=True):
        if not isinstance(declared, dict) or (declared.get("shape"), declared.get("dtype")) != typed:
            raise ValueError("captured result ABI differs from the actual typed program result")
        if _shape(golden) != declared["shape"]:
            raise ValueError("captured full result shape differs from its typed output ABI")
    return dict(zip(names, art.golden, strict=True))


# ---- capsule_capture_spec ----------------------------------------------------------------------
# Capture specifications projected from typed abstract capsule entries.


def _entry_seed(name: str) -> int:
    return sum((i + 1) * ord(c) for i, c in enumerate(name)) or 1


def _capture_spec(entry: dict, binding) -> dict:
    """Build the PytorchRefSource capture spec (op + shapes + dtype + seed) from an abstract entry."""
    dim = binding.tile_dim
    op = entry.get("op", "matmul")
    M = entry.get("M", entry.get("M_tiles", 1) * dim)
    K = entry.get("K", entry.get("K_tiles", 1) * dim)
    N = entry.get("N", entry.get("N_tiles", 1) * dim)
    if op == "attention_qk" and "N" not in entry and "N_tiles" not in entry:
        N = M  # Preserve the square default, while honoring explicit rectangular source geometry.
    spec = {
        "op": entry.get("capture_op", op),
        "dtype": entry.get("capture_dtype", binding.operand_dtype),
        "seed": _entry_seed(entry["name"]),
        "M": M,
        "K": K,
        "N": N,
    }
    if "shape" in entry:
        spec["shape"] = list(entry["shape"])
    if "permutation" in entry:
        spec["permutation"] = list(entry["permutation"])
    # WHICH QUANTIZATION the captured PROGRAM should carry, when the entry names one. The dtype default
    # for int8 is weight-only, which emits a float matmul over dequantized weights -- correct for a
    # model ladder, wrong for a capsule meant to exercise an integer datapath, and not fixable by
    # substituting a golden: the program itself contains no integer contraction.
    if entry.get("quant_scheme"):
        spec["quant_scheme"] = str(entry["quant_scheme"])
    if op in ("rmsnorm", "gemma_4norm", "layernorm"):
        spec["eps"] = entry.get("eps", 1.0 / 65536.0)
    # optional per-op fields (batch / soft-cap / conv geometry) pass through verbatim when present
    for f in (
        "B",
        "cap",
        "Cin",
        "Himg",
        "Wimg",
        "P",
        "Pool",
        "Dv",
        "causal",
        "eps",
        "input_palette",
        "producer_scale",
        "quant_scale",
        "native_input_dtype",
        "native_output_dtype",
        "native_source_abi",
        "frontend_op",
    ):
        if f in entry:
            spec[f] = entry[f]
    if any(key in entry for key in ("native_input_dtype", "native_output_dtype", "native_source_abi")):
        spec["generalization"] = entry.get("generalization")
    return spec


# ---- capsule_builtin_source --------------------------------------------------------------------
# Render builtin capture sources from explicit templates and typed input declarations.


def _permutation(spec: dict, rank: int) -> list[int]:
    """The axis order a ``permute`` program applies: the spec's, else the last two axes swapped (for
    the historic ``[M, K]`` form, the plain transpose)."""
    raw = spec.get("permutation")
    if raw is None:
        return [*range(rank - 2), rank - 1, rank - 2] if rank >= 2 else list(range(rank))
    if (
        not isinstance(raw, (list, tuple))
        or any(isinstance(axis, bool) or not isinstance(axis, int) for axis in raw)
        or sorted(raw) != list(range(rank))
    ):
        raise ValueError(f"permutation must order every axis of the rank-{rank} shape exactly once, got {raw!r}")
    return [int(axis) for axis in raw]


def render_builtin_source(
    spec: dict,
    *,
    preamble: str,
    integer_matmul: str,
    parametric_linear: str,
    bodies: dict[str, str],
    input_names: dict,
) -> str:
    """Render the PyTorch loader source for an op spec. ``spec`` carries ``op`` + the shape fields the op
    needs (M/K/N/Dv) + optional ``eps``/``causal``/``bias`` + ``seed``/``dtype``. Fail closed on an
    unknown op (never silently emit a wrong program)."""
    op = spec["op"]
    native = native_storage_source_spec(spec)
    if op == "dtype_cast" and native is None:
        raise ValueError("dtype_cast requires an explicit native source/result declaration")
    if spec.get("input_palette") is not None and (op == "int_matmul" or spec.get("quant_scheme")):
        raise ValueError("selected input palettes require the normal typed float builtin source path")
    if op != "int_matmul" and op not in bodies:
        raise KeyError(f"capsule_source has no PyTorch template for op {op!r} (have {sorted(bodies)})")
    raw_shape = spec.get("shape")
    if raw_shape is None:
        raw_shape = [spec.get("M", 16), spec.get("K", 16)]
    if (
        not isinstance(raw_shape, (list, tuple))
        or not raw_shape
        or any(isinstance(d, bool) or not isinstance(d, int) or d < 1 for d in raw_shape)
    ):
        raise ValueError(f"capsule shape must be a non-empty sequence of positive integers, got {raw_shape!r}")
    fields = {
        "op": op,
        "dtype": spec.get("dtype", "fp32"),
        "seed": int(spec.get("seed", 0)),
        "M": spec.get("M", 16),
        "K": spec.get("K", 16),
        "N": spec.get("N", 16),
        "Dv": spec.get("Dv", spec.get("K", 16)),
        "eps": spec.get("eps", 1e-5),
        "causal": bool(spec.get("causal", False)),
        "bias": bool(spec.get("bias", False)),
        # extra shape/scalar fields for the composite ops (batch, soft-cap, conv geometry)
        "B": spec.get("B", 2),
        "cap": spec.get("cap", 50.0),
        "Cin": spec.get("Cin", 1),
        "Himg": spec.get("Himg", 8),
        "Wimg": spec.get("Wimg", 8),
        "P": spec.get("P", 3 if op == "conv_residual_pool" else 2),
        "Pool": spec.get("Pool", 2),
        # Elementwise programs preserve the captured rank. Most synthetic probes use the historic
        # MxK form; an application-derived probe may instead carry its exact static shape.
        "shape_args": ", ".join(str(int(d)) for d in raw_shape),
        "perm_args": ", ".join(str(axis) for axis in _permutation(spec, len(raw_shape))),
    }
    for name in ("M", "K", "N", "Dv", "B", "Cin", "Himg", "Wimg", "P", "Pool"):
        if type(fields[name]) is not int or fields[name] < 1:
            raise ValueError(f"builtin source field {name} must be an explicit positive integer")
    for name in ("eps", "cap"):
        if type(fields[name]) not in (int, float) or not math.isfinite(fields[name]) or fields[name] <= 0:
            raise ValueError(f"builtin source field {name} must be an explicit positive finite scalar")
    for name in ("causal", "bias"):
        if name in spec and type(spec[name]) is not bool:
            raise ValueError(f"builtin source field {name} must be an explicit Boolean")
    fields["Ho"] = fields["Himg"] - fields["P"] + 1
    fields["Wo"] = fields["Wimg"] - fields["P"] + 1
    if native is not None:
        fields["native_output_torch_dtype"] = "float32"
    if op == "producer_quantizer_observer":
        fields.update(component_sources.parameters(spec))
    if op == "conv_residual_pool" and (
        fields["Ho"] < 1
        or fields["Wo"] < 1
        or type(fields["Pool"]) is not int
        or fields["Pool"] < 1
        or fields["Ho"] % fields["Pool"]
        or fields["Wo"] % fields["Pool"]
    ):
        raise ValueError("conv_residual_pool requires positive convolved extents divisible by the selected pool")
    if op == "int_matmul":
        if spec.get("quant_scheme"):
            raise ValueError("an isolated int_matmul has quantized operands; do not quantize it again")
        return preamble.format(**fields) + integer_matmul.format(**fields)
    # A QUANTIZED capture needs a weight PARAMETER for the scheme to bind to (see parametric_linear).
    # Only the contraction ops have a meaningful weight; asking for a quantized elementwise op is a
    # request that cannot be honoured, and saying so beats emitting an unquantized program under a
    # quantized name.
    if spec.get("quant_scheme"):
        if op not in ("linear", "matmul"):
            raise ValueError(
                f"quant_scheme is set for op {op!r}, but only a contraction carries a weight for a "
                f"torchAO scheme to quantize; an unquantized program under a quantized name is worse "
                f"than a refusal"
            )
        return preamble.format(**fields) + parametric_linear.format(**fields)
    source = preamble.format(**fields) + bodies[op].format(**fields)
    if native is not None:
        source += native_storage_loader(spec)
    if spec.get("input_palette") is not None:
        from merlin.development.targetgen.corpus.operands import render_source_inputs

        source += render_source_inputs(spec["input_palette"], input_names[op], fields["dtype"])
    return source
