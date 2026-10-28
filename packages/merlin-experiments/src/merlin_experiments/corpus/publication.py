"""Reject held-out source identities from public release and authoring inputs."""

from __future__ import annotations

import math
from collections import Counter
from functools import lru_cache
from pathlib import Path

import yaml

from ..phase0.claim_boundary import assert_no_claim_capsules, held_out_models
from ..spec import SpecError


def _contraction_kind(op):
    """Use the original capture census's ATen-to-semantic vocabulary."""
    from merlin.capture.contraction_formats import _CONTRACTIONS

    aliases = {name.split(".")[1]: kind for name, kind in _CONTRACTIONS.items()}
    aliases.update(_CONTRACTIONS)
    aliases.update({kind: kind for kind in _CONTRACTIONS.values()})
    return aliases.get(op)


def _pointwise_conv_operands(operands, attributes):
    """The capsule NHWC, flattened-KN convolution contract at a 1x1 window."""
    if (
        attributes.get("layout") != "nhwc"
        or attributes.get("kh") != 1
        or attributes.get("kw") != 1
        or attributes.get("stride") != [1, 1]
        or attributes.get("padding") != [0, 0, 0, 0]
        or attributes.get("dilation") != [1, 1]
        or attributes.get("groups", 1) != 1
    ):
        return None
    inputs = [shape for role, shape in operands if role == "input"]
    weights = [shape for role, shape in operands if role == "weight"]
    if len(inputs) != 1 or len(weights) != 1 or len(inputs[0]) != 4 or len(weights[0]) != 2:
        return None
    lhs, rhs = inputs[0], weights[0]
    if lhs[-1] != rhs[0] or attributes.get("ci") != lhs[-1]:
        return None
    return [(role, [math.prod(shape[:-1]), shape[-1]] if role == "input" else shape) for role, shape in operands]


def _canonical_operands(operation, operands):
    kind = _contraction_kind(operation["op"])
    if kind is None:
        # An unlisted spelling of a contraction (a fused or capsule-native op) still has
        # one input and one weight; identify it by its contraction shapes when they compose.
        roles = [role for role, _ in operands]
        if roles.count("input") == 1 and roles.count("weight") == 1:
            contraction = _canonical_operands({**operation, "op": "matmul"}, operands)
            if contraction is not None:
                return contraction
        return operation["op"], operands
    attributes = operation.get("attributes") or {}
    if not isinstance(attributes, dict):
        return None
    if kind == "conv2d":
        pointwise = _pointwise_conv_operands(operands, attributes)
        if pointwise is None:
            return kind, operands
        operands = pointwise
    elif kind not in {"matmul", "batch_matmul", "linear", "addmm"}:
        return operation["op"], operands
    # The publication boundary excludes the shape independently of epilogues.
    # In addmm(beta=0), the bias storage is not read at all.
    operands = [
        (role, shape)
        for role, shape in operands
        if not (kind == "addmm" and role == "bias" and attributes.get("beta") == 0)
    ]
    operands = [(role, shape[1:] if len(shape) == 3 and shape[0] == 1 else shape) for role, shape in operands]
    inputs = [shape for role, shape in operands if role == "input"]
    weights = [shape for role, shape in operands if role == "weight"]
    if len(inputs) != 1 or len(weights) != 1:
        return None
    lhs, rhs = inputs[0], weights[0]
    if len(lhs) < 2 or len(rhs) < 2:
        return None
    # Capsule linear sources store KN; ATen linear stores NK. Resolve the
    # rectangular case from K, and preserve the explicit ATen orientation.
    if kind == "linear" and (str(operation["op"]).startswith("aten.") or rhs[-2] != lhs[-1]):
        rhs = [*rhs[:-2], rhs[-1], rhs[-2]]
        operands = [(role, rhs if role == "weight" else shape) for role, shape in operands]
    if lhs[-1] != rhs[-2]:
        return None
    return "matmul", operands


def _residual_state_operands(operation, named):
    """Every contraction of a residual-state capsule: its main one and its independent ``prelude``.

    Each contraction is resolved by the input names its attributes declare and normalized exactly
    like a single-contraction capsule. All inputs must be claimed by exactly one contraction;
    anything else (a missing, repeated or unclaimed operand, a non-contraction) has no identity.
    """
    attributes = operation.get("attributes")
    prelude = attributes.get("prelude") if isinstance(attributes, dict) else None
    if not isinstance(prelude, dict):
        return None
    shapes = {}
    for name, role, shape in named:
        if not isinstance(name, str) or not name or name in shapes:
            return None
        shapes[name] = (role, shape)
    claimed = []
    resolved = []
    # The main contraction is the capsule's own operation (a matmul names its activation ``lhs``, a
    # convolution ``ifm``); the prelude is always an independent matmul.
    main_kind = _contraction_kind(operation["op"])
    for source, op, lhs_key in (
        (attributes, operation, "ifm" if main_kind == "conv2d" else "lhs"),
        (prelude, {"op": "matmul"}, "lhs"),
    ):
        names = [source.get(lhs_key), source.get("weight")]
        if any(not isinstance(name, str) or name not in shapes for name in names) or names[0] == names[1]:
            return None
        claimed.extend(names)
        canonical = _canonical_operands(op, [shapes[name] for name in names])
        if canonical is None or canonical[0] != (main_kind if op is operation else "matmul"):
            return None
        resolved.extend(canonical[1])
    if sorted(claimed) != sorted(shapes):
        return None
    return resolved


def operand_extents(document, *, residual_preludes=True) -> tuple[tuple[int, ...], ...] | None:
    """An operation's operands reduced to sorted non-unit extents, independent of spelling.

    After contraction normalization, neither the operation name, operand roles, tensor
    names, dimension order, unit dimensions, dtype nor numerical policy is kept: a held-out
    layer keeps its identity under any spelling, transposition or batch-1 wrapping.
    Missing roles or unresolved shapes have no established identity (``None``).

    A residual-state capsule (``attributes.prelude``) is identified by all of its contractions, so
    a held-out core in any one of them refuses it. ``residual_preludes=False`` keeps such a
    declaration unidentified (the held-out side stays strict).
    """
    operation = document.get("operation") if isinstance(document, dict) else None
    inputs = document.get("inputs") if isinstance(document, dict) else None
    if (
        not isinstance(operation, dict)
        or not isinstance(operation.get("op"), str)
        or not operation["op"]
        or not isinstance(inputs, list)
        or not inputs
    ):
        return None
    operands = []
    named = []
    for item in inputs:
        if not isinstance(item, dict) or not isinstance(item.get("role"), str) or not item["role"]:
            return None
        shape = item.get("shape")
        if not isinstance(shape, list) or not shape or any(type(size) is not int or size <= 0 for size in shape):
            return None
        operands.append((item["role"], shape))
        named.append((item.get("name"), item["role"], shape))
    attributes = operation.get("attributes")
    if isinstance(attributes, dict) and attributes.get("prelude") is not None:
        resolved = _residual_state_operands(operation, named) if residual_preludes else None
        if resolved is None:
            return None
        return tuple(sorted(tuple(sorted(size for size in shape if size != 1)) for _, shape in resolved))
    canonical = _canonical_operands(operation, operands)
    if canonical is None:
        return None
    return tuple(sorted(tuple(sorted(size for size in shape if size != 1)) for _, shape in canonical[1]))


def _core(extents):
    """The held-out identity: its multi-dimensional operands (bias vectors and scalars are optional)."""
    tensors = tuple(extent for extent in extents if len(extent) >= 2)
    return Counter(tensors or extents)


@lru_cache(maxsize=512)
def _excluded_extents(raw):
    # Cache only parsed immutable bytes; every boundary call reopens each file.
    return operand_extents(yaml.safe_load(raw), residual_preludes=False)


def _excluded_cores(experiment) -> list[Counter]:
    """Operand cores of every held-out capsule; exclusion binds to content, not mutable names."""
    source_root = getattr(experiment, "source_root", None)
    if not getattr(experiment, "held_out_inputs", ()):
        return []
    if source_root is None:
        from merlin.common.paths import repo_root

        source_root = repo_root()
    cores = []
    for relative in experiment.held_out_inputs:
        for path in (Path(source_root) / relative).rglob("capsule.yaml"):
            extents = _excluded_extents(path.read_bytes())
            if extents is None:
                raise SpecError("excluded historical capsule declaration is malformed")
            cores.append(_core(extents))
    return cores


def require_public_capsules(experiment, roots) -> None:
    """Inspect only selected public capsule metadata, never hidden source tensors."""
    held_out = held_out_models(experiment)
    excluded = _excluded_cores(experiment)
    for root in roots:
        for path in sorted(Path(root).rglob("capsule.yaml")):
            document = yaml.safe_load(path.read_bytes())
            if not isinstance(document, dict):
                raise SpecError("public capsule declaration is malformed")
            extents = operand_extents(document)
            if excluded and extents is None:
                raise SpecError("public capsule cannot be identified against the held-out inputs")
            # Extra operands (a full-output bias, a residual) cannot hide a held-out core.
            if extents is not None and any(not core - Counter(extents) for core in excluded):
                raise SpecError(
                    "public corpus contains an explicitly excluded historical input; use generated-only preparation"
                )
            operation = document.get("operation") or {}
            if not isinstance(operation, dict):
                raise SpecError("public capsule operation declaration is malformed")
            attributes = operation.get("attributes") or {}
            capture = document.get("materialized_capture") or {}
            reference = document.get("pytorch_ref") or {}
            if any(not isinstance(value, dict) for value in (attributes, capture, reference)):
                raise SpecError("public capsule source declaration is malformed")
            identities = [
                document.get("model"),
                attributes.get("model"),
                reference.get("model"),
                capture.get("workload_id"),
                capture.get("source_workload_id"),
            ]
            rows = [document, *({**document, "model": identity} for identity in identities if identity)]
            try:
                assert_no_claim_capsules(rows, held_out)
            except ValueError:
                raise SpecError(
                    "public corpus contains a held-out validation source; prepare a generated-only release"
                ) from None


#: Labels of the cohort an author's compiler is graded on in the open.
PUBLIC_LABELS = frozenset({"public", "dev"})


def require_accelerator_placement(roots) -> int:
    """Refuse any public capsule whose frozen Phase 0 placement puts nothing on the accelerator.

    Host placement is compiled by Merlin's own generic host codegen, so such a capsule would grade none
    of the author's work. The placement read is the one Phase 0 froze into ``software_screen``; a
    capsule without that record is refused as well, because its absence is not evidence of accelerator
    work. An unscored host guard is admitted: it is host-placed by design and grades no accelerator
    work. A composition refusal twin is admitted when a positive of its chain and scheme in the same roots is
    placed on the accelerator. Returns the number of public capsules checked.
    """
    from merlin.targetgen.capsule_common import is_unscored_host_guard

    from ..phase0.composition_generation import positive_is_placed, twin_group
    from ..phase0.program_admission import accelerator_placement_refusal

    documents = []
    for root in roots:
        for path in sorted(Path(root).rglob("capsule.yaml")):
            document = yaml.safe_load(path.read_bytes())
            if not isinstance(document, dict):
                raise SpecError("public capsule declaration is malformed")
            if document.get("label") in PUBLIC_LABELS:
                documents.append((path, document))
    # A refusal twin is graded on being refused, so it is admitted when a positive of its chain and scheme
    # in the same public cohort is placed on the accelerator; a twin with no such sibling stays refused.
    placed_groups = {
        twin_group(document)
        for _, document in documents
        if positive_is_placed(document, document.get("software_screen"))
    }
    refused, checked = [], 0
    for path, document in documents:
        checked += 1
        if is_unscored_host_guard(document):
            continue
        reason = accelerator_placement_refusal(document.get("software_screen"))
        if reason is not None and (
            document.get("operation", {}).get("op") == "composition_program"
            and ((document.get("semantic") or {}).get("composition_truth") or {}).get("outcome") == "refuse"
            and twin_group(document) in placed_groups
        ):
            reason = None
        if reason is not None:
            refused.append(f"{document.get('name') or path.parent.name}: {reason}")
    if refused:
        raise SpecError(
            f"{len(refused)} public capsule(s) carry no accelerator placement; public capsules grade the "
            "author's accelerator compiler only, and host placement is Merlin-supplied: " + "; ".join(refused)
        )
    return checked
