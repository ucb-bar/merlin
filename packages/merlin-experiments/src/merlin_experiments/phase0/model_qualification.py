"""Qualify an integer whole-model capsule by the isolated verification of every device group it holds.

A source-backed integer model capsule relies on arithmetic it cannot bound itself: its contractions'
operands are internal tensors, not the capsule's inputs, so the writer's partial-sum screen has
nothing concrete to check, and the model is refused ("lacks a verified isolated i8 matmul"). That
refusal stands unless the model's arithmetic HAS been verified in isolation for this target:

1. every device group the model forms -- through the one grouping (``compute_groups`` +
   ``group_command``, stated by ``group_capsule_entries.entries``) -- is covered by a per-group
   interface capsule built under the corpus binding by the Phase 0 writer, with its golden. An
   integer contraction also needs a proven internal partial-sum bound; a residual add has no MAC
   and instead needs a non-vacuous bounded-integer golden sampled from the full operand range. A group the
   vocabulary cannot state is uncovered;
2. where the experiment declares a whole-model reference arm for the model, its gate result records
   zero mismatching dispatches.

The group capsules are ordinary corpus members, so Phase 1 grades the very programs that qualified
the model. The evidence is recorded on the model capsule (``model_qualification``). Missing or failed
evidence refuses the capsule with the unchanged message. Nothing here names a target or a model.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA = "merlin.phase0.model_qualification.v1"
#: The refusal the writer has always issued for an unverified source-backed integer model.
REFUSAL = "source-backed integer contraction lacks a verified isolated i8 matmul and exact inputs"
_IDENTITY_KEY = "group_identity_sha256"


class QualificationMissing(ValueError):
    """Why a model capsule's isolated verification is absent or failed (chained under the refusal)."""


def _identity(entry: Mapping[str, Any]) -> str:
    # The command-stream identity deliberately drops multiplier *values* so
    # scheduling-equivalent groups can share a form. A numerical golden cannot:
    # two residual adds with identical shapes and different scales have different
    # answers. Bind every arithmetic field and the selected semantics here.
    presentation = {"name", "cat", "kind", "label", "source_role", "source_reference", "comment"}
    numerical = {key: value for key, value in entry.items() if key not in presentation}
    return hashlib.sha256(json.dumps(numerical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _weight_args(directory: Path, cap: Mapping[str, Any]):
    from merlin.xdsl_dialects.lowering import stream_plan

    attrs = (cap.get("operation") or {}).get("attributes") or {}
    weights = attrs.get("weights") if isinstance(attrs.get("weights"), Mapping) else {}
    name = weights.get("manifest") or attrs.get("weights_manifest")
    candidates = [directory / str(name)] if name else []
    candidates += sorted(directory.glob("*.manifest.json"))
    for path in candidates:
        if path.is_file():
            return stream_plan.weight_args_of(json.loads(path.read_text(encoding="utf-8")))
    return None


def reference_gate_verdict(declaration: Any) -> dict[str, Any]:
    """The declared whole-model reference gate result, read and judged: zero mismatches or refused.

    ``declaration`` is ``None`` (no reference arm declared for this model) or a path to the gate's
    result JSON. Accepted shapes are the gate's per-dispatch summary (``words.dispatches`` > 0 and
    ``words.differ`` == 0) or an explicit ``mismatches`` == 0 over ``of`` > 0 elements.
    """
    if declaration is None:
        return {"status": "not_declared"}
    path = Path(str(declaration))
    if path.is_symlink() or not path.is_file():
        raise QualificationMissing(f"the declared reference gate result {path} is absent")
    raw = path.read_bytes()
    document = json.loads(raw)
    words = document.get("words") if isinstance(document.get("words"), Mapping) else None
    if words is not None:
        dispatches, differ = words.get("dispatches"), words.get("differ")
    else:
        dispatches, differ = document.get("of"), document.get("mismatches")
    if type(dispatches) is not int or type(differ) is not int or dispatches <= 0:
        raise QualificationMissing("the declared reference gate result records no counted dispatches")
    if differ != 0:
        raise QualificationMissing(f"the reference gate records {differ} of {dispatches} mismatching dispatches")
    return {
        "status": "exact",
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "dispatches": dispatches,
        "mismatches": 0,
    }


def _covered(written: Path, identity: str, expected_cap: dict, expected_program: str) -> dict[str, Any]:
    """The evidence one written group capsule carries, or why it does not qualify."""
    import yaml

    capsule = yaml.safe_load((written / "capsule.yaml").read_text(encoding="utf-8")) or {}
    if any(capsule.get(key) != expected_cap.get(key) for key in ("operation", "inputs", "numeric_policy")):
        raise QualificationMissing(f"group capsule {written.name} differs from the requested numerical program")
    if (written / "capsule.interface.mlir").read_text(encoding="utf-8") != expected_program:
        raise QualificationMissing(f"group capsule {written.name} differs from the requested interface program")
    from merlin.targetgen import golden_store as GS

    golden_path = written / GS.DOCUMENT
    if not golden_path.is_file():
        raise QualificationMissing(f"group capsule {written.name} has no golden")
    # The document's digest below binds the archived arrays: it names each one's SHA-256.
    golden = GS.load_golden(written) or {}
    bound = golden.get("integer_partial_sum_bound") or capsule.get("integer_partial_sum_bound") or {}
    operation = (capsule.get("operation") or {}).get("op")
    if operation == "residual_add":
        # A residual add performs no multiply-accumulate. Treating its honest
        # ``not_applicable`` MAC bound as a failure excluded source models that
        # contain a perfectly materialized add group. It still needs an exact
        # integer reference with a declared, falsifiable output-step bound.
        from merlin.common import quant_formats

        attrs = (capsule.get("operation") or {}).get("attributes") or {}
        policy = capsule.get("numeric_policy") or {}
        falsifiability = capsule.get("numeric_falsifiability") or {}
        dtype = attrs.get("output_dtype")
        try:
            fmt = quant_formats.get(dtype)
        except (KeyError, ValueError, TypeError) as exc:
            raise QualificationMissing(f"group capsule {written.name} has no known integer output format") from exc
        full_range = [-(1 << (fmt.element_bits - 1)), (1 << (fmt.element_bits - 1)) - 1]
        if (
            fmt.kind != "int_affine"
            or bound.get("status") != "not_applicable"
            or bound.get("reason") != "this writer path has no modeled integer contraction"
            or golden.get("golden_source") != "merlin_tensor_int"
            or policy.get("compare") != "bounded_int"
            or policy.get("dtype") != dtype
            or type(attrs.get("bound_lsb")) is not int
            or attrs["bound_lsb"] < 0
            or policy.get("atol") != attrs["bound_lsb"]
            or policy.get("rtol") != 0
            or capsule.get("stimulus_range") != full_range
            or falsifiability.get("status") != "ok"
            or falsifiability.get("falsifiable") is not True
        ):
            raise QualificationMissing(
                f"group capsule {written.name} lacks a full-domain sampled, falsifiable residual-add integer reference"
            )
        arithmetic = {"kind": "bounded_integer_residual_add", "output_step_bound": attrs["bound_lsb"]}
    else:
        from .writer import INTEGER_CONTRACTION_BOUND_OPS

        if operation not in INTEGER_CONTRACTION_BOUND_OPS:
            raise QualificationMissing(f"group capsule {written.name} has no recognized integer arithmetic proof")
        if bound.get("status") != "proven_safe":
            raise QualificationMissing(
                f"group capsule {written.name} has no proven internal partial-sum bound ({bound.get('status')})"
            )
        arithmetic = {
            "kind": "proven_integer_partial_sums",
            "partial_sum_bound": bound.get("bound"),
            "mac_result_bits": bound.get("mac_result_bits"),
        }
    if capsule.get(_IDENTITY_KEY) != identity:
        raise QualificationMissing(f"group capsule {written.name} states a different group")
    result = {
        "capsule": f"{written.parent.name}/{written.name}",
        "golden_sha256": hashlib.sha256(golden_path.read_bytes()).hexdigest(),
        "arithmetic_qualification": arithmetic,
    }
    if arithmetic["kind"] == "proven_integer_partial_sums":
        result.update(partial_sum_bound=bound.get("bound"), mac_result_bits=bound.get("mac_result_bits"))
    return result


def qualify(entry: Mapping[str, Any], cap: Mapping[str, Any], directory: Path, binding, out_root: Path) -> dict:
    """The qualification record for one integer model capsule, or the unchanged refusal."""
    try:
        return _qualify(entry, cap, Path(directory), binding, Path(out_root))
    except QualificationMissing as missing:
        raise ValueError(REFUSAL) from missing


def _qualify(entry, cap, directory: Path, binding, out_root: Path) -> dict:
    import yaml

    from merlin.common import mlir_query as mq
    from merlin.targetgen import group_capsule_entries as G

    from .group_forms import write_group_capsule

    if "source_integer_proof" in entry:
        from .connected_integer_qualification import qualify as connected_qualify

        try:
            return connected_qualify(entry, cap, directory)
        except ValueError as error:
            raise QualificationMissing(str(error)) from error

    if (entry.get("numerical_semantics") or {}).get("internal_arithmetic") is None:
        raise QualificationMissing("no selected internal arithmetic to bound a group against")
    interface = directory / str(cap.get("interface_mlir") or "capsule.interface.mlir")
    if not interface.is_file():
        raise QualificationMissing("the model capsule carries no interface program to group")
    model = str(entry.get("name") or directory.name)
    stated = G.entries(
        binding.target,
        mq.parse(str(interface)),
        weight_args=_weight_args(directory, cap),
        model=model,
        with_raw=False,
        numerical_variants=True,
    )
    if stated.get("unstated"):
        raise QualificationMissing(f"device groups the capsule vocabulary cannot state: {stated['unstated']}")
    if not stated.get("entries") or not stated.get("accelerator_groups"):
        raise QualificationMissing("the model forms no device group whose arithmetic could be verified")
    if stated.get("stated") != stated.get("accelerator_groups"):
        raise QualificationMissing("not every device group of the model is stated")
    reference = reference_gate_verdict(entry.get("reference_gate"))
    groups = []
    for row in stated["entries"]:
        group_entry = {
            **dict(row["entry"]),
            # Public: Phase 1 grades the very programs that qualified the model.
            "label": "public",
            "numerical_semantics": entry["numerical_semantics"],
            "source_reference": f"isolated verification of a device group of {model}",
        }
        identity = _identity(group_entry)
        target_dir = out_root / str(group_entry.get("cat") or "layers") / str(group_entry["name"])
        if (target_dir / "capsule.yaml").is_file():
            existing = yaml.safe_load((target_dir / "capsule.yaml").read_text(encoding="utf-8")) or {}
            if existing.get(_IDENTITY_KEY) != identity:
                # Keep form-equivalent but numerically distinct groups separate.
                # The suffix is derived from the full arithmetic identity, never
                # from model names or target-specific heuristics.
                group_entry["name"] = f"{group_entry['name']}_q{identity[:12]}"
                target_dir = out_root / str(group_entry.get("cat") or "layers") / str(group_entry["name"])
                if (target_dir / "capsule.yaml").is_file():
                    collided = yaml.safe_load((target_dir / "capsule.yaml").read_text(encoding="utf-8")) or {}
                    if collided.get(_IDENTITY_KEY) != identity:
                        raise QualificationMissing(f"numerical group identity collision at {target_dir.name}")
        expected_cap, expected_program = G.interface_capsule(group_entry, binding)
        if (target_dir / "capsule.yaml").is_file():
            written = target_dir  # shared with another model: the identity check below decides
        else:
            try:
                written = write_group_capsule(group_entry, binding, out_root)
            except Exception as error:  # noqa: BLE001 -- a group the writer refuses leaves it uncovered
                raise QualificationMissing(f"group {row['name']} was refused: {error}") from error
            if written is None:
                raise QualificationMissing(f"group {row['name']} produced no capsule")
            capsule = yaml.safe_load((written / "capsule.yaml").read_text(encoding="utf-8")) or {}
            capsule[_IDENTITY_KEY] = identity
            (written / "capsule.yaml").write_text(yaml.safe_dump(capsule, sort_keys=False), encoding="utf-8")
        groups.append(
            {
                "groups": list(row["groups"]),
                "count": row["count"],
                **_covered(written, identity, expected_cap, expected_program),
            }
        )
    return {
        "schema": SCHEMA,
        "status": "qualified",
        "grouping": "merlin.targetgen.group_capsule_entries.entries",
        "device_groups": stated["accelerator_groups"],
        "covered_groups": stated["stated"],
        "group_capsules": groups,
        "reference_gate": reference,
        "scope": "every device group verified in isolation under the corpus binding; full-model execution unverified",
    }
