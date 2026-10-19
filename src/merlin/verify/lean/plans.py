"""Authored boundary rosters, resolved before execution and independent of observed records.

A matching roster establishes recording completeness at its stated granularity. Semantic checking
and byte-bound chain replay remain separate requirements; a plan never grants a pass permission.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from . import stages


class PlanMismatch(ValueError):
    """The capture does not contain exactly the independently expected ordered occurrences."""


@dataclass(frozen=True)
class Boundary:
    name: str
    pass_entry: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name
            or not isinstance(self.pass_entry, str)
            or not self.pass_entry
        ):
            raise ValueError("a planned boundary requires a name and pass entry")


@dataclass(frozen=True)
class Plan:
    scope: str
    boundaries: tuple[Boundary, ...]
    granularity: str = "individual-boundary-occurrences"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.scope, str)
            or not self.scope
            or not isinstance(self.granularity, str)
            or not self.granularity
        ):
            raise ValueError("a boundary plan requires its scope and granularity")
        if type(self.boundaries) is not tuple or not self.boundaries:
            raise ValueError("a boundary plan requires an immutable nonempty tuple")
        if any(type(b) is not Boundary for b in self.boundaries):
            raise ValueError("a boundary plan requires Boundary occurrences")

    def canonical_bytes(self) -> bytes:
        payload = {
            "schema": "merlin.lean.boundary_plan.v1",
            "scope": self.scope,
            "granularity": self.granularity,
            "boundaries": [{"boundary": b.name, "pass": b.pass_entry} for b in self.boundaries],
        }
        return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def validate(self, published: list[Path]) -> None:
        """Require every occurrence, including unchanged ones, with exact order and pass identity."""
        if len(published) != len(self.boundaries):
            raise PlanMismatch(f"expected {len(self.boundaries)} boundary occurrences, recorded {len(published)}")
        for ordinal, (path, expected) in enumerate(zip(published, self.boundaries, strict=True)):
            meta = json.loads((path / "meta.json").read_bytes())
            if path.name != f"{ordinal:04d}_{expected.name}":
                raise PlanMismatch(f"boundary occurrence {ordinal}: unexpected directory {path.name}")
            if (meta.get("boundary"), meta.get("pass")) != (expected.name, expected.pass_entry):
                raise PlanMismatch(f"boundary occurrence {ordinal}: expected {expected.name} / {expected.pass_entry}")


def replay(expected: Plan, manifest: Path, root: Path) -> None:
    """Check a retained roster against a caller's trusted expected plan, then check actual records.

    Loading expectations from the recorded boundaries, or trusting a recorded plan hash alone,
    would let an omitted boundary define its own denominator. The caller supplies the independently
    resolved plan. This replays only the roster obligation, not the semantic chain or source proofs.
    """
    if manifest.read_bytes() != expected.canonical_bytes():
        raise PlanMismatch("retained boundary plan differs from the independently expected plan")
    published = stages.completed(root)
    reserved = root / ".reserved"
    if not reserved.is_dir() or sorted(p.name for p in reserved.iterdir()) != [
        f"{n:06d}" for n in range(len(published))
    ]:
        raise PlanMismatch("recording has incomplete or unexpected reservations")
    published_set = set(published)
    if any(p.name.startswith(".writing-") for p in root.iterdir()) or any(
        p.is_dir() and not p.name.startswith(".") and p not in published_set for p in root.iterdir()
    ):
        raise PlanMismatch("recording has unpublished boundary directories")
    expected.validate(published)


# Independently authored expectations, including the parse occurrence. Production
# edits must not silently change the denominator of a verified shared prefix.
# Statistic identities also matter: they select the optional quantization anchor.
_ORDINARY_PREPROCESSING = (
    (None, Boundary("xdsl-parse", "merlin.llvmlower.passes_xdsl.parse_mlir_text")),
    (
        "dead_tensor_ops_pruned",
        Boundary("xdsl-pruned", "merlin.llvmlower.passes_xdsl.prune_dead_pure_tensor_ops"),
    ),
    ("quant_ext_lowered", Boundary("xdsl-quant-lowered", "merlin.llvmlower.passes_xdsl.lower_quant_ext")),
    ("c_interface_funcs", Boundary("xdsl-c-interface", "merlin.llvmlower.passes_xdsl.add_c_interface")),
)


def preprocessing() -> Plan:
    """Resolve the structural preprocessing roster before running its parser or transformations.

    The independent roster supplies one occurrence per slot, even when it changes no bytes.
    The actual production table must match it before any parser or pass can run.
    The optional reassociation occurrence is included exactly where preprocessing selects it. A
    changed selection during execution causes a roster mismatch rather than silently widening scope.
    """
    from merlin.llvmlower import passes_xdsl

    selected = passes_xdsl._layer_norm_chunked_sums_selected()
    actual = tuple(passes_xdsl._PREPROCESS)
    if selected and sum(statistic == "quant_ext_lowered" for statistic, _, _ in actual) != 1:
        raise ValueError("selected chunked sums requires exactly one quantization preprocessing slot")
    prefix = passes_xdsl.__name__
    observed = (
        (None, Boundary("xdsl-parse", f"{prefix}.parse_mlir_text")),
        *((statistic, Boundary(stage, f"{prefix}.{name}")) for statistic, stage, name in actual),
    )
    if observed != _ORDINARY_PREPROCESSING:
        raise PlanMismatch("production preprocessing differs from the independently authored ordinary roster")
    boundaries = []
    for statistic, boundary in _ORDINARY_PREPROCESSING:
        boundaries.append(boundary)
        if statistic == "quant_ext_lowered" and selected:
            boundaries.append(
                Boundary(
                    "xdsl-layer-norm-chunked-sums",
                    "merlin.llvmlower.normalization_reassociation.chunk_layer_norm_sums",
                )
            )
    return Plan("lower_model structural preprocessing only", tuple(boundaries), "individual-preprocessing-boundaries")


def preparation(*, int8_compute: bool = False, routing: bool = False) -> Plan:
    """Authored ordinary shared-preparation hook roster (no selected optional mutations).

    Configuration admission is performed before entry by ``preparation.capture``. The final slot
    records the whole remaining preparation tail against the actual returned file; it grants no
    semantic permission to an ABI or numerical repair.
    """
    lower = "merlin.llvmlower"
    slots = (
        ("prepare-parse", "merlin.frontends.linalg_mlir.parse_mlir_file"),
        ("prepare-torchao-affine", f"{lower}.torchao_affine.lower_torchao_affine_quant"),
        ("prepare-scalar-squares", f"{lower}.scalar_square.canonicalize_scalar_squares"),
        ("prepare-collapse-overrank-matmul", f"{lower}.passes_xdsl.collapse_overrank_matmul"),
        ("prepare-propagate-quant-inner", "merlin.runtime.dispatch_runtime._propagate_quant_inner"),
        ("prepare-qinner-lift", f"{lower}.qinner.lift"),
        ("prepare-lower-quant-ext", f"{lower}.passes_xdsl.lower_quant_ext"),
        ("prepare-bf16-matmul-f32acc", f"{lower}.passes_xdsl.lower_bf16_matmul_f32acc"),
        ("prepare-fix-bool-sitofp", f"{lower}.passes_xdsl.fix_bool_sitofp"),
        ("prepare-fix-bool-fptosi", f"{lower}.passes_xdsl.fix_bool_fptosi"),
        ("prepare-scalarize-splat-inputs", f"{lower}.splat_inputs.scalarize_splat_inputs"),
        ("prepare-print", "merlin.xdsl_dialects._common.text"),
        ("prepare-finish", "merlin.runtime.backends.zephyr_model.prepare_for_lowering"),
    )
    if int8_compute:
        index = next(i for i, slot in enumerate(slots) if slot[0] == "prepare-lower-quant-ext")
        slots = (*slots[:index], ("prepare-quantize-int8", f"{lower}.quant_passes.apply_quant"), *slots[index:])
    if routing:
        slots = (
            *slots[:-1],
            ("prepare-routing-prefix", "merlin.runtime.backends.zephyr_model.prepare_for_lowering"),
            ("prepare-device-routing", f"{lower}.device_offload.rewrite_prepared_file"),
            slots[-1],
        )
    return Plan(
        (
            "shared preparation with explicit assumed device routing; no optional mutations"
            if routing
            else "ordinary shared preparation only; no optional mutations or provider routing"
        ),
        tuple(Boundary(*slot) for slot in slots),
        "recorded-preparation-hooks-and-composite-tail",
    )
