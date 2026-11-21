"""Bind an existing compiler-only stage to public compilation and discovery.

These are source observations, not a numerical admission or a reference grant.
Only the closed staged roster is read; capture/evaluator directories are absent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path

from merlin.compiler.semantic.model import KernelRequest

# ---- staged ------------------------------------------------------------------------------------


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _snapshot(directory):
    from merlin.compiler.compilation import digest
    from merlin.compiler.frontend.inputs import _read_member, load_compile_inputs

    if directory.is_symlink() or any(parent.is_symlink() for parent in directory.parents) or not directory.is_dir():
        raise ValueError("compiler input directory must be real and present")
    manifest = _read_member(directory, "compile-inputs.json")
    staged = load_compile_inputs(directory)
    if staged.precision_policy_document is None:
        raise ValueError("staged program requires an explicit source-bound precision policy")
    # The loader has already checked the exact regular-file roster, every
    # signature/initializer and policy-to-captured-source correspondence.
    members = tuple((path.name, digest(path)) for path in sorted(directory.iterdir()))
    if _read_member(directory, "compile-inputs.json") != manifest:
        raise ValueError("compiler stage changed during selection")
    return staged, manifest, members


@dataclass(frozen=True)
class StagedProgram:
    """Select existing compiler-only inputs, never a model/evaluator bundle.

    Construction validates and freezes the actual stage. Numerical meaning must
    still be admitted by the original caller through the selected compiler.
    ``freeze_readonly_buffers`` explicitly chooses the freshly observed BUFFER
    bytes for every compiled invocation; later source-state updates are outside
    this snapshot contract. It grants no mutation, alias or accuracy permission.
    """

    directory: Path
    freeze_readonly_buffers: bool = False
    _selected_directory: Path = field(init=False, repr=False)
    _selected_freeze: bool = field(init=False, repr=False)
    _manifest: bytes = field(init=False, repr=False)
    _members: tuple = field(init=False, repr=False)

    def __post_init__(self):
        if type(self.freeze_readonly_buffers) is not bool:
            raise ValueError("freeze_readonly_buffers requires an explicit Boolean")
        root = Path(self.directory).absolute()
        _, manifest, members = _snapshot(root)
        object.__setattr__(self, "directory", root)
        object.__setattr__(self, "_selected_directory", root)
        object.__setattr__(self, "_selected_freeze", self.freeze_readonly_buffers)
        object.__setattr__(self, "_manifest", manifest)
        object.__setattr__(self, "_members", members)

    def revalidate(self):
        if self.freeze_readonly_buffers is not self._selected_freeze:
            raise ValueError("original buffer freeze consent changed after selection")
        if self.directory != self._selected_directory:
            raise ValueError("original compiler stage directory changed after selection")
        staged, manifest, members = _snapshot(self.directory)
        if manifest != self._manifest or members != self._members:
            raise ValueError("original compiler stage or precision policy changed after selection")
        return staged


@dataclass(frozen=True)
class StagedPreparation:
    """Actual stage/policy/bound-source identities; no semantic authorization."""

    stage: StagedProgram
    bound: object = field(repr=False)
    _record: str = field(init=False, repr=False)
    _policy: bytes = field(init=False, repr=False)
    _owners: tuple = field(init=False, repr=False)

    def __post_init__(self):
        from merlin.compiler.frontend.bindings import BoundCapture

        if type(self.stage) is not StagedProgram or type(self.bound) is not BoundCapture:
            raise ValueError("preparation requires the original typed stage and immutable binding")
        staged = self.stage.revalidate()
        self.bound.revalidate(self.stage.directory, staged)
        record = {
            "schema": "merlin.staged-preparation.v1",
            "directory": str(self.stage.directory),
            "preparation_sha256": _sha(self.stage._manifest),
            "preparation_member": "compile-inputs.json",
            "numerical_policy_sha256": staged.precision_policy_sha256,
            "captured_source_sha256": staged.original_source_sha256,
            "staged_source_sha256": staged.program_sha256,
            "bound_source_sha256": self.bound.record["bound_program_sha256"],
            "members": dict(self.stage._members),
            "scope": "compiler staging/preparation identity only; numerical admission and calibration/reference "
            "qualification are independent",
        }
        if self.stage.freeze_readonly_buffers:
            record["buffer_freeze"] = self.bound.record["buffer_freeze"]
        if "quantization_legalization" in self.bound.record:
            record["quantization_legalization"] = self.bound.record["quantization_legalization"]
        object.__setattr__(self, "_record", json.dumps(record, sort_keys=True, allow_nan=False))
        object.__setattr__(self, "_policy", staged.precision_policy_document)
        object.__setattr__(self, "_owners", (self.stage, self.bound))

    def record(self):
        return json.loads(self._record)

    def policy(self):
        """Return selected policy data for original-caller admission, never a verdict."""
        return json.loads(self._policy)

    def revalidate_ir(self, source):
        """Bind normalized discovery IR to the actual immutable argument substitution."""
        from merlin.compiler.frontend.linalg_mlir import parse_mlir_text

        self.revalidate()
        source.revalidate()
        if not source.module.is_structurally_equivalent(parse_mlir_text(self.bound.program)):
            raise ValueError("discovery source differs from its original staged preparation")

    def revalidate(self, source=None):
        if self.stage is not self._owners[0] or self.bound is not self._owners[1]:
            raise ValueError("original staged preparation owners changed")
        staged = self.stage.revalidate()
        self.bound.revalidate(self.stage.directory, staged)
        record = self.record()
        if (
            self.stage.freeze_readonly_buffers is not ("buffer_freeze" in record)
            or str(self.stage.directory) != record["directory"]
            or staged.precision_policy_document != self._policy
            or staged.precision_policy_sha256 != record["numerical_policy_sha256"]
            or _sha(self.stage._manifest) != record["preparation_sha256"]
            or _sha(self.bound.program.encode()) != record["bound_source_sha256"]
        ):
            raise ValueError("original staged preparation identity changed")
        if source is not None and (source.kind != "mlir" or source.text != self.bound.program):
            raise ValueError("prepared request differs from its original bound source")


def prepare(program: StagedProgram, *, max_initialization_bytes):
    """Bind only existing admitted immutable parameters; perform no quantization."""
    from merlin.compiler.compilation import Source
    from merlin.compiler.frontend.bindings import bind_lifted_arguments

    if type(program) is not StagedProgram:
        raise ValueError("staged preparation requires an explicit StagedProgram")
    staged = program.revalidate()
    bound = bind_lifted_arguments(
        program.directory,
        staged,
        max_initialization_bytes=max_initialization_bytes,
        freeze_readonly_buffers=program.freeze_readonly_buffers,
    )
    binding = StagedPreparation(program, bound)
    source = Source(bound.program, origin=str(program.directory / "program.mlir"))
    binding.revalidate(source)
    return source, binding


# ---- native ------------------------------------------------------------------------------------
# Compiler-only native request intake with exact staged source binding.


def _load_native_request(args: argparse.Namespace, *, target_identity: str, mode: str):
    """Read compiler inputs only; bind parsed Linalg to the selected target."""
    from merlin.compiler.frontend.inputs import load_compile_inputs, load_lifted_argument_bytes
    from merlin.compiler.semantic.linalg.linalg_bridge import LinalgBridgeError, translate_linalg_text
    from merlin.compiler.semantic.model import KernelRequest

    if getattr(args, "compile_inputs", None):
        if args.linalg_entry:
            raise ValueError("--linalg-entry cannot override a staged capture signature")
        staged = load_compile_inputs(args.compile_inputs)
        if staged.precision_policy is not None:
            if "matmul_regions" in staged.precision_policy:
                from merlin.compiler.quantization.contraction.intake import check_source_bound_f32_matmuls

                check_source_bound_f32_matmuls(
                    staged.program,
                    entry=staged.entry,
                    policy=staged.precision_policy,
                    original_source_sha256=staged.original_source_sha256,
                )
            raise LinalgBridgeError("selected precision policy has no admitted native transformation")
        translated = translate_linalg_text(
            staged.program,
            entry=staged.entry,
            target_identity=target_identity,
            lowering_policy=mode,
            argument_constants=load_lifted_argument_bytes(args.compile_inputs, staged),
        )
        return translated.request, "staged_capture", translated.source_operations
    if args.linalg:
        if not args.linalg_entry:
            raise ValueError("--linalg-entry is required with --linalg")
        source = Path(args.linalg).read_bytes().decode("utf-8")
        translated = translate_linalg_text(
            source,
            entry=args.linalg_entry,
            target_identity=target_identity,
            lowering_policy=mode,
        )
        return translated.request, "parsed_linalg", translated.source_operations
    if args.linalg_entry:
        raise ValueError("--linalg-entry requires --linalg")
    request = KernelRequest.from_record(json.loads(Path(args.request).read_text()))
    return request, "typed_request", ()


def load_selected_request(args, *, target_identity, mode, binding=None, load_request=_load_native_request):
    """Load unchanged inputs or the explicit source-bound diagnostic graph route."""
    from merlin.compiler.frontend.inputs import load_compile_inputs
    from merlin.compiler.quantization.contraction.constants import prepare_graph_request, prepare_tile_request
    from merlin.compiler.semantic.linalg.linalg_bridge import LinalgBridgeError

    graph_policy = getattr(args, "diagnostic_graph_policy", None)
    if graph_policy is None:
        region_id = getattr(args, "diagnostic_region", None)
        tile_index = getattr(args, "diagnostic_tile_index", None)
        whole = getattr(args, "whole_contraction", False)
        if region_id is not None:
            if not getattr(args, "compile_inputs", None) or mode != "diagnostic" or binding is None:
                raise LinalgBridgeError(
                    "source-bound precision tiles require staged compiler inputs, target support, and diagnostic mode"
                )
            if getattr(args, "linalg_entry", None):
                raise ValueError("--linalg-entry cannot override a staged capture signature")
            staged = load_compile_inputs(args.compile_inputs)
            return prepare_tile_request(
                staged,
                region_id,
                binding=binding,
                target_identity=target_identity,
                mode=mode,
                tile_index=tile_index,
                whole_contraction=whole,
            )
        if whole:
            raise LinalgBridgeError("--whole-contraction requires --diagnostic-region")
        if tile_index is not None:
            raise LinalgBridgeError("--diagnostic-tile-index requires --diagnostic-region")
        request, kind, operations = load_request(args, target_identity=target_identity, mode=mode)
        return request, kind, operations, None
    if (
        getattr(args, "whole_contraction", False)
        or getattr(args, "diagnostic_region", None) is not None
        or getattr(args, "diagnostic_tile_index", None) is not None
    ):
        raise LinalgBridgeError("graph precision policy cannot select a contraction or tile")
    if mode != "diagnostic" or binding is None:
        raise LinalgBridgeError("graph precision policy requires diagnostic mode and target support")
    if not callable(getattr(binding, "prepare_precision_graph", None)):
        raise LinalgBridgeError("selected target support has no graph precision lowering")
    policy = json.loads(Path(graph_policy).read_text())
    staged_graph = bool(getattr(args, "compile_inputs", None))
    original_source_sha256 = None
    if staged_graph:
        staged = load_compile_inputs(args.compile_inputs)
        if staged.precision_policy != policy or args.linalg or args.linalg_entry:
            raise LinalgBridgeError("staged graph precision must match its compiler-input policy and entry")
        source, entry = staged.program, staged.entry
        original_source_sha256 = staged.original_source_sha256
    else:
        if not args.linalg or not args.linalg_entry:
            raise LinalgBridgeError("graph precision policy requires --linalg/--linalg-entry or --compile-inputs")
        source, entry = Path(args.linalg).read_bytes().decode("utf-8"), args.linalg_entry
    request, operations, plan = prepare_graph_request(
        source,
        entry=entry,
        policy=policy,
        binding=binding,
        target_identity=target_identity,
        mode=mode,
        original_source_sha256=original_source_sha256,
    )
    kind = "source_bound_diagnostic_staged_graph" if staged_graph else "source_bound_diagnostic_graph"
    return request, kind, operations, plan


# ---- layouts -----------------------------------------------------------------------------------
# Bind an explicit physical boundary representation to an admitted source.
#
# This checks source/request correspondence. Layout legality and encodings remain
# obligations of the selected backend and approved hardware contract.


def bind_physical_layouts(args, request, precision_plan):
    """Restore the versioned native layout document without guessing a layout."""
    path = getattr(args, "physical_layouts", None)
    if path is None:
        return request, None
    if getattr(args, "request", None) or any(
        getattr(args, name, None) is not None
        for name in ("ordered_k_tile", "ordered_m_tile", "ordered_n_tile", "outer_tile_index")
    ):
        raise ValueError("physical layouts require parsed Linalg without ordered tile transformations")
    row = json.loads(Path(path).read_bytes())
    fields = {"schema", "target_identity", "source_identity", "physical_extents", "boundary_layouts", "output_layouts"}
    schema = row.get("schema") if isinstance(row, dict) else None
    if schema not in {"merlin.physical_boundary_layouts.v1", "merlin.physical_boundary_layouts.v2"} or (
        set(row) != (fields | {"request_digest"} if schema.endswith("v2") else fields)
    ):
        raise ValueError("physical layouts need the complete versioned boundary schema")
    if precision_plan is not None and schema.endswith("v1"):
        raise ValueError("precision-transformed layouts need the exact-request v2 schema")
    if schema.endswith("v2") and row["request_digest"] != request.digest():
        raise ValueError("physical layouts differ from the precision-transformed request")
    if row["target_identity"] != request.target_identity or row["source_identity"] != request.source_identity:
        raise ValueError("physical layouts differ from the parsed source or selected target")
    if (
        not isinstance(row["physical_extents"], dict)
        or not row["physical_extents"]
        or not isinstance(row["boundary_layouts"], dict)
        or not row["boundary_layouts"]
        or not isinstance(row["output_layouts"], list)
        or not row["output_layouts"]
    ):
        raise ValueError("physical layouts need typed extents, input layouts, and ordered output layouts")
    bound = replace(
        request,
        physical_extents=tuple(sorted(row["physical_extents"].items())),
        boundary_layouts=tuple(sorted(row["boundary_layouts"].items())),
        output_layouts=tuple(row["output_layouts"]),
    )
    canonical = {
        **row,
        "physical_extents": dict(bound.physical_extents),
        "boundary_layouts": dict(bound.boundary_layouts),
        "output_layouts": list(bound.output_layouts),
    }
    return bound, (json.dumps(canonical, sort_keys=True, indent=2) + "\n").encode()


def check_layout_artifact(artifact, manifest, document):
    """Require the selected document, or refuse an unrequested representation."""
    name = "physical-boundary-layouts.json"
    path = artifact / name
    tracked = manifest.get("stage_artifacts")
    if document is None:
        if path.exists() or path.is_symlink() or isinstance(tracked, dict) and name in tracked:
            raise ValueError("native artifact has an unrequested physical boundary layout")
    elif (
        path.is_symlink()
        or not path.is_file()
        or path.read_bytes() != document
        or not isinstance(tracked, dict)
        or tracked.get(name) != hashlib.sha256(document).hexdigest()
    ):
        raise ValueError("native physical boundary layout differs from the selected source representation")


# ---- tiling ------------------------------------------------------------------------------------
# Select explicit ordered native request partitions through the chosen backend.
#
# Geometry and numerical admission stay with the selected OOT implementation.
# The shared transformations preserve logical source boundaries and reduction order;
# these records do not qualify execution or hardware support.


@dataclass(frozen=True)
class OrderedNativeRequest:
    """Selected request and the exact two source boundaries needed for replay."""

    source: KernelRequest
    k_source: KernelRequest
    request: KernelRequest
    mn_record: bytes | None
    k_record: bytes | None


def prepare_ordered_request(args, binding, source_request, precision_plan=None):
    """Keep optional outer-M/N selection before ordered reduction partitioning."""
    if precision_plan is not None and any(
        getattr(args, name, None) is not None
        for name in ("ordered_k_tile", "ordered_m_tile", "ordered_n_tile", "outer_tile_index")
    ):
        raise ValueError("source-bound precision tiles cannot also use ordered matrix partitioning")
    outer, mn_record = prepare_ordered_mn(args, binding, source_request)
    selected, k_record = prepare_ordered_k(args, binding, outer)
    return OrderedNativeRequest(source_request, outer, selected, mn_record, k_record)


def prepare_ordered_k(args: argparse.Namespace, binding, source_request):
    """Ask the selected target to qualify exact ordered-K partitioning."""
    from merlin.compiler.semantic.target_binding import NativeCompilationError
    from merlin.compiler.transforms.tiling import OrderedKTiling

    tile_k = getattr(args, "ordered_k_tile", None)
    if tile_k is None:
        return source_request, None
    prepare = getattr(binding, "prepare_ordered_k", None) if binding is not None else None
    if not callable(prepare):
        raise NativeCompilationError("unsupported_semantics", "selected target has no qualified ordered-K tiling")
    prepared = prepare(source_request, tile_k)
    if (
        not isinstance(prepared, OrderedKTiling)
        or prepared.request.target_identity != source_request.target_identity
        or prepared.request.lowering_policy != source_request.lowering_policy
        or prepared.request.source_identity != source_request.source_identity
    ):
        raise ValueError("target returned an invalid ordered-K partition")
    record = {
        **prepared.record(),
        "source_request_digest": source_request.digest(),
        "tile_k": tile_k,
    }
    return prepared.request, (json.dumps(record, sort_keys=True, indent=2) + "\n").encode()


def prepare_ordered_mn(args: argparse.Namespace, binding, source_request):
    """Select one checked outer output tile from the selected target's geometry."""
    from merlin.compiler.semantic.target_binding import NativeCompilationError
    from merlin.compiler.transforms.tiling import OrderedMNTiling

    tile_m = getattr(args, "ordered_m_tile", None)
    tile_n = getattr(args, "ordered_n_tile", None)
    index = getattr(args, "outer_tile_index", None)
    if tile_m is None and tile_n is None and index is None:
        return source_request, None
    if tile_m is None or tile_n is None or index is None:
        raise ValueError("ordered M/N tiling needs both geometry dimensions and an output tile index")
    prepare = getattr(binding, "prepare_ordered_mn", None) if binding is not None else None
    if not callable(prepare):
        raise NativeCompilationError("unsupported_semantics", "selected target has no qualified M/N tiling")
    prepared = prepare(source_request, tile_m, tile_n)
    if (
        not isinstance(prepared, OrderedMNTiling)
        or prepared.source.digest() != source_request.digest()
        or type(index) is not int
        or not 0 <= index < len(prepared.tiles)
    ):
        raise ValueError("target returned an invalid ordered M/N partition or tile index")
    selected = prepared.tiles[index].request
    if (
        selected.target_identity != source_request.target_identity
        or selected.lowering_policy != source_request.lowering_policy
        or selected.source_identity != source_request.source_identity
    ):
        raise ValueError("outer output tile changed source identity or compilation policy")
    record = {
        **prepared.record(),
        "selected_tile_index": index,
        "selected_request_digest": selected.digest(),
    }
    return selected, (json.dumps(record, sort_keys=True, indent=2) + "\n").encode()


# ---- preflight ---------------------------------------------------------------------------------
# Read-only evidence before claiming a captured model compiled for an OOT target.
#
# Routing at a requested deployment format is a capability question. Grouping the
# captured IR is a different question: it sees the operand formats and quantization
# operations that actually exist. Keep both counts visible until a compiler pass
# bridges them and emits an executable target program.


def preflight_model(capture_bundle: str | Path, *, target: str, deployment_dtype: str) -> dict:
    """Compare declared-format routes with groups present in a real capture.

    This inspects inputs only. It does not run a model, generate a capsule, build a
    backend, or claim a target binary. An explicit capture path avoids silently
    resolving another model revision by name.
    """
    from merlin.capture import demands as source_demands
    from merlin.compiler.ir import mlir_query as mq
    from merlin.compiler.partition.groups import compute_groups
    from merlin.compiler.partition.groups import (
        formation_compute_groups as _merlin_formation_compute_groups_1,
    )
    from merlin.development.targetgen.corpus import group_entries as group_capsule_entries
    from merlin.target.capabilities import routing

    bundle = Path(capture_bundle)
    if bundle.is_symlink() or not bundle.is_dir():
        raise ValueError(f"capture bundle must be an existing non-symlink directory: {bundle}")
    model = bundle / "model.mlir"
    if model.is_symlink() or not model.is_file():
        raise ValueError(f"capture bundle has no regular, non-symlink model.mlir: {bundle}")
    if not deployment_dtype:
        raise ValueError("deployment_dtype must be an exact target format name")

    required = (
        "weights.safetensors",
        "weights.safetensors.manifest.json",
        "inputs.npz",
        "input_order.json",
        "golden.npy",
    )
    missing = [name for name in required if not (bundle / name).is_file() or (bundle / name).is_symlink()]
    text = model.read_text(encoding="utf-8")
    blockers: list[dict[str, object]] = []
    try:
        demands = source_demands.model_op_demands_checked(text, deployment_dtype)
        inventory_verified = True
    except source_demands.ModelDemandIncomplete as exc:
        # Keep the legacy tag inventory for a useful comparison, but make its
        # incompleteness an explicit blocker, never a successful route ledger.
        demands = source_demands.model_op_demands(text, deployment_dtype)
        inventory_verified = False
        blockers.append({"kind": "route_inventory_incomplete", "detail": str(exc)})
    route = routing.route_plan(demands, target)
    module = mq.parse(model)
    groups = _merlin_formation_compute_groups_1.form_groups(module, target)
    contractions = [group for group in groups if group.root is not None]
    on_device = [group for group in contractions if group.placement != compute_groups.HOST]
    tagged_contractions = [demand for demand in demands if demand.family == "contraction"]
    declared_mesh = [result for result in route["mesh"] if result.demand.family == "contraction"]
    stated = group_capsule_entries.entries(target, module, with_raw=False)

    captured_dtypes = Counter(str(group.in_dtype or "unknown") for group in contractions)
    refusals = Counter(
        str(group.refusal or "unknown") for group in contractions if group.placement == compute_groups.HOST
    )
    if missing:
        blockers.append({"kind": "incomplete_capture", "files": missing})
    # A raw count mismatch is not proof of missing operations: some structurally
    # contracting attention regions are classified under a different family.
    # The strict identity-and-shape check above is the authority for completeness.
    if len(declared_mesh) > len(on_device):
        blockers.append(
            {
                "kind": "declared_route_not_materialized",
                "declared_routes": len(declared_mesh),
                "captured_accelerator_groups": len(on_device),
                "explanation": (
                    "Routing at the requested deployment format is not an executable placement "
                    "of the captured IR. A proven dtype/quantization lowering may be required."
                ),
            }
        )
    incompatible = sum(count for dtype, count in captured_dtypes.items() if dtype != deployment_dtype)
    if incompatible:
        blockers.append(
            {
                "kind": "capture_dtype_needs_bridge",
                "contractions": incompatible,
                "requested": deployment_dtype,
                "explanation": "No target-native quantization bridge is established by this preflight.",
            }
        )
    if stated["unstated"]:
        blockers.append({"kind": "accelerator_group_has_no_capsule_form", "reasons": stated["unstated"]})
    if not on_device:
        blockers.append({"kind": "no_accelerator_groups_in_capture"})

    return {
        "schema": "whole_model_preflight_v1",
        "status": "blocked" if blockers else "analyzed_not_compiled",
        "reason": (
            f"{len(blockers)} readiness blocker(s); see blockers and contractions"
            if blockers
            else "capture analysis passed, but no target model binary was built or run"
        ),
        "target": target,
        "capture_bundle": str(bundle.resolve()),
        "deployment_dtype": deployment_dtype,
        "target_binary_emitted": False,
        "scope": "static capture and capability analysis; no model execution or target build",
        "route_inventory_verified": inventory_verified,
        "contractions": {
            "captured": len(contractions),
            "tagged_demands": len(tagged_contractions),
            "declared_format_routes": len(declared_mesh),
            "captured_accelerator_groups": len(on_device),
            "capsule_form_groups": stated["stated"],
            "captured_operand_dtypes": dict(sorted(captured_dtypes.items())),
            "host_refusals": dict(sorted(refusals.items())),
        },
        "blockers": blockers,
    }
