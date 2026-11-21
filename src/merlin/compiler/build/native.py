"""Selected native compilation and checked artifact publication.

Target preparation/encoding stays in the selected OOT provider.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from merlin.compiler.frontend.staged import bind_physical_layouts, load_selected_request, prepare_ordered_request
from merlin.compiler.quantization.contraction.constants import precision_plan_bytes, prepared_static_inputs

# ---- native_records ----------------------------------------------------------------------------
# Publish selected native preparation records inside unpublished staging.
#
# These records bind source transformations and declared parameter bytes to the
# artifact. They do not establish numerical quality or device execution.


def _destination(staged, relative):
    """Keep preparation artifacts inside the actual unpublished directory."""
    staged = Path(staged)
    relative = Path(relative)
    path = staged / relative
    parents = (staged.joinpath(*relative.parts[:index]) for index in range(len(relative.parts) + 1))
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or any(parent.is_symlink() for parent in parents)
        or path.exists()
    ):
        raise ValueError("target preparation destination already exists or is linked outside staging")
    return path


def _write_manifest(staged, manifest):
    (staged / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")


def publish_precision_plan(staged, manifest, precision_plan, *, binding=None, staged_root=None):
    """Track a graph plan or source-bound tile and its declared packed inputs."""
    from merlin.compiler.quantization.contraction.contract import DiagnosticMatmulTile

    raw_plan = precision_plan_bytes(precision_plan)
    tile = isinstance(precision_plan, DiagnosticMatmulTile)
    tracked = manifest.get("stage_artifacts")
    if not isinstance(tracked, dict) or "precision-plan.json" in tracked:
        raise ValueError("target manifest cannot track the source precision plan")
    if tile and "constant_inputs" in manifest:
        raise ValueError("target manifest cannot track the source precision plan")
    if not tile and "constant_inputs" not in manifest:
        raise ValueError("target graph manifest has no explicit constant input record")
    plan_path = staged / "precision-plan.json"
    if plan_path.exists() or plan_path.is_symlink():
        raise ValueError("target precision plan destination already exists")
    if tile and (binding is None or staged_root is None):
        raise ValueError("source-bound constants require their selected support and staged compiler inputs")
    static = (
        prepared_static_inputs(
            binding,
            precision_plan,
            Path(staged_root),
            json.loads((staged / "execution_plan.json").read_bytes()),
        )
        if tile
        else ()
    )
    destinations = []
    for item in static:
        if item.payload is None:
            continue
        relative = item.record()["path"]
        if relative in tracked:
            raise ValueError("target manifest already tracks a source constant")
        destinations.append((_destination(staged, relative), item.payload, relative))
    # Validate every selected destination before writing any parameter payload.
    for path, payload, _ in destinations:
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(payload)
    plan_path.write_bytes(raw_plan)
    manifest = {
        **manifest,
        "stage_artifacts": {
            **tracked,
            "precision-plan.json": hashlib.sha256(raw_plan).hexdigest(),
            **{relative: hashlib.sha256(payload).hexdigest() for _, payload, relative in destinations},
        },
    }
    if tile:
        manifest["constant_inputs"] = [item.record() for item in static]
    _write_manifest(staged, manifest)
    return manifest


def publish_ordered_records(
    staged, manifest, *, source_request, k_source_request, ordered_mn_record=None, ordered_k_record=None
):
    """Retain the original outer-M/N then ordered-K request digest chain."""
    rows = (
        ("ordered-k-tiling.json", ordered_k_record, "source_request_digest", k_source_request, "ordered-K"),
        ("ordered-mn-tiling.json", ordered_mn_record, "outer_source_request_digest", source_request, "ordered-M/N"),
    )
    for name, record, field, source, description in rows:
        if record is None:
            continue
        tracked = manifest.get("stage_artifacts")
        if not isinstance(tracked, dict) or name in tracked:
            raise ValueError(f"target manifest cannot track the {description} partition")
        path = _destination(staged, name)
        path.write_bytes(record)
        manifest = {
            **manifest,
            field: source.digest(),
            "stage_artifacts": {
                **tracked,
                name: hashlib.sha256(record).hexdigest(),
            },
        }
        _write_manifest(staged, manifest)
    return manifest


# ---- native ------------------------------------------------------------------------------------


def publish_layout(staged, manifest, document):
    """Track the explicitly selected source/request-bound physical representation."""
    name = "physical-boundary-layouts.json"
    tracked = manifest.get("stage_artifacts")
    path = staged / name
    if not isinstance(tracked, dict) or name in tracked or path.exists() or path.is_symlink():
        raise ValueError("target manifest cannot track the source physical boundary layout")
    path.write_bytes(document)
    manifest = {**manifest, "stage_artifacts": {**tracked, name: hashlib.sha256(document).hexdigest()}}
    (staged / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return manifest


def compile_report(args, *, load_request, load_limits, write_status):
    """Build and publish one actual selected native artifact."""
    from merlin.compiler.semantic.linalg.linalg_bridge import LinalgBridgeError
    from merlin.compiler.semantic.target_binding import (
        NativeCompilationError,
        load_native_target_binding,
        open_native_snapshot,
        verify_native_publication,
    )

    try:
        if (
            getattr(args, "application_state", None)
            or getattr(args, "application_parameters", "embedded") != "embedded"
        ):
            raise ValueError("application state and external parameters require --application")
        binding = load_native_target_binding(args.support)
        snapshot = open_native_snapshot(Path(args.snapshot))
        if snapshot.profile.digest() != binding.profile().digest():
            raise ValueError("native snapshot differs from selected support profile")
        request, source_kind, source_operations, precision_plan = load_selected_request(
            args,
            target_identity=snapshot.profile.target_identity,
            mode=args.mode,
            binding=binding,
            load_request=load_request,
        )
        request, layout_document = bind_physical_layouts(args, request, precision_plan)
        ordered = prepare_ordered_request(args, binding, request, precision_plan)
        request = ordered.request
        if request.lowering_policy != args.mode:
            raise ValueError("explicit compile mode differs from typed request")
        output = Path(args.out)
        if args.status_file and Path(args.status_file).resolve().is_relative_to(output.resolve()):
            raise ValueError("native compile status file must be outside the artifact directory")
        abi = json.loads(Path(args.abi).read_text())
        if (
            set(abi) != {"fixed_inputs", "fixed_outputs"}
            or not isinstance(abi["fixed_inputs"], dict)
            or (not isinstance(abi["fixed_outputs"], list))
        ):
            raise ValueError("native compile ABI needs fixed_inputs and fixed_outputs")
        if not all(type(value) is int for value in abi["fixed_inputs"].values()) or (
            not all(type(value) is int for value in abi["fixed_outputs"])
        ):
            raise ValueError("native compile ABI addresses must be integers")
        limits = load_limits(args.search_limits)
        if output.exists():
            raise FileExistsError(f"fresh native compilation output required: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="native-compile-", dir=output.parent) as temporary:
            staged = Path(temporary) / "compilation"
            manifest = binding.compile(
                snapshot,
                request,
                fixed_inputs=abi["fixed_inputs"],
                fixed_outputs=tuple(abi["fixed_outputs"]),
                target_source=Path(args.target_source),
                destination=staged,
                limits=limits,
            )
            if precision_plan is not None:
                manifest = publish_precision_plan(
                    staged,
                    manifest,
                    precision_plan,
                    binding=binding,
                    staged_root=getattr(args, "compile_inputs", None),
                )
            if layout_document is not None:
                manifest = publish_layout(staged, manifest, layout_document)
            manifest = publish_ordered_records(
                staged,
                manifest,
                source_request=ordered.source,
                k_source_request=ordered.k_source,
                ordered_mn_record=ordered.mn_record,
                ordered_k_record=ordered.k_record,
            )
            verify_native_publication(
                staged,
                manifest,
                engine=args.engine,
                request_digest=request.digest(),
                target_identity=snapshot.profile.target_identity,
            )
            staged.rename(output)
        report = {
            "schema": "merlin.native_compilation_status.v1",
            "status": "emitted",
            "engine": args.engine,
            "support": args.support,
            "out": str(args.out),
            "request_digest": request.digest(),
            "target_identity": snapshot.profile.target_identity,
            "source_kind": source_kind,
            "source_identity": request.source_identity,
            "source_operations": list(source_operations),
            "search_limits": limits.record(),
            "artifact_kind": manifest.get("artifact_kind", "program"),
        }
        if precision_plan is not None:
            report["precision_plan_path"] = str(output / "precision-plan.json")
            report["constant_inputs"] = manifest["constant_inputs"]
        if ordered.k_record is not None:
            report["source_request_digest"] = ordered.k_source.digest()
            report["ordered_k_tiling_path"] = str(output / "ordered-k-tiling.json")
        if ordered.mn_record is not None:
            report["outer_source_request_digest"] = ordered.source.digest()
            report["ordered_mn_tiling_path"] = str(output / "ordered-mn-tiling.json")
        if manifest.get("artifact_kind") == "program_set":
            report["program_count"] = len(manifest["segments"])
            report["segment_binary_sha256"] = [row["binary_sha256"] for row in manifest["segments"]]
        else:
            report["binary_sha256"] = manifest["binary_sha256"]
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, ImportError, UnicodeError) as error:
        status = (
            "unsupported_semantics"
            if isinstance(error, LinalgBridgeError)
            else error.status
            if isinstance(error, NativeCompilationError)
            else "compile_error"
        )
        report = {
            "schema": "merlin.native_compilation_status.v1",
            "status": status,
            "engine": args.engine,
            "support": args.support,
            "out": str(args.out),
            "reason": str(error),
        }
    if args.status_file and not Path(args.status_file).resolve().is_relative_to(Path(args.out).resolve()):
        write_status(Path(args.status_file), report)
    return report
