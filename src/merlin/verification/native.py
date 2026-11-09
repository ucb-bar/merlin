"""Replay selected source/snapshot/declared native artifact identities.

Successful replay establishes artifact identity only. It is not numerical,
implementation, execution, device or hardware qualification.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def verify_report(args, *, load_request):
    """Reconstruct the original request and check the published artifact bytes."""
    from merlin.compiler.frontend.layouts import bind_physical_layouts, check_layout_artifact
    from merlin.compiler.frontend.native import load_selected_request
    from merlin.compiler.frontend.tiling import prepare_ordered_request
    from merlin.semantic_compiler.linalg_bridge import LinalgBridgeError
    from merlin.semantic_compiler.snapshot import open_native_snapshot
    from merlin.semantic_compiler.target_binding import (
        NativeCompilationError,
        load_native_target_binding,
        verify_native_publication,
    )
    from merlin.verification.native_records import check_ordered_records, check_precision_plan

    artifact, status = Path(args.artifact), Path(args.out)
    safe_status = not (
        status.resolve().is_relative_to(artifact.resolve())
        or status.resolve().is_relative_to(Path(args.snapshot).resolve())
    )
    try:
        if not safe_status:
            raise ValueError("native verification status must be outside the artifact and snapshot directories")
        if getattr(args, "application", False) and getattr(args, "diagnostic_graph_policy", None):
            raise ValueError("diagnostic graph precision cannot be combined with --application")
        if getattr(args, "application", False) and getattr(args, "physical_layouts", None):
            raise ValueError("physical boundary layouts cannot be combined with --application")
        if getattr(args, "application", False) and any(
            getattr(args, name, None) is not None
            for name in (
                "diagnostic_region",
                "diagnostic_tile_index",
                "ordered_k_tile",
                "ordered_m_tile",
                "ordered_n_tile",
                "outer_tile_index",
            )
        ):
            raise ValueError("diagnostic tiles and ordered partitions cannot be combined with --application")
        if getattr(args, "application_state", None):
            raise ValueError("--application-state requires --application")
        if getattr(args, "application_parameters", "embedded") != "embedded":
            raise ValueError("--application-parameters requires --application")
        snapshot = open_native_snapshot(Path(args.snapshot))
        binding = load_native_target_binding(args.support) if args.support else None
        if binding is not None and snapshot.profile.digest() != binding.profile().digest():
            raise ValueError("native snapshot differs from selected support profile")
        request, kind, _, precision = load_selected_request(
            args,
            target_identity=snapshot.profile.target_identity,
            mode=args.mode,
            binding=binding,
            load_request=load_request,
        )
        request, layout_document = bind_physical_layouts(args, request, precision)
        ordered = prepare_ordered_request(args, binding, request, precision)
        request = ordered.request
        if request.target_identity != snapshot.profile.target_identity or request.lowering_policy != args.mode:
            raise ValueError("native verification source differs from selected target or compile mode")
        manifest_bytes = (artifact / "manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        if not isinstance(manifest, dict):
            raise ValueError("native artifact manifest must be an object")
        snapshot_digest = hashlib.sha256((snapshot.root / "manifest.json").read_bytes()).hexdigest()
        if manifest.get("native_snapshot_sha256") != snapshot_digest:
            raise ValueError("native artifact snapshot identity differs")
        if manifest.get("compile_mode") != args.mode:
            raise ValueError("native artifact compile mode differs")
        path = artifact / "row-softmax-tiling.json"
        if path.exists() or path.is_symlink() or "softmax_source_request_digest" in manifest:
            raise ValueError("native artifact has an unrequested decomposition or physical boundary layout")
        check_ordered_records(
            artifact,
            manifest,
            source_request=ordered.source,
            k_source_request=ordered.k_source,
            ordered_mn_record=ordered.mn_record,
            ordered_k_record=ordered.k_record,
        )
        check_layout_artifact(artifact, manifest, layout_document)
        verify_native_publication(
            artifact,
            manifest,
            engine=args.engine,
            request_digest=request.digest(),
            target_identity=snapshot.profile.target_identity,
        )
        check_precision_plan(
            artifact,
            manifest,
            precision,
            binding=binding,
            staged_root=getattr(args, "compile_inputs", None),
        )
        tracked = manifest.get("stage_artifacts")
        selected_path = artifact / "native-selection.json"
        if manifest.get("native_check_fingerprint") is not None:
            if not isinstance(tracked, dict) or "native-selection.json" not in tracked:
                raise ValueError("native selected graph is not tracked by the artifact manifest")
            selection = json.loads(selected_path.read_text())
            if not isinstance(selection, dict) or any(
                selection.get(key) != value
                for key, value in (
                    ("engine", args.engine),
                    ("request_digest", request.digest()),
                    ("target_identity", snapshot.profile.target_identity),
                    ("check_fingerprint", manifest.get("native_check_fingerprint")),
                )
            ):
                raise ValueError("native selected graph identity differs from artifact manifest")
        elif selected_path.exists():
            raise ValueError("native selected graph has no checked fingerprint")
        report = {
            "schema": "merlin.native_artifact_verification.v1",
            "status": "verified_artifacts",
            "scope": "source_snapshot_declared_artifact_identity",
            "engine": args.engine,
            "artifact": str(artifact),
            "source_kind": kind,
            "request_digest": request.digest(),
            "target_identity": snapshot.profile.target_identity,
            "binary_sha256": manifest.get("binary_sha256"),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "checked_stage_artifacts": sorted(tracked) if isinstance(tracked, dict) else [],
            "selected_record_checked": manifest.get("native_check_fingerprint") is not None,
        }
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, ImportError, UnicodeError) as error:
        report = {
            "schema": "merlin.native_artifact_verification.v1",
            "status": (
                "unsupported_semantics"
                if isinstance(error, LinalgBridgeError)
                else error.status
                if isinstance(error, NativeCompilationError)
                else "compile_error"
            ),
            "scope": "source_snapshot_declared_artifact_identity",
            "engine": args.engine,
            "artifact": str(artifact),
            "reason": str(error),
        }
    return report, safe_status
