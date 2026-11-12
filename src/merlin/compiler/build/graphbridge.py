"""Publish the typed host boundary of one checked native graph artifact.

This preserves the diagnostic source/ABI handoff. It neither implements the
external call nor establishes numerical, device, image or execution qualification.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def publish_graph_bridge(linalg, entry, *, compiled, device_symbol, out) -> tuple[int, dict]:
    """Return the original command status and response after checked publication."""
    from merlin.compiler.build.hostbridge import diagnostic_whole_graph_host_bridge
    from merlin.compiler.semantic.target_binding import verify_native_publication

    output = Path(out)
    compiled = Path(compiled)
    if output.exists() or output.is_symlink():
        return 2, {"status": "compile_error", "reason": "fresh output directory required"}
    # No failure record may land within the artifact being checked.
    if output.resolve().is_relative_to(compiled.resolve()):
        return 2, {"status": "compile_error", "reason": "host bridge output must be outside the native compilation"}
    try:
        manifest_path = compiled / "manifest.json"
        if manifest_path.is_symlink():
            raise ValueError("native compilation manifest is linked")
        compilation = json.loads(manifest_path.read_text())
        if not isinstance(compilation, dict):
            raise ValueError("native compilation manifest must be an object")
        verify_native_publication(
            compiled,
            compilation,
            engine="merlin_native",
            request_digest=compilation.get("request_digest"),
            target_identity=compilation.get("target_identity"),
        )
        stage_artifacts = compilation.get("stage_artifacts")
        required = {"native-selection.json", "precision-plan.json"}
        if not isinstance(stage_artifacts, dict) or not required <= stage_artifacts.keys():
            raise ValueError("native graph stages are not tracked by the compilation manifest")
        selection = json.loads((compiled / "native-selection.json").read_text())
        precision_plan = json.loads((compiled / "precision-plan.json").read_text())
        execution_plan = json.loads((compiled / "execution_plan.json").read_text())
        layouts_name = "physical-boundary-layouts.json"
        physical_layouts = (
            json.loads((compiled / layouts_name).read_text()) if layouts_name in stage_artifacts else None
        )
        record, text = diagnostic_whole_graph_host_bridge(
            Path(linalg).read_bytes().decode("utf-8"),
            entry,
            device_symbol=device_symbol,
            compilation=compilation,
            selection=selection,
            precision_plan=precision_plan,
            execution_plan=execution_plan,
            physical_layouts=physical_layouts,
        )
        output.mkdir(parents=True)
        (output / "host-bridge.mlir").write_text(text)
        record["host_bridge_sha256"] = hashlib.sha256(text.encode()).hexdigest()
        record["native_manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        (output / "manifest.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
        return 0, {"status": record["status"], "out": str(output)}
    except (OSError, ValueError, RuntimeError, TypeError, KeyError, UnicodeError) as error:
        output.mkdir(parents=True, exist_ok=True)
        (output / "manifest.json").write_text(
            json.dumps(
                {
                    "schema": "merlin.diagnostic_whole_graph_host_bridge.v1",
                    "status": "compile_error",
                    "reason": str(error),
                },
                sort_keys=True,
                indent=2,
            )
            + "\n"
        )
        return 2, {"status": "compile_error", "reason": str(error), "out": str(output)}
