"""Producer-side framework catalog and byte-bound capture sidecars."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path


def _framework_catalog(torch) -> dict:
    """Observe the registry in this exact capture interpreter, never another build."""
    owner = Path(__file__).with_name("_aten_opset_worker.py")
    try:
        spec = importlib.util.spec_from_file_location("_capture_opset", owner)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.core_opset()
    except Exception as exc:  # noqa: BLE001 -- unavailable is not an empty registry
        return {
            "schema": "merlin.pytorch_opset.v1",
            "status": "not_available",
            "torch": str(torch.__version__),
            "n_all_aten": None,
            "reason": f"capture framework catalog unavailable: {type(exc).__name__}: {exc}",
        }


def _framework_cpu_dispatch(torch) -> dict:
    """Observe this framework's CPU path without selecting or normalizing arithmetic."""
    record = {
        "status": "unavailable",
        "capability": None,
        "environment_override": os.environ.get("ATEN_CPU_CAPABILITY"),
    }
    getter = getattr(getattr(getattr(torch, "backends", None), "cpu", None), "get_cpu_capability", None)
    if callable(getter):
        try:
            capability = getter()
        except (AttributeError, RuntimeError, ValueError) as error:
            record["reason"] = f"{type(error).__name__}: {error}"
        else:
            if isinstance(capability, str) and capability:
                record.update(status="observed", capability=capability)
    return record


def frontend_trace_bytes(trace: dict) -> bytes:
    """Keep every trace member, with compact whitespace for the existing bounded readers."""
    return (json.dumps(trace, sort_keys=True, separators=(",", ":")) + "\n").encode()


def write_frontend_trace(path: Path, trace: dict) -> dict:
    """Write the producer's actual trace and return the pointer to those exact bytes."""
    raw = frontend_trace_bytes(trace)
    path.write_bytes(raw)
    return {"path": path.name, "sha256": hashlib.sha256(raw).hexdigest(), "status": trace["status"]}


def write_conversion_status(path: Path, result) -> None:
    """Keep the conversion boundary visible before later capture work can fail."""
    path.write_text(
        json.dumps(
            {
                "schema": "merlin.capture.conversion_status.v1",
                "conversion_ok": bool(result.ok),
                "mlir_bytes": len(result.mlir_text.encode("utf-8")),
                "path_taken": getattr(result, "path_taken", None),
                "diagnostics": [str(item)[:2000] for item in (getattr(result, "diagnostics", None) or [])[:100]],
                "numerical_execution_qualified": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def finalize_session_artifacts(root: Path, report: dict) -> None:
    """Commit final worker sidecars before inventorying the completed session bundle."""
    from m2m.capture.bundle_integrity import write_bundle_integrity

    (root / "session-receipt.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    write_bundle_integrity(root)
