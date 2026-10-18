"""Producer-side framework catalog and byte-bound capture sidecars."""

from __future__ import annotations

import hashlib
import importlib.util
import json
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


def frontend_trace_bytes(trace: dict) -> bytes:
    """Keep every trace member, with compact whitespace for the existing bounded readers."""
    return (json.dumps(trace, sort_keys=True, separators=(",", ":")) + "\n").encode()


def write_frontend_trace(path: Path, trace: dict) -> dict:
    """Write the producer's actual trace and return the pointer to those exact bytes."""
    raw = frontend_trace_bytes(trace)
    path.write_bytes(raw)
    return {"path": path.name, "sha256": hashlib.sha256(raw).hexdigest(), "status": trace["status"]}


def finalize_session_artifacts(root: Path, report: dict) -> None:
    """Commit final worker sidecars before inventorying the completed session bundle."""
    from m2m.capture.bundle_integrity import write_bundle_integrity

    (root / "session-receipt.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    write_bundle_integrity(root)
