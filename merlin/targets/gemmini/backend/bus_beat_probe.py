"""Gemmini boundary for the generic CIRCT bus-beat feasibility probe."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from merlin.perf.bus_beat_probe import assess_compiled_simulator, derive_counter_beat_monitors
from merlin.targetgen.rtl import mlc_bridge

from . import counter_byte_bindings, gemmini


_TARGET = "gemmini"


def _canonical_sha256(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _compiled_model_files(simulator: Path) -> tuple[Path | None, Path | None, str | None]:
    """Discover Verilator metadata from the selected binary's own generated-tree convention."""
    name = simulator.name
    _prefix, separator, config = name.partition("harness-")
    generated = simulator.parent / "generated-src"
    if not separator or not config or not generated.is_dir():
        return None, None, "simulator path does not identify a generated Verilator configuration"
    roots = sorted(path for path in generated.iterdir()
                   if path.is_dir() and path.name.endswith("." + config))
    if len(roots) != 1:
        return None, None, "generated Verilator configuration directory is absent or ambiguous"
    object_dir = roots[0] / roots[0].name
    public = sorted(path for path in object_dir.glob("V*.h")
                    if "___024" not in path.name and "__Syms" not in path.name
                    and "__Dpi" not in path.name and "__pch" not in path.name)
    metadata = sorted(object_dir.glob("V*_classes.mk"))
    if len(public) != 1 or len(metadata) != 1:
        return None, None, "public model header or Verilator classes metadata is absent or ambiguous"
    return public[0], metadata[0], None


def probe_bus_beat_traffic() -> dict[str, Any]:
    """Derive what can be monitored and refuse what the current binary cannot measure."""
    circt_path = mlc_bridge.core_hw_mlir(_TARGET)
    if circt_path is None or not Path(circt_path).is_file():
        artifact: dict[str, Any] = {
            "schema": "merlin.gemmini-bus-beat-probe.v1",
            "target": _TARGET,
            "status": "unknown",
            "physical_byte_facts": [],
            "why": "the active elaborated core CIRCT HW artifact is unavailable",
        }
    else:
        circt = Path(circt_path)
        structural = counter_byte_bindings.probe_counter_byte_bindings()
        monitor = derive_counter_beat_monitors(
            circt.read_text(encoding="utf-8", errors="replace"), structural,
            source=str(circt.resolve()))
        simulator = Path(gemmini.verilator_path())
        public, metadata, discovery_problem = _compiled_model_files(simulator)
        required = []
        for row in monitor.get("monitors", []):
            if isinstance(row, dict) and row.get("status") == "monitor_derivable":
                required.extend(str(row[key]).removeprefix("%")
                                for key in ("valid_port", "ready_port", "data_port"))
        availability = assess_compiled_simulator(
            simulator=simulator, public_header=public, build_metadata=metadata,
            required_ports=required, exact_window_marker=None)
        if discovery_problem:
            availability["problems"] = list(dict.fromkeys(
                [discovery_problem, *availability.get("problems", [])]))
            availability["status"] = "unknown"
        availability["source_binary_binding"] = {
            "status": "unknown",
            "why": ("the generated Verilator source directory and executable are not joined by a "
                    "content-addressed build manifest"),
        }
        artifact = {
            "schema": "merlin.gemmini-bus-beat-probe.v1",
            "target": _TARGET,
            "status": "unknown",
            "monitor_proof": monitor,
            "simulator_feasibility": availability,
            "physical_byte_facts": [],
            "why": ("physical traffic measurement refused: CIRCT monitor candidates lack an "
                    "independent host-memory protocol binding, and the selected prebuilt simulator "
                    "does not expose a content-bound exact-window beat observation path"),
            "required_rebuild_capabilities": [
                "content-addressed CIRCT-to-simulator build manifest",
                "trace or direct callback access to every proved valid/ready/data signal",
                "simulator-visible start and end markers for the exact measured region",
                "independent protocol binding from observed channel to host-memory payload direction",
            ],
        }
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact
