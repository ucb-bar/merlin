"""Export the ordinary whole-model host/device boundary without building an OOT backend.

Inputs are an explicitly selected capture, target and corpus binder. The same
group rewrite used by whole-model compilation owns layout, fusion, preparation
and call signatures. This endpoint neither changes quantization nor runs a
simulator; emitted interfaces are requests to a backend, not execution evidence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def export_rewrite(
    module, rewrite, *, binder, output: str | Path, source_reference: str, connect_spatial: bool = False
) -> dict:
    """Export a completed ordinary rewrite and its deduplicated backend requests.

    Accepting the rewrite itself lets a caller retain its earlier work. Every
    actual call must have a matching statement; a partial interface export is
    reported as incomplete, never as a completed model compilation.
    """
    from merlin.llvmlower.whole_program import group_interface
    from merlin.targetgen.contract.interface_emit import parse_interface_mlir

    connected = None
    if connect_spatial:
        from merlin.llvmlower.connected_spatial import group_connected_spatial

        selection = group_connected_spatial(module, rewrite)
        module, rewrite, connected = selection.module, selection.rewrite, selection.receipt

    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    module.verify()
    host_text = str(module) + "\n"
    (root / "host.mlir").write_text(host_text)
    sidecar = rewrite.to_sidecar()
    (root / "device_signatures.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    requests = []
    for index, (symbol, entry) in enumerate(sorted(rewrite.entries.items())):
        row = {"symbol": symbol, "entry": entry}
        try:
            text = group_interface(entry, binder, source_reference=source_reference)
            parse_interface_mlir(text)
            filename = f"interface-{index}.mlir"
            (root / filename).write_text(text)
            row.update(status="emitted", file=filename, sha256=hashlib.sha256(text.encode()).hexdigest())
        except Exception as error:
            row.update(status="refused", error=type(error).__name__, reason=str(error))
        requests.append(row)
    emitted = {row["symbol"] for row in requests if row["status"] == "emitted"}
    missing = sorted({row.symbol for row in rewrite.routed} - emitted)
    report = {
        "schema": "model_interface_export_v1",
        "status": "incomplete" if missing or rewrite.skipped else "interface_exported",
        "source_reference": source_reference,
        "host_ir_sha256": hashlib.sha256(host_text.encode()).hexdigest(),
        "device_calls": len(rewrite.routed),
        "calls_with_interface": sum(row.symbol in emitted for row in rewrite.routed),
        "unique_interfaces": len(emitted),
        "missing_call_interfaces": missing,
        "skipped": list(rewrite.skipped),
        "requests": requests,
        "scope": "host/device boundary only; host operations remain in host.mlir",
        "target_execution": False,
        "numerical_evaluation": False,
    }
    if connected is not None:
        report["connected_spatial_selection"] = connected
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def export_capture(
    capture: str | Path, *, target: str, binder, output: str | Path, connect_spatial: bool = False
) -> dict:
    """Run ordinary complete-group lowering up to the authored backend boundary.

    The caller selects the target contract/facts with the normal target registry
    context and supplies its corpus binder. No target or model defaults are
    discovered here. Source captures are read only.
    """
    from merlin.common import mlir_query as mq
    from merlin.llvmlower.device_offload import rewrite_groups_to_device

    source = Path(capture)
    source = source / "model.mlir" if source.is_dir() else source
    module = mq.parse(source.read_text())
    rewrite = rewrite_groups_to_device(
        module,
        target,
        select=lambda _: True,
        capture=source,
        require_complete_groups=True,
        blob_weights=True,
    )
    return export_rewrite(
        module, rewrite, binder=binder, output=output, source_reference=str(source), connect_spatial=connect_spatial
    )
