"""Fail-closed reading of unbound prebuilt evidence for diagnostic model checks."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from merlin.runtime.modelinputs import file_sha256, strict_tree_sha256


def _require_selected_rtl_facts(inputs: Mapping[str, Any], selected_rtl_facts: Mapping[str, str] | None) -> None:
    """Join an old path-only observation to current selected bytes, without producer credit."""
    observed = inputs.get("rtl_facts")
    identity = inputs.get("rtl_facts_identity")
    if observed is None:
        if identity is not None:
            raise ValueError("prebuilt diagnostic receipt has RTL facts identity without a selected path")
        return
    if not isinstance(observed, str) or selected_rtl_facts is None:
        raise ValueError("prebuilt diagnostic receipt has unselected RTL facts")
    selected_path = selected_rtl_facts.get("path")
    selected_sha256 = selected_rtl_facts.get("sha256")
    if not isinstance(selected_path, str) or not isinstance(selected_sha256, str):
        raise ValueError("prebuilt diagnostic receipt has no byte-pinned selected RTL facts")
    path = Path(observed)
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
        or path.resolve() != path
        or observed != selected_path
        or file_sha256(path) != selected_sha256
        or (identity is not None and identity != dict(selected_rtl_facts))
    ):
        raise ValueError("prebuilt diagnostic receipt differs from selected RTL facts")


def load_diagnostic_receipt(
    path: Path,
    *,
    capture: Path,
    host_package: Path,
    candidate: Path,
    host_package_tree: Mapping[str, Any],
    candidate_tree: Mapping[str, Any],
    arena_mb: int,
    target: str,
    device_selected: bool,
    selected_rtl_facts: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Read old build evidence without asserting its unrecorded producer identity."""
    if (
        not path.is_absolute()
        or path.resolve() != path
        or path.is_symlink()
        or path.name != "baremetal_model.json"
        or not path.is_file()
    ):
        raise ValueError("prebuilt diagnostic receipt is absent or indirect")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict) or receipt.get("schema") != "merlin.baremetal-saved-model.v1":
        raise ValueError("prebuilt diagnostic receipt has another schema")
    inputs = receipt.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("prebuilt diagnostic receipt has no immutable build inputs")
    device = inputs.get("device")
    if device_selected and not isinstance(device, Mapping):
        raise ValueError("prebuilt diagnostic receipt has no selected device")
    _require_selected_rtl_facts(inputs, selected_rtl_facts)
    if (
        inputs.get("capture") != str(capture.resolve())
        or inputs.get("package") != str(host_package.resolve())
        or inputs.get("capture_tree") != strict_tree_sha256(capture)
        or host_package_tree != strict_tree_sha256(host_package)
        or inputs.get("package_tree") != host_package_tree
        or inputs.get("arena_mb") != arena_mb
        or inputs.get("reference_file") is not None
        or (device_selected and device.get("package") != str(candidate.resolve()))
        or (device_selected and device.get("name") != target)
        or (device_selected and candidate_tree != strict_tree_sha256(candidate))
        or (device_selected and device.get("package_tree") != candidate_tree)
        or (not device_selected and device is not None)
    ):
        raise ValueError("prebuilt diagnostic receipt differs from current immutable build inputs")
    output = receipt.get("output")
    if not isinstance(output, Mapping):
        raise ValueError("prebuilt diagnostic receipt has no build output")
    sidecar_record = output.get("device_sidecar")
    if device_selected and not isinstance(sidecar_record, Mapping):
        raise ValueError("prebuilt diagnostic receipt has no selected device sidecar")
    if not device_selected and sidecar_record is not None:
        raise ValueError("prebuilt host-only diagnostic unexpectedly contains device sidecar")
    elf = Path(str(output.get("elf") or ""))
    sidecar = Path(str((sidecar_record or {}).get("path") or "")) if device_selected else None
    build = path.parent / "build"
    if elf.is_symlink() or not elf.is_file() or not elf.resolve().is_relative_to(build.resolve()):
        raise ValueError("prebuilt diagnostic receipt names an absent or external build artifact")
    if sidecar is not None and (
        sidecar.is_symlink() or not sidecar.is_file() or not sidecar.resolve().is_relative_to(build.resolve())
    ):
        raise ValueError("prebuilt diagnostic receipt names an absent or external device sidecar")
    return receipt
