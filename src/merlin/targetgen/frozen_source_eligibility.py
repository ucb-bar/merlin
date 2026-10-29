"""The trusted eligibility denominator of a frozen model source, from that source alone.

Core compiler code (sibling source routing) and the optional native model executor both reopen this
census; it lives here so the core never imports the optional executor. ``error`` selects the
exception type a caller's refusals already use.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _digest(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def census(capsule: Mapping[str, Any], *, target: str, error: type[Exception] = ValueError) -> dict[str, Any]:
    """Build only the trusted eligibility denominator from frozen model source.

    No host-dispatch compile, outline, tile ledger, or candidate placement is
    consulted. The router is used to enumerate demands; its *decisions* are
    not native-execution evidence. This can run even when the legacy model
    grade fails before producing a coverage certificate.
    """
    root_value = capsule.get("__dir__")
    interface_name = capsule.get("interface_mlir") or capsule.get("linalg_mlir")
    attrs = (capsule.get("operation") or {}).get("attributes") or {}
    fmt = attrs.get("dtype")
    if (
        not isinstance(root_value, str)
        or not isinstance(interface_name, str)
        or not interface_name
        or Path(interface_name).is_absolute()
        or ".." in Path(interface_name).parts
        or not isinstance(fmt, str)
        or not fmt
    ):
        raise error("frozen model has no exact local interface and routing dtype")
    root = Path(root_value)
    if not root.is_dir() or root.is_symlink():
        raise error("frozen model capsule directory is absent or indirect")
    source = root / interface_name
    if source.is_symlink() or not source.is_file():
        raise error("frozen model interface is absent or indirect")
    raw = source.read_bytes()
    import yaml

    from merlin.targetgen import target_registry
    from merlin.targetgen.capsule_source import model_op_demands_checked
    from merlin.targetgen.coverage_certificate import for_target
    from merlin.targetgen.routing import route_plan

    source_text = raw.decode("utf-8")
    selected_contract = target_registry.resolve(target).capability_contract_path
    if not selected_contract.is_absolute() or selected_contract.is_symlink() or not selected_contract.is_file():
        raise error("selected target capability contract is absent or indirect")
    contract_pin = _digest(selected_contract)
    contract = yaml.safe_load(selected_contract.read_text(encoding="utf-8"))
    if not isinstance(contract, dict) or contract.get("name") != target:
        raise error("selected target capability contract names a different target")
    with target_registry.observed_contract(target, contract, source_path=selected_contract):
        demands = model_op_demands_checked(source_text, fmt)
        certificate = for_target(route_plan(demands, target), target, linalg_mlir=source_text, execution=None)
    if _digest(source)["sha256"] != hashlib.sha256(raw).hexdigest():
        raise error("frozen source changed during independent eligibility census")
    if _digest(selected_contract) != contract_pin:
        raise error("selected capability contract changed during eligibility census")
    if (
        certificate.get("target") != target
        or certificate.get("source_mlir_sha256") != hashlib.sha256(raw).hexdigest()
        or certificate.get("n_regions") != len(demands)
    ):
        raise error("independent source denominator lacks exact identity")
    certificate["source_inventory_scope"] = "frozen_source_only_no_legacy_execution"
    certificate["eligibility_capability_contract"] = contract_pin
    return certificate
