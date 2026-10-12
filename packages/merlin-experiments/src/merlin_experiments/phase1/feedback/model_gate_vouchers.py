"""Private source-bound vouchers from actually closed L2 model completions.

Issue only inside the trusted completion producer, while source/audit assets
are owned. The public request supplies no authority or host pathname. Broker
lookup derives a private location; workers reopen its bytes and launch identity.
"""

import json
import os
from hashlib import sha256
from pathlib import Path

from merlin_experiments.phase1.feedback import promotion, snapshots


def _bytes(record):
    return json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _path(ws, digest, capsule):
    anchor = snapshots.promotion_snapshot_path(Path(ws), "0" * 32, create_store=True)
    root = anchor.parent / ".model-gates"
    if root.is_symlink() or root.resolve().is_relative_to(Path(ws).resolve()):
        raise ValueError("model voucher store must be outside candidate workspace")
    root.mkdir(mode=0o700, exist_ok=True)
    root.chmod(0o700)
    return root / (sha256(_bytes([digest, capsule])).hexdigest() + ".json")


def _context(context, cert, contract=None):
    instrument = promotion.cert_instrument(cert, context=context)
    if not instrument:
        raise ValueError("model continuation has no exact certifying instrument")
    descriptor = Path(context.descriptor).resolve(strict=True)
    from merlin.compile.model_execution_inputs import strict_tree_sha256

    contract_path = Path(contract) if contract is not None else Path(context.repo) / "merlin/contract"
    contract_digest = (
        strict_tree_sha256(contract_path) if contract_path.is_dir() else sha256(contract_path.read_bytes()).hexdigest()
    )
    return {
        "descriptor_sha256": sha256(descriptor.read_bytes()).hexdigest(),
        "instrument": instrument,
        "contract_sha256": contract_digest,
    }


def issue(ws, source_ws, result, *, capsule, context, cert_tier, receipt_path, contract=None):
    """Persist scheduling authority only for closed, independently audited L2."""
    from merlin.targetgen.capsule_grade import _qualified_model_check

    path = Path(receipt_path)
    receipt_bytes = path.read_bytes()
    if json.loads(receipt_bytes) != result or result.get("kind") != "model":
        raise ValueError("model voucher completion differs from owned durable receipt")
    check = _qualified_model_check(result)
    tiers = check.get("candidate_required_tiers", {}).get("tiers", {}) if isinstance(check, dict) else {}
    if (
        not isinstance(check, dict)
        or check.get("kind") != "owned_host_model_execution"
        or check.get("status") not in {"incomplete", "pass"}
        or tiers.get("L2", {}).get("status") != "pass"
        or any(
            tiers.get(t, {}).get("status") != "pass"
            for t in ("L0", "L1")
            if t in check.get("candidate_required_tiers", {}).get("required_tiers", [])
        )
        or check.get("candidate_source_placement", {}).get("status") != "clean"
        or check.get("candidate_completed_dispatch", {}).get("status") != "verified"
        or check.get("candidate_source_coverage", {}).get("status") != "verified"
    ):
        raise ValueError("model voucher lacks closed L2/source/dispatch certificate")
    gate = result.get("model_gate_admission")
    if (
        not isinstance(gate, dict)
        or gate.get("stage") != "L2"
        or gate.get("passed") is not True
        or type(gate.get("fraction")) not in (float, int)
        or not 0 <= gate["fraction"] <= 1
        or gate["fraction"] < float((capsule.get("gate") or {}).get("after_op_pass_fraction", 0))
    ):
        raise ValueError("model voucher lacks actual selected operator gate admission")
    digest = promotion._submission_digest(source_ws)
    name = capsule["name"]
    declaration = Path(capsule["__dir__"]) / "capsule.yaml"
    record = {
        "schema": "private_model_gate_v1",
        "workspace": str(Path(ws).resolve()),
        "source_digest": digest,
        "capsule": name,
        "target": capsule.get("target") or result.get("target"),
        "capsule_sha256": sha256(declaration.read_bytes()).hexdigest(),
        "cert_tier": cert_tier,
        "gate": gate,
        "closed_l2_certificate_sha256": sha256(_bytes(check)).hexdigest(),
        "closed_l2_receipt_sha256": sha256(receipt_bytes).hexdigest(),
        **_context(context, cert_tier, contract),
    }
    # Target belongs to the immutable selected invocation, not result fields.
    from merlin.targetgen.target_experiment import load_target_experiment

    record["target"] = load_target_experiment(context.descriptor).target
    destination = _path(ws, digest, name)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(_bytes(record))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return destination


def lookup(ws, *, digest, capsule, context, cert_tier, contract=None):
    """Broker-only derived lookup; caller requests cannot provide file paths."""
    path = _path(ws, digest, capsule)
    if not path.is_file():
        return None
    verify(path, source_digest=digest, capsules={capsule}, context=context, cert_tier=cert_tier, contract=contract)
    return path


def verify(path, *, source_digest, capsules, context, cert_tier, contract=None):
    from merlin.targetgen.model_gate_continuation import ModelGateContinuation
    from merlin.targetgen.target_experiment import load_target_experiment

    path = Path(path)
    if path.is_symlink() or path.resolve() != path or not path.is_file():
        raise ValueError("model voucher is absent or indirect")
    raw = path.read_bytes()
    record = json.loads(raw)
    name = record.get("capsule")
    if (
        record.get("schema") != "private_model_gate_v1"
        or name not in capsules
        or record.get("source_digest") != source_digest
        or record.get("cert_tier") != cert_tier
        or path != _path(record["workspace"], source_digest, name)
        or record.get("target") != load_target_experiment(context.descriptor).target
        or any(record.get(k) != v for k, v in _context(context, cert_tier, contract).items())
        or record.get("gate", {}).get("passed") is not True
    ):
        raise ValueError("model voucher differs from protected source/context/instrument")
    for field in ("capsule_sha256", "closed_l2_certificate_sha256", "closed_l2_receipt_sha256"):
        value = record.get(field)
        if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("model voucher has no exact original certificate identity")
    return ModelGateContinuation(
        name, record["capsule_sha256"], record["target"], source_digest, sha256(raw).hexdigest()
    )
