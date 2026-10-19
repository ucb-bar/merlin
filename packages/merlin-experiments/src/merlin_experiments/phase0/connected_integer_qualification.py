"""Original-owner source and realization boundary for connected integer proofs.

Only an explicitly scoped host-selected owner may connect a source proof to its
independently checked executable arithmetic. Serialized diagnostics are not an
owner, and absence of an actual realization checker remains a refusal.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
from pathlib import Path

import numpy as np

from merlin.common import mlir_query as mq
from merlin.targetgen import capsule_source, golden_store
from merlin.targetgen.frontend_use_def import original_use_def_semantics
from merlin_experiments.capture_execution import native_integer

from . import capture_selection
from . import connected_integer_bounds as B
from .capture_execution_attestation import require_verified_execution
from .provenance import _verify_frontend_evidence_dir
from .sealed_generation import verified_capture_failure

SELECTION = "checked_native_integer_v1"
_OWNER = contextvars.ContextVar("merlin_connected_integer_realization_owner", default=None)


def _sha(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("connected source member is absent or indirect")
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextlib.contextmanager
def scoped_realization_owner(owner):
    """Explicit original-host API; this provides no author/serialized-child grant."""
    if (
        owner is None
        or not callable(getattr(owner, "source_rule", None))
        or not callable(getattr(owner, "verify_source_execution", None))
    ):
        raise ValueError("connected integer admission requires an actual selected realization owner")
    token = _OWNER.set(owner)
    try:
        yield
    finally:
        _OWNER.reset(token)


def source_proof(cap, directory):
    """Reopen original sealed source, exact signed inputs and every complete output."""
    directory = Path(directory)
    reason = verified_capture_failure(cap)
    if reason is not None:
        raise ValueError(reason)
    attestations = cap.get("capture_execution_attestations") or []
    if len(attestations) != 1:
        raise ValueError("connected native integer proof currently requires one complete sealed source")
    attestation = attestations[0]
    require_verified_execution(attestation)
    selection = capture_selection.load(
        Path(attestation["selection"]["path"]), expected_sha256=attestation["selection"]["sha256"]
    )
    plan = selection["plan"]
    if not native_integer.selected(plan.get("worker_options")):
        raise ValueError("connected source proof requires explicit unchanged native_integer_v1 capture")
    capture = attestation["capture"]
    if capture.get("kind") == "session":
        raw_sha = (cap.get("frontend_trace") or {}).get("capture_mlir_sha256")
        matches = [p for p in capture.get("programs", []) if p.get("model_sha256") == raw_sha]
        if len(matches) != 1:
            raise ValueError("connected source must select one unique original verified session program")
        root = Path(capture["capture_path"]) / "stages" / matches[0]["name"]
    else:
        root = Path(capture["model_path"]).parent
    metadata = json.loads((root / "meta.json").read_bytes())
    native_integer.validate(root, metadata, plan)
    _verify_frontend_evidence_dir(directory)
    raw = root / "model.mlir"
    portable, _ = capsule_source._portable_weights_reference(raw.read_text(), metadata["weights"], sidecar=True)
    interface = directory / "capsule.interface.mlir"
    if interface.read_text() != portable:
        raise ValueError("connected source program differs from the exact sealed source portability edit")
    trace_path = root / "frontend-trace.json"
    trace = json.loads(trace_path.read_bytes())
    original = original_use_def_semantics(trace)
    if (directory / "frontend-trace.json").read_bytes() != trace_path.read_bytes():
        raise ValueError("connected source frontend ownership differs from original sealed use-def graph")
    attrs = cap["operation"]["attributes"]
    outputs = attrs.get("outs") or [attrs["out"]]
    input_names = attrs["arg_order"][: -len(outputs)]
    if attrs["arg_order"] != input_names + outputs or len(input_names) != len(metadata["input_abi"]):
        raise ValueError("connected source ordered inputs/outputs differ from its complete ABI")
    golden = golden_store.load_golden(directory)
    policy = cap.get("numeric_policy") or {}
    if len(outputs) == 1 and policy == dict(compare="exact_int", dtype=metadata["output_abi"][0]["dtype"]):
        output_policies = {outputs[0]: policy}
    elif policy.get("compare") == "per_output" and set(policy.get("outputs", {})) == set(outputs):
        output_policies = policy["outputs"]
    else:
        raise ValueError("connected source needs complete independently exact per-output numeric policies")
    captured_golden = json.loads((root / "golden.json").read_bytes())
    captured_golden = captured_golden if len(outputs) > 1 else [captured_golden]
    if set(golden["outputs"]) != set(outputs):
        raise ValueError("connected source golden live-out roster differs")
    intervals = []
    with np.load(root / "inputs.npz", allow_pickle=False) as inputs:
        for i, name in enumerate(input_names):
            value = inputs["in" + str(i)]
            observed = golden["oracle_provenance"]["inputs"][name]
            if observed["shape"] != list(value.shape) or not np.array_equal(
                np.asarray(observed["decoded"]).reshape(value.shape), value
            ):
                raise ValueError("connected capsule inputs differ from the exact sealed signed inputs")
            intervals.append((int(value.min()), int(value.max())))
    for name, value, abi in zip(outputs, captured_golden, metadata["output_abi"], strict=True):
        expected = dict(compare="exact_int", dtype=abi["dtype"])
        if output_policies[name] != expected or not np.array_equal(golden["outputs"][name], value):
            raise ValueError("connected capsule output or numeric policy differs from complete sealed source")
    proof = B.derive(mq.parse(raw), intervals)
    proof.update(
        source_sha256=_sha(raw),
        interface_sha256=_sha(interface),
        frontend_trace_sha256=_sha(trace_path),
        original_graph_sha256=original.graph_sha256,
        captured_input_sha256=_sha(root / "inputs.npz"),
        captured_golden_sha256=_sha(root / "golden.json"),
        source_mode=native_integer.MODE,
        selection_sha256=attestation["selection"]["sha256"],
        source_scope="original sealed integral source, exact inputs and every output; executable target still separate",
    )
    return proof


def qualify(entry, cap, directory):
    if entry.get("source_integer_proof") != SELECTION:
        raise ValueError("connected integer proof requires its explicit versioned selection")
    owner = _OWNER.get()
    if owner is None:
        raise ValueError("connected integer source has no original host-selected realization owner")
    proof = source_proof(cap, directory)
    before = json.dumps(proof, sort_keys=True)
    rule = owner.source_rule()
    arithmetic_bounds = B.check_arithmetic_rule(proof, rule)
    realized = owner.verify_source_execution(Path(directory) / "capsule.interface.mlir", proof, Path(directory))
    if not isinstance(realized, dict) or realized.get("status") != "verified_source_execution_correspondence":
        raise ValueError("selected connected integer realization has not verified complete source execution")
    if realized.get("source_sha256") != proof["interface_sha256"] or realized.get("source_rule") != rule:
        raise ValueError("selected connected integer realization differs from exact source or arithmetic scope")
    if json.dumps(proof, sort_keys=True) != before or owner.source_rule() != rule:
        raise ValueError("selected connected source proof or arithmetic rule changed during realization")
    if source_proof(cap, directory) != proof:
        raise ValueError("original sealed source or complete capsule changed during realization")
    return dict(
        schema="merlin.phase0.connected_integer_qualification.v1",
        status="qualified",
        method=SELECTION,
        source_proof=proof,
        source_rule=rule,
        arithmetic_bounds=arithmetic_bounds,
        realization=realized,
        scope=(
            "checked complete source arithmetic and selected executable correspondence; no timing or cohort admission"
        ),
    )
