"""Read one exact native-integer program from an original complete sealed capture.

This adapts receipt-bound relative paths and complete integral goldens to the
ordinary writer. It neither issues a new execution verdict nor grants arithmetic,
compiler, private-exclusion, hardware or performance-cohort admission.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np

from merlin.targetgen.capsule_source import CapsuleArtifacts
from merlin_experiments.capture_execution import native_integer

from . import capture_selection
from .capture_execution_attestation import require_verified_execution


def _root(attestation, program):
    capture = attestation["capture"]
    kind = capture.get("kind")
    if "kind" in capture and (type(kind) is not str or kind not in {"single", "session"}):
        raise ValueError("unsupported sealed capture kind")
    if "model_path" in capture and "capture_path" in capture:
        raise ValueError("ambiguous sealed capture root layout")
    root_key = "model_path" if "kind" not in capture else "capture_path"
    if type(capture.get(root_key)) is not str or not capture[root_key]:
        raise ValueError("sealed capture lacks its exact original root layout")
    root = Path(capture[root_key])
    if kind == "session":
        matches = [p for p in capture["programs"] if p["name"] == program]
        if len(matches) != 1:
            raise ValueError("select one exact program from the complete original sealed roster")
        return root / "stages" / matches[0]["name"]
    if program is not None:
        raise ValueError("single sealed source does not select a session program")
    return root.parent if "kind" not in capture else root


def _payload_limit(root, maximum):
    if type(maximum) is not int or maximum < 1:
        raise ValueError("sealed native payload needs an explicit positive public bound")
    receipt = root / "capture_receipt.json"
    if receipt.is_symlink() or receipt.stat().st_size > min(maximum, 1 << 20):
        raise ValueError("sealed native metadata exceeds its selected public bound")
    from merlin.common import strict_json

    document = strict_json.loads(receipt.read_bytes(), max_bytes=1 << 20)
    names = set(document["artifacts"]) | {"golden.json"}
    names.update(
        path.name for path in root.iterdir() if path.name in {"golden.npy", "goldens.npz", "pytorch-opset.json"}
    )
    total = receipt.stat().st_size
    for name in names:
        path = root / name
        if Path(name).name != name or path.is_symlink() or not path.is_file():
            raise ValueError("sealed native payload leaves are absent or indirect")
        total += path.stat().st_size
        if total > maximum:
            raise ValueError("sealed native payload exceeds its selected public bound before array loading")


def read(attestation, *, program, source_member, max_payload_bytes=None):
    """Reopen complete original source/inputs/outputs; relocate only metadata paths."""
    require_verified_execution(attestation)
    selection = capture_selection.load(
        Path(attestation["selection"]["path"]), expected_sha256=attestation["selection"]["sha256"]
    )
    plan, root = selection["plan"], _root(attestation, program)
    if max_payload_bytes is not None:
        _payload_limit(root, max_payload_bytes)
    member = PurePosixPath(source_member)
    if member.is_absolute() or ".." in member.parts or member.as_posix() != source_member:
        raise ValueError("sealed source member must be an exact relative workload leaf")
    source = Path(selection["run_dir"]) / "snapshots/source/workload" / source_member
    if not source.is_file() or source.is_symlink():
        raise ValueError("selected original PyTorch source leaf is unavailable")
    meta = json.loads((root / "meta.json").read_bytes())
    native_integer.validate(root, meta, plan)
    # Complete golden.json preserves every integral output. golden.npy stores only
    # output zero as FP32 and cannot certify this ABI.
    with np.load(root / "inputs.npz", allow_pickle=False) as archive:
        inputs = [archive[f"in{i}"].tolist() for i in range(len(meta["input_abi"]))]
    for key, name in (("frontend_trace", "frontend-trace.json"), ("framework_catalog", "pytorch-opset.json")):
        selected = dict(meta.get(key) or {})
        path = root / name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != selected.get("sha256"):
            raise ValueError("original sealed metadata leaf changed")
        selected["path"] = str(path)
        meta[key] = selected
    meta.update(
        frontend_raw_mlir_path=str(root / "model.mlir"),
        weights=str(root / "weights.safetensors"),
        weights_manifest=str(root / "weights.safetensors.manifest.json"),
    )
    return CapsuleArtifacts(
        op="model",
        dtype="int8",
        pytorch_src=source.read_text(),
        linalg_mlir=(root / "model.mlir").read_text(),
        inputs=inputs,
        golden=json.loads((root / "golden.json").read_bytes()),
        weights_path=meta["weights"],
        meta=meta,
    )


class Source:
    """Ordinary writer adapter for explicitly reused original sealed execution."""

    def __init__(self, attestation, *, program, source_member):
        self.attestation, self.program, self.source_member = attestation, program, source_member
        self.attestations = []
        selection = capture_selection.load(
            Path(attestation["selection"]["path"]), expected_sha256=attestation["selection"]["sha256"]
        )
        self.m2m_dir = Path(selection["plan"]["m2m_root"])

    def available(self):
        return True

    def capture_loader(self, loader_py, dtype, **options):
        if (
            dtype != "int8"
            or options.get("already_quantized") is not True
            or set(options)
            - {"workdir", "already_quantized", "scheme", "env", "python", "recipe", "agreement_tolerance"}
            or any(options.get(k) for k in ("scheme", "env", "python", "recipe"))
            or options.get("agreement_tolerance") not in (None, (0, 0), [0, 0])
        ):
            raise ValueError("sealed native program adapter admits no precision transformation")
        artifacts = read(self.attestation, program=self.program, source_member=self.source_member)
        if Path(loader_py).read_text() != artifacts.pytorch_src:
            raise ValueError("ordinary source request differs from the exact original sealed source leaf")
        self.attestations.append(self.attestation)
        return artifacts
