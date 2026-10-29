"""model_check's own linked-image admission for whole-model builds (``check --mode full``).

A device-routed whole-model build requires an independently selected linked-ELF admission service
(:class:`merlin.targetgen.contract.elf_admission.LinkedElfAdmissionService`). The service pins the
source file of its evaluator by digest for the whole build, so the evaluator lives in this small module
of its own: editing the orchestration modules never invalidates a build in flight.

This admission is structural -- it records the image's digest and declared ISA and accepts;
``compile_saved_model`` already holds the image's ISA to the board's DTS. An instruction policy (which
accelerator commands a submission may issue) is the caller's to impose: pass its own service to
:func:`merlin.model_check.full.check` instead.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def admit_structural(*, elf: Path, evidence_root: Path) -> dict[str, Any]:
    """Record the image's digest and declared ISA; accept."""
    from merlin.runtime.backends.spike_model import arch_extensions

    evidence_root = Path(evidence_root)
    evidence_root.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "merlin.model_check.elf_admission.v1",
        "status": "accepted",
        "elf_sha256": _sha256(Path(elf)),
        "arch_extensions": arch_extensions(Path(elf)),
        "scope": "structural: digest and declared ISA; no instruction policy",
    }
    path = evidence_root / "admission.json"
    path.write_text(json.dumps(report, indent=1) + "\n")
    return {**report, "report_path": str(path.resolve()), "report_sha256": _sha256(path)}


def service(target: str):
    """The structural admission as a selected service for ``target``."""
    from merlin.targetgen.contract.build_service import file_digest
    from merlin.targetgen.contract.elf_admission import LinkedElfAdmissionService

    owner = Path(__file__).resolve()
    return LinkedElfAdmissionService(target, admit_structural, ((str(owner), file_digest(owner)),))
