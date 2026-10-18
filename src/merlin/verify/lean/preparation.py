"""Capture ordinary shared preparation for a later strict source-to-preprocessing chain.

Capture is explicitly unverified. Only the consuming gate can establish source support,
initialized execution and semantic preservation of all preparation and preprocessing boundaries.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import check, plans, stages

if TYPE_CHECKING:
    from .handoff import Options


@dataclass(frozen=True)
class CapturedPreparation:
    """Immutable unverified evidence reference; retained files are rechecked on consumption."""

    root: Path
    run: str
    plan: plans.Plan
    source_path: Path
    source: bytes
    output: Path
    target: bytes
    features: frozenset[str]
    function: str
    index_bits: int
    compiler_python: Path
    fingerprints: tuple[tuple[str, str], ...]

    @property
    def manifest(self) -> Path:
        return self.root.parent / "expected-plan.json"


def _admit(*, int8_compute, quant_passes, features, matrix, device, outline_int8, prepared_model_transform) -> None:
    from merlin.llvmlower import impr_features, optional_passes

    selection = optional_passes.active()
    effective_features = impr_features.normalize(optional_passes.selected_features(features))
    legacy = [
        entry.name
        for entry in optional_passes.REGISTRY
        if entry.switch and os.environ.get(entry.switch) == "1" and selection.state(entry.name) is not False
    ]
    if int8_compute or quant_passes is not None:
        raise ValueError("preparation capture does not yet cover selected integer quantization")
    if features or effective_features or selection.enable or legacy or os.environ.get("MERLIN_VEC_RANK"):
        raise ValueError("preparation capture does not yet cover optional mutations or effective policy selections")
    if matrix is not None or device is not None or outline_int8 or prepared_model_transform is not None:
        raise ValueError("preparation capture does not cover routing, outlining or provider callbacks")


def capture(
    source: Path,
    work: Path,
    options: Options,
    *,
    int8_compute: bool = False,
    quant_passes: list[str] | None = None,
    features: frozenset[str] | None = None,
    blocking: bool = True,
    harts: int = 1,
    vlen: int | None = None,
    matrix: Any | None = None,
    device: Any | None = None,
    outline_int8: bool = False,
    prepared_model_transform=None,
) -> CapturedPreparation:
    """Run the existing tuple API unchanged, recording an independently authored complete roster.

    Unsupported effective configurations refuse before entering preparation or creating artifacts.
    The result confers no verification status and must be supplied explicitly to ``lower_model``.
    """
    _admit(
        int8_compute=int8_compute,
        quant_passes=quant_passes,
        features=features,
        matrix=matrix,
        device=device,
        outline_int8=outline_int8,
        prepared_model_transform=prepared_model_transform,
    )
    # Resolve the roster before executing even the parser. No expectation comes from stage records.
    expected = plans.preparation()
    from merlin.runtime.backends.zephyr_model import prepare_for_lowering

    from .handoff import _inventory

    source, work = Path(source).resolve(), Path(work).resolve()
    if source == work / "model.prepared.mlir":
        raise ValueError("preparation source aliases its output")
    original = source.read_bytes()
    evidence = work / "lean-preparation" / uuid.uuid4().hex
    evidence.mkdir(parents=True)
    (evidence / "source.mlir").write_bytes(original)
    (evidence / "expected-plan.json").write_bytes(expected.canonical_bytes())
    with stages.capture(evidence / "stages") as recording:
        output, concrete = prepare_for_lowering(
            source,
            work,
            int8_compute=int8_compute,
            quant_passes=quant_passes,
            features=features,
            blocking=blocking,
            harts=harts,
            vlen=vlen,
            matrix=matrix,
            device=device,
            outline_int8=outline_int8,
            prepared_model_transform=prepared_model_transform,
        )
        output = Path(output).resolve()
        target = output.read_bytes()
        published = stages.completed(recording.root)
        if not published:
            raise ValueError("preparation recorded no boundaries")
        stages.record_boundary(
            expected.boundaries[-1].name,
            expected.boundaries[-1].pass_entry,
            (published[-1] / "after.mlir").read_text(encoding="utf-8"),
            target.decode("utf-8"),
        )
        stages.record_handoff(output)
    _admit(
        int8_compute=int8_compute,
        quant_passes=quant_passes,
        features=features,
        matrix=matrix,
        device=device,
        outline_int8=outline_int8,
        prepared_model_transform=prepared_model_transform,
    )
    if concrete:
        raise ValueError("ordinary preparation unexpectedly selected concrete features")
    published, fingerprints = _inventory(recording)
    plans.replay(expected, evidence / "expected-plan.json", recording.root)
    if source.read_bytes() != original or (published[0] / "before.mlir").read_bytes() != original:
        raise ValueError("preparation did not preserve its original source binding")
    if output.read_bytes() != target or (published[-1] / "after.mlir").read_bytes() != target:
        raise ValueError("preparation does not end at the actual returned file")
    check._atomic_json(
        evidence / "capture.json",
        {
            "schema": "merlin.lean.preparation_capture.v1",
            "status": "captured_unverified",
            "scope": expected.scope,
            "configuration": {
                "int8_compute": False,
                "quant_passes": None,
                "features": [],
                "blocking": blocking,
                "harts": harts,
                "vlen": vlen,
                "routing_or_provider_callback": False,
                "optional_policy_enables": [],
            },
            "plan_sha256": expected.sha256,
            "source_sha256": hashlib.sha256(original).hexdigest(),
            "target_sha256": hashlib.sha256(target).hexdigest(),
            "reason": "semantic/source proofs must be established by the consuming shared-pass gate",
        },
    )
    return CapturedPreparation(
        recording.root,
        recording.run,
        expected,
        source,
        original,
        output,
        target,
        frozenset(concrete),
        options.function,
        options.index_bits,
        options.compiler_python,
        tuple(sorted(fingerprints.items())),
    )


def validate(captured: CapturedPreparation, options: Options, source: bytes) -> tuple[list[Path], dict[str, str]]:
    """Recheck an untrusted retained prefix against actual inputs; this grants no semantic proof."""
    from .handoff import _inventory

    expected = plans.preparation()
    if captured.plan != expected:
        raise ValueError("retained capture differs from the independently authored preparation roster")
    if captured.features:
        raise ValueError("retained capture claims unchecked concrete preparation features")

    if (captured.function, captured.index_bits, captured.compiler_python) != (
        options.function,
        options.index_bits,
        options.compiler_python,
    ):
        raise ValueError("preparation and preprocessing use different function/compiler domains")
    if captured.target != source or captured.output.read_bytes() != source:
        raise ValueError("preparation endpoint differs from the actual preprocessing source")
    if (
        captured.source_path.read_bytes() != captured.source
        or (captured.root.parent / "source.mlir").read_bytes() != captured.source
    ):
        raise ValueError("preparation original source changed")
    plans.replay(expected, captured.manifest, captured.root)
    published, fingerprints = _inventory(stages.Capture(captured.root, captured.run, complete=True))
    if tuple(sorted(fingerprints.items())) != captured.fingerprints:
        raise ValueError("preparation capture changed after completion")
    if (published[0] / "before.mlir").read_bytes() != captured.source or (
        published[-1] / "after.mlir"
    ).read_bytes() != source:
        raise ValueError("preparation stage endpoints differ from the captured source/handoff")
    return published, fingerprints
