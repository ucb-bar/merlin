"""Generation-time captures through the admitted sealed runner.

Verified Phase 0 admits a capture only from the sealed Model2MLIR runner, with its attestation
re-checked against the bytes on disk (operator policy, 2026-10-01). Capsules written from PyTorch
during generation (operation probes, derived micro models) capture at generation time, so in verified
mode each such capture is preselected, issued in the sandbox, replayed and attested exactly like a
declared iteration capture, and the capsule records the attestation it was built from. Without a
selected sealed runtime, generation-time capture keeps its historical diagnostic path.

The selection travels to the generation process in one environment variable, set when the run is
frozen. The capture request is checked against what the sealed policy can express; anything else --
a declared loader environment, a pinned interpreter, a quantization scheme instead of a recipe, an
already-materialized model -- fails closed instead of being captured outside the seal.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from merlin.targetgen.capsule_source import M2MUnavailable, PytorchRefSource

CONFIG_ENV = "MERLIN_PHASE0_SEALED_CAPTURE"
_BUNDLE_ONLY = frozenset({"capture_receipt.json", "model.mlir"})


def configured() -> dict[str, Any] | None:
    raw = os.environ.get(CONFIG_ENV)
    if not raw:
        return None
    config = json.loads(raw)
    required = {"m2m_root", "venv", "runs_root"}
    if not isinstance(config, dict) or not required <= set(config):
        raise ValueError(f"{CONFIG_ENV} must name {sorted(required)}")
    if "execution_timeout_seconds" in config:
        from .m2m_runtime import capture_timeout

        capture_timeout(config["execution_timeout_seconds"])
    return config


def capture_source() -> PytorchRefSource:
    """The capture source generation writes PyTorch capsules with: sealed when one is selected."""
    config = configured()
    return SealedCaptureSource(config) if config is not None else PytorchRefSource()


def bind_source(entries: list[dict[str, Any]], *, verified: bool) -> PytorchRefSource | None:
    """Select one capture adapter before any member is written.

    A materialized capture already carries its own sealed execution receipt. All
    generation-time frontend captures in a verified corpus must instead use the
    run's one selected sealed adapter; ambient fallback is diagnostic only.
    """
    needs_capture = any(
        not entry.get("materialized_capture")
        and (
            entry.get("kind") == "model"
            or entry.get("op") == "model"
            or entry.get("source") == "pytorch"
            or entry.get("pytorch_ref")
        )
        for entry in entries
    )
    if not needs_capture:
        return None
    source = capture_source()
    if verified and not isinstance(source, SealedCaptureSource):
        raise ValueError("verified Phase 0 requires a selected sealed Model2MLIR runtime before capsule writes")
    if verified and not source.available():
        raise ValueError("verified Phase 0 selected a sealed Model2MLIR runtime that is unavailable")
    return source


class SealedCaptureSource(PytorchRefSource):
    """A :class:`PytorchRefSource` whose every capture is a fresh, attested sealed run."""

    def __init__(self, config: dict[str, Any]):
        venv = Path(config["venv"])
        super().__init__(m2m_dir=Path(config["m2m_root"]), python=venv / "bin/python")
        self.config = dict(config)
        self.attestations: list[dict] = []
        if config.get("tmp_root"):
            # Capture work directories (and every other temporary file of this generation process)
            # live on the run's own filesystem rather than whatever host /tmp happens to have room.
            Path(config["tmp_root"]).mkdir(parents=True, exist_ok=True)
            tempfile.tempdir = str(config["tmp_root"])

    def _staged_merlin(self, package: Path, schemas: Path) -> tuple[Path, Path]:
        """A writable copy of the running Merlin package and its schemas, staged once per process.

        A frozen run executes from a read-only source snapshot, and the sealed runner bundles the
        selected schemas into its copy of the package, which a read-only package refuses. The copy
        holds the same bytes (checked here) with owner-writable modes; the sealed plan then binds the
        staged tree exactly as it binds any selected package.
        """
        staged = Path(self.config["runs_root"]) / "_merlin"
        package_copy, schemas_copy = staged / "src/merlin", staged / "merlin/schemas"
        if not getattr(self, "_staged", False):
            for source, destination in ((package, package_copy), (schemas, schemas_copy)):
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.copytree(source, destination, symlinks=False)
                for path in [destination, *destination.rglob("*")]:
                    path.chmod(path.stat().st_mode | 0o200)
                originals = sorted(p.relative_to(source) for p in source.rglob("*") if p.is_file())
                copies = sorted(p.relative_to(destination) for p in destination.rglob("*") if p.is_file())
                if originals != copies or any(
                    (source / name).read_bytes() != (destination / name).read_bytes() for name in originals
                ):
                    raise M2MUnavailable("the staged Merlin package differs from the running one")
            self._staged = True
        return package_copy, schemas_copy

    def _cache_slot(self, *args, **kwargs):  # noqa: ANN002, ANN003 -- every sealed capture is fresh
        return None

    def _launch_worker(self, cmd, *, env, **request):  # noqa: ANN001
        from merlin.common.paths import module_source_path, schemas_dir

        from .capture_execution_attestation import attest_sealed_m2m
        from .capture_selection import issue, select, verify

        if request.get("scheme") or request.get("already_quantized"):
            raise M2MUnavailable("a sealed capture selects a recipe or a float dtype, never a scheme or a prior model")
        if request.get("declared_env"):
            raise M2MUnavailable("a sealed capture admits no declared loader environment")
        if Path(request["interpreter"]) != self.python:
            raise M2MUnavailable("a sealed capture runs only the selected runtime's interpreter")
        stage_fp32 = request.get("stage_fp32", False)
        if type(stage_fp32) is not bool:
            raise M2MUnavailable("stage_fp32 must be an explicit boolean")
        if (
            stage_fp32
            and str(request["dtype"]) not in {"fp32", "f32"}
            and not (str(request["dtype"]) == "int8" and request.get("recipe_path") is not None)
        ):
            raise M2MUnavailable("FP32 staging requires a float capture or selected int8 recipe")
        root = Path(self.config["runs_root"])
        root.mkdir(parents=True, exist_ok=True)
        slot = root / f"{len(list(root.iterdir())):04d}-{request['op']}-{request['dtype']}"
        slot.mkdir()
        workload = slot / "workload"
        workload.mkdir()
        shutil.copyfile(request["loader_py"], workload / "loader.py")
        recipe = None
        if request.get("recipe_path") is not None:
            recipe = slot / "quant_recipe.json"
            shutil.copyfile(request["recipe_path"], recipe)
        tolerance = request.get("agreement_tolerance")
        merlin_root, schemas_root = self._staged_merlin(Path(module_source_path("merlin")).parent, Path(schemas_dir()))
        identity = select(
            m2m_root=Path(self.config["m2m_root"]),
            frozen_origin=self.config.get("frozen_origin"),
            workload_root=workload,
            worker=merlin_root / "targetgen/_m2m_capture_worker.py",
            venv=Path(self.config["venv"]),
            schemas_root=schemas_root,
            run_dir=slot / "run",
            output_dir=slot / "selection",
            dtype=str(request["dtype"]),
            recipe=recipe,
            bwrap_binary=Path(self.config["bwrap"]) if self.config.get("bwrap") else None,
            # Absent: the historical fixed 120 s and unchanged selection bytes.
            execution_timeout_seconds=self.config.get("execution_timeout_seconds"),
            worker_options={
                **({"agreement_tolerance": [float(value) for value in tolerance]} if tolerance else {}),
                **({"stage_fp32": True} if stage_fp32 else {}),
            }
            or None,
        )
        selection = Path(identity["path"])
        expected_package = self.config.get("package")
        if expected_package is not None:
            from .capture_selection import load

            selected_package = load(selection, expected_sha256=identity["sha256"])["plan"]["selected_trees"]["m2m"]
            if selected_package != expected_package:
                raise M2MUnavailable("the Model2MLIR package differs from the one selected when the run was frozen")
        issue(selection, expected_sha256=identity["sha256"])
        model = slot / "run/capture/model.mlir"
        replay = verify(selection, expected_sha256=identity["sha256"], model_path=model)
        attestation = attest_sealed_m2m(replay, selection_path=selection, model_path=model)
        capture = slot / "run/capture"
        if (capture / "linalg.mlir").read_bytes() != model.read_bytes():
            raise M2MUnavailable("sealed capture's program differs from the attested model bytes")
        workdir = Path(request["workdir"])
        # The worker's own result files, as an unsealed capture leaves them. The bundle's receipt and
        # model copy stay in the sealed run: the attestation, not a copy, is what binds them.
        for member in sorted(capture.iterdir()):
            if member.name in _BUNDLE_ONLY or member.is_symlink() or not member.is_file():
                continue
            shutil.copyfile(member, workdir / member.name)
        meta = json.loads((workdir / "meta.json").read_text(encoding="utf-8"))
        # A materialized bundle records its members relative to itself; resolve them in the copy.
        for key in ("frontend_trace", "framework_catalog"):
            row = meta.get(key)
            if isinstance(row, dict) and isinstance(row.get("path"), str) and not Path(row["path"]).is_absolute():
                row["path"] = str(workdir / row["path"])
        meta["capture_execution_attestation"] = attestation
        (workdir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        self.attestations.append(attestation)
        return SimpleNamespace(returncode=0, stdout="", stderr="")


def verified_capture_failure(capsule: dict) -> str | None:
    """Why a generation-time capture in this capsule is not an admitted sealed execution, or ``None``."""
    from .capture_execution_attestation import AttestationNotVerified, require_verified_execution

    if capsule.get("materialized_capture"):
        return None  # a declared iteration capture, attested at derivation and re-checked by evidence
    if not (capsule.get("pytorch_ref") or capsule.get("source") == "pytorch"):
        return None
    attestations = capsule.get("capture_execution_attestations") or []
    raw = (capsule.get("frontend_trace") or {}).get("capture_mlir_sha256")
    if not attestations:
        return "generation-time capture carries no sealed-runner attestation"
    try:
        for attestation in attestations:
            require_verified_execution(attestation)
    except AttestationNotVerified as exc:
        return str(exc)
    if raw is not None and raw not in {(row.get("capture") or {}).get("model_sha256") for row in attestations}:
        return "the capsule's captured program is not the attested capture"
    return None
