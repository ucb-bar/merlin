"""TargetGenRunLogger — single tracking API for the targetgen-evals harness."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

from harness.tracking.types import TrackingConfig, TRACKING_MODES
from harness.tracking.local_backend import LocalBackend
from harness.tracking.mlflow_backend import MLflowBackend
from harness.tracking.otel_backend import OtelBackend


class TargetGenRunLogger:
    """
    One canonical tracking abstraction for all harness code.

    Backends:
      local  — always active (pure stdlib, writes to logs/)
      mlflow — active when mode != "local" (optional import)
      otel   — active when mode in ("full", "debug") (optional import)

    All backend failures are caught, warned locally, and never propagate.
    """

    def __init__(self, config: TrackingConfig) -> None:
        self._config = config
        self._local = LocalBackend(config)
        self._mlflow: MLflowBackend | None = (
            MLflowBackend(config, self._local) if config.mode != "local" else None
        )
        self._otel: OtelBackend | None = (
            OtelBackend(config, self._local) if config.mode in ("full", "debug") else None
        )

    # ------------------------------------------------------------------
    @classmethod
    def start(
        cls,
        *,
        target: str,
        method: str,
        seed: int,
        run_id: str,
        run_path: Path,
        tracking_mode: str = "local",
        mlflow_tracking_uri: str | None = None,
        experiment_name: str | None = None,
        otel_endpoint: str | None = None,
    ) -> "TargetGenRunLogger":
        if tracking_mode not in TRACKING_MODES:
            tracking_mode = "local"
        config = TrackingConfig(
            mode=tracking_mode,
            run_id=run_id,
            target=target,
            method=method,
            seed=seed,
            run_path=Path(run_path),
            mlflow_tracking_uri=mlflow_tracking_uri,
            experiment_name=experiment_name,
            otel_endpoint=otel_endpoint,
        )
        return cls(config)

    # ------------------------------------------------------------------
    def log_param(self, name: str, value: Any) -> None:
        self._local.log_param(name, value)
        if self._mlflow:
            self._mlflow.log_param(name, value)

    def log_params(self, params: dict[str, Any]) -> None:
        self._local.log_params(params)
        if self._mlflow:
            self._mlflow.log_params(params)

    # ------------------------------------------------------------------
    def log_metric(
        self,
        name: str,
        value: int | float | bool | None,
        step: int | None = None,
        source: str | None = None,
    ) -> None:
        self._local.log_metric(name, value, step=step, source=source)
        if self._mlflow:
            self._mlflow.log_metric(name, value, step=step, source=source)

    def log_metrics(self, metrics: dict[str, Any], prefix: str | None = None) -> None:
        self._local.log_metrics(metrics, prefix=prefix)
        if self._mlflow:
            self._mlflow.log_metrics(metrics, prefix=prefix)

    # ------------------------------------------------------------------
    def log_event(self, name: str, payload: dict | None = None) -> None:
        self._local.log_event(name, payload)

    # ------------------------------------------------------------------
    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        self._local.log_artifact(path, artifact_path)
        if self._mlflow:
            self._mlflow.log_artifact(path, artifact_path)

    def log_artifacts(self, path: Path, artifact_path: str | None = None) -> None:
        self._local.log_artifacts(path, artifact_path)
        if self._mlflow:
            self._mlflow.log_artifacts(path, artifact_path)

    def log_generated_dir(self, generated_dir: Path, target: str) -> None:
        """Log generated/<target>-mlir/ as a tarball artifact (MLflow only)."""
        if self._mlflow:
            self._mlflow.log_generated_dir_as_tarball(generated_dir, target)

    # ------------------------------------------------------------------
    def start_span(self, name: str, attributes: dict[str, Any] | None = None):
        """Context manager. Returns real OTel span or nullcontext."""
        if self._otel:
            return self._otel.start_span(name, attributes)
        return nullcontext()

    # ------------------------------------------------------------------
    def finish(self, status: str, message: str | None = None) -> None:
        self._local.log_event("run.finished", {"status": status, "message": message})
        if self._mlflow:
            self._mlflow.finish(status, message)

    def close(self) -> None:
        """Flush and close all backends. Call after finish()."""
        pass  # local backend has no buffers; MLflow run already ended in finish()

    def patch_manifest(self, manifest_path: Path) -> None:
        """Write tracking IDs back into run_manifest.yaml."""
        if self._mlflow:
            self._mlflow.patch_manifest(manifest_path)

    # ------------------------------------------------------------------
    @property
    def mlflow_run_id(self) -> str | None:
        return self._mlflow.run_id if self._mlflow else None

    @property
    def otel_trace_id(self) -> str | None:
        return self._otel.trace_id if self._otel else None

    @property
    def mode(self) -> str:
        return self._config.mode

    @property
    def logs_dir(self) -> Path:
        return self._config.run_path / "logs"

    def print_summary(self) -> None:
        """Print tracking status to stdout."""
        mlf_status = "enabled" if (self._mlflow and self._mlflow._enabled) else "disabled"
        otel_status = "enabled" if (self._otel and self._otel._enabled) else "disabled"
        print(f"  tracking mode:  {self._config.mode}")
        print(f"  MLflow:         {mlf_status}" +
              (f" (run_id: {self.mlflow_run_id})" if self.mlflow_run_id else ""))
        print(f"  OTel:           {otel_status}")
        print(f"  local logs:     {self.logs_dir}")
