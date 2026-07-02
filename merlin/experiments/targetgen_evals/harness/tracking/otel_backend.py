"""OpenTelemetry backend — optional. Returns nullcontext spans when not available."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Any

from harness.tracking.types import TrackingConfig


class OtelBackend:
    def __init__(self, config: TrackingConfig, local) -> None:
        self._enabled = False
        self._tracer = None
        self._local = local
        self._config = config
        self._trace_id: str | None = None

        try:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
        except ImportError:
            local.warn(
                "opentelemetry-sdk not installed; OTel tracing disabled. "
                "Install with: uv pip install 'targetgen-evals[tracking]'"
            )
            return

        self._setup(config, local, trace)

    def _setup(self, config: TrackingConfig, local, trace_module) -> None:
        try:
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

            resource = Resource.create({
                "service.name": config.service_name,
                "target": config.target,
                "method": config.method,
                "seed": str(config.seed),
                "run_id": config.run_id,
            })
            provider = TracerProvider(resource=resource)

            if config.otel_endpoint:
                try:
                    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
                    exporter = OTLPSpanExporter(endpoint=config.otel_endpoint)
                except Exception as e:
                    local.warn(f"OTel OTLP exporter setup failed ({e}); using console exporter")
                    exporter = ConsoleSpanExporter()
            else:
                exporter = ConsoleSpanExporter()

            provider.add_span_processor(BatchSpanProcessor(exporter))
            trace_module.set_tracer_provider(provider)
            self._tracer = trace_module.get_tracer("targetgen-evals")
            self._enabled = True
        except Exception as e:
            local.warn(f"OTel setup failed ({e}); OTel tracing disabled")
            self._enabled = False

    @property
    def trace_id(self) -> str | None:
        return self._trace_id

    def start_span(self, name: str, attributes: dict[str, Any] | None = None):
        """Return a context manager — real OTel span or nullcontext."""
        if not self._enabled or self._tracer is None:
            return nullcontext()

        @contextmanager
        def _span_cm():
            with self._tracer.start_as_current_span(name, attributes=attributes or {}) as span:
                try:
                    from opentelemetry import trace
                    ctx = trace.get_current_span().get_span_context()
                    if ctx.is_valid:
                        self._trace_id = format(ctx.trace_id, "032x")
                except Exception:
                    pass
                yield span

        return _span_cm()
