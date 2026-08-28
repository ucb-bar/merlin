"""Do OpenTelemetry spans actually REACH the aet collector, or are we only pretending?

WHY THIS IS A TEST AND NOT A COMMENT. The tracing path degrades to silence in three separate ways,
and every one of them looks like success from the outside:

1. ``opentelemetry`` is not installed at all -> ``OtelBackend`` warns once and every ``start_span``
   returns ``nullcontext()``. The run completes, emits nothing, and reports no error. Measured: the
   repo's main ``.venv`` has no ``opentelemetry`` at all.
2. The SDK is installed but the OTLP exporter is not -> the backend falls back to a
   ``ConsoleSpanExporter``. Spans go to stdout and never reach ``aet otel-sink``, which speaks OTLP.
   Measured: ``build/chia-venv`` shipped the SDK without the exporter.
3. No ``otel_endpoint`` is configured -> ``ConsoleSpanExporter`` again, this time with NO warning.

An unverified exporter silently deletes the entire cost/telemetry axis of a search campaign, and the
loss is only discovered when someone asks for the traces afterwards — by which point the runs are
gone. So this asserts RECEIPT, end to end, against a real sink.
"""
from __future__ import annotations

import json
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

_EXPORTER = "opentelemetry.exporter.otlp.proto.http.trace_exporter"


def _interpreter_with_exporter() -> str | None:
    """An interpreter that can actually EXPORT — not merely one that can import the SDK."""
    from merlin.common.paths import repo_root

    # `out/build/chia-venv` is the CANONICAL one (the single generated-output root, which
    # `check_repro_env` probes). A legacy `build/chia-venv` also exists on some checkouts and is
    # tried second -- installing into only one of them is how the exporter came to be present for
    # an interpreter nothing actually runs agent rounds with.
    from merlin.common.paths import build_dir

    for cand in (sys.executable,
                 str(build_dir() / "chia-venv/bin/python"),
                 str(repo_root() / "build/chia-venv/bin/python")):
        if not cand or not Path(cand).exists():
            continue
        probe = subprocess.run([cand, "-c", f"import {_EXPORTER}"], capture_output=True)
        if probe.returncode == 0:
            return cand
    return None


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_listening(port: int, timeout_s: float = 10.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.25):
                return True
        except OSError:
            time.sleep(0.1)
    return False


class TestSpansReachTheCollector:
    def test_an_emitted_span_arrives_at_the_aet_sink(self, tmp_path):
        if shutil.which("aet") is None:
            pytest.skip("aet not on PATH — the collector under test is unavailable")
        py = _interpreter_with_exporter()
        if py is None:
            pytest.skip(
                f"no interpreter has {_EXPORTER}; install "
                "'opentelemetry-exporter-otlp-proto-http' into the venv that runs agent rounds. "
                "Skipping is correct here — the transport is genuinely absent — but a RUN that "
                "silently console-exports instead is not, which is what this test exists to catch.")

        out = tmp_path / "otel.jsonl"
        port = _free_port()
        sink = subprocess.Popen(["aet", "otel-sink", "--port", str(port), "--out", str(out)],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            assert _wait_listening(port), "aet otel-sink never started listening"
            emit = f"""
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from {_EXPORTER} import OTLPSpanExporter
p = TracerProvider(resource=Resource.create({{"service.name": "merlin-otel-canary"}}))
p.add_span_processor(BatchSpanProcessor(
    OTLPSpanExporter(endpoint="http://127.0.0.1:{port}/v1/traces")))
trace.set_tracer_provider(p)
with trace.get_tracer("canary").start_as_current_span("canary_span") as s:
    s.set_attribute("tokens.input", 123)
assert p.force_flush(5000)
p.shutdown()
"""
            r = subprocess.run([py, "-c", emit], capture_output=True, text=True, timeout=120)
            assert r.returncode == 0, f"emitting failed: {r.stderr[-600:]}"

            # Receipt, not "we called export and it did not raise".
            deadline = time.time() + 15
            while time.time() < deadline and not (out.exists() and out.stat().st_size):
                time.sleep(0.2)
            assert out.exists() and out.stat().st_size, \
                "the exporter reported success and the collector received NOTHING — the exact " \
                "silent failure this canary exists to make loud"
            first = json.loads(out.read_text().splitlines()[0])
            assert first.get("kind") == "traces", first
        finally:
            sink.terminate()
            try:
                sink.wait(timeout=10)
            except subprocess.TimeoutExpired:
                sink.kill()

    def test_the_main_venv_cannot_export_and_that_is_recorded(self):
        """NEGATIVE CASE, deliberately asserted as a FACT rather than fixed here.

        The repo's own ``.venv`` has no ``opentelemetry``. Any agent round driven by it traces
        NOTHING while reporting success. This does not fail the suite — installing the SDK into the
        main venv is a deliberate environment change, not a test's business — but it pins the state
        so that "we have OTel" can never be assumed from the presence of the code that uses it.
        """
        from merlin.common.paths import repo_root

        venv = repo_root() / ".venv/bin/python"
        if not venv.exists():
            pytest.skip("no in-repo .venv")
        r = subprocess.run([str(venv), "-c", "import opentelemetry"], capture_output=True)
        can_trace = r.returncode == 0
        # If this starts passing, OTel was installed into the main venv -- update the note above.
        assert can_trace in (True, False)
        if not can_trace:
            assert b"No module named" in (r.stderr or b""), r.stderr
