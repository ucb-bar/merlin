"""Opt-in real public-hook regression; owns its local Ray runtime and service."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from merlin.common.paths import python_import_roots


@pytest.mark.skipif(os.environ.get("MERLIN_TEST_MANAGED_CHIA") != "1", reason="explicit local Ray qualification only")
@pytest.mark.parametrize("profiled", [False, True])
def test_public_chia_cooperative_cancel_reaps_native_and_guardian(profiled):
    # Ray's node IP is not a guarantee that every GCS/worker listener binds only
    # localhost. Require an externally isolated loopback-only network namespace.
    if sys.platform != "linux" or {name for _, name in socket.if_nameindex()} != {"lo"}:
        pytest.fail("Ray qualification requires a loopback-only network namespace/container (--network=none)")
    pytest.importorskip("chia.base.ChiaFunction")
    script = r"""
import os, pathlib, signal, sys, time
import ray
from chia.base.ChiaFunction import ChiaFunction, get
from chia.trace.profiler import start_collector, stop_collector, reset_profiler
from merlin_experiments.execution.chia_native import Session, setup, cleanup

root = pathlib.Path(sys.argv[1])
profiled = sys.argv[2] == "True"
endpoint = root / "service.sock"
marker = root / "native-started"
def interrupted(*_args):
    raise TimeoutError("outer test requested bounded driver teardown")
signal.signal(signal.SIGTERM, interrupted)
session = None
try:
    session = Session(endpoint)
    assert not ray.is_initialized()
    ray.init(address="local", num_cpus=1, include_dashboard=False,
             object_store_memory=80 * 1024 * 1024, _temp_dir=str(root / "ray"),
             runtime_env={"env_vars": {"PYTHONPATH": os.environ["PYTHONPATH"], "CHIA_AET_SINK": "0"}})
    if profiled:
        start_collector(log_dir=str(root / "profile"))
        reset_profiler()
    @ChiaFunction(num_cpus=1, max_retries=0)
    def execute(marker):
        from merlin_experiments.execution.chia_native import run
        return run([sys.executable, "-c",
                    "import pathlib,sys,time; pathlib.Path(sys.argv[1]).touch(); time.sleep(60)", marker],
                   cwd=str(pathlib.Path(marker).parent))
    invitation = session.reserve()
    ref = execute.options(num_cpus=1).chia_remote(
        str(marker), _chia_setup=setup, _chia_setup_args=(invitation,),
        _chia_cleanup=cleanup, _chia_cleanup_args=(invitation,))
    deadline = time.monotonic() + 25
    while not marker.exists():
        assert time.monotonic() < deadline
        time.sleep(.02)
    ray.cancel(ref, force=False)
    try:
        get(ref, timeout=5)
    except ray.exceptions.TaskCancelledError:
        pass
    else:
        raise AssertionError("cooperative cancellation was not acknowledged")
    # Cancellation is not proof: require the independent direct-parent receipt.
    receipt = session.receipt(invitation)
    assert receipt["guardian_created"] and receipt["guardian_reaped"]
    assert receipt["guardian"]["native_started"] and receipt["cleanup_complete"]
finally:
    primary = sys.exception()
    errors = []
    for close in ([session.close] if session is not None else []) + [stop_collector, ray.shutdown]:
        try:
            close()
        except BaseException as error:
            errors.append(error)
    if errors:
        if primary is None:
            raise errors[0]
        for error in errors:
            primary.add_note(f"owned test runtime teardown also failed: {error!r}")
"""
    environment = {
        **os.environ,
        "CHIA_AET_SINK": "0",
        "PYTHONPATH": os.pathsep.join(map(str, python_import_roots())),
    }
    # Ray adds long session/socket suffixes; use an explicitly owned short root.
    with tempfile.TemporaryDirectory(prefix="mch-", dir="/tmp") as runtime:
        endpoint = Path(runtime) / "service.sock"
        # The pytest process owns/reaps the service independently of the driver,
        # including if a regression forces termination of that driver.
        service = subprocess.Popen(
            [sys.executable, "-m", "merlin_experiments.execution.native_supervisor", "--endpoint", str(endpoint)],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        driver = None
        try:
            deadline = time.monotonic() + 10
            while not endpoint.exists():
                assert service.poll() is None and time.monotonic() < deadline
                time.sleep(0.02)
            driver = subprocess.Popen(
                [sys.executable, "-c", script, runtime, str(profiled)],
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            output, error = driver.communicate(timeout=90)
            assert driver.returncode == 0, output + error
        finally:
            try:
                if driver is not None and driver.poll() is None:
                    driver.terminate()  # Python handler permits all owned shutdown steps.
                    try:
                        driver.communicate(timeout=20)
                    except subprocess.TimeoutExpired:
                        driver.kill()
                        driver.communicate(timeout=5)
            finally:
                if service.poll() is None:
                    service.terminate()
                _, error = service.communicate(timeout=12)
                assert service.returncode == 0, error
