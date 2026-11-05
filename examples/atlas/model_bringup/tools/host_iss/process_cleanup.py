"""Bounded cleanup of a directly owned QEMU process group."""
from contextlib import suppress
import os
import signal
import subprocess


def reap_qemu(process):
    if process.poll() is None:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
    return process.poll() is not None
