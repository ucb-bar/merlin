"""Installed payload/origin checks and pytest guard for qualify_installed.py.

Copied byte-for-byte into the retained external venv. Never used as an isolation sandbox.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.abc
import importlib.util
import json
import sys
import sysconfig
import zipfile
from pathlib import Path


class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {
            "_common",
            "_pbcommon",
            "run_baseline_qa_loop",
            "run_agent_experiment",
            "run_agentic_perf_experiment",
            "run_paired_perf_bench",
            "run_global_perf_experiment",
            "perf_agent_stage",
            "perf_snapshot",
            "chia_agentic_perf_experiment",
            "tooling_readiness",
            "chia",
            "ray",
        }:
            raise AssertionError("unexpected native or scheduler import: " + fullname)


sys.meta_path.insert(0, NoNative())


def assert_installed_origins():
    site = Path(sysconfig.get_path("purelib")).resolve()
    checked = []
    for name, module in tuple(sys.modules.items()):
        if name in {"merlin", "merlin_experiments"} or name.startswith(("merlin.", "merlin_experiments.")):
            filename = getattr(module, "__file__", None)
            if filename:
                assert Path(filename).resolve().is_relative_to(site), (name, filename)
                checked.append(name)
    return checked


def pytest_sessionfinish(session, exitstatus):
    assert_installed_origins()


def probe(wheels, *, modules=(), required_modules=()):
    site = Path(sysconfig.get_path("purelib")).resolve()
    count = 0
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            for name in archive.namelist():
                if name.endswith("/") or ".dist-info/" in name:
                    continue
                assert (site / name).read_bytes() == archive.read(name), name
                count += 1
    assert count, "empty installed payload check"
    for module in modules:
        importlib.import_module(module)
    for module in required_modules:
        assert importlib.util.find_spec(module) is not None, "suite requires module: " + module
    return {"verified_payloads": count, "site": str(site), "installed_modules": assert_installed_origins()}


if __name__ == "__main__":
    if sys.argv[1:2] == ["--guarded-tests"]:
        import socket
        import subprocess

        def forbidden(*args, **kwargs):
            raise AssertionError("installed pure qualification cannot launch processes or listeners")

        # Install before importing pytest or collecting source under test. This is
        # an accidental-execution tripwire, not isolation against adversarial code.
        subprocess.Popen = forbidden
        socket.socket.bind = forbidden
        import pytest

        raise SystemExit(pytest.main(sys.argv[2:]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", action="append", default=[])
    parser.add_argument("--require-module", action="append", default=[])
    parser.add_argument("wheels", nargs="+")
    args = parser.parse_args()
    print(json.dumps(probe(args.wheels, modules=args.module, required_modules=args.require_module), indent=2))
