"""Installed payload/origin checks and pytest guard for qualify_installed.py.

Copied byte-for-byte into the retained external venv. Never used as an isolation sandbox.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import os
import sys
import sysconfig
import zipfile
from importlib.metadata import entry_points
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


_TEST_SOURCES = None


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        assert key not in result, "archived test roster repeats a field"
        result[key] = value
    return result


def _read_test_roster():
    selected = os.environ.get("MERLIN_TEST_ARCHIVED_SOURCES")
    if selected is None:
        return None, b"", {}
    path = Path(selected)
    assert path.is_absolute() and path.is_file() and not path.is_symlink(), "archived test roster is unavailable"
    with path.open("rb") as stream:
        raw = stream.read(1048577)
    assert len(raw) <= 1048576, "archived test roster exceeds its byte bound"
    value = json.loads(raw, object_pairs_hook=_unique_object)
    assert (
        type(value) is dict
        and set(value) == {"schema", "files"}
        and value["schema"] == "merlin.installed_test_sources.v1"
        and type(value["files"]) is list
        and len(value["files"]) <= 10000
    ), "archived tests need their closed source roster"
    result = {}
    paths = set()
    for row in value["files"]:
        assert type(row) is dict and set(row) == {"module", "path", "sha256"}, "archived test source is incomplete"
        assert all(type(part) is str for part in row.values()), "archived test source has unsupported fields"
        name, source, digest = row["module"], Path(row["path"]), row["sha256"]
        assert (
            source.is_absolute()
            and source.suffix == ".py"
            and source.stem.startswith("test_")
            and name.split(".")[-1] == source.stem
            and name not in result
            and source not in paths
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
        ), "archived test source identity is unsupported or repeated"
        if name.startswith(("merlin.", "merlin_experiments.")):
            assert name.startswith(("merlin.tests.", "merlin_experiments.tests.")), (
                "production modules cannot be archived tests"
            )
        assert source.is_file() and not source.is_symlink(), "archived test source is unavailable"
        with source.open("rb") as stream:
            payload = stream.read(16777217)
        assert len(payload) <= 16777216, "archived test source exceeds its byte bound"
        assert hashlib.sha256(payload).hexdigest() == digest, "archived test source bytes changed"
        result[name] = (source.resolve(), digest)
        paths.add(source)
    return selected, raw, result


def assert_installed_origins(*, archived_tests=None):
    site = Path(sysconfig.get_path("purelib")).resolve()
    checked = []
    for name, module in tuple(sys.modules.items()):
        if name in {"merlin", "merlin_experiments"} or name.startswith(("merlin.", "merlin_experiments.")):
            filename = getattr(module, "__file__", None)
            if filename:
                source = Path(filename).resolve()
                if not source.is_relative_to(site) and archived_tests is not None:
                    assert name in archived_tests and source == archived_tests[name][0], (name, filename)
                    continue
                assert source.is_relative_to(site), (name, filename)
                checked.append(name)
    return checked


def pytest_sessionstart(session):
    global _TEST_SOURCES
    _TEST_SOURCES = _read_test_roster()
    # importlib-mode collection can otherwise invent an archive-only parent
    # named merlin, hiding the installed compiler package from these tests.
    importlib.import_module("merlin")
    if any(name.startswith("merlin_experiments.") for name in _TEST_SOURCES[2]):
        importlib.import_module("merlin_experiments")
    assert_installed_origins(archived_tests=_TEST_SOURCES[2])


def pytest_sessionfinish(session, exitstatus):
    if _TEST_SOURCES is None:
        assert_installed_origins()
        return
    current = _read_test_roster()
    assert current == _TEST_SOURCES, "archived test roster changed during execution"
    assert_installed_origins(archived_tests=current[2])


def probe(wheels, *, modules=(), required_modules=(), required_entry_points=()):
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
    for requirement in required_entry_points:
        group_and_name, separator, value = requirement.partition("=")
        group, name_separator, name = group_and_name.partition(":")
        assert separator and name_separator and group and name and value, requirement
        providers = tuple(entry_points(group=group))
        assert len(providers) == 1, (group, providers)
        provider = providers[0]
        assert (provider.name, provider.value) == (name, value), (requirement, provider)
        provider.load()
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
    parser.add_argument("--require-entry-point", action="append", default=[])
    parser.add_argument("wheels", nargs="+")
    args = parser.parse_args()
    print(
        json.dumps(
            probe(
                args.wheels,
                modules=args.module,
                required_modules=args.require_module,
                required_entry_points=args.require_entry_point,
            ),
            indent=2,
        )
    )
