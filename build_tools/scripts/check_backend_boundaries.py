"""Run pinned OOT boundary tests against installed wheels outside source checkouts.

This checks the explicitly selected software boundary cases. It downloads no native
toolchains and establishes no device execution or numerical qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from email.parser import BytesParser
from pathlib import Path
from zipfile import ZipFile

SCHEMA = "merlin.backend-boundaries.v1"
VERSIONS = ("0.68.0", "0.65.0")
FIELDS = {
    "id",
    "repository",
    "revision",
    "xdsl",
    "provider_module",
    "tests",
    "fixtures",
    "selection",
    "installed_service",
}

PROBE = r"""
import hashlib, importlib, json, sys
from email.parser import BytesParser
from importlib.metadata import distribution, entry_points, version
from pathlib import Path
from zipfile import ZipFile

assert version("xdsl") == sys.argv[1], (version("xdsl"), sys.argv[1])
rows = []
for wheel_path in sys.argv[2:]:
    wheel = Path(wheel_path)
    with ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata = next(name for name in names if name.endswith(".dist-info/METADATA"))
        message = BytesParser().parsebytes(archive.read(metadata))
        installed = distribution(message["Name"])
        assert installed.version == message["Version"]
        count = 0
        for name in names:
            if name.endswith("/") or name.endswith(".dist-info/RECORD"):
                continue
            assert ".data/" not in name, (wheel.name, "unsupported wheel data relocation", name)
            actual = Path(installed.locate_file(name)).resolve()
            assert actual.is_relative_to(Path(sys.prefix).resolve()), actual
            assert actual.read_bytes() == archive.read(name), (wheel.name, name)
            count += 1
        rows.append({"distribution": message["Name"], "version": installed.version,
                     "members_exact": count, "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest()})
import merlin
assert Path(merlin.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
print(json.dumps({"python": sys.executable, "python_version": sys.version, "prefix": sys.prefix,
                  "pytest": version("pytest"), "xdsl": version("xdsl"), "wheels": rows}))
"""

GUARD = """
from pathlib import Path
import importlib
import sys
import merlin

PREFIX = Path(sys.prefix).resolve()
ROOTS = [Path(path).resolve() for path in {roots!r}]
PROVIDER = importlib.import_module({provider!r})
assert Path(merlin.__file__).resolve().is_relative_to(PREFIX)
assert Path(PROVIDER.__file__).resolve().is_relative_to(PREFIX)

def pytest_collection_modifyitems(items):
    if {service!r}:
        for item in items:
            item.module.ROOT = Path(PROVIDER.__file__).resolve().parent

def pytest_sessionfinish(session, exitstatus):
    for name, module in list(sys.modules.items()):
        value = getattr(module, "__file__", None)
        if not value:
            continue
        path = Path(value).resolve()
        assert not any(path.is_relative_to(root) for root in ROOTS), (name, str(path))
"""


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(value: object) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"expected a nonempty POSIX relative file path: {value!r}")
    path = Path(value)
    if (
        value.startswith("/")
        or not path.parts
        or path.is_absolute()
        or any(part in {".", ".."} for part in path.parts)
        or ":" in value
    ):
        raise ValueError(f"path escapes its selected source: {value!r}")
    if path.name == "conftest.py":
        raise ValueError("the installed-owner guard cannot be replaced by a selected fixture")
    return path


def pins(path: Path) -> list[dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if set(document) != {"schema", "backends"} or document["schema"] != SCHEMA:
        raise ValueError(f"expected schema {SCHEMA!r} and a backends roster")
    rows = document["backends"]
    if not isinstance(rows, list) or len(rows) != 6:
        raise ValueError("the reviewed boundary roster must select exactly six providers")
    identities = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != FIELDS:
            raise ValueError(f"malformed backend pin: {row!r}")
        identity = row["id"]
        if (
            not isinstance(identity, str)
            or not identity
            or not all(c.isascii() and (c.isalnum() or c == "-") for c in identity)
        ):
            raise ValueError(f"unsafe provider identity: {identity!r}")
        if identity in identities:
            raise ValueError(f"duplicate provider pin: {identity}")
        identities.add(identity)
        revision = row["revision"]
        if not isinstance(revision, str) or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError(f"{identity}: revision must be a full lowercase Git SHA")
        if not isinstance(row["repository"], str) or not row["repository"].startswith("https://"):
            raise ValueError(f"{identity}: expected an HTTPS source repository")
        if row["xdsl"] not in VERSIONS:
            raise ValueError(f"{identity}: unadmitted xDSL version {row['xdsl']!r}")
        if not isinstance(row["provider_module"], str) or not all(
            p.isidentifier() for p in row["provider_module"].split(".")
        ):
            raise ValueError(f"{identity}: expected a dotted provider module")
        for key in ("tests", "fixtures"):
            if not isinstance(row[key], list) or key == "tests" and not row[key]:
                raise ValueError(f"{identity}: {key} must be a file roster")
            for name in row[key]:
                relative(name)
        if row["selection"] is not None and not isinstance(row["selection"], str):
            raise ValueError(f"{identity}: selection must be a pytest expression or null")
        if not isinstance(row["installed_service"], bool):
            raise ValueError(f"{identity}: installed_service must be boolean")
    return rows


def git(source: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(source), *arguments], check=True, capture_output=True, text=True
    ).stdout.strip()


def clean_source(source: Path, revision: str) -> None:
    if git(source, "rev-parse", "HEAD") != revision:
        raise ValueError(f"source HEAD does not match selected revision: {source}")
    if git(source, "status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none"):
        raise ValueError(f"selected source has tracked, untracked or submodule changes: {source}")


def run(argv: list[str], *, cwd: Path, log: Path, env: dict[str, str], records: list) -> None:
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(argv, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=900)
    records.append({"argv": argv, "returncode": result.returncode, "log": str(log)})
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log}")


def build_source(row: dict, source: Path, destination: Path, env: dict, commands: list) -> Path:
    clean_source(source, row["revision"])
    destination.mkdir()
    run(
        [sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", str(destination), str(source)],
        cwd=destination,
        log=destination / "build.log",
        env=env,
        records=commands,
    )
    clean_source(source, row["revision"])
    wheels = list(destination.glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError(f"expected exactly one provider wheel: {destination}")
    with ZipFile(wheels[0]) as archive:
        for name in archive.namelist():
            if name.endswith("/"):
                continue
            path = Path(name)
            if name.startswith("/") or path.is_absolute() or ".." in path.parts or ":" in name:
                raise ValueError(f"unsafe wheel member: {name}")
    return wheels[0]


def copy_inputs(row: dict, source: Path, destination: Path) -> list[dict]:
    result = []
    for name in sorted(set(row["tests"] + row["fixtures"])):
        path = source / relative(name)
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(source.resolve()):
            raise ValueError(f"selected fixture is not a contained regular file: {path}")
        git(source, "ls-files", "--error-unmatch", "--", name)
        target = destination / relative(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        result.append({"path": name, "sha256": sha(target)})
    return result


def test_lane(
    row: dict, source: Path, python: Path, temporary: Path, output: Path, env: dict, commands: list, roots: list
) -> dict:
    directory = temporary / row["id"]
    directory.mkdir()
    files = copy_inputs(row, source, directory)
    (directory / "conftest.py").write_text(
        GUARD.format(roots=roots, provider=row["provider_module"], service=row["installed_service"]),
        encoding="utf-8",
    )
    junit = output / (row["id"] + ".xml")
    args = [
        str(python),
        "-I",
        "-m",
        "pytest",
        "-q",
        "-rs",
        "-c",
        str(directory / "pytest.ini"),
        "--junitxml",
        str(junit),
        *row["tests"],
    ]
    (directory / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    if row["selection"]:
        args.extend(["-k", row["selection"]])
    run(args, cwd=directory, log=output / (row["id"] + ".log"), env=env, records=commands)
    suites = list(ET.parse(junit).getroot().iter("testsuite"))
    try:
        counts = {
            key: sum(int(suite.attrib[key]) for suite in suites) for key in ("tests", "failures", "errors", "skipped")
        }
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"JUnit must explicitly record valid tests/failures/errors/skipped counts: {row['id']}"
        ) from exc
    if counts["tests"] <= 0 or any(counts[key] for key in ("failures", "errors", "skipped")):
        raise ValueError(f"selected boundary cases must execute without skips: {row['id']}: {counts}")
    clean_source(source, row["revision"])
    return {
        "id": row["id"],
        "revision": row["revision"],
        "xdsl": row["xdsl"],
        "counts": counts,
        "selection": row["selection"],
        "fixtures": files,
        "junit": str(junit),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--pins", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", action="append", default=[], metavar="ID=PATH")
    parser.add_argument("--python068", type=Path)
    parser.add_argument("--python065", type=Path)
    args = parser.parse_args(argv)
    pins_sha256 = sha(args.pins)
    rows = pins(args.pins)
    if sha(args.pins) != pins_sha256:
        raise ValueError("selected pin manifest changed during intake")
    wheel = args.wheel.resolve(strict=True)
    with ZipFile(wheel) as archive:
        metadata = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(metadata) != 1 or BytesParser().parsebytes(archive.read(metadata[0]))["Name"].lower() != "merlin":
            raise ValueError("--wheel must select the main Merlin distribution")
    selected = {}
    for value in args.source:
        identity, separator, path = value.partition("=")
        if not separator or identity in selected or identity not in {row["id"] for row in rows}:
            parser.error(f"invalid or duplicate --source selection: {value}")
        selected[identity] = Path(path).resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": SCHEMA,
        "scope": "Pinned installed software boundary cases; no native/device execution qualification",
        "main_wheel_sha256": sha(wheel),
        "pins_sha256": pins_sha256,
        "commands": [],
        "lanes": [],
        "passed": False,
    }
    env = dict(os.environ)
    for key in (
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTEST_ADDOPTS",
        "PYTEST_PLUGINS",
        "MERLIN_REPO_ROOT",
        "MERLIN_TARGET_PATH",
    ):
        env.pop(key, None)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    try:
        with tempfile.TemporaryDirectory(prefix="merlin-backend-boundaries-") as temporary_name:
            temporary = Path(temporary_name)
            sources = {}
            provider_wheels = {}
            for row in rows:
                identity = row["id"]
                source = selected.get(identity, output / (identity + "-source"))
                if identity not in selected:
                    run(
                        ["git", "init", str(source)],
                        cwd=output,
                        log=output / (identity + "-init.log"),
                        env=env,
                        records=report["commands"],
                    )
                    run(
                        ["git", "-C", str(source), "remote", "add", "origin", row["repository"]],
                        cwd=output,
                        log=output / (identity + "-remote.log"),
                        env=env,
                        records=report["commands"],
                    )
                    run(
                        ["git", "-C", str(source), "fetch", "--depth=1", "origin", row["revision"]],
                        cwd=output,
                        log=output / (identity + "-fetch.log"),
                        env=env,
                        records=report["commands"],
                    )
                    run(
                        ["git", "-C", str(source), "checkout", "--detach", "FETCH_HEAD"],
                        cwd=output,
                        log=output / (identity + "-checkout.log"),
                        env=env,
                        records=report["commands"],
                    )
                sources[identity] = source
                provider_wheels[identity] = build_source(
                    row, source, output / (identity + "-wheel"), env, report["commands"]
                )
            interpreters = {"0.68.0": args.python068, "0.65.0": args.python065}
            for version in VERSIONS:
                grouped = [row for row in rows if row["xdsl"] == version]
                if not grouped:
                    continue
                python = interpreters[version]
                if python is None:
                    directory = temporary / ("env-" + version)
                    run(
                        [sys.executable, "-m", "venv", str(directory)],
                        cwd=temporary,
                        log=output / ("venv-" + version + ".log"),
                        env=env,
                        records=report["commands"],
                    )
                    python = directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
                    run(
                        [
                            str(python),
                            "-m",
                            "pip",
                            "install",
                            "--disable-pip-version-check",
                            "--no-input",
                            "--no-cache-dir",
                            str(wheel),
                            "xdsl==" + version,
                            "pytest",
                            *[str(provider_wheels[row["id"]]) for row in grouped],
                        ],
                        cwd=temporary,
                        log=output / ("install-" + version + ".log"),
                        env=env,
                        records=report["commands"],
                    )
                else:
                    python = python.absolute()
                interpreters[version] = python
                run(
                    [str(python), "-I", "-m", "pip", "check"],
                    cwd=temporary,
                    log=output / ("dependencies-" + version + ".log"),
                    env=env,
                    records=report["commands"],
                )
                run(
                    [
                        str(python),
                        "-I",
                        "-c",
                        PROBE,
                        version,
                        str(wheel),
                        *[str(provider_wheels[row["id"]]) for row in grouped],
                    ],
                    cwd=temporary,
                    log=output / ("payload-" + version + ".json"),
                    env=env,
                    records=report["commands"],
                )
            roots = [str(Path(__file__).resolve().parents[2]), *[str(source.resolve()) for source in sources.values()]]
            if any(temporary.resolve().is_relative_to(Path(root)) for root in roots):
                raise ValueError("test and environment temporary storage must be outside all source checkouts")
            for row in rows:
                report["lanes"].append(
                    test_lane(
                        row,
                        sources[row["id"]],
                        interpreters[row["xdsl"]],
                        temporary,
                        output,
                        env,
                        report["commands"],
                        roots,
                    )
                )
                print(f"{row['id']}: {report['lanes'][-1]['counts']['tests']} boundary checks passed", flush=True)
            for version in VERSIONS:
                grouped = [row for row in rows if row["xdsl"] == version]
                if grouped:
                    run(
                        [
                            str(interpreters[version]),
                            "-I",
                            "-c",
                            PROBE,
                            version,
                            str(wheel),
                            *[str(provider_wheels[row["id"]]) for row in grouped],
                        ],
                        cwd=temporary,
                        log=output / ("payload-after-" + version + ".json"),
                        env=env,
                        records=report["commands"],
                    )
            if sha(wheel) != report["main_wheel_sha256"] or sha(args.pins) != report["pins_sha256"]:
                raise ValueError("selected main wheel or pin manifest changed during the run")
            for row in rows:
                clean_source(sources[row["id"]], row["revision"])
            report["passed"] = True
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        report["error"] = str(exc)
        print(f"backend boundaries: {exc}", file=sys.stderr)
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
