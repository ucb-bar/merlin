"""Observe the actual runtime-host driver argv around one post-freeze execution.

The administrator installs this hook at the driver invocation, inside the
existing FireSim screen/script lifecycle. It neither discovers nor launches a
run farm. Its command is supplied by that reviewed lifecycle, never by a guest
ELF or a request field. Observations do not certify numerics or isolation.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time
from pathlib import Path

from . import elf_custody as C

SCHEMA = "merlin.post_freeze.firesim.runtime_driver.v1"


def execute(*, request, request_sha256, model, download_record, driver_script,
            argv, cwd, output, wall_seconds):
    """Retain failures and before/after observations on the actual driver host."""
    output, cwd = Path(output).absolute(), Path(cwd).absolute()
    if output.exists() or output.resolve() != output or cwd.resolve() != cwd:
        raise ValueError("runtime hook needs an unused ordinary output and actual working directory")
    output.mkdir(mode=0o700)
    started = time.monotonic()
    record = {"schema": SCHEMA, "status": "refused", "runtime": C.runtime_identity(),
              "scientific_qualification": False, "returncode": None, "timed_out": False}
    process = None
    before = None
    try:
        selected, _, _ = C.selected_model(request, request_sha256, model)
        contract = C.contract(selected)
        execution = contract.get("runtime_execution", {})
        C.require_identity(C.artifact(__file__), execution.get("helper", {}))
        if (not isinstance(argv, list) or not argv
                or not all(isinstance(v, str) and v and "\x00" not in v for v in argv)):
            raise ValueError("runtime hook requires the concrete driver argv")
        executable = Path(argv[0])
        executable = executable if executable.is_absolute() else cwd / executable
        driver = C.artifact(executable)
        C.require_identity(driver, execution.get("driver", {}))
        budget = selected.get("budget", {}).get("job_wall_s")
        if (type(wall_seconds) is not int or type(budget) is not int
                or not 0 < wall_seconds <= budget <= 21600):
            raise ValueError("runtime deadline must fit the selected remaining job budget")
        C.write(output / "simulator-argv.json", argv)
        before = C.before(download_record, request, request_sha256, model,
                          output / "simulator-argv.json", cwd,
                          contract["boot_argument_prefix"], driver_script)
        C.write(output / "elf-before.json", before)
        record.update(driver=driver, observer=C.artifact(__file__),
                      before=C.artifact(output / "elf-before.json"),
                      argv=argv, working_directory=str(cwd), wall_seconds=wall_seconds)
        # FireSim's surrounding script owns stdout/UART collection. Do not buffer
        # it or invent a simulator command by parsing a shell script.
        if time.monotonic() - started >= wall_seconds:
            raise TimeoutError("runtime preparation exhausted remaining job budget")
        process = subprocess.Popen(argv, cwd=cwd, start_new_session=True)
        try:
            record["returncode"] = process.wait(timeout=max(0.001, wall_seconds - (time.monotonic() - started)))
        except subprocess.TimeoutExpired:
            record["timed_out"] = True
            os.killpg(process.pid, signal.SIGKILL)
            record["returncode"] = process.wait()
        record["status"] = "driver_finished"
    except Exception as error:
        record["error"] = {"type": type(error).__name__, "reason": str(error)}
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            record["returncode"] = process.wait()
        if before is not None:
            try:
                C.write(output / "elf-after.json", C.after(output / "elf-before.json"))
                if C.artifact(record["driver"]["path"]) != record["driver"]:
                    raise ValueError("actual simulator driver changed during execution")
                record["after"] = C.artifact(output / "elf-after.json")
            except Exception as error:
                record["status"] = "custody_failed"
                record["custody_error"] = {"type": type(error).__name__, "reason": str(error)}
        record["elapsed_wall_seconds"] = time.monotonic() - started
        C.write(output / "runtime-driver.json", record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("request", "download-record", "driver-script", "cwd", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--wall-seconds", type=int, required=True)
    parser.add_argument("driver_argv", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    driver_argv = args.driver_argv
    if driver_argv[:1] == ["--"]:
        driver_argv = driver_argv[1:]
    result = execute(request=args.request, request_sha256=args.request_sha256,
                     model=args.model, download_record=args.download_record,
                     driver_script=args.driver_script, argv=driver_argv, cwd=args.cwd,
                     output=args.output, wall_seconds=args.wall_seconds)
    if result["status"] != "driver_finished" or result["timed_out"]:
        return 2
    code = result["returncode"]
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    raise SystemExit(main())
