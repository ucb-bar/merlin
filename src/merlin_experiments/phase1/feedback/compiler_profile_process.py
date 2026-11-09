"""Fresh sealed original-owner workers for the unchanged interface protocol.

The parent ordinary emission observer owns the live product. This transport
returns actual child products, never reconstructs a saved emission completion,
and leaves source/object/instruction/numerical consumers unchanged.
"""

from __future__ import annotations

import argparse
import functools
import inspect
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from merlin.common import cpu_pool, invocation_record
from merlin.targetgen import capsule_common as CC
from merlin.targetgen import package_runtime as P
from merlin_experiments.phase2.contracts import StageGateError

from . import compiler_profile_child as CHILD
from . import compiler_profile_runtime as R

SCHEMA = "merlin.original_interface_child.v1"
ENV = "MERLIN_ORIGINAL_EMISSION_PROCESSES"


def enabled():
    choice = os.environ.get(ENV, "0")
    if choice not in ("", "0", "1"):
        raise ValueError(f"{ENV} requires 0 or 1")
    return choice == "1"


def _implementation():
    return [CHILD._pin(Path(module.__file__).resolve()) for module in (sys.modules[__name__], R, CC)]


def _write(path, document):
    path.write_text(json.dumps(document, sort_keys=True, allow_nan=False), encoding="utf-8")


def _job(document, spec):
    expected = {
        "schema",
        "package_dir",
        "target",
        "source",
        "generated",
        "artifact_name",
        "timeout",
        "deadline",
        "implementation",
    }
    if (
        type(document) is not dict
        or set(document) != expected
        or document["schema"] != SCHEMA
        or document["implementation"] != _implementation()
        or type(document["timeout"]) is not int
        or document["timeout"] <= 0
        or type(document["deadline"]) not in (int, float)
        or not math.isfinite(document["deadline"])
    ):
        raise StageGateError("original interface worker lost its closed source/protocol selection")
    generated = CHILD._plain(document["generated"])
    if (
        spec.parent.parent != generated
        or spec.name != "spec.json"
        or CHILD._pin(document["source"]["path"]) != document["source"]
        or Path(document["artifact_name"]).name != document["artifact_name"]
        or document["artifact_name"]
        in ("", ".", "..", "command_buffer.json", "input.interface.mlir", "lowered.target.mlir")
    ):
        raise StageGateError("original interface worker changed its source or private product ownership")
    return generated


def _products(generated, artifact_name):
    return {
        name: CHILD._pin(generated / filename)
        for name, filename in (
            ("input", "input.interface.mlir"),
            ("target", "lowered.target.mlir"),
            ("command_buffer", "command_buffer.json"),
            ("artifact", artifact_name),
        )
    }


def _read_products(job, result):
    generated = CHILD._plain(job["generated"])
    pins = _products(generated, job["artifact_name"])
    if pins != result.get("products") or pins["input"]["sha256"] != job["source"]["sha256"]:
        raise StageGateError("original interface worker returned another source or product")
    if not (generated / "lowered.target.mlir").read_text().strip():
        raise StageGateError("original interface worker omitted the target plane")
    buffer = json.loads((generated / "command_buffer.json").read_text())
    CC.schemas.validate_command_buffer(buffer, contract=None)
    CC.validate_interface_tensor_dtypes(buffer, (generated / "input.interface.mlir").read_text())
    if buffer.get("declined"):
        raise StageGateError("original interface worker cannot publish a declined buffer as emitted")
    artifact = (generated / job["artifact_name"]).read_text()
    if not artifact.strip():
        raise StageGateError("original interface worker omitted the target artifact")
    invocations = sorted((generated / "invocations").glob("*/invocation.json"))
    if not invocations or result.get("invocations") != [CHILD._pin(path) for path in invocations]:
        raise StageGateError("original interface worker lost its actual command invocations")
    for path in invocations:
        observed = invocation_record.verify(path)
        if (
            observed["kind"] != "subprocess"
            or observed["stage"] != "original_profile_package"
            or observed["status"] != "completed"
            or observed["returncode"] != 0
        ):
            raise StageGateError("original interface worker has an incomplete original package invocation")
    return buffer, artifact


def _command(spec, result, transport):
    # Select the actual parent's imported framework roots, not an archived
    # common-owner repo or mutable PYTHONPATH chosen by an unrelated launcher.
    roots = [str(Path(P.__file__).resolve().parents[2]), str(Path(R.__file__).resolve().parents[3])]
    bootstrap = (
        f"import sys;sys.path[:0]={roots!r};"
        "from merlin_experiments.phase1.feedback.compiler_profile_process import main;"
        "sys.exit(main())"
    )
    return [sys.executable, "-I", "-B", "-c", bootstrap, "--spec", str(spec), "--result", str(result), *transport.argv]


def lower(executor, pkg, interface, generated, *, timeout, artifact_name):
    """A live parent issues one fresh child and reopens its completion before return."""
    from merlin.targetgen.capsule_runner import _kill_tree

    if (
        type(executor) is not R._Executor
        or not executor.concurrent_entrypoints_v1()
        or P.active_package_executor() is not executor
        or not enabled()
        or cpu_pool.current_thread_owns_cpu_lease()
    ):
        raise StageGateError("interface child requires its unleased exact original execution owner")
    generated, interface = CHILD._plain(generated), CHILD._plain(interface)
    if generated.is_relative_to(pkg.directory):
        raise StageGateError("original interface child cannot stage inside the compiler")
    root = generated / ".original-interface-child"
    root.mkdir(mode=0o700)  # A second request cannot overwrite an earlier completion.
    spec, result = root / "spec.json", root / "result.json"
    job = dict(
        schema=SCHEMA,
        package_dir=str(pkg.directory),
        target=executor.target.target,
        source=CHILD._pin(interface),
        generated=str(generated),
        artifact_name=artifact_name,
        timeout=timeout,
        deadline=time.monotonic() + 3 * timeout + 90,
        implementation=_implementation(),
    )
    _write(spec, job)
    environment = cpu_pool.child_environment()
    with CHILD.prepare_model_child(spec_path=spec, environment=environment, package_dir=pkg.directory) as transport:
        if transport is None:
            raise StageGateError("interface child lost the original sealed capability")
        command = cpu_pool.worker_pool_command(_command(spec, result, transport), environment)
        with (root / "stdout").open("w") as stdout, (root / "stderr").open("w") as stderr:
            with invocation_record.observe(
                root,
                stage="original_interface_child",
                argv=command,
                inputs=(spec, interface),
                outputs=(result,),
                dependencies=tuple(Path(pin["path"]) for pin in job["implementation"]),
            ) as observed:
                process = subprocess.Popen(
                    command,
                    env=environment,
                    pass_fds=transport.pass_fds,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                )
                worker = None
                try:
                    worker = CHILD._process(process.pid)
                    process.wait(timeout=max(0.001, job["deadline"] - time.monotonic()))
                except BaseException:
                    _kill_tree(process.pid)
                    process.wait(timeout=10)
                    raise
                finally:
                    _write(
                        root / "process.json",
                        {"worker": worker, "returncode": process.returncode, "reaped": process.poll() is not None},
                    )
                observed.complete(subprocess.CompletedProcess(command, process.returncode))
        if process.returncode or not result.is_file():
            detail = (root / "stderr").read_text()[-2000:]
            raise StageGateError(f"original interface child exited {process.returncode}: {detail}")
        row = transport.verify_result(json.loads(result.read_bytes()))
        if (
            row.get("schema") != SCHEMA
            or row.get("worker") != worker
            or row.get("spec") != CHILD._pin(spec)
            or _job(job, spec) != generated
            or CHILD._pin(interface) != job["source"]
        ):
            raise StageGateError("original interface child completion changed its selected request")
        if row["status"] == "refused":
            error = row["failure"]
            if error["type"] == "BackendDeclined":
                raise P.BackendDeclined(**error["fields"])
            if error["type"] == "CertFailure":
                raise P.CertFailure(**error["fields"])
            raise StageGateError("original interface child returned an unknown refusal")
        if row["status"] != "emitted":
            raise StageGateError("original interface child returned an unknown status")
        return _read_products(job, row)


def main(argv=None):
    from merlin.targetgen.capsule_runner import _die_with_parent

    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--original-profile-fd", type=int, required=True)
    args = parser.parse_args(argv)
    _die_with_parent()
    raw = args.spec.read_bytes()
    with CHILD.restore_model_child(args.original_profile_fd, spec_path=args.spec, spec_bytes=raw) as receipt:
        worker = CHILD._process(os.getpid())
        job = json.loads(raw)
        generated = _job(job, args.spec)
        if args.result != args.spec.parent / "result.json":
            raise StageGateError("original interface child has a foreign result owner")
        package = P.load_package(job["package_dir"])
        body = CC.lower_interface.__wrapped__
        if (
            body.__module__ != CC.__name__
            or body.__qualname__ != "lower_interface"
            or Path(inspect.getsourcefile(body)).resolve() != Path(CC.__file__).resolve()
        ):
            raise StageGateError("original interface child lost the actual ordinary protocol body")
        try:
            if time.monotonic() >= job["deadline"]:
                raise subprocess.TimeoutExpired("original interface child", job["timeout"])
            body(
                package,
                job["source"]["path"],
                generated,
                contract=None,
                timeout=job["timeout"],
                artifact_name=job["artifact_name"],
                invoke=functools.partial(P.run_entrypoint, invocation_directory=generated),
            )
            row = {
                "status": "emitted",
                "products": _products(generated, job["artifact_name"]),
                "invocations": [
                    CHILD._pin(path) for path in sorted((generated / "invocations").glob("*/invocation.json"))
                ],
            }
        except P.BackendDeclined as error:
            row = {"status": "refused", "failure": {"type": "BackendDeclined", "fields": error.to_dict()}}
        except P.CertFailure as error:
            row = {
                "status": "refused",
                "failure": {
                    "type": "CertFailure",
                    "fields": {"plane": error.plane, "category": error.category.value, "detail": error.detail},
                },
            }
        if _job(job, args.spec) != generated:
            raise StageGateError("original interface child selection changed before final checks")
    # Publication follows restoration's original final full verification.
    _write(
        args.result,
        {"schema": SCHEMA, "spec": CHILD._pin(args.spec), "worker": worker, **row, "original_compiler_child": receipt},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
