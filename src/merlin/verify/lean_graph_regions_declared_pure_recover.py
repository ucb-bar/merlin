"""Offline bind the reviewed successful Examples output; no native invocation.

The stopped observer profile remains PARTIAL. Recovery relies explicitly on
trusted exclusive ownership/immutability of its private produced objects and
the retained source/command/pre/post/trace association, not a compilation theorem.
No SDK/runtime file is opened or expected runtime hash refreshed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import stat
from pathlib import Path


def private_ref(path, directory):
    if not path.is_relative_to(directory):
        raise ValueError("recovery read outside exclusively owned private outputs")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("private output is not a regular file")
        data = stream.read()
        after = os.fstat(stream.fileno())
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in fields):
        raise ValueError("private output changed during offline association read")
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    api = runpy.run_path(
        str(Path(__file__).with_name("lean_graph_regions_declared_loader_plan.py")),
        run_name="_private_recovery_original_refs",
    )
    reader = runpy.run_path(str(Path(__file__).with_name("lean_graph_regions_declared_pure_trace.py")))
    reviewer = runpy.run_path(str(Path(__file__).with_name("lean_graph_regions_declared_pure_review.py")))
    root, profile = args.root.resolve(strict=True), args.profile.resolve(strict=True)
    build = profile / "build"
    context = api["parse_json"]((build / "CONTEXT.json").read_bytes())
    if context["schema"] != "merlin.graph-regions.original-module-runtime-development.v1":
        raise ValueError("recovery is limited to the new reviewed3276 context")
    guard = private_ref(profile / "controller-snapshot/lean_graph_regions_declared_pure_trace.py", profile)
    current_guard = private_ref(
        Path(__file__).with_name("lean_graph_regions_declared_pure_trace.py"), Path(__file__).parent
    )
    reviewed = reviewer["reviewed_pair"](root, args.review, context["original_provenance"], current_guard, api)
    if guard != reviewed["guard_identity"]:
        raise ValueError("recorded producer guard differs from the reviewed source identity")
    observation = api["parse_json"](api["read_original"](reviewed["input_ref"]))
    module = "Merlin.GraphRegionsDeclaredCaptureExamples"
    sources = api["parse_json"]((build / "SOURCES.json").read_bytes())
    source = sources[module]
    api["read_original"](source["source"])
    api["read_original"](source["original"])
    command_directory = build / "commands/0060-Merlin.GraphRegionsDeclaredCaptureExamples-compile"
    process = api["parse_json"]((command_directory / "PROCESS.json").read_bytes())
    precheck = api["parse_json"]((command_directory / "PRECHECK.json").read_bytes())
    postcheck = api["parse_json"]((command_directory / "POSTCHECK.json").read_bytes())
    target = build / "compiled/Merlin/GraphRegionsDeclaredCaptureExamples.olean"
    expected_command = context["loader"] + [
        "-j1",
        "--root=" + str(build / "source"),
        "-o",
        str(target),
        source["source"]["path"],
    ]
    if process != observation["process"] or process["command"] != expected_command or process["returncode"] != 0:
        raise ValueError("retained command/source/object/return association differs")
    if postcheck != observation["postcheck"]:
        raise ValueError("retained postcheck differs from reviewed evidence")
    for checked in (precheck, postcheck):
        if checked["original_sdk_runtime_bindings_checked"] != 3276 or not checked["all_bound_bytes_unchanged"]:
            raise ValueError("retained original selected bindings were not complete")
    index = api["parse_json"](sorted(build.glob("INDEX-*.json"))[-1].read_bytes())
    if module in index["project"]:
        raise ValueError("old profile already bound this module; never overwrite/promote it")
    for item in index["project"].values():
        for reference in item["outputs"].values():
            api["read_original"](reference)
    raw = (command_directory / "stderr.txt").read_text()
    runtime = context["original_provenance"]["original_runtime_refs"]
    misses = reader["baseline_misses"](args.baseline.read_text(), {pin["path"] for pin in runtime})
    initializers = {"initialize_" + name.replace(".", "_") for name in index["project"]}
    recovered_trace = reader["parse"](
        raw,
        runtime,
        runtime[0]["path"],
        context["loader"][0],
        misses | {reviewed["pair"]},
        expected_pid=process["pid"],
        project_initializers=initializers,
    )
    inputs = {}
    for dependency in index["actual_source_import_graph"][module]["project"]:
        inputs.update(index["project"][dependency]["project_inputs"])
        inputs[dependency] = index["project"][dependency]["outputs"]
    outputs = {}
    for suffix in (".olean", ".olean.private", ".olean.server", ".ilean"):
        path = target.with_suffix(suffix)
        if path.is_file():
            outputs[suffix] = private_ref(path, build / "compiled")
    if ".olean" not in outputs:
        raise ValueError("retained successful process lacks its produced object")
    process_ref = private_ref(command_directory / "PROCESS.json", profile)
    object_report_path = build / "commands/0059-Merlin.GraphRegionsDeclaredCaptureExamples-object-imports/stdout.txt"
    selected = api["parse_json"]((build / "ORIGINAL-BINDINGS.json").read_bytes())
    actual_imports, standard_imports = {}, []
    for line in object_report_path.read_text().splitlines():
        path = Path(line)
        if not path.is_absolute():
            raise ValueError("unexpected retained object-reporter line")
        if path.is_relative_to(build / "compiled"):
            dependency = ".".join(path.relative_to(build / "compiled").with_suffix("").parts)
            if dependency not in index["project"] or index["project"][dependency]["outputs"][".olean"]["path"] != line:
                raise ValueError("actual reported project import lacks its exact binding")
            actual_imports[dependency] = index["project"][dependency]["outputs"]
        elif line in selected and path.suffix == ".olean":
            standard_imports.append(selected[line])
        else:
            raise ValueError("retained actual import is outside original selected module/object set")
    index["project"][module] = dict(
        source=source["source"],
        project_inputs=inputs,
        outputs=outputs,
        actual_object_imports=actual_imports,
        actual_standard_object_imports=standard_imports,
        object_report=private_ref(object_report_path, profile),
        compilation=dict(
            kind="offline recovered rc0 source/command/object association",
            process=process_ref,
            trust="exclusive private output ownership/immutability + original loader/compiler/host transport",
        ),
        qualified=False,
    )
    index["status"] = "NEW OFFLINE CACHE ASSOCIATION ONLY; original observer profile remains PARTIAL"
    index["offline_recovery_review"] = reviewed
    index["sdk_runtime_files_read"] = 0
    index["native_invocations"] = 0
    (args.output / "CACHE.json").write_text(json.dumps(index, indent=2) + "\n")
    (args.output / "RUNTIME.json").write_text(json.dumps(recovered_trace, indent=2) + "\n")
    (args.output / "RECOVERY.json").write_text(
        json.dumps(
            dict(
                status=index["status"],
                module=module,
                source=source,
                process=process_ref,
                outputs=outputs,
                original_profile_status_unchanged=True,
                input_document_ref=reviewed["input_ref"],
                sdk_runtime_files_read=0,
                native_invocations=0,
                new_runtime_expected_hashes=False,
                qualification=False,
                trust=[
                    "private produced-object association/immutability",
                    "retained original pre/post guards",
                    "pinned loader/compiler/module reporting",
                    "OS/kernel/host transport",
                ],
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(dict(module=module, produced_objects=len(outputs), native_invocations=0, sdk_runtime_files_read=0))
    )


if __name__ == "__main__":
    main()
