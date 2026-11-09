"""Independent fixed public-framework argument getter preparation and replay.

This owner binds actual public API/header-generation sources and native getter
processes. Installed framework, SDK, linker and transitive dependency historical
correspondence remain unqualified. It supplies source argument observations to
the versioned schema intake, never compiler or numerical implementations.
"""

from __future__ import annotations

import json
import resource
import shlex
from pathlib import Path

from merlin.common import invocation_record as I
from merlin.common.paths import module_source_path
from merlin.targetgen import torch_tensor_argument_observer as T
from merlin.targetgen.frontend_operator_effects import original_tensor_argument_requests

from .command_intake import _git
from .rtl_intake import RtlIntakeRefusal, _json, _outside, _plain

SCHEMA = "merlin.independent_tensor_argument_getter.v1"
ENVIRONMENT = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}


def _run(argv, *, owner, stage, inputs=(), outputs=(), dependencies=(), timeout=60, **kwargs):
    result = I.run(
        argv,
        directory=owner,
        stage=stage,
        inputs=inputs,
        outputs=outputs,
        dependencies=dependencies,
        cwd=owner,
        env=ENVIRONMENT,
        capture_output=True,
        timeout=timeout,
        **kwargs,
    )
    result.check_returncode()
    receipt = next(
        path
        for path in owner.glob("invocations/*/invocation.json")
        if json.loads(path.read_bytes())["argv"] == list(argv)
    )
    I.require_environment(receipt, environment=ENVIRONMENT)
    return result, str(receipt)


def _compile_limit():
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _compile_command(compiler, sdk, cpp, getter, dependencies):
    includes, library = Path(sdk["torch_include"]), Path(sdk["torch_library"])
    return [
        str(compiler),
        "-O0",
        "-std=c++17",
        "-shared",
        "-fPIC",
        "-MMD",
        "-MF",
        str(dependencies),
        "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(sdk["cxx11_abi"])),
        "-I" + str(includes),
        "-I" + str(includes / "torch/csrc/api/include"),
        "-I" + sdk["python_include"],
        str(cpp),
        "-L" + str(library),
        "-Wl,-rpath," + str(library),
        "-ltorch_python",
        "-ltorch_cpu",
        "-lc10",
        "-o",
        str(getter),
    ]


def _dependency_path(raw):
    """Normalize compiler-emitted parent components only after each link check."""
    source = Path(raw)
    if not source.is_absolute():
        raise RtlIntakeRefusal("Tensor argument compiler dependency needs an actual absolute source path")
    current = Path(source.anchor)
    for part in source.parts[1:]:
        current = current.parent if part == ".." else current / part
        if current.is_symlink():
            raise RtlIntakeRefusal("Tensor argument compiler dependency cannot traverse a symlink")
    return _plain(current)


def prepare_getter(*, python, compiler, checkout, commit, forbidden, output):
    """Build only the fixed getter; public source APIs choose the numeric guard."""
    output.mkdir(mode=0o700)
    observer = module_source_path("merlin.targetgen.torch_tensor_argument_observer")
    compiler_path = Path(compiler).absolute()
    _outside(compiler_path, forbidden)
    compiler = _plain(compiler_path.resolve(strict=True))
    for path in (observer, compiler):
        _outside(path, forbidden)
    if _git(checkout, "rev-parse", "HEAD") != commit or _git(checkout, "status", "--porcelain", "--untracked-files=no"):
        raise RtlIntakeRefusal("Tensor argument APIs require the selected clean public checkout")
    public = output / "public-source"
    sources, invocations = [], []
    for relative in T.PUBLIC_PATHS:
        result, invocation = _run(
            ["/usr/bin/git", "-C", str(checkout), "show", commit + ":" + relative],
            owner=output,
            stage="public_tensor_argument_source_readback",
            inputs=(observer,),
        )
        destination = public / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(result.stdout)
        sources.append(
            {
                "relative": relative,
                "path": str(destination),
                "blob": _git(checkout, "rev-parse", commit + ":" + relative),
                "invocation": invocation,
            }
        )
        invocations.append(invocation)
    result, invocation = _run(
        [str(python), "-I", str(observer), "sdk"],
        owner=output,
        stage="native_tensor_argument_sdk_selection",
        inputs=(observer,),
    )
    sdk = json.loads(result.stdout)
    if sdk["reported_git_version"] != commit:
        raise RtlIntakeRefusal("native framework reported revision differs from selected public declarations")
    sdk_path = output / "sdk.json"
    sdk_path.write_bytes(_json(sdk))
    invocations.append(invocation)
    generated = output / "generated-headers"
    _, invocation = _run(
        [str(python), "-I", str(observer), "headers", str(public), str(generated)],
        owner=output,
        stage="public_tensor_argument_header_generation",
        inputs=(observer, *(public / relative for relative in (*T.PRIMARY_HEADERS, "setup.py"))),
        outputs=tuple(generated / relative for relative in T.PRIMARY_HEADERS),
    )
    invocations.append(invocation)
    includes = Path(sdk["torch_include"])
    for relative in T.PRIMARY_HEADERS:
        actual = _plain(includes / relative)
        _outside(actual, forbidden)
        if actual.read_bytes() != (generated / relative).read_bytes():
            raise RtlIntakeRefusal("native SDK header differs from the exact public packaging product")
    cpp, dependencies = output / "getter.cpp", output / "getter.d"
    cpp.write_text(T.CPP_SOURCE)
    suffix = sdk["extension_suffix"]
    if not isinstance(suffix, str) or "/" in suffix or not suffix.endswith(".so") or type(sdk["cxx11_abi"]) is not bool:
        raise RtlIntakeRefusal("native framework extension metadata is unsupported")
    getter = output / ("native_tensor_argument_getter" + suffix)
    library = Path(sdk["torch_library"])
    selected_paths = [_plain(Path(sdk[name]).resolve(strict=True)) for name in ("torch_module", "native_module")] + [
        _plain(library / name) for name in ("libtorch_python.so", "libtorch_cpu.so", "libc10.so")
    ]
    for path in (*selected_paths, includes, Path(sdk["python_include"]), library):
        _outside(path, forbidden)
    command = _compile_command(compiler, sdk, cpp, getter, dependencies)
    _, invocation = _run(
        command,
        owner=output,
        stage="native_tensor_argument_getter_build",
        timeout=180,
        inputs=(observer, cpp, sdk_path, *(includes / relative for relative in T.PRIMARY_HEADERS), *selected_paths),
        outputs=(getter, dependencies),
        preexec_fn=_compile_limit,
    )
    invocations.append(invocation)
    dependency_names = shlex.split(dependencies.read_text().split(":", 1)[1].replace("\\\n", " "))
    dependency_paths = tuple(sorted({_dependency_path(path) for path in dependency_names}))
    for path in dependency_paths:
        _outside(path, forbidden)
    return {
        "schema": SCHEMA,
        "python": str(python),
        "compiler": str(compiler),
        "observer": str(observer),
        "checkout": str(checkout),
        "commit": commit,
        "public_sources": sources,
        "sdk": str(sdk_path),
        "getter": str(getter),
        "cpp": str(cpp),
        "dependencies": str(dependencies),
        "dependency_paths": [str(path) for path in dependency_paths],
        "invocations": invocations,
        "unknowns": ["installed_framework_sdk_build_correspondence", "complete_host_linker_loader_dependency_closure"],
    }


def verify_getter(record):
    """Replay observations only: record JSON does not issue a live authority."""
    if (
        set(record)
        != {
            "schema",
            "python",
            "compiler",
            "observer",
            "checkout",
            "commit",
            "public_sources",
            "sdk",
            "getter",
            "cpp",
            "dependencies",
            "dependency_paths",
            "invocations",
            "unknowns",
        }
        or record["schema"] != SCHEMA
        or record["unknowns"]
        != ["installed_framework_sdk_build_correspondence", "complete_host_linker_loader_dependency_closure"]
        or record["observer"] != str(module_source_path("merlin.targetgen.torch_tensor_argument_observer"))
        or Path(record["cpp"]).read_text() != T.CPP_SOURCE
    ):
        raise RtlIntakeRefusal("Tensor argument getter lost its fixed original source preparation")
    checkout, commit = Path(record["checkout"]), record["commit"]
    if _git(checkout, "rev-parse", "HEAD") != commit or _git(checkout, "status", "--porcelain", "--untracked-files=no"):
        raise RtlIntakeRefusal("Tensor argument public checkout changed")
    if [row["relative"] for row in record["public_sources"]] != list(T.PUBLIC_PATHS):
        raise RtlIntakeRefusal("Tensor argument getter lost its complete public API source roster")
    for source in record["public_sources"]:
        if _git(checkout, "rev-parse", commit + ":" + source["relative"]) != source["blob"]:
            raise RtlIntakeRefusal("Tensor argument source Git binding changed")
        actual = I.require_environment(Path(source["invocation"]), environment=ENVIRONMENT)
        if (
            actual["argv"] != ["/usr/bin/git", "-C", str(checkout), "show", commit + ":" + source["relative"]]
            or Path(actual["stdout"]["path"]).read_bytes() != Path(source["path"]).read_bytes()
        ):
            raise RtlIntakeRefusal("Tensor argument source differs from actual public Git readback")
    for path in record["invocations"]:
        I.require_environment(Path(path), environment=ENVIRONMENT)
    if len(record["invocations"]) != len(T.PUBLIC_PATHS) + 3 or record["invocations"][: len(T.PUBLIC_PATHS)] != [
        row["invocation"] for row in record["public_sources"]
    ]:
        raise RtlIntakeRefusal("Tensor argument getter lost its complete native preparation roster")
    sdk_receipt, header_receipt, build_receipt = (
        I.require_environment(Path(path), environment=ENVIRONMENT) for path in record["invocations"][-3:]
    )
    sdk = json.loads(Path(record["sdk"]).read_bytes())
    output, observer, python = Path(record["cpp"]).parent, record["observer"], record["python"]
    if (
        sdk_receipt["argv"] != [python, "-I", observer, "sdk"]
        or sdk_receipt["stage"] != "native_tensor_argument_sdk_selection"
        or json.loads(Path(sdk_receipt["stdout"]["path"]).read_bytes()) != sdk
        or sdk["reported_git_version"] != commit
        or header_receipt["argv"]
        != [python, "-I", observer, "headers", str(output / "public-source"), str(output / "generated-headers")]
        or header_receipt["stage"] != "public_tensor_argument_header_generation"
        or build_receipt["argv"]
        != _compile_command(record["compiler"], sdk, record["cpp"], record["getter"], record["dependencies"])
        or build_receipt["stage"] != "native_tensor_argument_getter_build"
        or Path(record["getter"]) != output / ("native_tensor_argument_getter" + sdk["extension_suffix"])
    ):
        raise RtlIntakeRefusal("Tensor argument getter does not replay its fixed native SDK/header/build processes")
    for relative in T.PRIMARY_HEADERS:
        public_header = output / "generated-headers" / relative
        native_header = Path(sdk["torch_include"]) / relative
        if public_header.read_bytes() != native_header.read_bytes():
            raise RtlIntakeRefusal("Tensor argument public header product no longer matches the observed SDK")
    dependency_names = shlex.split(Path(record["dependencies"]).read_text().split(":", 1)[1].replace("\\\n", " "))
    actual_dependencies = [str(path) for path in sorted({_dependency_path(path) for path in dependency_names})]
    if record["dependency_paths"] != actual_dependencies:
        raise RtlIntakeRefusal("Tensor argument getter lost the actual complete non-system header dependency roster")
    return record


def observe_arguments(*, trace, schema_observation, getter, output):
    request = original_tensor_argument_requests(trace, schema_observation)
    request_path, observation_path = output / "tensor-request.json", output / "tensor-observation.json"
    request_path.write_bytes(_json(request))
    result, invocation = _run(
        [getter["python"], "-I", getter["observer"], "observe", str(request_path), getter["getter"]],
        owner=output,
        stage="native_original_tensor_argument_observation",
        inputs=(request_path, Path(getter["observer"]), Path(getter["getter"]), Path(getter["sdk"])),
    )
    observation_path.write_bytes(result.stdout)
    return {"request": str(request_path), "observation": str(observation_path), "invocation": invocation}


def verify_arguments(*, trace, schema_observation, getter, member):
    if set(member) != {"request", "observation", "invocation"}:
        raise RtlIntakeRefusal("Tensor argument observations need their complete native request/output")
    request = original_tensor_argument_requests(trace, schema_observation)
    if json.loads(Path(member["request"]).read_bytes()) != request:
        raise RtlIntakeRefusal("Tensor argument request changed original source slots or literals")
    actual = I.require_environment(Path(member["invocation"]), environment=ENVIRONMENT)
    if (
        actual["argv"] != [getter["python"], "-I", getter["observer"], "observe", member["request"], getter["getter"]]
        or actual["stage"] != "native_original_tensor_argument_observation"
        or Path(actual["stdout"]["path"]).read_bytes() != Path(member["observation"]).read_bytes()
    ):
        raise RtlIntakeRefusal("Tensor argument observation differs from its actual fixed native API invocation")
    return json.loads(Path(member["observation"]).read_bytes())
