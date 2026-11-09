"""Public compile/run boundary admission; independent of a hardware toolchain."""

import copy
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin import api
from merlin.compiler.artifact import Artifact, Compilation, Deployment
from merlin.runtime.service import RunOptions, RunResult


class BinaryBackend:
    name = "fixture.binary"

    def validate(self, request):
        if request.options.parameters:
            raise ValueError("unsupported option")

    def compile(self, request):
        request.output.mkdir()
        device = request.output / "device.bin"
        commands = request.output / "launch.json"
        device.write_bytes(b"device-commands")
        commands.write_text('{"entry": 0}')
        return Compilation(
            request.output,
            self.name,
            request.source.sha256,
            (
                Artifact.from_file(device, kind="device-binary", role="device"),
                Artifact.from_file(commands, kind="json", role="launch"),
            ),
        )


@pytest.fixture
def linux_image(tmp_path):
    if sys.platform != "linux" or os.uname().machine != "x86_64":
        pytest.skip("native x86_64 Linux execution fixture")
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("native C compiler required for actual Linux execution")
    source, image = tmp_path / "process.c", tmp_path / "process.elf"
    source.write_text('''#define _GNU_SOURCE
#include <sched.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
int main(int argc, char **argv) {
  if (argc > 1 && !strcmp(argv[1], "fail")) { fputc(255, stderr); return 7; }
  if (argc > 2 && !strcmp(argv[1], "spawn")) {
    if (!fork()) { sleep(1); FILE *f=fopen(argv[2],"w"); if(f) { fputs("escaped",f); fclose(f); } _exit(0); }
    puts("child-started"); fflush(stdout); sleep(30); return 9;
  }
  cpu_set_t cpus; CPU_ZERO(&cpus);
  if (sched_getaffinity(0, sizeof cpus, &cpus)) return 8;
  printf("%d/%d\\n", argc-1, CPU_COUNT(&cpus));
  for(int i=1;i<argc;++i) puts(argv[i]);
  return 0;
}
''')
    subprocess.run([compiler, "-O0", str(source), "-o", str(image)], check=True, capture_output=True)
    return image


def test_native_linux_arguments_affinity_streams_and_cli_status(linux_image, monkeypatch, capsys):
    import base64

    from merlin.compiler.backend import service
    from merlin.runtime import linux
    from merlin.service_cli import run_main

    before = os.sched_getaffinity(0)
    harts = min(2, len(before))
    deployment = Deployment("host", "linux", "x86_64", "sysv", harts)
    arguments = ("space word", "--flag=value", "")
    result = api.run(linux_image, runner=linux, options=RunOptions(deployment=deployment, arguments=arguments))
    assert result.status == "completed" and result.stdout == f"3/{harts}\nspace word\n--flag=value\n\n"
    assert result.metadata["exit_code"] == 0
    assert result.metadata["initial_cpu_affinity"] == sorted(before)[:harts]
    assert result.metadata["argv"][1:] == list(arguments) and os.sched_getaffinity(0) == before
    assert base64.b64decode(result.metadata["stdout_base64"]) == result.stdout.encode()
    assert result.metadata["verification"] == "not_requested" and not result.metadata["device_execution_verified"]
    monkeypatch.setattr(service, "entry_points", lambda **kwargs: [SimpleNamespace(load=lambda: linux)])
    code = run_main([str(linux_image), "--runner", linux.name, "--system", "host", "--platform", "linux",
                     "--isa", "x86_64", "--abi", "sysv", "--harts", "1", "--argument=fail"])
    observed = json.loads(capsys.readouterr().out)
    assert code == 1 and observed["status"] == "failed" and observed["metadata"]["exit_code"] == 7
    assert base64.b64decode(observed["metadata"]["stderr_base64"]) == b"\xff"


def test_native_linux_refuses_header_platform_abi_and_unavailable_cpu(linux_image, monkeypatch):
    from merlin.runtime import linux

    deployment = Deployment("host", "linux", "x86_64", "sysv", 1)
    monkeypatch.setattr(linux, "_execute", lambda *args: pytest.fail("refused inputs must not launch"))
    for contract in (replace(deployment, platform="baremetal"), replace(deployment, abi="wrong"),
                     replace(deployment, isa="rv64imac"), replace(deployment, harts=len(os.sched_getaffinity(0)) + 1)):
        with pytest.raises(ValueError):
            api.run(linux_image, runner=linux, options=RunOptions(deployment=contract))
    original = linux_image.read_bytes()
    for offset, value in ((18, (243).to_bytes(2, "little")), (16, (1).to_bytes(2, "little")), (5, b"\x02")):
        changed = bytearray(original)
        changed[offset:offset + len(value)] = value
        linux_image.write_bytes(changed)
        with pytest.raises(ValueError):
            api.run(linux_image, runner=linux, options=RunOptions(deployment=deployment))
    linux_image.write_bytes(original)
    linux_image.chmod(0o600)
    with pytest.raises(ValueError, match="executable"):
        api.run(linux_image, runner=linux, options=RunOptions(deployment=deployment))
    with pytest.raises(ValueError, match="arguments"):
        RunOptions(arguments=("invalid\x00argument",))


def test_native_linux_timeout_terminates_its_spawned_process_group(linux_image, tmp_path):
    import time

    from merlin.runtime import linux

    marker = tmp_path / "child-survived"
    result = api.run(linux_image, runner=linux,
                     options=RunOptions(deployment=Deployment("host", "linux", "x86_64", "sysv", 1),
                                        arguments=("spawn", str(marker)), timeout=0.3))
    assert result.status == "timed_out" and result.metadata["exit_code"] != 0
    assert "child-started" in result.stdout and result.metadata["timeout_process_group_cleanup"]
    time.sleep(1.1)
    assert not marker.exists()


def test_multi_product_compilation_and_cli_share_request_and_refuse_bad_options(tmp_path, monkeypatch, capsys):
    from merlin.cli import main
    from merlin.compiler.backend import service

    backend = BinaryBackend()
    monkeypatch.setattr(service, "entry_points", lambda **kwargs: [SimpleNamespace(load=lambda: backend)])
    source = tmp_path / "kernel.mlir"
    source.write_text("builtin.module {}")
    assert main(["compile", str(source), "--backend", backend.name, "--output", str(tmp_path / "compiled")]) == 0
    observed = json.loads(capsys.readouterr().out)
    assert [x["kind"] for x in observed["artifacts"]] == ["device-binary", "json"]
    loaded = Compilation.load(observed["manifest"])
    assert api.inspect(loaded)["metadata"]["request"]["source_origin"] == str(source.resolve())
    assert main(["inspect", str(loaded.manifest)]) == 0
    assert json.loads(capsys.readouterr().out)["source_sha256"] == loaded.source_sha256
    with pytest.raises(ValueError, match="unsupported option"):
        api.compile(
            source,
            backend=backend,
            output=tmp_path / "rejected",
            options=api.CompileOptions(parameters={"fixture.unknown": 1}),
        )
    assert not (tmp_path / "rejected").exists()
    with pytest.raises(FileExistsError):
        api.compile(source, backend=backend, output=loaded.directory)


def test_manifest_reopens_bytes_and_refuses_escape_or_substituted_request(tmp_path):
    result = api.compile("builtin.module {}", backend=BinaryBackend(), output=tmp_path / "out")
    raw = result.to_dict()
    result.artifacts[0].path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes do not match"):
        api.inspect(result.manifest)
    raw["artifacts"][0]["path"] = "../escape.bin"
    result.manifest.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="traversal"):
        Compilation.load(result.manifest)

    class WrongSource(BinaryBackend):
        def compile(self, request):
            return replace(super().compile(request), source_sha256="0" * 64)

    with pytest.raises(ValueError, match="identity"):
        api.compile("builtin.module {}", backend=WrongSource(), output=tmp_path / "wrong")

    class ChangedSource(BinaryBackend):
        def validate(self, request):
            object.__setattr__(request.source, "text", "builtin.module {func.func private @changed()}")

    with pytest.raises(ValueError, match="changed the original"):
        api.compile("builtin.module {}", backend=ChangedSource(), output=tmp_path / "changed")
    assert not (tmp_path / "changed").exists()


def test_external_runtime_contract_harts_and_no_compile(tmp_path, monkeypatch):
    from merlin.runtime.backends import spike

    elf = tmp_path / "image.unclassified"
    header = bytearray(64)
    header[:6] = b"\x7fELF\x02\x01"
    header[18:20] = (243).to_bytes(2, "little")
    header[48:52] = (4).to_bytes(4, "little")  # double-float ABI
    elf.write_bytes(header)
    calls = []
    monkeypatch.setattr(spike, "run_elf", lambda *args, **kwargs: calls.append((args, kwargs)) or "console")
    contract = Deployment("fixture-soc", "zephyr", "rv64imafdc", "lp64d", 2, 0x80000000, 0x100000)
    result = api.run(elf, runner="spike", options=RunOptions(deployment=contract))
    assert result.status == "completed" and result.stdout == "console"
    assert calls[0][1] == {
        "harts": 2,
        "isa": contract.isa,
        "timeout": 300,
        "dram_base": contract.dram_base,
        "mem_bytes": contract.memory_bytes,
    }
    with pytest.raises(ValueError, match="explicit Deployment"):
        api.run(elf, runner="spike")
    with pytest.raises(ValueError, match="hart count"):
        api.run(elf, runner="spike", options=RunOptions(deployment=contract, harts=3))
    with pytest.raises(ValueError, match="ABI"):
        api.run(elf, runner="spike", options=RunOptions(deployment=replace(contract, abi="lp64")))
    with pytest.raises(ValueError, match="does not support.*arguments"):
        api.run(elf, runner="spike", options=RunOptions(deployment=contract, arguments=("--unexpected",)))
    with pytest.raises(ValueError, match="FireSim supports prepare/collect"):
        api.run(elf, runner="firesim")
    assert len(calls) == 1
    prepared = api.run(
        elf, runner="firesim", options=RunOptions(deployment=contract, mode="prepare", output=tmp_path / "handoff")
    )
    record = json.loads((prepared.directory / "external-image.json").read_text())
    assert record["compilation_provenance"] == "not_supplied"
    assert record["deployment_authority"] == "caller"
    assert len(calls) == 1  # preparing an external image does not execute Spike or FireSim


def test_saved_deployment_is_not_runtime_overridable_and_products_stay_distinct(tmp_path):
    from hashlib import sha256

    result = api.compile("builtin.module {}", backend=BinaryBackend(), output=tmp_path / "out")
    deployment = Deployment("soc", "baremetal", "rv64imac", "lp64", 1)
    product = replace(result, deployment=deployment)
    def completed(request):
        (tmp_path / "runtime-output.txt").write_text("fixture output, not an execution input")
        return RunResult("device-fixture", request.artifact.sha256, "completed")

    runner = SimpleNamespace(
        name="device-fixture",
        validate=lambda request: None,
        run=completed,
    )
    result = api.run(product, runner=runner, options=RunOptions(role="device"))
    assert result.status == "completed"
    with pytest.raises(ValueError, match="cannot override"):
        api.run(
            product, runner=runner, options=RunOptions(role="device", deployment=replace(deployment, platform="zephyr"))
        )
    with pytest.raises(ValueError, match="expected one artifact"):
        api.run(product, runner=runner)
    # Neither callback may replace inputs or mutate the reachable caller option
    # mapping. A self-consistent replacement digest cannot renew the selection.
    for stage in ("validate", "run"):
        for defect in ("bytes", "artifact", "deployment", "harts", "options", "inputs", "arguments"):
            path = tmp_path / f"{stage}-{defect}.bin"
            path.write_bytes(b"original device input")
            artifact = Artifact.from_file(path, kind="device-binary", role="device")
            options = RunOptions(deployment=Deployment("soc", "baremetal", "rv64imac", "lp64", 1))
            calls = []

            def callback(request, boundary):
                calls.append(boundary)
                if boundary == stage:
                    if defect in {"bytes", "artifact"}:
                        path.write_bytes(b"replaced device input")
                        if defect == "artifact":
                            object.__setattr__(request.artifact, "sha256", sha256(path.read_bytes()).hexdigest())
                    elif defect == "deployment":
                        object.__setattr__(request.deployment, "harts", 2)
                    elif defect == "harts":
                        object.__setattr__(request, "harts", 2)
                    elif defect == "options":
                        object.__setattr__(options, "timeout", 99)
                    elif defect == "arguments":
                        object.__setattr__(request, "arguments", ("substituted",))
                    else:
                        options.inputs["replacement"] = path
                return RunResult("device-fixture", request.artifact.sha256, "completed")

            selected = SimpleNamespace(name="device-fixture", validate=lambda request: callback(request, "validate"),
                                       run=lambda request: callback(request, "run"))
            with pytest.raises(ValueError, match="original execution|bytes do not match"):
                api.run(artifact, runner=selected, options=options)
            assert calls == (["validate"] if stage == "validate" else ["validate", "run"]), (stage, defect)


def test_compile_cli_retains_legacy_model_dispatch(monkeypatch):
    import sys

    from merlin.service_cli import compile_main

    calls = []
    monkeypatch.setitem(sys.modules, "merlin.compile_cli", SimpleNamespace(main=lambda args: calls.append(args) or 7))
    assert compile_main(["--workload", "fixture", "--target", "scalar"]) == 7
    assert calls == [["--workload", "fixture", "--target", "scalar"]]


@pytest.mark.parametrize(
    ("filename", "schema"),
    [("software-spec.yaml", "merlin.software_spec.v1"), ("spec.yaml", "merlin.software_spec.v2")],
)
def test_spec_inspection_uses_real_semantics_without_compilation(tmp_path, capsys, filename, schema):
    from merlin.cli import main
    from merlin.targetgen.software_spec import load_software_spec

    bundle = Path(__file__).resolve().parents[3] / "examples/gemmini/target"
    original = bundle / filename
    selected = tmp_path / "selected.yaml"
    selected.write_bytes(original.read_bytes())
    compact = api.inspect_spec(selected)
    assert compact["source_schema"] == schema
    assert compact["qualification"] == "not_established"
    assert compact["projection"]["status"] == "supported"
    assert "software_spec" not in compact
    assert "authored_text" not in compact
    assert main(["inspect", "--spec", str(selected)]) == 0
    assert json.loads(capsys.readouterr().out) == compact
    assert main(["inspect", "--spec", str(selected), "--expand"]) == 0
    expanded = json.loads(capsys.readouterr().out)
    assert expanded["software_spec"] == load_software_spec(selected)
    assert expanded["canonical_sha256"] == compact["canonical_sha256"]
    assert expanded["authored_text"] == selected.read_bytes().decode("utf-8")
    if schema == "merlin.software_spec.v2":
        from merlin.compiler.quantization.preparation import LEGACY_OUTPUT_PATH, load_profile

        expected = copy.deepcopy(load_software_spec(bundle / "software-spec.yaml"))
        expected["status"] = "unreviewed"
        methodology = expected["quantization"].pop("methodology")
        assert expanded["software_spec"] == expected
        profile = load_profile(bundle.parent / "preparation.yaml")
        assert profile.document()["methodology"] == methodology
        assert profile.document()["compatibility_policy"] == LEGACY_OUTPUT_PATH
    assert sorted(p.name for p in tmp_path.iterdir()) == [selected.name]
    with pytest.raises(ValueError, match="Boolean"):
        api.inspect_spec(selected, expanded=1)


@pytest.mark.parametrize("arguments", [[], ["a.json", "--spec", "a.yaml"], ["a.json", "--expand"],
                                      ["a.json", "--backends"], ["--backends", "--expand"]])
def test_inspection_refuses_ambiguous_selection_before_input_io(arguments):
    from merlin.service_cli import inspect_main

    with pytest.raises(SystemExit) as failure:
        inspect_main(arguments)
    assert failure.value.code == 2


def test_backend_inspection_uses_installed_metadata_without_importing_provider(tmp_path, monkeypatch, capsys):
    import sys

    from merlin.cli import main
    from merlin.compiler.backend.service import resolve

    module = "fixture_unloaded_backend"
    (tmp_path / (module + ".py")).write_text('raise AssertionError("provider imported by inspection")\n')
    for name, provider_name, origin in (
        ("fixture_editable", "fixture.shared", {"url": tmp_path.as_uri(), "dir_info": {"editable": True}}),
        ("fixture_upstream", "fixture.shared", {
            "url": "https://user:secret@example.test/backend.git?token=secret#private",
            "vcs_info": {"vcs": "git", "commit_id": "a" * 40},
        }),
        ("fixture_invalid", "fixture.invalid_origin", None),
    ):
        metadata = tmp_path / (name + "-1.0.dist-info")
        metadata.mkdir()
        (metadata / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n")
        (metadata / "entry_points.txt").write_text(f"[merlin.backends]\n{provider_name} = {module}:plugin\n")
        (metadata / "direct_url.json").write_text(json.dumps(origin) if origin else "broken JSON")
    monkeypatch.syspath_prepend(str(tmp_path))
    report = api.inspect_backends()
    rows = {row["distribution"]["name"]: row for row in report["providers"]
            if row["name"].startswith("fixture.")}
    assert rows["fixture_editable"]["origin"]["editable"] is True
    assert rows["fixture_upstream"]["origin"]["url"] == "https://example.test/backend.git"
    assert rows["fixture_upstream"]["origin"]["url_redacted"] is True
    assert rows["fixture_upstream"]["origin"]["vcs"]["commit_id"] == "a" * 40
    assert rows["fixture_invalid"]["origin"]["status"] == "invalid_metadata"
    assert all(rows[name]["selection"] == "ambiguous" for name in ("fixture_editable", "fixture_upstream"))
    assert report["code_loaded_by_inspection"] is False
    assert all(row["compatibility"] == "not_checked" for row in rows.values())
    assert module not in sys.modules
    assert main(["inspect", "--backends"]) == 0
    assert json.loads(capsys.readouterr().out) == report
    with pytest.raises(ValueError, match="found 2.*fixture_editable.*fixture_upstream"):
        resolve("fixture.shared", group="merlin.backends", methods=("validate", "compile"))
    assert module not in sys.modules


_PACKAGE_INTERFACE = (
    'module attributes {merlin_iface.version = "0.1", merlin_iface.target = "fixture", '
    'merlin_iface.abi_version = "0.1"} {\n'
    '  %X = merlin_iface.tensor {name = "X", role = "input"} : tensor<2x2xi8>\n'
    '  %Y = merlin_iface.movement %X {name = "Y", semantic = "mvin_mvout", output_dtype = "i32"} '
    ': (tensor<2x2xi8>) -> tensor<2x2xi32>\n}\n'
)


def _process_package(tmp_path, mode="valid"):
    """Actual fixture processes exercise transport, never target computation."""
    import shutil
    import sys

    import yaml

    from merlin.targetgen.contract import schemas
    from merlin.targetgen.contract.interface_emit import parse_interface_mlir

    root, contract = tmp_path / "package", tmp_path / "contract"
    root.mkdir()
    shutil.copytree(schemas.contract_dir() / "schemas", contract / "schemas")
    (root / "buffer.json").write_text(json.dumps(parse_interface_mlir(_PACKAGE_INTERFACE)))
    (root / "mode.json").write_text(json.dumps({"mode": mode, "contract": str(contract)}))
    (root / "tool.py").write_text('''import json,sys
from pathlib import Path
root=Path(__file__).parent
mode=json.loads((root/'mode.json').read_text())
entry=sys.argv[1]
source=Path(sys.argv[2])
if entry=='parse':
    if mode['mode']=='source':
        (root/'tool.py').write_text(Path(__file__).read_text()+'\\n# changed\\n')
    if mode['mode']=='schema':
        path=Path(mode['contract'])/'schemas/command_buffer.schema.json'
        path.write_text(path.read_text()+'\\n')
    if mode['mode']=='input':
        source.write_text('changed interface')
elif entry=='lower_interface_to_target':
    print('fixture target IR transport; no executable semantics')
elif entry=='emit_command_buffer':
    command=json.loads((root/'buffer.json').read_text())
    if mode['mode']=='dtype': command['tensors']['Y']['dtype']='i16'
    if mode['mode']=='invalid': command['commands']='invalid'
    if mode['mode']=='decline': command.update(commands=[],declined={'reason':'fixture decline'})
    Path(sys.argv[3]).write_text(json.dumps(command))
elif entry=='lower_target_to_llvm':
    print('fixture opaque target output, deliberately not LLVM')
else:
    raise AssertionError(entry)
''')
    commands = {entry: {"argv": [sys.executable, "{tool}", entry, "{input_mlir}"]}
                for entry in ("parse", "lower_interface_to_target", "emit_command_buffer", "lower_target_to_llvm")}
    commands["emit_command_buffer"]["argv"].append("{output_json}")
    (root / "manifest.yaml").write_text(yaml.safe_dump({
        "artifact_type": "mlir_oot_target_backend", "target": "fixture", "language": "python",
        "authoring": {"mode": "hand_curated"}, "integrity_exempt": False,
        "entrypoints": {"tool": "tool.py"}, "commands": commands,
    }))
    return root, api.Source(_PACKAGE_INTERFACE, kind="merlin-interface"), api.CompileOptions(parameters={
        "package.root": str(root), "package.contract": str(contract), "package.timeout": 30,
    })


def test_process_package_transport_retains_actual_invocations_and_opaque_products(tmp_path, monkeypatch):
    from merlin.compiler.backend import package

    root, source, options = _process_package(tmp_path)
    reference = tmp_path / "contract/independent-reference.bin"
    reference.write_bytes(b"fixture reference outside compiler schema selection")
    read_bytes = Path.read_bytes

    def read_selected(path):
        assert path != reference, "compiler inspected an independent reference"
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_selected)
    result = api.compile(source, backend=package, options=options, output=tmp_path / "compiled")
    assert result.deployment is None
    assert result.metadata["numerical_verification"] == "not_run"
    assert result.metadata["execution_observed"] is False
    records = sorted(result.directory.glob("invocations/*/invocation.json"),
                     key=lambda path: json.loads(path.read_text())["started_ns"])
    assert [json.loads(path.read_text())["stage"] for path in records] == [
        "parse", "lower_interface_to_target", "emit_command_buffer", "emit_target_artifact",
    ]
    command = next(item for item in result.artifacts if item.role == "command-buffer")
    assert json.loads(command.path.read_text())["tensors"]["Y"]["dtype"] == "i32"
    assert result.metadata["selected_package"]["source"]["path"] == str(root)
    assert "deliberately not LLVM" in (result.directory / "stages/target-artifact.txt").read_text()
    assert api.inspect(result.manifest)["backend"] == package.name


@pytest.mark.parametrize("mode", ["source", "schema", "input", "dtype", "invalid", "decline"])
def test_process_package_transport_refuses_changes_and_preserves_original_decline(tmp_path, mode):
    from merlin.compiler.backend import package
    from merlin.targetgen.package_runtime import BackendDeclined, CertFailure

    _, source, options = _process_package(tmp_path, mode)
    expected = BackendDeclined if mode == "decline" else CertFailure if mode in {"dtype", "invalid"} else ValueError
    with pytest.raises(expected):
        api.compile(source, backend=package, options=options, output=tmp_path / "refused")
    assert not (tmp_path / "refused/compilation.json").exists()


@pytest.mark.parametrize("defect", ["model", "deployment", "unknown-option", "target", "duplicate", "dtype",
                                    "noncompiler", "abi2", "overlap", "scoped"])
def test_process_package_transport_admits_only_its_selected_contract_before_execution(tmp_path, defect):
    import yaml

    from merlin.compiler.backend import package
    from merlin.targetgen.package_runtime import CertFailure, scoped_package_executor

    root, source, options = _process_package(tmp_path)
    output = tmp_path / "refused"
    if defect == "model":
        source = api.Source("builtin.module {}")
    elif defect == "deployment":
        options = replace(options, platform="zephyr")
    elif defect == "unknown-option":
        options = replace(options, parameters={**options.parameters, "package.unsupported": True})
    elif defect == "target":
        source = replace(source, text=source.text.replace('target = "fixture"', 'target = "other"'))
    elif defect == "duplicate":
        duplicate = '  %Z = merlin_iface.tensor {name = "X", role = "input"} : tensor<2x2xi8>\n'
        source = replace(source, text=source.text.replace("  %Y =", duplicate + "  %Y ="))
    elif defect == "dtype":
        source = replace(source, text=source.text.replace("xi8", "x!unknown.element"))
    elif defect in {"noncompiler", "abi2"}:
        path = root / "manifest.yaml"
        manifest = yaml.safe_load(path.read_text())
        manifest.update({"artifact_type": "not-a-compiler"} if defect == "noncompiler" else {"abi_version": "0.2"})
        path.write_text(yaml.safe_dump(manifest))
    elif defect == "overlap":
        output = root / "compiled"
    elif defect == "scoped":
        def refuse(*args, **kwargs):
            raise RuntimeError("fixture scoped executor refuses")

        executor = SimpleNamespace(run_entrypoint=refuse, build_package=refuse)
        with scoped_package_executor(executor), pytest.raises(RuntimeError, match="scoped executor"):
            api.compile(source, backend=package, options=options, output=output)
        assert not list(output.rglob("invocation.json"))
        assert not (output / "compilation.json").exists()
        return
    with pytest.raises((ValueError, CertFailure)):
        api.compile(source, backend=package, options=options, output=output)
    assert not output.exists()
