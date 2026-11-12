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
from merlin.runtime.service import RunOptions


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
    source, image = (tmp_path / "process.c", tmp_path / "process.elf")
    source.write_text(
        '#define _GNU_SOURCE\n#include <sched.h>\n#include <stdio.h>\n#include <string.h>\n#include <unistd.h>\nint main(int argc, char **argv) {\n  if (argc > 1 && !strcmp(argv[1], "fail")) { fputc(255, stderr); return 7; }\n  if (argc > 2 && !strcmp(argv[1], "spawn")) {\n    if (!fork()) { sleep(1); FILE *f=fopen(argv[2],"w"); if(f) { fputs("escaped",f); fclose(f); } _exit(0); }\n    puts("child-started"); fflush(stdout); sleep(30); return 9;\n  }\n  cpu_set_t cpus; CPU_ZERO(&cpus);\n  if (sched_getaffinity(0, sizeof cpus, &cpus)) return 8;\n  printf("%d/%d\\n", argc-1, CPU_COUNT(&cpus));\n  for(int i=1;i<argc;++i) puts(argv[i]);\n  return 0;\n}\n'
    )
    subprocess.run([compiler, "-O0", str(source), "-o", str(image)], check=True, capture_output=True)
    return image


def test_native_linux_arguments_affinity_streams_and_cli_status(linux_image, monkeypatch, capsys):
    import base64

    from merlin.cli.service import run_main
    from merlin.compiler.backend import service
    from merlin.runtime import linux

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
    assert result.metadata["verification"] == "not_requested" and (not result.metadata["device_execution_verified"])
    monkeypatch.setattr(service, "entry_points", lambda **kwargs: [SimpleNamespace(load=lambda: linux)])
    code = run_main(
        [
            str(linux_image),
            "--runner",
            linux.name,
            "--system",
            "host",
            "--platform",
            "linux",
            "--isa",
            "x86_64",
            "--abi",
            "sysv",
            "--harts",
            "1",
            "--argument=fail",
        ]
    )
    observed = json.loads(capsys.readouterr().out)
    assert code == 1 and observed["status"] == "failed" and (observed["metadata"]["exit_code"] == 7)
    assert base64.b64decode(observed["metadata"]["stderr_base64"]) == b"\xff"


def test_native_linux_timeout_terminates_its_spawned_process_group(linux_image, tmp_path):
    import time

    from merlin.runtime import linux

    marker = tmp_path / "child-survived"
    result = api.run(
        linux_image,
        runner=linux,
        options=RunOptions(
            deployment=Deployment("host", "linux", "x86_64", "sysv", 1), arguments=("spawn", str(marker)), timeout=0.3
        ),
    )
    assert result.status == "timed_out" and result.metadata["exit_code"] != 0
    assert "child-started" in result.stdout and result.metadata["timeout_process_group_cleanup"]
    time.sleep(1.1)
    assert not marker.exists()


def test_multi_product_compilation_and_cli_share_request_and_refuse_bad_options(tmp_path, monkeypatch, capsys):
    from merlin.cli.main import main
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


def test_external_runtime_contract_harts_and_no_compile(tmp_path, monkeypatch):
    from merlin.runtime.backends import spike

    elf = tmp_path / "image.unclassified"
    header = bytearray(64)
    header[:6] = b"\x7fELF\x02\x01"
    header[18:20] = (243).to_bytes(2, "little")
    header[48:52] = (4).to_bytes(4, "little")
    elf.write_bytes(header)
    calls = []
    monkeypatch.setattr(spike, "run_elf", lambda *args, **kwargs: calls.append((args, kwargs)) or "console")
    contract = Deployment("fixture-soc", "zephyr", "rv64imafdc", "lp64d", 2, 2147483648, 1048576)
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
    assert len(calls) == 1


_PACKAGE_INTERFACE = 'module attributes {merlin_iface.version = "0.1", merlin_iface.target = "fixture", merlin_iface.abi_version = "0.1"} {\n  %X = merlin_iface.tensor {name = "X", role = "input"} : tensor<2x2xi8>\n  %Y = merlin_iface.movement %X {name = "Y", semantic = "mvin_mvout", output_dtype = "i32"} : (tensor<2x2xi8>) -> tensor<2x2xi32>\n}\n'


def _process_package(tmp_path, mode="valid"):
    """Actual fixture processes exercise transport, never target computation."""
    import shutil
    import sys

    import yaml

    from merlin.targetgen.contract import schemas
    from merlin.targetgen.contract.interface_emit import parse_interface_mlir

    root, contract = (tmp_path / "package", tmp_path / "contract")
    root.mkdir()
    shutil.copytree(schemas.contract_dir() / "schemas", contract / "schemas")
    (root / "buffer.json").write_text(json.dumps(parse_interface_mlir(_PACKAGE_INTERFACE)))
    (root / "mode.json").write_text(json.dumps({"mode": mode, "contract": str(contract)}))
    (root / "tool.py").write_text(
        "import json,sys\nfrom pathlib import Path\nroot=Path(__file__).parent\nmode=json.loads((root/'mode.json').read_text())\nentry=sys.argv[1]\nsource=Path(sys.argv[2])\nif entry=='parse':\n    if mode['mode']=='source':\n        (root/'tool.py').write_text(Path(__file__).read_text()+'\\n# changed\\n')\n    if mode['mode']=='schema':\n        path=Path(mode['contract'])/'schemas/command_buffer.schema.json'\n        path.write_text(path.read_text()+'\\n')\n    if mode['mode']=='input':\n        source.write_text('changed interface')\nelif entry=='lower_interface_to_target':\n    print('fixture target IR transport; no executable semantics')\nelif entry=='emit_command_buffer':\n    command=json.loads((root/'buffer.json').read_text())\n    if mode['mode']=='dtype': command['tensors']['Y']['dtype']='i16'\n    if mode['mode']=='invalid': command['commands']='invalid'\n    if mode['mode']=='decline': command.update(commands=[],declined={'reason':'fixture decline'})\n    Path(sys.argv[3]).write_text(json.dumps(command))\nelif entry=='lower_target_to_llvm':\n    print('fixture opaque target output, deliberately not LLVM')\nelse:\n    raise AssertionError(entry)\n"
    )
    commands = {
        entry: {"argv": [sys.executable, "{tool}", entry, "{input_mlir}"]}
        for entry in ("parse", "lower_interface_to_target", "emit_command_buffer", "lower_target_to_llvm")
    }
    commands["emit_command_buffer"]["argv"].append("{output_json}")
    (root / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "artifact_type": "mlir_oot_target_backend",
                "target": "fixture",
                "language": "python",
                "authoring": {"mode": "hand_curated"},
                "integrity_exempt": False,
                "entrypoints": {"tool": "tool.py"},
                "commands": commands,
            }
        )
    )
    return (
        root,
        api.Source(_PACKAGE_INTERFACE, kind="merlin-interface"),
        api.CompileOptions(
            parameters={"package.root": str(root), "package.contract": str(contract), "package.timeout": 30}
        ),
    )


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
    records = sorted(
        result.directory.glob("invocations/*/invocation.json"),
        key=lambda path: json.loads(path.read_text())["started_ns"],
    )
    assert [json.loads(path.read_text())["stage"] for path in records] == [
        "parse",
        "lower_interface_to_target",
        "emit_command_buffer",
        "emit_target_artifact",
    ]
    command = next((item for item in result.artifacts if item.role == "command-buffer"))
    assert json.loads(command.path.read_text())["tensors"]["Y"]["dtype"] == "i32"
    assert result.metadata["selected_package"]["source"]["path"] == str(root)
    assert "deliberately not LLVM" in (result.directory / "stages/target-artifact.txt").read_text()
    assert api.inspect(result.manifest)["backend"] == package.name
