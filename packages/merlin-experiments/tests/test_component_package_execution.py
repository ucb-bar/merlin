"""Private compiler subprocess mounts and ordinary scoped execution routing."""
import hashlib
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest
from merlin_experiments.phase1 import component_package_execution as E
from merlin_experiments.phase2.component_experiment import ComponentView, RuntimeGrant
from merlin_experiments.phase2.contracts import StageGateError

from merlin.targetgen import package_runtime as P


def test_compiler_namespace_receives_single_input_and_isolated_output_not_private_parent(tmp_path, monkeypatch):
    candidate, evidence = tmp_path / "compiler", tmp_path / "private" / "grade"
    candidate.mkdir()
    evidence.mkdir(parents=True)
    tool = candidate / "tool.py"
    tool.write_text("owned compiler script")
    source = evidence / "input.interface.mlir"
    source.write_text("module {}")
    output = evidence / "command_buffer.json"
    (evidence / "golden.json").write_text("MALICIOUS PRIVATE SIBLING")
    sandbox = tmp_path / "owned-bwrap"
    sandbox.write_text("reviewed tool bytes")
    grant = RuntimeGrant(sandbox, "/usr/bin/bwrap", hashlib.sha256(sandbox.read_bytes()).hexdigest())
    view = ComponentView(tmp_path / "public", "1" * 64, "2" * 64, "3" * 64)
    package = P.Package(candidate, {"language": "python"}, tool)
    monkeypatch.setattr(P, "_resolve_argv", lambda *a: ["/usr/bin/python3", str(tool), str(a[2]), str(a[3])])
    def readonly_policy(*_args, **options):
        assert options["candidate_writable"] is False
        return (str(sandbox), "--unshare-all", "--ro-bind", str(candidate), str(candidate))

    monkeypatch.setattr(E, "strict_tool_policy", readonly_policy)
    observed = []

    def run(command, **kwargs):
        observed.append(command)
        index = command.index("/evaluation-output")
        (Path(command[index - 1]) / "command_buffer.json").write_text('{"actual":"returned"}')
        return subprocess.CompletedProcess(command, 0, stdout="actual compiler stdout", stderr="")

    monkeypatch.setattr(E.subprocess, "run", run)
    executor = E.ComponentPackageExecutor(candidate, view, (grant,), evidence)
    result = executor.run_entrypoint(package, "emit_command_buffer", source, output, invocation_directory=evidence)
    command = observed[0]
    assert result.returncode == 0 and output.read_text() == '{"actual":"returned"}'
    assert str(evidence) not in command and str(evidence / "golden.json") not in command
    assert ["--ro-bind", str(candidate), str(candidate)] == command[2:5]
    assert ["--ro-bind", str(source), "/evaluation-input/interface.mlir"] == command[5:8]
    assert "/evaluation-output/command_buffer.json" in command
    assert next(evidence.rglob("invocation.json")).is_file()


def test_declared_build_scripts_require_separately_boxed_service(tmp_path):
    executor = E.ComponentPackageExecutor(tmp_path, None, (), tmp_path / "grade")
    package = P.Package(tmp_path, {"build": {"command": [sys.executable, "host-script.py"]}}, tmp_path / "tool")
    with pytest.raises(StageGateError, match="boxed build service"):
        executor.build_package(package)


def test_actual_component_package_namespace_denies_private_sibling_writes_and_network(tmp_path):
    """Actual namespace canary only; no target, functional or model qualification."""
    from merlin_experiments.phase2.component_runtime import inventory_runtime
    from test_component_experiment import _view

    from merlin.common import invocation_record

    sandbox, compiler = shutil.which("bwrap"), shutil.which("cc")
    if sandbox is None or compiler is None:
        pytest.skip("actual namespace canary requires reviewed host bwrap and C compiler")
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    code = candidate / "probe.c"
    code.write_text(r'''
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <arpa/inet.h>
int main(int argc, char **argv) {
  if (argc != 5) return 11;
  FILE *input = fopen(argv[1], "r"); char text[80] = {0};
  if (!input || !fgets(text, sizeof(text), input)) return 12;
  fclose(input); if (strcmp(text, "namespace input only\n")) return 13;
  if (open(argv[3], O_RDONLY) >= 0) return 14;
  if (open("/evaluation-input/golden.json", O_RDONLY) >= 0) return 15;
  if (open("readonly-write", O_WRONLY | O_CREAT, 0600) >= 0) return 16;
  int socket_fd = socket(AF_INET, SOCK_STREAM, 0);
  struct sockaddr_in address = {0}; address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK); address.sin_port = htons(atoi(argv[4]));
  alarm(10); if (socket_fd >= 0 && connect(socket_fd, (struct sockaddr *)&address, sizeof(address)) == 0) return 17;
  if (socket_fd >= 0) close(socket_fd);
  FILE *output = fopen(argv[2], "w"); if (!output) return 18;
  fputs("{\"namespace_canary\":\"observed\"}\n", output); fclose(output);
  puts("namespace input/output, readonly compiler, denied sibling and network"); return 0;
}
''')
    tool = candidate / "probe"
    build = subprocess.run([compiler, "-static", "-O2", str(code), "-o", str(tool)],
                           capture_output=True, text=True, timeout=60)
    if build.returncode:
        pytest.skip("actual namespace canary requires the selected host static C runtime")
    private = tmp_path / "private" / "grade"
    private.mkdir(parents=True)
    source, output = private / "interface.mlir", private / "command_buffer.json"
    source.write_text("namespace input only\n")
    golden = private / "golden.json"
    golden.write_text("MALICIOUS PRIVATE SIBLING")
    runtime = inventory_runtime(files=(), trees=(), executables=((Path(sandbox), "/usr/bin/bwrap"),))
    view = _view(tmp_path)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        with socket.create_connection(listener.getsockname(), timeout=2):
            pass
        manifest = {"language": "c", "commands": {"parse": {"argv": ["{tool}", "{input_mlir}", "{output_json}",
                                                                             str(golden),
                                                                             str(listener.getsockname()[1])]}}}
        package = P.Package(candidate, manifest, tool)
        executor = E.ComponentPackageExecutor(candidate, view, runtime, private)
        result = executor.run_entrypoint(package, "parse", source, output, invocation_directory=private)
    if result.returncode and "Operation not permitted" in result.stderr:
        pytest.skip("native namespace creation is unavailable in this process; no qualification")
    assert result.returncode == 0, result.stderr
    assert output.read_text() == '{"namespace_canary":"observed"}\n'
    assert not (candidate / "readonly-write").exists() and golden.read_text() == "MALICIOUS PRIVATE SIBLING"
    record = invocation_record.verify(next(private.rglob("invocation.json")))
    assert record["status"] == "completed" and record["stage"] == "parse"
