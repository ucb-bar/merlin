"""Actual source-selected host-filesystem controls, never experiment inputs.

The public host syscall proxy can read files even when the target program is
freestanding. The same ELF must complete with full outputs in both namespaces.
This tests only process/file containment, not target ISA/effect/timing authority.
The independently qualified runtime must still select a closed runner itself.
"""

import json
import os
from pathlib import Path

import pytest
from merlin_experiments.phase2 import component_experiment as E
from merlin_experiments.phase2 import component_runtime as R
from merlin_experiments.phase2.contracts import StageGateError, sha256_file

from merlin.common import invocation_record as I
from merlin.runtime.out_b64 import OutB64Decoder
from merlin.targetgen.compiler_library import freeze_compiler_library

_ENV = {"PATH": "/usr/bin:/bin", "LC_ALL": "C"}
_NAMES = (
    "MERLIN_TEST_BWRAP",
    "MERLIN_TEST_CROSS_GCC",
    "MERLIN_TEST_FESVR_ENGINE",
    "MERLIN_TEST_DTC",
    "MERLIN_TEST_PUBLIC_SPIKE_SOURCE",
    "MERLIN_TEST_PUBLIC_RISCV_TESTS_SOURCE",
    "MERLIN_TEST_BAREMETAL_PLATFORM",
    "MERLIN_TEST_FESVR_BUILD_RECEIPT",
)


def _selected():
    values = {name: os.environ.get(name) for name in _NAMES}
    commits = tuple(
        os.environ.get(name)
        for name in (
            "MERLIN_TEST_PUBLIC_SPIKE_COMMIT",
            "MERLIN_TEST_PUBLIC_RISCV_TESTS_COMMIT",
        )
    )
    if not all((*values.values(), *commits)):
        pytest.skip("requires exact public source commits, engine/build citation, compiler, CRT and namespace tools")
    assert all(Path(value).is_absolute() for value in values.values())
    assert all(len(commit) == 40 and all(c in "0123456789abcdef" for c in commit) for commit in commits)
    return tuple(Path(value).resolve(strict=True) for value in values.values()), commits


def _run(owner, name, argv, *, inputs=(), dependencies=(), outputs=(), cwd=None, timeout=30):
    result = I.run(
        argv,
        directory=owner / name,
        stage="private_fesvr_" + name,
        inputs=inputs,
        dependencies=(Path(__file__), Path(I.__file__), *dependencies),
        outputs=outputs,
        cwd=cwd or owner,
        env=_ENV,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    return result


def _public_source(owner, name, root, commit, paths):
    head = _run(owner, name + "_head", ["/usr/bin/git", "-C", str(root), "rev-parse", "HEAD"])
    status = _run(owner, name + "_status", ["/usr/bin/git", "-C", str(root), "status", "--porcelain"])
    assert head.returncode == status.returncode == 0 and not status.stdout
    assert head.stdout.decode().strip() == commit
    selected = []
    for index, relative in enumerate(paths):
        path = root / relative
        tracked = _run(
            owner,
            name + "_tracked_" + str(index),
            [
                "/usr/bin/git",
                "-C",
                str(root),
                "ls-files",
                "--error-unmatch",
                "--",
                relative,
            ],
            inputs=(path,),
        )
        assert tracked.returncode == 0 and tracked.stdout.decode().strip() == relative
        assert path.is_file() and not any(p.is_symlink() for p in (path, *path.parents))
        selected.append(path)
    return tuple(selected)


def _control_source(public_syscall, public_helper, admitted, excluded, counts, write_flag):
    numbers = {}
    for method in ("sys_openat", "sys_read", "sys_close"):
        rows = [
            line.strip()
            for line in public_syscall.read_text().splitlines()
            if line.strip().endswith("= &syscall_t::" + method + ";")
        ]
        assert len(rows) == 1 and rows[0].startswith("table[")
        numbers[method] = int(rows[0][len("table[") :].partition("]")[0])
    text = public_helper.read_text()
    start = text.index("extern volatile uint64_t tohost;")
    function = text[start : text.index("\n}\n", text.index("static uintptr_t syscall(", start)) + 3]
    array = "volatile uint64_t magic_mem[8] __attribute__((aligned(64)));"
    assert function.count(array) == 1
    # The exact public helper supplies three arguments. Its unused argument
    # slots are explicitly zeroed: this owned control requests read-only open.
    function = function.replace(array, array[:-1] + " = {0};")
    # The public proxy forwards openat's fourth argument to the native OS.
    # Extend the observed helper with that explicitly initialized argument.
    # The live native os flag must succeed outside the namespace as well.
    assert "reg_t pname, reg_t len, reg_t flags," in public_syscall.read_text()
    signature, slot = "uint64_t arg2)", "magic_mem[3] = arg2;"
    assert function.count(signature) == function.count(slot) == 1
    function = function.replace(signature, "uint64_t arg2, uint64_t arg3)").replace(
        slot, slot + "\n  magic_mem[4] = arg3;"
    )
    definitions = {
        "OPEN_NUMBER": numbers["sys_openat"],
        "READ_NUMBER": numbers["sys_read"],
        "CLOSE_NUMBER": numbers["sys_close"],
        "ADMITTED_COUNT": counts[0],
        "EXCLUDED_COUNT": counts[1],
        "NATIVE_WRITE_FLAG": write_flag,
    }
    return (
        '#include <stdint.h>\n#include "htif.h"\n#include "out_b64.h"\n'
        + function
        + "\n"
        + "".join("#define " + key + " " + str(value) + "\n" for key, value in definitions.items())
        + "#define ADMITTED_PATH "
        + json.dumps(str(admitted))
        + "\n#define EXCLUDED_PATH "
        + json.dumps(str(excluded))
        + r"""
static void packet(const char *name, const unsigned char *data, unsigned long count) {
  htif_puts("OUT_B64_BEGIN v1 "); htif_puts(name); htif_puts(" 1 ");
  htif_putd(count); htif_puts(" 1 u\n");
  merlin_out_b64 p; merlin_out_b64_init(&p, count, 1, 0, htif_puts);
  for (unsigned long i = 0; i < count; ++i)
    if (!merlin_out_b64_word(&p, data[i])) htif_exit(2);
  if (!merlin_out_b64_finish(&p)) htif_exit(3);
  htif_puts("OUT_B64_END\n");
}
int main(void) {
  unsigned char admitted[ADMITTED_COUNT] = {0}, excluded[EXCLUDED_COUNT] = {0}, status[3] = {0};
  uintptr_t fd = syscall(OPEN_NUMBER, 0, (uintptr_t)ADMITTED_PATH, sizeof(ADMITTED_PATH), 0);
  if ((intptr_t)fd >= 0) {
    status[0] = syscall(READ_NUMBER, fd, (uintptr_t)admitted, ADMITTED_COUNT, 0) == ADMITTED_COUNT;
    syscall(CLOSE_NUMBER, fd, 0, 0, 0);
  }
  fd = syscall(OPEN_NUMBER, 0, (uintptr_t)EXCLUDED_PATH, sizeof(EXCLUDED_PATH), 0);
  if ((intptr_t)fd >= 0) {
    status[1] = syscall(READ_NUMBER, fd, (uintptr_t)excluded, EXCLUDED_COUNT, 0) == EXCLUDED_COUNT;
    syscall(CLOSE_NUMBER, fd, 0, 0, 0);
  }
  fd = syscall(OPEN_NUMBER, 0, (uintptr_t)ADMITTED_PATH, sizeof(ADMITTED_PATH), NATIVE_WRITE_FLAG);
  if ((intptr_t)fd >= 0) { status[2] = 1; syscall(CLOSE_NUMBER, fd, 0, 0, 0); }
  packet("admitted", admitted, ADMITTED_COUNT); packet("excluded", excluded, EXCLUDED_COUNT);
  packet("status", status, 3); htif_puts("DONE\n"); htif_exit(0); return 0;
}
"""
    )


def _build(owner, compiler, platform, source, elf, dependencies):
    tools = []
    for name in ("cc1", "as", "ld", "collect2"):
        result = _run(owner, "compiler_" + name, [str(compiler), "-print-prog-name=" + name])
        assert result.returncode == 0
        raw = Path(result.stdout.decode().strip())
        assert raw.is_absolute() and raw.is_file()
        tools.append(raw.resolve(strict=True))
    result = _run(owner, "compiler_libgcc", [str(compiler), "-print-libgcc-file-name"])
    assert result.returncode == 0
    archive = Path(result.stdout.decode().strip()).resolve(strict=True)
    assert archive.is_file()
    loader_members = set()
    for index, tool in enumerate((compiler, *tools)):
        result = _run(owner, "compiler_loader_" + str(index), ["/usr/bin/ldd", str(tool)], dependencies=(tool,))
        assert result.returncode == 0
        assert "not found" not in result.stdout.decode()
        for token in result.stdout.decode().split():
            if token.startswith("/"):
                path = Path(token).resolve(strict=True)
                assert path.is_file()
                loader_members.add(path)
    members = (platform / "crt.S", platform / "htif.c", platform / "libc_min.c", source)
    includes = ["-I" + str(platform), "-I" + str(platform.parent)]
    headers = set()
    for index, member in enumerate(members):
        result = _run(
            owner,
            "compiler_headers_" + str(index),
            [
                str(compiler),
                "-M",
                "-MT",
                "source_dependencies",
                *includes,
                str(member),
            ],
            inputs=(member,),
        )
        assert result.returncode == 0
        declaration, separator, paths = result.stdout.decode().replace("\\\n", " ").partition(":")
        assert declaration == "source_dependencies" and separator
        for raw in paths.split():
            path = Path(raw)
            assert path.is_absolute() and path.is_file() and "\\" not in raw
            headers.add(path.resolve(strict=True))
    result = _run(
        owner,
        "elf_build",
        [
            str(compiler),
            "-O2",
            "-ffreestanding",
            "-fno-builtin",
            "-fno-use-linker-plugin",
            "-mcmodel=medany",
            "-nostdlib",
            "-static",
            "-Wl,-T," + str(platform / "link.ld"),
            *includes,
            *map(str, members),
            "-lgcc",
            "-o",
            str(elf),
        ],
        inputs=(*members, platform / "link.ld"),
        dependencies=(*dependencies, *tools, archive, *headers, *loader_members),
        outputs=(elf,),
        timeout=60,
    )
    assert result.returncode == 0, result.stderr.decode()
    assert elf.read_bytes().startswith(b"\x7fELF")


def _outputs(result):
    assert result.returncode == 0, result.stderr.decode()
    decoder, outputs = OutB64Decoder(), {}
    lines = result.stdout.decode("ascii").splitlines()
    assert lines[-1] == "DONE" and lines.count("DONE") == 1 and not result.stderr
    for line in lines[:-1]:
        assert decoder.consume(line.split(), outputs)
    decoder.require_closed()
    assert set(outputs) == {"admitted", "excluded", "status"}
    return outputs


def test_same_public_host_proxy_elf_retains_outputs_but_cannot_read_excluded_file(tmp_path):
    (outer, compiler, engine, dtc, public, public_tests, platform, build_citation), commits = _selected()
    public_members = _public_source(
        tmp_path,
        "public_engine",
        public,
        commits[0],
        (
            "fesvr/syscall.cc",
            "fesvr/elfloader.cc",
            "fesvr/htif.cc",
            "riscv/platform.h",
            "riscv/sim.cc",
        ),
    )
    (helper,) = _public_source(tmp_path, "public_tests", public_tests, commits[1], ("benchmarks/common/syscalls.c",))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    admitted, excluded = workspace / "admitted.txt", tmp_path / "excluded.txt"
    admitted_payload, excluded_payload = b"OWNED_ADMITTED_FILE_BYTES", b"OWNED_EXCLUDED_FILE_BYTES"
    admitted.write_bytes(admitted_payload)
    excluded.write_bytes(excluded_payload)
    source, elf = tmp_path / "control.c", workspace / "control.elf"
    source.write_text(
        _control_source(
            public_members[0],
            helper,
            admitted,
            excluded,
            (
                len(admitted_payload),
                len(excluded_payload),
            ),
            os.O_WRONLY,
        )
    )
    _build(tmp_path, compiler, platform, source, elf, (*public_members, helper, build_citation))
    # This older engine-build citation is retained as diagnostic provenance.
    # It does not grant an independently qualified physical/runtime owner.
    dependencies = (source, *public_members, helper, engine, dtc, build_citation)
    host = _run(
        tmp_path,
        "host_visible",
        [str(engine), "-p1", str(elf)],
        inputs=(elf, admitted, excluded),
        dependencies=dependencies,
        cwd=workspace,
    )
    assert _outputs(host) == {
        "admitted": [list(admitted_payload)],
        "excluded": [list(excluded_payload)],
        "status": [[1, 1, 1]],
    }
    library = tmp_path / "library"
    library.mkdir()
    (library / "api.py").write_text('"""Private namespace canary; no compiler implementation."""\n')
    contract = freeze_compiler_library(
        library,
        review_id="owned-filesystem-control",
        public_modules=("api",),
        sources=(("api.py", "api"),),
    )
    view = E.materialize_component_view(
        tmp_path / "view",
        library=contract,
        library_root=library,
        generation_sha256="1" * 64,
        inputs=(E.ApprovedInput(admitted, "generated_input/admitted.txt", sha256_file(admitted), "generated_input"),),
    )
    for index, tool in enumerate((engine, dtc)):
        result = _run(tmp_path, "loader_" + str(index), ["/usr/bin/ldd", str(tool)], dependencies=(tool,))
        assert result.returncode == 0
    runtime = R.inventory_runtime(
        files=(),
        trees=(),
        executables=((engine, "/usr/bin/diagnostic-engine"), (dtc, "/usr/bin/dtc")),
    )
    policy = E.strict_tool_policy(
        view,
        workspace,
        runtime=runtime,
        candidate_destination=str(workspace),
        bwrap_binary=outer,
        candidate_writable=False,
    )
    assert "--bind" not in policy and "--share-net" not in policy
    snapshot = {p.name: sha256_file(p) for p in workspace.iterdir()}
    scoped = _run(
        tmp_path,
        "scoped_visible",
        [*policy, "--", "/usr/bin/diagnostic-engine", "-p1", str(elf)],
        inputs=(elf, admitted, excluded),
        dependencies=(
            *dependencies,
            *(row.source for row in runtime),
            Path(E.__file__),
            Path(R.__file__),
            *(p for p in view.root.rglob("*") if p.is_file()),
        ),
        cwd=workspace,
    )
    assert _outputs(scoped) == {
        "admitted": [list(admitted_payload)],
        "excluded": [[0] * len(excluded_payload)],
        "status": [[1, 0, 0]],
    }
    assert {p.name: sha256_file(p) for p in workspace.iterdir()} == snapshot
    assert excluded.read_bytes() == excluded_payload
    # Source symlinks are refused before namespace construction/execution.
    linked = tmp_path / "linked_workspace"
    linked.mkdir()
    (linked / "escape").symlink_to(excluded)
    with pytest.raises(StageGateError, match="linked files"):
        E.strict_tool_policy(view, linked, runtime=runtime, bwrap_binary=outer, candidate_writable=False)
    for record in tmp_path.rglob("invocation.json"):
        I.require_environment(record, environment=_ENV)
    (tmp_path / "selection.json").write_text(
        json.dumps(
            {
                "scope": "process and host-filesystem containment only; no target/runtime/origin/timing authority",
                "engine_build_scope": (
                    "older public-source diagnostic citation; transitive build environment unqualified"
                ),
                "elf": {"path": str(elf), "sha256": sha256_file(elf)},
                "source": {"path": str(source), "sha256": sha256_file(source)},
                "outer": {"path": str(outer), "sha256": sha256_file(outer)},
                "public_source_commits": list(commits),
                "environment": I.environment_identity(_ENV),
                "view": {"path": str(view.root), "manifest_sha256": view.manifest_sha256},
                "runtime": [
                    {"source": str(row.source), "destination": row.destination, "sha256": row.sha256} for row in runtime
                ],
                "command": [*policy, "--", "/usr/bin/diagnostic-engine", "-p1", str(elf)],
                "original_output_roster": {
                    "admitted": [1, len(admitted_payload)],
                    "excluded": [1, len(excluded_payload)],
                    "status": [1, 3],
                },
                "original_workspace": snapshot,
                "source_symlink_negative": {
                    "path": str(linked / "escape"),
                    "target": str(excluded),
                    "refused_by": "strict_tool_policy before process launch",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
